// E9b CPU-MoE prefill feasibility bench (2026-10-15). Question: can the host
// CPU take over a share of the 4096-token prefill MoE so the GPU staging H2D
// (46.8 GiB/block, ~5.0 s copy + 2.4 s fill) shrinks?
//
// The production decode CPU executor aggregates ~45 GB/s / ~54 us per
// single-row (expert, 1 token) activation (tmp/q4e-sprint-r1-20261014 d1/e2
// logs; briefing number 53.7 us). Prefill rows are grouped: one expert can
// receive ~80-100 rows per 4096-token layer. This bench measures, on the real
// production entry point ops::cpu::cpu_expert_batch_forward (the only caller
// of the Q8/A64 kernels, src/models/Qwen4ExpExpertExecutor.cpp:488):
//   1. decode16   16 tasks x 1 row, tokens=1   (production decode granularity)
//   2. flat40     40 tasks x 1 row, tokens=1   (max task parallelism, no reuse)
//   3. quad10x4   10 experts x 4 rows, tokens=4, one call (MTP-verify shape)
//   4. oneex4     1 expert x 4 rows, tokens=4  (per-expert workhorse call)
//   5. grp1xR     1 expert x R rows (R in {16,32}) as R/4 consecutive oneex4
//                 calls, weights hot in L1/L2 across calls
// and derives rows/s, window weight GB/s and the equivalent 4096-token block
// time for the all-cold 40,960 rows/layer x 48 layers case.
//
// API caps this bench must respect (src/ops/Qwen4ExpCpuExpertOps.cpp):
//   - cpu_expert_batch_forward rejects tokens>4 or count>40 (:1581).
//   - the per-expert AVX2 entries fall back to scalar above 4 tokens
//     (:1372, :1475, :1514); there is NO vectorized >4-row path today.
// Slots are synthetic host memory in the compact-nested layout
// (2,534,400 B/expert, codes 51200/25600, nested absmax 200/100 floats),
// identical to q4e_decode_moe_bench.cpp:48-58. No model files, no GPU, no
// SYCL context is created; this executable runs standalone on the host.
//
// Build:  AilaQwen4ExpCpuMoePrefillBench (CMakeLists.txt)
// Run:    build-refactor/AilaQwen4ExpCpuMoePrefillBench.exe --arms all
// Gate:   compare us/row against the decode anchor; the coordinator owns the
//         e2e projection (charter sec 5.3), this tool prints the arithmetic.

#include "ops/Qwen4ExpCpuExpertOps.hpp"

#include <tbb/task_arena.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kHidden = ops::cpu::kHiddenDim;       // 2560
constexpr int kInter = ops::cpu::kIntermediateDim;  // 640
constexpr int kBlock = ops::cpu::kBlockSize;        // 64
constexpr int kNestedBlock = 256;
constexpr int kMaxTasks = ops::cpu::kMaxExpertBatchTasks; // 40

// Compact nested slot geometry (q4e_decode_moe_bench.cpp:48-56).
constexpr size_t kGuPacked = static_cast<size_t>(2 * kInter) * (kHidden / 2); // 1,638,400
constexpr size_t kGuCodes = static_cast<size_t>(2 * kInter) * (kHidden / kBlock); // 51,200
constexpr size_t kDnPacked = static_cast<size_t>(kHidden) * (kInter / 2); // 819,200
constexpr size_t kDnCodes = static_cast<size_t>(kHidden) * (kInter / kBlock); // 25,600
constexpr size_t kSlotBytes = kGuPacked + kGuCodes + kDnPacked + kDnCodes; // 2,534,400
constexpr size_t kGuScaleNat = kGuCodes / kNestedBlock; // 200
constexpr size_t kDnScaleNat = kDnCodes / kNestedBlock; // 100

// 4096-token prefill block, top_k 10 (Qwen4ExpBnb4Backend.cpp:1179 route shape).
constexpr int kRowsPerLayer = 4096 * 10; // 40,960 (token, rank) rows
constexpr int kLayers = 48;

constexpr float kOffsets[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double cv_pct(const std::vector<double>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double var = 0;
    for (double v : values) var += (v - mean) * (v - mean);
    return std::sqrt(var / values.size()) / mean * 100.0;
}

uint32_t splitmix32(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    v ^= v >> 16;
    return v;
}

// Host-side synthetic pool: one compact-nested slot per expert plus the
// per-expert nested absmax slices and a shared 256-entry quant map (the NF4
// LUT replicated 16x so nested.codes[*] can index it directly).
struct HostSlotPool {
    std::vector<uint8_t> data;     // pool * kSlotBytes
    std::vector<float> gu_absmax;  // pool * kGuScaleNat
    std::vector<float> dn_absmax;  // pool * kDnScaleNat
    std::vector<float> quant_map;  // 256 floats
    std::vector<ops::MoeExpertSlotHost> slots;
    size_t pool = 0;
};

HostSlotPool make_pool(size_t pool, uint32_t seed) {
    HostSlotPool p;
    p.pool = pool;
    p.data.resize(pool * kSlotBytes);
    p.gu_absmax.resize(pool * kGuScaleNat);
    p.dn_absmax.resize(pool * kDnScaleNat);
    p.quant_map.resize(256);
    for (size_t i = 0; i < 256; ++i) p.quant_map[i] = kOffsets[i % 16];

    auto* words = reinterpret_cast<uint32_t*>(p.data.data());
    const size_t count = p.data.size() / sizeof(uint32_t);
    for (size_t i = 0; i < count; ++i)
        words[i] = splitmix32(static_cast<uint32_t>(i) + seed);
    for (size_t i = 0; i < p.gu_absmax.size(); ++i)
        p.gu_absmax[i] = 0.01f + static_cast<float>(i % 29) * 0.0007f;
    for (size_t i = 0; i < p.dn_absmax.size(); ++i)
        p.dn_absmax[i] = 0.01f + static_cast<float>(i % 23) * 0.0009f;

    p.slots.resize(pool);
    for (size_t e = 0; e < pool; ++e) {
        uint8_t* base = p.data.data() + e * kSlotBytes;
        auto& slot = p.slots[e];
        slot.gate_up_packed = base;
        slot.down_packed = base + kGuPacked + kGuCodes;
        slot.gate_up_nested = {base + kGuPacked,
                               p.gu_absmax.data() + e * kGuScaleNat,
                               p.quant_map.data(), 0.0f, kNestedBlock};
        slot.down_nested = {base + kGuPacked + kGuCodes + kDnPacked,
                            p.dn_absmax.data() + e * kDnScaleNat,
                            p.quant_map.data(), 0.0f, kNestedBlock};
    }
    return p;
}

struct RunIO {
    std::vector<bf16> x;    // 4 * kHidden
    std::vector<bf16> h;    // 40 * kInter (+64 sentinel)
    std::vector<float> out; // 40 * kHidden (+256 sentinel)
};

constexpr float kSentinel = -777.0f;

RunIO make_io() {
    RunIO io;
    io.x.resize(static_cast<size_t>(4) * kHidden);
    for (size_t i = 0; i < io.x.size(); ++i)
        io.x[i] = bf16(-0.2f + 0.4f * static_cast<float>((i * 2654435761u) % 1000) / 1000.0f);
    io.h.assign(static_cast<size_t>(kMaxTasks) * kInter + 64, bf16(kSentinel));
    io.out.assign(static_cast<size_t>(kMaxTasks) * kHidden + 256, kSentinel);
    return io;
}

bool all_finite(const std::vector<float>& v) {
    for (float f : v) if (!std::isfinite(f)) return false;
    return true;
}

ops::cpu::CpuKernelBackend parse_backend(const std::string& name) {
    using B = ops::cpu::CpuKernelBackend;
    if (name == "a64") return B::Nf4Q8Activation64Avx2;
    if (name == "h8") return B::Nf4Q8Avx2;
    if (name == "row4") return B::Nf4Q8Row4Avx2;
    if (name == "block64") return B::Nf4Q8Block64Avx2;
    if (name == "gpuorder") return B::GpuOrderAvx2;
    if (name == "avx2") return B::Avx2;
    throw std::invalid_argument("unknown --backend: " + name);
}

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> values;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = text.find(',', begin);
        const std::string token = text.substr(begin, end == std::string::npos ? end : end - begin);
        if (token.empty()) throw std::invalid_argument("empty list element");
        const int value = std::stoi(token);
        if (value <= 0) throw std::invalid_argument("list values must be positive");
        values.push_back(value);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

struct BenchConfig {
    ops::cpu::CpuKernelBackend backend = ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2;
    std::string backend_name = "a64";
    int threads = 18;             // production AILA_Q4E_CPU_THREADS
    size_t pool = 512;            // experts in the synthetic arena (~1.24 GiB)
    int iters = 20, rounds = 5;
    bool scale_shift = true;      // AILA_Q4E_CPU_SCALE_SHIFT=1
    bool a64_specialize = true;   // AILA_Q4E_CPU_A64_SPECIALIZE=1
    bool a64_row2 = true;         // AILA_Q4E_CPU_A64_ROW2=1
    bool verify_scalar = true;
    std::vector<int> grp_rows{16, 32};
    bool arm_decode16 = true, arm_flat40 = true, arm_quad = true;
    bool arm_oneex4 = true, arm_grp = true;
};

// One arm: `window_experts` distinct slots per call; "quad" arms place 4 rows
// (tokens 0..3) on each expert, "single" arms one row with tokens=1. `reps`
// consecutive calls reuse the same window (weights stay hot in cache).
struct ArmSpec {
    std::string name;
    bool quad = false;
    int window_experts = 1;
    int tokens = 1;
    int reps = 1;
};

struct ArmResult {
    std::string name;
    int rows_per_call = 0, reps = 0, window_experts = 0;
    double call_ms = 0, cv = 0;
    double us_per_row = 0, rows_per_s = 0, window_gbps = 0;
};

ArmResult run_arm(const BenchConfig& cfg, HostSlotPool& pool, RunIO& io,
                  tbb::task_arena& arena, const ArmSpec& spec, size_t& cursor) {
    const int rows_per_call = spec.quad ? spec.window_experts * 4 : spec.window_experts;
    if (rows_per_call > kMaxTasks || spec.tokens > 4 || spec.tokens < 1)
        throw std::invalid_argument("arm exceeds batch caps: " + spec.name);
    if (pool.pool <= static_cast<size_t>(spec.window_experts) + 1)
        throw std::invalid_argument("pool too small for arm: " + spec.name);

    std::vector<ops::cpu::CpuExpertTask> tasks(static_cast<size_t>(rows_per_call));
    const auto fill_tasks = [&](size_t window) {
        if (spec.quad) {
            for (int e = 0; e < spec.window_experts; ++e)
                for (int t = 0; t < 4; ++t)
                    tasks[static_cast<size_t>(e) * 4 + t] = {pool.slots[window + e], t};
        } else {
            for (int i = 0; i < spec.window_experts; ++i)
                tasks[static_cast<size_t>(i)] = {pool.slots[window + i], 0};
        }
    };
    const auto launch = [&] {
        ops::cpu::cpu_expert_batch_forward(tasks.data(), rows_per_call, io.x.data(), spec.tokens,
                                           io.h.data(), io.out.data(), nullptr, cfg.backend, nullptr,
                                           cfg.scale_shift, cfg.a64_specialize, cfg.a64_row2,
                                           false, nullptr);
    };

    // Warmup: 3 timed-equivalent iterations.
    for (int i = 0; i < 3; ++i) {
        const size_t window = (cursor + static_cast<size_t>(i) * spec.window_experts) %
                              (pool.pool - spec.window_experts);
        fill_tasks(window);
        for (int rep = 0; rep < spec.reps; ++rep) arena.execute(launch);
    }

    std::vector<double> samples;
    samples.reserve(cfg.rounds);
    for (int r = 0; r < cfg.rounds; ++r) {
        const auto t0 = Clock::now();
        for (int i = 0; i < cfg.iters; ++i) {
            const size_t window = cursor % (pool.pool - spec.window_experts);
            cursor += static_cast<size_t>(spec.window_experts);
            fill_tasks(window);
            for (int rep = 0; rep < spec.reps; ++rep) arena.execute(launch);
        }
        const auto t1 = Clock::now();
        samples.push_back(elapsed_ms(t0, t1) / cfg.iters);
    }

    const double iter_ms = median_of(samples);
    ArmResult result;
    result.name = spec.name;
    result.rows_per_call = rows_per_call;
    result.reps = spec.reps;
    result.window_experts = spec.window_experts;
    result.call_ms = iter_ms / spec.reps;
    result.cv = cv_pct(samples);
    const double rows_per_iter = static_cast<double>(rows_per_call) * spec.reps;
    result.us_per_row = iter_ms * 1000.0 / rows_per_iter;
    result.rows_per_s = rows_per_iter / (iter_ms / 1000.0);
    // First-pass weight stream of one call: distinct slots / call latency.
    result.window_gbps = static_cast<double>(spec.window_experts) * kSlotBytes / 1e9 /
                         (result.call_ms / 1000.0);
    std::cout << std::fixed << std::setprecision(4)
              << "[CPU MOE BENCH] {\"arm\":\"" << spec.name
              << "\",\"quad\":" << (spec.quad ? 1 : 0)
              << ",\"window_experts\":" << spec.window_experts
              << ",\"rows_per_call\":" << rows_per_call
              << ",\"reps\":" << spec.reps
              << ",\"rows_per_iter\":" << rows_per_iter
              << ",\"call_ms\":" << result.call_ms
              << ",\"iter_ms\":" << iter_ms
              << ",\"cv_pct\":" << result.cv
              << ",\"us_per_row\":" << result.us_per_row
              << ",\"rows_per_s\":" << result.rows_per_s
              << ",\"window_weight_gbps\":" << result.window_gbps << "}"
              << std::endl;
    return result;
}

void print_equivalence(const ArmResult& r) {
    const double layer_s = r.us_per_row * 1e-6 * kRowsPerLayer;
    const double block_s = layer_s * kLayers;
    const double block_tok_s = 4096.0 / block_s;
    std::cout << std::fixed << std::setprecision(4)
              << "[CPU MOE EQ] {\"arm\":\"" << r.name
              << "\",\"us_per_row\":" << r.us_per_row
              << ",\"rows_per_layer\":" << kRowsPerLayer
              << ",\"layers\":" << kLayers
              << ",\"eq_layer_s\":" << layer_s
              << ",\"eq_block_s\":" << block_s
              << ",\"eq_block_tok_s\":" << block_tok_s
              << ",\"cpu_rows_in_6p8s\":" << (6.8e6 / r.us_per_row)
              << ",\"assumption\":\"all-cold rows on CPU, no GPU overlap\"}"
              << std::endl;
}

bool verify_avx2_vs_scalar(HostSlotPool& pool, RunIO& io) {
    if (!pool.pool) return true;
    const ops::cpu::CpuExpertTask task{pool.slots[0], 0};
    std::vector<bf16> h_ref(static_cast<size_t>(kInter));
    std::vector<float> out_ref(static_cast<size_t>(kHidden));
    ops::cpu::cpu_expert_forward_scalar(pool.slots[0], io.x.data(), out_ref.data(), 1,
                                        h_ref.data(), nullptr, 1.0f, false, kHidden, kInter,
                                        kHidden, kBlock);
    ops::cpu::cpu_expert_batch_forward(&task, 1, io.x.data(), 1, io.h.data(), io.out.data(),
                                       nullptr, ops::cpu::CpuKernelBackend::Avx2);
    double worst = 0;
    for (int i = 0; i < kHidden; ++i) {
        const double a = io.out[i], b = out_ref[i];
        worst = std::max(worst, std::abs(a - b) / (1.0 + std::abs(b)));
    }
    const bool ok = worst <= 2e-3;
    std::cout << std::fixed << std::setprecision(6)
              << "[CPU MOE GUARD] {\"scalar_canary_rel\":" << worst
              << ",\"tol\":0.002,\"ok\":" << (ok ? "true" : "false") << "}" << std::endl;
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    try {
        BenchConfig cfg;
        std::string arms = "all";
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("expected value after option");
            const std::string key(argv[i]);
            const std::string value(argv[i + 1]);
            if (key == "--backend") { cfg.backend = parse_backend(value); cfg.backend_name = value; }
            else if (key == "--threads") cfg.threads = std::stoi(value);
            else if (key == "--pool") cfg.pool = static_cast<size_t>(std::stoi(value));
            else if (key == "--iters") cfg.iters = std::stoi(value);
            else if (key == "--rounds") cfg.rounds = std::stoi(value);
            else if (key == "--grp-rows") cfg.grp_rows = parse_list(value);
            else if (key == "--scale-shift") cfg.scale_shift = std::stoi(value) != 0;
            else if (key == "--a64-specialize") cfg.a64_specialize = std::stoi(value) != 0;
            else if (key == "--a64-row2") cfg.a64_row2 = std::stoi(value) != 0;
            else if (key == "--verify-scalar") cfg.verify_scalar = std::stoi(value) != 0;
            else if (key == "--arms") arms = value;
            else throw std::invalid_argument("unknown option: " + key);
        }
        if (cfg.threads <= 0 || cfg.iters <= 0 || cfg.rounds <= 0 || cfg.pool < 80)
            throw std::invalid_argument("threads/iters/rounds positive, pool >= 80 required");
        if (arms != "all") {
            cfg.arm_decode16 = cfg.arm_flat40 = cfg.arm_quad = cfg.arm_oneex4 = cfg.arm_grp = false;
            size_t begin = 0;
            while (begin <= arms.size()) {
                const size_t end = arms.find(',', begin);
                const std::string name = arms.substr(begin, end == std::string::npos ? end : end - begin);
                if (name == "decode16") cfg.arm_decode16 = true;
                else if (name == "flat40") cfg.arm_flat40 = true;
                else if (name == "quad10x4") cfg.arm_quad = true;
                else if (name == "oneex4") cfg.arm_oneex4 = true;
                else if (name == "grp") cfg.arm_grp = true;
                else throw std::invalid_argument("unknown arm: " + name);
                if (end == std::string::npos) break;
                begin = end + 1;
            }
        }

        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const int threads = std::min(cfg.threads, static_cast<int>(hw));
        std::cout << "[CPU MOE DEVICE] {\"host_threads\":" << hw
                  << ",\"requested_threads\":" << cfg.threads
                  << ",\"effective_threads\":" << threads << "}" << std::endl;
        std::cout << "[CPU MOE CONFIG] {\"backend\":\"" << cfg.backend_name
                  << "\",\"slot_bytes\":" << kSlotBytes
                  << ",\"pool\":" << cfg.pool
                  << ",\"pool_gib\":" << static_cast<double>(cfg.pool * kSlotBytes) / double(1ull << 30)
                  << ",\"iters\":" << cfg.iters << ",\"rounds\":" << cfg.rounds
                  << ",\"scale_shift\":" << (cfg.scale_shift ? 1 : 0)
                  << ",\"a64_specialize\":" << (cfg.a64_specialize ? 1 : 0)
                  << ",\"a64_row2\":" << (cfg.a64_row2 ? 1 : 0) << "}" << std::endl;

        HostSlotPool pool = make_pool(cfg.pool, 0x9e3779b9u);
        RunIO io = make_io();
        tbb::task_arena arena(threads);

        if (cfg.verify_scalar && !verify_avx2_vs_scalar(pool, io)) {
            std::cout << "[CPU MOE GUARD] scalar canary failed; failing closed" << std::endl;
            return 2;
        }

        std::vector<ArmResult> results;
        size_t cursor = 0;
        const auto add = [&](const ArmSpec& spec) {
            results.push_back(run_arm(cfg, pool, io, arena, spec, cursor));
        };

        if (cfg.arm_decode16) add({"decode16", false, 16, 1, 1});
        if (cfg.arm_flat40) add({"flat40", false, 40, 1, 1});
        if (cfg.arm_quad) add({"quad10x4", true, 10, 4, 1});
        if (cfg.arm_oneex4) add({"oneex4", true, 1, 4, 1});
        if (cfg.arm_grp) {
            for (int rows : cfg.grp_rows) {
                if (rows % 4 != 0 || rows <= 0) throw std::invalid_argument("--grp-rows must be positive multiples of 4");
                add({"grp1x" + std::to_string(rows), true, 1, 4, rows / 4});
            }
        }

        // Guards: repeat determinism on one expert + finiteness + sentinel tails.
        std::vector<float> copy(static_cast<size_t>(kHidden));
        {
            const size_t window = cursor % (cfg.pool - 1);
            const ops::cpu::CpuExpertTask task{pool.slots[window], 0};
            for (int pass = 0; pass < 2; ++pass) {
                ops::cpu::cpu_expert_batch_forward(&task, 1, io.x.data(), 1, io.h.data(),
                                                   io.out.data(), nullptr, cfg.backend, nullptr,
                                                   cfg.scale_shift, cfg.a64_specialize,
                                                   cfg.a64_row2, false, nullptr);
                if (pass == 0)
                    std::memcpy(copy.data(), io.out.data(), static_cast<size_t>(kHidden) * sizeof(float));
            }
        }
        const bool deterministic =
            std::memcmp(copy.data(), io.out.data(), static_cast<size_t>(kHidden) * sizeof(float)) == 0;
        const bool finite = all_finite(io.out);
        const float h_sentinel = static_cast<float>(bf16(kSentinel));
        const bool sentinel_h =
            static_cast<float>(io.h[static_cast<size_t>(kMaxTasks) * kInter]) == h_sentinel;
        const bool sentinel_out = io.out[static_cast<size_t>(kMaxTasks) * kHidden] == kSentinel;
        std::cout << "[CPU MOE GUARD2] {\"deterministic\":" << (deterministic ? "true" : "false")
                  << ",\"finite\":" << (finite ? "true" : "false")
                  << ",\"sentinel_h\":" << (sentinel_h ? "true" : "false")
                  << ",\"sentinel_out\":" << (sentinel_out ? "true" : "false") << "}" << std::endl;
        if (!deterministic || !finite || !sentinel_h || !sentinel_out) {
            std::cout << "[CPU MOE GUARD2] guard failed; failing closed" << std::endl;
            return 3;
        }

        for (const auto& r : results) print_equivalence(r);
        std::cout << "[CPU MOE NOTE] eq_* assumes every routed row runs on the CPU; "
                     "the decision rule is rows_cpu_budget = overlap_budget_s / us_per_row"
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CPU MOE FAIL] " << e.what() << std::endl;
        return 1;
    }
}
