// R1 decode-MoE kernel bench (2026-10-15). Production adaptive-decode shapes:
// per layer/token the GPU computes the cached hits of 10 routed routes with
// moe_decode_gate_up_swiglu_grouped + moe_decode_down_to_rank_f32 +
// moe_ordered_accumulate_down_rows (Qwen4ExpBnb4Backend.cpp:1179-1365
// RunResidentDecode: array-form descriptors, nested_fixed256 = true, hot and
// staging subsets, CPU arena for the rest). This bench measures those kernels
// with production descriptor geometry (compact nested codes, slot =
// 2,534,400 B), the fixed-256 / uniform-scale variants
// (src/ops/MoeDecodeVariant.*), an E sweep (5/10/20) for small-grid
// efficiency, and a 48-layer per-layer chain (gu -> down -> merge) with a
// busy/gap split so the 27.49 ms/token experts_dev span
// (cost-model section 6.17, dec-instr-b2) can be attributed to kernel time
// vs queue gaps.
//
// Zero model files: slots, codes, nested absmax and activations are synthetic.
//
// Build:  AilaQwen4ExpDecodeMoeBench (CMakeLists.txt)
// Run:    build-refactor/AilaQwen4ExpDecodeMoeBench.exe --experts 10,20
// Gate:   kernel-level >= 3% at two shapes (charter sec 5.3); the e2e
//         projection is printed by the coordinator, not by this tool.

#include "core/Context.hpp"
#include "ops/MoeDecodeVariant.hpp"
#include "ops/MoeExpertOps.hpp"

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
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kHidden = 2560;
constexpr int kInter = 640;
constexpr int kTopK = 10;
constexpr int kBlock = 64;
constexpr int kNestedBlock = 256;

// Adaptive-cache slot geometry, Qwen4ExpAdaptiveCache.cpp:100-119 with
// compact_nested = true (validated mandatory there): 2,534,400 B per expert.
constexpr size_t kGuPacked = static_cast<size_t>(2 * kInter) * (kHidden / 2); // 1,638,400
constexpr size_t kGuCodes = static_cast<size_t>(2 * kInter) * (kHidden / kBlock); // 51,200
constexpr size_t kDnPacked = static_cast<size_t>(kHidden) * (kInter / 2); // 819,200
constexpr size_t kDnCodes = static_cast<size_t>(kHidden) * (kInter / kBlock); // 25,600
constexpr size_t kSlotBytes = kGuPacked + kGuCodes + kDnPacked + kDnCodes; // 2,534,400
constexpr size_t kGuScaleNat = kGuCodes / kNestedBlock; // 200
constexpr size_t kDnScaleNat = kDnCodes / kNestedBlock; // 100
constexpr size_t kFp32SlotBytes = kGuPacked + kGuCodes * sizeof(float) + kDnPacked +
                                  kDnCodes * sizeof(float); // 2,764,800 (non-nested arm)

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

double quantile_of(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - static_cast<double>(lo));
}

double cv_pct(const std::vector<double>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double var = 0;
    for (double v : values) var += (v - mean) * (v - mean);
    return std::sqrt(var / values.size()) / mean * 100.0;
}

uint64_t event_start_ns(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_start>();
}

uint64_t event_end_ns(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_end>();
}

void print_bench(const char* arm, int experts, int rows, double bytes, double med,
                 double p10, double p90, double cv, size_t iters) {
    const double gbps = bytes / 1e9 / (med / 1e3);
    std::cout << "[DECODE MOE BENCH] {\"arm\":\"" << arm << "\",\"experts\":" << experts
              << ",\"rows\":" << rows << ",\"iters\":" << iters
              << ",\"median_ms\":" << med << ",\"p10_ms\":" << p10
              << ",\"p90_ms\":" << p90 << ",\"cv_pct\":" << cv
              << ",\"mb\":" << bytes / 1e6 << ",\"gbps\":" << gbps << "}" << std::endl;
}

struct BitCompare {
    size_t mismatch = 0;
    size_t first = 0;
};

template <typename T>
BitCompare compare_bits(const std::vector<T>& a, const std::vector<T>& b) {
    BitCompare result;
    const size_t count = std::min(a.size(), b.size());
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            if (result.mismatch == 0) result.first = i;
            ++result.mismatch;
        }
    }
    return result;
}

bool all_finite_bf16(const std::vector<bf16>& values) {
    for (bf16 v : values)
        if (!std::isfinite(static_cast<float>(v))) return false;
    return true;
}

bool all_finite_f32(const std::vector<float>& values) {
    for (float v : values)
        if (!std::isfinite(v)) return false;
    return true;
}

struct SlotPool {
    uint8_t* base = nullptr;
    float* metadata = nullptr;
    size_t slots = 0;
    std::vector<ops::MoeExpertSlotDev> host_slots;
    ops::MoeExpertSlotDev* dev_slots = nullptr;
    float* gu_map = nullptr;
    float* dn_map = nullptr;
};

// One pool == one "layer frame": slot geometry identical to production, with
// per-expert nested metadata slices (gu: 200 floats, dn: 100 floats).
SlotPool make_nested_pool(Context& ctx, size_t slots) {
    SlotPool pool;
    pool.slots = slots;
    pool.base = static_cast<uint8_t*>(ctx.alloc_device(slots * kSlotBytes));
    const size_t meta_floats = 32 + slots * (kGuScaleNat + kDnScaleNat);
    pool.metadata = static_cast<float*>(ctx.alloc_device(meta_floats * sizeof(float)));
    pool.gu_map = pool.metadata;
    pool.dn_map = pool.metadata + 16;
    pool.host_slots.resize(slots);
    for (size_t s = 0; s < slots; ++s) {
        uint8_t* slot_base = pool.base + s * kSlotBytes;
        auto& slot = pool.host_slots[s];
        slot.gate_up_packed = slot_base;
        slot.down_packed = slot_base + kGuPacked + kGuCodes;
        slot.gate_up_nested = {slot_base + kGuPacked,
                               pool.metadata + 32 + s * (kGuScaleNat + kDnScaleNat),
                               pool.gu_map, 0.0f, kNestedBlock};
        slot.down_nested = {slot_base + kGuPacked + kGuCodes + kDnPacked,
                            pool.metadata + 32 + s * (kGuScaleNat + kDnScaleNat) + kGuScaleNat,
                            pool.dn_map, 0.0f, kNestedBlock};
    }
    pool.dev_slots = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(slots * sizeof(ops::MoeExpertSlotDev)));
    return pool;
}

void init_nested_pool(Context& ctx, SlotPool& pool, uint32_t seed) {
    const size_t words = pool.slots * kSlotBytes / sizeof(uint32_t);
    auto* words_ptr = reinterpret_cast<uint32_t*>(pool.base);
    // Splitmix-style hash so packed codes and both scale tables look random.
    ctx.queue().parallel_for(sycl::range<1>(words), [=](sycl::id<1> id) {
        uint32_t v = static_cast<uint32_t>(id[0]) + seed;
        v ^= v >> 16;
        v *= 0x7feb352dU;
        v ^= v >> 15;
        v *= 0x846ca68bU;
        v ^= v >> 16;
        words_ptr[id[0]] = v;
    });
    const size_t meta_floats = 32 + pool.slots * (kGuScaleNat + kDnScaleNat);
    auto* meta = pool.metadata;
    ctx.queue().parallel_for(sycl::range<1>(meta_floats), [=](sycl::id<1> id) {
        const size_t index = id[0];
        if (index < 16) {
            meta[index] = kOffsets[index]; // gu nested quant map
        } else if (index < 32) {
            meta[index] = kOffsets[index - 16]; // dn nested quant map
        } else {
            const size_t rel = index - 32;
            const size_t within = rel % (kGuScaleNat + kDnScaleNat);
            meta[index] = 0.01f + static_cast<float>(within % 29) * 0.0007f;
        }
    });
    ctx.memcpy_h2d(pool.dev_slots, pool.host_slots.data(),
                   pool.slots * sizeof(ops::MoeExpertSlotDev));
    ctx.synchronize();
}

// Non-nested (fp32 absmax) descriptors for the K1-iso arm.
struct Fp32Pool {
    uint8_t* base = nullptr;
    std::vector<ops::MoeExpertSlotDev> host_slots;
    ops::MoeExpertSlotDev* dev_slots = nullptr;
};

Fp32Pool make_fp32_pool(Context& ctx, size_t slots, uint32_t seed) {
    Fp32Pool pool;
    pool.base = static_cast<uint8_t*>(ctx.alloc_device(slots * kFp32SlotBytes));
    pool.host_slots.resize(slots);
    for (size_t s = 0; s < slots; ++s) {
        uint8_t* slot_base = pool.base + s * kFp32SlotBytes;
        auto& slot = pool.host_slots[s];
        slot.gate_up_packed = slot_base;
        slot.gate_up_absmax = reinterpret_cast<const float*>(slot_base + kGuPacked);
        slot.down_packed = slot_base + kGuPacked + kGuCodes * sizeof(float);
        slot.down_absmax =
            reinterpret_cast<const float*>(slot_base + kGuPacked + kGuCodes * sizeof(float) + kDnPacked);
    }
    auto* words_ptr = reinterpret_cast<uint32_t*>(pool.base);
    ctx.queue().parallel_for(sycl::range<1>(slots * kFp32SlotBytes / sizeof(uint32_t)),
        [=](sycl::id<1> id) {
            uint32_t v = static_cast<uint32_t>(id[0]) + seed;
            v ^= v >> 16;
            v *= 0x7feb352dU;
            v ^= v >> 15;
            v *= 0x846ca68bU;
            v ^= v >> 16;
            words_ptr[id[0]] = v;
        });
    // Rewrite the absmax regions with sane positive scales, on device (the
    // pool is USM device memory and must not be touched from the host).
    auto* fp32_base = pool.base;
    ctx.queue().parallel_for(sycl::range<1>(slots * kGuCodes), [=](sycl::id<1> id) {
        const size_t slot = id[0] / kGuCodes;
        const size_t index = id[0] % kGuCodes;
        auto* gu = reinterpret_cast<float*>(fp32_base + slot * kFp32SlotBytes + kGuPacked);
        gu[index] = 0.01f + static_cast<float>(index % 37) * 0.0005f;
    });
    ctx.queue().parallel_for(sycl::range<1>(slots * kDnCodes), [=](sycl::id<1> id) {
        const size_t slot = id[0] / kDnCodes;
        const size_t index = id[0] % kDnCodes;
        auto* dn = reinterpret_cast<float*>(fp32_base + slot * kFp32SlotBytes + kGuPacked +
                                            kGuCodes * sizeof(float) + kDnPacked);
        dn[index] = 0.01f + static_cast<float>(index % 23) * 0.0007f;
    });
    pool.dev_slots = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(slots * sizeof(ops::MoeExpertSlotDev)));
    ctx.memcpy_h2d(pool.dev_slots, pool.host_slots.data(),
                   slots * sizeof(ops::MoeExpertSlotDev));
    ctx.synchronize();
    return pool;
}

struct Shapes {
    std::vector<int> experts;
    std::vector<int> rows;
};

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

} // namespace

int main(int argc, char** argv) {
    try {
        Shapes shapes;
        shapes.experts = {10, 20};
        shapes.rows = {1};
        int iters = 100;
        int rounds = 3;
        int chain_layers = 48;
        int chain_experts = 5;
        bool want_chain = true;
        bool want_fp32 = true;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument("expected value after option");
            const std::string key(argv[i]);
            const std::string value(argv[i + 1]);
            if (key == "--experts") shapes.experts = parse_list(value);
            else if (key == "--rows") shapes.rows = parse_list(value);
            else if (key == "--iters") iters = std::stoi(value);
            else if (key == "--rounds") rounds = std::stoi(value);
            else if (key == "--chain-layers") chain_layers = std::stoi(value);
            else if (key == "--chain-experts") chain_experts = std::stoi(value);
            else if (key == "--chain") want_chain = std::stoi(value) != 0;
            else if (key == "--fp32") want_fp32 = std::stoi(value) != 0;
            else throw std::invalid_argument("unknown option: " + key);
        }
        if (iters <= 0 || rounds <= 0) throw std::invalid_argument("iters/rounds must be positive");
        if (chain_layers <= 0 || chain_experts <= 0 || chain_experts > 16)
            throw std::invalid_argument("chain bounds");

        Context ctx(nullptr, true);
        auto& q = ctx.queue();
        std::cout << std::fixed << std::setprecision(4)
                  << "[DECODE MOE DEVICE] {\"name\":\""
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\",\"driver\":\""
                  << q.get_device().get_info<sycl::info::device::driver_version>() << "\"}"
                  << std::endl;
        std::cout << "[DECODE MOE CONFIG] {\"slot_bytes\":" << kSlotBytes
                  << ",\"fp32_slot_bytes\":" << kFp32SlotBytes << ",\"hidden\":" << kHidden
                  << ",\"intermediate\":" << kInter << ",\"top_k\":" << kTopK
                  << ",\"gu_mb_per_expert\":" << double(kGuPacked + kGuCodes) / 1e6
                  << ",\"dn_mb_per_expert\":" << double(kDnPacked + kDnCodes) / 1e6
                  << ",\"iters\":" << iters << ",\"rounds\":" << rounds << "}" << std::endl;

        // Per-shape descriptor pools are re-created per shape below; the
        // structural buffers are sized for the maximum shape.
        int max_experts = 0;
        for (int e : shapes.experts) max_experts = std::max(max_experts, e);
        const int max_rows = *std::max_element(shapes.rows.begin(), shapes.rows.end());

        auto* x = static_cast<bf16*>(ctx.alloc_device(
            static_cast<size_t>(max_rows) * kHidden * sizeof(bf16)));
        auto* weights = static_cast<float*>(ctx.alloc_device(
            static_cast<size_t>(max_rows) * kTopK * sizeof(float)));
        auto* y = static_cast<bf16*>(ctx.alloc_device(
            static_cast<size_t>(max_rows) * kHidden * sizeof(bf16)));
        const size_t h_count = static_cast<size_t>(max_experts) * kInter + 64; // +sentinel
        auto* h = static_cast<bf16*>(ctx.alloc_device(h_count * sizeof(bf16)));
        const size_t down_count = static_cast<size_t>(max_rows) * kTopK * kHidden + 256;
        auto* down = static_cast<float*>(ctx.alloc_device(down_count * sizeof(float)));
        auto* token_ids = static_cast<int*>(ctx.alloc_device(max_experts * sizeof(int)));
        auto* ranks = static_cast<int*>(ctx.alloc_device(max_experts * sizeof(int)));

        {
            std::vector<bf16> x_host(static_cast<size_t>(max_rows) * kHidden);
            for (size_t i = 0; i < x_host.size(); ++i)
                x_host[i] = bf16(-0.2f + 0.4f * static_cast<float>((i * 2654435761u) % 1000) / 1000.0f);
            std::vector<float> w_host(static_cast<size_t>(max_rows) * kTopK, 1.0f / kTopK);
            ctx.memcpy_h2d(x, x_host.data(), x_host.size() * sizeof(bf16));
            ctx.memcpy_h2d(weights, w_host.data(), w_host.size() * sizeof(float));
        }

        // ---- shape loop -----------------------------------------------------
        struct ShapeResult {
            int experts = 0;
            int rows = 0;
            double gu_ref = 0, gu_fixed = 0, gu_uniform = 0, gu_fixed_uniform = 0;
            double dn_ref = 0, dn_fixed = 0, dn_uniform = 0, dn_fixed_uniform = 0;
        };
        std::vector<ShapeResult> results;

        for (int experts : shapes.experts) {
            for (int rows : shapes.rows) {
                std::cout << "[DECODE MOE SHAPE] {\"experts\":" << experts
                          << ",\"rows\":" << rows << "}" << std::endl;
                SlotPool pool = make_nested_pool(ctx, static_cast<size_t>(experts));
                Fp32Pool fp32_pool;
                init_nested_pool(ctx, pool, 0x9e3779b9u + static_cast<uint32_t>(experts * 131 + rows));
                if (want_fp32)
                    fp32_pool = make_fp32_pool(ctx, static_cast<size_t>(experts),
                                               0x517cc1b7u + static_cast<uint32_t>(experts));

                std::vector<int> token_host(experts), rank_host(experts);
                for (int e = 0; e < experts; ++e) {
                    token_host[e] = e % rows;
                    rank_host[e] = e % kTopK;
                }
                ctx.memcpy_h2d(token_ids, token_host.data(), experts * sizeof(int));
                ctx.memcpy_h2d(ranks, rank_host.data(), experts * sizeof(int));
                // h/down sentinels so an out-of-bounds write fails the guard.
                {
                    std::vector<bf16> sentinel(64, bf16(-777.0f));
                    std::vector<float> dsent(256, -777.0f);
                    ctx.memcpy_h2d(h + static_cast<size_t>(experts) * kInter, sentinel.data(),
                                   sentinel.size() * sizeof(bf16));
                    ctx.memcpy_h2d(down + static_cast<size_t>(rows) * kTopK * kHidden,
                                   dsent.data(), dsent.size() * sizeof(float));
                }
                const double gu_bytes = double(experts) * (kGuPacked + kGuCodes);
                const double dn_bytes = double(experts) * (kDnPacked + kDnCodes);
                const double gu_fp32_bytes = double(experts) * (kGuPacked + kGuCodes * sizeof(float));
                const double dn_fp32_bytes = double(experts) * (kDnPacked + kDnCodes * sizeof(float));

                using Opts = ops::DecodeMoeVariantOptions;
                auto gu_ref = [&]() {
                    return ops::moe_decode_gate_up_swiglu_grouped(
                        ctx, pool.dev_slots, nullptr, x, h, experts, kHidden, kInter, kBlock,
                        nullptr, token_ids, true);
                };
                auto gu_array = [&]() {
                    return ops::moe_decode_gate_up_swiglu_grouped(
                        ctx, pool.host_slots.data(), nullptr, x, h, experts, kHidden, kInter, kBlock,
                        nullptr, token_host.data(), true);
                };
                auto gu_fp32 = [&]() {
                    return ops::moe_decode_gate_up_swiglu_grouped(
                        ctx, fp32_pool.dev_slots, nullptr, x, h, experts, kHidden, kInter, kBlock,
                        nullptr, token_ids, false);
                };
                auto gu_var = [&](const Opts& opts) {
                    return ops::moe_decode_gate_up_variant(
                        ctx, pool.dev_slots, token_ids, nullptr, x, h, experts, kHidden, kInter,
                        kBlock, opts, nullptr);
                };
                auto dn_ref = [&]() {
                    return ops::moe_decode_down_to_rank_f32(
                        ctx, pool.dev_slots, ranks, h, down, experts, kTopK, kInter, kHidden, kBlock,
                        token_ids, nullptr, nullptr, true);
                };
                auto dn_array = [&]() {
                    return ops::moe_decode_down_to_rank_f32(
                        ctx, pool.host_slots.data(), rank_host.data(), h, down, experts, kTopK,
                        kInter, kHidden, kBlock, token_host.data(), nullptr, nullptr, true);
                };
                auto dn_fp32 = [&]() {
                    return ops::moe_decode_down_to_rank_f32(
                        ctx, fp32_pool.dev_slots, ranks, h, down, experts, kTopK, kInter, kHidden,
                        kBlock, token_ids, nullptr, nullptr, false);
                };
                auto dn_var = [&](const Opts& opts) {
                    return ops::moe_decode_down_rank_variant(
                        ctx, pool.dev_slots, ranks, token_ids, h, down, experts, kTopK, kInter,
                        kHidden, kBlock, opts, nullptr);
                };

                const Opts opt_div_uniform{false, true};
                const Opts opt_fixed_div{true, false};
                const Opts opt_fixed_uniform{true, true};

                // Warmup every arm (first launch JITs the kernel).
                for (int i = 0; i < 3; ++i) {
                    gu_ref(); gu_var(opt_div_uniform); gu_var(opt_fixed_div); gu_var(opt_fixed_uniform);
                    if (want_fp32) gu_fp32();
                    if (experts <= 16) gu_array();
                    dn_ref(); dn_var(opt_div_uniform); dn_var(opt_fixed_div); dn_var(opt_fixed_uniform);
                    if (want_fp32) dn_fp32();
                    if (experts <= 16) dn_array();
                    ctx.synchronize();
                }

                auto time_arm = [&](auto&& launch, double bytes, const char* name) {
                    std::vector<double> samples;
                    samples.reserve(static_cast<size_t>(rounds));
                    for (int r = 0; r < rounds; ++r) {
                        const auto t0 = Clock::now();
                        for (int i = 0; i < iters; ++i) launch();
                        ctx.synchronize();
                        const auto t1 = Clock::now();
                        samples.push_back(elapsed_ms(t0, t1) / iters);
                    }
                    const double med = median_of(samples);
                    print_bench(name, experts, rows, bytes, med, quantile_of(samples, 0.1),
                                quantile_of(samples, 0.9), cv_pct(samples), static_cast<size_t>(iters));
                    return med;
                };

                // ---- bitwise parity vs the production nested dispatch ---------
                std::vector<bf16> ref_h(static_cast<size_t>(experts) * kInter);
                std::vector<float> ref_down(static_cast<size_t>(rows) * kTopK * kHidden);
                gu_ref();
                dn_ref();
                ctx.synchronize();
                ctx.memcpy_d2h(ref_h.data(), h, ref_h.size() * sizeof(bf16));
                ctx.memcpy_d2h(ref_down.data(), down, ref_down.size() * sizeof(float));
                ctx.synchronize();
                if (!all_finite_bf16(ref_h) || !all_finite_f32(ref_down))
                    throw std::runtime_error("reference arm produced non-finite values");

                std::vector<bf16> var_h(ref_h.size());
                std::vector<float> var_down(ref_down.size());
                auto parity = [&](const char* name, const Opts& opts) {
                    ctx.memcpy_h2d(h, ref_h.data(), ref_h.size() * sizeof(bf16));
                    gu_var(opts);
                    ctx.synchronize();
                    ctx.memcpy_d2h(var_h.data(), h, var_h.size() * sizeof(bf16));
                    ctx.synchronize();
                    const auto h_cmp = compare_bits(var_h, ref_h);
                    gu_ref(); // restore the reference h for the down comparison
                    ctx.synchronize();
                    dn_var(opts);
                    ctx.synchronize();
                    ctx.memcpy_d2h(var_down.data(), down, var_down.size() * sizeof(float));
                    ctx.synchronize();
                    const auto d_cmp = compare_bits(var_down, ref_down);
                    std::cout << "[DECODE MOE PARITY] {\"arm\":\"" << name
                              << "\",\"ref\":\"production-nested\",\"h_mismatch\":" << h_cmp.mismatch
                              << ",\"h_first\":" << h_cmp.first
                              << ",\"down_mismatch\":" << d_cmp.mismatch
                              << ",\"down_first\":" << d_cmp.first << "}" << std::endl;
                    if (h_cmp.mismatch || d_cmp.mismatch)
                        throw std::runtime_error(std::string("parity failure for ") + name);
                };
                parity("div-uniform", opt_div_uniform);
                parity("fixed-div", opt_fixed_div);
                parity("fixed-uniform", opt_fixed_uniform);
                if (experts <= 16) {
                    // Array-form production dispatch must match the device form.
                    std::vector<bf16> arr_h(ref_h.size());
                    gu_array();
                    ctx.synchronize();
                    ctx.memcpy_d2h(arr_h.data(), h, arr_h.size() * sizeof(bf16));
                    ctx.synchronize();
                    const auto cmp = compare_bits(arr_h, ref_h);
                    std::cout << "[DECODE MOE PARITY] {\"arm\":\"array-vs-dev\",\"ref\":\"production-nested\""
                              << ",\"h_mismatch\":" << cmp.mismatch << "}" << std::endl;
                }

                // ---- timings --------------------------------------------------
                ShapeResult result;
                result.experts = experts;
                result.rows = rows;
                result.gu_ref = time_arm(gu_ref, gu_bytes, "gu-ref-nested-div");
                if (want_fp32) time_arm(gu_fp32, gu_fp32_bytes, "gu-ref-fp32");
                if (experts <= 16) time_arm(gu_array, gu_bytes, "gu-ref-arrayform");
                result.gu_uniform = time_arm([&] { return gu_var(opt_div_uniform); }, gu_bytes,
                                             "gu-var-uniform");
                result.gu_fixed = time_arm([&] { return gu_var(opt_fixed_div); }, gu_bytes,
                                           "gu-var-fixed256");
                result.gu_fixed_uniform = time_arm([&] { return gu_var(opt_fixed_uniform); }, gu_bytes,
                                                   "gu-var-fixed256-uniform");
                dn_ref(); // make sure h holds reference data for the down timing loop
                ctx.synchronize();
                result.dn_ref = time_arm(dn_ref, dn_bytes, "dn-ref-nested-div");
                if (want_fp32) time_arm(dn_fp32, dn_fp32_bytes, "dn-ref-fp32");
                if (experts <= 16) time_arm(dn_array, dn_bytes, "dn-ref-arrayform");
                result.dn_uniform = time_arm([&] { return dn_var(opt_div_uniform); }, dn_bytes,
                                             "dn-var-uniform");
                result.dn_fixed = time_arm([&] { return dn_var(opt_fixed_div); }, dn_bytes,
                                           "dn-var-fixed256");
                result.dn_fixed_uniform = time_arm([&] { return dn_var(opt_fixed_uniform); }, dn_bytes,
                                                   "dn-var-fixed256-uniform");
                results.push_back(result);

                // ---- guards ---------------------------------------------------
                {
                    std::vector<bf16> sentinel(64);
                    std::vector<float> dsent(256);
                    ctx.memcpy_d2h(sentinel.data(), h + static_cast<size_t>(experts) * kInter,
                                   sentinel.size() * sizeof(bf16));
                    ctx.memcpy_d2h(dsent.data(), down + static_cast<size_t>(rows) * kTopK * kHidden,
                                   dsent.size() * sizeof(float));
                    ctx.synchronize();
                    bool ok = true;
                    for (bf16 v : sentinel) ok = ok && static_cast<float>(v) == -777.0f;
                    for (float v : dsent) ok = ok && v == -777.0f;
                    std::cout << "[DECODE MOE GUARD] {\"experts\":" << experts << ",\"rows\":" << rows
                              << ",\"sentinel_ok\":" << (ok ? 1 : 0) << "}" << std::endl;
                    if (!ok) throw std::runtime_error("sentinel guard violated");
                }

                ctx.free_device(pool.base);
                ctx.free_device(pool.metadata);
                ctx.free_device(pool.dev_slots);
                if (want_fp32) {
                    ctx.free_device(fp32_pool.base);
                    ctx.free_device(fp32_pool.dev_slots);
                }
            }
        }

        // ---- per-shape gate summary ------------------------------------------
        auto gain = [](double ref, double var) { return (ref - var) / ref * 100.0; };
        for (const auto& r : results) {
            std::cout << "[DECODE MOE GATE] {\"experts\":" << r.experts << ",\"rows\":" << r.rows
                      << ",\"gu_fixed_gain_pct\":" << gain(r.gu_ref, r.gu_fixed)
                      << ",\"gu_uniform_gain_pct\":" << gain(r.gu_ref, r.gu_uniform)
                      << ",\"gu_fixed_uniform_gain_pct\":" << gain(r.gu_ref, r.gu_fixed_uniform)
                      << ",\"dn_fixed_gain_pct\":" << gain(r.dn_ref, r.dn_fixed)
                      << ",\"dn_uniform_gain_pct\":" << gain(r.dn_ref, r.dn_uniform)
                      << ",\"dn_fixed_uniform_gain_pct\":" << gain(r.dn_ref, r.dn_fixed_uniform)
                      << "}" << std::endl;
        }
        if (results.size() >= 2) {
            double gu_min = 1e9, dn_min = 1e9;
            for (const auto& r : results) {
                gu_min = std::min(gu_min, gain(r.gu_ref, r.gu_fixed));
                dn_min = std::min(dn_min, gain(r.dn_ref, r.dn_fixed));
            }
            std::cout << "[DECODE MOE CANDIDATE] {\"candidate\":\"fixed_scale_256\""
                      << ",\"shapes\":" << results.size() << ",\"min_gu_gain_pct\":" << gu_min
                      << ",\"min_dn_gain_pct\":" << dn_min
                      << ",\"verdict\":\"" << ((gu_min >= 3.0 && dn_min >= 3.0) ? "GO" : "NO-GO")
                      << "\",\"rule\":\"charter 5.3: kernel >=3% at both shapes\"}" << std::endl;
        }

        // ---- 48-layer chain (kernel-side floor for experts_dev) --------------
        if (want_chain) {
            const int chain_count = chain_layers * chain_experts;
            SlotPool chain_pool = make_nested_pool(ctx, static_cast<size_t>(chain_count));
            init_nested_pool(ctx, chain_pool, 0xc2b2ae35u);
            std::vector<int> chain_tokens(chain_count, 0);
            std::vector<int> chain_ranks(chain_count);
            for (int l = 0; l < chain_layers; ++l)
                for (int e = 0; e < chain_experts; ++e)
                    chain_ranks[static_cast<size_t>(l) * chain_experts + e] = e % kTopK;
            auto* chain_token_dev = static_cast<int*>(ctx.alloc_device(chain_count * sizeof(int)));
            auto* chain_rank_dev = static_cast<int*>(ctx.alloc_device(chain_count * sizeof(int)));
            ctx.memcpy_h2d(chain_token_dev, chain_tokens.data(), chain_count * sizeof(int));
            ctx.memcpy_h2d(chain_rank_dev, chain_ranks.data(), chain_count * sizeof(int));

            auto run_chain = [&](const char* name, bool variant_fixed, bool variant_uniform) {
                std::vector<sycl::event> events;
                events.reserve(static_cast<size_t>(chain_layers) * 3);
                // Warmup pass (with sync) then one measured pass.
                for (int pass = 0; pass < 2; ++pass) {
                    events.clear();
                    for (int layer = 0; layer < chain_layers; ++layer) {
                        const auto* slots =
                            chain_pool.dev_slots + static_cast<size_t>(layer) * chain_experts;
                        const int* tokens = chain_token_dev + static_cast<size_t>(layer) * chain_experts;
                        const int* layer_ranks =
                            chain_rank_dev + static_cast<size_t>(layer) * chain_experts;
                        sycl::event gu, dn, merge;
                        if (variant_fixed || variant_uniform) {
                            ops::DecodeMoeVariantOptions opts{variant_fixed, variant_uniform};
                            gu = ops::moe_decode_gate_up_variant(
                                ctx, slots, tokens, nullptr, x, h, chain_experts, kHidden, kInter,
                                kBlock, opts, nullptr);
                            dn = ops::moe_decode_down_rank_variant(
                                ctx, slots, layer_ranks, tokens, h, down, chain_experts, kTopK,
                                kInter, kHidden, kBlock, opts, &gu);
                        } else {
                            gu = ops::moe_decode_gate_up_swiglu_grouped(
                                ctx, slots, nullptr, x, h, chain_experts, kHidden, kInter, kBlock,
                                nullptr, tokens, true);
                            dn = ops::moe_decode_down_to_rank_f32(
                                ctx, slots, layer_ranks, h, down, chain_experts, kTopK, kInter,
                                kHidden, kBlock, tokens, nullptr, &gu, true);
                        }
                        merge = ops::moe_ordered_accumulate_down_rows(
                            ctx, down, weights, y, 1, kTopK, kHidden, nullptr, nullptr, &dn);
                        events.push_back(gu);
                        events.push_back(dn);
                        events.push_back(merge);
                    }
                    ctx.synchronize();
                    if (pass == 0) continue;
                    double busy = 0;
                    for (const auto& e : events)
                        busy += static_cast<double>(event_end_ns(e) - event_start_ns(e)) / 1e6;
                    const double wall =
                        static_cast<double>(event_end_ns(events.back()) - event_start_ns(events.front())) / 1e6;
                    std::cout << "[DECODE MOE CHAIN] {\"arm\":\"" << name
                              << "\",\"layers\":" << chain_layers
                              << ",\"experts\":" << chain_experts
                              << ",\"wall_ms\":" << wall << ",\"busy_ms\":" << busy
                              << ",\"gap_ms\":" << (wall - busy)
                              << ",\"per_layer_ms\":" << wall / chain_layers
                              << ",\"kernel_share_pct\":" << busy / wall * 100.0 << "}" << std::endl;
                }
            };
            run_chain("ref-nested-div", false, false);
            run_chain("var-fixed256", true, false);
            run_chain("var-fixed256-uniform", true, true);
            ctx.free_device(chain_pool.base);
            ctx.free_device(chain_pool.metadata);
            ctx.free_device(chain_pool.dev_slots);
            ctx.free_device(chain_token_dev);
            ctx.free_device(chain_rank_dev);
        }

        ctx.free_device(x);
        ctx.free_device(weights);
        ctx.free_device(y);
        ctx.free_device(h);
        ctx.free_device(down);
        ctx.free_device(token_ids);
        ctx.free_device(ranks);
        std::cout << "[DECODE MOE DONE]" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[DECODE MOE FAIL] " << error.what() << std::endl;
        return 1;
    }
}
