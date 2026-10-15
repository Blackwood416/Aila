// Qwen4Exp layer-stage overlap bench (role E7, 2026-10-15).
//
// Question: can speculative whole-layer expert staging (host fill + copy
// queue H2D into a 454-slot arena, ~1.07 GiB) run concurrently with a
// VRAM-bound attention/GDN-type device load without stealing more than a
// fraction of it? The load is the production GDN prefill kernel at the real
// layer geometry (vh=48, rows=4096; K4 anchor 74.8 ms/layer). One round is
// `--gdn-calls` GDN calls on the compute queue plus one staging pass.
//
// Arms:
//   load     GDN batch alone
//   stage    staging alone (Begin -> WaitReady)
//   overlap  GDN batch submitted, then staging kicked off while it runs
//
// Gate (charter section 4): median overlap load inflation <= 30% of the
// median stage-only wall, otherwise the overlap premise is NO-GO.
//
// Guards: GDN output finite check; arena payload bit-compare against the
// source pool (first and last slot headers); module slot/descriptor math.
// ASCII only; no allocation inside the timed rounds.

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "models/Qwen4ExpLayerStage.hpp"
#include "ops/GatedDeltaNetOps.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int H = 2560;
constexpr int I = 640;
constexpr size_t Packed = 2 * size_t(I) * H / 2;
constexpr size_t Codes = 2 * size_t(I) * H / 64;
constexpr size_t DownPacked = size_t(H) * I / 2;
constexpr size_t DownCodes = size_t(H) * I / 64;
constexpr size_t DownOffset = Packed + Codes;
constexpr size_t DownCodesOffset = DownOffset + DownPacked;
constexpr size_t SlotBytes = DownCodesOffset + DownCodes;
static_assert(SlotBytes == 2534400, "slot layout must match the production expert slot");

double ms_since(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

aila::ExpertSlotGeometry compact_geometry() {
    aila::ExpertSlotGeometry g;
    g.gu_packed_bytes = Packed;
    g.gu_absmax_count = Codes;
    g.gu_absmax_bytes = Codes;
    g.dn_packed_bytes = DownPacked;
    g.dn_absmax_count = DownCodes;
    g.dn_absmax_bytes = DownCodes;
    g.gu_packed_offset = 0;
    g.gu_absmax_offset = Packed;
    g.dn_packed_offset = DownOffset;
    g.dn_absmax_offset = DownCodesOffset;
    g.slot_bytes = SlotBytes;
    g.compact_nested = true;
    return g;
}

struct Params {
    int experts = 454;
    int gdn_calls = 2;
    int samples = 5;
    int chunk_slots = 16;
    int fill_banks = 2;
    int qh = 16;
    int vh = 48;
    int rows = 4096;
    bool arm_load = true;
    bool arm_stage = true;
    bool arm_overlap = true;
};

struct Sample {
    double load_ms = 0;
    double stage_ms = 0;
    double fill_ms = 0;
    double bankwait_ms = 0;
};

// Minimal production GDN prefill fixture (mirrors perf/qwen4exp/q4e_gdn_bench.cpp).
struct GdnFixture {
    Context& ctx;
    ops::DeltaNetGeometry g;
    int count;
    int initial_head = 2;
    std::vector<bf16> input, cw, dt;
    std::vector<float> initial, history, nw, al;
    Tensor in, conv_w, norm_w, a_log, dt_bias, state, conv_state, out;

    GdnFixture(Context& c, int tokens, int qh, int vh)
        : ctx(c),
          g(ops::DeltaNetGeometry::create(qh, vh, 128, 128, 4, 1e-6f,
                                          ops::GateActivation::Sigmoid)),
          count(tokens) {
        std::mt19937 rng(4917);
        const auto rnd = [&]() { return (int(rng() % 2001) - 1000) / 1000.f; };
        input.resize(size_t(count) * g.all_dim);
        for (auto& x : input) x = bf16(rnd() * .3f);
        initial.resize(size_t(vh) * 128 * 128);
        for (auto& x : initial) x = rnd() * .02f;
        history.resize(3 * g.qkv_dim);
        for (auto& x : history) x = rnd() * .2f;
        cw.resize(g.qkv_dim * 4);
        for (auto& x : cw) x = bf16(rnd() * .4f);
        nw.resize(128);
        for (auto& x : nw) x = 1.f + rnd() * .1f;
        al.resize(vh);
        for (auto& x : al) x = -2.f + rnd() * .5f;
        dt.resize(vh);
        for (auto& x : dt) x = bf16(-1.f + rnd() * .5f);
        const auto alloc = [&](size_t n, dnnl::memory::data_type t) {
            return Tensor::allocate(ctx, {static_cast<int64_t>(n)}, t);
        };
        using type = dnnl::memory::data_type;
        in = alloc(input.size(), type::bf16);
        conv_w = alloc(cw.size(), type::bf16);
        norm_w = alloc(nw.size(), type::f32);
        a_log = alloc(al.size(), type::f32);
        dt_bias = alloc(dt.size(), type::bf16);
        state = alloc(initial.size(), type::f32);
        conv_state = alloc(history.size(), type::f32);
        out = alloc(size_t(count) * g.kv_dim + 16, type::bf16);
        ctx.memcpy_h2d(in.data(), input.data(), in.size_bytes());
        ctx.memcpy_h2d(conv_w.data(), cw.data(), conv_w.size_bytes());
        ctx.memcpy_h2d(norm_w.data(), nw.data(), norm_w.size_bytes());
        ctx.memcpy_h2d(a_log.data(), al.data(), a_log.size_bytes());
        ctx.memcpy_h2d(dt_bias.data(), dt.data(), dt_bias.size_bytes());
    }

    ops::DeltaNetWeights weights() {
        return {conv_w.data_as<bf16>(), norm_w.data_as<float>(),
                a_log.data_as<float>(), dt_bias.data_as<bf16>()};
    }

    // Async state restore on the compute queue (in-order): keeps successive
    // rounds independent without draining the pipeline.
    void queue_reset() {
        ctx.queue().memcpy(state.data(), initial.data(), state.size_bytes());
        ctx.queue().memcpy(conv_state.data(), history.data(), conv_state.size_bytes());
    }

    void queue_run() {
        ops::DeltaNetState st{state.data_as<float>(), conv_state.data_as<float>(),
                              initial_head, false};
        ops::run_linear_delta_prefill_gpu_batched(ctx, g, weights(), st,
                                                  in.data_as<bf16>(),
                                                  out.data_as<bf16>(), count);
    }
};

Params parse(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--experts") p.experts = std::stoi(value);
        else if (key == "--gdn-calls") p.gdn_calls = std::stoi(value);
        else if (key == "--samples") p.samples = std::stoi(value);
        else if (key == "--chunk-slots") p.chunk_slots = std::stoi(value);
        else if (key == "--fill-banks") p.fill_banks = std::stoi(value);
        else if (key == "--rows") p.rows = std::stoi(value);
        else if (key == "--arms") {
            p.arm_load = value.find("load") != std::string::npos;
            p.arm_stage = value.find("stage") != std::string::npos;
            p.arm_overlap = value.find("overlap") != std::string::npos;
        } else {
            throw std::runtime_error("unknown flag " + key);
        }
    }
    check(p.experts >= 1 && p.experts <= 512, "experts must be 1..512");
    check(p.gdn_calls >= 1 && p.samples >= 1, "bad gdn-calls/samples");
    check(p.chunk_slots >= 1 && p.fill_banks >= 2 && p.fill_banks <= 8, "bad chunk/banks");
    check(p.rows > 0 && p.rows % 4 == 0, "rows must be a multiple of 4");
    check(p.arm_load || p.arm_stage || p.arm_overlap, "no arms selected");
    return p;
}

void run_load(Context& ctx, GdnFixture& f, const Params& p, Sample& s) {
    const auto t0 = Clock::now();
    for (int k = 0; k < p.gdn_calls; ++k) {
        f.queue_reset();
        f.queue_run();
    }
    auto barrier = ctx.queue().ext_oneapi_submit_barrier();
    barrier.wait_and_throw();
    s.load_ms = ms_since(t0);
}

uint64_t run_stage(aila::Qwen4ExpLayerStage& stage, const std::vector<int>& ids,
                   const std::vector<const uint8_t*>& srcs, sycl::queue& copy,
                   int layer, Sample& s) {
    const auto t0 = Clock::now();
    const uint64_t epoch = stage.Begin(layer, ids, srcs, copy);
    stage.WaitReady(epoch);
    s.stage_ms = ms_since(t0);
    const auto stats = stage.Stats();
    s.fill_ms = stats.last_fill_ms;
    s.bankwait_ms = stats.last_bank_wait_ms;
    return epoch;
}

void run_overlap(Context& ctx, GdnFixture& f, aila::Qwen4ExpLayerStage& stage,
                 const std::vector<int>& ids, const std::vector<const uint8_t*>& srcs,
                 sycl::queue& copy, int layer, const Params& p, Sample& s) {
    const auto t0 = Clock::now();
    for (int k = 0; k < p.gdn_calls; ++k) {
        f.queue_reset();
        f.queue_run();
    }
    auto barrier = ctx.queue().ext_oneapi_submit_barrier();
    const uint64_t epoch = stage.Begin(layer, ids, srcs, copy);
    std::exception_ptr worker_error;
    std::thread waiter([&] {
        try {
            stage.WaitReady(epoch);
        } catch (...) {
            worker_error = std::current_exception();
        }
    });
    barrier.wait_and_throw();
    s.load_ms = ms_since(t0);
    // Mimics production: the next epoch's first H2D waits this round's load.
    stage.RecordArenaConsumer(epoch, barrier);
    waiter.join();
    if (worker_error) std::rethrow_exception(worker_error);
    s.stage_ms = ms_since(t0);
    const auto stats = stage.Stats();
    s.fill_ms = stats.last_fill_ms;
    s.bankwait_ms = stats.last_bank_wait_ms;
}

void guard_outputs(Context& ctx, const GdnFixture& f) {
    std::array<bf16, 8> head{};
    ctx.memcpy_d2h(head.data(), f.out.data_as<bf16>(), head.size() * sizeof(bf16));
    for (const auto v : head) check(std::isfinite(float(v)), "non-finite GDN output");
}

void guard_arena(Context& ctx, aila::Qwen4ExpLayerStage& stage, uint64_t epoch,
                 const std::vector<uint8_t>& pool, int experts) {
    const size_t probe = 4096;
    check(stage.SlotOf(epoch, 0) == 0, "slot map mismatch for expert 0");
    check(stage.SlotOf(epoch, experts - 1) == experts - 1, "slot map mismatch for last expert");
    const auto d = stage.GetSlotDev(epoch, experts - 1, nullptr);
    check(d.gate_up_packed != nullptr && d.down_packed != nullptr,
          "stage descriptor is empty");
    std::vector<uint8_t> a(probe), b(probe);
    ctx.memcpy_d2h(a.data(), d.gate_up_packed, probe);
    ctx.memcpy_d2h(b.data(), d.down_packed, probe);
    check(std::memcmp(a.data(), pool.data() + size_t(experts - 1) * SlotBytes, probe) == 0,
          "arena gate_up payload mismatch");
    check(std::memcmp(b.data(), pool.data() + size_t(experts - 1) * SlotBytes + DownOffset,
                      probe) == 0,
          "arena down payload mismatch");
    const int routes[3] = {0, experts - 1, experts < 512 ? experts : 511};
    const auto delta = stage.MissedExperts(epoch, routes, 3);
    const bool edge_is_staged = routes[2] <= experts - 1;
    check(delta.staged_hits == (edge_is_staged ? 3 : 2), "route delta hits mismatch");
    check(edge_is_staged ? delta.missed.empty() : delta.missed.size() == 1,
          "route delta miss mismatch");
}
}  // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf << std::fixed << std::setprecision(3);
    aila::env::enable_l0_copy_engine();
    try {
        const Params p = parse(argc, argv);
        Context ctx(nullptr, true);
        sycl::queue& q = ctx.queue();
        auto* copy = new sycl::queue(
            q.get_context(), q.get_device(),
            sycl::property_list{sycl::property::queue::in_order(),
                                sycl::ext::intel::property::queue::immediate_command_list(),
                                sycl::property::queue::enable_profiling()});
        struct Drain {
            Context& ctx;
            sycl::queue* copy;
            ~Drain() {
                try { copy->wait_and_throw(); } catch (...) {}
                try { ctx.synchronize(); } catch (...) {}
                delete copy;
            }
        } drain{ctx, copy};

        std::vector<uint8_t> pool(size_t(p.experts) * SlotBytes);
        for (size_t w = 0; w < pool.size() / 4; ++w)
            reinterpret_cast<uint32_t*>(pool.data())[w] = mix(uint32_t(w) + 41);
        std::vector<int> ids(static_cast<size_t>(p.experts));
        std::vector<const uint8_t*> srcs(static_cast<size_t>(p.experts));
        for (int e = 0; e < p.experts; ++e) {
            ids[static_cast<size_t>(e)] = e;
            srcs[static_cast<size_t>(e)] = pool.data() + size_t(e) * SlotBytes;
        }

        aila::Qwen4ExpLayerStage stage;
        std::string error;
        check(stage.Allocate(ctx, compact_geometry(),
                             aila::LayerStageConfig{p.experts, p.experts, p.chunk_slots,
                                                    p.fill_banks},
                             &error),
              error.empty() ? "layer stage allocation failed" : error.c_str());

        GdnFixture f(ctx, p.rows, p.qh, p.vh);
        std::cout << "[LS DEVICE] " << q.get_device().get_info<sycl::info::device::name>()
                  << " driver=" << q.get_device().get_info<sycl::info::device::driver_version>()
                  << '\n'
                  << "[LS CONFIG] experts=" << p.experts << " chunk_slots=" << p.chunk_slots
                  << " fill_banks=" << p.fill_banks << " gdn_calls=" << p.gdn_calls
                  << " rows=" << p.rows << " vh=" << p.vh << " qh=" << p.qh
                  << " arena_mb=" << (stage.ArenaBytes() >> 20)
                  << " host_banks_mb=" << (stage.HostBytes() >> 20)
                  << " stage_bytes=" << uint64_t(p.experts) * SlotBytes << '\n';

        std::vector<double> load_w, stage_w, overlap_load_w, overlap_stage_w;
        std::vector<double> fill_w, bank_w;
        for (int rep = -1; rep < p.samples; ++rep) {
            Sample s;
            if (p.arm_load) {
                run_load(ctx, f, p, s);
                if (rep >= 0) {
                    load_w.push_back(s.load_ms);
                    std::cout << "[LS TIME] arm=load sample=" << rep
                              << " load_ms=" << s.load_ms << '\n';
                }
            }
            if (p.arm_stage) {
                run_stage(stage, ids, srcs, *copy, 0, s);
                if (rep >= 0) {
                    stage_w.push_back(s.stage_ms);
                    fill_w.push_back(s.fill_ms);
                    bank_w.push_back(s.bankwait_ms);
                    std::cout << "[LS TIME] arm=stage sample=" << rep
                              << " stage_ms=" << s.stage_ms << " fill_ms=" << s.fill_ms
                              << " bankwait_ms=" << s.bankwait_ms << '\n';
                }
            }
            if (p.arm_overlap) {
                run_overlap(ctx, f, stage, ids, srcs, *copy, 0, p, s);
                if (rep >= 0) {
                    overlap_load_w.push_back(s.load_ms);
                    overlap_stage_w.push_back(s.stage_ms);
                    std::cout << "[LS TIME] arm=overlap sample=" << rep
                              << " load_ms=" << s.load_ms << " stage_ms=" << s.stage_ms
                              << " fill_ms=" << s.fill_ms
                              << " bankwait_ms=" << s.bankwait_ms << '\n';
                }
            }
        }
        stage.WaitIdle();
        stage.Reset();
        // Guards run outside the timed rounds, once each, on a fresh round.
        {
            Sample s;
            run_load(ctx, f, p, s);
            guard_outputs(ctx, f);
            const uint64_t e = run_stage(stage, ids, srcs, *copy, 0, s);
            guard_arena(ctx, stage, e, pool, p.experts);
        }

        const double load_m = median(load_w);
        const double stage_m = median(stage_w);
        const double ov_load_m = median(overlap_load_w);
        const double ov_stage_m = median(overlap_stage_w);
        if (p.arm_load && p.arm_stage && p.arm_overlap) {
            const double inflation = ov_load_m - load_m;
            const double ratio = stage_m > 0 ? inflation / stage_m : 1.0;
            const double stage_gbps =
                stage_m > 0 ? double(p.experts) * double(SlotBytes) / (stage_m * 1e6) : 0.0;
            std::cout << "[LS MEDIAN] load_ms=" << load_m << " stage_ms=" << stage_m
                      << " overlap_load_ms=" << ov_load_m
                      << " overlap_stage_ms=" << ov_stage_m
                      << " stage_gbps=" << stage_gbps
                      << " fill_ms=" << median(fill_w)
                      << " bankwait_ms=" << median(bank_w) << '\n';
            std::cout << "[LS GATE] {\"inflation_ms\":" << inflation
                      << ",\"stage_ms\":" << stage_m << ",\"ratio\":" << ratio
                      << ",\"stage_slow_ratio\":"
                      << (stage_m > 0 ? ov_stage_m / stage_m : 0.0)
                      << ",\"verdict\":\"" << (ratio <= 0.30 ? "overlap_ok" : "no_go")
                      << "\"}\n";
        } else if (p.arm_stage) {
            std::cout << "[LS MEDIAN] stage_ms=" << stage_m << " fill_ms=" << median(fill_w)
                      << " bankwait_ms=" << median(bank_w) << '\n';
        } else {
            std::cout << "[LS MEDIAN] load_ms=" << load_m << '\n';
        }
        std::cout << "[LS GUARD] finite=pass arena_bitwise=pass slot_map=pass\n";
        std::cout << "[LS DONE]\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[LS FAIL] " << e.what() << '\n';
        return 1;
    }
}
