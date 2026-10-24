// Micro-probe for the QSA decode attention core.
//
// Why this exists: full-workload VTune runs are far too slow to iterate on, and
// the in-code profiler inflates short spans by 2.5-5x (round-1 document
// section 22). This probe times the kernel in isolation at the production shape
// (24 query heads, 2 KV heads, head_dim 256, 2052 cells) with SYCL events, so a
// kernel change can be judged in seconds instead of through a 46 ms token that
// also contains twenty other stages.
//
// It is a benchmark, not a correctness test: the point is to know what the
// operator costs on its own. When a change wins here but not in the model, the
// cost has moved somewhere else or the operator is not on the critical path, and
// that has to be explained rather than averaged away.

#include "src/ops/AttentionOps.cpp"

// The same stubs the other kernel probes provide: these globals live in the
// engine translation unit, not in the ops implementation.
namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

int g_failures = 0;

void expect(bool condition, const char *what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

}  // namespace

int main(int argc, char **argv) {
    // Cells can be overridden so the operator's scaling can be measured without
    // rebuilding: linear in cells means throughput-bound (fewer reads would
    // help), flat means the per-work-group serial work dominates (more
    // work-groups would help).
    int cells_arg = 2052;
    if (argc > 1) {
        cells_arg = std::atoi(argv[1]);
        if (cells_arg < 256) cells_arg = 256;
    }
    constexpr int kHeads = 24;
    constexpr int kKvHeads = 2;
    constexpr int kHeadDim = 256;
    const int kCells = cells_arg;
    constexpr int kTile = 128;
    constexpr int kStride = 272;
    const int kTiles = (kCells + kTile - 1) / kTile;

    Context ctx(nullptr, /*profiling=*/true);
    auto q = Tensor::allocate(ctx, {1, kHeads * kHeadDim},
                              dnnl::memory::data_type::bf16);
    auto k = Tensor::allocate(ctx, {kKvHeads, kCells, kHeadDim},
                              dnnl::memory::data_type::bf16);
    auto v = Tensor::allocate(ctx, {kKvHeads, kCells, kHeadDim},
                              dnnl::memory::data_type::bf16);
    auto out = Tensor::allocate(ctx, {1, kHeads * kHeadDim},
                                dnnl::memory::data_type::bf16);
    auto partials = Tensor::allocate(ctx, {kHeads, kTiles, kStride},
                                     dnnl::memory::data_type::f32);

    // Deterministic, small-magnitude inputs. Values only have to be realistic in
    // size; the probe measures time, not accuracy.
    std::vector<bf16> host_q(static_cast<size_t>(kHeads) * kHeadDim);
    std::vector<bf16> host_kv(static_cast<size_t>(kKvHeads) * kCells * kHeadDim);
    for (size_t i = 0; i < host_q.size(); ++i) {
        host_q[i] = bf16(0.01f * static_cast<float>((i * 37) % 61) - 0.3f);
    }
    for (size_t i = 0; i < host_kv.size(); ++i) {
        host_kv[i] = bf16(0.01f * static_cast<float>((i * 53) % 67) - 0.33f);
    }
    ctx.memcpy_h2d(q.data(), host_q.data(), host_q.size() * sizeof(bf16));
    ctx.memcpy_h2d(k.data(), host_kv.data(), host_kv.size() * sizeof(bf16));
    ctx.memcpy_h2d(v.data(), host_kv.data(), host_kv.size() * sizeof(bf16));
    ctx.synchronize();

    auto run_once = [&] {
        ops::attention_decode_exact_head256_partial_merge(
            ctx, q, k, v, out, partials, kHeads, kKvHeads,
            /*tail_start=*/0, /*sink_len=*/0, /*effective_len=*/kCells);
    };

    // Warm up off the clock: first call builds the USM residency and the JIT
    // cache, and the pipeline under study has no persistent cache either.
    for (int i = 0; i < 5; ++i) {
        run_once();
    }
    ctx.synchronize();

    constexpr int kSamples = 30;
    // Batch the timed calls between one pair of barriers. A single call between
    // its own barrier pair measures ~180 us regardless of how many cells it
    // touches, which is the profiling-queue barrier and wait cost, not the
    // kernel: at 256 cells the operator still reported 180 us against 217 us at
    // 2052. Amortising over kBatch calls removes that constant.
    constexpr int kBatch = 10;
    std::vector<double> samples;
    samples.reserve(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        auto begin = ctx.queue().ext_oneapi_submit_barrier();
        for (int b = 0; b < kBatch; ++b) {
            run_once();
        }
        auto end = ctx.queue().ext_oneapi_submit_barrier();
        end.wait_and_throw();
        const auto t0 =
            begin.get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto t1 =
            end.get_profiling_info<sycl::info::event_profiling::command_end>();
        samples.push_back(static_cast<double>(t1 - t0) * 1e-3 / kBatch);
    }

    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double best = samples.front();
    const double worst = samples.back();

    // Same batch on a queue WITHOUT enable_profiling, timed with the host clock.
    // The event-timed figure above includes whatever the profiling queue adds to
    // each submission, and the production pipeline does not enable profiling, so
    // this is the number that describes the token.
    double wall_us = -1.0;
    {
        Context plain(nullptr, /*profiling=*/false);
        auto q2 = Tensor::allocate(plain, {1, kHeads * kHeadDim},
                                   dnnl::memory::data_type::bf16);
        auto k2 = Tensor::allocate(plain, {kKvHeads, kCells, kHeadDim},
                                   dnnl::memory::data_type::bf16);
        auto v2 = Tensor::allocate(plain, {kKvHeads, kCells, kHeadDim},
                                   dnnl::memory::data_type::bf16);
        auto o2 = Tensor::allocate(plain, {1, kHeads * kHeadDim},
                                   dnnl::memory::data_type::bf16);
        auto p2 = Tensor::allocate(plain, {kHeads, kTiles, kStride},
                                   dnnl::memory::data_type::f32);
        plain.memcpy_h2d(q2.data(), host_q.data(), host_q.size() * sizeof(bf16));
        plain.memcpy_h2d(k2.data(), host_kv.data(), host_kv.size() * sizeof(bf16));
        plain.memcpy_h2d(v2.data(), host_kv.data(), host_kv.size() * sizeof(bf16));
        plain.synchronize();
        auto call = [&] {
            ops::attention_decode_exact_head256_partial_merge(
                plain, q2, k2, v2, o2, p2, kHeads, kKvHeads, 0, 0, kCells);
        };
        for (int i = 0; i < 5; ++i) call();
        plain.synchronize();
        std::vector<double> walls;
        for (int i = 0; i < kSamples; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int b = 0; b < kBatch; ++b) call();
            plain.synchronize();
            const auto t1 = std::chrono::steady_clock::now();
            walls.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count() /
                kBatch);
        }
        std::sort(walls.begin(), walls.end());
        wall_us = walls[walls.size() / 2];
    }

    // Reference point: 12 QSA layers per token, so this operator's contribution
    // to a token is twelve times the figure below.
    std::printf("cells=%d heads=%d kv_heads=%d head_dim=%d tiles=%d\n", kCells,
                kHeads, kKvHeads, kHeadDim, kTiles);
    std::printf("qsa_decode_attention_us: median=%.1f  best=%.1f  worst=%.1f\n",
                median, best, worst);
    std::printf("qsa_decode_wall_no_profiling_us: %.1f\n", wall_us);
    std::printf("per_token_12_layers_ms: %.2f\n", median * 12.0 / 1000.0);

    // Sanity: the output must not be all zeros, otherwise the probe may have
    // measured an early-out path rather than the operator.
    std::vector<bf16> host_out(static_cast<size_t>(kHeads) * kHeadDim);
    ctx.memcpy_d2h(host_out.data(), out.data(), host_out.size() * sizeof(bf16));
    float magnitude = 0.0f;
    for (const auto value : host_out) {
        magnitude += sycl::fabs(static_cast<float>(value));
    }
    expect(magnitude > 0.0f, "attention output is not all zeros");

    std::printf("%s\n", g_failures == 0 ? "PROBE_OK" : "PROBE_FAILED");
    return g_failures == 0 ? 0 : 1;
}
