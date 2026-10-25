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

int run_prefill_probe() {
    constexpr int kCount = 1024;
    constexpr int kStartPos = 4096;
    constexpr int kMaxSeq = 8192;
    constexpr int kHeads = 24;
    constexpr int kKvHeads = 2;
    constexpr int kHeadDim = 256;
    constexpr int kBlockTopK = 512;
    constexpr int kRatio = 4;
    constexpr int kStride = kBlockTopK + 4;

    Context ctx(nullptr, /*profiling=*/false);
    auto q = Tensor::allocate(ctx, {kCount, kHeads * kHeadDim}, dnnl::memory::data_type::bf16);
    auto gate = Tensor::allocate(ctx, {kCount, kHeads * kHeadDim}, dnnl::memory::data_type::bf16);
    auto k = Tensor::allocate(ctx, {kKvHeads, kMaxSeq, kHeadDim}, dnnl::memory::data_type::bf16);
    auto v = Tensor::allocate(ctx, {kKvHeads, kMaxSeq, kHeadDim}, dnnl::memory::data_type::bf16);
    auto active_blocks = Tensor::allocate(ctx, {kCount, kStride}, dnnl::memory::data_type::s32);
    auto valid_counts = Tensor::allocate(ctx, {kCount, 2}, dnnl::memory::data_type::s32);
    auto out = Tensor::allocate(ctx, {kCount, kHeads * kHeadDim}, dnnl::memory::data_type::bf16);

    std::vector<bf16> h_q(size_t(kCount) * kHeads * kHeadDim);
    std::vector<bf16> h_gate(size_t(kCount) * kHeads * kHeadDim);
    std::vector<bf16> h_kv(size_t(kKvHeads) * kMaxSeq * kHeadDim);
    std::vector<int> h_blocks(size_t(kCount) * kStride, 0);
    std::vector<int> h_counts(size_t(kCount) * 2, 0);

    for (size_t i = 0; i < h_q.size(); ++i) {
        h_q[i] = bf16(0.01f * float((i * 37) % 61) - 0.3f);
        h_gate[i] = bf16(0.02f * float((i * 19) % 43) - 0.4f);
    }
    for (size_t i = 0; i < h_kv.size(); ++i) {
        h_kv[i] = bf16(0.01f * float((i * 53) % 67) - 0.33f);
    }
    for (int r = 0; r < kCount; ++r) {
        const int pos = kStartPos + r;
        const int num_blocks = (pos + 1) / kRatio;
        const int tail = (pos + 1) - num_blocks * kRatio;
        h_counts[r * 2 + 0] = kBlockTopK;
        h_counts[r * 2 + 1] = tail;
        for (int b = 0; b < kBlockTopK; ++b) {
            h_blocks[size_t(r) * kStride + b] = (b * 2) % num_blocks;
        }
    }
    ctx.memcpy_h2d(q.data(), h_q.data(), h_q.size() * sizeof(bf16));
    ctx.memcpy_h2d(gate.data(), h_gate.data(), h_gate.size() * sizeof(bf16));
    ctx.memcpy_h2d(k.data(), h_kv.data(), h_kv.size() * sizeof(bf16));
    ctx.memcpy_h2d(v.data(), h_kv.data(), h_kv.size() * sizeof(bf16));
    ctx.memcpy_h2d(active_blocks.data(), h_blocks.data(), h_blocks.size() * sizeof(int));
    ctx.memcpy_h2d(valid_counts.data(), h_counts.data(), h_counts.size() * sizeof(int));
    ctx.synchronize();

    std::vector<bf16> ref_out(h_q.size());
    auto run_mode = [&](const char* mode_val) {
#ifdef _WIN32
        _putenv_s("AILA_QSA_HEADS_PER_WG", mode_val);
#else
        setenv("AILA_QSA_HEADS_PER_WG", mode_val, 1);
#endif
        auto call = [&] {
            ops::attention_prefill_qsa_batched(
                ctx, q, k, v, active_blocks, out, kCount, kStartPos,
                kHeads, kKvHeads, kHeadDim, kBlockTopK, kRatio,
                &gate, &valid_counts, nullptr, nullptr);
        };
        for (int w = 0; w < 2; ++w) call();
        ctx.synchronize();
        std::vector<double> ms;
        for (int s = 0; s < 7; ++s) {
            const auto t0 = std::chrono::steady_clock::now();
            call();
            ctx.synchronize();
            ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };

    const double ms2 = run_mode("2");
    ctx.memcpy_d2h(ref_out.data(), out.data(), ref_out.size() * sizeof(bf16));

    for (const char* m : {"25"}) {
        const double t_ms = run_mode(m);
        std::vector<bf16> cand_out(h_q.size());
        ctx.memcpy_d2h(cand_out.data(), out.data(), cand_out.size() * sizeof(bf16));
        double max_abs = 0.0, sq_err = 0.0, sq_ref = 0.0;
        for (size_t i = 0; i < ref_out.size(); ++i) {
            const double r = double(float(ref_out[i]));
            const double c = double(float(cand_out[i]));
            const double d = std::abs(r - c);
            if (d > max_abs) max_abs = d;
            sq_err += d * d;
            sq_ref += r * r;
        }
        const double nrmse = std::sqrt(sq_err / std::max(sq_ref, 1e-30));
        std::printf("[QSA Prefill 1024-tile] mode=%s median=%.2fms (84x=%.2fs) vs mode2=%.2fms (84x=%.2fs) speedup=%.2fx nrmse=%.2e max_abs=%.4f\n",
                    m, t_ms, t_ms * 84.0 / 1000.0, ms2, ms2 * 84.0 / 1000.0, ms2 / t_ms, nrmse, max_abs);
        expect(nrmse < 1e-4 && max_abs <= 0.02, "QSA prefill mode parity within bf16 ulp");
    }
    return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1 && std::strcmp(argv[1], "--prefill") == 0) {
        return run_prefill_probe();
    }
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
