// ESIMD DPAS probe suite for A770/DG2 (probe-first, opt-in; no production wiring).
//
// Answers, on this machine/toolchain and before any production integration:
//   1. exactness of esimd::xmx::dpas s8/u8 K-chained accumulation vs a CPU
//      int32 reference (the MoE path must keep integer partials exact);
//   2. operand sourcing costs: registers vs per-iteration global reload vs SLM;
//   3. micro-throughput vs the SG8 joint_matrix path (6b7286a) and vs scalar
//      int8 dots (the current wgshare-style math);
//   4. VNNI activation packing and NF4->s8 staging cost at tile sizes;
//   5. one MoE-shaped tile (M=8 rows x N=8 tokens x K=256, per-64 scales) with
//      the fp32 scale epilogue a production kernel would use.
//
// Layout contract exercised here (staged, produced by a packing step):
//   W: [k_tile][row][32]   contiguous int8 codes per K=32 tile
//   X: [k_tile][32][N]     VNNI: index(k,n) = (k/4)*N*4 + n*4 + (k%4)
//   C: M x N int32 per dpas result, row-major
// Upstream dpas e2e tests use the same VNNI index formula ("vertically packed").
//
// Included inside the anonymous namespace of Qwen4ExpKernelTests.cpp.
// Requires <sycl/ext/intel/esimd.hpp> (and intel info) at TU scope.

#include <sycl/ext/intel/esimd.hpp>

// The EU-width query lives behind an optional header that is not present in
// every oneAPI install; keep the probe compiling without it.
#if __has_include(<sycl/ext/intel/info/device.hpp>)
#include <sycl/ext/intel/info/device.hpp>
#define EPROBE_HAS_INTEL_INFO 1
#else
#define EPROBE_HAS_INTEL_INFO 0
#endif

namespace esimd_ns = sycl::ext::intel::esimd;
namespace esimd_xmx_ns = sycl::ext::intel::esimd::xmx;

template <class E>
constexpr esimd_xmx_ns::dpas_argument_type eprobe_prec() {
    if constexpr (std::is_same_v<E, int8_t>) {
        return esimd_xmx_ns::dpas_argument_type::s8;
    } else if constexpr (std::is_same_v<E, uint8_t>) {
        return esimd_xmx_ns::dpas_argument_type::u8;
    } else {
        return esimd_xmx_ns::dpas_argument_type::Invalid;
    }
}

constexpr int eprobe_vnni_index(int kk, int n, int ncols) {
    return ((kk / 4) * ncols + n) * 4 + (kk % 4);
}

// Deterministic code patterns; a and b differ so a swapped operand is caught.
template <class E>
int eprobe_weight_code(int row, int k) {
    const int span = std::is_signed_v<E> ? 255 : 256;
    const int raw = (row * 131 + k * 17 + 401) % span;
    return std::is_signed_v<E> ? raw - 127 : raw;
}

template <class E>
int eprobe_act_code(int k, int n) {
    const int span = std::is_signed_v<E> ? 255 : 256;
    const int raw = (k * 29 + n * 11 + 907) % span;
    return std::is_signed_v<E> ? raw - 127 : raw;
}

template <class E>
dnnl::memory::data_type eprobe_dtype() {
    return std::is_signed_v<E> ? dnnl::memory::data_type::s8
                               : dnnl::memory::data_type::u8;
}

template <int RC, int K, int N, class WT, class AT>
struct EprobeExactTag {};

// Exact M x K x N case: K-chained dpas tiles, staged layouts, integer compare.
template <int RC, int K, int N, class WT, class AT>
void eprobe_run_exact(Context& ctx, const std::string& label) {
    static_assert(K % 32 == 0, "K must be a multiple of 32");
    constexpr int KT = K / 32;
    std::vector<WT> w(size_t(RC) * K);
    std::vector<AT> x(size_t(K) * N);
    for (int kt = 0; kt < KT; ++kt) {
        for (int m = 0; m < RC; ++m) {
            for (int kk = 0; kk < 32; ++kk) {
                w[(size_t(kt) * RC + m) * 32 + kk] =
                    static_cast<WT>(eprobe_weight_code<WT>(m, kt * 32 + kk));
            }
        }
        for (int kk = 0; kk < 32; ++kk) {
            for (int n = 0; n < N; ++n) {
                x[size_t(kt) * 32 * N + eprobe_vnni_index(kk, n, N)] =
                    static_cast<AT>(eprobe_act_code<AT>(kt * 32 + kk, n));
            }
        }
    }
    std::vector<int32_t> ref(size_t(RC) * N, 0);
    for (int m = 0; m < RC; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                const int wv = int(w[(size_t(k / 32) * RC + m) * 32 + (k % 32)]);
                // X is staged per K32 tile with tile stride 32*N bytes.
                const int xv = int(x[(size_t(k / 32) * 32 * N + eprobe_vnni_index(k % 32, n, N))]);
                acc += wv * xv;
            }
            ref[size_t(m) * N + n] = acc;
        }
    }
    auto dw = Tensor::allocate(ctx, {int64_t(RC) * K}, eprobe_dtype<WT>());
    auto dx = Tensor::allocate(ctx, {int64_t(K) * N}, eprobe_dtype<AT>());
    auto dout = Tensor::allocate(ctx, {int64_t(RC) * N}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(dw.data(), w.data(), w.size());
    ctx.memcpy_h2d(dx.data(), x.data(), x.size());
    WT* wp = dw.data_as<WT>();
    AT* xp = dx.data_as<AT>();
    int32_t* op = dout.data_as<int32_t>();
    ctx.queue().submit([&](sycl::handler& h) {
        h.single_task<EprobeExactTag<RC, K, N, WT, AT>>([=]() SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            simd<int32_t, RC * N> c(0);
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                simd<WT, RC * 32> wt(wp + size_t(kt) * RC * 32, overaligned_tag<16>{});
                simd<AT, 32 * N> at(xp + size_t(kt) * 32 * N, overaligned_tag<16>{});
                c = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, AT, WT,
                                       eprobe_prec<AT>(), eprobe_prec<WT>()>(c, at, wt);
            }
            c.copy_to(op);
        });
    }).wait_and_throw();
    std::vector<int32_t> got(ref.size());
    ctx.memcpy_d2h(got.data(), dout.data(), got.size() * sizeof(int32_t));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (got[i] != ref[i]) {
            throw std::runtime_error(
                "ESIMD DPAS exact mismatch [" + label + "] idx=" + std::to_string(i) +
                " got=" + std::to_string(got[i]) + " ref=" + std::to_string(ref[i]));
        }
    }
    std::cout << "[ESIMD-DPAS] exact " << label << ": " << ref.size()
              << "/" << ref.size() << " int32 match (K-chained, staged layouts)\n";
}

// SLM operand source: stage A tile through SLM, then read it back per dpas.
template <int RC, int K, int N>
struct EprobeSlmTag {};

template <int RC, int K, int N>
void eprobe_run_slm(Context& ctx, const std::string& label) {
    static_assert(K % 32 == 0, "K must be a multiple of 32");
    constexpr int KT = K / 32;
    std::vector<int8_t> w(size_t(RC) * K);
    std::vector<int8_t> x(size_t(K) * N);
    for (int kt = 0; kt < KT; ++kt) {
        for (int m = 0; m < RC; ++m) {
            for (int kk = 0; kk < 32; ++kk) {
                w[(size_t(kt) * RC + m) * 32 + kk] =
                    int8_t(eprobe_weight_code<int8_t>(m, kt * 32 + kk));
            }
        }
        for (int kk = 0; kk < 32; ++kk) {
            for (int n = 0; n < N; ++n) {
                x[size_t(kt) * 32 * N + eprobe_vnni_index(kk, n, N)] =
                    int8_t(eprobe_act_code<int8_t>(kt * 32 + kk, n));
            }
        }
    }
    std::vector<int32_t> ref(size_t(RC) * N, 0);
    for (int m = 0; m < RC; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += int(w[(size_t(k / 32) * RC + m) * 32 + (k % 32)]) *
                       int(x[(size_t(k / 32) * 32 * N + eprobe_vnni_index(k % 32, n, N))]);
            }
            ref[size_t(m) * N + n] = acc;
        }
    }
    auto dw = Tensor::allocate(ctx, {int64_t(RC) * K}, dnnl::memory::data_type::s8);
    auto dx = Tensor::allocate(ctx, {int64_t(K) * N}, dnnl::memory::data_type::s8);
    auto dout = Tensor::allocate(ctx, {int64_t(RC) * N}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(dw.data(), w.data(), w.size());
    ctx.memcpy_h2d(dx.data(), x.data(), x.size());
    int8_t* wp = dw.data_as<int8_t>();
    int8_t* xp = dx.data_as<int8_t>();
    int32_t* op = dout.data_as<int32_t>();
    ctx.queue().submit([&](sycl::handler& h) {
        sycl::local_accessor<int8_t, 1> slm(size_t(RC) * K + 16, h);
        h.single_task<EprobeSlmTag<RC, K, N>>([=]() SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            simd<int32_t, RC * N> c(0);
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                simd<int8_t, RC * 32> wt_global(wp + size_t(kt) * RC * 32,
                                                overaligned_tag<16>{});
                slm_block_store<int8_t, RC * 32>(size_t(kt) * RC * 32, wt_global);
                simd<int8_t, RC * 32> wt = slm_block_load<int8_t, RC * 32>(
                    size_t(kt) * RC * 32);
                simd<int8_t, 32 * N> at(xp + size_t(kt) * 32 * N, overaligned_tag<16>{});
                c = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                       eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(c, at, wt);
            }
            c.copy_to(op);
        });
    }).wait_and_throw();
    std::vector<int32_t> got(ref.size());
    ctx.memcpy_d2h(got.data(), dout.data(), got.size() * sizeof(int32_t));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (got[i] != ref[i]) {
            throw std::runtime_error("ESIMD DPAS SLM mismatch [" + label + "] idx=" +
                                     std::to_string(i) + " got=" + std::to_string(got[i]) +
                                     " ref=" + std::to_string(ref[i]));
        }
    }
    std::cout << "[ESIMD-DPAS] exact " << label << ": " << ref.size()
              << " int32 match (A staged via SLM)\n";
}

// MoE-shaped tile: M=8 weight rows, N=8 tokens, K=256, scales per 64-K block.
// The combined scale (w_absmax/127)*(x_absmax/127) is materialized per
// (row, token, block) and applied as an fp32 vector epilogue.
template <int ROWS, int TOKS, int K, int BLOCK>
struct EprobeMoeTileTag {};

template <int ROWS, int TOKS, int K, int BLOCK>
void eprobe_run_moe_tile(Context& ctx) {
    static_assert(K % BLOCK == 0 && BLOCK % 32 == 0, "block geometry");
    static_assert(BLOCK == 64, "per-64 nested scales (production convention)");
    constexpr int KT = K / 32;
    constexpr int NB = K / BLOCK;
    std::vector<int8_t> w(size_t(ROWS) * K);
    std::vector<int8_t> x(size_t(K) * TOKS);
    std::vector<float> w_abs(size_t(ROWS) * NB), x_abs(size_t(TOKS) * NB);
    for (int kt = 0; kt < KT; ++kt) {
        for (int m = 0; m < ROWS; ++m) {
            for (int kk = 0; kk < 32; ++kk) {
                w[(size_t(kt) * ROWS + m) * 32 + kk] =
                    int8_t(eprobe_weight_code<int8_t>(m, kt * 32 + kk));
            }
        }
        for (int kk = 0; kk < 32; ++kk) {
            for (int n = 0; n < TOKS; ++n) {
                x[size_t(kt) * 32 * TOKS + eprobe_vnni_index(kk, n, TOKS)] =
                    int8_t(eprobe_act_code<int8_t>(kt * 32 + kk, n));
            }
        }
    }
    for (int m = 0; m < ROWS; ++m) {
        for (int b = 0; b < NB; ++b) {
            w_abs[size_t(m) * NB + b] = 0.030f + 0.0015f * float((m * 7 + b * 3) % 11);
        }
    }
    for (int n = 0; n < TOKS; ++n) {
        for (int b = 0; b < NB; ++b) {
            x_abs[size_t(n) * NB + b] = 0.120f + 0.006f * float((n * 5 + b * 2) % 9);
        }
    }
    // combined scale vectors, one ROWS*TOKS vector per 64-K block
    std::vector<float> comb(size_t(NB) * ROWS * TOKS);
    for (int b = 0; b < NB; ++b) {
        for (int m = 0; m < ROWS; ++m) {
            for (int n = 0; n < TOKS; ++n) {
                comb[(size_t(b) * ROWS + m) * TOKS + n] =
                    (w_abs[size_t(m) * NB + b] / 127.f) * (x_abs[size_t(n) * NB + b] / 127.f);
            }
        }
    }
    std::vector<float> ref(size_t(ROWS) * TOKS, 0.f);
    for (int m = 0; m < ROWS; ++m) {
        for (int n = 0; n < TOKS; ++n) {
            double acc = 0.0;
            for (int b = 0; b < NB; ++b) {
                int64_t partial = 0;
                for (int k = b * BLOCK; k < (b + 1) * BLOCK; ++k) {
                    partial += int64_t(w[(size_t(k / 32) * ROWS + m) * 32 + (k % 32)]) *
                               int64_t(x[size_t(k / 32) * 32 * TOKS +
                                         eprobe_vnni_index(k % 32, n, TOKS)]);
                }
                acc += double(partial) * double(comb[(size_t(b) * ROWS + m) * TOKS + n]);
            }
            ref[size_t(m) * TOKS + n] = float(acc);
        }
    }
    auto dw = Tensor::allocate(ctx, {int64_t(ROWS) * K}, dnnl::memory::data_type::s8);
    auto dx = Tensor::allocate(ctx, {int64_t(K) * TOKS}, dnnl::memory::data_type::s8);
    auto ds = Tensor::allocate(ctx, {int64_t(NB) * ROWS * TOKS}, dnnl::memory::data_type::f32);
    auto dout = Tensor::allocate(ctx, {int64_t(ROWS) * TOKS}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(dw.data(), w.data(), w.size());
    ctx.memcpy_h2d(dx.data(), x.data(), x.size());
    ctx.memcpy_h2d(ds.data(), comb.data(), comb.size() * sizeof(float));
    int8_t* wp = dw.data_as<int8_t>();
    int8_t* xp = dx.data_as<int8_t>();
    float* sp = ds.data_as<float>();
    float* op = dout.data_as<float>();
    ctx.queue().submit([&](sycl::handler& h) {
        h.single_task<EprobeMoeTileTag<ROWS, TOKS, K, BLOCK>>([=]() SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            simd<float, ROWS * TOKS> acc(0.f);
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                simd<int32_t, ROWS * TOKS> p(0);
#pragma unroll
                for (int half = 0; half < BLOCK / 32; ++half) {
                    const int kt = b * (BLOCK / 32) + half;
                    simd<int8_t, ROWS * 32> wt(wp + size_t(kt) * ROWS * 32,
                                               overaligned_tag<16>{});
                    simd<int8_t, 32 * TOKS> at(xp + size_t(kt) * 32 * TOKS,
                                               overaligned_tag<16>{});
                    p = esimd_xmx_ns::dpas<8, ROWS, int32_t, int32_t, int8_t, int8_t,
                                           eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(p, at, wt);
                }
                simd<float, ROWS * TOKS> s(sp + size_t(b) * ROWS * TOKS,
                                           overaligned_tag<16>{});
                simd<float, ROWS * TOKS> pf(p);
                acc = acc + pf * s;
            }
            acc.copy_to(op);
        });
    }).wait_and_throw();
    std::vector<float> got(ref.size());
    ctx.memcpy_d2h(got.data(), dout.data(), got.size() * sizeof(float));
    float max_abs = 0.f, max_rel = 0.f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        const float denom = std::max(1e-3f, std::fabs(ref[i]));
        max_rel = std::max(max_rel, d / denom);
    }
    std::cout << "[ESIMD-DPAS] moe-tile m" << ROWS << " k" << K << " n" << TOKS
              << " per-64 scales: max_abs=" << max_abs << " max_rel=" << max_rel
              << " (fp32 epilogue, int partials exact)\n";
    if (max_rel > 1e-4f) {
        throw std::runtime_error("ESIMD DPAS moe-tile scale epilogue mismatch: max_rel=" +
                                 std::to_string(max_rel));
    }
}

// ---------------------------------------------------------------------------
// Throughput section
// ---------------------------------------------------------------------------

// Timing contract: 3 warmup submissions, then 20 timed samples; report median
// and p10/p90 over the samples (linear interpolation percentile).
struct EprobeTiming {
    double median = 0.0;
    double p10 = 0.0;
    double p90 = 0.0;
};

inline double eprobe_percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double idx = q * double(sorted.size() - 1);
    const size_t lo = size_t(idx);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - double(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

inline EprobeTiming eprobe_summarize(std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    return EprobeTiming{eprobe_percentile(ms, 0.5), eprobe_percentile(ms, 0.1),
                        eprobe_percentile(ms, 0.9)};
}

template <int RC, int N, int ILP, bool ReloadW>
struct EprobeBenchTag {};

// esimd dpas microbench: one staged K=32 tile per iteration; ILP independent
// accumulators; ReloadW alternates two tiles in global so the weight load is
// not loop-invariant.
template <int RC, int N, int ILP, bool ReloadW>
EprobeTiming eprobe_bench_esimd_ms(Context& ctx, int threads, int iters, double& dpas_total) {
    std::vector<int8_t> w(size_t(2) * RC * 32);
    std::vector<int8_t> x(size_t(32) * N);
    for (size_t i = 0; i < w.size(); ++i) w[i] = int8_t(int(i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < x.size(); ++i) x[i] = int8_t(int(i * 11 + 5) % 255 - 127);
    auto dw = Tensor::allocate(ctx, {int64_t(w.size())}, dnnl::memory::data_type::s8);
    auto dx = Tensor::allocate(ctx, {int64_t(x.size())}, dnnl::memory::data_type::s8);
    auto dout = Tensor::allocate(ctx, {int64_t(threads) * RC * N}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(dw.data(), w.data(), w.size());
    ctx.memcpy_h2d(dx.data(), x.data(), x.size());
    int8_t* wp = dw.data_as<int8_t>();
    int8_t* xp = dx.data_as<int8_t>();
    int32_t* op = dout.data_as<int32_t>();
    auto submit = [&] {
        ctx.queue().submit([&](sycl::handler& h) {
            h.parallel_for<EprobeBenchTag<RC, N, ILP, ReloadW>>(
                sycl::range<1>(size_t(threads)), [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
                    using namespace sycl::ext::intel::esimd;
                    simd<int8_t, 32 * N> at(xp, overaligned_tag<16>{});
                    simd<int32_t, RC * N> c0(1);
                    simd<int32_t, RC * N> c1(2), c2(3), c3(4);
                    for (int it = 0; it < iters; ++it) {
                        int8_t* waddr = wp;
                        if constexpr (ReloadW) waddr = wp + ((it & 1) ? RC * 32 : 0);
                        simd<int8_t, RC * 32> wt(waddr, overaligned_tag<16>{});
                        c0 = esimd_xmx_ns::dpas<8, RC, int32_t>(c0, at, wt);
                        if constexpr (ILP > 1) c1 = esimd_xmx_ns::dpas<8, RC, int32_t>(c1, at, wt);
                        if constexpr (ILP > 2) {
                            c2 = esimd_xmx_ns::dpas<8, RC, int32_t>(c2, at, wt);
                            c3 = esimd_xmx_ns::dpas<8, RC, int32_t>(c3, at, wt);
                        }
                    }
                    if constexpr (ILP > 1) c0 = c0 + c1;
                    if constexpr (ILP > 2) c0 = c0 + c2 + c3;
                    c0.copy_to(op + int(i[0]) * RC * N);
                });
        });
    };
    for (int r = 0; r < 3; ++r) {
        submit();
        ctx.synchronize();
    }
    std::vector<double> ms;
    for (int r = 0; r < 20; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        submit();
        ctx.synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    dpas_total = double(threads) * iters * ILP;
    return eprobe_summarize(std::move(ms));
}

// SG8 joint_matrix microloop with register-resident operands (gate comparator).
EprobeTiming eprobe_bench_jm_ms(Context& ctx, int sgs, int iters, double& mad_total) {
    std::vector<int8_t> w(8 * 32), x(32 * 8);
    for (size_t i = 0; i < w.size(); ++i) w[i] = int8_t(int(i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < x.size(); ++i) x[i] = int8_t(int(i * 11 + 5) % 255 - 127);
    auto dw = Tensor::allocate(ctx, {int64_t(w.size())}, dnnl::memory::data_type::s8);
    auto dx = Tensor::allocate(ctx, {int64_t(x.size())}, dnnl::memory::data_type::s8);
    auto dout = Tensor::allocate(ctx, {int64_t(sgs) * 8 * 8}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(dw.data(), w.data(), w.size());
    ctx.memcpy_h2d(dx.data(), x.data(), x.size());
    const int8_t* wp = dw.data_as<int8_t>();
    const int8_t* xp = dx.data_as<int8_t>();
    int32_t* op = dout.data_as<int32_t>();
    auto submit = [&] {
        ctx.queue().submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(size_t(sgs) * 8, 8),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
                using namespace sycl::ext::oneapi::experimental::matrix;
                auto sg = it.get_sub_group();
                joint_matrix<sycl::sub_group, int8_t, use::a, 8, 32, layout::row_major> am;
                joint_matrix<sycl::sub_group, int8_t, use::b, 32, 8, layout::ext_intel_packed> bm;
                joint_matrix<sycl::sub_group, int32_t, use::accumulator, 8, 8> c0, c1, c2, c3;
                joint_matrix_load(sg, am,
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(wp), 32);
                joint_matrix_load(sg, bm,
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(xp), 32);
                joint_matrix_fill(sg, c0, 1);
                joint_matrix_fill(sg, c1, 2);
                joint_matrix_fill(sg, c2, 3);
                joint_matrix_fill(sg, c3, 4);
                for (int it2 = 0; it2 < iters; ++it2) {
                    joint_matrix_mad(sg, c0, am, bm, c0);
                    joint_matrix_mad(sg, c1, am, bm, c1);
                    joint_matrix_mad(sg, c2, am, bm, c2);
                    joint_matrix_mad(sg, c3, am, bm, c3);
                }
                joint_matrix_apply(sg, c0, c1, [](int32_t& a, int32_t& b) { a += b; });
                joint_matrix_apply(sg, c0, c2, [](int32_t& a, int32_t& b) { a += b; });
                joint_matrix_apply(sg, c0, c3, [](int32_t& a, int32_t& b) { a += b; });
                joint_matrix_store(sg, c0,
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
                        op + size_t(it.get_global_linear_id() / 8) * 64),
                    8, layout::row_major);
            });
        });
    };
    for (int r = 0; r < 3; ++r) {
        submit();
        ctx.synchronize();
    }
    std::vector<double> ms;
    for (int r = 0; r < 20; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        submit();
        ctx.synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    mad_total = double(sgs) * iters * 4;
    return eprobe_summarize(std::move(ms));
}

// Scalar int8 dot per output element (no DP4A): the current wgshare math.
EprobeTiming eprobe_bench_idot_ms(Context& ctx, int threads, int iters) {
    std::vector<int8_t> w(32), x(32 * 8);
    for (size_t i = 0; i < w.size(); ++i) w[i] = int8_t(int(i * 7 + 3) % 255 - 127);
    for (size_t i = 0; i < x.size(); ++i) x[i] = int8_t(int(i * 11 + 5) % 255 - 127);
    auto dw = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::s8);
    auto dx = Tensor::allocate(ctx, {int64_t(x.size())}, dnnl::memory::data_type::s8);
    auto dout = Tensor::allocate(ctx, {int64_t(threads)}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(dw.data(), w.data(), w.size());
    ctx.memcpy_h2d(dx.data(), x.data(), x.size());
    const int8_t* wp = dw.data_as<int8_t>();
    const int8_t* xp = dx.data_as<int8_t>();
    int32_t* op = dout.data_as<int32_t>();
    auto submit = [&] {
        ctx.queue().submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(size_t(threads), 64), [=](sycl::nd_item<1> it) {
                const int t = int(it.get_global_linear_id());
                const int n = t & 7;
                int acc = 0;
                for (int it2 = 0; it2 < iters; ++it2) {
                    int s = 0;
#pragma unroll
                    for (int k = 0; k < 32; ++k) {
                        s += int(wp[k]) * int(xp[((k / 4) * 8 + n) * 4 + (k % 4)]);
                    }
                    acc += s;
                }
                op[t] = acc;
            });
        });
    };
    for (int r = 0; r < 3; ++r) {
        submit();
        ctx.synchronize();
    }
    std::vector<double> ms;
    for (int r = 0; r < 20; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        submit();
        ctx.synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return eprobe_summarize(std::move(ms));
}

void eprobe_report_bench(const std::string& label, const EprobeTiming& t, double ops,
                         double macs_per_op) {
    const double ns_per_op = t.median * 1e6 / ops;
    const double gmac = ops * macs_per_op / (t.median * 1e-3) / 1e9;
    std::cout << "[ESIMD-DPAS] bench " << label << ": total=" << ops
              << " ops, median=" << t.median << " ms, p10=" << t.p10
              << " ms, p90=" << t.p90 << " ms, " << ns_per_op << " ns/op, " << gmac
              << " GMAC/s (" << gmac * 2.0 << " GFLOP/s)\n" << std::flush;
}

// VNNI pack + NF4 unpack staging cost (plain SYCL, mirrors the R2b prototype).
void eprobe_bench_staging_ms(Context& ctx, int unpack_rows, int pack_rows, int width,
                             double& unpack_ms, double& pack_ms) {
    const size_t weights = size_t(unpack_rows) * width;
    std::vector<uint8_t> packed(weights / 2);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = uint8_t((i * 37 + 11) & 0xff);
    std::vector<int8_t> rows(size_t(pack_rows) * width);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = int8_t(int(i * 13 + 7) % 255 - 127);
    auto dpacked = Tensor::allocate(ctx, {int64_t(packed.size())}, dnnl::memory::data_type::u8);
    auto dcodes = Tensor::allocate(ctx, {int64_t(weights)}, dnnl::memory::data_type::s8);
    auto drows = Tensor::allocate(ctx, {int64_t(rows.size())}, dnnl::memory::data_type::s8);
    auto dvnni = Tensor::allocate(ctx, {int64_t(rows.size())}, dnnl::memory::data_type::s8);
    ctx.memcpy_h2d(dpacked.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(drows.data(), rows.data(), rows.size());
    const uint8_t* pp = dpacked.data_as<uint8_t>();
    int8_t* cp = dcodes.data_as<int8_t>();
    const int8_t* rp = drows.data_as<int8_t>();
    int8_t* vp = dvnni.data_as<int8_t>();
    static constexpr int8_t kNf4[16] = {-127, -88, -67, -50, -36, -23, -12, 0,
                                        10, 20, 31, 43, 56, 71, 92, 127};
    auto unpack = [&] {
        ctx.queue().submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(packed.size()), [=](sycl::id<1> i) {
                const uint32_t word = reinterpret_cast<const uint32_t*>(pp)[i[0]];
                for (int j = 0; j < 8; ++j) {
                    cp[size_t(i[0]) * 8 + j] =
                        kNf4[(word >> ((j / 2) * 8 + (j % 2 ? 0 : 4))) & 15];
                }
            });
        });
    };
    auto vpack = [&] {
        const int words = width / 4;
        ctx.queue().submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(size_t(pack_rows) * words), [=](sycl::id<1> i) {
                const int row = int(i[0]) / words, w = int(i[0]) % words;
                reinterpret_cast<uint32_t*>(vp)[size_t(w) * pack_rows + row] =
                    reinterpret_cast<const uint32_t*>(rp)[size_t(row) * words + w];
            });
        });
    };
    struct Timed {
        Context& ctx;
        std::function<void()> fn;
        double median(int reps) {
            for (int w = 0; w < 3; ++w) {
                fn();
                ctx.synchronize();
            }
            std::vector<double> ms;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                fn();
                ctx.synchronize();
                const auto t1 = std::chrono::steady_clock::now();
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2];
        }
    };
    unpack_ms = Timed{ctx, unpack}.median(20);
    pack_ms = Timed{ctx, vpack}.median(20);
    const double unpack_gbps = (double(weights) * 1.5) / (unpack_ms * 1e-3) / 1e9;
    const double pack_gbps = (double(rows.size()) * 2.0) / (pack_ms * 1e-3) / 1e9;
    std::cout << "[ESIMD-DPAS] staging nf4->s8 " << weights << " weights: "
              << unpack_ms << " ms, " << unpack_gbps << " GB/s (r+w)\n";
    std::cout << "[ESIMD-DPAS] staging vnni-pack " << pack_rows << "x" << width
              << ": " << pack_ms << " ms, " << pack_gbps << " GB/s (r+w)\n";
}

void eprobe_run_benches(Context& ctx) {
    double gate_esimd = 0.0, gate_jm = 0.0;
    {
        double ops = 0;
        const EprobeTiming t = eprobe_bench_esimd_ms<8, 8, 4, false>(ctx, 4096, 1500, ops);
        eprobe_report_bench("esimd rc=8 n=8 k=32 ilp=4 reg-resident", t, ops, 2048.0);
        gate_esimd = ops * 2048.0 / (t.median * 1e-3) / 1e9;
    }
    {
        double ops = 0;
        const EprobeTiming t = eprobe_bench_esimd_ms<8, 8, 1, false>(ctx, 4096, 6000, ops);
        eprobe_report_bench("esimd rc=8 n=8 k=32 ilp=1 dependent", t, ops, 2048.0);
    }
    {
        double ops = 0;
        const EprobeTiming t = eprobe_bench_esimd_ms<8, 8, 4, true>(ctx, 4096, 1500, ops);
        eprobe_report_bench("esimd rc=8 n=8 k=32 ilp=4 weight-reload", t, ops, 2048.0);
    }
    {
        double ops = 0;
        const EprobeTiming t = eprobe_bench_esimd_ms<4, 8, 4, false>(ctx, 4096, 2000, ops);
        eprobe_report_bench("esimd rc=4 n=8 k=32 ilp=4 reg-resident", t, ops, 1024.0);
    }
    {
        double ops = 0;
        const EprobeTiming t = eprobe_bench_esimd_ms<1, 8, 4, false>(ctx, 4096, 2000, ops);
        eprobe_report_bench("esimd rc=1 n=8 k=32 ilp=4 reg-resident", t, ops, 256.0);
    }
    {
        double ops = 0;
        const EprobeTiming t = eprobe_bench_jm_ms(ctx, 2048, 3000, ops);
        eprobe_report_bench("joint-matrix sg8 8x8x32 ilp=4 reg-resident", t, ops, 2048.0);
        gate_jm = ops * 2048.0 / (t.median * 1e-3) / 1e9;
    }
    {
        const EprobeTiming t = eprobe_bench_idot_ms(ctx, 8192, 30000);
        const double ops = double(8192) * 30000;
        eprobe_report_bench("scalar int8 dot (32 MAC/thread/iter)", t, ops, 32.0);
    }
    const double ratio = gate_jm > 0 ? gate_esimd / gate_jm : 0.0;
    std::cout << "[ESIMD-DPAS] gate check: esimd(rc=8,ilp=4)=" << gate_esimd
              << " GMAC/s vs joint-matrix=" << gate_jm << " GMAC/s -> ratio "
              << ratio << " (esimd>=1.20x required by the ESIMD path doc)\n"
              << std::flush;
}

// ---------------------------------------------------------------------------
// Real production shapes: staged grid GEMM, per-64 scale epilogue, staging e2e
// ---------------------------------------------------------------------------
// Staged layouts (tile-major; every K32 tile is contiguous and 32B aligned):
//   Ws [e][kt][row][32]        int8 weight codes (row = output channel)
//   Xs [e][kt][mt][256]        int8 activations, tile VNNI:
//                              idx = (kk/4)*32 + n*4 + (kk%4), n = token-in-tile
//   out [e*outputs][rows]      int32 (pure GEMM) or f32 (scaled epilogue)
// Grid mirrors Qwen4ExpDpasMoeProbe.hpp: experts x mt-groups x output blocks;
// one work-group covers ILP 8-token tiles x RC output rows.

template <typename Fn>
static EprobeTiming eprobe_time_submit(Context& ctx, Fn&& fn) {
    for (int w = 0; w < 3; ++w) {
        fn();
        ctx.synchronize();
    }
    std::vector<double> ms(20);
    for (int r = 0; r < 20; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        ctx.synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return eprobe_summarize(std::move(ms));
}

template <int RC, int ILP, bool ShareA>
struct EprobeGridTag {};

// Pure int GEMM over full K (staged tiles, K chained in one accumulator).
template <int RC, int ILP, bool ShareA>
static sycl::event eprobe_grid_submit(Context& ctx, const int8_t* ws, const int8_t* xs,
                                      int32_t* out, int experts, int rows, int outputs, int k) {
    const int kc = k / 32, mt_count = rows / 8;
    const int mts = (mt_count + ILP - 1) / ILP, nblocks = outputs / RC;
    const size_t nwg = size_t(experts) * mts * nblocks;
    return ctx.queue().submit([&](sycl::handler& h) {
        h.parallel_for<EprobeGridTag<RC, ILP, ShareA>>(sycl::range<1>(nwg),
            [=](sycl::id<1> wg) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                const int e = int(wg[0]) / (mts * nblocks);
                const int mg = (int(wg[0]) / nblocks) % mts;
                const int nblk = int(wg[0]) % nblocks;
                const int mt0 = mg * ILP;
                const int8_t* wrow = ws + ((size_t(e) * kc) * outputs + size_t(nblk) * RC) * 32;
                const int8_t* xbase = xs + size_t(e) * kc * mt_count * 256;
                simd<int32_t, RC * 8> acc0(0), acc1(0), acc2(0), acc3(0);
#pragma unroll 4
                for (int kt = 0; kt < kc; ++kt) {
                    const int8_t* wa = wrow + size_t(kt) * outputs * 32;
                    simd<int8_t, RC * 32> a0(wa, overaligned_tag<16>{});
                    simd<int8_t, RC * 32> a1(0), a2(0), a3(0);
                    if constexpr (ILP > 1) {
                        if constexpr (ShareA) {
                            a1 = a0;
                            a2 = a0;
                            a3 = a0;
                        } else {
                            a1 = simd<int8_t, RC * 32>(wa, overaligned_tag<16>{});
                            a2 = simd<int8_t, RC * 32>(wa, overaligned_tag<16>{});
                            a3 = simd<int8_t, RC * 32>(wa, overaligned_tag<16>{});
                        }
                    }
                    const int8_t* xt = xbase + (size_t(kt) * mt_count + mt0) * 256;
                    const simd<int8_t, 256> b0(xt, overaligned_tag<16>{});
                    acc0 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                              eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(acc0, b0, a0);
                    if constexpr (ILP > 1) {
                        const simd<int8_t, 256> b1(xt + 256, overaligned_tag<16>{});
                        const simd<int8_t, 256> b2(xt + 512, overaligned_tag<16>{});
                        const simd<int8_t, 256> b3(xt + 768, overaligned_tag<16>{});
                        acc1 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                                  eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(acc1, b1, a1);
                        acc2 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                                  eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(acc2, b2, a2);
                        acc3 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                                  eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(acc3, b3, a3);
                    }
                }
                const size_t obase = size_t(e) * outputs * rows;
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const size_t o = obase + size_t(nblk * RC + r) * rows + size_t(mt0) * 8;
                    const simd<int32_t, 8> r0 = acc0.template select<8, 1>(r * 8);
                    block_store<int32_t, 8>(out + o, r0);
                    if constexpr (ILP > 1) {
                        block_store<int32_t, 8>(out + o + 8, acc1.template select<8, 1>(r * 8));
                        block_store<int32_t, 8>(out + o + 16, acc2.template select<8, 1>(r * 8));
                        block_store<int32_t, 8>(out + o + 24, acc3.template select<8, 1>(r * 8));
                    }
                }
            });
    });
}

template <int ILP>
struct EprobeGridJmTag {};

// SG8 joint_matrix comparator: same grid, same staged tiles, same ILP grouping.
template <int ILP>
static sycl::event eprobe_grid_jm_submit(Context& ctx, const int8_t* ws, const int8_t* xs,
                                         int32_t* out, int experts, int rows, int outputs, int k) {
    const int kc = k / 32, mt_count = rows / 8;
    const int mts = (mt_count + ILP - 1) / ILP, nblocks = outputs / 8;
    const size_t nwg = size_t(experts) * mts * nblocks;
    return ctx.queue().submit([&](sycl::handler& h) {
        h.parallel_for<EprobeGridJmTag<ILP>>(sycl::nd_range<1>(nwg * 8, 8),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
                using namespace sycl::ext::oneapi::experimental::matrix;
                auto sg = it.get_sub_group();
                const int wg = int(it.get_group(0));
                const int e = wg / (mts * nblocks);
                const int mg = (wg / nblocks) % mts;
                const int nblk = wg % nblocks;
                const int mt0 = mg * ILP;
                const int8_t* wrow = ws + ((size_t(e) * kc) * outputs + size_t(nblk) * 8) * 32;
                const int8_t* xbase = xs + size_t(e) * kc * mt_count * 256;
                joint_matrix<sycl::sub_group, int8_t, use::a, 8, 32, layout::row_major> am0;
                joint_matrix<sycl::sub_group, int8_t, use::b, 32, 8, layout::ext_intel_packed> bm0;
                joint_matrix<sycl::sub_group, int32_t, use::accumulator, 8, 8> c0, c1, c2, c3;
                joint_matrix<sycl::sub_group, int8_t, use::b, 32, 8, layout::ext_intel_packed> bm1, bm2, bm3;
                joint_matrix_fill(sg, c0, 0);
                if constexpr (ILP > 1) {
                    joint_matrix_fill(sg, c1, 0);
                    joint_matrix_fill(sg, c2, 0);
                    joint_matrix_fill(sg, c3, 0);
                }
#pragma unroll 4
                for (int kt = 0; kt < kc; ++kt) {
                    joint_matrix_load(sg, am0,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(
                            wrow + size_t(kt) * outputs * 32), 32);
                    const int8_t* xt = xbase + (size_t(kt) * mt_count + mt0) * 256;
                    joint_matrix_load(sg, bm0,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(xt), 32);
                    joint_matrix_mad(sg, c0, am0, bm0, c0);
                    if constexpr (ILP > 1) {
                        joint_matrix_load(sg, bm1,
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(xt + 256), 32);
                        joint_matrix_load(sg, bm2,
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(xt + 512), 32);
                        joint_matrix_load(sg, bm3,
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(xt + 768), 32);
                        joint_matrix_mad(sg, c1, am0, bm1, c1);
                        joint_matrix_mad(sg, c2, am0, bm2, c2);
                        joint_matrix_mad(sg, c3, am0, bm3, c3);
                    }
                }
                const size_t obase = size_t(e) * outputs * rows;
                const size_t o = obase + size_t(nblk * 8) * rows + size_t(mt0) * 8;
                joint_matrix_store(sg, c0,
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(out + o), rows,
                    layout::row_major);
                if constexpr (ILP > 1) {
                    joint_matrix_store(sg, c1,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(out + o + 8), rows,
                        layout::row_major);
                    joint_matrix_store(sg, c2,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(out + o + 16), rows,
                        layout::row_major);
                    joint_matrix_store(sg, c3,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(out + o + 24), rows,
                        layout::row_major);
                }
            });
    });
}

struct EprobeShapeBuffers {
    Tensor ws, xs, out;
    int experts = 0, rows = 0, outputs = 0, k = 0;

    void init(Context& ctx, int e, int m, int n, int kk) {
        experts = e; rows = m; outputs = n; k = kk;
        const int kc = k / 32, mt_count = rows / 8;
        std::vector<int8_t> wh(size_t(experts) * kc * outputs * 32);
        std::vector<int8_t> xh(size_t(experts) * kc * mt_count * 256);
        for (size_t i = 0; i < wh.size(); ++i) wh[i] = int8_t(int(i * 13 + 5) % 255 - 127);
        for (size_t i = 0; i < xh.size(); ++i) xh[i] = int8_t(int(i * 7 + 11) % 255 - 127);
        ws = Tensor::allocate(ctx, {int64_t(wh.size())}, dnnl::memory::data_type::s8);
        xs = Tensor::allocate(ctx, {int64_t(xh.size())}, dnnl::memory::data_type::s8);
        out = Tensor::allocate(ctx, {int64_t(experts) * outputs * rows}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(ws.data(), wh.data(), wh.size());
        ctx.memcpy_h2d(xs.data(), xh.data(), xh.size());
    }
};

// Decode staged codes back to (row, k) for the CPU reference.
static inline int eprobe_staged_w_get(const int8_t* ws, int e, int row, int k, int outputs,
                                      int kk_total) {
    const int kc = kk_total / 32;
    return int(ws[((size_t(e) * kc + (k / 32)) * outputs + row) * 32 + (k % 32)]);
}

static void eprobe_grid_verify_small(Context& ctx) {
    constexpr int E = 2, M = 16, N = 16, K = 64;
    EprobeShapeBuffers b;
    b.init(ctx, E, M, N, K);
    const int kc = K / 32, mt_count = M / 8;
    std::vector<int8_t> wh(size_t(E) * kc * N * 32), xh(size_t(E) * kc * mt_count * 256);
    ctx.memcpy_d2h(wh.data(), b.ws.data(), wh.size());
    ctx.memcpy_d2h(xh.data(), b.xs.data(), xh.size());
    // CPU reference from the staged layouts.
    std::vector<int32_t> ref(size_t(E) * N * M, 0);
    for (int e = 0; e < E; ++e) {
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                int64_t acc = 0;
                for (int kk = 0; kk < K; ++kk) {
                    const int wv = eprobe_staged_w_get(wh.data(), e, n, kk, N, K);
                    const int mt = m / 8, nm = m % 8;
                    const int kt = kk / 32, kx = kk % 32;
                    const int xv = int(xh[((size_t(e) * kc + kt) * mt_count + mt) * 256 +
                                         (kx / 4) * 32 + nm * 4 + (kx % 4)]);
                    acc += int64_t(wv) * int64_t(xv);
                }
                ref[(size_t(e) * N + n) * M + m] = int32_t(acc);
            }
        }
    }
    eprobe_grid_submit<8, 1, true>(ctx, b.ws.data_as<int8_t>(), b.xs.data_as<int8_t>(),
                                   b.out.data_as<int32_t>(), E, M, N, K).wait_and_throw();
    std::vector<int32_t> got(ref.size());
    ctx.memcpy_d2h(got.data(), b.out.data(), got.size() * sizeof(int32_t));
    int64_t max_abs = 0;
    for (size_t i = 0; i < ref.size(); ++i) max_abs = std::max(max_abs, std::abs(int64_t(got[i]) - ref[i]));
    // Cross-check the SG8 comparator arm on the identical staged inputs so a
    // "faster" comparator cannot be an artifact of wrong work.
    auto dout_jm = Tensor::allocate(ctx, {int64_t(E) * N * M}, dnnl::memory::data_type::s32);
    eprobe_grid_jm_submit<1>(ctx, b.ws.data_as<int8_t>(), b.xs.data_as<int8_t>(),
                             dout_jm.data_as<int32_t>(), E, M, N, K).wait_and_throw();
    std::vector<int32_t> got_jm(ref.size());
    ctx.memcpy_d2h(got_jm.data(), dout_jm.data(), got_jm.size() * sizeof(int32_t));
    int64_t max_abs_jm = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs_jm = std::max(max_abs_jm, std::abs(int64_t(got_jm[i]) - ref[i]));
    }
    std::cout << "[ESIMD-DPAS] grid verify small E" << E << " M" << M << " N" << N << " K" << K
              << ": esimd_max_abs=" << max_abs << " jm_max_abs=" << max_abs_jm
              << ((max_abs == 0 && max_abs_jm == 0) ? " PASS" : " FAIL") << "\n" << std::flush;
    if (max_abs != 0 || max_abs_jm != 0) throw std::runtime_error("ESIMD DPAS staged grid mismatch");
}

template <int ILP>
static void eprobe_grid_arms_impl(Context& ctx, EprobeShapeBuffers& b, const char* shape_tag,
                                  double macs, double& best_esimd, std::string& best_arm) {
    auto run = [&](const std::string& name, const EprobeTiming& t) {
        const double gmac = macs / (t.median * 1e-3) / 1e9;
        std::cout << "[ESIMD-DPAS] grid " << shape_tag << " " << name << ": median=" << t.median
                  << " p10=" << t.p10 << " p90=" << t.p90 << " ms, " << gmac << " GMAC/s\n"
                  << std::flush;
        return gmac;
    };
    const int8_t* ws = b.ws.data_as<int8_t>();
    const int8_t* xs = b.xs.data_as<int8_t>();
    int32_t* op = b.out.data_as<int32_t>();
    {
        const EprobeTiming t = eprobe_time_submit(ctx, [&] {
            eprobe_grid_submit<8, ILP, true>(ctx, ws, xs, op, b.experts, b.rows, b.outputs, b.k);
        });
        const double g = run("esimd rc=8 ilp="
                             + std::to_string(ILP) + " share", t);
        if (g > best_esimd) { best_esimd = g; best_arm = "esimd rc=8 share"; }
    }
    {
        const EprobeTiming t = eprobe_time_submit(ctx, [&] {
            eprobe_grid_submit<8, ILP, false>(ctx, ws, xs, op, b.experts, b.rows, b.outputs, b.k);
        });
        const double g = run("esimd rc=8 ilp=" + std::to_string(ILP) + " reload", t);
        if (g > best_esimd) { best_esimd = g; best_arm = "esimd rc=8 reload"; }
    }
    {
        const EprobeTiming t = eprobe_time_submit(ctx, [&] {
            eprobe_grid_submit<4, ILP, true>(ctx, ws, xs, op, b.experts, b.rows, b.outputs, b.k);
        });
        const double g = run("esimd rc=4 ilp=" + std::to_string(ILP) + " share", t);
        if (g > best_esimd) { best_esimd = g; best_arm = "esimd rc=4 share"; }
    }
    {
        const EprobeTiming t = eprobe_time_submit(ctx, [&] {
            eprobe_grid_submit<4, ILP, false>(ctx, ws, xs, op, b.experts, b.rows, b.outputs, b.k);
        });
        const double g = run("esimd rc=4 ilp=" + std::to_string(ILP) + " reload", t);
        if (g > best_esimd) { best_esimd = g; best_arm = "esimd rc=4 reload"; }
    }
    const EprobeTiming t = eprobe_time_submit(ctx, [&] {
        eprobe_grid_jm_submit<ILP>(ctx, ws, xs, op, b.experts, b.rows, b.outputs, b.k);
    });
    const double jm = run("sg8 joint_matrix ilp=" + std::to_string(ILP), t);
    const double ratio = jm > 0 ? best_esimd / jm : 0.0;
    std::cout << "[ESIMD-DPAS] grid compare " << shape_tag << ": best_esimd=" << best_esimd
              << " GMAC/s (" << best_arm << ") vs sg8=" << jm << " GMAC/s -> ratio " << ratio
              << (ratio >= 1.20 ? " GO" : " BELOW-GATE") << "\n" << std::flush;
}

static void eprobe_run_grid_shape(Context& ctx, int experts, int rows, int outputs, int k) {
    EprobeShapeBuffers b;
    b.init(ctx, experts, rows, outputs, k);
    const double macs = double(experts) * rows * outputs * k;
    const std::string tag = "k" + std::to_string(k) + " n" + std::to_string(outputs) + " m"
                            + std::to_string(rows) + " e" + std::to_string(experts);
    double best = 0.0;
    std::string best_arm;
    if (rows >= 32) {
        eprobe_grid_arms_impl<4>(ctx, b, tag.c_str(), macs, best, best_arm);
    } else {
        eprobe_grid_arms_impl<1>(ctx, b, tag.c_str(), macs, best, best_arm);
    }
}

// --------------------- per-64 scale epilogue comparison --------------------
// w_abs [e*outputs][64] padded (per-row absmax/127 per 64-K block);
// x_abs_t [nb][e*rows] transposed (8 consecutive tokens load 32B-aligned).
template <int RC, int ILP, bool ShareA>
struct EprobeScaledTag {};

template <int RC, int ILP, bool ShareA>
static sycl::event eprobe_scaled_submit(Context& ctx, const int8_t* ws, const int8_t* xs,
                                        const float* w_abs, const float* x_abs_t, float* out,
                                        int experts, int rows, int outputs, int k) {
    const int kc = k / 32, mt_count = rows / 8;
    const int mts = (mt_count + ILP - 1) / ILP, nblocks = outputs / RC;
    const int nb_count = k / 64;
    const size_t nwg = size_t(experts) * mts * nblocks;
    return ctx.queue().submit([&](sycl::handler& h) {
        h.parallel_for<EprobeScaledTag<RC, ILP, ShareA>>(sycl::range<1>(nwg),
            [=](sycl::id<1> wg) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                const int e = int(wg[0]) / (mts * nblocks);
                const int mg = (int(wg[0]) / nblocks) % mts;
                const int nblk = int(wg[0]) % nblocks;
                const int mt0 = mg * ILP;
                const int8_t* wrow = ws + ((size_t(e) * kc) * outputs + size_t(nblk) * RC) * 32;
                const int8_t* xbase = xs + size_t(e) * kc * mt_count * 256;
                const float* xrow = x_abs_t + size_t(e) * rows + size_t(mt0) * 8;
                const float* wrow_s = w_abs + size_t(e) * outputs + size_t(nblk) * RC;
                const size_t xstep = size_t(experts) * rows;
                simd<float, RC * 8> acc0(0), acc1(0), acc2(0), acc3(0);
#pragma unroll 2
                for (int nb = 0; nb < nb_count; ++nb) {
                    simd<int32_t, RC * 8> ip0(0), ip1(0), ip2(0), ip3(0);
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int kt = nb * 2 + j;
                        const int8_t* wa = wrow + size_t(kt) * outputs * 32;
                        const simd<int8_t, RC * 32> a0(wa, overaligned_tag<16>{});
                        simd<int8_t, RC * 32> a1(0), a2(0), a3(0);
                        if constexpr (ILP > 1) {
                            if constexpr (ShareA) {
                                a1 = a0;
                                a2 = a0;
                                a3 = a0;
                            } else {
                                a1 = simd<int8_t, RC * 32>(wa, overaligned_tag<16>{});
                                a2 = simd<int8_t, RC * 32>(wa, overaligned_tag<16>{});
                                a3 = simd<int8_t, RC * 32>(wa, overaligned_tag<16>{});
                            }
                        }
                        const int8_t* xt = xbase + (size_t(kt) * mt_count + mt0) * 256;
                        const simd<int8_t, 256> b0(xt, overaligned_tag<16>{});
                        ip0 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                                 eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(ip0, b0, a0);
                        if constexpr (ILP > 1) {
                            const simd<int8_t, 256> b1(xt + 256, overaligned_tag<16>{});
                            const simd<int8_t, 256> b2(xt + 512, overaligned_tag<16>{});
                            const simd<int8_t, 256> b3(xt + 768, overaligned_tag<16>{});
                            ip1 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                                     eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(ip1, b1, a1);
                            ip2 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                                     eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(ip2, b2, a2);
                            ip3 = esimd_xmx_ns::dpas<8, RC, int32_t, int32_t, int8_t, int8_t,
                                                     eprobe_prec<int8_t>(), eprobe_prec<int8_t>()>(ip3, b3, a3);
                        }
                    }
                    const float* xb = xrow + size_t(nb) * xstep;
                    const simd<float, 8> xs0 = block_load<float, 8>(xb);
                    simd<float, 8> xs1(0), xs2(0), xs3(0);
                    if constexpr (ILP > 1) {
                        xs1 = block_load<float, 8>(xb + 8);
                        xs2 = block_load<float, 8>(xb + 16);
                        xs3 = block_load<float, 8>(xb + 24);
                    }
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const float wmul = wrow_s[size_t(r) * 64 + nb];
                        const simd<float, 8> f0 = simd<float, 8>(ip0.template select<8, 1>(r * 8));
                        simd<float, 8> cur0 = acc0.template select<8, 1>(r * 8);
                        acc0.template select<8, 1>(r * 8) = cur0 + f0 * (xs0 * wmul);
                        if constexpr (ILP > 1) {
                            const simd<float, 8> f1 = simd<float, 8>(ip1.template select<8, 1>(r * 8));
                            const simd<float, 8> f2 = simd<float, 8>(ip2.template select<8, 1>(r * 8));
                            const simd<float, 8> f3 = simd<float, 8>(ip3.template select<8, 1>(r * 8));
                            simd<float, 8> cur1 = acc1.template select<8, 1>(r * 8);
                            simd<float, 8> cur2 = acc2.template select<8, 1>(r * 8);
                            simd<float, 8> cur3 = acc3.template select<8, 1>(r * 8);
                            acc1.template select<8, 1>(r * 8) = cur1 + f1 * (xs1 * wmul);
                            acc2.template select<8, 1>(r * 8) = cur2 + f2 * (xs2 * wmul);
                            acc3.template select<8, 1>(r * 8) = cur3 + f3 * (xs3 * wmul);
                        }
                    }
                }
                const size_t obase = size_t(e) * outputs * rows;
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const size_t o = obase + size_t(nblk * RC + r) * rows + size_t(mt0) * 8;
                    const simd<float, 8> r0 = acc0.template select<8, 1>(r * 8);
                    block_store<float, 8>(out + o, r0);
                    if constexpr (ILP > 1) {
                        block_store<float, 8>(out + o + 8, acc1.template select<8, 1>(r * 8));
                        block_store<float, 8>(out + o + 16, acc2.template select<8, 1>(r * 8));
                        block_store<float, 8>(out + o + 24, acc3.template select<8, 1>(r * 8));
                    }
                }
            });
    });
}

template <int ILP>
struct EprobeScaledJmTag {};

template <int ILP>
static sycl::event eprobe_scaled_jm_submit(Context& ctx, const int8_t* ws, const int8_t* xs,
                                           const float* w_abs, const float* x_abs_t, float* out,
                                           int experts, int rows, int outputs, int k) {
    namespace intel_matrix = sycl::ext::intel::experimental::matrix;
    const int kc = k / 32, mt_count = rows / 8;
    const int mts = (mt_count + ILP - 1) / ILP, nblocks = outputs / 8;
    const int nb_count = k / 64;
    const size_t nwg = size_t(experts) * mts * nblocks;
    return ctx.queue().submit([&](sycl::handler& h) {
        h.parallel_for<EprobeScaledJmTag<ILP>>(sycl::nd_range<1>(nwg * 8, 8),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
                using namespace sycl::ext::oneapi::experimental::matrix;
                auto sg = it.get_sub_group();
                const int wg = int(it.get_group(0));
                const int e = wg / (mts * nblocks);
                const int mg = (wg / nblocks) % mts;
                const int nblk = wg % nblocks;
                const int mt0 = mg * ILP;
                const int8_t* wrow = ws + ((size_t(e) * kc) * outputs + size_t(nblk) * 8) * 32;
                const int8_t* xbase = xs + size_t(e) * kc * mt_count * 256;
                const size_t xstep = size_t(experts) * rows;
                joint_matrix<sycl::sub_group, int8_t, use::a, 8, 32, layout::row_major> am0;
                joint_matrix<sycl::sub_group, int8_t, use::b, 32, 8, layout::ext_intel_packed> bm0, bm1, bm2, bm3;
                joint_matrix<sycl::sub_group, int32_t, use::accumulator, 8, 8> ip0, ip1, ip2, ip3;
                joint_matrix<sycl::sub_group, float, use::accumulator, 8, 8> part, accf0, accf1, accf2, accf3;
                joint_matrix_fill(sg, accf0, 0.f);
                if constexpr (ILP > 1) {
                    joint_matrix_fill(sg, accf1, 0.f);
                    joint_matrix_fill(sg, accf2, 0.f);
                    joint_matrix_fill(sg, accf3, 0.f);
                }
#pragma unroll 2
                for (int nb = 0; nb < nb_count; ++nb) {
                    joint_matrix_fill(sg, ip0, 0);
                    if constexpr (ILP > 1) {
                        joint_matrix_fill(sg, ip1, 0);
                        joint_matrix_fill(sg, ip2, 0);
                        joint_matrix_fill(sg, ip3, 0);
                    }
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int kt = nb * 2 + j;
                        joint_matrix_load(sg, am0,
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(
                                wrow + size_t(kt) * outputs * 32), 32);
                        const int8_t* xt = xbase + (size_t(kt) * mt_count + mt0) * 256;
                        joint_matrix_load(sg, bm0,
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(xt), 32);
                        joint_matrix_mad(sg, ip0, am0, bm0, ip0);
                        if constexpr (ILP > 1) {
                            joint_matrix_load(sg, bm1,
                                sycl::address_space_cast<sycl::access::address_space::global_space,
                                                         sycl::access::decorated::no>(xt + 256), 32);
                            joint_matrix_load(sg, bm2,
                                sycl::address_space_cast<sycl::access::address_space::global_space,
                                                         sycl::access::decorated::no>(xt + 512), 32);
                            joint_matrix_load(sg, bm3,
                                sycl::address_space_cast<sycl::access::address_space::global_space,
                                                         sycl::access::decorated::no>(xt + 768), 32);
                            joint_matrix_mad(sg, ip1, am0, bm1, ip1);
                            joint_matrix_mad(sg, ip2, am0, bm2, ip2);
                            joint_matrix_mad(sg, ip3, am0, bm3, ip3);
                        }
                    }
                    const int lane_off = mt0 * 8;
                    intel_matrix::joint_matrix_apply(sg, part, [&](float& s, size_t r, size_t c) {
                        s = w_abs[(size_t(e) * outputs + nblk * 8 + r) * 64 + nb] *
                            x_abs_t[size_t(nb) * xstep + size_t(e) * rows + lane_off + c];
                    });
                    joint_matrix_apply(sg, part, ip0, [](float& s, int32_t& v) {
                        s *= float(v);
                    });
                    joint_matrix_apply(sg, accf0, part, [](float& a, float& p) {
                        a += p;
                    });
                    if constexpr (ILP > 1) {
#define EPROBE_SCALED_JM_LANE(LANE, OFF, IP, ACCF)                                              \
                        intel_matrix::joint_matrix_apply(sg, part, [&](float& s, size_t r, size_t c) { \
                            s = w_abs[(size_t(e) * outputs + nblk * 8 + r) * 64 + nb] *              \
                                x_abs_t[size_t(nb) * xstep + size_t(e) * rows + lane_off + (OFF) + c]; \
                        });                                                                            \
                        joint_matrix_apply(sg, part, IP, [](float& s, int32_t& v) {                    \
                            s *= float(v);                                                             \
                        });                                                                            \
                        joint_matrix_apply(sg, ACCF, part, [](float& a, float& p) {                    \
                            a += p;                                                                     \
                        })
                        EPROBE_SCALED_JM_LANE(1, 8, ip1, accf1);
                        EPROBE_SCALED_JM_LANE(2, 16, ip2, accf2);
                        EPROBE_SCALED_JM_LANE(3, 24, ip3, accf3);
#undef EPROBE_SCALED_JM_LANE
                    }
                }
                const size_t obase = size_t(e) * outputs * rows;
                const size_t o = obase + size_t(nblk * 8) * rows + size_t(mt0) * 8;
                joint_matrix_store(sg, accf0,
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(out + o), rows,
                    layout::row_major);
                if constexpr (ILP > 1) {
                    joint_matrix_store(sg, accf1,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(out + o + 8), rows,
                        layout::row_major);
                    joint_matrix_store(sg, accf2,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(out + o + 16), rows,
                        layout::row_major);
                    joint_matrix_store(sg, accf3,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(out + o + 24), rows,
                        layout::row_major);
                }
            });
    });
}

static void eprobe_scaled_verify(Context& ctx) {
    constexpr int E = 1, M = 8, N = 8, K = 64;
    const int nb_count = K / 64, kc = K / 32, mt_count = M / 8;
    EprobeShapeBuffers b;
    b.init(ctx, E, M, N, K);
    std::vector<float> w_abs(size_t(E) * N * 64), x_abs_t(size_t(nb_count) * E * M);
    for (size_t i = 0; i < w_abs.size(); ++i) w_abs[i] = 0.021f + 0.0017f * float(i % 11);
    for (size_t i = 0; i < x_abs_t.size(); ++i) x_abs_t[i] = 0.115f + 0.006f * float(i % 7);
    auto dw = Tensor::allocate(ctx, {int64_t(w_abs.size())}, dnnl::memory::data_type::f32);
    auto dx = Tensor::allocate(ctx, {int64_t(x_abs_t.size())}, dnnl::memory::data_type::f32);
    auto dout = Tensor::allocate(ctx, {int64_t(E) * N * M}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(dw.data(), w_abs.data(), w_abs.size() * sizeof(float));
    ctx.memcpy_h2d(dx.data(), x_abs_t.data(), x_abs_t.size() * sizeof(float));
    eprobe_scaled_submit<8, 1, true>(ctx, b.ws.data_as<int8_t>(), b.xs.data_as<int8_t>(),
                                     dw.data_as<float>(), dx.data_as<float>(),
                                     dout.data_as<float>(), E, M, N, K).wait_and_throw();
    auto dout_jm = Tensor::allocate(ctx, {int64_t(E) * N * M}, dnnl::memory::data_type::f32);
    eprobe_scaled_jm_submit<1>(ctx, b.ws.data_as<int8_t>(), b.xs.data_as<int8_t>(),
                               dw.data_as<float>(), dx.data_as<float>(),
                               dout_jm.data_as<float>(), E, M, N, K).wait_and_throw();
    std::vector<int8_t> wh(size_t(E) * kc * N * 32), xh(size_t(E) * kc * mt_count * 256);
    ctx.memcpy_d2h(wh.data(), b.ws.data(), wh.size());
    ctx.memcpy_d2h(xh.data(), b.xs.data(), xh.size());
    std::vector<float> got(size_t(E) * N * M);
    ctx.memcpy_d2h(got.data(), dout.data(), got.size() * sizeof(float));
    std::vector<float> got_jm(got.size());
    ctx.memcpy_d2h(got_jm.data(), dout_jm.data(), got_jm.size() * sizeof(float));
    double max_rel = 0.0, max_rel_jm = 0.0;
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            double acc = 0.0;
            for (int nb = 0; nb < nb_count; ++nb) {
                int64_t part = 0;
                for (int kk = nb * 64; kk < (nb + 1) * 64; ++kk) {
                    const int wv = eprobe_staged_w_get(wh.data(), 0, n, kk, N, K);
                    const int mt = m / 8, nm = m % 8, kt = kk / 32, kx = kk % 32;
                    const int xv = int(xh[(size_t(kt) * mt_count + mt) * 256 + (kx / 4) * 32 +
                                          nm * 4 + (kx % 4)]);
                    part += int64_t(wv) * int64_t(xv);
                }
                acc += double(part) * double(w_abs[size_t(n) * 64 + nb]) *
                       double(x_abs_t[size_t(nb) * M + m]);
            }
            const double denom = std::max(1e-6, std::abs(acc));
            max_rel = std::max(max_rel, std::abs(double(got[size_t(n) * M + m]) - acc) / denom);
            max_rel_jm =
                std::max(max_rel_jm, std::abs(double(got_jm[size_t(n) * M + m]) - acc) / denom);
        }
    }
    std::cout << "[ESIMD-DPAS] scaled verify k64 n8 m8: max_rel=" << max_rel
              << " jm_max_rel=" << max_rel_jm
              << ((max_rel <= 1e-5 && max_rel_jm <= 1e-5) ? " PASS" : " FAIL") << "\n"
              << std::flush;
    if (max_rel > 1e-5 || max_rel_jm > 1e-5)
        throw std::runtime_error("ESIMD DPAS scaled epilogue mismatch");
}

static void eprobe_run_scaled_shape(Context& ctx, int experts, int rows, int outputs, int k,
                                    double& esimd_gmac, double& jm_gmac) {
    EprobeShapeBuffers b;
    b.init(ctx, experts, rows, outputs, k);
    const int nb_count = k / 64;
    std::vector<float> w_abs(size_t(experts) * outputs * 64);
    std::vector<float> x_abs_t(size_t(nb_count) * size_t(experts) * rows);
    for (size_t i = 0; i < w_abs.size(); ++i) w_abs[i] = 0.019f + 0.0013f * float(i % 13);
    for (size_t i = 0; i < x_abs_t.size(); ++i) x_abs_t[i] = 0.105f + 0.0047f * float(i % 9);
    auto dw = Tensor::allocate(ctx, {int64_t(w_abs.size())}, dnnl::memory::data_type::f32);
    auto dx = Tensor::allocate(ctx, {int64_t(x_abs_t.size())}, dnnl::memory::data_type::f32);
    auto dout = Tensor::allocate(ctx, {int64_t(experts) * outputs * rows}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(dw.data(), w_abs.data(), w_abs.size() * sizeof(float));
    ctx.memcpy_h2d(dx.data(), x_abs_t.data(), x_abs_t.size() * sizeof(float));
    const int8_t* ws = b.ws.data_as<int8_t>();
    const int8_t* xs = b.xs.data_as<int8_t>();
    float* op = dout.data_as<float>();
    const double macs = double(experts) * rows * outputs * k;
    const std::string tag = "scaled k" + std::to_string(k) + " n" + std::to_string(outputs)
                            + " m" + std::to_string(rows) + " e" + std::to_string(experts);
    auto report = [&](const std::string& name, const EprobeTiming& t) {
        const double gmac = macs / (t.median * 1e-3) / 1e9;
        std::cout << "[ESIMD-DPAS] " << tag << " " << name << ": median=" << t.median
                  << " p10=" << t.p10 << " p90=" << t.p90 << " ms, " << gmac << " GMAC/s\n"
                  << std::flush;
        return gmac;
    };
    if (rows >= 32) {
        const EprobeTiming t0 = eprobe_time_submit(ctx, [&] {
            eprobe_scaled_submit<8, 4, true>(ctx, ws, xs, dw.data_as<float>(),
                                             dx.data_as<float>(), op, experts, rows, outputs, k);
        });
        esimd_gmac = report("esimd rc=8 ilp=4 share", t0);
        const EprobeTiming t1 = eprobe_time_submit(ctx, [&] {
            eprobe_scaled_submit<8, 4, false>(ctx, ws, xs, dw.data_as<float>(),
                                              dx.data_as<float>(), op, experts, rows, outputs, k);
        });
        report("esimd rc=8 ilp=4 reload", t1);
        const EprobeTiming t2 = eprobe_time_submit(ctx, [&] {
            eprobe_scaled_jm_submit<4>(ctx, ws, xs, dw.data_as<float>(), dx.data_as<float>(),
                                       op, experts, rows, outputs, k);
        });
        jm_gmac = report("sg8 joint_matrix ilp=4 share", t2);
    } else {
        const EprobeTiming t0 = eprobe_time_submit(ctx, [&] {
            eprobe_scaled_submit<8, 1, true>(ctx, ws, xs, dw.data_as<float>(),
                                             dx.data_as<float>(), op, experts, rows, outputs, k);
        });
        esimd_gmac = report("esimd rc=8 ilp=1 share", t0);
        const EprobeTiming t2 = eprobe_time_submit(ctx, [&] {
            eprobe_scaled_jm_submit<1>(ctx, ws, xs, dw.data_as<float>(), dx.data_as<float>(),
                                       op, experts, rows, outputs, k);
        });
        jm_gmac = report("sg8 joint_matrix ilp=1 share", t2);
    }
    const double ratio = jm_gmac > 0 ? esimd_gmac / jm_gmac : 0.0;
    std::cout << "[ESIMD-DPAS] scaled compare " << tag << ": esimd=" << esimd_gmac
              << " GMAC/s vs sg8=" << jm_gmac << " GMAC/s -> ratio " << ratio
              << (ratio >= 1.20 ? " GO" : " BELOW-GATE") << "\n" << std::flush;
}

// --------------------- staging (NF4 unpack + VNNI pack) e2e ----------------
static constexpr int8_t kEprobeNf4Lut[16] = {-127, -88, -67, -50, -36, -23, -12, 0,
                                             10,   20,  31,  43,  56,  71,  92,  127};

static sycl::event eprobe_stage_w_submit(Context& ctx, const uint8_t* packed, int8_t* ws,
                                         int experts, int outputs, int k) {
    const int chunks = k / 8, kc = k / 32;
    const size_t total = size_t(experts) * outputs * chunks;
    return ctx.queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
            const int e = int(i[0]) / (outputs * chunks);
            const int row = (int(i[0]) / chunks) % outputs;
            const int chunk = int(i[0]) % chunks;
            const uint32_t word = reinterpret_cast<const uint32_t*>(
                packed + (size_t(e) * outputs + row) * (k / 2))[chunk];
            const int kt = chunk / 4, kk = (chunk % 4) * 8;
            int8_t* dst = ws + ((size_t(e) * kc + kt) * outputs + row) * 32 + kk;
            for (int j = 0; j < 8; ++j) {
                dst[j] = kEprobeNf4Lut[(word >> ((j / 2) * 8 + (j % 2 ? 0 : 4))) & 15];
            }
        });
    });
}

static sycl::event eprobe_pack_x_submit(Context& ctx, const int8_t* x0, int8_t* xs,
                                        int experts, int rows, int k) {
    const int words = k / 4, mt_count = rows / 8, kc = k / 32;
    const size_t total = size_t(experts) * rows * words;
    return ctx.queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
            const int e = int(i[0]) / (rows * words);
            const int row = (int(i[0]) / words) % rows;
            const int w = int(i[0]) % words;
            const uint32_t v =
                reinterpret_cast<const uint32_t*>(x0 + (size_t(e) * rows + row) * k)[w];
            const int kt = w / 8, kk4 = w % 8, mt = row / 8, n = row % 8;
            reinterpret_cast<uint32_t*>(
                xs + ((size_t(e) * kc + kt) * mt_count + mt) * 256 + kk4 * 32 + n * 4)[0] = v;
        });
    });
}

static void eprobe_staged_verify(Context& ctx) {
    constexpr int E = 1, M = 8, N = 8, K = 64;
    const int kc = K / 32, mt_count = M / 8;
    std::vector<uint8_t> packed(size_t(E) * N * (K / 2));
    std::vector<int8_t> x0(size_t(E) * M * K);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = uint8_t((i * 37 + 11) & 0xff);
    for (size_t i = 0; i < x0.size(); ++i) x0[i] = int8_t(int(i * 13 + 7) % 255 - 127);
    auto dp = Tensor::allocate(ctx, {int64_t(packed.size())}, dnnl::memory::data_type::u8);
    auto dx0 = Tensor::allocate(ctx, {int64_t(x0.size())}, dnnl::memory::data_type::s8);
    auto dws = Tensor::allocate(ctx, {int64_t(E) * kc * N * 32}, dnnl::memory::data_type::s8);
    auto dxs = Tensor::allocate(ctx, {int64_t(E) * kc * mt_count * 256}, dnnl::memory::data_type::s8);
    auto dout = Tensor::allocate(ctx, {int64_t(E) * N * M}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(dp.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(dx0.data(), x0.data(), x0.size());
    eprobe_stage_w_submit(ctx, dp.data_as<uint8_t>(), dws.data_as<int8_t>(), E, N, K);
    eprobe_pack_x_submit(ctx, dx0.data_as<int8_t>(), dxs.data_as<int8_t>(), E, M, K);
    ctx.synchronize();
    // Verify staged layouts against the source data, then the GEMM vs int64.
    std::vector<int8_t> wh(size_t(E) * kc * N * 32), xh(size_t(E) * kc * mt_count * 256);
    ctx.memcpy_d2h(wh.data(), dws.data(), wh.size());
    ctx.memcpy_d2h(xh.data(), dxs.data(), xh.size());
    int staged_errors = 0;
    for (int row = 0; row < N; ++row) {
        for (int kk = 0; kk < K; ++kk) {
            const uint8_t byte = packed[size_t(row) * (K / 2) + kk / 2];
            const int code = (kk % 2) ? (byte & 15) : (byte >> 4);
            if (eprobe_staged_w_get(wh.data(), 0, row, kk, N, K) != kEprobeNf4Lut[code]) ++staged_errors;
        }
    }
    for (int mt = 0; mt < mt_count; ++mt) {
        for (int n = 0; n < 8; ++n) {
            for (int kx = 0; kx < K; ++kx) {
                const int kt = kx / 32, kxr = kx % 32;
                const int got = int(xh[(size_t(kt) * mt_count + mt) * 256 + (kxr / 4) * 32 +
                                       n * 4 + (kxr % 4)]);
                if (got != int(x0[size_t(mt * 8 + n) * K + kx])) ++staged_errors;
            }
        }
    }
    eprobe_grid_submit<8, 1, true>(ctx, dws.data_as<int8_t>(), dxs.data_as<int8_t>(),
                                   dout.data_as<int32_t>(), E, M, N, K).wait_and_throw();
    std::vector<int32_t> got(size_t(E) * N * M);
    ctx.memcpy_d2h(got.data(), dout.data(), got.size() * sizeof(int32_t));
    int64_t gemm_errors = 0;
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            int64_t ref = 0;
            for (int kk = 0; kk < K; ++kk) {
                const uint8_t byte = packed[size_t(n) * (K / 2) + kk / 2];
                const int code = (kk % 2) ? (byte & 15) : (byte >> 4);
                ref += int64_t(kEprobeNf4Lut[code]) * int64_t(x0[size_t(m) * K + kk]);
            }
            if (int64_t(got[size_t(n) * M + m]) != ref) ++gemm_errors;
        }
    }
    std::cout << "[ESIMD-DPAS] staged verify k64 n8 m8: staged_errors=" << staged_errors
              << " gemm_errors=" << gemm_errors
              << ((staged_errors == 0 && gemm_errors == 0) ? " PASS" : " FAIL") << "\n"
              << std::flush;
    if (staged_errors || gemm_errors) throw std::runtime_error("ESIMD DPAS staged pipeline mismatch");
}

static void eprobe_run_staged_e2e(Context& ctx, int experts, int rows, int outputs, int k) {
    const int kc = k / 32, mt_count = rows / 8, kc4 = k / 4;
    std::vector<uint8_t> packed(size_t(experts) * outputs * (k / 2));
    std::vector<int8_t> x0(size_t(experts) * rows * k);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = uint8_t((i * 37 + 11) & 0xff);
    for (size_t i = 0; i < x0.size(); ++i) x0[i] = int8_t(int(i * 13 + 7) % 255 - 127);
    auto dp = Tensor::allocate(ctx, {int64_t(packed.size())}, dnnl::memory::data_type::u8);
    auto dx0 = Tensor::allocate(ctx, {int64_t(x0.size())}, dnnl::memory::data_type::s8);
    auto dws = Tensor::allocate(ctx, {int64_t(experts) * kc * outputs * 32}, dnnl::memory::data_type::s8);
    auto dxs = Tensor::allocate(ctx, {int64_t(experts) * kc * mt_count * 256}, dnnl::memory::data_type::s8);
    auto dout = Tensor::allocate(ctx, {int64_t(experts) * outputs * rows}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(dp.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(dx0.data(), x0.data(), x0.size());
    const EprobeTiming t_w = eprobe_time_submit(ctx, [&] {
        eprobe_stage_w_submit(ctx, dp.data_as<uint8_t>(), dws.data_as<int8_t>(), experts, outputs, k);
    });
    const EprobeTiming t_x = eprobe_time_submit(ctx, [&] {
        eprobe_pack_x_submit(ctx, dx0.data_as<int8_t>(), dxs.data_as<int8_t>(), experts, rows, k);
    });
    const int ilp = rows >= 32 ? 4 : 1;
    EprobeTiming t_g;
    if (rows >= 32) {
        t_g = eprobe_time_submit(ctx, [&] {
            eprobe_grid_submit<8, 4, true>(ctx, dws.data_as<int8_t>(), dxs.data_as<int8_t>(),
                                           dout.data_as<int32_t>(), experts, rows, outputs, k);
        });
    } else {
        t_g = eprobe_time_submit(ctx, [&] {
            eprobe_grid_submit<8, 1, true>(ctx, dws.data_as<int8_t>(), dxs.data_as<int8_t>(),
                                           dout.data_as<int32_t>(), experts, rows, outputs, k);
        });
    }
    const double total = t_w.median + t_x.median + t_g.median;
    const double tiles = double(experts) * (rows / 8) * (outputs / 8);
    const double weight_gbps = double(packed.size()) * 1.5 / (t_w.median * 1e-3) / 1e9;
    const double pack_gbps = double(x0.size()) * 2.0 / (t_x.median * 1e-3) / 1e9;
    std::cout << "[ESIMD-DPAS] e2e k" << k << " n" << outputs << " m" << rows << " e" << experts
              << " (ilp=" << ilp << "): stage_w=" << t_w.median << " ms (" << weight_gbps
              << " GB/s), pack_x=" << t_x.median << " ms (" << pack_gbps
              << " GB/s), gemm=" << t_g.median << " ms, total=" << total
              << " ms, per-8x8-tile=" << (total * 1e3 / tiles) << " us"
              << " (kc4=" << kc4 << ")\n"
              << std::flush;
}

static void eprobe_run_real_shapes(Context& ctx) {
    const int m_sweep[4] = {8, 32, 64, 128};
    for (int m : m_sweep) eprobe_run_grid_shape(ctx, 8, m, 1280, 2560);  // gate/up
    for (int m : m_sweep) eprobe_run_grid_shape(ctx, 8, m, 2560, 640);   // down
    double e1 = 0.0, j1 = 0.0, e2 = 0.0, j2 = 0.0;
    eprobe_run_scaled_shape(ctx, 8, 64, 1280, 2560, e1, j1);
    eprobe_run_scaled_shape(ctx, 8, 64, 2560, 640, e2, j2);
    eprobe_run_staged_e2e(ctx, 8, 128, 1280, 2560);
    eprobe_run_staged_e2e(ctx, 8, 128, 2560, 640);
}

// N=16 is not in the DG2 capability list (upstream reports exec size 8 on DG2).
// Attempt it in isolation so a toolchain rejection cannot poison the main arm.
void test_esimd_dpas_n16_attempt(Context& ctx) {
    std::cout << "[ESIMD-DPAS] N16 probe: attempting M8/K32/N16 s8 x s8\n";
    try {
        eprobe_run_exact<8, 32, 16, int8_t, int8_t>(ctx, "m8k32n16");
        std::cout << "[ESIMD-DPAS] N16 result: compiled and EXACT (device supports N16)\n";
    } catch (const sycl::exception& e) {
        std::cout << "[ESIMD-DPAS] N16 result: rejected by toolchain/device: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cout << "[ESIMD-DPAS] N16 result: FAIL: " << e.what() << "\n";
    }
}

static void test_esimd_dpas_probe(Context& ctx) {
    std::cout << "\n======================================================================\n";
    std::cout << "  [ESIMD] WP0 probe: xmx::dpas coverage, exactness, micro-throughput\n";
    std::cout << "======================================================================\n";
    try {
        const auto dev = ctx.queue().get_device();
        std::cout << "[ESIMD-DPAS] device: "
                  << dev.get_info<sycl::info::device::name>();
#if EPROBE_HAS_INTEL_INFO
        std::cout << " EU simd width="
                  << dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
#endif
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cout << "[ESIMD-DPAS] device info probe failed: " << e.what() << "\n";
    }
    eprobe_run_exact<1, 32, 8, int8_t, int8_t>(ctx, "m1k32n8 s8xs8");
    eprobe_run_exact<2, 32, 8, int8_t, int8_t>(ctx, "m2k32n8 s8xs8");
    eprobe_run_exact<4, 32, 8, int8_t, int8_t>(ctx, "m4k32n8 s8xs8");
    eprobe_run_exact<8, 32, 8, int8_t, int8_t>(ctx, "m8k32n8 s8xs8");
    eprobe_run_exact<8, 64, 8, int8_t, int8_t>(ctx, "m8k64n8 s8xs8 chain2");
    eprobe_run_exact<8, 128, 8, int8_t, int8_t>(ctx, "m8k128n8 s8xs8 chain4");
    eprobe_run_exact<8, 256, 8, int8_t, int8_t>(ctx, "m8k256n8 s8xs8 chain8");
    eprobe_run_exact<8, 256, 8, uint8_t, int8_t>(ctx, "m8k256n8 u8w x s8x");
    eprobe_run_exact<8, 256, 8, int8_t, uint8_t>(ctx, "m8k256n8 s8w x u8x");
    eprobe_run_slm<8, 64, 8>(ctx, "m8k64n8 A-via-SLM");
    eprobe_run_moe_tile<8, 8, 256, 64>(ctx);
    double unpack_ms = 0.0;
    double pack_ms = 0.0;
    eprobe_bench_staging_ms(ctx, 4096, 512, 2560, unpack_ms, pack_ms);
    eprobe_run_benches(ctx);
    eprobe_grid_verify_small(ctx);
    eprobe_scaled_verify(ctx);
    eprobe_staged_verify(ctx);
    eprobe_run_real_shapes(ctx);
    std::cout << "[ESIMD-DPAS] main arms complete; running isolated N16 attempt last\n"
              << std::flush;
    test_esimd_dpas_n16_attempt(ctx);
    std::cout << "[ESIMD-DPAS] WP0 probe suite COMPLETE\n" << std::flush;
}
