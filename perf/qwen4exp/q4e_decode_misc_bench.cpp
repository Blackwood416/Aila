// E5 (2026-10-15): decode device-side remaining items bench for Qwen4Exp T1.
//
// Families (production rows=1 shapes, production code paths):
//   hc      ops::hc_decode_down_inject / ops::hc_decode_up_mix (+ streams and
//           the pipelined HcDecodeVariant), token scale = 96 inject pairs +
//           1 mixer pair = 194 kernels (Qwen4ExpBnb4Backend.cpp:2169-2210,
//           2568-2570)
//   gdn     ops::run_linear_delta_decode_gpu, 36 linear layers per token
//           (Qwen4Prefill.cpp:63-91)
//   shared  Bnb4BitLinear fused gate/up/SwiGLU + down (K=2560,N=1280 and
//           K=640,N=2560), 48 layers per token (Qwen4ExpBnb4Backend.cpp:2393-2430)
//   head    ops::Linear [2560 -> 248320] bf16 GEMV (+ optional Q8 head),
//           1 call per token (Qwen4ExpBnb4Backend.cpp:2578-2595)
//
// Every arm reports host submit time, device span (profiling events) and wall
// time, all medians, so a family can be classified as device- or
// submit-bound. Synthetic weights only: no model file, no worker, no server.
//
// Guards (fail closed): HC pipelined/streams arms must be bitwise identical to
// the production kernels; the shared NF4 stack is checked against a CPU f64
// dequantized oracle; output sentinel tails must stay -9.
//
// Usage (see docs/plans/qwen38-r1-decode-misc-2026-10-15.md section 3):
//   AilaQwen4ExpDecodeMiscBench.exe --arms all --samples 20 --warm 3
#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/GatedDeltaNetOps.hpp"
#include "ops/HcDecodeVariant.hpp"
#include "ops/HyperConnectionOps.hpp"
#include "ops/Ops.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using bf16 = sycl::ext::oneapi::bfloat16;
using DataType = dnnl::memory::data_type;

// Production Qwen3.8-Flash-Next shapes (config.json + backend load code).
constexpr int kHidden = 2560;
constexpr int kHcGroups = 4;
constexpr int kHcGroupDim = 2560;
constexpr int kHcLowrank = 320;
constexpr int kDecodeLayers = 48;
constexpr int kLinearLayers = 36;
constexpr int kSharedLayers = 48;
constexpr int kVocab = 248320;
constexpr int kSharedInter = 640;
constexpr int kGdnQHeads = 16;
constexpr int kGdnVHeads = 48;
constexpr int kGdnHeadDim = 128;
constexpr int kGdnKernel = 4;
constexpr int kHcPairCalls = 2 * kDecodeLayers + 1; // 97 calls = 194 kernels

// NF4 dequantization LUT, mirrored from
// tests/inference/DecodeRowProjectionTests.cpp:26 and SafeTensors load.
constexpr float kQuantMap[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};
constexpr int kBlock = 64;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("BENCH FAIL: " + message);
}

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

uint32_t mix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

std::vector<std::string> split_list(const std::string& text) {
    std::vector<std::string> tokens;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const size_t end = (comma == std::string::npos) ? text.size() : comma;
        if (end > start) tokens.push_back(text.substr(start, end - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return tokens;
}

std::vector<bf16> random_bf16(size_t n, uint32_t seed, float scale) {
    std::vector<bf16> out(n);
    uint32_t state = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        state = mix(state + static_cast<uint32_t>(i));
        const float unit = static_cast<float>(state % 4097) / 2048.0f - 1.0f;
        out[i] = bf16(unit * scale);
    }
    return out;
}

std::vector<float> random_f32(size_t n, uint32_t seed, float scale) {
    std::vector<float> out(n);
    uint32_t state = seed * 40503u + 7u;
    for (size_t i = 0; i < n; ++i) {
        state = mix(state + static_cast<uint32_t>(i));
        out[i] = (static_cast<float>(state % 4097) / 2048.0f - 1.0f) * scale;
    }
    return out;
}

Tensor make_bf16(Context& ctx, const std::vector<bf16>& host) {
    Tensor t = Tensor::allocate(ctx, {static_cast<int64_t>(host.size())}, DataType::bf16);
    ctx.memcpy_h2d(t.data(), host.data(), host.size() * sizeof(bf16));
    return t;
}

Tensor make_f32(Context& ctx, const std::vector<float>& host) {
    Tensor t = Tensor::allocate(ctx, {static_cast<int64_t>(host.size())}, DataType::f32);
    ctx.memcpy_h2d(t.data(), host.data(), host.size() * sizeof(float));
    return t;
}

struct Sample {
    double host_ms = 0;  // host time issuing the whole batch (no drain wait)
    double wall_ms = 0;  // host time including the drain wait
    double span_ms = 0;  // profiling span between the two barriers
};

template <typename Setup, typename Submit>
Sample measure(Context& ctx, int warm, int samples, Setup&& setup, Submit&& submit) {
    std::vector<double> host, wall, span;
    host.reserve(samples);
    wall.reserve(samples);
    span.reserve(samples);
    for (int s = 0; s < warm + samples; ++s) {
        setup();
        const auto t0 = Clock::now();
        auto a = ctx.queue().ext_oneapi_submit_barrier();
        submit();
        auto b = ctx.queue().ext_oneapi_submit_barrier();
        const auto t1 = Clock::now();
        b.wait_and_throw();
        const auto t2 = Clock::now();
        if (s < warm) continue;
        host.push_back(elapsed_ms(t0, t1));
        wall.push_back(elapsed_ms(t0, t2));
        span.push_back(
            static_cast<double>(b.get_profiling_info<sycl::info::event_profiling::command_start>() -
                                a.get_profiling_info<sycl::info::event_profiling::command_end>()) *
            1e-6);
    }
    Sample out;
    out.host_ms = median(host);
    out.wall_ms = median(wall);
    out.span_ms = median(span);
    return out;
}

void print_sample(const std::string& tag, const Sample& s, int kernels, double bytes) {
    std::cout << "[MISC] " << tag << " host_ms=" << std::fixed << std::setprecision(4) << s.host_ms
              << " span_ms=" << s.span_ms << " wall_ms=" << s.wall_ms
              << " kernels=" << kernels;
    if (bytes > 0)
        std::cout << " gbps_span=" << std::setprecision(2) << bytes / (s.span_ms * 1e6);
    if (kernels > 0)
        std::cout << " host_us_per_kernel=" << std::setprecision(2) << (s.host_ms * 1000.0 / kernels);
    std::cout << "\n";
}

bool bit_equal_bf16(const std::vector<bf16>& a, const std::vector<bf16>& b, long long* first) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        uint16_t x = 0, y = 0;
        std::memcpy(&x, &a[i], sizeof(x));
        std::memcpy(&y, &b[i], sizeof(y));
        if (x != y) {
            if (first) *first = static_cast<long long>(i);
            return false;
        }
    }
    return true;
}

bool bit_equal_f32(const std::vector<float>& a, const std::vector<float>& b, long long* first) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (first) *first = static_cast<long long>(i);
            return false;
        }
    }
    return true;
}

struct Error {
    double err = 0, ref = 0, max = 0;
    void add(double a, double b) {
        require(std::isfinite(a) && std::isfinite(b), "non-finite value");
        const double d = a - b;
        err += d * d;
        ref += a * a;
        max = std::max(max, std::abs(d));
    }
    double nrmse() const { return std::sqrt(err / std::max(ref, 1e-24)); }
};

// ---------------------------------------------------------------------------
// hyper-connection (hc)
// ---------------------------------------------------------------------------
struct HcFixture {
    Context& ctx;
    std::vector<bf16> host_n, host_wd, host_wi, host_wu;
    std::vector<float> host_act, host_inj, host_x;
    Tensor N, Wd, Wi, Wu, act, inj, x;

    explicit HcFixture(Context& c)
        : ctx(c),
          host_n(random_bf16(size_t(kHcGroups) * kHcGroupDim, 11, 0.35f)),
          host_wd(random_bf16(size_t(kHcLowrank) * kHcGroups * kHcGroupDim, 12, 0.05f)),
          host_wi(random_bf16(size_t(kHcGroups) * kHcGroups * kHcGroupDim, 13, 0.05f)),
          host_wu(random_bf16(size_t(kHcGroups) * kHcGroupDim * kHcLowrank, 14, 0.05f)) {
        N = make_bf16(ctx, host_n);
        Wd = make_bf16(ctx, host_wd);
        Wi = make_bf16(ctx, host_wi);
        Wu = make_bf16(ctx, host_wu);
        act = Tensor::allocate(ctx, {kHcLowrank}, DataType::bf16);
        inj = Tensor::allocate(ctx, {kHcGroups}, DataType::f32);
        x = Tensor::allocate(ctx, {kHcGroupDim}, DataType::bf16);
    }

    const bf16* n() const { return N.data_as<bf16>(); }
    const bf16* wd() const { return Wd.data_as<bf16>(); }
    const bf16* wi() const { return Wi.data_as<bf16>(); }
    const bf16* wu() const { return Wu.data_as<bf16>(); }
    bf16* act_p() { return act.data_as<bf16>(); }
    float* inj_p() { return inj.data_as<float>(); }
    bf16* x_p() { return x.data_as<bf16>(); }

    std::vector<bf16> read_act() {
        std::vector<bf16> out(kHcLowrank);
        ctx.memcpy_d2h(out.data(), act.data(), out.size() * sizeof(bf16));
        return out;
    }
    std::vector<bf16> read_x() {
        std::vector<bf16> out(kHcGroupDim);
        ctx.memcpy_d2h(out.data(), x.data(), out.size() * sizeof(bf16));
        return out;
    }
    std::vector<float> read_inj() {
        std::vector<float> out(kHcGroups);
        ctx.memcpy_d2h(out.data(), inj.data(), out.size() * sizeof(float));
        return out;
    }
};

void hc_pair_prod(HcFixture& f, bool with_inject) {
    ops::hc_decode_down_inject(f.ctx, f.n(), f.wd(), with_inject ? f.wi() : nullptr,
                               f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 256);
    ops::hc_decode_up_mix(f.ctx, f.act_p(), f.wu(), f.n(), f.x_p(), kHcLowrank, kHcGroupDim,
                          kHcGroups, false);
}

void hc_pair_streams(HcFixture& f, bool with_inject) {
    ops::hc_decode_down_inject(f.ctx, f.n(), f.wd(), with_inject ? f.wi() : nullptr,
                               f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 256);
    ops::hc_decode_up_mix(f.ctx, f.act_p(), f.wu(), f.n(), f.x_p(), kHcLowrank, kHcGroupDim,
                          kHcGroups, true);
}

void hc_pair_pl(HcFixture& f, bool with_inject, int depth, int wg) {
    ops::hc_decode_down_inject_pl(f.ctx, f.n(), f.wd(), with_inject ? f.wi() : nullptr,
                                  f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, depth, wg);
    ops::hc_decode_up_mix_pl(f.ctx, f.act_p(), f.wu(), f.n(), f.x_p(), kHcLowrank, kHcGroupDim,
                             kHcGroups, depth);
}

void run_hc(Context& ctx, int samples, int warm, std::vector<std::pair<std::string, double>>& results) {
    HcFixture f(ctx);
    const double bytes_pair =
        (double(size_t(kHcLowrank) * kHcGroups * kHcGroupDim) +
         double(size_t(kHcGroups) * kHcGroups * kHcGroupDim) +
         double(size_t(kHcGroups) * kHcGroupDim * kHcLowrank)) *
        sizeof(bf16);

    // Bitwise parity: production vs streams vs pipelined (down and up).
    {
        hc_pair_prod(f, true);
        ctx.synchronize();
        const auto act_ref = f.read_act();
        const auto inj_ref = f.read_inj();
        const auto x_ref = f.read_x();
        long long first = -1;
        for (int depth : {1, 2, 3}) {
            for (int wg : {256, 128}) {
                if (depth == 3 && wg == 128) continue;  // depth sweep on the default WG only
                hc_pair_pl(f, true, depth, wg);
                ctx.synchronize();
                const bool ok = bit_equal_bf16(act_ref, f.read_act(), &first) &&
                                bit_equal_f32(inj_ref, f.read_inj(), &first) &&
                                bit_equal_bf16(x_ref, f.read_x(), &first);
                std::cout << "[MISC] hc_parity_pl_depth" << depth << "_wg" << wg
                          << " bitwise=" << (ok ? 1 : 0);
                if (!ok) std::cout << " first_diff=" << first;
                std::cout << "\n";
                require(ok, "hc pipelined depth " + std::to_string(depth) + " wg " +
                                std::to_string(wg) + " diverged from production");
            }
        }
        hc_pair_streams(f, true);
        ctx.synchronize();
        const bool ok = bit_equal_bf16(act_ref, f.read_act(), &first) &&
                        bit_equal_f32(inj_ref, f.read_inj(), &first) &&
                        bit_equal_bf16(x_ref, f.read_x(), &first);
        std::cout << "[MISC] hc_parity_streams bitwise=" << (ok ? 1 : 0) << "\n";
        require(ok, "hc streams diverged from production");
    }

    auto bench_single = [&](const std::string& tag, const std::function<void()>& fn, int kernels) {
        const Sample s = measure(ctx, warm, samples, [] {}, fn);
        print_sample(tag, s, kernels, bytes_pair);
        results.emplace_back(tag, s.span_ms);
    };

    bench_single("hc_down_wg256", [&] { ops::hc_decode_down_inject(f.ctx, f.n(), f.wd(), f.wi(), f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 256); }, 1);
    bench_single("hc_down_wg128", [&] { ops::hc_decode_down_inject(f.ctx, f.n(), f.wd(), f.wi(), f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 128); }, 1);
    bench_single("hc_down_wg64", [&] { ops::hc_decode_down_inject(f.ctx, f.n(), f.wd(), f.wi(), f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 64); }, 1);
    bench_single("hc_down_pl2_wg256", [&] { ops::hc_decode_down_inject_pl(f.ctx, f.n(), f.wd(), f.wi(), f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 2, 256); }, 1);
    bench_single("hc_down_pl2_wg128", [&] { ops::hc_decode_down_inject_pl(f.ctx, f.n(), f.wd(), f.wi(), f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 2, 128); }, 1);
    bench_single("hc_down_pl3_wg256", [&] { ops::hc_decode_down_inject_pl(f.ctx, f.n(), f.wd(), f.wi(), f.act_p(), f.inj_p(), kHcLowrank, kHcGroupDim, kHcGroups, 3, 256); }, 1);
    bench_single("hc_up_prod", [&] { ops::hc_decode_up_mix(f.ctx, f.act_p(), f.wu(), f.n(), f.x_p(), kHcLowrank, kHcGroupDim, kHcGroups, false); }, 1);
    bench_single("hc_up_streams", [&] { ops::hc_decode_up_mix(f.ctx, f.act_p(), f.wu(), f.n(), f.x_p(), kHcLowrank, kHcGroupDim, kHcGroups, true); }, 1);
    bench_single("hc_up_pl2", [&] { ops::hc_decode_up_mix_pl(f.ctx, f.act_p(), f.wu(), f.n(), f.x_p(), kHcLowrank, kHcGroupDim, kHcGroups, 2); }, 1);
    bench_single("hc_up_pl3", [&] { ops::hc_decode_up_mix_pl(f.ctx, f.act_p(), f.wu(), f.n(), f.x_p(), kHcLowrank, kHcGroupDim, kHcGroups, 3); }, 1);

    auto token_submit = [&](const std::function<void(HcFixture&, bool)>& pair) {
        return [&f, pair] {
            for (int call = 0; call < 2 * kDecodeLayers; ++call) pair(f, true);
            pair(f, false);  // final mixer, no inject weight
        };
    };
    const int token_samples = std::max(3, samples / 3);
    const int token_warm = 1;
    auto bench_token = [&](const std::string& tag, const std::function<void(HcFixture&, bool)>& pair) {
        auto submit = token_submit(pair);
        const Sample s = measure(ctx, token_warm, token_samples, [] {}, submit);
        print_sample(tag, s, 2 * kHcPairCalls, (double)kHcPairCalls * bytes_pair);
        results.emplace_back(tag, s.span_ms);
    };
    bench_token("hc_token_prod", hc_pair_prod);
    bench_token("hc_token_streams", hc_pair_streams);
    bench_token("hc_token_pl2_wg256", [](HcFixture& fx, bool inj) { hc_pair_pl(fx, inj, 2, 256); });
    bench_token("hc_token_pl2_wg128", [](HcFixture& fx, bool inj) { hc_pair_pl(fx, inj, 2, 128); });
    bench_token("hc_token_pl3_wg256", [](HcFixture& fx, bool inj) { hc_pair_pl(fx, inj, 3, 256); });
}

// ---------------------------------------------------------------------------
// linear (GDN decode)
// ---------------------------------------------------------------------------
struct GdnFixture {
    Context& ctx;
    ops::DeltaNetGeometry g;
    std::vector<bf16> host_in, host_cw, host_dt;
    std::vector<float> host_nw, host_al, host_state, host_hist;
    Tensor in, conv_w, norm_w, a_log, dt_bias, state, conv_state, out;
    int conv_head = 2;

    explicit GdnFixture(Context& c)
        : ctx(c),
          g(ops::DeltaNetGeometry::create(kGdnQHeads, kGdnVHeads, kGdnHeadDim, kGdnHeadDim,
                                          kGdnKernel, 1e-6f, ops::GateActivation::Sigmoid)),
          host_in(random_bf16(size_t(g.all_dim), 21, 0.3f)),
          host_cw(random_bf16(size_t(g.qkv_dim) * kGdnKernel, 22, 0.4f)),
          host_dt(random_bf16(size_t(kGdnVHeads), 23, 0.5f)),
          host_nw(random_f32(size_t(kGdnHeadDim), 24, 0.05f)),
          host_al(random_f32(size_t(kGdnVHeads), 25, 0.2f)),
          host_state(random_f32(size_t(kGdnVHeads) * kGdnHeadDim * kGdnHeadDim, 26, 0.02f)),
          host_hist(random_f32(size_t(3) * g.qkv_dim, 27, 0.2f)) {
        in = make_bf16(ctx, host_in);
        conv_w = make_bf16(ctx, host_cw);
        dt_bias = make_bf16(ctx, host_dt);
        norm_w = make_f32(ctx, host_nw);
        a_log = make_f32(ctx, host_al);
        state = make_f32(ctx, host_state);
        conv_state = make_f32(ctx, host_hist);
        out = Tensor::allocate(ctx, {int64_t(g.kv_dim) + 16}, DataType::bf16);
        std::vector<bf16> sentinel(out.numel(), bf16(-9.f));
        ctx.memcpy_h2d(out.data(), sentinel.data(), out.size_bytes());
    }

    ops::DeltaNetWeights weights() const {
        return {conv_w.data_as<bf16>(), norm_w.data_as<float>(), a_log.data_as<float>(),
                dt_bias.data_as<bf16>()};
    }

    void decode_once() {
        const bf16* base = in.data_as<bf16>();
        ops::DeltaNetState st{state.data_as<float>(), conv_state.data_as<float>(), conv_head, false};
        ops::run_linear_delta_decode_gpu(ctx, g, weights(), st, base,
                                         base + g.qkv_dim,
                                         base + g.qkv_dim + g.z_dim,
                                         base + g.qkv_dim + g.z_dim + kGdnVHeads,
                                         out.data_as<bf16>(), false);
        conv_head = (conv_head + 1) % 3;
    }
};

void run_gdn(Context& ctx, int samples, int warm, std::vector<std::pair<std::string, double>>& results) {
    GdnFixture f(ctx);
    f.decode_once();
    ctx.synchronize();
    {
        std::vector<bf16> probe(8);
        ctx.memcpy_d2h(probe.data(), f.out.data(), probe.size() * sizeof(bf16));
        for (const auto v : probe) require(std::isfinite(float(v)), "GDN output not finite");
    }
    const Sample single = measure(ctx, warm, samples, [] {}, [&] { f.decode_once(); });
    print_sample("gdn_single", single, 1, 0);
    results.emplace_back("gdn_single", single.span_ms);

    const Sample token = measure(ctx, 1, std::max(3, samples / 3), [] {}, [&] {
        for (int layer = 0; layer < kLinearLayers; ++layer) f.decode_once();
    });
    print_sample("gdn_token36", token, kLinearLayers, 0);
    results.emplace_back("gdn_token36", token.span_ms);
}

// ---------------------------------------------------------------------------
// shared expert (NF4 gate/up/SwiGLU + down)
// ---------------------------------------------------------------------------
struct SharedFixture {
    Context& ctx;
    Bnb4BitLinear gu, down;
    Bnb4BitLinearScratch scratch;
    std::vector<uint8_t> packed_gu, packed_down;
    std::vector<float> abs_gu, abs_down;
    std::vector<bf16> host_x;
    Tensor map, wgu_p, wgu_a, wd_p, wd_a, x, inter, gu_raw, out;
    int gu_rows = 2 * kSharedInter;

    explicit SharedFixture(Context& c) : ctx(c), host_x(random_bf16(kHidden, 31, 0.3f)) {
        const size_t gu_values = size_t(gu_rows) * kHidden;
        const size_t down_values = size_t(kHidden) * kSharedInter;
        packed_gu.resize(gu_values / 2);
        packed_down.resize(down_values / 2);
        uint32_t seed = 41;
        for (auto& b : packed_gu) b = static_cast<uint8_t>(mix(seed++ + 0x51));
        for (auto& b : packed_down) b = static_cast<uint8_t>(mix(seed++ + 0xD3));
        abs_gu.resize(gu_values / kBlock);
        abs_down.resize(down_values / kBlock);
        for (size_t i = 0; i < abs_gu.size(); ++i) abs_gu[i] = std::ldexp(1.0f, int(i % 5) - 4);
        for (size_t i = 0; i < abs_down.size(); ++i) abs_down[i] = std::ldexp(1.0f, int(i % 5) - 4);

        Tensor map_dev = Tensor::allocate(ctx, {16}, DataType::f32);
        ctx.memcpy_h2d(map_dev.data(), kQuantMap, sizeof(kQuantMap));
        map = std::move(map_dev);
        wgu_p = Tensor::allocate(ctx, {int64_t(packed_gu.size())}, DataType::u8);
        wgu_a = Tensor::allocate(ctx, {int64_t(abs_gu.size())}, DataType::f32);
        wd_p = Tensor::allocate(ctx, {int64_t(packed_down.size())}, DataType::u8);
        wd_a = Tensor::allocate(ctx, {int64_t(abs_down.size())}, DataType::f32);
        ctx.memcpy_h2d(wgu_p.data(), packed_gu.data(), packed_gu.size());
        ctx.memcpy_h2d(wgu_a.data(), abs_gu.data(), abs_gu.size() * sizeof(float));
        ctx.memcpy_h2d(wd_p.data(), packed_down.data(), packed_down.size());
        ctx.memcpy_h2d(wd_a.data(), abs_down.data(), abs_down.size() * sizeof(float));
        x = make_bf16(ctx, host_x);
        inter = Tensor::allocate(ctx, {kSharedInter}, DataType::bf16);
        gu_raw = Tensor::allocate(ctx, {gu_rows}, DataType::bf16);
        out = Tensor::allocate(ctx, {kHidden}, DataType::bf16);

        Bnb4BitWeightRef ref_gu;
        ref_gu.name = "bench.shared.gate_up";
        ref_gu.packed_weight = &wgu_p;
        ref_gu.absmax = &wgu_a;
        ref_gu.quant_map = &map;
        ref_gu.quant_state.quant_type = "nf4";
        ref_gu.quant_state.dtype = "bfloat16";
        ref_gu.quant_state.shape = {gu_rows, kHidden};
        ref_gu.quant_state.blocksize = kBlock;
        std::string error;
        require(gu.init(ctx, ref_gu, &error), "shared gate_up init failed: " + error);

        Bnb4BitWeightRef ref_down;
        ref_down.name = "bench.shared.down";
        ref_down.packed_weight = &wd_p;
        ref_down.absmax = &wd_a;
        ref_down.quant_map = &map;
        ref_down.quant_state.quant_type = "nf4";
        ref_down.quant_state.dtype = "bfloat16";
        ref_down.quant_state.shape = {kHidden, kSharedInter};
        ref_down.quant_state.blocksize = kBlock;
        require(down.init(ctx, ref_down, &error), "shared down init failed: " + error);
    }

    void stack() {
        require(Bnb4BitLinear::try_forward_decode_gate_up_swiglu(ctx, gu, x, inter, kSharedInter),
                "shared fused gate/up/swiglu is unsupported for the bench shape");
        down.ForwardDecodeRows(ctx, scratch, inter, out, 1);
    }

    double weight_bytes() const {
        return double(packed_gu.size() + packed_down.size()) +
               4.0 * double(abs_gu.size() + abs_down.size());
    }
};

double cpu_nf4_dot(const std::vector<uint8_t>& packed, const std::vector<float>& absmax,
                   int row, int k, const std::vector<bf16>& input) {
    double sum = 0.0;
    const size_t base = size_t(row) * k;
    for (int i = 0; i < k; ++i) {
        const uint8_t byte = packed[(base + i) / 2];
        const int code = (base + i) % 2 ? byte & 15 : byte >> 4;
        const float w = float(bf16(kQuantMap[code] * absmax[(base + i) / kBlock]));
        sum += double(w) * double(float(input[i]));
    }
    return sum;
}

void run_shared(Context& ctx, int samples, int warm, std::vector<std::pair<std::string, double>>& results) {
    SharedFixture f(ctx);
    f.stack();
    ctx.synchronize();

    // Determinism: two production stacks must agree bitwise.
    std::vector<bf16> out_a(kHidden), out_b(kHidden);
    ctx.memcpy_d2h(out_a.data(), f.out.data(), out_a.size() * sizeof(bf16));
    f.stack();
    ctx.synchronize();
    ctx.memcpy_d2h(out_b.data(), f.out.data(), out_b.size() * sizeof(bf16));
    long long first = -1;
    require(bit_equal_bf16(out_a, out_b, &first), "shared stack is not deterministic");

    // CPU f64 oracle for the down projection, fed with the GPU's own SiLU output.
    {
        std::vector<bf16> inter_host(kSharedInter);
        ctx.memcpy_d2h(inter_host.data(), f.inter.data(), inter_host.size() * sizeof(bf16));
        Error e;
        for (int row = 0; row < kHidden; row += 64) {
            const double ref = cpu_nf4_dot(f.packed_down, f.abs_down, row, kSharedInter, inter_host);
            e.add(ref, double(float(out_a[row])));
        }
        std::cout << "[MISC] shared_cpu_oracle nrmse=" << std::setprecision(5) << e.nrmse()
                  << " max_abs=" << e.max << " rows=sampled64\n";
        require(e.nrmse() < 5e-2, "shared CPU oracle mismatch");
    }

    const double bytes = f.weight_bytes();
    auto bench = [&](const std::string& tag, const std::function<void()>& fn, int kernels, double by) {
        const Sample s = measure(ctx, warm, samples, [] {}, fn);
        print_sample(tag, s, kernels, by);
        results.emplace_back(tag, s.span_ms);
    };
    bench("shared_gu", [&] {
        require(Bnb4BitLinear::try_forward_decode_gate_up_swiglu(ctx, f.gu, f.x, f.inter, kSharedInter),
                "shared fused gate/up/swiglu unsupported");
    }, 1, 0);
    bench("shared_down", [&] { f.down.ForwardDecodeRows(ctx, f.scratch, f.inter, f.out, 1); }, 1, 0);
    bench("shared_stack", [&] { f.stack(); }, 2, bytes);
    bench("shared_token48", [&] {
        for (int layer = 0; layer < kSharedLayers; ++layer) f.stack();
    }, 2 * kSharedLayers, bytes * kSharedLayers);
}

// ---------------------------------------------------------------------------
// lm head (bf16 and optional Q8)
// ---------------------------------------------------------------------------
void fill_bf16(Context& ctx, Tensor& t, float value) {
    bf16* p = t.data_as<bf16>();
    const int64_t n = t.numel();
    ctx.queue().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) { p[i] = bf16(value); });
    });
}

void run_head(Context& ctx, int samples, int warm, bool run_q8,
              std::vector<std::pair<std::string, double>>& results) {
    Tensor weight = Tensor::allocate(ctx, {kHidden, kVocab}, DataType::bf16);
    fill_bf16(ctx, weight, 0.01f);
    Linear head;
    head.init(ctx, weight, kHidden, kVocab, /*preprocessed=*/true, /*prepare_gemv=*/false);
    std::vector<bf16> host_x = random_bf16(kHidden, 51, 0.3f);
    Tensor x = make_bf16(ctx, host_x);
    Tensor logits = Tensor::allocate(ctx, {kVocab}, DataType::bf16);
    const double bytes = double(kHidden) * kVocab * sizeof(bf16);

    const Sample head_sample = measure(ctx, warm, samples, [] {}, [&] {
        head.forward(ctx, x, logits, 1);
    });
    print_sample("head_bf16", head_sample, 1, bytes);
    results.emplace_back("head_bf16", head_sample.span_ms);

    if (run_q8) {
        if (!ops::bf16_q8_head_supported(ctx, kHidden, kVocab)) {
            std::cout << "[MISC] head_q8 supported=0\n";
            return;
        }
        Tensor q8 = Tensor::allocate(ctx, {kVocab, kHidden}, DataType::u8);
        Tensor scales = Tensor::allocate(ctx, {kVocab, kHidden / kBlock}, DataType::f32);
        ops::quantize_bf16_weights_q8_64(ctx, weight, q8, scales).wait_and_throw();
        Tensor xq = Tensor::allocate(ctx, {kHidden}, DataType::u8);
        Tensor xs = Tensor::allocate(ctx, {kHidden / kBlock}, DataType::f32);
        Tensor q8_logits = Tensor::allocate(ctx, {kVocab}, DataType::bf16);
        const Sample q8_sample = measure(ctx, warm, samples, [] {}, [&] {
            ops::bf16_q8_head_forward(ctx, q8, scales, x, q8_logits, xq, xs, 1, kHidden, kVocab);
        });
        const double q8_bytes = double(kVocab) * kHidden + double(kVocab) * (kHidden / kBlock) * 4;
        print_sample("head_q8", q8_sample, 2, q8_bytes);
        results.emplace_back("head_q8", q8_sample.span_ms);
    }
}

// ---------------------------------------------------------------------------
// raw submit cost on the same in-order queue
// ---------------------------------------------------------------------------
struct TrivialKernel {
    void operator()(sycl::id<1>) const {}
};

void run_submit_cost(Context& ctx, int kernels) {
    auto submit_batch = [&](int n) {
        for (int i = 0; i < n; ++i)
            ctx.queue().submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(1), TrivialKernel{});
            });
    };
    submit_batch(64);
    ctx.synchronize();
    const auto t0 = Clock::now();
    submit_batch(kernels);
    const auto t1 = Clock::now();
    ctx.synchronize();
    const auto t2 = Clock::now();
    std::cout << "[MISC] submit_cost kernels=" << kernels
              << " host_us_per_kernel=" << std::fixed << std::setprecision(2)
              << (elapsed_ms(t0, t1) * 1000.0 / kernels)
              << " wall_us_per_kernel=" << (elapsed_ms(t0, t2) * 1000.0 / kernels) << "\n";
}

double lookup(const std::vector<std::pair<std::string, double>>& results, const std::string& tag) {
    for (const auto& r : results)
        if (r.first == tag) return r.second;
    return -1.0;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    aila::env::enable_l0_copy_engine();
    try {
        int samples = 20;
        int warm = 3;
        int submit_kernels = 512;
        bool want_hc = true, want_gdn = true, want_shared = true, want_head = true, want_q8 = false;
        bool want_submit = true;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, std::string("missing value after ") + argv[i]);
            const std::string key(argv[i]);
            const std::string value(argv[i + 1]);
            if (key == "--samples") samples = std::stoi(value);
            else if (key == "--warm") warm = std::stoi(value);
            else if (key == "--submit-kernels") submit_kernels = std::stoi(value);
            else if (key == "--arms") {
                want_hc = want_gdn = want_shared = want_head = want_q8 = want_submit = false;
                for (const auto& arm : split_list(value)) {
                    if (arm == "all") want_hc = want_gdn = want_shared = want_head = want_submit = true;
                    else if (arm == "hc") want_hc = true;
                    else if (arm == "gdn") want_gdn = true;
                    else if (arm == "shared") want_shared = true;
                    else if (arm == "head") want_head = true;
                    else if (arm == "head_q8") { want_head = true; want_q8 = true; }
                    else if (arm == "submit") want_submit = true;
                    else throw std::invalid_argument("unknown arm: " + arm);
                }
            } else {
                throw std::invalid_argument("unknown option: " + key);
            }
        }
        require(samples > 0 && warm >= 0 && submit_kernels > 0, "invalid bench arguments");

        Context ctx(nullptr, true);
        require(ctx.queue().has_property<sycl::property::queue::enable_profiling>(),
                "profiling queue required for device spans");
        std::cout << "[MISC DEVICE] " << ctx.queue().get_device().get_info<sycl::info::device::name>()
                  << " driver=" << ctx.queue().get_device().get_info<sycl::info::device::driver_version>()
                  << " samples=" << samples << " warm=" << warm << "\n";

        std::vector<std::pair<std::string, double>> results;
        if (want_submit) run_submit_cost(ctx, submit_kernels);
        if (want_hc) run_hc(ctx, samples, warm, results);
        if (want_gdn) run_gdn(ctx, samples, warm, results);
        if (want_shared) run_shared(ctx, samples, warm, results);
        if (want_head) run_head(ctx, samples, warm, want_q8, results);
        ctx.synchronize();

        // Gates (charter section 5.3): kernel >=3% at both shapes, e2e >=5%.
        auto gate = [&](const std::string& tag, const std::string& base, double clean_share_ms) {
            const double a = lookup(results, base);
            const double b = lookup(results, tag);
            if (a <= 0 || b <= 0) {
                std::cout << "[MISC GATE] " << tag << " status=missing\n";
                return;
            }
            const double pct = 100.0 * (a - b) / a;
            const double e2e = pct * clean_share_ms / 58.8;
            std::cout << "[MISC GATE] " << tag << " vs " << base << " kernel_pct=" << std::fixed
                      << std::setprecision(2) << pct << " projected_e2e_pct=" << e2e
                      << " kernel_gate=" << (pct >= 3.0 ? "PASS" : "FAIL")
                      << " e2e_gate=" << (e2e >= 5.0 ? "PASS" : "FAIL")
                      << " basis=clean_share_" << clean_share_ms << "ms_of_58.8ms\n";
        };
        gate("hc_token_streams", "hc_token_prod", 8.4);
        gate("hc_token_pl2_wg256", "hc_token_prod", 8.4);
        gate("hc_token_pl2_wg128", "hc_token_prod", 8.4);
        gate("hc_token_pl3_wg256", "hc_token_prod", 8.4);
        std::cout << "[MISC NOTE] head floor: "
                  << (double(kHidden) * kVocab * 2 / 291.11) / 1000.0
                  << " ms at the measured 291.11 GB/s streaming bound\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
