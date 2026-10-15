// Independent kernel-level bench for the GDN (linear attention) prefill
// recurrence and its SLM-warm-row variant
// (ops::run_linear_delta_prefill_gpu_batched_slm).
//
// Pseudo-random weights and inputs: no model file, no worker, no server.
//
// Guards before any timing (fail closed):
//   - CPU canary: f64 recurrence on a 64-token window for the baseline and
//     for the SLM arm (state/output NRMSE < 1.5e-2, the GdnChunkTests canary);
//   - parity: the SLM arm must reproduce the baseline state / conv history /
//     output bit-for-bit; if not, the tolerance fallback must satisfy
//     state_nrmse <= 1e-3 and out_max_abs <= 2e-2, otherwise the bench fails;
//   - output sentinel tail (out tensor has +16 extra elements) must stay -9.
//
// Protocol (K4 section 4.3, docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md):
// reset before the timed barrier (reset excluded), `--warm` warm-ups per
// (cell, arm), samples round-robin across cells, timing = wall clock between
// ext_oneapi_submit_barrier calls plus the device span from profiling events.
//
// Output: one [GDN BENCH] JSON line per (arm, vh, rows) plus [GDN GATE] lines
// that apply the frozen K4 R3 thresholds (kernel >= 3% at both shapes,
// projected e2e >= 5% TTFT on the 14K fixture).
#include "core/Context.hpp"
#include "ops/GatedDeltaNetOps.hpp"
#include "ops/GdnPrefillVariant.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using bf16 = sycl::ext::oneapi::bfloat16;

// K4 constants for the projected-e2e gate (docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md
// sections 3c/4.4): 36 linear layers, 13944-token 14K request in 4096-token
// chunks (3 x 4096 + 1656 = 3.4045 chunk equivalents), 71.82 s TTFT.
constexpr double kGdnLayers = 36.0;
constexpr double kRequestChunkEquivalents = 13944.0 / 4096.0;
constexpr double kTtftBaselineMs = 71820.0;
constexpr double kKernelGatePct = 3.0;
constexpr double kE2eGatePct = 5.0;
// M1 anchor: vh=48 n=4096 mode 0 wall median on 2026-10-10 (HEAD e808f63).
constexpr double kAnchorMsVh48Rows4096 = 74.8172;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - static_cast<double>(lo));
}

struct Error {
    double err = 0, ref = 0, max = 0;
    void add(double a, double b) {
        require(std::isfinite(a) && std::isfinite(b), "non-finite GDN value");
        const double d = a - b;
        err += d * d;
        ref += a * a;
        max = std::max(max, std::abs(d));
    }
    double nrmse() const { return std::sqrt(err / std::max(ref, 1e-24)); }
};

double silu(double x) { return x / (1 + std::exp(-x)); }
double softplus(double x) { return x > 20 ? x : x < -20 ? std::exp(x) : std::log1p(std::exp(x)); }

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

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    for (const auto& token : split_list(text)) values.push_back(std::stoi(token));
    return values;
}

// Mirrors ops::clamp_slm_rows so the bench can name arms and model the
// expected global traffic without touching the kernel.
int effective_slm_rows(int rows) {
    if (rows < 0) rows = 0;
    if (rows > ops::kGdnSlmRowsMax) rows = ops::kGdnSlmRowsMax;
    return rows - (rows % 4);
}

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
          g(ops::DeltaNetGeometry::create(qh, vh, 128, 128, 4, 1e-6f, ops::GateActivation::Sigmoid)),
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
        return {conv_w.data_as<bf16>(), norm_w.data_as<float>(), a_log.data_as<float>(),
                dt_bias.data_as<bf16>()};
    }

    ops::DeltaNetState reset(bool full) {
        ctx.memcpy_h2d(state.data(), initial.data(), state.size_bytes());
        ctx.memcpy_h2d(conv_state.data(), history.data(), conv_state.size_bytes());
        if (full) {
            std::vector<bf16> sentinel(out.numel(), bf16(-9.f));
            ctx.memcpy_h2d(out.data(), sentinel.data(), out.size_bytes());
        }
        return {state.data_as<float>(), conv_state.data_as<float>(), initial_head, false};
    }
};

struct Prepared {
    std::vector<double> q, k, v, beta, log_decay, history;
};

Prepared prepare_cpu(const GdnFixture& f) {
    Prepared p;
    const auto& g = f.g;
    p.q.resize(size_t(f.count) * g.kv_dim);
    p.k.resize(p.q.size());
    p.v.resize(p.q.size());
    p.beta.resize(f.count * g.v_heads);
    p.log_decay.resize(p.beta.size());
    p.history.assign(f.history.begin(), f.history.end());
    std::vector<double> conv(g.qkv_dim);
    int ring = f.initial_head;
    for (int t = 0; t < f.count; ++t) {
        const auto* row = f.input.data() + size_t(t) * g.all_dim;
        for (int c = 0; c < g.qkv_dim; ++c) {
            double v = double(float(row[c])) * float(f.cw[c * 4 + 3]);
            for (int j = 0; j < 3; ++j) v += p.history[((ring + j) % 3) * g.qkv_dim + c] * float(f.cw[c * 4 + j]);
            conv[c] = silu(v);
        }
        for (int c = 0; c < g.qkv_dim; ++c) p.history[ring * g.qkv_dim + c] = float(row[c]);
        ring = (ring + 1) % 3;
        for (int h = 0; h < g.v_heads; ++h) {
            const int hk = h / (g.v_heads / g.q_heads);
            double qs = 0, ks = 0;
            for (int j = 0; j < 128; ++j) {
                qs += conv[hk * 128 + j] * conv[hk * 128 + j];
                ks += conv[g.q_dim + hk * 128 + j] * conv[g.q_dim + hk * 128 + j];
            }
            const size_t base = (size_t(t) * g.v_heads + h) * 128;
            for (int j = 0; j < 128; ++j) {
                p.q[base + j] = conv[hk * 128 + j] / std::sqrt(qs + g.eps);
                p.k[base + j] = conv[g.q_dim + hk * 128 + j] / std::sqrt(ks + g.eps);
                p.v[base + j] = conv[2 * g.q_dim + h * 128 + j];
            }
            const size_t scalar = size_t(t) * g.v_heads + h;
            p.beta[scalar] = 1 / (1 + std::exp(-double(float(row[g.qkv_dim + g.z_dim + g.v_heads + h]))));
            p.log_decay[scalar] = -std::exp(double(f.al[h])) *
                                  softplus(double(float(row[g.qkv_dim + g.z_dim + h])) + float(f.dt[h]));
        }
    }
    return p;
}

struct CpuResult {
    std::vector<double> state, raw;
};

CpuResult recurrent_cpu(const GdnFixture& f, const Prepared& p) {
    CpuResult r;
    r.state.assign(f.initial.begin(), f.initial.end());
    r.raw.resize(size_t(f.count) * f.g.kv_dim);
    for (int t = 0; t < f.count; ++t)
        for (int h = 0; h < f.g.v_heads; ++h) {
            const size_t base = (size_t(t) * f.g.v_heads + h) * 128;
            auto* s = r.state.data() + h * 128 * 128;
            const double a = std::exp(p.log_decay[t * f.g.v_heads + h]);
            const double b = p.beta[t * f.g.v_heads + h];
            for (int v = 0; v < 128; ++v) {
                double dot = 0;
                for (int k = 0; k < 128; ++k) dot += a * s[k * 128 + v] * p.k[base + k];
                const double d = b * (p.v[base + v] - dot);
                double o = 0;
                for (int k = 0; k < 128; ++k) {
                    s[k * 128 + v] = a * s[k * 128 + v] + p.k[base + k] * d;
                    o += s[k * 128 + v] * p.q[base + k];
                }
                r.raw[base + v] = o;
            }
        }
    return r;
}

// Mirrors tests/ops/GdnChunkTests.cpp check_gpu: state NRMSE, conv history
// equality and the normalized/gated bf16 output against the f64 reference.
void check_gpu(const GdnFixture& f, const Prepared& p, const CpuResult& ref, const char* label) {
    std::vector<float> state(ref.state.size()), history(p.history.size());
    std::vector<bf16> output(static_cast<size_t>(f.out.numel()));
    f.ctx.memcpy_d2h(state.data(), f.state.data(), state.size() * sizeof(float));
    f.ctx.memcpy_d2h(history.data(), f.conv_state.data(), history.size() * sizeof(float));
    f.ctx.memcpy_d2h(output.data(), f.out.data(), output.size() * sizeof(bf16));
    Error es, eo;
    for (size_t i = 0; i < state.size(); ++i) es.add(ref.state[i], state[i]);
    for (size_t i = 0; i < history.size(); ++i) require(history[i] == p.history[i], "conv history changed");
    for (int t = 0; t < f.count; ++t)
        for (int h = 0; h < f.g.v_heads; ++h) {
            const size_t base = (size_t(t) * f.g.v_heads + h) * 128;
            double ss = 0;
            for (int v = 0; v < 128; ++v) ss += std::pow(ref.raw[base + v] * f.g.scale, 2);
            const double norm = 1 / std::sqrt(ss / 128 + f.g.eps);
            for (int v = 0; v < 128; ++v) {
                const double z = float(f.input[size_t(t) * f.g.all_dim + f.g.qkv_dim + h * 128 + v]);
                const double gate = f.g.gate_act == ops::GateActivation::Silu ? silu(z) : 1 / (1 + std::exp(-z));
                const double expected = float(bf16(float(ref.raw[base + v] * f.g.scale * norm * f.nw[v] * gate)));
                eo.add(expected, float(output[base + v]));
            }
        }
    for (size_t i = ref.raw.size(); i < output.size(); ++i)
        require(float(output[i]) == -9.f, "GDN output guard overwritten");
    std::cout << "[GDN GUARD] label=" << label << " state_nrmse=" << es.nrmse()
              << " state_max_abs=" << es.max << " out_nrmse=" << eo.nrmse()
              << " out_max_abs=" << eo.max << "\n";
    require(es.nrmse() < .015 && eo.nrmse() < .015, "GDN numerical canary failed");
}

void run_arm(GdnFixture& f, const ops::DeltaNetWeights& w, int variant, int r, ops::DeltaNetState& st) {
    if (variant == 0) {
        ops::run_linear_delta_prefill_gpu_batched(f.ctx, f.g, w, st, f.in.data_as<bf16>(),
                                                  f.out.data_as<bf16>(), f.count);
        return;
    }
    require(ops::run_linear_delta_prefill_gpu_batched_slm(f.ctx, f.g, w, st, f.in.data_as<bf16>(),
                                                          f.out.data_as<bf16>(), f.count, r),
            "SLM variant rejected the bench geometry");
}

struct Snapshot {
    std::vector<float> state, conv;
    std::vector<bf16> out;
};

Snapshot capture(GdnFixture& f, const ops::DeltaNetWeights& w, int variant, int r) {
    auto st = f.reset(true);
    run_arm(f, w, variant, r, st);
    f.ctx.synchronize();
    Snapshot s;
    s.state.resize(f.initial.size());
    s.conv.resize(f.history.size());
    s.out.resize(static_cast<size_t>(f.out.numel()));
    f.ctx.memcpy_d2h(s.state.data(), f.state.data(), f.state.size_bytes());
    f.ctx.memcpy_d2h(s.conv.data(), f.conv_state.data(), f.conv_state.size_bytes());
    f.ctx.memcpy_d2h(s.out.data(), f.out.data(), f.out.size_bytes());
    return s;
}

struct Parity {
    long long state_first_diff = -1, conv_first_diff = -1, out_first_diff = -1;
    double state_nrmse = 0, out_max_abs = 0;
    bool bitwise() const { return state_first_diff < 0 && conv_first_diff < 0 && out_first_diff < 0; }
    bool pass() const { return bitwise() || (state_nrmse <= 1e-3 && out_max_abs <= 2e-2); }
    const char* label() const { return bitwise() ? "bitwise" : "tolerance"; }
};

Parity compare_snapshots(const Snapshot& a, const Snapshot& b) {
    require(a.state.size() == b.state.size() && a.conv.size() == b.conv.size() &&
                a.out.size() == b.out.size(),
            "snapshot shape mismatch");
    Parity p;
    for (size_t i = 0; i < a.state.size(); ++i)
        if (std::memcmp(&a.state[i], &b.state[i], sizeof(float)) != 0) {
            p.state_first_diff = static_cast<long long>(i);
            break;
        }
    for (size_t i = 0; i < a.conv.size(); ++i)
        if (std::memcmp(&a.conv[i], &b.conv[i], sizeof(float)) != 0) {
            p.conv_first_diff = static_cast<long long>(i);
            break;
        }
    for (size_t i = 0; i < a.out.size(); ++i)
        if (std::memcmp(&a.out[i], &b.out[i], sizeof(bf16)) != 0) {
            p.out_first_diff = static_cast<long long>(i);
            break;
        }
    Error es;
    for (size_t i = 0; i < a.state.size(); ++i) es.add(a.state[i], b.state[i]);
    p.state_nrmse = es.nrmse();
    for (size_t i = 0; i < a.out.size(); ++i) {
        const double diff = std::abs(double(a.out[i]) - double(b.out[i]));
        p.out_max_abs = std::max(p.out_max_abs, diff);
    }
    return p;
}

struct Sample {
    double wall_ms = 0;
    double span_ms = 0;
};

Sample time_once(GdnFixture& f, const ops::DeltaNetWeights& w, int variant, int r) {
    auto st = f.reset(false);
    const auto start = Clock::now();
    auto a = f.ctx.queue().ext_oneapi_submit_barrier();
    run_arm(f, w, variant, r, st);
    auto b = f.ctx.queue().ext_oneapi_submit_barrier();
    b.wait_and_throw();
    Sample s;
    s.wall_ms = elapsed_ms(start, Clock::now());
    s.span_ms = static_cast<double>(
                    b.get_profiling_info<sycl::info::event_profiling::command_start>() -
                    a.get_profiling_info<sycl::info::event_profiling::command_end>()) *
                1e-6;
    return s;
}

struct Arm {
    std::string name;
    int variant = 0; // 0 = production kernel, 1 = SLM-warm-row variant
    int r = 0;       // requested SLM rows (variant 1 only)
};

struct Cell {
    int vh = 0;
    int rows = 0;
    std::unique_ptr<GdnFixture> f;
};

} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    aila::env::enable_l0_copy_engine();
    try {
        std::vector<int> vh_list{48};
        std::vector<int> rows_list{4096, 8192};
        std::vector<int> r_list{120};
        std::vector<std::string> arm_names{"baseline", "slm"};
        int samples_n = 15;
        int warm_n = 2;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value after ") + argv[i]);
            const std::string key(argv[i]);
            const std::string value(argv[i + 1]);
            if (key == "--vh") vh_list = parse_int_list(value);
            else if (key == "--rows") rows_list = parse_int_list(value);
            else if (key == "--r") r_list = parse_int_list(value);
            else if (key == "--samples") samples_n = std::stoi(value);
            else if (key == "--warm") warm_n = std::stoi(value);
            else if (key == "--arms") arm_names = split_list(value);
            else throw std::invalid_argument("unknown option: " + key);
        }
        require(!vh_list.empty() && !rows_list.empty() && !r_list.empty() && samples_n > 0 &&
                    warm_n >= 0,
                "invalid bench arguments");
        for (int vh : vh_list) require(vh > 0 && vh % 16 == 0, "vh must be a positive multiple of q_heads=16");
        for (int rows : rows_list) require(rows > 0 && rows % 4 == 0, "rows must be a positive multiple of 4");

        Context ctx(nullptr, true);
        std::cout << "[GDN DEVICE] " << ctx.queue().get_device().get_info<sycl::info::device::name>()
                  << " driver=" << ctx.queue().get_device().get_info<sycl::info::device::driver_version>()
                  << "\n";

        std::vector<Arm> arms;
        for (const auto& name : arm_names) {
            if (name == "baseline") {
                arms.push_back({"baseline", 0, 0});
            } else if (name == "slm") {
                for (int r : r_list) {
                    const int effective = effective_slm_rows(r);
                    arms.push_back({"slm_r" + std::to_string(effective), 1, r});
                }
            } else {
                throw std::invalid_argument("unknown arm: " + name);
            }
        }
        require(!arms.empty(), "no arms selected");

        // --- CPU canary on a short window (64 tokens) ---------------------
        {
            const int canary_vh = vh_list.front();
            int canary_r = 0;
            for (const auto& arm : arms)
                if (arm.variant == 1) canary_r = std::max(canary_r, effective_slm_rows(arm.r));
            GdnFixture f(ctx, 64, 16, canary_vh);
            const auto w = f.weights();
            const auto p = prepare_cpu(f);
            const auto ref = recurrent_cpu(f, p);
            {
                auto st = f.reset(true);
                run_arm(f, w, 0, 0, st);
                check_gpu(f, p, ref, "canary_baseline");
            }
            {
                auto st = f.reset(true);
                run_arm(f, w, 1, canary_r, st);
                check_gpu(f, p, ref, "canary_slm");
            }
        }

        // --- cells and parity guards (untimed) ----------------------------
        std::vector<Cell> cells;
        for (int vh : vh_list)
            for (int rows : rows_list) {
                Cell c;
                c.vh = vh;
                c.rows = rows;
                c.f = std::make_unique<GdnFixture>(ctx, rows, 16, vh);
                cells.push_back(std::move(c));
            }

        std::vector<std::vector<Parity>> parities(
            cells.size(), std::vector<Parity>(arms.size()));
        const bool has_slm = std::any_of(arms.begin(), arms.end(),
                                         [](const Arm& a) { return a.variant == 1; });
        if (has_slm) {
            for (size_t ci = 0; ci < cells.size(); ++ci) {
                auto& cell = cells[ci];
                const auto w = cell.f->weights();
                const auto base = capture(*cell.f, w, 0, 0);
                for (size_t ai = 0; ai < arms.size(); ++ai) {
                    if (arms[ai].variant == 0) continue;
                    const auto v = capture(*cell.f, w, arms[ai].variant, arms[ai].r);
                    parities[ci][ai] = compare_snapshots(base, v);
                    const auto& p = parities[ci][ai];
                    std::cout << "[GDN PARITY] vh=" << cell.vh << " rows=" << cell.rows
                              << " arm=" << arms[ai].name << " bitwise=" << (p.bitwise() ? 1 : 0)
                              << " state_first_diff=" << p.state_first_diff
                              << " conv_first_diff=" << p.conv_first_diff
                              << " out_first_diff=" << p.out_first_diff
                              << " state_nrmse=" << p.state_nrmse
                              << " out_max_abs=" << p.out_max_abs
                              << " pass=" << (p.pass() ? "yes" : "FAIL") << "\n";
                    require(p.pass(), "SLM variant parity guard failed");
                }
            }
        }

        // --- warm-ups and round-robin samples -----------------------------
        std::vector<std::vector<std::vector<Sample>>> raw(
            cells.size(), std::vector<std::vector<Sample>>(arms.size()));
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            const auto w = cells[ci].f->weights();
            for (size_t ai = 0; ai < arms.size(); ++ai)
                for (int warm = 0; warm < warm_n; ++warm)
                    time_once(*cells[ci].f, w, arms[ai].variant, arms[ai].r);
        }
        for (int sample = 0; sample < samples_n; ++sample)
            for (size_t ci = 0; ci < cells.size(); ++ci) {
                const auto w = cells[ci].f->weights();
                for (size_t ai = 0; ai < arms.size(); ++ai)
                    raw[ci][ai].push_back(
                        time_once(*cells[ci].f, w, arms[ai].variant, arms[ai].r));
            }

        // --- report --------------------------------------------------------
        struct Result {
            double median_ms = 0, p10_ms = 0, p90_ms = 0, span_ms = 0;
            double gbps = 0, dram_gbps = 0;
        };
        std::vector<std::vector<Result>> results(cells.size(), std::vector<Result>(arms.size()));
        std::cout << std::fixed << std::setprecision(3);
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            const double vh = cells[ci].vh;
            const double rows = cells[ci].rows;
            // Nominal state stream: 2R + 1W per token over all heads.
            const double nominal_bytes = 3.0 * 65536.0 * vh * rows;
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                std::vector<double> wall, span;
                for (const auto& s : raw[ci][ai]) {
                    wall.push_back(s.wall_ms);
                    span.push_back(s.span_ms);
                }
                Result r;
                r.median_ms = quantile(wall, 0.5);
                r.p10_ms = quantile(wall, 0.1);
                r.p90_ms = quantile(wall, 0.9);
                r.span_ms = quantile(span, 0.5);
                r.gbps = nominal_bytes / (r.median_ms * 1e6);
                const int staged = arms[ai].variant == 0 ? 0 : effective_slm_rows(arms[ai].r);
                const double cold_frac = (128.0 - staged) / 128.0;
                // Staging + unstaging touches the SLM rows once (1R + 1W).
                const double dram_bytes = nominal_bytes * cold_frac +
                                          2.0 * 65536.0 * vh * (staged / 128.0);
                r.dram_gbps = dram_bytes / (r.median_ms * 1e6);
                results[ci][ai] = r;
                std::ostringstream js;
                js << "[GDN BENCH] {\"schema\":\"aila.gdn-slm-bench.v1\""
                   << ",\"rows\":" << cells[ci].rows
                   << ",\"median_ms\":" << r.median_ms
                   << ",\"gbps\":" << r.gbps
                   << ",\"arm\":\"" << arms[ai].name << "\""
                   << ",\"variant\":" << arms[ai].variant
                   << ",\"vh\":" << cells[ci].vh
                   << ",\"r_slm\":" << (arms[ai].variant == 0 ? 0 : effective_slm_rows(arms[ai].r))
                   << ",\"p10_ms\":" << r.p10_ms
                   << ",\"p90_ms\":" << r.p90_ms
                   << ",\"device_span_ms\":" << r.span_ms
                   << ",\"wall_over_span\":" << (r.span_ms > 0 ? r.median_ms / r.span_ms : 0.0)
                   << ",\"samples\":" << wall.size()
                   << ",\"nominal_bytes\":" << static_cast<long long>(nominal_bytes)
                   << ",\"dram_gbps\":" << r.dram_gbps
                   << ",\"parity\":\"" << (arms[ai].variant == 0 ? "n/a" : parities[ci][ai].label())
                   << "\"}\n";
                std::cout << js.str();
            }
        }

        // --- anchor re-check (K4 section 4.2) -----------------------------
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            if (cells[ci].vh != 48 || cells[ci].rows != 4096) continue;
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                if (arms[ai].variant != 0) continue;
                const double rel = (results[ci][ai].median_ms - kAnchorMsVh48Rows4096) /
                                   kAnchorMsVh48Rows4096 * 100.0;
                std::cout << "[GDN ANCHOR] vh=48 rows=4096 anchor_ms=" << kAnchorMsVh48Rows4096
                          << " median_ms=" << results[ci][ai].median_ms
                          << " rel_diff_pct=" << rel
                          << " within_3pct=" << (std::abs(rel) <= 3.0 ? 1 : 0) << "\n";
            }
        }

        // --- K4 R3 gates ---------------------------------------------------
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            int base_ai = -1;
            for (size_t ai = 0; ai < arms.size(); ++ai)
                if (arms[ai].variant == 0) base_ai = static_cast<int>(ai);
            if (base_ai < 0) continue;
            const double base_ms = results[ci][base_ai].median_ms;
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                if (arms[ai].variant == 0) continue;
                const double saving = base_ms - results[ci][ai].median_ms;
                const double pct = 100.0 * saving / base_ms;
                const bool kernel_gate = pct >= kKernelGatePct;
                const double e2e_pct = cells[ci].rows == 4096
                                           ? 100.0 * saving * kGdnLayers * kRequestChunkEquivalents /
                                                 kTtftBaselineMs
                                           : -1.0;
                const bool e2e_gate = e2e_pct >= kE2eGatePct;
                std::cout << "[GDN GATE] {\"vh\":" << cells[ci].vh
                          << ",\"rows\":" << cells[ci].rows
                          << ",\"r_slm\":" << effective_slm_rows(arms[ai].r)
                          << ",\"baseline_ms\":" << base_ms
                          << ",\"slm_ms\":" << results[ci][ai].median_ms
                          << ",\"kernel_saving_pct\":" << pct
                          << ",\"kernel_gate\":" << (kernel_gate ? 1 : 0)
                          << ",\"e2e_projected_pct\":" << e2e_pct
                          << ",\"e2e_gate\":" << (e2e_gate ? 1 : 0) << "}\n";
            }
        }
        // Two-shape confirmation per (vh, slm arm): >= 3% at rows=4096 and
        // rows=8192 (only shapes actually measured are considered).
        for (int vh : vh_list) {
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                if (arms[ai].variant == 0) continue;
                double min_pct = 1e30;
                bool have_any = false;
                double e2e_pct = -1.0;
                for (size_t ci = 0; ci < cells.size(); ++ci) {
                    if (cells[ci].vh != vh) continue;
                    int base_ai = -1;
                    for (size_t aj = 0; aj < arms.size(); ++aj)
                        if (arms[aj].variant == 0) base_ai = static_cast<int>(aj);
                    if (base_ai < 0) continue;
                    const double base_ms = results[ci][base_ai].median_ms;
                    const double pct =
                        100.0 * (base_ms - results[ci][ai].median_ms) / base_ms;
                    min_pct = std::min(min_pct, pct);
                    have_any = true;
                    if (cells[ci].rows == 4096)
                        e2e_pct = 100.0 * (base_ms - results[ci][ai].median_ms) * kGdnLayers *
                                  kRequestChunkEquivalents / kTtftBaselineMs;
                }
                if (!have_any) continue;
                const bool two_shape = min_pct >= kKernelGatePct;
                const bool pass = two_shape && e2e_pct >= kE2eGatePct;
                std::cout << "[GDN TWO-SHAPE] {\"vh\":" << vh
                          << ",\"r_slm\":" << effective_slm_rows(arms[ai].r)
                          << ",\"min_saving_pct\":" << min_pct
                          << ",\"two_shape_kernel_gate\":" << (two_shape ? 1 : 0)
                          << ",\"e2e_projected_pct\":" << e2e_pct
                          << ",\"pass\":" << (pass ? 1 : 0) << "}\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[GDN FAIL] " << e.what() << "\n";
        return 1;
    }
}
