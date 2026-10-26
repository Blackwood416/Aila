#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/HyperConnectionOps.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/GatedDeltaNetOps.hpp"
#include "ops/QsaIndexerOps.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/Ops.hpp"
#include "ops/AttentionOps.hpp"
#include "ops/Qwen4ExpAttentionPair.hpp"
#include "utils/EnvUtils.hpp"
#include "../models/Qwen4ExpGreedyProbe.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

template <typename Fn>
void expect_invalid_argument(const std::string& name, Fn&& fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(name + ": expected invalid_argument");
    }
}

void check_close(const std::vector<bf16>& actual, const std::vector<float>& expected,
                 const std::string& name, float tol = 2e-2f) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + " size mismatch: actual=" +
                                 std::to_string(actual.size()) + " expected=" +
                                 std::to_string(expected.size()));
    }
    float max_diff = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float act = static_cast<float>(actual[i]);
        float exp = expected[i];
        if (!std::isfinite(act) || !std::isfinite(exp)) throw std::runtime_error(name + " nonfinite");
        float diff = std::abs(act - exp);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    if (max_diff > tol) {
        throw std::runtime_error(name + " failed: max_diff=" + std::to_string(max_diff) +
                                 " at index " + std::to_string(max_idx) +
                                 " (actual=" + std::to_string(static_cast<float>(actual[max_idx])) +
                                 ", expected=" + std::to_string(expected[max_idx]) + ")");
    }
    std::cout << "[PASS] " << name << " (max_diff=" << max_diff << " <= " << tol << ")\n";
}

void check_close_f32(const std::vector<float>& actual, const std::vector<float>& expected,
                     const std::string& name, float tol = 1e-4f) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + " size mismatch");
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw std::runtime_error(name + " nonfinite");
        float diff = std::abs(actual[i] - expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    if (max_diff > tol) {
        throw std::runtime_error(name + " failed: max_diff=" + std::to_string(max_diff) +
                                 " > " + std::to_string(tol));
    }
    std::cout << "[PASS] " << name << " (max_diff=" << max_diff << " <= " << tol << ")\n";
}

static void run_grouped_rms_norm_test(Context& ctx, int rows) {
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    constexpr float eps = 1e-6f;
    const size_t total = rows * groups * group_dim;

    std::mt19937 rng(42 + rows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> x_host(total);
    std::vector<float> w_host(groups * group_dim);
    for (size_t i = 0; i < total; ++i) x_host[i] = bf16(dist(rng));
    for (size_t i = 0; i < w_host.size(); ++i) w_host[i] = dist(rng) * 0.1f;

    // CPU Reference
    std::vector<float> ref_out(total);
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < groups; ++g) {
            size_t base = static_cast<size_t>(r) * (groups * group_dim) + g * group_dim;
            size_t w_base = static_cast<size_t>(g) * group_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < group_dim; ++d) {
                float v = static_cast<float>(x_host[base + d]);
                sum_sq += v * v;
            }
            float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(group_dim) + eps);
            for (int d = 0; d < group_dim; ++d) {
                float v = static_cast<float>(x_host[base + d]);
                ref_out[base + d] = v * inv_rms * (1.0f + w_host[w_base + d]);
            }
        }
    }

    Tensor x_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);
    Tensor w_dev = Tensor::allocate(ctx, {groups * group_dim}, dnnl::memory::data_type::f32);
    Tensor out_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(x_dev.data(), x_host.data(), total * sizeof(bf16));
    ctx.memcpy_h2d(w_dev.data(), w_host.data(), w_host.size() * sizeof(float));

    ops::grouped_rms_norm(ctx, static_cast<const bf16*>(x_dev.data()),
                          static_cast<const float*>(w_dev.data()), eps,
                          static_cast<bf16*>(out_dev.data()),
                          rows, groups, group_dim, /*plus_one=*/true);
    ctx.synchronize();

    std::vector<bf16> gpu_out(total);
    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), total * sizeof(bf16));
    check_close(gpu_out, ref_out, "grouped_rms_norm rows=" + std::to_string(rows) + " (plus_one=true)", 1e-2f);
}

void test_grouped_rms_norm(Context& ctx) {
    run_grouped_rms_norm_test(ctx, 1);
    run_grouped_rms_norm_test(ctx, 2);
}

static void run_hc_mix_reduce_test(Context& ctx, int rows, bool apply_sigmoid_to_mix) {
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    const size_t total_in = static_cast<size_t>(rows) * groups * group_dim;
    const size_t total_out = static_cast<size_t>(rows) * group_dim;

    std::mt19937 rng(123 + rows + (apply_sigmoid_to_mix ? 500 : 0));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> N_host(total_in);
    std::vector<bf16> mix_host(total_in);
    for (size_t i = 0; i < total_in; ++i) {
        N_host[i] = bf16(dist(rng));
        mix_host[i] = bf16(dist(rng));
    }

    std::vector<float> ref_out(total_out);
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < group_dim; ++d) {
            float sum = 0.0f;
            for (int g = 0; g < groups; ++g) {
                size_t off = static_cast<size_t>(r) * (groups * group_dim) + g * group_dim + d;
                float m = static_cast<float>(mix_host[off]);
                if (apply_sigmoid_to_mix) {
                    m = static_cast<float>(bf16(1.0f / (1.0f + std::exp(-m))));
                }
                sum += m * static_cast<float>(N_host[off]);
            }
            ref_out[r * group_dim + d] = sum / static_cast<float>(groups);
        }
    }

    Tensor N_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);
    Tensor mix_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);
    Tensor out_dev = Tensor::allocate(ctx, {rows, group_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(N_dev.data(), N_host.data(), total_in * sizeof(bf16));
    ctx.memcpy_h2d(mix_dev.data(), mix_host.data(), total_in * sizeof(bf16));

    ops::hc_mix_reduce(ctx, static_cast<const bf16*>(N_dev.data()),
                       static_cast<const bf16*>(mix_dev.data()),
                       static_cast<bf16*>(out_dev.data()),
                       rows, group_dim, groups, apply_sigmoid_to_mix);
    ctx.synchronize();

    std::vector<bf16> gpu_out(total_out);
    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), total_out * sizeof(bf16));
    std::string tag = "hc_mix_reduce rows=" + std::to_string(rows) +
                      (apply_sigmoid_to_mix ? " (fused sigmoid)" : " (plain)");
    check_close(gpu_out, ref_out, tag, 1e-2f);
}

void test_hc_mix_reduce(Context& ctx) {
    for (int r : {1, 2, 4, 32}) {
        run_hc_mix_reduce_test(ctx, r, false);
        run_hc_mix_reduce_test(ctx, r, true);
    }
}

static void run_hc_inject_add_test(Context& ctx, int rows) {
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    const size_t total_H = rows * groups * group_dim;
    const size_t total_y = rows * group_dim;
    const size_t total_inj = rows * groups;

    std::mt19937 rng(456 + rows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> H_host(total_H);
    std::vector<bf16> y_host(total_y);
    std::vector<float> inj_host(total_inj);

    for (size_t i = 0; i < total_H; ++i) H_host[i] = bf16(dist(rng));
    for (size_t i = 0; i < total_y; ++i) y_host[i] = bf16(dist(rng));
    for (size_t i = 0; i < total_inj; ++i) inj_host[i] = 2.0f / (1.0f + std::exp(-dist(rng)));

    std::vector<float> ref_out(total_H);
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < groups; ++g) {
            float inj_val = inj_host[r * groups + g];
            for (int d = 0; d < group_dim; ++d) {
                size_t off = static_cast<size_t>(r) * (groups * group_dim) + g * group_dim + d;
                float h_val = static_cast<float>(H_host[off]);
                float y_val = static_cast<float>(y_host[r * group_dim + d]);
                ref_out[off] = h_val + inj_val * y_val;
            }
        }
    }

    Tensor H_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);
    Tensor y_dev = Tensor::allocate(ctx, {rows, group_dim}, dnnl::memory::data_type::bf16);
    Tensor inj_dev = Tensor::allocate(ctx, {rows, groups}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(H_dev.data(), H_host.data(), total_H * sizeof(bf16));
    ctx.memcpy_h2d(y_dev.data(), y_host.data(), total_y * sizeof(bf16));
    ctx.memcpy_h2d(inj_dev.data(), inj_host.data(), total_inj * sizeof(float));

    ops::hc_inject_add(ctx, static_cast<bf16*>(H_dev.data()),
                       static_cast<const bf16*>(y_dev.data()),
                       static_cast<const float*>(inj_dev.data()),
                       rows, group_dim, groups);
    ctx.synchronize();

    std::vector<bf16> gpu_out(total_H);
    ctx.memcpy_d2h(gpu_out.data(), H_dev.data(), total_H * sizeof(bf16));
    check_close(gpu_out, ref_out, "hc_inject_add rows=" + std::to_string(rows), 1e-2f);
}

void test_hc_inject_add(Context& ctx) {
    run_hc_inject_add_test(ctx, 1);
    run_hc_inject_add_test(ctx, 2);
}

static void run_hc_inject_add_and_grouped_rms_norm_test(Context& ctx, int rows) {
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    constexpr float eps = 1e-6f;
    const size_t total_H = rows * groups * group_dim;
    const size_t total_y = rows * group_dim;
    const size_t total_inj = rows * groups;
    const size_t total_w = groups * group_dim;

    std::mt19937 rng(888 + rows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> H_init(total_H);
    std::vector<bf16> y_host(total_y);
    std::vector<float> inj_host(total_inj);
    std::vector<float> norm_w_host(total_w);

    for (size_t i = 0; i < total_H; ++i) H_init[i] = bf16(dist(rng));
    for (size_t i = 0; i < total_y; ++i) y_host[i] = bf16(dist(rng));
    for (size_t i = 0; i < total_inj; ++i) inj_host[i] = 2.0f / (1.0f + std::exp(-dist(rng)));
    for (size_t i = 0; i < total_w; ++i) norm_w_host[i] = dist(rng) * 0.1f;

    Tensor H_sep_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);
    Tensor H_fused_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);
    Tensor y_dev = Tensor::allocate(ctx, {rows, group_dim}, dnnl::memory::data_type::bf16);
    Tensor inj_dev = Tensor::allocate(ctx, {rows, groups}, dnnl::memory::data_type::f32);
    Tensor norm_w_dev = Tensor::allocate(ctx, {groups * group_dim}, dnnl::memory::data_type::f32);
    Tensor out_sep_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);
    Tensor out_fused_dev = Tensor::allocate(ctx, {rows, groups * group_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(H_sep_dev.data(), H_init.data(), total_H * sizeof(bf16));
    ctx.memcpy_h2d(H_fused_dev.data(), H_init.data(), total_H * sizeof(bf16));
    ctx.memcpy_h2d(y_dev.data(), y_host.data(), total_y * sizeof(bf16));
    ctx.memcpy_h2d(inj_dev.data(), inj_host.data(), total_inj * sizeof(float));
    ctx.memcpy_h2d(norm_w_dev.data(), norm_w_host.data(), total_w * sizeof(float));

    // 1. Separate: hc_inject_add followed by grouped_rms_norm
    ops::hc_inject_add(ctx, static_cast<bf16*>(H_sep_dev.data()),
                       static_cast<const bf16*>(y_dev.data()),
                       static_cast<const float*>(inj_dev.data()),
                       rows, group_dim, groups);
    ops::grouped_rms_norm(ctx, static_cast<const bf16*>(H_sep_dev.data()),
                          static_cast<const float*>(norm_w_dev.data()),
                          eps, static_cast<bf16*>(out_sep_dev.data()),
                          rows, groups, group_dim, true);
    ctx.synchronize();

    // 2. Fused: hc_inject_add_and_grouped_rms_norm
    ops::hc_inject_add_and_grouped_rms_norm(
        ctx, static_cast<bf16*>(H_fused_dev.data()),
        static_cast<const bf16*>(y_dev.data()),
        static_cast<const float*>(inj_dev.data()),
        static_cast<const float*>(norm_w_dev.data()),
        eps, static_cast<bf16*>(out_fused_dev.data()),
        rows, groups, group_dim, true);
    ctx.synchronize();

    std::vector<bf16> H_sep_h(total_H);
    std::vector<bf16> H_fused_h(total_H);
    std::vector<bf16> out_sep_h(total_H);
    std::vector<bf16> out_fused_h(total_H);

    ctx.memcpy_d2h(H_sep_h.data(), H_sep_dev.data(), total_H * sizeof(bf16));
    ctx.memcpy_d2h(H_fused_h.data(), H_fused_dev.data(), total_H * sizeof(bf16));
    ctx.memcpy_d2h(out_sep_h.data(), out_sep_dev.data(), total_H * sizeof(bf16));
    ctx.memcpy_d2h(out_fused_h.data(), out_fused_dev.data(), total_H * sizeof(bf16));

    // Generic scalar fusion permits 1 ULP; this is not a bitwise assertion.
    int h_ulp_diffs = 0;
    for (size_t i = 0; i < total_H; ++i) {
        uint16_t b_sep = *reinterpret_cast<const uint16_t*>(&H_sep_h[i]);
        uint16_t b_fused = *reinterpret_cast<const uint16_t*>(&H_fused_h[i]);
        if (b_sep != b_fused) {
            int diff = std::abs(static_cast<int>(b_sep) - static_cast<int>(b_fused));
            if (diff > 1) {
                throw std::runtime_error("H mismatch > 1 ULP at index " + std::to_string(i) +
                                         " sep=" + std::to_string(static_cast<float>(H_sep_h[i])) +
                                         " fused=" + std::to_string(static_cast<float>(H_fused_h[i])));
            }
            h_ulp_diffs++;
        }
    }

    // The same generic 1-ULP bound for normalized output.
    int out_ulp_diffs = 0;
    for (size_t i = 0; i < total_H; ++i) {
        uint16_t b_sep = *reinterpret_cast<const uint16_t*>(&out_sep_h[i]);
        uint16_t b_fused = *reinterpret_cast<const uint16_t*>(&out_fused_h[i]);
        if (b_sep != b_fused) {
            int diff = std::abs(static_cast<int>(b_sep) - static_cast<int>(b_fused));
            if (diff > 1) {
                throw std::runtime_error("out_N mismatch > 1 ULP at index " + std::to_string(i) +
                                         " sep=" + std::to_string(static_cast<float>(out_sep_h[i])) +
                                         " fused=" + std::to_string(static_cast<float>(out_fused_h[i])));
            }
            out_ulp_diffs++;
        }
    }

    std::cout << "[PASS] hc_inject_add_and_grouped_rms_norm rows=" << rows
              << " (parity verified, h_ulp_diffs=" << h_ulp_diffs
              << ", out_ulp_diffs=" << out_ulp_diffs << " / " << total_H << ")\n";
}

void test_hc_inject_add_and_grouped_rms_norm(Context& ctx) {
    run_hc_inject_add_and_grouped_rms_norm_test(ctx, 1);
    run_hc_inject_add_and_grouped_rms_norm_test(ctx, 2);
    run_hc_inject_add_and_grouped_rms_norm_test(ctx, 4);
}

void run_hc_decode_fused_case(Context& ctx, int groups, int group_dim, int lowrank, bool with_inject) {
    const int k_dim = groups * group_dim;
    auto round_bf16 = [](float v) { return static_cast<float>(bf16(v)); };
    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };

    std::mt19937 rng(with_inject ? (789 + groups + lowrank) : (790 + groups + lowrank));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bf16> N_host(k_dim), w_down(static_cast<size_t>(lowrank) * k_dim),
        w_inject(static_cast<size_t>(groups) * k_dim), w_up(static_cast<size_t>(k_dim) * lowrank);
    for (auto& v : N_host) v = bf16(dist(rng));
    for (auto& v : w_down) v = bf16(dist(rng) * 0.02f);
    for (auto& v : w_inject) v = bf16(dist(rng) * 0.02f);
    for (auto& v : w_up) v = bf16(dist(rng) * 0.2f);

    // CPU reference, rounding to bf16 where the PyTorch reference does.
    std::vector<float> ref_act(lowrank), ref_inj(groups), ref_x(group_dim);
    for (int r = 0; r < lowrank; ++r) {
        double dot = 0.0;
        for (int k = 0; k < k_dim; ++k) {
            dot += static_cast<double>(w_down[static_cast<size_t>(r) * k_dim + k]) * static_cast<float>(N_host[k]);
        }
        const float v = round_bf16(static_cast<float>(dot)) / groups;
        ref_act[r] = round_bf16(v * sigmoid(v));
    }
    for (int g = 0; g < groups; ++g) {
        double dot = 0.0;
        for (int k = 0; k < k_dim; ++k) {
            dot += static_cast<double>(w_inject[static_cast<size_t>(g) * k_dim + k]) * static_cast<float>(N_host[k]);
        }
        ref_inj[g] = 2.0f * sigmoid(round_bf16(static_cast<float>(dot)) / groups);
    }
    for (int d = 0; d < group_dim; ++d) {
        float mixed = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const size_t row = static_cast<size_t>(g) * group_dim + d;
            double dot = 0.0;
            for (int k = 0; k < lowrank; ++k) {
                dot += static_cast<double>(w_up[row * lowrank + k]) * ref_act[k];
            }
            mixed += round_bf16(sigmoid(round_bf16(static_cast<float>(dot)))) * static_cast<float>(N_host[row]);
        }
        ref_x[d] = mixed / groups;
    }

    Tensor N_dev = Tensor::allocate(ctx, {k_dim}, dnnl::memory::data_type::bf16);
    Tensor wd_dev = Tensor::allocate(ctx, {lowrank, k_dim}, dnnl::memory::data_type::bf16);
    Tensor wi_dev = Tensor::allocate(ctx, {groups, k_dim}, dnnl::memory::data_type::bf16);
    Tensor wu_dev = Tensor::allocate(ctx, {k_dim, lowrank}, dnnl::memory::data_type::bf16);
    Tensor act_dev = Tensor::allocate(ctx, {lowrank}, dnnl::memory::data_type::bf16);
    Tensor inj_dev = Tensor::allocate(ctx, {groups}, dnnl::memory::data_type::f32);
    Tensor x_dev = Tensor::allocate(ctx, {group_dim}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(N_dev.data(), N_host.data(), N_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(wd_dev.data(), w_down.data(), w_down.size() * sizeof(bf16));
    ctx.memcpy_h2d(wi_dev.data(), w_inject.data(), w_inject.size() * sizeof(bf16));
    ctx.memcpy_h2d(wu_dev.data(), w_up.data(), w_up.size() * sizeof(bf16));
    const std::vector<float> inj_sentinel(groups, -7.0f);
    ctx.memcpy_h2d(inj_dev.data(), inj_sentinel.data(), groups * sizeof(float));

    ops::hc_decode_down_inject(ctx, N_dev.data_as<bf16>(), wd_dev.data_as<bf16>(),
                               with_inject ? wi_dev.data_as<bf16>() : nullptr,
                               act_dev.data_as<bf16>(), inj_dev.data_as<float>(),
                               lowrank, group_dim, groups);
    ops::hc_decode_up_mix(ctx, act_dev.data_as<bf16>(), wu_dev.data_as<bf16>(), N_dev.data_as<bf16>(),
                          x_dev.data_as<bf16>(), lowrank, group_dim, groups);
    ctx.synchronize();

    std::vector<bf16> act(lowrank), x(group_dim);
    std::vector<float> inj(groups);
    ctx.memcpy_d2h(act.data(), act_dev.data(), act.size() * sizeof(bf16));
    ctx.memcpy_d2h(x.data(), x_dev.data(), x.size() * sizeof(bf16));
    ctx.memcpy_d2h(inj.data(), inj_dev.data(), inj.size() * sizeof(float));
    const std::string tag = (with_inject ? " (inject" : " (mixer") +
                            std::string(" g=") + std::to_string(groups) +
                            " d=" + std::to_string(group_dim) +
                            " lr=" + std::to_string(lowrank) + ")";
    check_close(act, ref_act, "hc_decode_down_inject act" + tag, 1e-2f);
    check_close(x, ref_x, "hc_decode_up_mix x" + tag, 1e-2f);
    check_close_f32(inj, with_inject ? ref_inj : inj_sentinel, "hc_decode_down_inject inj" + tag, 1e-2f);
}

void test_hc_decode_fused(Context& ctx, bool with_inject) {
    run_hc_decode_fused_case(ctx, 4, 2560, 320, with_inject);
    if (with_inject) {
        // Non-standard & odd shapes testing subgroup tail loop and scalar unaligned fallback
        run_hc_decode_fused_case(ctx, 2, 256, 45, true);
        run_hc_decode_fused_case(ctx, 3, 31, 35, true);
    }
}

#include "Qwen4ExpHcRowsTests.hpp"

void test_ple_gate_value(Context& ctx) {
    constexpr int rows = 2;
    constexpr int hc_count = 4;
    constexpr int hidden_size = 2560;
    const size_t total_KQ = rows * hc_count * hidden_size;
    const size_t total_V = rows * hidden_size;

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> K_host(total_KQ);
    std::vector<bf16> Q_host(total_KQ);
    std::vector<bf16> V_host(total_V);
    for (size_t i = 0; i < total_KQ; ++i) {
        K_host[i] = bf16(dist(rng));
        Q_host[i] = bf16(dist(rng));
    }
    for (size_t i = 0; i < total_V; ++i) {
        V_host[i] = bf16(dist(rng));
    }

    std::vector<float> ref_GV(total_KQ);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    for (int r = 0; r < rows; ++r) {
        for (int s = 0; s < hc_count; ++s) {
            size_t base = static_cast<size_t>(r) * (hc_count * hidden_size) + s * hidden_size;
            float dot = 0.0f;
            for (int d = 0; d < hidden_size; ++d) {
                dot += static_cast<float>(K_host[base + d]) * static_cast<float>(Q_host[base + d]);
            }
            float g_raw = dot * inv_sqrt_d;
            float sign = (g_raw >= 0.0f ? 1.0f : -1.0f);
            float g_clamped = std::max(std::abs(g_raw), 1.0e-6f);
            float g = sign * std::sqrt(g_clamped);
            float gate = 1.0f / (1.0f + std::exp(-g));
            for (int d = 0; d < hidden_size; ++d) {
                ref_GV[base + d] = gate * static_cast<float>(V_host[r * hidden_size + d]);
            }
        }
    }

    Tensor K_dev = Tensor::allocate(ctx, {rows, hc_count * hidden_size}, dnnl::memory::data_type::bf16);
    Tensor Q_dev = Tensor::allocate(ctx, {rows, hc_count * hidden_size}, dnnl::memory::data_type::bf16);
    Tensor V_dev = Tensor::allocate(ctx, {rows, hidden_size}, dnnl::memory::data_type::bf16);
    Tensor GV_dev = Tensor::allocate(ctx, {rows, hc_count * hidden_size}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(K_dev.data(), K_host.data(), total_KQ * sizeof(bf16));
    ctx.memcpy_h2d(Q_dev.data(), Q_host.data(), total_KQ * sizeof(bf16));
    ctx.memcpy_h2d(V_dev.data(), V_host.data(), total_V * sizeof(bf16));

    ops::ple_gate_value(ctx, static_cast<const bf16*>(K_dev.data()),
                        static_cast<const bf16*>(Q_dev.data()),
                        static_cast<const bf16*>(V_dev.data()),
                        static_cast<bf16*>(GV_dev.data()),
                        rows, hc_count, hidden_size);
    ctx.synchronize();

    std::vector<bf16> gpu_GV(total_KQ);
    ctx.memcpy_d2h(gpu_GV.data(), GV_dev.data(), total_KQ * sizeof(bf16));
    check_close(gpu_GV, ref_GV, "ple_gate_value", 1e-2f);
}

void test_ple_dilated_conv_silu_add(Context& ctx) {
    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    constexpr int dilation = 3;
    constexpr int state_len = 9;

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> conv_w(channels * kernel_size);
    for (size_t i = 0; i < conv_w.size(); ++i) conv_w[i] = bf16(dist(rng) * 0.1f);

    std::vector<float> conv_state_ref(state_len * channels, 0.0f);
    int conv_head_ref = 0;

    Tensor conv_w_dev = Tensor::allocate(ctx, {channels, kernel_size}, dnnl::memory::data_type::bf16);
    Tensor conv_state_dev = Tensor::allocate(ctx, {state_len, channels}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(conv_w_dev.data(), conv_w.data(), conv_w.size() * sizeof(bf16));
    std::vector<float> zeros(state_len * channels, 0.0f);
    ctx.memcpy_h2d(conv_state_dev.data(), zeros.data(), zeros.size() * sizeof(float));
    int conv_head_dev = 0;

    // Test 3 consecutive decode steps (rows = 1)
    for (int step = 0; step < 3; ++step) {
        std::vector<bf16> gv_normed(channels), gv(channels);
        for (int c = 0; c < channels; ++c) {
            gv_normed[c] = bf16(dist(rng));
            gv[c] = bf16(dist(rng));
        }

        // CPU ref
        std::vector<float> ref_out(channels);
        const int r0 = conv_head_ref;
        const int r1 = (conv_head_ref + 3) % state_len;
        const int r2 = (conv_head_ref + 6) % state_len;
        for (int c = 0; c < channels; ++c) {
            float t0 = conv_state_ref[r0 * channels + c];
            float t1 = conv_state_ref[r1 * channels + c];
            float t2 = conv_state_ref[r2 * channels + c];
            float t3 = static_cast<float>(gv_normed[c]);
            const bf16* w = conv_w.data() + c * kernel_size;
            float val = t0 * static_cast<float>(w[0]) + t1 * static_cast<float>(w[1]) +
                        t2 * static_cast<float>(w[2]) + t3 * static_cast<float>(w[3]);
            float silu = val / (1.0f + std::exp(-val));
            ref_out[c] = static_cast<float>(gv[c]) + silu;
            conv_state_ref[conv_head_ref * channels + c] = t3;
        }
        conv_head_ref = (conv_head_ref + 1) % state_len;

        Tensor gv_normed_dev = Tensor::allocate(ctx, {1, channels}, dnnl::memory::data_type::bf16);
        Tensor gv_dev = Tensor::allocate(ctx, {1, channels}, dnnl::memory::data_type::bf16);
        Tensor out_dev = Tensor::allocate(ctx, {1, channels}, dnnl::memory::data_type::bf16);

        ctx.memcpy_h2d(gv_normed_dev.data(), gv_normed.data(), channels * sizeof(bf16));
        ctx.memcpy_h2d(gv_dev.data(), gv.data(), channels * sizeof(bf16));

        ops::ple_dilated_conv_silu_add(
            ctx, static_cast<const bf16*>(gv_normed_dev.data()),
            static_cast<const bf16*>(gv_dev.data()),
            static_cast<float*>(conv_state_dev.data()),
            conv_head_dev,
            static_cast<const bf16*>(conv_w_dev.data()),
            static_cast<bf16*>(out_dev.data()),
            1, channels, kernel_size, dilation);
        ctx.synchronize();

        std::vector<bf16> gpu_out(channels);
        ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), channels * sizeof(bf16));
        check_close(gpu_out, ref_out, "ple_dilated_conv_silu_add decode step " + std::to_string(step), 1e-2f);
        if (conv_head_dev != conv_head_ref) {
            throw std::runtime_error("conv_head mismatch: dev=" + std::to_string(conv_head_dev) +
                                     " ref=" + std::to_string(conv_head_ref));
        }
    }
}

void test_ple_dilated_conv_silu_add_device_head_graph(Context& ctx) {
    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    constexpr int dilation = 3;
    constexpr int state_len = 9;
    constexpr int steps = 11;
    constexpr int initial_head = 5;
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;

    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error("PLE device-head test requires SYCL graph support");
    }

    std::mt19937 rng(8038);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<bf16> conv_w(channels * kernel_size);
    std::vector<float> initial_conv_state(state_len * channels);
    for (auto& value : conv_w) value = bf16(dist(rng) * 0.1f);
    for (auto& value : initial_conv_state) value = dist(rng);

    Tensor conv_w_dev = Tensor::allocate(ctx, {channels, kernel_size}, dnnl::memory::data_type::bf16);
    Tensor state_host_dev = Tensor::allocate(ctx, {state_len, channels}, dnnl::memory::data_type::f32);
    Tensor state_graph_dev = Tensor::allocate(ctx, {state_len, channels}, dnnl::memory::data_type::f32);
    Tensor head_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
    Tensor gv_normed_dev = Tensor::allocate(ctx, {1, channels}, dnnl::memory::data_type::bf16);
    Tensor gv_dev = Tensor::allocate(ctx, {1, channels}, dnnl::memory::data_type::bf16);
    Tensor out_host_dev = Tensor::allocate(ctx, {1, channels}, dnnl::memory::data_type::bf16);
    Tensor out_graph_dev = Tensor::allocate(ctx, {1, channels}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(conv_w_dev.data(), conv_w.data(), conv_w.size() * sizeof(bf16));
    ctx.memcpy_h2d(state_host_dev.data(), initial_conv_state.data(),
                   initial_conv_state.size() * sizeof(float));
    ctx.memcpy_h2d(state_graph_dev.data(), initial_conv_state.data(),
                   initial_conv_state.size() * sizeof(float));
    ctx.memcpy_h2d(head_dev.data(), &initial_head, sizeof(int));

    ExecutableGraph graph = [&]() {
        ex::command_graph<ex::graph_state::modifiable> recording(queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::ple_dilated_conv_silu_add_device_head(
                ctx, gv_normed_dev.data_as<bf16>(), gv_dev.data_as<bf16>(),
                state_graph_dev.data_as<float>(), head_dev.data_as<int>(),
                conv_w_dev.data_as<bf16>(), out_graph_dev.data_as<bf16>(),
                1, channels, kernel_size, dilation);
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    }();

    int host_head = initial_head;
    std::vector<bf16> host_out(channels);
    std::vector<bf16> graph_out(channels);
    std::vector<float> host_state(state_len * channels);
    std::vector<float> graph_state(state_len * channels);
    int graph_head = -1;

    for (int step = 0; step < steps; ++step) {
        std::vector<bf16> gv_normed(channels);
        std::vector<bf16> gv(channels);
        for (int c = 0; c < channels; ++c) {
            gv_normed[c] = bf16(dist(rng));
            gv[c] = bf16(dist(rng));
        }
        ctx.memcpy_h2d(gv_normed_dev.data(), gv_normed.data(), channels * sizeof(bf16));
        ctx.memcpy_h2d(gv_dev.data(), gv.data(), channels * sizeof(bf16));

        ops::ple_dilated_conv_silu_add(
            ctx, gv_normed_dev.data_as<bf16>(), gv_dev.data_as<bf16>(),
            state_host_dev.data_as<float>(), host_head,
            conv_w_dev.data_as<bf16>(), out_host_dev.data_as<bf16>(),
            1, channels, kernel_size, dilation);
        queue.ext_oneapi_graph(graph);
        ctx.synchronize();

        ctx.memcpy_d2h(host_out.data(), out_host_dev.data(), channels * sizeof(bf16));
        ctx.memcpy_d2h(graph_out.data(), out_graph_dev.data(), channels * sizeof(bf16));
        ctx.memcpy_d2h(host_state.data(), state_host_dev.data(), host_state.size() * sizeof(float));
        ctx.memcpy_d2h(graph_state.data(), state_graph_dev.data(), graph_state.size() * sizeof(float));
        ctx.memcpy_d2h(&graph_head, head_dev.data(), sizeof(int));

        if (std::memcmp(host_out.data(), graph_out.data(), channels * sizeof(bf16)) != 0 ||
            std::memcmp(host_state.data(), graph_state.data(),
                        host_state.size() * sizeof(float)) != 0 ||
            graph_head != host_head) {
            throw std::runtime_error(
                "PLE graph device-head mismatch at replay " + std::to_string(step) +
                ": host_head=" + std::to_string(host_head) +
                " graph_head=" + std::to_string(graph_head));
        }
    }

    std::cout << "[PASS] PLE graph device conv head (replays=" << steps
              << ", ring wrap, output/state/head bitwise)\n";
}

void test_qsa_graph_position_and_block_advance(Context& ctx) {
    constexpr int capacity = 2060;
    constexpr int head_dim = 128;
    constexpr int total_dim = 640;
    constexpr int max_blocks = 516;
    constexpr int initial_pos = 2051;
    constexpr int initial_blocks = (initial_pos + 1) / 4;
    constexpr int steps = 5;
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;
    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error("QSA graph test requires SYCL graph support");
    }

    std::vector<bf16> initial_k(capacity * head_dim);
    std::vector<bf16> q_norm(head_dim);
    std::vector<bf16> k_norm(head_dim);
    for (int i = 0; i < capacity * head_dim; ++i) {
        initial_k[i] = bf16(float((i * 13 + 7) % 127 - 63) * 0.002f);
    }
    for (int d = 0; d < head_dim; ++d) {
        q_norm[d] = bf16(0.9f + float(d % 19) * 0.01f);
        k_norm[d] = bf16(0.95f + float((d * 3) % 17) * 0.01f);
    }

    Tensor qk_host_dev = Tensor::allocate(
        ctx, {1, total_dim}, dnnl::memory::data_type::bf16);
    Tensor qk_graph_dev = Tensor::allocate(
        ctx, {1, total_dim}, dnnl::memory::data_type::bf16);
    Tensor q_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_host_dev = Tensor::allocate(
        ctx, {capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_graph_dev = Tensor::allocate(
        ctx, {capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_host_dev = Tensor::allocate(
        ctx, {4, head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_graph_dev = Tensor::allocate(
        ctx, {4, head_dim}, dnnl::memory::data_type::bf16);
    Tensor compressed_host_dev = Tensor::allocate(
        ctx, {max_blocks, head_dim}, dnnl::memory::data_type::bf16);
    Tensor compressed_graph_dev = Tensor::allocate(
        ctx, {max_blocks, head_dim}, dnnl::memory::data_type::bf16);
    Tensor control_dev = Tensor::allocate(
        ctx, {ops::graph_control::words}, dnnl::memory::data_type::s32);
    Tensor scores_host_dev = Tensor::allocate(
        ctx, {max_blocks}, dnnl::memory::data_type::f32);
    Tensor scores_graph_dev = Tensor::allocate(
        ctx, {max_blocks}, dnnl::memory::data_type::f32);
    Tensor active_host_dev = Tensor::allocate(
        ctx, {512 + 4}, dnnl::memory::data_type::s32);
    Tensor active_graph_dev = Tensor::allocate(
        ctx, {512 + 4}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(q_norm_dev.data(), q_norm.data(), q_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_norm_dev.data(), k_norm.data(), k_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_host_dev.data(), initial_k.data(), initial_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_graph_dev.data(), initial_k.data(), initial_k.size() * sizeof(bf16));
    queue.memset(compressed_host_dev.data(), 0, compressed_host_dev.size_bytes());
    queue.memset(compressed_graph_dev.data(), 0, compressed_graph_dev.size_bytes());
    ops::qsa_compress_key_blocks(
        ctx, k_host_dev, k_norm_dev, compressed_host_dev,
        0, initial_blocks, 10000.0f);
    ctx.synchronize();
    queue.memcpy(compressed_graph_dev.data(), compressed_host_dev.data(),
                 size_t(initial_blocks) * head_dim * sizeof(bf16));
    int graph_blocks = initial_blocks;
    int position = initial_pos;
    std::array<int, ops::graph_control::words> control{};
    control[ops::graph_control::start_pos] = initial_pos;
    control[ops::graph_control::compressed_blocks] = initial_blocks;
    control[3] = 0; // test-only invalid marker
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    ctx.synchronize();

    Tensor narrow_k_cache = Tensor::allocate(
        ctx, {capacity, head_dim - 64}, dnnl::memory::data_type::bf16);
    Tensor narrow_compressed_k = Tensor::allocate(
        ctx, {max_blocks, head_dim - 64}, dnnl::memory::data_type::bf16);
    expect_invalid_argument("QSA prepare graph cache capacity", [&] {
        ops::qsa_prepare_q_and_cache_k_graph(
            ctx, qk_graph_dev, q_norm_dev, narrow_k_cache, q_graph_dev,
            control_dev.data_as<int>(), 10000.0f);
    });
    expect_invalid_argument("QSA compress graph compressed capacity", [&] {
        ops::qsa_compress_one_completed_block_graph(
            ctx, k_graph_dev, k_norm_dev, narrow_compressed_k,
            control_dev.data_as<int>(), 10000.0f);
    });

    ExecutableGraph graph = [&]() {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::qsa_prepare_q_and_cache_k_graph(
                ctx, qk_graph_dev, q_norm_dev, k_graph_dev, q_graph_dev,
                control_dev.data_as<int>(), 10000.0f);
            ops::qsa_compress_one_completed_block_graph(
                ctx, k_graph_dev, k_norm_dev, compressed_graph_dev,
                control_dev.data_as<int>(), 10000.0f);
            ops::qsa_compute_block_scores_compressed_graph(
                ctx, q_graph_dev, compressed_graph_dev, scores_graph_dev,
                control_dev.data_as<int>(), max_blocks);
            ops::qsa_select_topk_blocks_graph(
                ctx, scores_graph_dev, active_graph_dev,
                control_dev.data_as<int>(), 512, 4, max_blocks);
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    }();
    ExecutableGraph scoring_guard_graph = [&]() {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::qsa_compute_block_scores_compressed_graph(
                ctx, q_graph_dev, compressed_graph_dev, scores_graph_dev,
                control_dev.data_as<int>(), max_blocks);
            ops::qsa_select_topk_blocks_graph(
                ctx, scores_graph_dev, active_graph_dev,
                control_dev.data_as<int>(), 512, 4, max_blocks);
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    }();

    int host_blocks = initial_blocks;
    for (int step = 0; step < steps; ++step) {
        position = initial_pos + step;
        std::vector<bf16> qk(total_dim);
        for (int i = 0; i < total_dim; ++i) {
            qk[i] = bf16(float(((i + 3) * 17 + position * 11) % 251 - 125) *
                         0.006f);
        }
        ctx.memcpy_h2d(qk_host_dev.data(), qk.data(), qk.size() * sizeof(bf16));
        ctx.memcpy_h2d(qk_graph_dev.data(), qk.data(), qk.size() * sizeof(bf16));
        control[ops::graph_control::start_pos] = position;
        control[ops::graph_control::compressed_blocks] = host_blocks;
        control[3] = 0;
        ctx.memcpy_h2d(control_dev.data(), control.data(),
                       control.size() * sizeof(int));

        ops::qsa_prepare_q_and_cache_k(
            ctx, qk_host_dev, q_norm_dev, k_host_dev, q_host_dev,
            position, 10000.0f);
        const int target_blocks = (position + 1) / 4;
        if (target_blocks > host_blocks) {
            if (target_blocks != host_blocks + 1) {
                throw std::runtime_error("QSA direct test skipped more than one block");
            }
            ops::qsa_compress_key_blocks(
                ctx, k_host_dev, k_norm_dev, compressed_host_dev,
                host_blocks, 1, 10000.0f);
            host_blocks = target_blocks;
        }

        auto event = queue.ext_oneapi_graph(graph);
        event.wait_and_throw();
        ctx.synchronize();
        ctx.memcpy_d2h(control.data(), control_dev.data(),
                       control.size() * sizeof(int));
        graph_blocks = control[ops::graph_control::compressed_blocks];
        if (control[3] != 0) {
            throw std::runtime_error("QSA graph reported an invalid block gap");
        }
        if (graph_blocks != host_blocks) {
            throw std::runtime_error(
                "QSA graph block count mismatch at pos=" +
                std::to_string(position) + ": graph=" +
                std::to_string(graph_blocks) + " host=" +
                std::to_string(host_blocks));
        }

        std::vector<bf16> q_host(4 * head_dim), q_graph(4 * head_dim);
        std::vector<bf16> k_host(capacity * head_dim);
        std::vector<bf16> k_graph(capacity * head_dim);
        std::vector<bf16> comp_host(max_blocks * head_dim);
        std::vector<bf16> comp_graph(max_blocks * head_dim);
        ctx.memcpy_d2h(q_host.data(), q_host_dev.data(), q_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(q_graph.data(), q_graph_dev.data(), q_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_host.data(), k_host_dev.data(), k_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_graph.data(), k_graph_dev.data(), k_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(comp_host.data(), compressed_host_dev.data(),
                       comp_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(comp_graph.data(), compressed_graph_dev.data(),
                       comp_graph.size() * sizeof(bf16));
        if (std::memcmp(q_host.data(), q_graph.data(), q_host.size() * sizeof(bf16)) != 0 ||
            std::memcmp(k_host.data(), k_graph.data(), k_host.size() * sizeof(bf16)) != 0 ||
            std::memcmp(comp_host.data(), comp_graph.data(),
                        comp_host.size() * sizeof(bf16)) != 0) {
            throw std::runtime_error(
                "QSA graph position/block output mismatch at pos=" +
                std::to_string(position));
        }

        ops::qsa_compute_block_scores_compressed(
            ctx, q_host_dev, compressed_host_dev, scores_host_dev,
            host_blocks);
        const int active_count = ops::qsa_select_topk_blocks(
            ctx, scores_host_dev, host_blocks, 512, position + 1, 4,
            active_host_dev);
        ctx.synchronize();
        const int selected_blocks = std::min(host_blocks, 512);
        const int tail_len = position + 1 - host_blocks * 4;
        if (active_count != selected_blocks || tail_len < 0 || tail_len > 4) {
            throw std::runtime_error("QSA host selector geometry mismatch");
        }
        std::vector<float> score_host(host_blocks), score_graph(host_blocks);
        std::vector<int> active_host(selected_blocks + tail_len);
        std::vector<int> active_graph(selected_blocks + tail_len);
        ctx.memcpy_d2h(score_host.data(), scores_host_dev.data(),
                       score_host.size() * sizeof(float));
        ctx.memcpy_d2h(score_graph.data(), scores_graph_dev.data(),
                       score_graph.size() * sizeof(float));
        ctx.memcpy_d2h(active_host.data(), active_host_dev.data(),
                       active_host.size() * sizeof(int));
        ctx.memcpy_d2h(active_graph.data(), active_graph_dev.data(),
                       active_graph.size() * sizeof(int));
        if (std::memcmp(score_host.data(), score_graph.data(),
                        score_host.size() * sizeof(float)) != 0 ||
            std::memcmp(active_host.data(), active_graph.data(),
                        active_host.size() * sizeof(int)) != 0) {
            throw std::runtime_error(
                "QSA graph scoring/selector mismatch at pos=" +
                std::to_string(position));
        }
    }

    if (graph_blocks != 514 || host_blocks != 514) {
        throw std::runtime_error("QSA graph test did not reach the 514-block boundary");
    }

    std::vector<float> scores_before(max_blocks);
    std::vector<int> active_before(512 + 4);
    std::vector<bf16> q_before(4 * head_dim);
    std::vector<bf16> k_row_before(head_dim);
    ctx.memcpy_d2h(scores_before.data(), scores_graph_dev.data(),
                   scores_before.size() * sizeof(float));
    ctx.memcpy_d2h(active_before.data(), active_graph_dev.data(),
                   active_before.size() * sizeof(int));
    ctx.memcpy_d2h(q_before.data(), q_graph_dev.data(),
                   q_before.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_row_before.data(),
                   k_graph_dev.data_as<bf16>() +
                       size_t(initial_pos + steps - 1) * head_dim,
                   k_row_before.size() * sizeof(bf16));

    std::vector<bf16> compressed_guard(head_dim, bf16(1.2345f));
    ctx.memcpy_h2d(compressed_graph_dev.data_as<bf16>() +
                       size_t(initial_blocks) * head_dim,
                   compressed_guard.data(),
                   compressed_guard.size() * sizeof(bf16));
    control[ops::graph_control::start_pos] = initial_pos + steps - 1;
    control[ops::graph_control::compressed_blocks] = initial_blocks;
    control[ops::graph_control::abort] = 1;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto abort_event = queue.ext_oneapi_graph(graph);
    abort_event.wait_and_throw();
    ctx.synchronize();
    ctx.memcpy_d2h(control.data(), control_dev.data(),
                   control.size() * sizeof(int));
    std::vector<bf16> q_abort(4 * head_dim);
    std::vector<bf16> k_row_abort(head_dim);
    std::vector<bf16> compressed_abort(head_dim);
    ctx.memcpy_d2h(q_abort.data(), q_graph_dev.data(),
                   q_abort.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_row_abort.data(),
                   k_graph_dev.data_as<bf16>() +
                       size_t(initial_pos + steps - 1) * head_dim,
                   k_row_abort.size() * sizeof(bf16));
    ctx.memcpy_d2h(compressed_abort.data(),
                   compressed_graph_dev.data_as<bf16>() +
                       size_t(initial_blocks) * head_dim,
                   compressed_abort.size() * sizeof(bf16));
    if (control[ops::graph_control::abort] == 0 ||
        control[ops::graph_control::compressed_blocks] != initial_blocks ||
        std::memcmp(q_before.data(), q_abort.data(),
                    q_before.size() * sizeof(bf16)) != 0 ||
        std::memcmp(k_row_before.data(), k_row_abort.data(),
                    k_row_before.size() * sizeof(bf16)) != 0 ||
        std::memcmp(compressed_guard.data(), compressed_abort.data(),
                    compressed_guard.size() * sizeof(bf16)) != 0) {
        throw std::runtime_error(
            "QSA compression abort-only modified output/cache/count");
    }

    control[ops::graph_control::start_pos] = initial_pos + steps - 1;
    control[ops::graph_control::compressed_blocks] = 512;
    control[ops::graph_control::abort] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto scoring_guard_event = queue.ext_oneapi_graph(scoring_guard_graph);
    scoring_guard_event.wait_and_throw();
    ctx.synchronize();
    ctx.memcpy_d2h(control.data(), control_dev.data(),
                   control.size() * sizeof(int));
    std::vector<float> scores_guard_abort(max_blocks);
    std::vector<int> active_guard_abort(512 + 4);
    ctx.memcpy_d2h(scores_guard_abort.data(), scores_graph_dev.data(),
                   scores_guard_abort.size() * sizeof(float));
    ctx.memcpy_d2h(active_guard_abort.data(), active_graph_dev.data(),
                   active_guard_abort.size() * sizeof(int));
    if (control[ops::graph_control::abort] == 0 ||
        control[ops::graph_control::compressed_blocks] != 512 ||
        std::memcmp(scores_before.data(), scores_guard_abort.data(),
                    scores_before.size() * sizeof(float)) != 0 ||
        std::memcmp(active_before.data(), active_guard_abort.data(),
                    active_before.size() * sizeof(int)) != 0) {
        throw std::runtime_error(
            "QSA scoring graph accepted an inconsistent block count");
    }

    std::vector<bf16> poisoned_qk(total_dim);
    for (int i = 0; i < total_dim; ++i) {
        poisoned_qk[i] = bf16(float((i * 7) % 61 - 30) * 0.01f);
    }
    ctx.memcpy_h2d(qk_graph_dev.data(), poisoned_qk.data(),
                   poisoned_qk.size() * sizeof(bf16));
    control[ops::graph_control::start_pos] = initial_pos + steps - 1;
    control[ops::graph_control::compressed_blocks] = 512;
    control[ops::graph_control::abort] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto invalid_event = queue.ext_oneapi_graph(graph);
    invalid_event.wait_and_throw();
    ctx.synchronize();
    ctx.memcpy_d2h(control.data(), control_dev.data(),
                   control.size() * sizeof(int));
    std::vector<float> scores_after(max_blocks);
    std::vector<int> active_after(512 + 4);
    std::vector<bf16> q_after(4 * head_dim);
    std::vector<bf16> k_row_after(head_dim);
    ctx.memcpy_d2h(scores_after.data(), scores_graph_dev.data(),
                   scores_after.size() * sizeof(float));
    ctx.memcpy_d2h(active_after.data(), active_graph_dev.data(),
                   active_after.size() * sizeof(int));
    ctx.memcpy_d2h(q_after.data(), q_graph_dev.data(),
                   q_after.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_row_after.data(),
                   k_graph_dev.data_as<bf16>() +
                       size_t(initial_pos + steps - 1) * head_dim,
                   k_row_after.size() * sizeof(bf16));
    if (control[ops::graph_control::abort] == 0 ||
        control[ops::graph_control::compressed_blocks] != 512 ||
        std::memcmp(q_before.data(), q_after.data(),
                    q_before.size() * sizeof(bf16)) != 0 ||
        std::memcmp(k_row_before.data(), k_row_after.data(),
                    k_row_before.size() * sizeof(bf16)) != 0 ||
        std::memcmp(scores_before.data(), scores_after.data(),
                    scores_before.size() * sizeof(float)) != 0 ||
        std::memcmp(active_before.data(), active_after.data(),
                    active_before.size() * sizeof(int)) != 0) {
        throw std::runtime_error(
            "QSA graph accepted an inconsistent compressed block count");
    }
    std::cout << "[PASS] QSA graph device position + block advance + scoring/selector"
              << " (pos=" << initial_pos << ".." << initial_pos + steps - 1
              << ", blocks=513..514, scores/active-blocks bitwise,"
              << " abort-only + stale-count no-op)\n";
}

void test_qsa_graph_per_layer_block_advance(Context& ctx) {
    constexpr int capacity = 2060;
    constexpr int head_dim = 128;
    constexpr int total_dim = 640;
    constexpr int max_blocks = 516;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int first_pos = 2054;
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;
    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error(
            "QSA per-layer graph test requires SYCL graph support");
    }

    const int layer1_slot = ops::graph_control_layer_blocks_slot(1);
    if (layer1_slot <= ops::graph_control::abort ||
        layer1_slot == ops::graph_control::compressed_blocks ||
        layer1_slot >= ops::graph_control::words) {
        throw std::runtime_error("QSA per-layer slot mapping is invalid");
    }

    std::vector<bf16> initial_k(capacity * head_dim);
    std::vector<bf16> q_norm(head_dim);
    std::vector<bf16> k_norm(head_dim);
    for (int i = 0; i < capacity * head_dim; ++i) {
        initial_k[i] =
            bf16(float((i * 5 + 3) % 101 - 50) * 0.001f);
    }
    for (int d = 0; d < head_dim; ++d) {
        q_norm[d] = bf16(0.9f + float(d % 17) * 0.01f);
        k_norm[d] = bf16(0.95f + float((d * 7) % 19) * 0.01f);
    }

    Tensor qk0_dev = Tensor::allocate(
        ctx, {1, total_dim}, dnnl::memory::data_type::bf16);
    Tensor qk1_dev = Tensor::allocate(
        ctx, {1, total_dim}, dnnl::memory::data_type::bf16);
    Tensor q_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor k0_dev = Tensor::allocate(
        ctx, {capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor k1_dev = Tensor::allocate(
        ctx, {capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor q0_dev = Tensor::allocate(
        ctx, {4, head_dim}, dnnl::memory::data_type::bf16);
    Tensor q1_dev = Tensor::allocate(
        ctx, {4, head_dim}, dnnl::memory::data_type::bf16);
    Tensor compressed0_dev = Tensor::allocate(
        ctx, {max_blocks, head_dim}, dnnl::memory::data_type::bf16);
    Tensor compressed1_dev = Tensor::allocate(
        ctx, {max_blocks, head_dim}, dnnl::memory::data_type::bf16);
    Tensor scores0_dev = Tensor::allocate(
        ctx, {max_blocks}, dnnl::memory::data_type::f32);
    Tensor scores1_dev = Tensor::allocate(
        ctx, {max_blocks}, dnnl::memory::data_type::f32);
    Tensor active0_dev = Tensor::allocate(
        ctx, {block_topk + compress_ratio}, dnnl::memory::data_type::s32);
    Tensor active1_dev = Tensor::allocate(
        ctx, {block_topk + compress_ratio}, dnnl::memory::data_type::s32);
    Tensor control_dev = Tensor::allocate(
        ctx, {ops::graph_control::words}, dnnl::memory::data_type::s32);
    expect_invalid_argument("QSA per-layer slot range", [&] {
        ops::qsa_compute_block_scores_compressed_graph(
            ctx, q0_dev, compressed0_dev, scores0_dev,
            control_dev.data_as<int>(), max_blocks,
            ops::graph_control::max_full_layers);
    });

    std::vector<bf16> compressed0(max_blocks * head_dim, bf16(0.25f));
    std::vector<bf16> compressed1(max_blocks * head_dim, bf16(-0.25f));
    const bf16 block512_guard0 = bf16(20.0f);
    const bf16 block512_guard1 = bf16(-20.0f);
    const bf16 block513_guard0 = bf16(21.0f);
    const bf16 block513_guard1 = bf16(-21.0f);
    std::fill_n(
        compressed0.data() + size_t(512) * head_dim, head_dim,
        block512_guard0);
    std::fill_n(
        compressed1.data() + size_t(512) * head_dim, head_dim,
        block512_guard1);
    std::fill_n(
        compressed0.data() + size_t(513) * head_dim, head_dim,
        block513_guard0);
    std::fill_n(
        compressed1.data() + size_t(513) * head_dim, head_dim,
        block513_guard1);

    ctx.memcpy_h2d(q_norm_dev.data(), q_norm.data(),
                   q_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_norm_dev.data(), k_norm.data(),
                   k_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(k0_dev.data(), initial_k.data(),
                   initial_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(k1_dev.data(), initial_k.data(),
                   initial_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(compressed0_dev.data(), compressed0.data(),
                   compressed0.size() * sizeof(bf16));
    ctx.memcpy_h2d(compressed1_dev.data(), compressed1.data(),
                   compressed1.size() * sizeof(bf16));

    std::array<int, ops::graph_control::words> control{};
    control[ops::graph_control::start_pos] = first_pos;
    control[ops::graph_control::compressed_blocks] = 513;
    control[layer1_slot] = 512;
    control[ops::graph_control::abort] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    ctx.synchronize();

    auto record_lane = [&](int layer, Tensor& qk_dev, Tensor& k_dev,
                           Tensor& q_dev, Tensor& compressed_dev,
                           Tensor& scores_dev, Tensor& active_dev) {
        ops::qsa_prepare_q_and_cache_k_graph(
            ctx, qk_dev, q_norm_dev, k_dev, q_dev,
            control_dev.data_as<int>(), 10000.0f, false, nullptr, nullptr,
            nullptr, 0, 0, 0, 0, 0, layer);
        ops::qsa_compress_one_completed_block_graph(
            ctx, k_dev, k_norm_dev, compressed_dev,
            control_dev.data_as<int>(), 10000.0f, false, nullptr, nullptr,
            nullptr, 0, 0, 0, 0, 0, layer);
        ops::qsa_compute_block_scores_compressed_graph(
            ctx, q_dev, compressed_dev, scores_dev,
            control_dev.data_as<int>(), max_blocks, layer);
        ops::qsa_select_topk_blocks_graph(
            ctx, scores_dev, active_dev, control_dev.data_as<int>(),
            block_topk, compress_ratio, max_blocks, layer);
    };

    ExecutableGraph graph = [&]() {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            record_lane(
                0, qk0_dev, k0_dev, q0_dev, compressed0_dev,
                scores0_dev, active0_dev);
            record_lane(
                1, qk1_dev, k1_dev, q1_dev, compressed1_dev,
                scores1_dev, active1_dev);
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    }();

    auto read_block = [&](Tensor& compressed_dev, int block) {
        std::vector<bf16> row(head_dim);
        ctx.memcpy_d2h(
            row.data(),
            compressed_dev.data_as<bf16>() +
                static_cast<size_t>(block) * head_dim,
            row.size() * sizeof(bf16));
        return row;
    };
    auto all_equal = [](const std::vector<bf16>& row, bf16 expected) {
        for (bf16 value : row) {
            if (static_cast<float>(value) != static_cast<float>(expected)) {
                return false;
            }
        }
        return true;
    };

    for (int step = 0; step < 2; ++step) {
        const int position = first_pos + step;
        std::vector<bf16> qk0(total_dim);
        std::vector<bf16> qk1(total_dim);
        for (int i = 0; i < total_dim; ++i) {
            qk0[i] = bf16(
                float(((i + 7) * 17 + position * 11) % 251 - 125) *
                0.004f);
            qk1[i] = bf16(
                float(((i + 29) * 13 + position * 19) % 257 - 128) *
                0.003f);
        }
        ctx.memcpy_h2d(qk0_dev.data(), qk0.data(),
                       qk0.size() * sizeof(bf16));
        ctx.memcpy_h2d(qk1_dev.data(), qk1.data(),
                       qk1.size() * sizeof(bf16));
        control[ops::graph_control::start_pos] = position;
        control[ops::graph_control::abort] = 0;
        ctx.memcpy_h2d(control_dev.data(), control.data(),
                       control.size() * sizeof(int));

        auto event = queue.ext_oneapi_graph(graph);
        event.wait_and_throw();
        ctx.synchronize();
        ctx.memcpy_d2h(control.data(), control_dev.data(),
                       control.size() * sizeof(int));
        if (control[ops::graph_control::abort] != 0) {
            throw std::runtime_error(
                "QSA per-layer graph aborted at pos=" +
                std::to_string(position));
        }
        if (control[ops::graph_control::compressed_blocks] !=
                513 + step ||
            control[layer1_slot] != 512 + step + 1) {
            throw std::runtime_error(
                "QSA per-layer counters overlapped at pos=" +
                std::to_string(position));
        }

        const std::vector<bf16> block512_l0 =
            read_block(compressed0_dev, 512);
        const std::vector<bf16> block512_l1 =
            read_block(compressed1_dev, 512);
        if (!all_equal(block512_l0, block512_guard0) ||
            all_equal(block512_l1, block512_guard1)) {
            throw std::runtime_error(
                "QSA per-layer block 512 state crossed lanes");
        }
    }

    const std::vector<bf16> block513_l0 = read_block(compressed0_dev, 513);
    const std::vector<bf16> block513_l1 = read_block(compressed1_dev, 513);
    if (all_equal(block513_l0, block513_guard0) ||
        all_equal(block513_l1, block513_guard1)) {
        throw std::runtime_error(
            "QSA per-layer layers did not both advance block 513");
    }

    std::cout << "[PASS] QSA per-layer graph counters"
              << " (pos=2054..2055, layer0=513..514, layer1=512..514,"
              << " block output isolated)\n";
}

void test_attention_decode_block_native_graph(Context& ctx) {
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int capacity = 2060;
    constexpr int max_tiles = 17;
    constexpr int partial_stride = 272;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;
    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error("attention graph test requires SYCL graph support");
    }

    std::mt19937 rng(911);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<bf16> q_host(num_heads * head_dim);
    std::vector<bf16> k_host(num_kv_heads * capacity * head_dim);
    std::vector<bf16> v_host(k_host.size());
    for (auto& value : q_host) value = bf16(dist(rng));
    for (auto& value : k_host) value = bf16(dist(rng));
    for (auto& value : v_host) value = bf16(dist(rng));

    Tensor q_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor out_host_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor out_graph_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor out_cached_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(
        ctx, {num_heads, capacity}, dnnl::memory::data_type::f32);
    Tensor partials_host_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride}, dnnl::memory::data_type::f32);
    Tensor partials_graph_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride}, dnnl::memory::data_type::f32);
    Tensor partials_cached_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride}, dnnl::memory::data_type::f32);
    Tensor active_blocks_dev = Tensor::allocate(
        ctx, {block_topk + compress_ratio}, dnnl::memory::data_type::s32);
    Tensor control_dev = Tensor::allocate(
        ctx, {ops::graph_control::words}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(q_dev.data(), q_host.data(), q_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_dev.data(), k_host.data(), k_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), v_host.data(), v_host.size() * sizeof(bf16));

    auto capture_graph = [&](bool cache_active_tokens) -> ExecutableGraph {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::attention_validate_qsa_active_blocks_graph(
                ctx, k_dev, v_dev, active_blocks_dev,
                control_dev.data_as<int>(), num_kv_heads);
            if (cache_active_tokens) {
                ops::attention_decode_exact_head256_block_native_graph_token_cache(
                    ctx, q_dev, k_dev, v_dev, out_cached_dev,
                    partials_cached_dev, active_blocks_dev,
                    control_dev.data_as<int>(), num_heads, num_kv_heads);
            } else {
                ops::attention_decode_exact_head256_block_native_graph(
                    ctx, q_dev, k_dev, v_dev, out_graph_dev,
                    partials_graph_dev, active_blocks_dev,
                    control_dev.data_as<int>(), num_heads, num_kv_heads);
            }
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    };
    ExecutableGraph graph = capture_graph(false);
    ExecutableGraph graph_cached = capture_graph(true);

    std::array<int, ops::graph_control::words> control{};
    control[ops::graph_control::abort] = 1;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto warm_old = queue.ext_oneapi_graph(graph);
    auto warm_cached = queue.ext_oneapi_graph(graph_cached);
    warm_old.wait_and_throw();
    warm_cached.wait_and_throw();
    ctx.synchronize();

    double old_replay_wall_ms = 0.0;
    double cached_replay_wall_ms = 0.0;
    auto timed_replay = [&](auto& executable) {
        const auto start = std::chrono::steady_clock::now();
        auto event = queue.ext_oneapi_graph(executable);
        event.wait_and_throw();
        ctx.synchronize();
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
            .count();
    };
    for (int step = 0; step < 4; ++step) {
        const int pos = 2052 + step;
        const int visible_len = pos + 1;
        const int complete_blocks = visible_len / compress_ratio;
        const int tail_len = visible_len % compress_ratio;
        std::array<int, block_topk + compress_ratio> active{};
        for (int b = 0; b < block_topk; ++b) active[b] = b;
        for (int t = 0; t < tail_len; ++t) {
            active[block_topk + t] =
                complete_blocks * compress_ratio + t;
        }
        ctx.memcpy_h2d(active_blocks_dev.data(), active.data(),
                       active.size() * sizeof(int));
        control[ops::graph_control::start_pos] = pos;
        control[ops::graph_control::compressed_blocks] = complete_blocks;
        control[ops::graph_control::abort] = 0;
        ctx.memcpy_h2d(control_dev.data(), control.data(),
                       control.size() * sizeof(int));

        ops::attention_decode(
            ctx, q_dev, k_dev, v_dev, out_host_dev, scores_dev,
            num_heads, num_kv_heads, head_dim, visible_len,
            &partials_host_dev, nullptr, nullptr, 0,
            active_blocks_dev.data_as<int>(), block_topk);
        ctx.synchronize();
        old_replay_wall_ms += timed_replay(graph);
        cached_replay_wall_ms += timed_replay(graph_cached);

        std::vector<bf16> out_host(num_heads * head_dim);
        std::vector<bf16> out_graph(num_heads * head_dim);
        std::vector<bf16> out_cached(num_heads * head_dim);
        std::vector<float> partial_host(
            size_t(num_heads) * max_tiles * partial_stride);
        std::vector<float> partial_graph(partial_host.size());
        std::vector<float> partial_cached(partial_host.size());
        ctx.memcpy_d2h(out_host.data(), out_host_dev.data(),
                       out_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(out_graph.data(), out_graph_dev.data(),
                       out_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(out_cached.data(), out_cached_dev.data(),
                       out_cached.size() * sizeof(bf16));
        ctx.memcpy_d2h(partial_host.data(), partials_host_dev.data(),
                       partial_host.size() * sizeof(float));
        ctx.memcpy_d2h(partial_graph.data(), partials_graph_dev.data(),
                       partial_graph.size() * sizeof(float));
        ctx.memcpy_d2h(partial_cached.data(), partials_cached_dev.data(),
                       partial_cached.size() * sizeof(float));
        if (std::memcmp(out_host.data(), out_graph.data(),
                        out_host.size() * sizeof(bf16)) != 0 ||
            std::memcmp(partial_host.data(), partial_graph.data(),
                        partial_host.size() * sizeof(float)) != 0 ||
            std::memcmp(out_host.data(), out_cached.data(),
                        out_host.size() * sizeof(bf16)) != 0 ||
            std::memcmp(partial_host.data(), partial_cached.data(),
                        partial_host.size() * sizeof(float)) != 0) {
            throw std::runtime_error(
                "block-native attention graph mismatch at start_pos=" +
                std::to_string(pos));
        }
    }
    std::cout << "[ATTENTION GRAPH A/B] start_pos=2052..2055"
              << " old_replay_wall_ms=" << old_replay_wall_ms / 4.0
              << " cached_replay_wall_ms=" << cached_replay_wall_ms / 4.0
              << "\n";

    std::vector<uint8_t> output_before(
        size_t(num_heads) * head_dim * sizeof(bf16));
    std::vector<uint8_t> partials_before(
        size_t(num_heads) * max_tiles * partial_stride * sizeof(float));
    std::vector<uint8_t> cached_output_before(output_before.size());
    std::vector<uint8_t> cached_partials_before(partials_before.size());
    ctx.memcpy_d2h(output_before.data(), out_graph_dev.data(),
                   output_before.size());
    ctx.memcpy_d2h(partials_before.data(), partials_graph_dev.data(),
                   partials_before.size());
    ctx.memcpy_d2h(cached_output_before.data(), out_cached_dev.data(),
                   cached_output_before.size());
    ctx.memcpy_d2h(cached_partials_before.data(), partials_cached_dev.data(),
                   cached_partials_before.size());
    control[ops::graph_control::abort] = 1;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto abort_event = queue.ext_oneapi_graph(graph);
    auto cached_abort_event = queue.ext_oneapi_graph(graph_cached);
    abort_event.wait_and_throw();
    cached_abort_event.wait_and_throw();
    ctx.synchronize();
    std::vector<uint8_t> output_after(output_before.size());
    std::vector<uint8_t> partials_after(partials_before.size());
    std::vector<uint8_t> cached_output_after(cached_output_before.size());
    std::vector<uint8_t> cached_partials_after(cached_partials_before.size());
    ctx.memcpy_d2h(output_after.data(), out_graph_dev.data(),
                   output_after.size());
    ctx.memcpy_d2h(partials_after.data(), partials_graph_dev.data(),
                   partials_after.size());
    ctx.memcpy_d2h(cached_output_after.data(), out_cached_dev.data(),
                   cached_output_after.size());
    ctx.memcpy_d2h(cached_partials_after.data(), partials_cached_dev.data(),
                   cached_partials_after.size());
    if (std::memcmp(output_before.data(), output_after.data(),
                    output_before.size()) != 0 ||
        std::memcmp(partials_before.data(), partials_after.data(),
                    partials_before.size()) != 0 ||
        std::memcmp(cached_output_before.data(), cached_output_after.data(),
                    cached_output_before.size()) != 0 ||
        std::memcmp(cached_partials_before.data(), cached_partials_after.data(),
                    cached_partials_before.size()) != 0) {
        throw std::runtime_error(
            "block-native attention graph abort modified output/partials");
    }

    auto replay_guard = [&](const std::string& label,
                            std::array<int, block_topk + compress_ratio> active,
                            int start_pos, int complete_blocks) {
        ctx.memcpy_h2d(active_blocks_dev.data(), active.data(),
                       active.size() * sizeof(int));
        auto run_variant = [&](auto& executable, Tensor& output_tensor,
                               Tensor& partials_tensor,
                               const std::vector<uint8_t>& expected_output,
                               const std::vector<uint8_t>& expected_partials,
                               const char* variant) {
            control[ops::graph_control::start_pos] = start_pos;
            control[ops::graph_control::compressed_blocks] = complete_blocks;
            control[ops::graph_control::abort] = 0;
            ctx.memcpy_h2d(control_dev.data(), control.data(),
                           control.size() * sizeof(int));
            auto event = queue.ext_oneapi_graph(executable);
            event.wait_and_throw();
            ctx.synchronize();
            ctx.memcpy_d2h(control.data(), control_dev.data(),
                           control.size() * sizeof(int));
            std::vector<uint8_t> output_guard(expected_output.size());
            std::vector<uint8_t> partials_guard(expected_partials.size());
            ctx.memcpy_d2h(output_guard.data(), output_tensor.data(),
                           output_guard.size());
            ctx.memcpy_d2h(partials_guard.data(), partials_tensor.data(),
                           partials_guard.size());
            if (control[ops::graph_control::abort] == 0 ||
                std::memcmp(expected_output.data(), output_guard.data(),
                            output_guard.size()) != 0 ||
                std::memcmp(expected_partials.data(), partials_guard.data(),
                            partials_guard.size()) != 0) {
                throw std::runtime_error(
                    std::string("block-native attention graph ") + variant +
                    " accepted " + label);
            }
        };
        run_variant(graph, out_graph_dev, partials_graph_dev,
                    output_before, partials_before, "old");
        run_variant(graph_cached, out_cached_dev, partials_cached_dev,
                    cached_output_before, cached_partials_before, "cached");
    };

    std::array<int, block_topk + compress_ratio> guarded_active{};
    for (int b = 0; b < block_topk; ++b) guarded_active[b] = b;
    guarded_active[block_topk] = 2052;
    replay_guard("an inconsistent compressed block count", guarded_active, 2052, 512);

    guarded_active[0] = 513;
    replay_guard("an out-of-range sparse block id", guarded_active, 2052, 513);

    guarded_active[0] = 0;
    guarded_active[block_topk] = capacity;
    replay_guard("an out-of-range tail token id", guarded_active, 2052, 513);

    std::cout << "[PASS] block-native attention graph"
              << " (start_pos=2052..2055, tiles=17/17/17/16, bitwise,"
              << " abort + invalid-id no-op)\n";
}

void test_attention_decode_block_native_graph_long_capacity(Context& ctx) {
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int capacity = 32768;
    constexpr int max_tiles = 17;
    constexpr int partial_stride = 272;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int start_pos = 32760;
    constexpr int visible_len = start_pos + 1;
    constexpr int compressed_blocks = visible_len / compress_ratio;
    constexpr int tail_len = visible_len % compress_ratio;
    constexpr int effective_len = block_topk * compress_ratio + tail_len;
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;
    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error(
            "long-capacity attention graph test requires SYCL graph support");
    }

    std::array<int, block_topk + compress_ratio> active{};
    for (int i = 0; i < block_topk; ++i) {
        active[i] = (i * 37 + 19) % compressed_blocks;
    }
    std::sort(active.begin(), active.begin() + block_topk);
    active[block_topk] = start_pos;
    auto cache_token = [&](int t) {
        if (t < block_topk * compress_ratio) {
            return active[t / compress_ratio] * compress_ratio +
                   (t % compress_ratio);
        }
        return active[block_topk + (t - block_topk * compress_ratio)];
    };

    std::mt19937 rng(917);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<bf16> q_host(num_heads * head_dim);
    for (auto& value : q_host) value = bf16(dist(rng));
    const size_t kv_elements =
        size_t(num_kv_heads) * capacity * head_dim;
    std::vector<bf16> k_host(kv_elements, bf16(0.0f));
    std::vector<bf16> v_host(kv_elements, bf16(0.0f));
    for (int t = 0; t < effective_len; ++t) {
        const int row = cache_token(t);
        for (int kv = 0; kv < num_kv_heads; ++kv) {
            for (int d = 0; d < head_dim; ++d) {
                const int k_code =
                    ((row * 131 + d * 17 + kv * 29) & 255) - 128;
                const int v_code =
                    ((row * 97 + d * 31 + kv * 13) & 255) - 128;
                const size_t offset =
                    (size_t(kv) * capacity + row) * head_dim + d;
                k_host[offset] = bf16(float(k_code) * (1.0f / 512.0f));
                v_host[offset] = bf16(float(v_code) * (1.0f / 512.0f));
            }
        }
    }

    Tensor q_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor direct_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor old_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor cached_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(
        ctx, {num_heads, capacity}, dnnl::memory::data_type::f32);
    Tensor direct_partials_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride},
        dnnl::memory::data_type::f32);
    Tensor old_partials_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride},
        dnnl::memory::data_type::f32);
    Tensor cached_partials_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride},
        dnnl::memory::data_type::f32);
    Tensor active_dev = Tensor::allocate(
        ctx, {block_topk + compress_ratio}, dnnl::memory::data_type::s32);
    Tensor control_dev = Tensor::allocate(
        ctx, {ops::graph_control::words}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(q_dev.data(), q_host.data(),
                   q_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_dev.data(), k_host.data(),
                   k_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), v_host.data(),
                   v_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(active_dev.data(), active.data(),
                   active.size() * sizeof(int));
    std::array<int, ops::graph_control::words> control{};
    control[ops::graph_control::start_pos] = start_pos;
    control[ops::graph_control::compressed_blocks] = compressed_blocks;
    control[ops::graph_control::abort] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    ctx.synchronize();

    auto capture_graph = [&](bool cache_active_tokens) -> ExecutableGraph {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::attention_validate_qsa_active_blocks_graph(
                ctx, k_dev, v_dev, active_dev, control_dev.data_as<int>(),
                num_kv_heads);
            if (cache_active_tokens) {
                ops::attention_decode_exact_head256_block_native_graph_token_cache(
                    ctx, q_dev, k_dev, v_dev, cached_dev,
                    cached_partials_dev, active_dev, control_dev.data_as<int>(),
                    num_heads, num_kv_heads);
            } else {
                ops::attention_decode_exact_head256_block_native_graph(
                    ctx, q_dev, k_dev, v_dev, old_dev,
                    old_partials_dev, active_dev, control_dev.data_as<int>(),
                    num_heads, num_kv_heads);
            }
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    };
    ExecutableGraph old_graph = capture_graph(false);
    ExecutableGraph cached_graph = capture_graph(true);

    ops::attention_decode(
        ctx, q_dev, k_dev, v_dev, direct_dev, scores_dev,
        num_heads, num_kv_heads, head_dim, visible_len,
        &direct_partials_dev, nullptr, nullptr, 0,
        active_dev.data_as<int>(), block_topk);
    auto old_event = queue.ext_oneapi_graph(old_graph);
    auto cached_event = queue.ext_oneapi_graph(cached_graph);
    old_event.wait_and_throw();
    cached_event.wait_and_throw();
    ctx.synchronize();

    std::vector<bf16> direct_out(num_heads * head_dim);
    std::vector<bf16> old_out(direct_out.size());
    std::vector<bf16> cached_out(direct_out.size());
    std::vector<float> direct_partials(
        size_t(num_heads) * max_tiles * partial_stride);
    std::vector<float> old_partials(direct_partials.size());
    std::vector<float> cached_partials(direct_partials.size());
    ctx.memcpy_d2h(direct_out.data(), direct_dev.data(),
                   direct_out.size() * sizeof(bf16));
    ctx.memcpy_d2h(old_out.data(), old_dev.data(),
                   old_out.size() * sizeof(bf16));
    ctx.memcpy_d2h(cached_out.data(), cached_dev.data(),
                   cached_out.size() * sizeof(bf16));
    ctx.memcpy_d2h(direct_partials.data(), direct_partials_dev.data(),
                   direct_partials.size() * sizeof(float));
    ctx.memcpy_d2h(old_partials.data(), old_partials_dev.data(),
                   old_partials.size() * sizeof(float));
    ctx.memcpy_d2h(cached_partials.data(), cached_partials_dev.data(),
                   cached_partials.size() * sizeof(float));
    if (std::memcmp(direct_out.data(), old_out.data(),
                    direct_out.size() * sizeof(bf16)) != 0 ||
        std::memcmp(direct_out.data(), cached_out.data(),
                    direct_out.size() * sizeof(bf16)) != 0 ||
        std::memcmp(direct_partials.data(), old_partials.data(),
                    direct_partials.size() * sizeof(float)) != 0 ||
        std::memcmp(direct_partials.data(), cached_partials.data(),
                    direct_partials.size() * sizeof(float)) != 0) {
        throw std::runtime_error(
            "long-capacity block-native attention graph mismatch");
    }

    std::cout << "[PASS] block-native attention graph long capacity"
              << " (max_seq=32768, start_pos=32760, selected_blocks=512,"
              << " tail=1, non-contiguous ids, bitwise)\n";
}

void test_decode_prepare_qgkv_graph(Context& ctx) {
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int capacity = 2060;
    constexpr int packed_dim = num_heads * head_dim * 2;
    constexpr int kv_dim = num_kv_heads * head_dim;
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;
    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error("QGKV graph test requires SYCL graph support");
    }

    std::mt19937 rng(913);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    std::vector<bf16> packed(packed_dim);
    std::vector<bf16> k_input(kv_dim);
    std::vector<bf16> v_input(kv_dim);
    std::vector<bf16> q_norm(head_dim);
    std::vector<bf16> k_norm(head_dim);
    std::vector<bf16> initial_kv(size_t(num_kv_heads) * capacity * head_dim);
    for (auto& value : packed) value = bf16(dist(rng));
    for (auto& value : k_input) value = bf16(dist(rng));
    for (auto& value : v_input) value = bf16(dist(rng));
    for (int d = 0; d < head_dim; ++d) {
        q_norm[d] = bf16(0.9f + float(d % 13) * 0.01f);
        k_norm[d] = bf16(0.95f + float(d % 11) * 0.01f);
    }
    for (auto& value : initial_kv) value = bf16(dist(rng) * 0.1f);

    Tensor packed_dev = Tensor::allocate(
        ctx, {1, packed_dim}, dnnl::memory::data_type::bf16);
    Tensor k_direct_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor k_graph_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor v_direct_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor v_graph_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor q_direct_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_graph_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor gate_direct_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor gate_graph_dev = Tensor::allocate(
        ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor q_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_direct_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_graph_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_cache_direct_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_cache_graph_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor control_dev = Tensor::allocate(
        ctx, {ops::graph_control::words}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(packed_dev.data(), packed.data(), packed.size() * sizeof(bf16));
    ctx.memcpy_h2d(q_norm_dev.data(), q_norm.data(), q_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_norm_dev.data(), k_norm.data(), k_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_direct_dev.data(), initial_kv.data(),
                   initial_kv.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_graph_dev.data(), initial_kv.data(),
                   initial_kv.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_cache_direct_dev.data(), initial_kv.data(),
                   initial_kv.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_cache_graph_dev.data(), initial_kv.data(),
                   initial_kv.size() * sizeof(bf16));

    auto reset_kv_inputs = [&]() {
        ctx.memcpy_h2d(k_direct_dev.data(), k_input.data(),
                       k_input.size() * sizeof(bf16));
        ctx.memcpy_h2d(k_graph_dev.data(), k_input.data(),
                       k_input.size() * sizeof(bf16));
        ctx.memcpy_h2d(v_direct_dev.data(), v_input.data(),
                       v_input.size() * sizeof(bf16));
        ctx.memcpy_h2d(v_graph_dev.data(), v_input.data(),
                       v_input.size() * sizeof(bf16));
    };
    reset_kv_inputs();
    std::array<int, ops::graph_control::words> control{};
    control[ops::graph_control::start_pos] = 2052;
    control[ops::graph_control::compressed_blocks] = 513;
    control[ops::graph_control::abort] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));

    ExecutableGraph graph = [&]() {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::decode_prepare_qgkv_packed_partial_graph(
                ctx, packed_dev, k_graph_dev, v_graph_dev, q_graph_dev,
                gate_graph_dev, q_norm_dev, k_norm_dev, k_cache_graph_dev,
                v_cache_graph_dev, control_dev.data_as<int>(), num_heads,
                num_kv_heads, head_dim, 1e-6f, 64, 10000000.0f);
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    }();

    for (int step = 0; step < 4; ++step) {
        const int position = 2052 + step;
        reset_kv_inputs();
        control[ops::graph_control::start_pos] = position;
        control[ops::graph_control::abort] = 0;
        ctx.memcpy_h2d(control_dev.data(), control.data(),
                       control.size() * sizeof(int));
        ops::decode_prepare_qgkv_packed_partial(
            ctx, packed_dev, k_direct_dev, v_direct_dev, q_direct_dev,
            gate_direct_dev, q_norm_dev, k_norm_dev, k_cache_direct_dev,
            v_cache_direct_dev, position, num_heads, num_kv_heads,
            head_dim, 1e-6f, 64, 10000000.0f);
        auto event = queue.ext_oneapi_graph(graph);
        event.wait_and_throw();
        ctx.synchronize();

        std::vector<bf16> q_direct(num_heads * head_dim);
        std::vector<bf16> q_graph(num_heads * head_dim);
        std::vector<bf16> gate_direct(num_heads * head_dim);
        std::vector<bf16> gate_graph(num_heads * head_dim);
        std::vector<bf16> k_direct(kv_dim), k_graph(kv_dim);
        std::vector<bf16> v_direct(kv_dim), v_graph(kv_dim);
        std::vector<bf16> kc_direct(initial_kv.size());
        std::vector<bf16> kc_graph(initial_kv.size());
        std::vector<bf16> vc_direct(initial_kv.size());
        std::vector<bf16> vc_graph(initial_kv.size());
        ctx.memcpy_d2h(q_direct.data(), q_direct_dev.data(),
                       q_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(q_graph.data(), q_graph_dev.data(),
                       q_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(gate_direct.data(), gate_direct_dev.data(),
                       gate_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(gate_graph.data(), gate_graph_dev.data(),
                       gate_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_direct.data(), k_direct_dev.data(),
                       k_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_graph.data(), k_graph_dev.data(),
                       k_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(v_direct.data(), v_direct_dev.data(),
                       v_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(v_graph.data(), v_graph_dev.data(),
                       v_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(kc_direct.data(), k_cache_direct_dev.data(),
                       kc_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(kc_graph.data(), k_cache_graph_dev.data(),
                       kc_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(vc_direct.data(), v_cache_direct_dev.data(),
                       vc_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(vc_graph.data(), v_cache_graph_dev.data(),
                       vc_graph.size() * sizeof(bf16));
        if (std::memcmp(q_direct.data(), q_graph.data(),
                        q_direct.size() * sizeof(bf16)) != 0 ||
            std::memcmp(gate_direct.data(), gate_graph.data(),
                        gate_direct.size() * sizeof(bf16)) != 0 ||
            std::memcmp(k_direct.data(), k_graph.data(),
                        k_direct.size() * sizeof(bf16)) != 0 ||
            std::memcmp(v_direct.data(), v_graph.data(),
                        v_direct.size() * sizeof(bf16)) != 0 ||
            std::memcmp(kc_direct.data(), kc_graph.data(),
                        kc_direct.size() * sizeof(bf16)) != 0 ||
            std::memcmp(vc_direct.data(), vc_graph.data(),
                        vc_direct.size() * sizeof(bf16)) != 0) {
            throw std::runtime_error(
                "QGKV graph position mismatch at pos=" +
                std::to_string(position));
        }
    }

    std::vector<bf16> q_guard(num_heads * head_dim);
    std::vector<bf16> gate_guard(num_heads * head_dim);
    std::vector<bf16> k_guard(kv_dim);
    std::vector<bf16> v_guard(kv_dim);
    std::vector<bf16> kc_guard(initial_kv.size());
    std::vector<bf16> vc_guard(initial_kv.size());
    ctx.memcpy_d2h(q_guard.data(), q_graph_dev.data(),
                   q_guard.size() * sizeof(bf16));
    ctx.memcpy_d2h(gate_guard.data(), gate_graph_dev.data(),
                   gate_guard.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_guard.data(), k_graph_dev.data(),
                   k_guard.size() * sizeof(bf16));
    ctx.memcpy_d2h(v_guard.data(), v_graph_dev.data(),
                   v_guard.size() * sizeof(bf16));
    ctx.memcpy_d2h(kc_guard.data(), k_cache_graph_dev.data(),
                   kc_guard.size() * sizeof(bf16));
    ctx.memcpy_d2h(vc_guard.data(), v_cache_graph_dev.data(),
                   vc_guard.size() * sizeof(bf16));
    control[ops::graph_control::start_pos] = 2055;
    control[ops::graph_control::compressed_blocks] = 512;
    control[ops::graph_control::abort] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto stale_event = queue.ext_oneapi_graph(graph);
    stale_event.wait_and_throw();
    ctx.synchronize();
    ctx.memcpy_d2h(control.data(), control_dev.data(),
                   control.size() * sizeof(int));
    std::vector<bf16> q_after(num_heads * head_dim);
    std::vector<bf16> gate_after(num_heads * head_dim);
    std::vector<bf16> k_after(kv_dim);
    std::vector<bf16> v_after(kv_dim);
    std::vector<bf16> kc_after(initial_kv.size());
    std::vector<bf16> vc_after(initial_kv.size());
    ctx.memcpy_d2h(q_after.data(), q_graph_dev.data(),
                   q_after.size() * sizeof(bf16));
    ctx.memcpy_d2h(gate_after.data(), gate_graph_dev.data(),
                   gate_after.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_after.data(), k_graph_dev.data(),
                   k_after.size() * sizeof(bf16));
    ctx.memcpy_d2h(v_after.data(), v_graph_dev.data(),
                   v_after.size() * sizeof(bf16));
    ctx.memcpy_d2h(kc_after.data(), k_cache_graph_dev.data(),
                   kc_after.size() * sizeof(bf16));
    ctx.memcpy_d2h(vc_after.data(), v_cache_graph_dev.data(),
                   vc_after.size() * sizeof(bf16));
    if (control[ops::graph_control::abort] == 0 ||
        std::memcmp(q_guard.data(), q_after.data(),
                    q_guard.size() * sizeof(bf16)) != 0 ||
        std::memcmp(gate_guard.data(), gate_after.data(),
                    gate_guard.size() * sizeof(bf16)) != 0 ||
        std::memcmp(k_guard.data(), k_after.data(),
                    k_guard.size() * sizeof(bf16)) != 0 ||
        std::memcmp(v_guard.data(), v_after.data(),
                    v_guard.size() * sizeof(bf16)) != 0 ||
        std::memcmp(kc_guard.data(), kc_after.data(),
                    kc_guard.size() * sizeof(bf16)) != 0 ||
        std::memcmp(vc_guard.data(), vc_after.data(),
                    vc_guard.size() * sizeof(bf16)) != 0) {
        throw std::runtime_error(
            "QGKV graph accepted a stale compressed block count");
    }

    const size_t cache_bytes =
        initial_kv.size() * sizeof(bf16);
    std::vector<uint8_t> cache_before(cache_bytes);
    ctx.memcpy_d2h(cache_before.data(), k_cache_graph_dev.data(),
                   cache_before.size());
    control[ops::graph_control::abort] = 1;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto abort_event = queue.ext_oneapi_graph(graph);
    abort_event.wait_and_throw();
    ctx.synchronize();
    std::vector<uint8_t> cache_after(cache_before.size());
    ctx.memcpy_d2h(cache_after.data(), k_cache_graph_dev.data(),
                   cache_after.size());
    if (std::memcmp(cache_before.data(), cache_after.data(),
                    cache_before.size()) != 0) {
        throw std::runtime_error(
            "QGKV graph abort modified K cache");
    }

    std::cout << "[PASS] graph-safe decode_prepare_qgkv"
              << " (pos=2052..2055, q/gate/K/V/cache bitwise,"
              << " abort + stale-count no-op)\n";
}

void test_qsa_layer_graph_composite(Context& ctx) {
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int qsa_heads = 4;
    constexpr int qsa_head_dim = 128;
    constexpr int capacity = 2060;
    constexpr int max_blocks = 516;
    constexpr int first_pos = 2052;
    constexpr int first_blocks = (first_pos + 1) / 4;
    constexpr int steps = 4;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int max_tiles = 17;
    constexpr int partial_stride = 272;
    constexpr int q_dim = num_heads * head_dim;
    constexpr int packed_dim = q_dim * 2;
    constexpr int kv_dim = num_kv_heads * head_dim;
    constexpr int qsa_q_dim = qsa_heads * qsa_head_dim;
    constexpr int qsa_qk_dim = (qsa_heads + 1) * qsa_head_dim;
    constexpr float theta = 10000.0f;
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;

    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error("QSA layer graph test requires SYCL graph support");
    }

    std::mt19937 init_rng(20261018);
    std::uniform_real_distribution<float> init_dist(-0.20f, 0.20f);
    std::vector<bf16> initial_k(size_t(num_kv_heads) * capacity * head_dim);
    std::vector<bf16> initial_v(initial_k.size());
    std::vector<bf16> initial_indexer_k(size_t(capacity) * qsa_head_dim);
    std::vector<bf16> initial_compressed(size_t(max_blocks) * qsa_head_dim);
    for (auto& value : initial_k) value = bf16(init_dist(init_rng));
    for (auto& value : initial_v) value = bf16(init_dist(init_rng));
    for (auto& value : initial_indexer_k) value = bf16(init_dist(init_rng));
    for (auto& value : initial_compressed) value = bf16(init_dist(init_rng));

    std::vector<bf16> main_q_norm(head_dim);
    std::vector<bf16> main_k_norm(head_dim);
    std::vector<bf16> qsa_q_norm(qsa_head_dim);
    std::vector<bf16> qsa_k_norm(qsa_head_dim);
    for (int d = 0; d < head_dim; ++d) {
        main_q_norm[d] = bf16(0.9f + float(d % 13) * 0.01f);
        main_k_norm[d] = bf16(0.95f + float(d % 11) * 0.01f);
    }
    for (int d = 0; d < qsa_head_dim; ++d) {
        qsa_q_norm[d] = bf16(0.9f + float(d % 17) * 0.01f);
        qsa_k_norm[d] = bf16(0.95f + float(d % 19) * 0.01f);
    }

    Tensor packed_dev = Tensor::allocate(
        ctx, {1, packed_dim}, dnnl::memory::data_type::bf16);
    Tensor k_input_direct_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor k_input_graph_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor v_input_direct_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor v_input_graph_dev = Tensor::allocate(
        ctx, {1, kv_dim}, dnnl::memory::data_type::bf16);
    Tensor indexer_qk_dev = Tensor::allocate(
        ctx, {1, qsa_qk_dim}, dnnl::memory::data_type::bf16);
    Tensor main_q_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor main_k_norm_dev = Tensor::allocate(
        ctx, {head_dim}, dnnl::memory::data_type::bf16);
    Tensor qsa_q_norm_dev = Tensor::allocate(
        ctx, {qsa_head_dim}, dnnl::memory::data_type::bf16);
    Tensor qsa_k_norm_dev = Tensor::allocate(
        ctx, {qsa_head_dim}, dnnl::memory::data_type::bf16);

    Tensor q_direct_dev = Tensor::allocate(
        ctx, {1, q_dim}, dnnl::memory::data_type::bf16);
    Tensor q_graph_dev = Tensor::allocate(
        ctx, {1, q_dim}, dnnl::memory::data_type::bf16);
    Tensor gate_direct_dev = Tensor::allocate(
        ctx, {1, q_dim}, dnnl::memory::data_type::bf16);
    Tensor gate_graph_dev = Tensor::allocate(
        ctx, {1, q_dim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_direct_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_graph_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_cache_direct_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_cache_graph_dev = Tensor::allocate(
        ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);

    Tensor qsa_q_direct_dev = Tensor::allocate(
        ctx, {1, qsa_q_dim}, dnnl::memory::data_type::bf16);
    Tensor qsa_q_graph_dev = Tensor::allocate(
        ctx, {1, qsa_q_dim}, dnnl::memory::data_type::bf16);
    Tensor qsa_k_direct_dev = Tensor::allocate(
        ctx, {capacity, qsa_head_dim}, dnnl::memory::data_type::bf16);
    Tensor qsa_k_graph_dev = Tensor::allocate(
        ctx, {capacity, qsa_head_dim}, dnnl::memory::data_type::bf16);
    Tensor compressed_direct_dev = Tensor::allocate(
        ctx, {max_blocks, qsa_head_dim}, dnnl::memory::data_type::bf16);
    Tensor compressed_graph_dev = Tensor::allocate(
        ctx, {max_blocks, qsa_head_dim}, dnnl::memory::data_type::bf16);
    Tensor scores_direct_dev = Tensor::allocate(
        ctx, {max_blocks}, dnnl::memory::data_type::f32);
    Tensor scores_graph_dev = Tensor::allocate(
        ctx, {max_blocks}, dnnl::memory::data_type::f32);
    Tensor active_direct_dev = Tensor::allocate(
        ctx, {block_topk + compress_ratio}, dnnl::memory::data_type::s32);
    Tensor active_graph_dev = Tensor::allocate(
        ctx, {block_topk + compress_ratio}, dnnl::memory::data_type::s32);
    Tensor partials_direct_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride},
        dnnl::memory::data_type::f32);
    Tensor partials_graph_dev = Tensor::allocate(
        ctx, {num_heads, max_tiles, partial_stride},
        dnnl::memory::data_type::f32);
    Tensor attention_scores_direct_dev = Tensor::allocate(
        ctx, {num_heads, capacity}, dnnl::memory::data_type::f32);
    Tensor output_direct_dev = Tensor::allocate(
        ctx, {1, q_dim}, dnnl::memory::data_type::bf16);
    Tensor output_graph_dev = Tensor::allocate(
        ctx, {1, q_dim}, dnnl::memory::data_type::bf16);
    Tensor control_dev = Tensor::allocate(
        ctx, {ops::graph_control::words}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(main_q_norm_dev.data(), main_q_norm.data(),
                   main_q_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(main_k_norm_dev.data(), main_k_norm.data(),
                   main_k_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(qsa_q_norm_dev.data(), qsa_q_norm.data(),
                   qsa_q_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(qsa_k_norm_dev.data(), qsa_k_norm.data(),
                   qsa_k_norm.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_direct_dev.data(), initial_k.data(),
                   initial_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_graph_dev.data(), initial_k.data(),
                   initial_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_cache_direct_dev.data(), initial_v.data(),
                   initial_v.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_cache_graph_dev.data(), initial_v.data(),
                   initial_v.size() * sizeof(bf16));
    ctx.memcpy_h2d(qsa_k_direct_dev.data(), initial_indexer_k.data(),
                   initial_indexer_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(qsa_k_graph_dev.data(), initial_indexer_k.data(),
                   initial_indexer_k.size() * sizeof(bf16));
    ctx.memcpy_h2d(compressed_direct_dev.data(), initial_compressed.data(),
                   initial_compressed.size() * sizeof(bf16));
    ctx.memcpy_h2d(compressed_graph_dev.data(), initial_compressed.data(),
                   initial_compressed.size() * sizeof(bf16));

    std::vector<float> initial_scores(max_blocks, -12345.0f);
    std::vector<int> initial_active(block_topk + compress_ratio, -1);
    std::vector<float> initial_partials(
        size_t(num_heads) * max_tiles * partial_stride, -54321.0f);
    ctx.memcpy_h2d(scores_direct_dev.data(), initial_scores.data(),
                   initial_scores.size() * sizeof(float));
    ctx.memcpy_h2d(scores_graph_dev.data(), initial_scores.data(),
                   initial_scores.size() * sizeof(float));
    ctx.memcpy_h2d(active_direct_dev.data(), initial_active.data(),
                   initial_active.size() * sizeof(int));
    ctx.memcpy_h2d(active_graph_dev.data(), initial_active.data(),
                   initial_active.size() * sizeof(int));
    ctx.memcpy_h2d(partials_direct_dev.data(), initial_partials.data(),
                   initial_partials.size() * sizeof(float));
    ctx.memcpy_h2d(partials_graph_dev.data(), initial_partials.data(),
                   initial_partials.size() * sizeof(float));
    ctx.synchronize();

    std::array<int, ops::graph_control::words> control{};
    control[ops::graph_control::start_pos] = first_pos;
    control[ops::graph_control::compressed_blocks] = first_blocks;
    control[ops::graph_control::abort] = 0;
    control[3] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    ctx.synchronize();

    ExecutableGraph graph = [&]() {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::decode_prepare_qgkv_packed_partial_graph(
                ctx, packed_dev, k_input_graph_dev, v_input_graph_dev,
                q_graph_dev, gate_graph_dev, main_q_norm_dev, main_k_norm_dev,
                k_cache_graph_dev, v_cache_graph_dev,
                control_dev.data_as<int>(), num_heads, num_kv_heads, head_dim,
                1e-6f, 64, theta);
            ops::qsa_prepare_q_and_cache_k_graph(
                ctx, indexer_qk_dev, qsa_q_norm_dev, qsa_k_graph_dev,
                qsa_q_graph_dev, control_dev.data_as<int>(), theta);
            ops::qsa_compress_one_completed_block_graph(
                ctx, qsa_k_graph_dev, qsa_k_norm_dev, compressed_graph_dev,
                control_dev.data_as<int>(), theta);
            ops::qsa_compute_block_scores_compressed_graph(
                ctx, qsa_q_graph_dev, compressed_graph_dev, scores_graph_dev,
                control_dev.data_as<int>(), max_blocks);
            ops::qsa_select_topk_blocks_graph(
                ctx, scores_graph_dev, active_graph_dev,
                control_dev.data_as<int>(), block_topk, compress_ratio,
                max_blocks);
            ops::attention_validate_qsa_active_blocks_graph(
                ctx, k_cache_graph_dev, v_cache_graph_dev, active_graph_dev,
                control_dev.data_as<int>(), num_kv_heads);
            ops::attention_decode_exact_head256_block_native_graph(
                ctx, q_graph_dev, k_cache_graph_dev, v_cache_graph_dev,
                output_graph_dev, partials_graph_dev, active_graph_dev,
                control_dev.data_as<int>(), num_heads, num_kv_heads,
                &gate_graph_dev);
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    }();

    auto require_bitwise = [](const auto& direct, const auto& graph_out,
                              const char* label, int position) {
        using value_type =
            typename std::decay_t<decltype(direct)>::value_type;
        if (direct.size() != graph_out.size() ||
            std::memcmp(direct.data(), graph_out.data(),
                        direct.size() * sizeof(value_type)) != 0) {
            throw std::runtime_error(
                "QSA layer graph " + std::string(label) +
                " mismatch at pos=" + std::to_string(position));
        }
    };

    int host_blocks = first_blocks;
    int graph_blocks = first_blocks;
    std::uniform_real_distribution<float> step_dist(-0.35f, 0.35f);
    for (int step = 0; step < steps; ++step) {
        const int position = first_pos + step;
        const int visible_len = position + 1;
        const int target_blocks = visible_len / compress_ratio;

        std::mt19937 step_rng(9000 + position);
        std::vector<bf16> packed(packed_dim);
        std::vector<bf16> k_input(kv_dim);
        std::vector<bf16> v_input(kv_dim);
        std::vector<bf16> indexer_qk(qsa_qk_dim);
        for (auto& value : packed) value = bf16(step_dist(step_rng));
        for (auto& value : k_input) value = bf16(step_dist(step_rng));
        for (auto& value : v_input) value = bf16(step_dist(step_rng));
        for (auto& value : indexer_qk) value = bf16(step_dist(step_rng));

        ctx.memcpy_h2d(packed_dev.data(), packed.data(),
                       packed.size() * sizeof(bf16));
        ctx.memcpy_h2d(k_input_direct_dev.data(), k_input.data(),
                       k_input.size() * sizeof(bf16));
        ctx.memcpy_h2d(k_input_graph_dev.data(), k_input.data(),
                       k_input.size() * sizeof(bf16));
        ctx.memcpy_h2d(v_input_direct_dev.data(), v_input.data(),
                       v_input.size() * sizeof(bf16));
        ctx.memcpy_h2d(v_input_graph_dev.data(), v_input.data(),
                       v_input.size() * sizeof(bf16));
        ctx.memcpy_h2d(indexer_qk_dev.data(), indexer_qk.data(),
                       indexer_qk.size() * sizeof(bf16));

        control[ops::graph_control::start_pos] = position;
        control[ops::graph_control::compressed_blocks] = graph_blocks;
        control[ops::graph_control::abort] = 0;
        control[3] = 0;
        ctx.memcpy_h2d(control_dev.data(), control.data(),
                       control.size() * sizeof(int));

        ops::decode_prepare_qgkv_packed_partial(
            ctx, packed_dev, k_input_direct_dev, v_input_direct_dev,
            q_direct_dev, gate_direct_dev, main_q_norm_dev, main_k_norm_dev,
            k_cache_direct_dev, v_cache_direct_dev, position, num_heads,
            num_kv_heads, head_dim, 1e-6f, 64, theta);
        ops::qsa_prepare_q_and_cache_k(
            ctx, indexer_qk_dev, qsa_q_norm_dev, qsa_k_direct_dev,
            qsa_q_direct_dev, position, theta);
        if (target_blocks > host_blocks) {
            if (target_blocks != host_blocks + 1) {
                throw std::runtime_error(
                    "QSA layer graph direct reference skipped a block");
            }
            ops::qsa_compress_key_blocks(
                ctx, qsa_k_direct_dev, qsa_k_norm_dev, compressed_direct_dev,
                host_blocks, 1, theta);
            host_blocks = target_blocks;
        }
        ops::qsa_compute_block_scores_compressed(
            ctx, qsa_q_direct_dev, compressed_direct_dev, scores_direct_dev,
            host_blocks);
        const int direct_active_blocks = ops::qsa_select_topk_blocks(
            ctx, scores_direct_dev, host_blocks, block_topk, visible_len,
            compress_ratio, active_direct_dev);
        const int expected_active_blocks =
            std::min(host_blocks, block_topk);
        if (direct_active_blocks != expected_active_blocks) {
            throw std::runtime_error(
                "QSA layer graph direct selector count mismatch at pos=" +
                std::to_string(position));
        }
        ops::attention_decode(
            ctx, q_direct_dev, k_cache_direct_dev, v_cache_direct_dev,
            output_direct_dev, attention_scores_direct_dev, num_heads,
            num_kv_heads, head_dim, visible_len, &partials_direct_dev,
            &gate_direct_dev, nullptr, 0, active_direct_dev.data_as<int>(),
            block_topk);

        auto event = queue.ext_oneapi_graph(graph);
        event.wait_and_throw();
        ctx.synchronize();

        ctx.memcpy_d2h(control.data(), control_dev.data(),
                       control.size() * sizeof(int));
        graph_blocks = control[ops::graph_control::compressed_blocks];
        if (control[3] != 0) {
            throw std::runtime_error(
                "QSA layer graph reported an invalid block gap at pos=" +
                std::to_string(position));
        }
        if (control[ops::graph_control::abort] != 0) {
            throw std::runtime_error(
                "QSA layer graph aborted at pos=" + std::to_string(position));
        }
        if (graph_blocks != host_blocks) {
            throw std::runtime_error(
                "QSA layer graph block count mismatch at pos=" +
                std::to_string(position) + ": graph=" +
                std::to_string(graph_blocks) + " direct=" +
                std::to_string(host_blocks));
        }

        std::vector<bf16> q_direct(q_dim), q_graph(q_dim);
        std::vector<bf16> gate_direct(q_dim), gate_graph(q_dim);
        std::vector<bf16> k_scratch_direct(kv_dim), k_scratch_graph(kv_dim);
        std::vector<bf16> v_scratch_direct(kv_dim), v_scratch_graph(kv_dim);
        std::vector<bf16> k_cache_direct(initial_k.size());
        std::vector<bf16> k_cache_graph(initial_k.size());
        std::vector<bf16> v_cache_direct(initial_v.size());
        std::vector<bf16> v_cache_graph(initial_v.size());
        std::vector<bf16> qsa_q_direct(qsa_q_dim), qsa_q_graph(qsa_q_dim);
        std::vector<bf16> qsa_k_direct(initial_indexer_k.size());
        std::vector<bf16> qsa_k_graph(initial_indexer_k.size());
        std::vector<bf16> compressed_block_direct(qsa_head_dim);
        std::vector<bf16> compressed_block_graph(qsa_head_dim);
        std::vector<float> scores_direct(max_blocks), scores_graph(max_blocks);
        std::vector<int> active_direct(block_topk + compress_ratio);
        std::vector<int> active_graph(block_topk + compress_ratio);
        std::vector<bf16> output_direct(q_dim), output_graph(q_dim);
        std::vector<float> partials_direct(
            size_t(num_heads) * max_tiles * partial_stride);
        std::vector<float> partials_graph(partials_direct.size());

        ctx.memcpy_d2h(q_direct.data(), q_direct_dev.data(),
                       q_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(q_graph.data(), q_graph_dev.data(),
                       q_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(gate_direct.data(), gate_direct_dev.data(),
                       gate_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(gate_graph.data(), gate_graph_dev.data(),
                       gate_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_scratch_direct.data(), k_input_direct_dev.data(),
                       k_scratch_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_scratch_graph.data(), k_input_graph_dev.data(),
                       k_scratch_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(v_scratch_direct.data(), v_input_direct_dev.data(),
                       v_scratch_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(v_scratch_graph.data(), v_input_graph_dev.data(),
                       v_scratch_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_cache_direct.data(), k_cache_direct_dev.data(),
                       k_cache_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(k_cache_graph.data(), k_cache_graph_dev.data(),
                       k_cache_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(v_cache_direct.data(), v_cache_direct_dev.data(),
                       v_cache_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(v_cache_graph.data(), v_cache_graph_dev.data(),
                       v_cache_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(qsa_q_direct.data(), qsa_q_direct_dev.data(),
                       qsa_q_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(qsa_q_graph.data(), qsa_q_graph_dev.data(),
                       qsa_q_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(qsa_k_direct.data(), qsa_k_direct_dev.data(),
                       qsa_k_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(qsa_k_graph.data(), qsa_k_graph_dev.data(),
                       qsa_k_graph.size() * sizeof(bf16));
        const size_t compressed_offset =
            size_t(513) * qsa_head_dim;
        ctx.memcpy_d2h(
            compressed_block_direct.data(),
            compressed_direct_dev.data_as<bf16>() + compressed_offset,
            compressed_block_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(
            compressed_block_graph.data(),
            compressed_graph_dev.data_as<bf16>() + compressed_offset,
            compressed_block_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(scores_direct.data(), scores_direct_dev.data(),
                       scores_direct.size() * sizeof(float));
        ctx.memcpy_d2h(scores_graph.data(), scores_graph_dev.data(),
                       scores_graph.size() * sizeof(float));
        ctx.memcpy_d2h(active_direct.data(), active_direct_dev.data(),
                       active_direct.size() * sizeof(int));
        ctx.memcpy_d2h(active_graph.data(), active_graph_dev.data(),
                       active_graph.size() * sizeof(int));
        ctx.memcpy_d2h(output_direct.data(), output_direct_dev.data(),
                       output_direct.size() * sizeof(bf16));
        ctx.memcpy_d2h(output_graph.data(), output_graph_dev.data(),
                       output_graph.size() * sizeof(bf16));
        ctx.memcpy_d2h(partials_direct.data(), partials_direct_dev.data(),
                       partials_direct.size() * sizeof(float));
        ctx.memcpy_d2h(partials_graph.data(), partials_graph_dev.data(),
                       partials_graph.size() * sizeof(float));

        require_bitwise(q_direct, q_graph, "q", position);
        require_bitwise(gate_direct, gate_graph, "gate", position);
        require_bitwise(k_scratch_direct, k_scratch_graph,
                        "K scratch", position);
        require_bitwise(v_scratch_direct, v_scratch_graph,
                        "V scratch", position);
        require_bitwise(k_cache_direct, k_cache_graph,
                        "full K cache", position);
        require_bitwise(v_cache_direct, v_cache_graph,
                        "full V cache", position);
        require_bitwise(qsa_q_direct, qsa_q_graph,
                        "indexer q", position);
        require_bitwise(qsa_k_direct, qsa_k_graph,
                        "indexer K cache", position);
        require_bitwise(compressed_block_direct, compressed_block_graph,
                        "compressed K[513]", position);
        require_bitwise(scores_direct, scores_graph, "scores", position);
        require_bitwise(active_direct, active_graph,
                        "active blocks", position);
        require_bitwise(output_direct, output_graph,
                        "attention/final output", position);
        require_bitwise(partials_direct, partials_graph,
                        "partials", position);
    }

    if (host_blocks != 514 || graph_blocks != 514) {
        throw std::runtime_error(
            "QSA layer graph did not reach the 514-block boundary");
    }

    struct GraphBytes {
        std::vector<uint8_t> q;
        std::vector<uint8_t> gate;
        std::vector<uint8_t> k_scratch;
        std::vector<uint8_t> v_scratch;
        std::vector<uint8_t> k_cache;
        std::vector<uint8_t> v_cache;
        std::vector<uint8_t> indexer_q;
        std::vector<uint8_t> indexer_k;
        std::vector<uint8_t> compressed_block;
        std::vector<uint8_t> scores;
        std::vector<uint8_t> active;
        std::vector<uint8_t> output;
        std::vector<uint8_t> partials;
    };
    auto read_bytes = [&](const void* device_ptr, size_t bytes) {
        std::vector<uint8_t> out(bytes);
        ctx.memcpy_d2h(out.data(), device_ptr, bytes);
        return out;
    };
    auto capture_graph_bytes = [&]() {
        GraphBytes snapshot;
        snapshot.q = read_bytes(q_graph_dev.data(), size_t(q_dim) * sizeof(bf16));
        snapshot.gate =
            read_bytes(gate_graph_dev.data(), size_t(q_dim) * sizeof(bf16));
        snapshot.k_scratch =
            read_bytes(k_input_graph_dev.data(), size_t(kv_dim) * sizeof(bf16));
        snapshot.v_scratch =
            read_bytes(v_input_graph_dev.data(), size_t(kv_dim) * sizeof(bf16));
        snapshot.k_cache = read_bytes(
            k_cache_graph_dev.data(), initial_k.size() * sizeof(bf16));
        snapshot.v_cache = read_bytes(
            v_cache_graph_dev.data(), initial_v.size() * sizeof(bf16));
        snapshot.indexer_q =
            read_bytes(qsa_q_graph_dev.data(), size_t(qsa_q_dim) * sizeof(bf16));
        snapshot.indexer_k = read_bytes(
            qsa_k_graph_dev.data(), initial_indexer_k.size() * sizeof(bf16));
        const size_t compressed_bytes =
            size_t(qsa_head_dim) * sizeof(bf16);
        snapshot.compressed_block = read_bytes(
            compressed_graph_dev.data_as<bf16>() +
                size_t(513) * qsa_head_dim,
            compressed_bytes);
        snapshot.scores =
            read_bytes(scores_graph_dev.data(), size_t(max_blocks) * sizeof(float));
        snapshot.active = read_bytes(
            active_graph_dev.data(),
            size_t(block_topk + compress_ratio) * sizeof(int));
        snapshot.output =
            read_bytes(output_graph_dev.data(), size_t(q_dim) * sizeof(bf16));
        snapshot.partials = read_bytes(
            partials_graph_dev.data(),
            size_t(num_heads) * max_tiles * partial_stride * sizeof(float));
        return snapshot;
    };
    auto require_unchanged = [](const std::vector<uint8_t>& before,
                                const std::vector<uint8_t>& after,
                                const char* label) {
        if (before != after) {
            throw std::runtime_error(
                "QSA layer graph abort modified " + std::string(label));
        }
    };

    GraphBytes before_abort = capture_graph_bytes();
    control[ops::graph_control::start_pos] = -1;
    control[ops::graph_control::compressed_blocks] = graph_blocks;
    control[ops::graph_control::abort] = 1;
    control[3] = 0;
    ctx.memcpy_h2d(control_dev.data(), control.data(),
                   control.size() * sizeof(int));
    auto abort_event = queue.ext_oneapi_graph(graph);
    abort_event.wait_and_throw();
    ctx.synchronize();
    ctx.memcpy_d2h(control.data(), control_dev.data(),
                   control.size() * sizeof(int));
    if (control[ops::graph_control::abort] == 0 ||
        control[ops::graph_control::compressed_blocks] != graph_blocks) {
        throw std::runtime_error(
            "QSA layer graph abort changed control state");
    }

    GraphBytes after_abort = capture_graph_bytes();
    require_unchanged(before_abort.q, after_abort.q, "q");
    require_unchanged(before_abort.gate, after_abort.gate, "gate");
    require_unchanged(before_abort.k_scratch, after_abort.k_scratch,
                      "K scratch");
    require_unchanged(before_abort.v_scratch, after_abort.v_scratch,
                      "V scratch");
    require_unchanged(before_abort.k_cache, after_abort.k_cache,
                      "K cache");
    require_unchanged(before_abort.v_cache, after_abort.v_cache,
                      "V cache");
    require_unchanged(before_abort.indexer_q, after_abort.indexer_q,
                      "indexer q");
    require_unchanged(before_abort.indexer_k, after_abort.indexer_k,
                      "indexer K cache");
    require_unchanged(before_abort.compressed_block,
                      after_abort.compressed_block, "compressed K[513]");
    require_unchanged(before_abort.scores, after_abort.scores, "scores");
    require_unchanged(before_abort.active, after_abort.active,
                      "active blocks");
    require_unchanged(before_abort.output, after_abort.output,
                      "attention/final output");
    require_unchanged(before_abort.partials, after_abort.partials,
                      "partials");

    std::cout << "[PASS] QSA layer graph composite"
              << " (six nodes, pos=2052..2055, blocks=513..514, bitwise,"
              << " abort=1 no-op)\n";
}

static void run_router_softmax_topk_test(Context& ctx, int tokens) {
    constexpr int num_experts = 512;
    constexpr int top_k = 10;

    std::mt19937 rng(2026 + tokens);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<bf16> scores_host(tokens * num_experts);
    for (size_t i = 0; i < scores_host.size(); ++i) {
        scores_host[i] = bf16(dist(rng));
    }

    std::vector<int> ref_ids(tokens * top_k);
    std::vector<float> ref_weights(tokens * top_k);
    ops::router_softmax_topk_host(
        scores_host.data(), nullptr, ref_ids.data(), ref_weights.data(),
        tokens, num_experts, top_k, /*norm_topk_prob=*/true);

    Tensor scores_dev = Tensor::allocate(ctx, {tokens, num_experts}, dnnl::memory::data_type::bf16);
    Tensor ids_dev = Tensor::allocate(ctx, {tokens, top_k}, dnnl::memory::data_type::s32);
    Tensor weights_dev = Tensor::allocate(ctx, {tokens, top_k}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(scores_dev.data(), scores_host.data(), scores_host.size() * sizeof(bf16));

    ops::router_softmax_topk_device(
        ctx, static_cast<const bf16*>(scores_dev.data()), nullptr,
        static_cast<int*>(ids_dev.data()),
        static_cast<float*>(weights_dev.data()),
        tokens, num_experts, top_k, /*norm_topk_prob=*/true);
    ctx.synchronize();

    std::vector<int> gpu_ids(tokens * top_k);
    std::vector<float> gpu_weights(tokens * top_k);
    ctx.memcpy_d2h(gpu_ids.data(), ids_dev.data(), gpu_ids.size() * sizeof(int));
    ctx.memcpy_d2h(gpu_weights.data(), weights_dev.data(), gpu_weights.size() * sizeof(float));

    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            size_t idx = static_cast<size_t>(t) * top_k + k;
            if (gpu_ids[idx] != ref_ids[idx]) {
                throw std::runtime_error("Router expert ID mismatch at token " + std::to_string(t) +
                                         " k=" + std::to_string(k) + ": gpu=" +
                                         std::to_string(gpu_ids[idx]) + " ref=" +
                                         std::to_string(ref_ids[idx]));
            }
        }
    }
    check_close_f32(gpu_weights, ref_weights, "router_softmax_topk_device weights tokens=" + std::to_string(tokens), 1e-4f);
}

static void run_router_topk_ids_test(Context& ctx, int tokens, int test_mode = 0) {
    constexpr int num_experts = 512;
    constexpr int top_k = 10;

    std::mt19937 rng(2027 + tokens + test_mode * 100);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<bf16> scores_host(tokens * num_experts);
    for (size_t i = 0; i < scores_host.size(); ++i) {
        if (test_mode == 1) {
            // Negative logits
            scores_host[i] = bf16(dist(rng) - 50.0f);
        } else if (test_mode == 2) {
            // Extremely negative logits that broke naive thresholding
            scores_host[i] = bf16(-1.0e35f);
        } else if (test_mode == 3) {
            // Tied logits to verify deterministic index tie-breaking
            scores_host[i] = bf16(static_cast<float>((i % 16) * 2));
        } else {
            scores_host[i] = bf16(dist(rng));
        }
    }
    if (test_mode == 2) {
        // Give top experts distinctly ordered scores
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                scores_host[t * num_experts + k * 3] = bf16(-1.0e32f + static_cast<float>(top_k - k));
            }
        }
    }

    std::vector<int> ref_ids(tokens * top_k);
    ops::router_topk_ids_host(
        scores_host.data(), ref_ids.data(), tokens, num_experts, top_k);

    Tensor scores_dev = Tensor::allocate(ctx, {tokens, num_experts}, dnnl::memory::data_type::bf16);
    Tensor ids_dev = Tensor::allocate(ctx, {tokens, top_k}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(scores_dev.data(), scores_host.data(), scores_host.size() * sizeof(bf16));

    ops::router_topk_ids_device(
        ctx, static_cast<const bf16*>(scores_dev.data()),
        static_cast<int*>(ids_dev.data()),
        tokens, num_experts, top_k);
    ctx.synchronize();

    std::vector<int> gpu_ids(tokens * top_k);
    ctx.memcpy_d2h(gpu_ids.data(), ids_dev.data(), gpu_ids.size() * sizeof(int));

    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            size_t idx = static_cast<size_t>(t) * top_k + k;
            if (gpu_ids[idx] != ref_ids[idx]) {
                throw std::runtime_error("Router topk IDs mismatch at mode=" + std::to_string(test_mode) +
                                         " token " + std::to_string(t) +
                                         " k=" + std::to_string(k) + ": gpu=" +
                                         std::to_string(gpu_ids[idx]) + " ref=" +
                                         std::to_string(ref_ids[idx]));
            }
        }
    }
}

void test_router_softmax_topk(Context& ctx) {
    run_router_softmax_topk_test(ctx, 1);
    run_router_softmax_topk_test(ctx, 4);
    // Phase E1: Test standard, negative, extreme negative, and tied logits
    run_router_topk_ids_test(ctx, 1, 0);
    run_router_topk_ids_test(ctx, 4, 0);
    run_router_topk_ids_test(ctx, 16, 0);
    run_router_topk_ids_test(ctx, 4, 1); // Negative logits
    run_router_topk_ids_test(ctx, 4, 2); // Extremely negative logits
    run_router_topk_ids_test(ctx, 4, 3); // Tied logits
}

void test_moe_decode_gate_up_and_down(Context& ctx) {
    constexpr int in_features = 2560;
    constexpr int intermediate_size = 640;
    constexpr int out_features = 2560;
    constexpr int blocksize = 64;
    constexpr int num_active_experts = 4;

    std::mt19937 rng(333);
    std::uniform_int_distribution<int> u8_dist(0, 255);
    std::uniform_real_distribution<float> f_dist(0.01f, 0.5f);
    std::uniform_real_distribution<float> in_dist(-0.2f, 0.2f);

    static constexpr float kQuantMap[16] = {
        -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
        -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
        0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
        0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
    };

    const size_t gu_packed_bytes = static_cast<size_t>(2 * intermediate_size) * (in_features / 2);
    const size_t gu_absmax_count = static_cast<size_t>(2 * intermediate_size) * (in_features / blocksize);
    const size_t down_packed_bytes = static_cast<size_t>(out_features) * (intermediate_size / 2);
    const size_t down_absmax_count = static_cast<size_t>(out_features) * (intermediate_size / blocksize);

    std::vector<std::vector<uint8_t>> gu_packed(num_active_experts, std::vector<uint8_t>(gu_packed_bytes));
    std::vector<std::vector<float>> gu_absmax(num_active_experts, std::vector<float>(gu_absmax_count));
    std::vector<std::vector<uint8_t>> down_packed(num_active_experts, std::vector<uint8_t>(down_packed_bytes));
    std::vector<std::vector<float>> down_absmax(num_active_experts, std::vector<float>(down_absmax_count));

    for (int e = 0; e < num_active_experts; ++e) {
        for (size_t i = 0; i < gu_packed_bytes; ++i) gu_packed[e][i] = static_cast<uint8_t>(u8_dist(rng));
        for (size_t i = 0; i < gu_absmax_count; ++i) gu_absmax[e][i] = f_dist(rng);
        for (size_t i = 0; i < down_packed_bytes; ++i) down_packed[e][i] = static_cast<uint8_t>(u8_dist(rng));
        for (size_t i = 0; i < down_absmax_count; ++i) down_absmax[e][i] = f_dist(rng);
    }

    std::vector<bf16> x_host(in_features);
    for (int i = 0; i < in_features; ++i) x_host[i] = bf16(in_dist(rng));

    std::vector<float> weights_host(num_active_experts);
    for (int e = 0; e < num_active_experts; ++e) weights_host[e] = 1.0f / num_active_experts;

    // CPU reference for gate_up
    std::vector<float> ref_h(num_active_experts * intermediate_size);
    for (int e = 0; e < num_active_experts; ++e) {
        for (int r = 0; r < intermediate_size; ++r) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            for (int d = 0; d < in_features; ++d) {
                int byte_idx = d / 2;
                int nibble = (d % 2 == 0) ? (gu_packed[e][r * (in_features / 2) + byte_idx] >> 4)
                                          : (gu_packed[e][r * (in_features / 2) + byte_idx] & 0xF);
                float w_gate = kQuantMap[nibble] * gu_absmax[e][r * (in_features / blocksize) + (d / blocksize)];
                gate_sum += static_cast<float>(x_host[d]) * w_gate;

                int up_row = r + intermediate_size;
                int up_nibble = (d % 2 == 0) ? (gu_packed[e][up_row * (in_features / 2) + byte_idx] >> 4)
                                             : (gu_packed[e][up_row * (in_features / 2) + byte_idx] & 0xF);
                float w_up = kQuantMap[up_nibble] * gu_absmax[e][up_row * (in_features / blocksize) + (d / blocksize)];
                up_sum += static_cast<float>(x_host[d]) * w_up;
            }
            float silu_g = gate_sum / (1.0f + std::exp(-gate_sum));
            ref_h[e * intermediate_size + r] = silu_g * up_sum;
        }
    }

    // CPU reference for down weighted accumulate
    std::vector<float> ref_out(out_features, 0.0f);
    for (int e = 0; e < num_active_experts; ++e) {
        float w_e = weights_host[e];
        for (int row = 0; row < out_features; ++row) {
            float acc = 0.0f;
            for (int d = 0; d < intermediate_size; ++d) {
                int byte_idx = d / 2;
                int nibble = (d % 2 == 0) ? (down_packed[e][row * (intermediate_size / 2) + byte_idx] >> 4)
                                          : (down_packed[e][row * (intermediate_size / 2) + byte_idx] & 0xF);
                float w_down = kQuantMap[nibble] * down_absmax[e][row * (intermediate_size / blocksize) + (d / blocksize)];
                acc += ref_h[e * intermediate_size + d] * w_down;
            }
            ref_out[row] += w_e * acc;
        }
    }

    // Allocate GPU tensors and slot views
    std::vector<Tensor> gu_packed_dev, gu_absmax_dev, down_packed_dev, down_absmax_dev;
    std::vector<ops::MoeExpertSlotDev> slots(num_active_experts);
    for (int e = 0; e < num_active_experts; ++e) {
        gu_packed_dev.push_back(Tensor::allocate(ctx, {gu_packed_bytes}, dnnl::memory::data_type::u8));
        gu_absmax_dev.push_back(Tensor::allocate(ctx, {gu_absmax_count}, dnnl::memory::data_type::f32));
        down_packed_dev.push_back(Tensor::allocate(ctx, {down_packed_bytes}, dnnl::memory::data_type::u8));
        down_absmax_dev.push_back(Tensor::allocate(ctx, {down_absmax_count}, dnnl::memory::data_type::f32));

        ctx.memcpy_h2d(gu_packed_dev[e].data(), gu_packed[e].data(), gu_packed_bytes);
        ctx.memcpy_h2d(gu_absmax_dev[e].data(), gu_absmax[e].data(), gu_absmax_count * sizeof(float));
        ctx.memcpy_h2d(down_packed_dev[e].data(), down_packed[e].data(), down_packed_bytes);
        ctx.memcpy_h2d(down_absmax_dev[e].data(), down_absmax[e].data(), down_absmax_count * sizeof(float));

        slots[e].gate_up_packed = static_cast<const uint8_t*>(gu_packed_dev[e].data());
        slots[e].gate_up_absmax = static_cast<const float*>(gu_absmax_dev[e].data());
        slots[e].down_packed = static_cast<const uint8_t*>(down_packed_dev[e].data());
        slots[e].down_absmax = static_cast<const float*>(down_absmax_dev[e].data());
    }

    Tensor x_dev = Tensor::allocate(ctx, {in_features}, dnnl::memory::data_type::bf16);
    Tensor h_dev = Tensor::allocate(ctx, {num_active_experts * intermediate_size}, dnnl::memory::data_type::bf16);
    Tensor out_dev = Tensor::allocate(ctx, {out_features}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_dev.data(), x_host.data(), in_features * sizeof(bf16));

    ops::moe_decode_gate_up_swiglu_grouped(
        ctx, slots.data(), nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(h_dev.data()),
        num_active_experts, in_features, intermediate_size, blocksize);
    ctx.synchronize();

    std::vector<bf16> gpu_h(num_active_experts * intermediate_size);
    ctx.memcpy_d2h(gpu_h.data(), h_dev.data(), gpu_h.size() * sizeof(bf16));
    check_close(gpu_h, ref_h, "moe_decode_gate_up_swiglu_grouped", 2e-2f);

    // Verify hardware event dependency: submit barrier, pass as dep_event
    sycl::event dummy_ev = ctx.queue().ext_oneapi_submit_barrier();
    ops::moe_decode_gate_up_swiglu_grouped(
        ctx, slots.data(), nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(h_dev.data()),
        num_active_experts, in_features, intermediate_size, blocksize,
        dummy_ev);
    ctx.synchronize();
    ctx.memcpy_d2h(gpu_h.data(), h_dev.data(), gpu_h.size() * sizeof(bf16));
    check_close(gpu_h, ref_h, "moe_decode_gate_up_swiglu_grouped with dep_event", 2e-2f);

    // Verify true cross-queue hardware event dependency from separate in-order queue
    sycl::queue copy_q(ctx.queue().get_context(), ctx.queue().get_device(),
                       sycl::property::queue::in_order());
    sycl::event cross_q_ev = copy_q.ext_oneapi_submit_barrier();
    ops::moe_decode_gate_up_swiglu_grouped(
        ctx, slots.data(), nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(h_dev.data()),
        num_active_experts, in_features, intermediate_size, blocksize,
        cross_q_ev);
    ctx.synchronize();
    ctx.memcpy_d2h(gpu_h.data(), h_dev.data(), gpu_h.size() * sizeof(bf16));
    check_close(gpu_h, ref_h, "moe_decode_gate_up_swiglu_grouped with cross_q dep_event", 2e-2f);

    // Verify empty/default-constructed event does not throw and completes correctly
    sycl::event empty_ev{};
    ops::moe_decode_gate_up_swiglu_grouped(
        ctx, slots.data(), nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(h_dev.data()),
        num_active_experts, in_features, intermediate_size, blocksize,
        empty_ev);
    ctx.synchronize();
    ctx.memcpy_d2h(gpu_h.data(), h_dev.data(), gpu_h.size() * sizeof(bf16));
    check_close(gpu_h, ref_h, "moe_decode_gate_up_swiglu_grouped with empty dep_event", 2e-2f);

    ops::moe_decode_down_weighted_accumulate(
        ctx, slots.data(), nullptr, weights_host.data(),
        static_cast<const bf16*>(h_dev.data()), nullptr,
        static_cast<bf16*>(out_dev.data()),
        num_active_experts, intermediate_size, out_features, blocksize);
    ctx.synchronize();

    std::vector<bf16> gpu_out(out_features);
    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), out_features * sizeof(bf16));
    check_close(gpu_out, ref_out, "moe_decode_down_weighted_accumulate", 2e-2f);

    // Fused shared-expert addition must preserve the bf16 rounding point of
    // the previous two-kernel sequence: first route, then add gated shared.
    std::vector<bf16> shared_host(out_features);
    for (bf16& value : shared_host) value = bf16(in_dist(rng));
    const bf16 shared_gate_host(0.3f);
    Tensor shared_dev = Tensor::allocate(ctx, {out_features}, dnnl::memory::data_type::bf16);
    Tensor gate_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(shared_dev.data(), shared_host.data(), out_features * sizeof(bf16));
    ctx.memcpy_h2d(gate_dev.data(), &shared_gate_host, sizeof(bf16));
    const float gate = static_cast<float>(shared_gate_host);
    const float gate_act = 1.0f / (1.0f + std::exp(-gate));
    std::vector<float> ref_fused(out_features);
    for (int i = 0; i < out_features; ++i) {
        ref_fused[i] = static_cast<float>(bf16(
            static_cast<float>(gpu_out[i]) + gate_act * static_cast<float>(shared_host[i])));
    }
    Tensor separate_dev = Tensor::allocate(ctx, {out_features}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(separate_dev.data(), gpu_out.data(), out_features * sizeof(bf16));
    auto* separate_ptr = static_cast<bf16*>(separate_dev.data());
    const auto* shared_ptr = static_cast<const bf16*>(shared_dev.data());
    const auto* gate_ptr = static_cast<const bf16*>(gate_dev.data());
    ctx.queue().parallel_for(sycl::range<1>(out_features), [=](sycl::id<1> id) {
        const float gate_value = static_cast<float>(gate_ptr[0]);
        const float sigmoid = 1.0f / (1.0f + sycl::native::exp(-gate_value));
        separate_ptr[id] = bf16(static_cast<float>(separate_ptr[id]) +
                                sigmoid * static_cast<float>(shared_ptr[id]));
    });
    ctx.synchronize();
    std::vector<bf16> separate_host(out_features);
    ctx.memcpy_d2h(separate_host.data(), separate_dev.data(), out_features * sizeof(bf16));
    ops::moe_decode_down_weighted_accumulate(
        ctx, slots.data(), nullptr, weights_host.data(),
        static_cast<const bf16*>(h_dev.data()), nullptr,
        static_cast<bf16*>(out_dev.data()),
        num_active_experts, intermediate_size, out_features, blocksize,
        static_cast<const bf16*>(shared_dev.data()),
        static_cast<const bf16*>(gate_dev.data()));
    ctx.synchronize();
    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), out_features * sizeof(bf16));
    check_close(gpu_out, ref_fused, "moe_decode_down_fused_shared", 2e-2f);
    int exact_mismatches = 0;
    float max_separate_diff = 0.0f;
    for (int i = 0; i < out_features; ++i) {
        if (gpu_out[i] != separate_host[i]) {
            ++exact_mismatches;
            max_separate_diff = std::max(max_separate_diff,
                std::abs(static_cast<float>(gpu_out[i]) - static_cast<float>(separate_host[i])));
        }
    }
    if (exact_mismatches != 0) {
        throw std::runtime_error("fused shared add differs from two-kernel result: " +
                                 std::to_string(exact_mismatches) + " values, max abs " +
                                 std::to_string(max_separate_diff));
    }
}

static constexpr float kTestQuantMap[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

static void run_moe_prefill_case(Context& ctx, int tokens, int num_active_experts, bool use_nested) {
    constexpr int in_features = 2560;
    constexpr int intermediate_size = 640;
    constexpr int out_features = 2560;
    constexpr int blocksize = 64;
    constexpr int nested_blocksize = 256;

    std::mt19937 rng(444 + tokens * 100 + num_active_experts * 10 + (use_nested ? 1 : 0));
    std::uniform_int_distribution<int> u8_dist(0, 255);
    std::uniform_int_distribution<int> nibble_dist(0, 15);
    std::uniform_real_distribution<float> f_dist(0.01f, 0.5f);
    std::uniform_real_distribution<float> in_dist(-0.2f, 0.2f);

    const size_t gu_packed_bytes = static_cast<size_t>(2 * intermediate_size) * (in_features / 2);
    const size_t gu_absmax_count = static_cast<size_t>(2 * intermediate_size) * (in_features / blocksize);
    const size_t down_packed_bytes = static_cast<size_t>(out_features) * (intermediate_size / 2);
    const size_t down_absmax_count = static_cast<size_t>(out_features) * (intermediate_size / blocksize);

    const int num_unique_experts = std::max(16, num_active_experts);
    std::vector<Tensor> gu_packed_dev, gu_absmax_dev, down_packed_dev, down_absmax_dev;
    std::vector<Tensor> gu_codes_dev, gu_nested_absmax_dev, down_codes_dev, down_nested_absmax_dev;
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(qmap_dev.data(), kTestQuantMap, 16 * sizeof(float));

    std::vector<ops::MoeExpertSlotDev> expert_pool(num_unique_experts);
    for (int e = 0; e < num_unique_experts; ++e) {
        gu_packed_dev.push_back(Tensor::allocate(ctx, {gu_packed_bytes}, dnnl::memory::data_type::u8));
        gu_absmax_dev.push_back(Tensor::allocate(ctx, {gu_absmax_count}, dnnl::memory::data_type::f32));
        down_packed_dev.push_back(Tensor::allocate(ctx, {down_packed_bytes}, dnnl::memory::data_type::u8));
        down_absmax_dev.push_back(Tensor::allocate(ctx, {down_absmax_count}, dnnl::memory::data_type::f32));

        std::vector<uint8_t> gu_p(gu_packed_bytes), dn_p(down_packed_bytes);
        std::vector<float> gu_a(gu_absmax_count), dn_a(down_absmax_count);
        for (auto& v : gu_p) v = static_cast<uint8_t>(u8_dist(rng));
        for (auto& v : dn_p) v = static_cast<uint8_t>(u8_dist(rng));
        for (auto& v : gu_a) v = f_dist(rng);
        for (auto& v : dn_a) v = f_dist(rng);

        ctx.memcpy_h2d(gu_packed_dev[e].data(), gu_p.data(), gu_packed_bytes);
        ctx.memcpy_h2d(gu_absmax_dev[e].data(), gu_a.data(), gu_absmax_count * sizeof(float));
        ctx.memcpy_h2d(down_packed_dev[e].data(), dn_p.data(), down_packed_bytes);
        ctx.memcpy_h2d(down_absmax_dev[e].data(), dn_a.data(), down_absmax_count * sizeof(float));

        expert_pool[e].gate_up_packed = static_cast<const uint8_t*>(gu_packed_dev[e].data());
        expert_pool[e].gate_up_absmax = static_cast<const float*>(gu_absmax_dev[e].data());
        expert_pool[e].down_packed = static_cast<const uint8_t*>(down_packed_dev[e].data());
        expert_pool[e].down_absmax = static_cast<const float*>(down_absmax_dev[e].data());

        if (use_nested) {
            const size_t gu_nested_count = (gu_absmax_count + nested_blocksize - 1) / nested_blocksize;
            const size_t dn_nested_count = (down_absmax_count + nested_blocksize - 1) / nested_blocksize;

            gu_codes_dev.push_back(Tensor::allocate(ctx, {gu_absmax_count}, dnnl::memory::data_type::u8));
            gu_nested_absmax_dev.push_back(Tensor::allocate(ctx, {gu_nested_count}, dnnl::memory::data_type::f32));
            down_codes_dev.push_back(Tensor::allocate(ctx, {down_absmax_count}, dnnl::memory::data_type::u8));
            down_nested_absmax_dev.push_back(Tensor::allocate(ctx, {dn_nested_count}, dnnl::memory::data_type::f32));

            std::vector<uint8_t> gu_c(gu_absmax_count), dn_c(down_absmax_count);
            std::vector<float> gu_na(gu_nested_count), dn_na(dn_nested_count);
            for (auto& v : gu_c) v = static_cast<uint8_t>(nibble_dist(rng));
            for (auto& v : dn_c) v = static_cast<uint8_t>(nibble_dist(rng));
            for (auto& v : gu_na) v = f_dist(rng);
            for (auto& v : dn_na) v = f_dist(rng);

            ctx.memcpy_h2d(gu_codes_dev[e].data(), gu_c.data(), gu_absmax_count);
            ctx.memcpy_h2d(gu_nested_absmax_dev[e].data(), gu_na.data(), gu_nested_count * sizeof(float));
            ctx.memcpy_h2d(down_codes_dev[e].data(), dn_c.data(), down_absmax_count);
            ctx.memcpy_h2d(down_nested_absmax_dev[e].data(), dn_na.data(), dn_nested_count * sizeof(float));

            expert_pool[e].gate_up_nested.codes = static_cast<const uint8_t*>(gu_codes_dev[e].data());
            expert_pool[e].gate_up_nested.nested_absmax = static_cast<const float*>(gu_nested_absmax_dev[e].data());
            expert_pool[e].gate_up_nested.nested_quant_map = static_cast<const float*>(qmap_dev.data());
            expert_pool[e].gate_up_nested.offset = 0.0f;
            expert_pool[e].gate_up_nested.blocksize = nested_blocksize;

            expert_pool[e].down_nested.codes = static_cast<const uint8_t*>(down_codes_dev[e].data());
            expert_pool[e].down_nested.nested_absmax = static_cast<const float*>(down_nested_absmax_dev[e].data());
            expert_pool[e].down_nested.nested_quant_map = static_cast<const float*>(qmap_dev.data());
            expert_pool[e].down_nested.offset = 0.0f;
            expert_pool[e].down_nested.blocksize = nested_blocksize;
        }
    }

    const int total_slots = tokens * num_active_experts;
    std::vector<ops::MoeExpertSlotDev> slots(total_slots);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < num_active_experts; ++k) {
            int exp_id = (t * 7 + k) % num_unique_experts;
            slots[t * num_active_experts + k] = expert_pool[exp_id];
        }
    }

    std::vector<bf16> x_host(tokens * in_features);
    for (auto& v : x_host) v = bf16(in_dist(rng));

    std::vector<float> weights_host(total_slots);
    for (int t = 0; t < tokens; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < num_active_experts; ++k) {
            float w = f_dist(rng);
            weights_host[t * num_active_experts + k] = w;
            sum += w;
        }
        for (int k = 0; k < num_active_experts; ++k) {
            weights_host[t * num_active_experts + k] /= sum;
        }
    }

    std::vector<bf16> shared_host(tokens * out_features);
    std::vector<bf16> gate_host(tokens);
    for (auto& v : shared_host) v = bf16(in_dist(rng));
    for (int t = 0; t < tokens; ++t) gate_host[t] = bf16(in_dist(rng));

    Tensor x_dev = Tensor::allocate(ctx, {tokens, in_features}, dnnl::memory::data_type::bf16);
    Tensor shared_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);
    Tensor gate_dev = Tensor::allocate(ctx, {tokens, 1}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(shared_dev.data(), shared_host.data(), shared_host.size() * sizeof(bf16));
    ctx.memcpy_h2d(gate_dev.data(), gate_host.data(), gate_host.size() * sizeof(bf16));

    auto* slots_dev_buf = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(total_slots * sizeof(ops::MoeExpertSlotDev)));
    ctx.memcpy_h2d(slots_dev_buf, slots.data(), total_slots * sizeof(ops::MoeExpertSlotDev));

    auto* weights_dev = static_cast<float*>(
        ctx.alloc_device(total_slots * sizeof(float)));
    ctx.memcpy_h2d(weights_dev, weights_host.data(), total_slots * sizeof(float));

    // 1. Reference using decode kernel per token
    Tensor ref_h_dev = Tensor::allocate(ctx, {tokens * num_active_experts * intermediate_size}, dnnl::memory::data_type::bf16);
    Tensor ref_y_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);
    Tensor ref_y_fused_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

    for (int t = 0; t < tokens; ++t) {
        const auto* x_t = static_cast<const bf16*>(x_dev.data()) + t * in_features;
        auto* h_t = static_cast<bf16*>(ref_h_dev.data()) + t * num_active_experts * intermediate_size;
        auto* y_t = static_cast<bf16*>(ref_y_dev.data()) + t * out_features;
        auto* y_fused_t = static_cast<bf16*>(ref_y_fused_dev.data()) + t * out_features;
        const auto* t_slots = slots.data() + t * num_active_experts;
        const auto* t_weights = weights_host.data() + t * num_active_experts;

        ops::moe_decode_gate_up_swiglu_grouped(
            ctx, t_slots, nullptr, x_t, h_t,
            num_active_experts, in_features, intermediate_size, blocksize);

        ops::moe_decode_down_weighted_accumulate(
            ctx, t_slots, nullptr, t_weights,
            h_t, nullptr, y_t,
            num_active_experts, intermediate_size, out_features, blocksize);

        ops::moe_decode_down_weighted_accumulate(
            ctx, t_slots, nullptr, t_weights,
            h_t, nullptr, y_fused_t,
            num_active_experts, intermediate_size, out_features, blocksize,
            static_cast<const bf16*>(shared_dev.data()) + t * out_features,
            static_cast<const bf16*>(gate_dev.data()) + t);
    }
    ctx.synchronize();

    std::vector<bf16> ref_h(tokens * num_active_experts * intermediate_size);
    std::vector<bf16> ref_y(tokens * out_features);
    std::vector<bf16> ref_y_fused(tokens * out_features);
    ctx.memcpy_d2h(ref_h.data(), ref_h_dev.data(), ref_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(ref_y.data(), ref_y_dev.data(), ref_y.size() * sizeof(bf16));
    ctx.memcpy_d2h(ref_y_fused.data(), ref_y_fused_dev.data(), ref_y_fused.size() * sizeof(bf16));

    // 2. Batched prefill kernels (with device weights)
    Tensor prefill_h_dev = Tensor::allocate(ctx, {tokens * num_active_experts * intermediate_size}, dnnl::memory::data_type::bf16);
    Tensor prefill_y_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);
    Tensor prefill_y_fused_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

    ops::moe_prefill_gate_up_rows(
        ctx, slots_dev_buf, nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(prefill_h_dev.data()),
        tokens, num_active_experts, in_features, intermediate_size, blocksize,
        nullptr, use_nested);

    ops::moe_prefill_down_rows(
        ctx, slots_dev_buf, weights_dev, nullptr,
        static_cast<const bf16*>(prefill_h_dev.data()),
        static_cast<bf16*>(prefill_y_dev.data()),
        tokens, num_active_experts, intermediate_size, out_features, blocksize,
        nullptr, nullptr, nullptr, use_nested);

    ops::moe_prefill_down_rows(
        ctx, slots_dev_buf, weights_dev, nullptr,
        static_cast<const bf16*>(prefill_h_dev.data()),
        static_cast<bf16*>(prefill_y_fused_dev.data()),
        tokens, num_active_experts, intermediate_size, out_features, blocksize,
        static_cast<const bf16*>(shared_dev.data()),
        static_cast<const bf16*>(gate_dev.data()),
        nullptr, use_nested);

    ctx.synchronize();

    std::vector<bf16> prefill_h(tokens * num_active_experts * intermediate_size);
    std::vector<bf16> prefill_y(tokens * out_features);
    std::vector<bf16> prefill_y_fused(tokens * out_features);
    ctx.memcpy_d2h(prefill_h.data(), prefill_h_dev.data(), prefill_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(prefill_y.data(), prefill_y_dev.data(), prefill_y.size() * sizeof(bf16));
    ctx.memcpy_d2h(prefill_y_fused.data(), prefill_y_fused_dev.data(), prefill_y_fused.size() * sizeof(bf16));

    // 3. Verify bit-exact match against per-token decode kernels
    int h_mismatches = 0;
    for (size_t i = 0; i < ref_h.size(); ++i) {
        if (prefill_h[i] != ref_h[i]) ++h_mismatches;
    }
    if (h_mismatches != 0) {
        throw std::runtime_error("moe_prefill_gate_up_rows bit-exact mismatch: " +
                                 std::to_string(h_mismatches) + " of " + std::to_string(ref_h.size()));
    }

    int y_mismatches = 0;
    for (size_t i = 0; i < ref_y.size(); ++i) {
        if (prefill_y[i] != ref_y[i]) ++y_mismatches;
    }
    if (y_mismatches != 0) {
        throw std::runtime_error("moe_prefill_down_rows bit-exact mismatch: " +
                                 std::to_string(y_mismatches) + " of " + std::to_string(ref_y.size()));
    }

    int y_fused_mismatches = 0;
    for (size_t i = 0; i < ref_y_fused.size(); ++i) {
        if (prefill_y_fused[i] != ref_y_fused[i]) ++y_fused_mismatches;
    }
    if (y_fused_mismatches != 0) {
        throw std::runtime_error("moe_prefill_down_rows (fused) bit-exact mismatch: " +
                                 std::to_string(y_fused_mismatches) + " of " + std::to_string(ref_y_fused.size()));
    }

    // 4. Expert-major prefill kernels (Phase S4)
    // Construct true expert-major bucketing matching Qwen4ExpBnb4Backend:
    const int total_acts = tokens * num_active_experts;
    std::vector<int> em_tokens(total_acts);
    std::vector<int> em_ranks(total_acts);
    std::vector<ops::MoeExpertSlotDev> em_slots(total_acts);

    std::vector<int> expert_counts(num_unique_experts, 0);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < num_active_experts; ++k) {
            int e = (t * 7 + k) % num_unique_experts;
            expert_counts[e]++;
        }
    }
    std::vector<int> expert_offsets(num_unique_experts + 1, 0);
    for (int e = 0; e < num_unique_experts; ++e) {
        expert_offsets[e + 1] = expert_offsets[e] + expert_counts[e];
    }
    std::vector<int> expert_pos = expert_offsets;
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < num_active_experts; ++k) {
            int e = (t * 7 + k) % num_unique_experts;
            int pos = expert_pos[e]++;
            em_tokens[pos] = t;
            em_ranks[pos] = k;
            em_slots[pos] = expert_pool[e];
        }
    }

    Tensor em_tokens_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
    Tensor em_ranks_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
    ops::MoeExpertSlotDev* em_slots_dev = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(total_acts * sizeof(ops::MoeExpertSlotDev)));
    Tensor em_h_dev = Tensor::allocate(ctx, {total_acts * intermediate_size}, dnnl::memory::data_type::bf16);
    Tensor em_down_acc_dev = Tensor::allocate(ctx, {total_acts * out_features}, dnnl::memory::data_type::f32);
    Tensor em_y_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);
    Tensor em_y_fused_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(em_tokens_dev.data(), em_tokens.data(), total_acts * sizeof(int));
    ctx.memcpy_h2d(em_ranks_dev.data(), em_ranks.data(), total_acts * sizeof(int));
    ctx.memcpy_h2d(em_slots_dev, em_slots.data(), total_acts * sizeof(ops::MoeExpertSlotDev));

    sycl::event em_gu = ops::moe_expert_major_prefill_gate_up(
        ctx, em_slots_dev,
        static_cast<const int*>(em_tokens_dev.data()),
        static_cast<const int*>(em_ranks_dev.data()),
        nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(em_h_dev.data()),
        total_acts, num_active_experts, in_features, intermediate_size, blocksize,
        nullptr, use_nested);

    sycl::event em_dn = ops::moe_expert_major_prefill_down(
        ctx, em_slots_dev,
        static_cast<const int*>(em_tokens_dev.data()),
        static_cast<const int*>(em_ranks_dev.data()),
        nullptr,
        static_cast<const bf16*>(em_h_dev.data()),
        static_cast<float*>(em_down_acc_dev.data()),
        total_acts, num_active_experts, intermediate_size, out_features, blocksize,
        &em_gu, use_nested);

    ops::moe_ordered_accumulate_down_rows(
        ctx, static_cast<const float*>(em_down_acc_dev.data()),
        weights_dev,
        static_cast<bf16*>(em_y_dev.data()),
        tokens, num_active_experts, out_features,
        nullptr, nullptr, &em_dn);

    ops::moe_ordered_accumulate_down_rows(
        ctx, static_cast<const float*>(em_down_acc_dev.data()),
        weights_dev,
        static_cast<bf16*>(em_y_fused_dev.data()),
        tokens, num_active_experts, out_features,
        static_cast<const bf16*>(shared_dev.data()),
        static_cast<const bf16*>(gate_dev.data()),
        &em_dn);

    ctx.synchronize();

    std::vector<bf16> em_h(total_acts * intermediate_size);
    std::vector<bf16> em_y(tokens * out_features);
    std::vector<bf16> em_y_fused(tokens * out_features);
    ctx.memcpy_d2h(em_h.data(), em_h_dev.data(), em_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(em_y.data(), em_y_dev.data(), em_y.size() * sizeof(bf16));
    ctx.memcpy_d2h(em_y_fused.data(), em_y_fused_dev.data(), em_y_fused.size() * sizeof(bf16));

    for (size_t i = 0; i < ref_h.size(); ++i) {
        if (em_h[i] != ref_h[i]) {
            throw std::runtime_error("moe_expert_major_prefill_gate_up bit-exact mismatch at index " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < ref_y.size(); ++i) {
        if (em_y[i] != ref_y[i]) {
            throw std::runtime_error("moe_expert_major_prefill_down bit-exact mismatch at index " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < ref_y_fused.size(); ++i) {
        if (em_y_fused[i] != ref_y_fused[i]) {
            throw std::runtime_error("moe_expert_major_prefill_down (fused) bit-exact mismatch at index " + std::to_string(i));
        }
    }
    ctx.free_device(em_slots_dev);

    // 5. Expert-major prefill with explicit 2-token weight reuse (Phase W4 C1/C2)
    int total_pairs = 0;
    for (int e = 0; e < num_unique_experts; ++e) {
        total_pairs += expert_counts[e] / 2;
    }
    std::vector<int> wr_pair_pos(num_unique_experts, 0);
    std::vector<int> wr_single_pos(num_unique_experts, 0);
    int p_off = 0;
    int s_off = 2 * total_pairs;
    for (int e = 0; e < num_unique_experts; ++e) {
        wr_pair_pos[e] = p_off;
        wr_single_pos[e] = s_off;
        p_off += (expert_counts[e] / 2) * 2;
        s_off += (expert_counts[e] % 2);
    }
    std::vector<int> wr_tokens(total_acts);
    std::vector<int> wr_ranks(total_acts);
    std::vector<ops::MoeExpertSlotDev> wr_slots(total_acts);
    std::vector<int> wr_expert_seen(num_unique_experts, 0);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < num_active_experts; ++k) {
            int e = (t * 7 + k) % num_unique_experts;
            int seen = wr_expert_seen[e]++;
            int pairs_limit = (expert_counts[e] / 2) * 2;
            int pos = (seen < pairs_limit)
                ? (wr_pair_pos[e] + seen)
                : (wr_single_pos[e] + (seen - pairs_limit));
            wr_tokens[pos] = t;
            wr_ranks[pos] = k;
            wr_slots[pos] = expert_pool[e];
        }
    }

    Tensor wr_tokens_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
    Tensor wr_ranks_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
    ops::MoeExpertSlotDev* wr_slots_dev = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(total_acts * sizeof(ops::MoeExpertSlotDev)));
    Tensor wr_h_dev = Tensor::allocate(ctx, {total_acts * intermediate_size}, dnnl::memory::data_type::bf16);
    Tensor wr_down_acc_dev = Tensor::allocate(ctx, {total_acts * out_features}, dnnl::memory::data_type::f32);
    Tensor wr_y_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);
    Tensor wr_y_fused_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(wr_tokens_dev.data(), wr_tokens.data(), total_acts * sizeof(int));
    ctx.memcpy_h2d(wr_ranks_dev.data(), wr_ranks.data(), total_acts * sizeof(int));
    ctx.memcpy_h2d(wr_slots_dev, wr_slots.data(), total_acts * sizeof(ops::MoeExpertSlotDev));

    sycl::event wr_gu = ops::moe_expert_major_prefill_gate_up(
        ctx, wr_slots_dev,
        static_cast<const int*>(wr_tokens_dev.data()),
        static_cast<const int*>(wr_ranks_dev.data()),
        nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(wr_h_dev.data()),
        total_acts, num_active_experts, in_features, intermediate_size, blocksize,
        nullptr, use_nested, total_pairs);

    sycl::event wr_dn = ops::moe_expert_major_prefill_down(
        ctx, wr_slots_dev,
        static_cast<const int*>(wr_tokens_dev.data()),
        static_cast<const int*>(wr_ranks_dev.data()),
        nullptr,
        static_cast<const bf16*>(wr_h_dev.data()),
        static_cast<float*>(wr_down_acc_dev.data()),
        total_acts, num_active_experts, intermediate_size, out_features, blocksize,
        &wr_gu, use_nested, total_pairs);

    ops::moe_ordered_accumulate_down_rows(
        ctx, static_cast<const float*>(wr_down_acc_dev.data()),
        weights_dev,
        static_cast<bf16*>(wr_y_dev.data()),
        tokens, num_active_experts, out_features,
        nullptr, nullptr, &wr_dn);

    ops::moe_ordered_accumulate_down_rows(
        ctx, static_cast<const float*>(wr_down_acc_dev.data()),
        weights_dev,
        static_cast<bf16*>(wr_y_fused_dev.data()),
        tokens, num_active_experts, out_features,
        static_cast<const bf16*>(shared_dev.data()),
        static_cast<const bf16*>(gate_dev.data()),
        &wr_dn);

    ctx.synchronize();

    std::vector<bf16> wr_h(total_acts * intermediate_size);
    std::vector<bf16> wr_y(tokens * out_features);
    std::vector<bf16> wr_y_fused(tokens * out_features);
    ctx.memcpy_d2h(wr_h.data(), wr_h_dev.data(), wr_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(wr_y.data(), wr_y_dev.data(), wr_y.size() * sizeof(bf16));
    ctx.memcpy_d2h(wr_y_fused.data(), wr_y_fused_dev.data(), wr_y_fused.size() * sizeof(bf16));

    for (size_t i = 0; i < ref_h.size(); ++i) {
        if (wr_h[i] != ref_h[i]) {
            throw std::runtime_error("moe_expert_major_prefill_gate_up (2-token reuse) bit-exact mismatch at index " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < ref_y.size(); ++i) {
        if (wr_y[i] != ref_y[i]) {
            throw std::runtime_error("moe_expert_major_prefill_down (2-token reuse) bit-exact mismatch at index " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < ref_y_fused.size(); ++i) {
        if (wr_y_fused[i] != ref_y_fused[i]) {
            throw std::runtime_error("moe_expert_major_prefill_down (2-token reuse fused) bit-exact mismatch at index " + std::to_string(i));
        }
    }
    ctx.free_device(wr_slots_dev);

    // 6. Expert-major prefill with 4-token weight reuse (Phase G3)
    int total_quads = 0;
    int total_pairs_g3 = 0;
    for (int e = 0; e < num_unique_experts; ++e) {
        int cnt = expert_counts[e];
        total_quads += cnt / 4;
        total_pairs_g3 += (cnt % 4) / 2;
    }
    std::vector<int> g3_quad_pos(num_unique_experts, 0);
    std::vector<int> g3_pair_pos(num_unique_experts, 0);
    std::vector<int> g3_single_pos(num_unique_experts, 0);
    int q_off = 0;
    int p_off_g3 = 4 * total_quads;
    int s_off_g3 = 4 * total_quads + 2 * total_pairs_g3;
    for (int e = 0; e < num_unique_experts; ++e) {
        int cnt = expert_counts[e];
        g3_quad_pos[e] = q_off;
        g3_pair_pos[e] = p_off_g3;
        g3_single_pos[e] = s_off_g3;
        q_off += (cnt / 4) * 4;
        p_off_g3 += ((cnt % 4) / 2) * 2;
        s_off_g3 += (cnt % 2);
    }
    std::vector<int> g3_tokens(total_acts);
    std::vector<int> g3_ranks(total_acts);
    std::vector<ops::MoeExpertSlotDev> g3_slots(total_acts);
    std::vector<int> g3_expert_seen(num_unique_experts, 0);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < num_active_experts; ++k) {
            int e = (t * 7 + k) % num_unique_experts;
            int seen = g3_expert_seen[e]++;
            int cnt = expert_counts[e];
            int q_limit = (cnt / 4) * 4;
            int p_limit = q_limit + ((cnt % 4) / 2) * 2;
            int pos;
            if (seen < q_limit) {
                pos = g3_quad_pos[e] + seen;
            } else if (seen < p_limit) {
                pos = g3_pair_pos[e] + (seen - q_limit);
            } else {
                pos = g3_single_pos[e] + (seen - p_limit);
            }
            g3_tokens[pos] = t;
            g3_ranks[pos] = k;
            g3_slots[pos] = expert_pool[e];
        }
    }

    Tensor g3_tokens_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
    Tensor g3_ranks_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
    ops::MoeExpertSlotDev* g3_slots_dev = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(total_acts * sizeof(ops::MoeExpertSlotDev)));
    Tensor g3_h_dev = Tensor::allocate(ctx, {total_acts * intermediate_size}, dnnl::memory::data_type::bf16);
    Tensor g3_down_acc_dev = Tensor::allocate(ctx, {total_acts * out_features}, dnnl::memory::data_type::f32);
    Tensor g3_y_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);
    Tensor g3_y_fused_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(g3_tokens_dev.data(), g3_tokens.data(), total_acts * sizeof(int));
    ctx.memcpy_h2d(g3_ranks_dev.data(), g3_ranks.data(), total_acts * sizeof(int));
    ctx.memcpy_h2d(g3_slots_dev, g3_slots.data(), total_acts * sizeof(ops::MoeExpertSlotDev));

    sycl::event g3_gu = ops::moe_expert_major_prefill_gate_up(
        ctx, g3_slots_dev,
        static_cast<const int*>(g3_tokens_dev.data()),
        static_cast<const int*>(g3_ranks_dev.data()),
        nullptr,
        static_cast<const bf16*>(x_dev.data()),
        static_cast<bf16*>(g3_h_dev.data()),
        total_acts, num_active_experts, in_features, intermediate_size, blocksize,
        nullptr, use_nested, total_pairs_g3, total_quads);

    sycl::event g3_dn = ops::moe_expert_major_prefill_down(
        ctx, g3_slots_dev,
        static_cast<const int*>(g3_tokens_dev.data()),
        static_cast<const int*>(g3_ranks_dev.data()),
        nullptr,
        static_cast<const bf16*>(g3_h_dev.data()),
        static_cast<float*>(g3_down_acc_dev.data()),
        total_acts, num_active_experts, intermediate_size, out_features, blocksize,
        &g3_gu, use_nested, total_pairs_g3, total_quads);

    ops::moe_ordered_accumulate_down_rows(
        ctx, static_cast<const float*>(g3_down_acc_dev.data()),
        weights_dev,
        static_cast<bf16*>(g3_y_dev.data()),
        tokens, num_active_experts, out_features,
        nullptr, nullptr, &g3_dn);

    ops::moe_ordered_accumulate_down_rows(
        ctx, static_cast<const float*>(g3_down_acc_dev.data()),
        weights_dev,
        static_cast<bf16*>(g3_y_fused_dev.data()),
        tokens, num_active_experts, out_features,
        static_cast<const bf16*>(shared_dev.data()),
        static_cast<const bf16*>(gate_dev.data()),
        &g3_dn);

    ctx.synchronize();

    std::vector<bf16> g3_h(total_acts * intermediate_size);
    std::vector<bf16> g3_y(tokens * out_features);
    std::vector<bf16> g3_y_fused(tokens * out_features);
    ctx.memcpy_d2h(g3_h.data(), g3_h_dev.data(), g3_h.size() * sizeof(bf16));
    ctx.memcpy_d2h(g3_y.data(), g3_y_dev.data(), g3_y.size() * sizeof(bf16));
    ctx.memcpy_d2h(g3_y_fused.data(), g3_y_fused_dev.data(), g3_y_fused.size() * sizeof(bf16));

    for (size_t i = 0; i < ref_h.size(); ++i) {
        if (g3_h[i] != ref_h[i]) {
            throw std::runtime_error("moe_expert_major_prefill_gate_up (4-token reuse) bit-exact mismatch at index " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < ref_y.size(); ++i) {
        if (g3_y[i] != ref_y[i]) {
            throw std::runtime_error("moe_expert_major_prefill_down (4-token reuse) bit-exact mismatch at index " + std::to_string(i));
        }
    }
    for (size_t i = 0; i < ref_y_fused.size(); ++i) {
        if (g3_y_fused[i] != ref_y_fused[i]) {
            throw std::runtime_error("moe_expert_major_prefill_down (4-token reuse fused) bit-exact mismatch at index " + std::to_string(i));
        }
    }

    if (total_quads > 0) {
        Tensor direct_quad_h = Tensor::allocate(ctx, {tokens * num_active_experts * intermediate_size}, dnnl::memory::data_type::bf16);
        Tensor direct_quad_dn = Tensor::allocate(ctx, {tokens * num_active_experts * out_features}, dnnl::memory::data_type::f32);
        sycl::event quad_gu = ops::moe_expert_major_prefill_gate_up_4token(
            ctx, g3_slots_dev,
            static_cast<const int*>(g3_tokens_dev.data()),
            static_cast<const int*>(g3_ranks_dev.data()),
            nullptr,
            static_cast<const bf16*>(x_dev.data()),
            static_cast<bf16*>(direct_quad_h.data()),
            total_quads, num_active_experts, in_features, intermediate_size, blocksize,
            nullptr, use_nested);

        sycl::event quad_dn = ops::moe_expert_major_prefill_down_4token(
            ctx, g3_slots_dev,
            static_cast<const int*>(g3_tokens_dev.data()),
            static_cast<const int*>(g3_ranks_dev.data()),
            nullptr,
            static_cast<const bf16*>(direct_quad_h.data()),
            static_cast<float*>(direct_quad_dn.data()),
            total_quads, num_active_experts, intermediate_size, out_features, blocksize,
            &quad_gu, use_nested);
        quad_dn.wait();

        std::vector<bf16> quad_h_host(tokens * num_active_experts * intermediate_size);
        std::vector<float> quad_dn_host(tokens * num_active_experts * out_features);
        ctx.memcpy_d2h(quad_h_host.data(), direct_quad_h.data(), quad_h_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(quad_dn_host.data(), direct_quad_dn.data(), quad_dn_host.size() * sizeof(float));

        std::vector<float> g3_dn_acc_host(total_acts * out_features);
        ctx.memcpy_d2h(g3_dn_acc_host.data(), g3_down_acc_dev.data(), g3_dn_acc_host.size() * sizeof(float));

        for (int q = 0; q < total_quads * 4; ++q) {
            int t = g3_tokens[q];
            int r = g3_ranks[q];
            size_t base_h = (static_cast<size_t>(t) * num_active_experts + r) * intermediate_size;
            for (int d = 0; d < intermediate_size; ++d) {
                if (quad_h_host[base_h + d] != ref_h[base_h + d]) {
                    throw std::runtime_error("moe_expert_major_prefill_gate_up_4token direct bit-exact mismatch");
                }
            }
            size_t base_dn = (static_cast<size_t>(t) * num_active_experts + r) * out_features;
            for (int d = 0; d < out_features; ++d) {
                if (quad_dn_host[base_dn + d] != g3_dn_acc_host[base_dn + d]) {
                    throw std::runtime_error("moe_expert_major_prefill_down_4token direct bit-exact mismatch");
                }
            }
        }

        // Phase Q4E WG-share: the same quad section computed with the subgroup
        // regrouped mapping (ShareQuads in {2,4,8}) must stay bit-exact against
        // the plain 4-token kernels (and therefore against the serial reference).
        // The output buffers are pre-filled with NaN so any (quad,row) pair the
        // new mapping drops fails the comparison.
        bf16 nan_bf16;
        {
            const uint16_t nan_bits = 0x7FC0u;
            std::memcpy(&nan_bf16, &nan_bits, sizeof(nan_bf16));
        }
        for (int share : {2, 4, 8}) {
            Tensor ws_quad_h = Tensor::allocate(ctx, {tokens * num_active_experts * intermediate_size}, dnnl::memory::data_type::bf16);
            Tensor ws_quad_dn = Tensor::allocate(ctx, {tokens * num_active_experts * out_features}, dnnl::memory::data_type::f32);
            std::vector<bf16> sentinel_h(tokens * num_active_experts * intermediate_size, nan_bf16);
            std::vector<float> sentinel_dn(tokens * num_active_experts * out_features,
                                           std::numeric_limits<float>::quiet_NaN());
            ctx.memcpy_h2d(ws_quad_h.data(), sentinel_h.data(), sentinel_h.size() * sizeof(bf16));
            ctx.memcpy_h2d(ws_quad_dn.data(), sentinel_dn.data(), sentinel_dn.size() * sizeof(float));

            sycl::event ws_gu = ops::moe_expert_major_prefill_gate_up_4token_wgshare(
                ctx, g3_slots_dev,
                static_cast<const int*>(g3_tokens_dev.data()),
                static_cast<const int*>(g3_ranks_dev.data()),
                nullptr,
                static_cast<const bf16*>(x_dev.data()),
                static_cast<bf16*>(ws_quad_h.data()),
                total_quads, num_active_experts, in_features, intermediate_size, blocksize,
                nullptr, use_nested, share, false);

            sycl::event ws_dn = ops::moe_expert_major_prefill_down_4token_wgshare(
                ctx, g3_slots_dev,
                static_cast<const int*>(g3_tokens_dev.data()),
                static_cast<const int*>(g3_ranks_dev.data()),
                nullptr,
                static_cast<const bf16*>(ws_quad_h.data()),
                static_cast<float*>(ws_quad_dn.data()),
                total_quads, num_active_experts, intermediate_size, out_features, blocksize,
                &ws_gu, use_nested, share, false);
            ws_dn.wait();

            std::vector<bf16> ws_h_host(tokens * num_active_experts * intermediate_size);
            std::vector<float> ws_dn_host(tokens * num_active_experts * out_features);
            ctx.memcpy_d2h(ws_h_host.data(), ws_quad_h.data(), ws_h_host.size() * sizeof(bf16));
            ctx.memcpy_d2h(ws_dn_host.data(), ws_quad_dn.data(), ws_dn_host.size() * sizeof(float));

            for (int q = 0; q < total_quads * 4; ++q) {
                int t = g3_tokens[q];
                int r = g3_ranks[q];
                size_t base_h = (static_cast<size_t>(t) * num_active_experts + r) * intermediate_size;
                for (int d = 0; d < intermediate_size; ++d) {
                    if (ws_h_host[base_h + d] != quad_h_host[base_h + d]) {
                        throw std::runtime_error("moe_expert_major_prefill_gate_up_4token_wgshare mismatch vs 4token at share=" +
                                                 std::to_string(share) + " index=" + std::to_string(base_h + d));
                    }
                }
                size_t base_dn = (static_cast<size_t>(t) * num_active_experts + r) * out_features;
                for (int d = 0; d < out_features; ++d) {
                    if (ws_dn_host[base_dn + d] != quad_dn_host[base_dn + d]) {
                        throw std::runtime_error("moe_expert_major_prefill_down_4token_wgshare mismatch vs 4token at share=" +
                                                 std::to_string(share) + " index=" + std::to_string(base_dn + d));
                    }
                }
            }
        }
    }
    ctx.free_device(g3_slots_dev);

    // 7. Expert-major prefill with 8-token weight reuse (G4). Mirrors the G3
    // layout exactly, but the leading bucket packs eight tokens instead of
    // four. Only exercised when eights is enabled so the default runs keep the
    // exact G3 topology.
    static const bool eights = aila::env::read_flag("AILA_Q4E_8TOKEN_REUSE", false);
    if (eights) {
        int total_eights = 0;
        int total_quads_g4 = 0;
        int total_pairs_g4 = 0;
        for (int e = 0; e < num_unique_experts; ++e) {
            int cnt = expert_counts[e];
            total_eights += cnt / 8;
            total_quads_g4 += (cnt % 8) / 4;
            total_pairs_g4 += ((cnt % 8) % 4) / 2;
        }
        std::vector<int> g4_eight_pos(num_unique_experts, 0);
        std::vector<int> g4_quad_pos(num_unique_experts, 0);
        std::vector<int> g4_pair_pos(num_unique_experts, 0);
        std::vector<int> g4_single_pos(num_unique_experts, 0);
        int e_off = 0;
        int q_off_g4 = 8 * total_eights;
        int p_off_g4 = q_off_g4 + 4 * total_quads_g4;
        int s_off_g4 = p_off_g4 + 2 * total_pairs_g4;
        for (int e = 0; e < num_unique_experts; ++e) {
            int cnt = expert_counts[e];
            int cnt8 = cnt % 8;
            g4_eight_pos[e] = e_off;
            g4_quad_pos[e] = q_off_g4;
            g4_pair_pos[e] = p_off_g4;
            g4_single_pos[e] = s_off_g4;
            e_off += (cnt / 8) * 8;
            q_off_g4 += (cnt8 / 4) * 4;
            p_off_g4 += ((cnt8 % 4) / 2) * 2;
            s_off_g4 += cnt8 % 2;
        }
        std::vector<int> g4_tokens(total_acts);
        std::vector<int> g4_ranks(total_acts);
        std::vector<ops::MoeExpertSlotDev> g4_slots(total_acts);
        std::vector<int> g4_expert_seen(num_unique_experts, 0);
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < num_active_experts; ++k) {
                int e = (t * 7 + k) % num_unique_experts;
                int seen = g4_expert_seen[e]++;
                int cnt = expert_counts[e];
                int e_limit = (cnt / 8) * 8;
                int q_limit = e_limit + ((cnt % 8) / 4) * 4;
                int p_limit = q_limit + (((cnt % 8) % 4) / 2) * 2;
                int pos;
                if (seen < e_limit) {
                    pos = g4_eight_pos[e] + seen;
                } else if (seen < q_limit) {
                    pos = g4_quad_pos[e] + (seen - e_limit);
                } else if (seen < p_limit) {
                    pos = g4_pair_pos[e] + (seen - q_limit);
                } else {
                    pos = g4_single_pos[e] + (seen - p_limit);
                }
                g4_tokens[pos] = t;
                g4_ranks[pos] = k;
                g4_slots[pos] = expert_pool[e];
            }
        }

        Tensor g4_tokens_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
        Tensor g4_ranks_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
        ops::MoeExpertSlotDev* g4_slots_dev = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(total_acts * sizeof(ops::MoeExpertSlotDev)));
        Tensor g4_h_dev = Tensor::allocate(ctx, {total_acts * intermediate_size}, dnnl::memory::data_type::bf16);
        Tensor g4_down_acc_dev = Tensor::allocate(ctx, {total_acts * out_features}, dnnl::memory::data_type::f32);
        Tensor g4_y_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);
        Tensor g4_y_fused_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

        ctx.memcpy_h2d(g4_tokens_dev.data(), g4_tokens.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(g4_ranks_dev.data(), g4_ranks.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(g4_slots_dev, g4_slots.data(), total_acts * sizeof(ops::MoeExpertSlotDev));

        sycl::event g4_gu = ops::moe_expert_major_prefill_gate_up(
            ctx, g4_slots_dev,
            static_cast<const int*>(g4_tokens_dev.data()),
            static_cast<const int*>(g4_ranks_dev.data()),
            nullptr,
            static_cast<const bf16*>(x_dev.data()),
            static_cast<bf16*>(g4_h_dev.data()),
            total_acts, num_active_experts, in_features, intermediate_size, blocksize,
            nullptr, use_nested, total_pairs_g4, total_quads_g4, total_eights);

        sycl::event g4_dn = ops::moe_expert_major_prefill_down(
            ctx, g4_slots_dev,
            static_cast<const int*>(g4_tokens_dev.data()),
            static_cast<const int*>(g4_ranks_dev.data()),
            nullptr,
            static_cast<const bf16*>(g4_h_dev.data()),
            static_cast<float*>(g4_down_acc_dev.data()),
            total_acts, num_active_experts, intermediate_size, out_features, blocksize,
            &g4_gu, use_nested, total_pairs_g4, total_quads_g4, total_eights);

        ops::moe_ordered_accumulate_down_rows(
            ctx, static_cast<const float*>(g4_down_acc_dev.data()),
            weights_dev,
            static_cast<bf16*>(g4_y_dev.data()),
            tokens, num_active_experts, out_features,
            nullptr, nullptr, &g4_dn);

        ops::moe_ordered_accumulate_down_rows(
            ctx, static_cast<const float*>(g4_down_acc_dev.data()),
            weights_dev,
            static_cast<bf16*>(g4_y_fused_dev.data()),
            tokens, num_active_experts, out_features,
            static_cast<const bf16*>(shared_dev.data()),
            static_cast<const bf16*>(gate_dev.data()),
            &g4_dn);

        ctx.synchronize();

        std::vector<bf16> g4_h(total_acts * intermediate_size);
        std::vector<bf16> g4_y(tokens * out_features);
        std::vector<bf16> g4_y_fused(tokens * out_features);
        ctx.memcpy_d2h(g4_h.data(), g4_h_dev.data(), g4_h.size() * sizeof(bf16));
        ctx.memcpy_d2h(g4_y.data(), g4_y_dev.data(), g4_y.size() * sizeof(bf16));
        ctx.memcpy_d2h(g4_y_fused.data(), g4_y_fused_dev.data(), g4_y_fused.size() * sizeof(bf16));

        for (size_t i = 0; i < ref_h.size(); ++i) {
            if (g4_h[i] != ref_h[i]) {
                throw std::runtime_error("moe_expert_major_prefill_gate_up (8-token reuse) bit-exact mismatch at index " + std::to_string(i));
            }
        }
        for (size_t i = 0; i < ref_y.size(); ++i) {
            if (g4_y[i] != ref_y[i]) {
                throw std::runtime_error("moe_expert_major_prefill_down (8-token reuse) bit-exact mismatch at index " + std::to_string(i));
            }
        }
        for (size_t i = 0; i < ref_y_fused.size(); ++i) {
            if (g4_y_fused[i] != ref_y_fused[i]) {
                throw std::runtime_error("moe_expert_major_prefill_down (8-token reuse fused) bit-exact mismatch at index " + std::to_string(i));
            }
        }

        if (total_eights > 0) {
            Tensor direct_eight_h = Tensor::allocate(ctx, {tokens * num_active_experts * intermediate_size}, dnnl::memory::data_type::bf16);
            Tensor direct_eight_dn = Tensor::allocate(ctx, {tokens * num_active_experts * out_features}, dnnl::memory::data_type::f32);
            std::vector<float> g4_down_acc_host(total_acts * out_features);
            ctx.memcpy_d2h(g4_down_acc_host.data(), g4_down_acc_dev.data(), g4_down_acc_host.size() * sizeof(float));
            sycl::event eight_gu = ops::moe_expert_major_prefill_gate_up_8token(
                ctx, g4_slots_dev,
                static_cast<const int*>(g4_tokens_dev.data()),
                static_cast<const int*>(g4_ranks_dev.data()),
                nullptr,
                static_cast<const bf16*>(x_dev.data()),
                static_cast<bf16*>(direct_eight_h.data()),
                total_eights, num_active_experts, in_features, intermediate_size, blocksize,
                nullptr, use_nested);

            sycl::event eight_dn = ops::moe_expert_major_prefill_down_8token(
                ctx, g4_slots_dev,
                static_cast<const int*>(g4_tokens_dev.data()),
                static_cast<const int*>(g4_ranks_dev.data()),
                nullptr,
                static_cast<const bf16*>(direct_eight_h.data()),
                static_cast<float*>(direct_eight_dn.data()),
                total_eights, num_active_experts, intermediate_size, out_features, blocksize,
                &eight_gu, use_nested);
            eight_dn.wait();

            std::vector<bf16> eight_h_host(tokens * num_active_experts * intermediate_size);
            std::vector<float> eight_dn_host(tokens * num_active_experts * out_features);
            ctx.memcpy_d2h(eight_h_host.data(), direct_eight_h.data(), eight_h_host.size() * sizeof(bf16));
            ctx.memcpy_d2h(eight_dn_host.data(), direct_eight_dn.data(), eight_dn_host.size() * sizeof(float));

            for (int o = 0; o < total_eights * 8; ++o) {
                int t = g4_tokens[o];
                int r = g4_ranks[o];
                size_t base_h = (static_cast<size_t>(t) * num_active_experts + r) * intermediate_size;
                for (int d = 0; d < intermediate_size; ++d) {
                    if (eight_h_host[base_h + d] != ref_h[base_h + d]) {
                        throw std::runtime_error("moe_expert_major_prefill_gate_up_8token direct bit-exact mismatch");
                    }
                }
                size_t base_dn = (static_cast<size_t>(t) * num_active_experts + r) * out_features;
                for (int d = 0; d < out_features; ++d) {
                    if (eight_dn_host[base_dn + d] != g4_down_acc_host[base_dn + d]) {
                        throw std::runtime_error("moe_expert_major_prefill_down_8token direct bit-exact mismatch");
                    }
                }
            }
        }
        ctx.free_device(g4_slots_dev);
    }

    std::cout << "[PASS] moe_prefill tokens=" << tokens
              << " k=" << num_active_experts
              << " nested=" << (use_nested ? "true" : "false")
              << " bit-exact match (0 mismatches across " << ref_y.size() << " values, all S1, S4, W4 C1/C2, G3 4-token, and G4 8-token)\n";

    ctx.free_device(slots_dev_buf);
    ctx.free_device(weights_dev);
}

void test_moe_prefill_gate_up_and_down(Context& ctx) {
    run_moe_prefill_case(ctx, 2, 4, false);
    run_moe_prefill_case(ctx, 2, 10, true);
    run_moe_prefill_case(ctx, 26, 10, false);
    run_moe_prefill_case(ctx, 26, 10, true);
    run_moe_prefill_case(ctx, 64, 10, true);
    // Larger routings (expert counts >= 8) also exercise the AILA_Q4E_8TOKEN_REUSE
    // 8-token bucket when the environment enables it; env-off runs stay at G3 size.
    run_moe_prefill_case(ctx, 16, 4, false);
    run_moe_prefill_case(ctx, 16, 4, true);
    run_moe_prefill_case(ctx, 32, 4, false);
    run_moe_prefill_case(ctx, 32, 4, true);
}

// Phase Q4E WG-share mapping edge cases. Both wgshare kernels are compared
// directly (bit for bit) against their plain 4-token counterparts on shapes
// that the production 640/2560 layouts cannot reach: Q not divisible by
// ShareQuads (5 % 2, 5 % 4 and 5 % 8 are all non-zero) and short row-block
// tails (13 % {8,4,2} and 7 % {8,4,2} are all non-zero). Output buffers start
// at a sentinel value, so a dropped (quad,row) pair shows up as a mismatch.
void test_moe_prefill_wgshare_mapping_edges(Context& ctx) {
    constexpr int in_features = 256;       // 128 packed bytes per row
    constexpr int gu_intermediate = 13;    // rows_per_wg in {8,4,2}: all leave a tail
    constexpr int down_in_features = 128;  // 64 packed bytes per row
    constexpr int down_out_features = 7;   // row-block tail for every ShareQuads
    constexpr int blocksize = 64;
    constexpr int top_k = 3;
    constexpr int tokens = 8;
    constexpr int acts = tokens * top_k;
    constexpr int quads = 5;
    constexpr int num_experts = 2;
    constexpr float sentinel = -12345.0f;

    std::mt19937 rng(20261010);
    std::uniform_int_distribution<int> u8_dist(0, 255);
    std::uniform_real_distribution<float> f_dist(0.01f, 0.5f);
    std::uniform_real_distribution<float> in_dist(-0.2f, 0.2f);

    const size_t gu_packed_bytes = static_cast<size_t>(2 * gu_intermediate) * (in_features / 2);
    const size_t gu_absmax_count = static_cast<size_t>(2 * gu_intermediate) * (in_features / blocksize);
    const size_t dn_packed_bytes = static_cast<size_t>(down_out_features) * (down_in_features / 2);
    const size_t dn_absmax_count = static_cast<size_t>(down_out_features) * (down_in_features / blocksize);

    std::vector<Tensor> gu_packed_dev, gu_absmax_dev, dn_packed_dev, dn_absmax_dev;
    std::vector<ops::MoeExpertSlotDev> pool(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        gu_packed_dev.push_back(Tensor::allocate(ctx, {gu_packed_bytes}, dnnl::memory::data_type::u8));
        gu_absmax_dev.push_back(Tensor::allocate(ctx, {gu_absmax_count}, dnnl::memory::data_type::f32));
        dn_packed_dev.push_back(Tensor::allocate(ctx, {dn_packed_bytes}, dnnl::memory::data_type::u8));
        dn_absmax_dev.push_back(Tensor::allocate(ctx, {dn_absmax_count}, dnnl::memory::data_type::f32));

        std::vector<uint8_t> gu_p(gu_packed_bytes), dn_p(dn_packed_bytes);
        std::vector<float> gu_a(gu_absmax_count), dn_a(dn_absmax_count);
        for (auto& v : gu_p) v = static_cast<uint8_t>(u8_dist(rng));
        for (auto& v : dn_p) v = static_cast<uint8_t>(u8_dist(rng));
        for (auto& v : gu_a) v = f_dist(rng);
        for (auto& v : dn_a) v = f_dist(rng);
        ctx.memcpy_h2d(gu_packed_dev[e].data(), gu_p.data(), gu_packed_bytes);
        ctx.memcpy_h2d(gu_absmax_dev[e].data(), gu_a.data(), gu_absmax_count * sizeof(float));
        ctx.memcpy_h2d(dn_packed_dev[e].data(), dn_p.data(), dn_packed_bytes);
        ctx.memcpy_h2d(dn_absmax_dev[e].data(), dn_a.data(), dn_absmax_count * sizeof(float));
        pool[e].gate_up_packed = static_cast<const uint8_t*>(gu_packed_dev[e].data());
        pool[e].gate_up_absmax = static_cast<const float*>(gu_absmax_dev[e].data());
        pool[e].down_packed = static_cast<const uint8_t*>(dn_packed_dev[e].data());
        pool[e].down_absmax = static_cast<const float*>(dn_absmax_dev[e].data());
    }

    // G3-style quad layout: slot 4q..4q+3 belong to one expert quad, and the
    // (token,rank) pair of position p maps to output row p (act p).
    std::vector<int> tok(acts), rnk(acts);
    std::vector<ops::MoeExpertSlotDev> slots(acts);
    for (int p = 0; p < acts; ++p) {
        tok[p] = p / top_k;
        rnk[p] = p % top_k;
        slots[p] = pool[(p / 4) % num_experts];
    }

    std::vector<bf16> x_host(tokens * in_features);
    for (auto& v : x_host) v = bf16(in_dist(rng));

    Tensor x_dev = Tensor::allocate(ctx, {tokens, in_features}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));

    Tensor tok_dev = Tensor::allocate(ctx, {acts}, dnnl::memory::data_type::s32);
    Tensor rnk_dev = Tensor::allocate(ctx, {acts}, dnnl::memory::data_type::s32);
    auto* slots_dev = static_cast<ops::MoeExpertSlotDev*>(ctx.alloc_device(acts * sizeof(ops::MoeExpertSlotDev)));
    ctx.memcpy_h2d(tok_dev.data(), tok.data(), acts * sizeof(int));
    ctx.memcpy_h2d(rnk_dev.data(), rnk.data(), acts * sizeof(int));
    ctx.memcpy_h2d(slots_dev, slots.data(), acts * sizeof(ops::MoeExpertSlotDev));

    const int gu_values = acts * gu_intermediate;
    const int dn_values = acts * down_out_features;

    auto run_gu = [&](bool wgshare, int share, Tensor& h_out, const sycl::event* dep) {
        if (!wgshare) {
            return ops::moe_expert_major_prefill_gate_up_4token(
                ctx, slots_dev, static_cast<const int*>(tok_dev.data()), static_cast<const int*>(rnk_dev.data()),
                nullptr, static_cast<const bf16*>(x_dev.data()), static_cast<bf16*>(h_out.data()),
                quads, top_k, in_features, gu_intermediate, blocksize, dep, false, false);
        }
        return ops::moe_expert_major_prefill_gate_up_4token_wgshare(
            ctx, slots_dev, static_cast<const int*>(tok_dev.data()), static_cast<const int*>(rnk_dev.data()),
            nullptr, static_cast<const bf16*>(x_dev.data()), static_cast<bf16*>(h_out.data()),
            quads, top_k, in_features, gu_intermediate, blocksize, dep, false, share, false);
    };
    auto run_dn = [&](bool wgshare, int share, const Tensor& h_in, Tensor& dn_out, const sycl::event* dep) {
        if (!wgshare) {
            return ops::moe_expert_major_prefill_down_4token(
                ctx, slots_dev, static_cast<const int*>(tok_dev.data()), static_cast<const int*>(rnk_dev.data()),
                nullptr, static_cast<const bf16*>(h_in.data()), static_cast<float*>(dn_out.data()),
                quads, top_k, down_in_features, down_out_features, blocksize, dep, false, false);
        }
        return ops::moe_expert_major_prefill_down_4token_wgshare(
            ctx, slots_dev, static_cast<const int*>(tok_dev.data()), static_cast<const int*>(rnk_dev.data()),
            nullptr, static_cast<const bf16*>(h_in.data()), static_cast<float*>(dn_out.data()),
            quads, top_k, down_in_features, down_out_features, blocksize, dep, false, share, false);
    };

    Tensor ref_gu_dev = Tensor::allocate(ctx, {gu_values}, dnnl::memory::data_type::bf16);
    Tensor ref_dn_dev = Tensor::allocate(ctx, {dn_values}, dnnl::memory::data_type::f32);
    std::vector<bf16> ref_gu(gu_values, bf16(sentinel));
    std::vector<float> ref_dn(dn_values, sentinel);
    ctx.memcpy_h2d(ref_gu_dev.data(), ref_gu.data(), ref_gu.size() * sizeof(bf16));
    ctx.memcpy_h2d(ref_dn_dev.data(), ref_dn.data(), ref_dn.size() * sizeof(float));
    sycl::event ref_gu_ev = run_gu(false, 0, ref_gu_dev, nullptr);
    sycl::event ref_dn_ev = run_dn(false, 0, ref_gu_dev, ref_dn_dev, &ref_gu_ev);
    ref_dn_ev.wait();
    ctx.memcpy_d2h(ref_gu.data(), ref_gu_dev.data(), ref_gu.size() * sizeof(bf16));
    ctx.memcpy_d2h(ref_dn.data(), ref_dn_dev.data(), ref_dn.size() * sizeof(float));

    int mismatches = 0;
    for (int share : {2, 4, 8}) {
        Tensor ws_gu_dev = Tensor::allocate(ctx, {gu_values}, dnnl::memory::data_type::bf16);
        Tensor ws_dn_dev = Tensor::allocate(ctx, {dn_values}, dnnl::memory::data_type::f32);
        std::vector<bf16> ws_gu(gu_values, bf16(sentinel));
        std::vector<float> ws_dn(dn_values, sentinel);
        ctx.memcpy_h2d(ws_gu_dev.data(), ws_gu.data(), ws_gu.size() * sizeof(bf16));
        ctx.memcpy_h2d(ws_dn_dev.data(), ws_dn.data(), ws_dn.size() * sizeof(float));
        sycl::event ws_gu_ev = run_gu(true, share, ws_gu_dev, nullptr);
        sycl::event ws_dn_ev = run_dn(true, share, ws_gu_dev, ws_dn_dev, &ws_gu_ev);
        ws_dn_ev.wait();
        ctx.memcpy_d2h(ws_gu.data(), ws_gu_dev.data(), ws_gu.size() * sizeof(bf16));
        ctx.memcpy_d2h(ws_dn.data(), ws_dn_dev.data(), ws_dn.size() * sizeof(float));

        for (int i = 0; i < gu_values; ++i) {
            if (ws_gu[i] != ref_gu[i]) {
                ++mismatches;
                if (mismatches == 1) {
                    std::cout << "  [wgshare] gate_up mismatch at share=" << share << " index=" << i << "\n";
                }
            }
        }
        for (int i = 0; i < dn_values; ++i) {
            if (ws_dn[i] != ref_dn[i]) {
                ++mismatches;
                if (mismatches == 1) {
                    std::cout << "  [wgshare] down mismatch at share=" << share << " index=" << i << "\n";
                }
            }
        }
    }
    if (mismatches != 0) {
        throw std::runtime_error("moe_expert_major_prefill wgshare mapping edge mismatch: " +
                                 std::to_string(mismatches) + " values differ from the 4-token path");
    }

    ctx.free_device(slots_dev);
    std::cout << "[PASS] moe_prefill wgshare mapping edges (quads=" << quads
              << ", I=" << gu_intermediate << ", H=" << down_out_features
              << ", shares 2/4/8 bit-exact vs 4-token, 0 mismatches)\n";
}

void test_gated_delta_net_sigmoid(Context& ctx) {
    constexpr int q_heads = 16;
    constexpr int v_heads = 48;
    constexpr int head_k = 128;
    constexpr int head_v = 128;
    constexpr int kernel = 4;
    constexpr float eps = 1e-6f;

    auto geom_silu = ops::DeltaNetGeometry::create(
        q_heads, v_heads, head_k, head_v, kernel, eps, ops::GateActivation::Silu);
    auto geom_sigmoid = ops::DeltaNetGeometry::create(
        q_heads, v_heads, head_k, head_v, kernel, eps, ops::GateActivation::Sigmoid);

    // Verify dimensions
    if (geom_silu.qkv_dim != 10240 || geom_silu.z_dim != 6144 || geom_silu.all_dim != 16480) {
        throw std::runtime_error("DeltaNetGeometry dimension calculation unexpected");
    }
    std::cout << "[PASS] DeltaNetGeometry dimension calculations validated\n";

    // -------------------------------------------------------------------------
    // Phase G2 Parity Check: Verify GDN Single-Write Pass vs CPU Reference
    // -------------------------------------------------------------------------
    const int qkv_dim = geom_silu.qkv_dim;
    const int z_dim = geom_silu.z_dim;
    const int state_elements = v_heads * head_k * head_v;

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist_s(-0.05f, 0.05f);
    std::uniform_real_distribution<float> dist_x(-0.2f, 0.2f);
    std::uniform_real_distribution<float> dist_ab(-0.5f, 0.5f);

    std::vector<float> S_init(state_elements);
    for (auto& s : S_init) s = dist_s(rng);

    std::vector<bf16> conv_w(static_cast<size_t>(qkv_dim) * kernel, bf16(0.0f));
    for (int c = 0; c < qkv_dim; ++c) {
        conv_w[static_cast<size_t>(c) * kernel + 3] = bf16(1.0f); // Identity tap
    }

    std::vector<float> norm_w(head_v, 1.0f);
    std::vector<float> a_log(v_heads, -1.0f);
    std::vector<bf16> dt_bias(v_heads, bf16(0.0f));
    std::vector<float> conv_state(static_cast<size_t>(geom_silu.conv_rows) * qkv_dim, 0.0f);

    std::vector<bf16> qkv(qkv_dim), z(z_dim), a(v_heads), b(v_heads);
    for (auto& x : qkv) x = bf16(dist_x(rng));
    for (auto& x : z) x = bf16(dist_x(rng));
    for (auto& x : a) x = bf16(dist_ab(rng));
    for (auto& x : b) x = bf16(dist_ab(rng));

    // Allocate GPU tensors
    Tensor conv_w_dev = Tensor::allocate(ctx, {qkv_dim, kernel}, dnnl::memory::data_type::bf16);
    Tensor norm_w_dev = Tensor::allocate(ctx, {head_v}, dnnl::memory::data_type::f32);
    Tensor a_log_dev = Tensor::allocate(ctx, {v_heads}, dnnl::memory::data_type::f32);
    Tensor dt_bias_dev = Tensor::allocate(ctx, {v_heads}, dnnl::memory::data_type::bf16);

    Tensor state_dev = Tensor::allocate(ctx, {v_heads, head_k, head_v}, dnnl::memory::data_type::f32);
    Tensor conv_state_dev = Tensor::allocate(ctx, {geom_silu.conv_rows, qkv_dim}, dnnl::memory::data_type::f32);

    Tensor qkv_dev = Tensor::allocate(ctx, {qkv_dim}, dnnl::memory::data_type::bf16);
    Tensor z_dev = Tensor::allocate(ctx, {z_dim}, dnnl::memory::data_type::bf16);
    Tensor a_dev = Tensor::allocate(ctx, {v_heads}, dnnl::memory::data_type::bf16);
    Tensor b_dev = Tensor::allocate(ctx, {v_heads}, dnnl::memory::data_type::bf16);
    Tensor out_dev = Tensor::allocate(ctx, {v_heads * head_v}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(conv_w_dev.data(), conv_w.data(), conv_w.size() * sizeof(bf16));
    ctx.memcpy_h2d(norm_w_dev.data(), norm_w.data(), norm_w.size() * sizeof(float));
    ctx.memcpy_h2d(a_log_dev.data(), a_log.data(), a_log.size() * sizeof(float));
    ctx.memcpy_h2d(dt_bias_dev.data(), dt_bias.data(), dt_bias.size() * sizeof(bf16));

    ctx.memcpy_h2d(state_dev.data(), S_init.data(), S_init.size() * sizeof(float));
    ctx.memcpy_h2d(conv_state_dev.data(), conv_state.data(), conv_state.size() * sizeof(float));

    ctx.memcpy_h2d(qkv_dev.data(), qkv.data(), qkv.size() * sizeof(bf16));
    ctx.memcpy_h2d(z_dev.data(), z.data(), z.size() * sizeof(bf16));
    ctx.memcpy_h2d(a_dev.data(), a.data(), a.size() * sizeof(bf16));
    ctx.memcpy_h2d(b_dev.data(), b.data(), b.size() * sizeof(bf16));

    ops::DeltaNetWeights weights{
        conv_w_dev.data_as<bf16>(),
        norm_w_dev.data_as<float>(),
        a_log_dev.data_as<float>(),
        dt_bias_dev.data_as<bf16>()
    };

    ops::DeltaNetState state{
        state_dev.data_as<float>(),
        conv_state_dev.data_as<float>(),
        0,
        false
    };

    // CPU Reference Calculation for Decode Step
    auto silu_fn = [](float v) { return v / (1.0f + std::exp(-v)); };
    auto softplus_fn = [](float v) {
        if (v > 20.0f) return v;
        if (v < -20.0f) return std::exp(v);
        return std::log1p(std::exp(v));
    };

    std::vector<float> q_conv(q_heads * head_k), k_conv(q_heads * head_k), v_conv(v_heads * head_v);
    for (int h = 0; h < q_heads; ++h) {
        float q_sq = 0.0f, k_sq = 0.0f;
        for (int d = 0; d < head_k; ++d) {
            float q_val = silu_fn(static_cast<float>(qkv[h * head_k + d]));
            float k_val = silu_fn(static_cast<float>(qkv[geom_silu.q_dim + h * head_k + d]));
            q_conv[h * head_k + d] = q_val;
            k_conv[h * head_k + d] = k_val;
            q_sq += q_val * q_val;
            k_sq += k_val * k_val;
        }
        float q_sc = 1.0f / std::sqrt(q_sq + eps);
        float k_sc = 1.0f / std::sqrt(k_sq + eps);
        for (int d = 0; d < head_k; ++d) {
            q_conv[h * head_k + d] *= q_sc;
            k_conv[h * head_k + d] *= k_sc;
        }
    }
    for (int i = 0; i < v_heads * head_v; ++i) {
        v_conv[i] = silu_fn(static_cast<float>(qkv[2 * geom_silu.q_dim + i]));
    }

    std::vector<float> S_ref = S_init;
    std::vector<bf16> out_ref(v_heads * head_v);

    for (int hv = 0; hv < v_heads; ++hv) {
        int hk = ops::linear_delta_key_head_for_value_head(hv, q_heads, v_heads);
        float beta = 1.0f / (1.0f + std::exp(-static_cast<float>(b[hv])));
        float alpha = static_cast<float>(a[hv]);
        float g_pre = softplus_fn(alpha + static_cast<float>(dt_bias[hv])) * (-std::exp(a_log[hv]));
        float decay = std::exp(g_pre);

        float* S_head = S_ref.data() + static_cast<size_t>(hv) * head_k * head_v;
        std::vector<float> out_h(head_v, 0.0f);

        for (int dv = 0; dv < head_v; ++dv) {
            float acc = 0.0f;
            for (int j = 0; j < head_k; ++j) {
                acc += S_head[j * head_v + dv] * decay * k_conv[hk * head_k + j];
            }
            float dval = (v_conv[hv * head_v + dv] - acc) * beta;
            for (int j = 0; j < head_k; ++j) {
                float sv = S_head[j * head_v + dv] * decay + k_conv[hk * head_k + j] * dval;
                S_head[j * head_v + dv] = sv;
                out_h[dv] += sv * q_conv[hk * head_k + j];
            }
            out_h[dv] *= geom_silu.scale;
        }

        float out_sq = 0.0f;
        for (int dv = 0; dv < head_v; ++dv) out_sq += out_h[dv] * out_h[dv];
        float out_scale = 1.0f / std::sqrt(out_sq / static_cast<float>(head_v) + eps);
        for (int dv = 0; dv < head_v; ++dv) {
            float n = out_h[dv] * out_scale * norm_w[dv];
            float gate = silu_fn(static_cast<float>(z[hv * head_v + dv]));
            out_ref[hv * head_v + dv] = bf16(n * gate);
        }
    }

    // Run GPU kernel
    ops::run_linear_delta_decode_gpu(
        ctx, geom_silu, weights, state,
        qkv_dev.data_as<bf16>(), z_dev.data_as<bf16>(),
        a_dev.data_as<bf16>(), b_dev.data_as<bf16>(),
        out_dev.data_as<bf16>());
    ctx.synchronize();

    std::vector<float> S_gpu(state_elements);
    std::vector<bf16> out_gpu(v_heads * head_v);
    ctx.memcpy_d2h(S_gpu.data(), state_dev.data(), S_gpu.size() * sizeof(float));
    ctx.memcpy_d2h(out_gpu.data(), out_dev.data(), out_gpu.size() * sizeof(bf16));

    check_close_f32(S_gpu, S_ref, "GDN single-write state parity (decode)", 2e-4f);
    check_close(out_gpu, std::vector<float>(out_ref.begin(), out_ref.end()), "GDN single-write out parity (decode)", 2e-2f);
}

void test_gdn_device_conv_head(Context& ctx) {
    constexpr int q_heads = 16;
    constexpr int v_heads = 48;
    constexpr int head_dim = 128;
    constexpr int kernel = 4;
    constexpr int steps = 5;
    constexpr int initial_head = 2;
    auto geom = ops::DeltaNetGeometry::create(
        q_heads, v_heads, head_dim, head_dim, kernel, 1e-6f,
        ops::GateActivation::Sigmoid);
    const int conv_rows = geom.conv_rows;
    const int final_head = (initial_head + steps) % conv_rows;
    const int last_written_head = (initial_head + steps - 1) % conv_rows;

    std::mt19937 rng(8042);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    auto make_bf16 = [&](size_t count) {
        std::vector<bf16> values(count);
        for (auto& value : values) value = bf16(dist(rng));
        return values;
    };
    auto make_f32 = [&](size_t count) {
        std::vector<float> values(count);
        for (auto& value : values) value = dist(rng);
        return values;
    };

    auto conv_w = make_bf16(static_cast<size_t>(geom.qkv_dim) * kernel);
    auto dt_bias = make_bf16(v_heads);
    auto fused = make_bf16(static_cast<size_t>(steps) * geom.all_dim);
    auto initial_state = make_f32(static_cast<size_t>(v_heads) * head_dim * head_dim);
    auto initial_history = make_f32(static_cast<size_t>(conv_rows) * geom.qkv_dim);
    std::vector<float> norm_w(head_dim, 1.0f);
    std::vector<float> a_log(v_heads, -1.0f);

    Tensor conv_w_dev = Tensor::allocate(ctx, {geom.qkv_dim, kernel}, dnnl::memory::data_type::bf16);
    Tensor norm_w_dev = Tensor::allocate(ctx, {head_dim}, dnnl::memory::data_type::f32);
    Tensor a_log_dev = Tensor::allocate(ctx, {v_heads}, dnnl::memory::data_type::f32);
    Tensor dt_bias_dev = Tensor::allocate(ctx, {v_heads}, dnnl::memory::data_type::bf16);
    Tensor fused_dev = Tensor::allocate(
        ctx, {steps, geom.all_dim}, dnnl::memory::data_type::bf16);
    Tensor host_state_dev = Tensor::allocate(
        ctx, {v_heads, head_dim, head_dim}, dnnl::memory::data_type::f32);
    Tensor host_history_dev = Tensor::allocate(
        ctx, {conv_rows, geom.qkv_dim}, dnnl::memory::data_type::f32);
    Tensor device_state_dev = Tensor::allocate(
        ctx, {v_heads, head_dim, head_dim}, dnnl::memory::data_type::f32);
    Tensor device_history_dev = Tensor::allocate(
        ctx, {conv_rows, geom.qkv_dim}, dnnl::memory::data_type::f32);
    Tensor host_out_dev = Tensor::allocate(
        ctx, {steps, geom.kv_dim}, dnnl::memory::data_type::bf16);
    Tensor device_out_dev = Tensor::allocate(
        ctx, {steps, geom.kv_dim}, dnnl::memory::data_type::bf16);
    Tensor conv_head_dev = Tensor::allocate(
        ctx, {ops::delta_net_conv_head_device_slots}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(conv_w_dev.data(), conv_w.data(), conv_w.size() * sizeof(bf16));
    ctx.memcpy_h2d(norm_w_dev.data(), norm_w.data(), norm_w.size() * sizeof(float));
    ctx.memcpy_h2d(a_log_dev.data(), a_log.data(), a_log.size() * sizeof(float));
    ctx.memcpy_h2d(dt_bias_dev.data(), dt_bias.data(), dt_bias.size() * sizeof(bf16));
    ctx.memcpy_h2d(fused_dev.data(), fused.data(), fused.size() * sizeof(bf16));
    ctx.memcpy_h2d(host_state_dev.data(), initial_state.data(), initial_state.size() * sizeof(float));
    ctx.memcpy_h2d(host_history_dev.data(), initial_history.data(), initial_history.size() * sizeof(float));
    ctx.memcpy_h2d(device_state_dev.data(), initial_state.data(), initial_state.size() * sizeof(float));
    ctx.memcpy_h2d(device_history_dev.data(), initial_history.data(), initial_history.size() * sizeof(float));

    ops::DeltaNetWeights weights{
        conv_w_dev.data_as<bf16>(),
        norm_w_dev.data_as<float>(),
        a_log_dev.data_as<float>(),
        dt_bias_dev.data_as<bf16>()};
    auto run = [&](ops::DeltaNetState& state, Tensor& out) {
        for (int step = 0; step < steps; ++step) {
            const bf16* row = fused_dev.data_as<bf16>() + size_t(step) * geom.all_dim;
            ops::run_linear_delta_decode_gpu(
                ctx, geom, weights, state, row, row + geom.qkv_dim,
                row + geom.qkv_dim + geom.z_dim,
                row + geom.qkv_dim + geom.z_dim + geom.v_heads,
                out.data_as<bf16>() + size_t(step) * geom.kv_dim);
        }
    };

    ops::DeltaNetState host_meta{
        host_state_dev.data_as<float>(), host_history_dev.data_as<float>(),
        initial_head, false};
    run(host_meta, host_out_dev);

    int slots_host[3] = {initial_head, initial_head, 0};
    ctx.memcpy_h2d(conv_head_dev.data(), slots_host, sizeof(slots_host));
    ops::DeltaNetState device_meta{
        device_state_dev.data_as<float>(), device_history_dev.data_as<float>(),
        initial_head, false};
    device_meta.conv_head_dev = conv_head_dev.data_as<int>();
    run(device_meta, device_out_dev);
    ctx.synchronize();

    std::vector<bf16> host_out(static_cast<size_t>(steps) * geom.kv_dim);
    std::vector<bf16> device_out(host_out.size());
    std::vector<float> host_state_after(initial_state.size());
    std::vector<float> device_state_after(initial_state.size());
    std::vector<float> host_history(initial_history.size());
    std::vector<float> device_history(initial_history.size());
    ctx.memcpy_d2h(host_out.data(), host_out_dev.data(), host_out.size() * sizeof(bf16));
    ctx.memcpy_d2h(device_out.data(), device_out_dev.data(), device_out.size() * sizeof(bf16));
    ctx.memcpy_d2h(host_state_after.data(), host_state_dev.data(), host_state_after.size() * sizeof(float));
    ctx.memcpy_d2h(device_state_after.data(), device_state_dev.data(), device_state_after.size() * sizeof(float));
    ctx.memcpy_d2h(host_history.data(), host_history_dev.data(), host_history.size() * sizeof(float));
    ctx.memcpy_d2h(device_history.data(), device_history_dev.data(), device_history.size() * sizeof(float));
    ctx.memcpy_d2h(slots_host, conv_head_dev.data(), sizeof(slots_host));

    if (std::memcmp(host_out.data(), device_out.data(), host_out.size() * sizeof(bf16)) != 0 ||
        std::memcmp(host_state_after.data(), device_state_after.data(), host_state_after.size() * sizeof(float)) != 0 ||
        std::memcmp(host_history.data(), device_history.data(), host_history.size() * sizeof(float)) != 0) {
        throw std::runtime_error("GDN device conv head changed output or recurrent state");
    }
    if (slots_host[0] != final_head || slots_host[1] != last_written_head ||
        slots_host[2] != 0) {
        throw std::runtime_error(
            "GDN device conv slots mismatch: head=" + std::to_string(slots_host[0]) +
            " previous=" + std::to_string(slots_host[1]) +
            " arrivals=" + std::to_string(slots_host[2]));
    }
    if (host_meta.conv_head != final_head || device_meta.conv_head != initial_head) {
        throw std::runtime_error("GDN host/device conv head mirrors diverged unexpectedly");
    }
    if (!host_meta.device_state_dirty || !device_meta.device_state_dirty) {
        throw std::runtime_error("GDN device conv head did not mark state dirty");
    }
    std::cout << "[PASS] GDN device conv head graph-safe advance (steps=" << steps
              << ", final=" << slots_host[0] << ", prior=" << slots_host[1]
              << ", bitwise state/history/output)\n";
}

static void run_attention_decode_gate_fusion_case(
    Context& ctx, int num_heads, int num_kv_heads, int head_dim,
    int cached_len, int max_seq_len, bool with_partials) {
    const int total_q = num_heads * head_dim;
    const size_t total_kv = static_cast<size_t>(num_kv_heads) * max_seq_len * head_dim;

    std::mt19937 rng(1337 + head_dim + cached_len);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> q_host(total_q);
    std::vector<bf16> k_host(total_kv);
    std::vector<bf16> v_host(total_kv);
    std::vector<bf16> gate_host(total_q);

    for (size_t i = 0; i < total_q; ++i) {
        q_host[i] = bf16(dist(rng));
        gate_host[i] = bf16(dist(rng));
    }
    for (size_t i = 0; i < total_kv; ++i) {
        k_host[i] = bf16(dist(rng));
        v_host[i] = bf16(dist(rng));
    }

    Tensor q_dev = Tensor::allocate(ctx, {1, total_q}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(ctx, {(int64_t)num_kv_heads, (int64_t)max_seq_len, (int64_t)head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(ctx, {(int64_t)num_kv_heads, (int64_t)max_seq_len, (int64_t)head_dim}, dnnl::memory::data_type::bf16);
    Tensor gate_dev = Tensor::allocate(ctx, {1, total_q}, dnnl::memory::data_type::bf16);
    Tensor out_fused_dev = Tensor::allocate(ctx, {1, total_q}, dnnl::memory::data_type::bf16);
    Tensor out_sep_dev = Tensor::allocate(ctx, {1, total_q}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(ctx, {(int64_t)num_heads, (int64_t)max_seq_len}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(q_dev.data(), q_host.data(), total_q * sizeof(bf16));
    ctx.memcpy_h2d(k_dev.data(), k_host.data(), total_kv * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), v_host.data(), total_kv * sizeof(bf16));
    ctx.memcpy_h2d(gate_dev.data(), gate_host.data(), total_q * sizeof(bf16));

    Tensor partials_dev;
    Tensor* partials_ptr = nullptr;
    if (with_partials) {
        constexpr int kDecodeExactTile = 128;
        const int partial_stride = (head_dim == 256) ? 272 : 144;
        const int max_tiles = (max_seq_len + kDecodeExactTile - 1) / kDecodeExactTile;
        partials_dev = Tensor::allocate(
            ctx,
            {(int64_t)num_heads, (int64_t)max_tiles, (int64_t)partial_stride},
            dnnl::memory::data_type::f32);
        partials_ptr = &partials_dev;
    }

    // 1. Run fused attention decode
    ops::attention_decode(
        ctx, q_dev, k_dev, v_dev, out_fused_dev, scores_dev,
        num_heads, num_kv_heads, head_dim, cached_len,
        partials_ptr, &gate_dev);
    ctx.synchronize();

    // 2. Run unfused attention decode followed by separate sigmoid_mul
    ops::attention_decode(
        ctx, q_dev, k_dev, v_dev, out_sep_dev, scores_dev,
        num_heads, num_kv_heads, head_dim, cached_len,
        partials_ptr, nullptr);
    ops::sigmoid_mul(ctx, out_sep_dev, gate_dev, out_sep_dev, total_q);
    ctx.synchronize();

    std::vector<bf16> fused_host(total_q);
    std::vector<bf16> sep_host(total_q);
    ctx.memcpy_d2h(fused_host.data(), out_fused_dev.data(), total_q * sizeof(bf16));
    ctx.memcpy_d2h(sep_host.data(), out_sep_dev.data(), total_q * sizeof(bf16));

    int exact_mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < total_q; ++i) {
        if (fused_host[i] != sep_host[i]) {
            ++exact_mismatches;
            max_diff = std::max(max_diff, std::abs(static_cast<float>(fused_host[i]) - static_cast<float>(sep_host[i])));
        }
    }

    const std::string tag = "attention_decode_gate_fusion (h=" + std::to_string(num_heads) +
                            ", kv=" + std::to_string(num_kv_heads) +
                            ", dim=" + std::to_string(head_dim) +
                            ", len=" + std::to_string(cached_len) +
                            (with_partials ? ", partials=true" : ", partials=false") + ")";
    if (exact_mismatches != 0) {
        throw std::runtime_error(tag + " failed: " + std::to_string(exact_mismatches) +
                                 " mismatches, max_diff=" + std::to_string(max_diff));
    }
    std::cout << "[PASS] " << tag << " (0 bit mismatches)\n";
}

void test_attention_decode_gate_fusion(Context& ctx) {
    // Qwen4Exp geometry: head_dim=128, num_heads=20, num_kv_heads=4
    run_attention_decode_gate_fusion_case(ctx, 20, 4, 128, 32, 256, false);
    run_attention_decode_gate_fusion_case(ctx, 20, 4, 128, 160, 256, true);
    // Standard head_dim=256 geometry
    run_attention_decode_gate_fusion_case(ctx, 4, 2, 256, 32, 1024, false);
    run_attention_decode_gate_fusion_case(ctx, 4, 2, 256, 1024, 1024, true);
}

void test_split_qgkv(Context& ctx) {
    constexpr int num_heads = 16;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int q_proj_dim = num_heads * 2 * head_dim;
    constexpr int kv_dim = num_kv_heads * head_dim;
    constexpr int total_row = q_proj_dim + 2 * kv_dim;

    for (int seq_len : {1, 4, 32, 256}) {
        const size_t total_src = static_cast<size_t>(seq_len) * total_row;
        const size_t total_q = static_cast<size_t>(seq_len) * num_heads * head_dim;
        const size_t total_kv = static_cast<size_t>(seq_len) * num_kv_heads * head_dim;

        std::mt19937 rng(12345 + seq_len);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<bf16> src_host(total_src);
        for (size_t i = 0; i < total_src; ++i) src_host[i] = bf16(dist(rng));

        Tensor src_dev = Tensor::allocate(ctx, {seq_len, total_row}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(src_dev.data(), src_host.data(), total_src * sizeof(bf16));

        // 1. Unified split_qgkv
        Tensor q_dev1 = Tensor::allocate(ctx, {seq_len, (int64_t)num_heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor g_dev1 = Tensor::allocate(ctx, {seq_len, (int64_t)num_heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor k_dev1 = Tensor::allocate(ctx, {seq_len, (int64_t)num_kv_heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor v_dev1 = Tensor::allocate(ctx, {seq_len, (int64_t)num_kv_heads * head_dim}, dnnl::memory::data_type::bf16);

        ops::split_qgkv(ctx, src_dev, q_dev1, g_dev1, k_dev1, v_dev1, seq_len, num_heads, num_kv_heads, head_dim);

        // 2. Reference two-step: split_qkv + split_q_gate
        Tensor q_proj_dev = Tensor::allocate(ctx, {seq_len, q_proj_dim}, dnnl::memory::data_type::bf16);
        Tensor q_dev2 = Tensor::allocate(ctx, {seq_len, (int64_t)num_heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor g_dev2 = Tensor::allocate(ctx, {seq_len, (int64_t)num_heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor k_dev2 = Tensor::allocate(ctx, {seq_len, (int64_t)num_kv_heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor v_dev2 = Tensor::allocate(ctx, {seq_len, (int64_t)num_kv_heads * head_dim}, dnnl::memory::data_type::bf16);

        ops::split_qkv(ctx, src_dev, q_proj_dev, k_dev2, v_dev2, seq_len, q_proj_dim, kv_dim);
        ops::split_q_gate(ctx, q_proj_dev, q_dev2, g_dev2, seq_len, num_heads, head_dim);
        ctx.synchronize();

        std::vector<bf16> q1(total_q), q2(total_q), g1(total_q), g2(total_q);
        std::vector<bf16> k1(total_kv), k2(total_kv), v1(total_kv), v2(total_kv);
        ctx.memcpy_d2h(q1.data(), q_dev1.data(), total_q * sizeof(bf16));
        ctx.memcpy_d2h(q2.data(), q_dev2.data(), total_q * sizeof(bf16));
        ctx.memcpy_d2h(g1.data(), g_dev1.data(), total_q * sizeof(bf16));
        ctx.memcpy_d2h(g2.data(), g_dev2.data(), total_q * sizeof(bf16));
        ctx.memcpy_d2h(k1.data(), k_dev1.data(), total_kv * sizeof(bf16));
        ctx.memcpy_d2h(k2.data(), k_dev2.data(), total_kv * sizeof(bf16));
        ctx.memcpy_d2h(v1.data(), v_dev1.data(), total_kv * sizeof(bf16));
        ctx.memcpy_d2h(v2.data(), v_dev2.data(), total_kv * sizeof(bf16));

        if (q1 != q2 || g1 != g2 || k1 != k2 || v1 != v2) {
            throw std::runtime_error("test_split_qgkv bitwise mismatch at seq_len=" + std::to_string(seq_len));
        }
        std::cout << "[PASS] split_qgkv seq_len=" << seq_len << " (0 bit mismatches vs split_qkv + split_q_gate)\n";
    }
}

void test_ple_dilated_conv_silu_add_residual(Context& ctx) {
    constexpr int channels = 10240;
    constexpr int kernel_size = 4;
    constexpr int dilation = 3;
    constexpr int state_len = 9;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> conv_w(channels * kernel_size);
    for (size_t i = 0; i < conv_w.size(); ++i) conv_w[i] = bf16(dist(rng) * 0.1f);
    Tensor conv_w_dev = Tensor::allocate(ctx, {channels, kernel_size}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(conv_w_dev.data(), conv_w.data(), conv_w.size() * sizeof(bf16));

    for (int rows : {1, 4}) {
        std::vector<bf16> gv_normed(rows * channels), gv(rows * channels), h_init(rows * channels);
        for (size_t i = 0; i < gv_normed.size(); ++i) {
            gv_normed[i] = bf16(dist(rng));
            gv[i] = bf16(dist(rng));
            h_init[i] = bf16(dist(rng));
        }

        Tensor gv_normed_dev = Tensor::allocate(ctx, {rows, channels}, dnnl::memory::data_type::bf16);
        Tensor gv_dev = Tensor::allocate(ctx, {rows, channels}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(gv_normed_dev.data(), gv_normed.data(), gv_normed.size() * sizeof(bf16));
        ctx.memcpy_h2d(gv_dev.data(), gv.data(), gv.size() * sizeof(bf16));

        // 1. Fused path (accumulate directly into h1)
        Tensor h1_dev = Tensor::allocate(ctx, {rows, channels}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(h1_dev.data(), h_init.data(), h_init.size() * sizeof(bf16));
        Tensor state1_dev = Tensor::allocate(ctx, {state_len, channels}, dnnl::memory::data_type::f32);
        std::vector<float> zeros(state_len * channels, 0.0f);
        ctx.memcpy_h2d(state1_dev.data(), zeros.data(), zeros.size() * sizeof(float));
        int head1 = 0;

        ops::ple_dilated_conv_silu_add(
            ctx, static_cast<const bf16*>(gv_normed_dev.data()),
            static_cast<const bf16*>(gv_dev.data()),
            static_cast<float*>(state1_dev.data()),
            head1,
            static_cast<const bf16*>(conv_w_dev.data()),
            nullptr, rows, channels, kernel_size, dilation,
            static_cast<bf16*>(h1_dev.data()));

        // 2. Unfused path (out into ple_out, then residual_add into h2)
        Tensor h2_dev = Tensor::allocate(ctx, {rows, channels}, dnnl::memory::data_type::bf16);
        Tensor ple_out_dev = Tensor::allocate(ctx, {rows, channels}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(h2_dev.data(), h_init.data(), h_init.size() * sizeof(bf16));
        Tensor state2_dev = Tensor::allocate(ctx, {state_len, channels}, dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(state2_dev.data(), zeros.data(), zeros.size() * sizeof(float));
        int head2 = 0;

        ops::ple_dilated_conv_silu_add(
            ctx, static_cast<const bf16*>(gv_normed_dev.data()),
            static_cast<const bf16*>(gv_dev.data()),
            static_cast<float*>(state2_dev.data()),
            head2,
            static_cast<const bf16*>(conv_w_dev.data()),
            static_cast<bf16*>(ple_out_dev.data()), rows, channels, kernel_size, dilation,
            nullptr);
        ops::residual_add(ctx, h2_dev, ple_out_dev, rows * channels);
        ctx.synchronize();

        std::vector<bf16> h1(rows * channels), h2(rows * channels);
        ctx.memcpy_d2h(h1.data(), h1_dev.data(), h1.size() * sizeof(bf16));
        ctx.memcpy_d2h(h2.data(), h2_dev.data(), h2.size() * sizeof(bf16));

        if (h1 != h2) {
            throw std::runtime_error("ple_dilated_conv_silu_add residual fusion mismatch at rows=" + std::to_string(rows));
        }
        std::cout << "[PASS] ple_dilated_conv_silu_add residual fusion rows=" << rows << " (0 bit mismatches)\n";
    }
}

void test_attention_prefill_gate_fusion(Context& ctx) {
    for (auto [head_dim, seq_len] : {
             std::pair{128, 8},
             std::pair{128, 256},
             std::pair{256, 8}
         }) {
        const int num_heads = (head_dim == 128) ? 8 : 4;
        const int num_kv_heads = 2;
        const int q_dim = num_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;

        std::mt19937 rng(888 + head_dim + seq_len);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<bf16> q_h(seq_len * q_dim), k_h(seq_len * kv_dim), v_h(seq_len * kv_dim), z_h(seq_len * q_dim);
        for (auto& x : q_h) x = bf16(dist(rng));
        for (auto& x : k_h) x = bf16(dist(rng));
        for (auto& x : v_h) x = bf16(dist(rng));
        for (auto& x : z_h) x = bf16(dist(rng));

        Tensor q_dev = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
        Tensor k_dev = Tensor::allocate(ctx, {seq_len, kv_dim}, dnnl::memory::data_type::bf16);
        Tensor v_dev = Tensor::allocate(ctx, {seq_len, kv_dim}, dnnl::memory::data_type::bf16);
        Tensor z_dev = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);

        ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(k_dev.data(), k_h.data(), k_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(v_dev.data(), v_h.data(), v_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(z_dev.data(), z_h.data(), z_h.size() * sizeof(bf16));

        Tensor out_fused = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
        Tensor out_sep = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
        Tensor scores = Tensor::allocate(ctx, {num_heads, seq_len, seq_len}, dnnl::memory::data_type::f32);

        // 1. Fused attention_prefill
        ops::attention_prefill(ctx, q_dev, k_dev, v_dev, out_fused, scores, seq_len, num_heads, num_kv_heads, head_dim, &z_dev);
        ctx.synchronize();

        // 2. Unfused attention_prefill + sigmoid_mul
        ops::attention_prefill(ctx, q_dev, k_dev, v_dev, out_sep, scores, seq_len, num_heads, num_kv_heads, head_dim, nullptr);
        ops::sigmoid_mul(ctx, out_sep, z_dev, out_sep, seq_len * q_dim);
        ctx.synchronize();

        std::vector<bf16> fused(seq_len * q_dim), sep(seq_len * q_dim);
        ctx.memcpy_d2h(fused.data(), out_fused.data(), fused.size() * sizeof(bf16));
        ctx.memcpy_d2h(sep.data(), out_sep.data(), sep.size() * sizeof(bf16));

        float max_diff = 0.0f;
        int bit_diffs = 0;
        for (size_t i = 0; i < fused.size(); ++i) {
            float f_v = static_cast<float>(fused[i]);
            float s_v = static_cast<float>(sep[i]);
            float diff = std::abs(f_v - s_v);
            if (diff > max_diff) max_diff = diff;
            if (fused[i] != sep[i]) bit_diffs++;
        }
        if (max_diff > 0.002f) {
            throw std::runtime_error("attention_prefill gate fusion mismatch at head_dim=" +
                                     std::to_string(head_dim) + " seq_len=" + std::to_string(seq_len) +
                                     " max_diff=" + std::to_string(max_diff));
        }
        std::cout << "[PASS] attention_prefill gate fusion head_dim=" << head_dim
                  << " seq_len=" << seq_len
                  << " (max_diff=" << max_diff << " <= 0.002, 1-ULP diffs=" << bit_diffs << "/" << fused.size() << ")\n" << std::flush;
    }
}

void test_attention_prefill_cached_gate_fusion(Context& ctx) {
    for (auto [head_dim, seq_len, start_pos] : {
             std::tuple{128, 8, 16},
             std::tuple{128, 128, 128},
             std::tuple{256, 8, 16}
         }) {
        const int num_heads = (head_dim == 128) ? 8 : 4;
        const int num_kv_heads = 2;
        const int max_seq_len = 512;
        const int q_dim = num_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;
        const int total_len = start_pos + seq_len;

        std::mt19937 rng(999 + head_dim + seq_len);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<bf16> q_h(seq_len * q_dim), k_h(num_kv_heads * max_seq_len * head_dim),
                          v_h(num_kv_heads * max_seq_len * head_dim), z_h(seq_len * q_dim);
        for (auto& x : q_h) x = bf16(dist(rng));
        for (auto& x : k_h) x = bf16(dist(rng));
        for (auto& x : v_h) x = bf16(dist(rng));
        for (auto& x : z_h) x = bf16(dist(rng));

        Tensor q_dev = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
        Tensor k_dev = Tensor::allocate(ctx, {num_kv_heads, max_seq_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor v_dev = Tensor::allocate(ctx, {num_kv_heads, max_seq_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor z_dev = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);

        ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(k_dev.data(), k_h.data(), k_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(v_dev.data(), v_h.data(), v_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(z_dev.data(), z_h.data(), z_h.size() * sizeof(bf16));

        Tensor out_fused = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
        Tensor out_sep = Tensor::allocate(ctx, {seq_len, q_dim}, dnnl::memory::data_type::bf16);
        Tensor scores = Tensor::allocate(ctx, {num_heads, seq_len, total_len}, dnnl::memory::data_type::f32);

        // 1. Fused attention_prefill_cached
        ops::attention_prefill_cached(ctx, q_dev, k_dev, v_dev, out_fused, scores,
                                      seq_len, start_pos, num_heads, num_kv_heads, head_dim, max_seq_len, &z_dev);
        ctx.synchronize();

        // 2. Unfused attention_prefill_cached + sigmoid_mul
        ops::attention_prefill_cached(ctx, q_dev, k_dev, v_dev, out_sep, scores,
                                      seq_len, start_pos, num_heads, num_kv_heads, head_dim, max_seq_len, nullptr);
        ops::sigmoid_mul(ctx, out_sep, z_dev, out_sep, seq_len * q_dim);
        ctx.synchronize();

        std::vector<bf16> fused(seq_len * q_dim), sep(seq_len * q_dim);
        ctx.memcpy_d2h(fused.data(), out_fused.data(), fused.size() * sizeof(bf16));
        ctx.memcpy_d2h(sep.data(), out_sep.data(), sep.size() * sizeof(bf16));

        float max_diff = 0.0f;
        int bit_diffs = 0;
        for (size_t i = 0; i < fused.size(); ++i) {
            float f_v = static_cast<float>(fused[i]);
            float s_v = static_cast<float>(sep[i]);
            float diff = std::abs(f_v - s_v);
            if (diff > max_diff) max_diff = diff;
            if (fused[i] != sep[i]) bit_diffs++;
        }
        if (max_diff > 0.002f) {
            throw std::runtime_error("attention_prefill_cached gate fusion mismatch at head_dim=" +
                                     std::to_string(head_dim) + " start_pos=" + std::to_string(start_pos) +
                                     " seq_len=" + std::to_string(seq_len) + " max_diff=" + std::to_string(max_diff));
        }
        std::cout << "[PASS] attention_prefill_cached gate fusion head_dim=" << head_dim
                  << " seq_len=" << seq_len << " start_pos=" << start_pos
                  << " (max_diff=" << max_diff << " <= 0.002, 1-ULP diffs=" << bit_diffs << "/" << fused.size() << ")\n" << std::flush;
    }
}

static void test_qsa_prepare_q_and_cache_k(Context& ctx) {
    std::cout << "[Test] Running QSA prepare Q and cache K test...\n";
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int kTotalDim = 640;
    constexpr int capacity = 32;
    constexpr int pos = 7;
    constexpr float theta = 10000000.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> qk_h(kTotalDim);
    std::vector<bf16> qn_w_h(kHeadDim);
    std::vector<bf16> k_cache_init(capacity * kHeadDim, bf16(0.0f));

    for (auto& v : qk_h) v = bf16(dist(rng));
    for (auto& v : qn_w_h) v = bf16(1.0f + dist(rng) * 0.1f);

    // Host reference run
    std::vector<bf16> k_cache_ref = k_cache_init;
    std::vector<bf16> q_out_ref(kNumHeads * kHeadDim);
    ops::qsa_prepare_q_and_cache_k_host(
        qk_h.data(), qn_w_h.data(), k_cache_ref.data(), q_out_ref.data(),
        pos, capacity, theta);

    // GPU device run
    Tensor qk_dev = Tensor::allocate(ctx, {1, kTotalDim}, dnnl::memory::data_type::bf16);
    Tensor qn_dev = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_dev = Tensor::allocate(ctx, {capacity, kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor q_out_dev = Tensor::allocate(ctx, {1, kNumHeads * kHeadDim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(qk_dev.data(), qk_h.data(), qk_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(qn_dev.data(), qn_w_h.data(), qn_w_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_dev.data(), k_cache_init.data(), k_cache_init.size() * sizeof(bf16));

    ops::qsa_prepare_q_and_cache_k(ctx, qk_dev, qn_dev, k_cache_dev, q_out_dev, pos, theta);
    ctx.synchronize();

    std::vector<bf16> q_out_gpu(kNumHeads * kHeadDim);
    std::vector<bf16> k_cache_gpu(capacity * kHeadDim);
    ctx.memcpy_d2h(q_out_gpu.data(), q_out_dev.data(), q_out_gpu.size() * sizeof(bf16));
    ctx.memcpy_d2h(k_cache_gpu.data(), k_cache_dev.data(), k_cache_gpu.size() * sizeof(bf16));

    // Check q_out
    float max_q_diff = 0.0f;
    for (size_t i = 0; i < q_out_gpu.size(); ++i) {
        float diff = std::abs(static_cast<float>(q_out_gpu[i]) - static_cast<float>(q_out_ref[i]));
        if (diff > max_q_diff) max_q_diff = diff;
    }
    if (max_q_diff > 0.01f) {
        throw std::runtime_error("test_qsa_prepare_q_and_cache_k: q_out mismatch, max_diff=" + std::to_string(max_q_diff));
    }

    // Check k_cache at pos
    for (int d = 0; d < kHeadDim; ++d) {
        if (k_cache_gpu[pos * kHeadDim + d] != qk_h[kNumHeads * kHeadDim + d]) {
            throw std::runtime_error("test_qsa_prepare_q_and_cache_k: k_cache mismatch at d=" + std::to_string(d));
        }
    }
    std::cout << "[PASS] QSA prepare Q and cache K (max_q_diff=" << max_q_diff << " <= 0.01)\n";
}

static void test_qsa_cache_k_batched(Context& ctx) {
    std::cout << "[Test] Running QSA batched K cache test...\n";
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int kTotalDim = 640;
    constexpr int capacity = 64;
    constexpr int start_pos = 16;
    constexpr int count = 20;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> qk_h(count * kTotalDim);
    std::vector<bf16> k_cache_init(capacity * kHeadDim, bf16(0.0f));
    for (auto& v : qk_h) v = bf16(dist(rng));

    Tensor qk_dev = Tensor::allocate(ctx, {count, kTotalDim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_dev = Tensor::allocate(ctx, {capacity, kHeadDim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(qk_dev.data(), qk_h.data(), qk_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_dev.data(), k_cache_init.data(), k_cache_init.size() * sizeof(bf16));

    ops::qsa_cache_k_batched(ctx, qk_dev, k_cache_dev, start_pos, count);
    ctx.synchronize();

    std::vector<bf16> k_cache_gpu(capacity * kHeadDim);
    ctx.memcpy_d2h(k_cache_gpu.data(), k_cache_dev.data(), k_cache_gpu.size() * sizeof(bf16));

    for (int t = 0; t < count; ++t) {
        for (int d = 0; d < kHeadDim; ++d) {
            bf16 expected = qk_h[t * kTotalDim + kNumHeads * kHeadDim + d];
            bf16 actual = k_cache_gpu[(start_pos + t) * kHeadDim + d];
            if (expected != actual) {
                throw std::runtime_error("test_qsa_cache_k_batched mismatch at t=" + std::to_string(t) + " d=" + std::to_string(d));
            }
        }
    }
    std::cout << "[PASS] QSA batched K cache (count=" << count << ", 0 bit mismatches)\n";
}

static void test_qsa_compute_block_scores(Context& ctx) {
    std::cout << "[Test] Running QSA compute block scores test...\n";
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int num_blocks = 32;
    constexpr int num_tokens = num_blocks * 4;
    constexpr float theta = 10000000.0f;

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> q_h(kNumHeads * kHeadDim);
    std::vector<bf16> k_cache_h(num_tokens * kHeadDim);
    std::vector<bf16> kn_w_h(kHeadDim);
    for (auto& v : q_h) v = bf16(dist(rng));
    for (auto& v : k_cache_h) v = bf16(dist(rng));
    for (auto& v : kn_w_h) v = bf16(1.0f + dist(rng) * 0.1f);

    // Host reference run
    std::vector<float> scores_ref(num_blocks);
    ops::qsa_compute_block_scores_host(
        q_h.data(), k_cache_h.data(), kn_w_h.data(), scores_ref.data(),
        num_blocks, num_tokens, theta);

    // GPU device run
    Tensor q_dev = Tensor::allocate(ctx, {1, kNumHeads * kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor k_cache_dev = Tensor::allocate(ctx, {num_tokens, kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor kn_dev = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_cache_dev.data(), k_cache_h.data(), k_cache_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(kn_dev.data(), kn_w_h.data(), kn_w_h.size() * sizeof(bf16));

    ops::qsa_compute_block_scores(ctx, q_dev, k_cache_dev, kn_dev, scores_dev, num_blocks, theta);
    ctx.synchronize();

    std::vector<float> scores_gpu(num_blocks);
    ctx.memcpy_d2h(scores_gpu.data(), scores_dev.data(), scores_gpu.size() * sizeof(float));

    float max_diff = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        float diff = std::abs(scores_gpu[b] - scores_ref[b]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 0.01f) {
        throw std::runtime_error("test_qsa_compute_block_scores: score mismatch, max_diff=" + std::to_string(max_diff));
    }
    std::cout << "[PASS] QSA compute block scores (num_blocks=" << num_blocks << ", max_diff=" << max_diff << " <= 0.01)\n";
}

static void test_qsa_select_topk_and_expand_tokens(Context& ctx) {
    std::cout << "[Test] Running QSA select top-k and expand tokens test...\n";
    // Case 1: num_complete_blocks > 512 (e.g. 600 blocks)
    {
        constexpr int num_blocks = 600;
        constexpr int block_topk = 512;
        constexpr int compress_ratio = 4;
        constexpr int cached_len = num_blocks * compress_ratio + 3; // 2403 tokens

        std::mt19937 rng(789);
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        std::vector<float> scores_h(num_blocks);
        for (auto& s : scores_h) s = dist(rng);

        std::vector<int> tokens_ref(block_topk * compress_ratio + 3);
        int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_h.data(), num_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());

        Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
        Tensor tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), scores_h.size() * sizeof(float));

        int gpu_count = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_blocks, block_topk, cached_len, compress_ratio, tokens_dev);
        ctx.synchronize();

        if (gpu_count != ref_count) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: token count mismatch, gpu=" +
                                     std::to_string(gpu_count) + " ref=" + std::to_string(ref_count));
        }

        std::vector<int> tokens_gpu(gpu_count);
        ctx.memcpy_d2h(tokens_gpu.data(), tokens_dev.data(), gpu_count * sizeof(int));

        for (int i = 0; i < gpu_count; ++i) {
            if (tokens_gpu[i] != tokens_ref[i]) {
                throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: index mismatch at i=" +
                                         std::to_string(i) + " gpu=" + std::to_string(tokens_gpu[i]) +
                                         " ref=" + std::to_string(tokens_ref[i]));
            }
        }
        std::cout << "[PASS] QSA select top-512 and expand tokens (num_blocks=600 -> " << gpu_count << " active tokens, 0 mismatches)\n";
    }

    // Case 2: num_complete_blocks <= 512 (e.g. 50 blocks)
    {
        constexpr int num_blocks = 50;
        constexpr int block_topk = 512;
        constexpr int compress_ratio = 4;
        constexpr int cached_len = num_blocks * compress_ratio + 2; // 202 tokens

        std::vector<float> scores_h(num_blocks, 1.0f);
        std::vector<int> tokens_ref(cached_len);
        int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_h.data(), num_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());

        Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
        Tensor tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), scores_h.size() * sizeof(float));

        int gpu_count = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_blocks, block_topk, cached_len, compress_ratio, tokens_dev);
        ctx.synchronize();

        if (gpu_count != ref_count || gpu_count != cached_len) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: dense count mismatch");
        }
        std::vector<int> tokens_gpu(gpu_count);
        ctx.memcpy_d2h(tokens_gpu.data(), tokens_dev.data(), gpu_count * sizeof(int));
        for (int i = 0; i < gpu_count; ++i) {
            if (tokens_gpu[i] != i) {
                throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: dense identity mismatch at " + std::to_string(i));
            }
        }
        std::cout << "[PASS] QSA select top-k dense identity fallback (cached_len=202, 0 mismatches)\n";
    }

    // Case 3: Clustered/tied scores (blocks with highest scores placed at the tail of the block array)
    {
        constexpr int num_blocks = 600;
        constexpr int block_topk = 512;
        constexpr int compress_ratio = 4;
        constexpr int cached_len = num_blocks * compress_ratio + 1; // 2401 tokens (1 tail token)

        std::vector<float> scores_h(num_blocks);
        for (int i = 0; i < 100; ++i) scores_h[i] = 1.0f;
        for (int i = 100; i < 300; ++i) scores_h[i] = 2.0f;
        for (int i = 300; i < 500; ++i) scores_h[i] = 3.0f;
        for (int i = 500; i < 600; ++i) scores_h[i] = 5.0f; // highest scores at the end

        std::vector<int> tokens_ref(block_topk * compress_ratio + 1);
        int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_h.data(), num_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());

        Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
        Tensor tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), scores_h.size() * sizeof(float));

        int gpu_count = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_blocks, block_topk, cached_len, compress_ratio, tokens_dev);
        ctx.synchronize();

        if (gpu_count != ref_count) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: clustered count mismatch");
        }
        std::vector<int> tokens_gpu(gpu_count);
        ctx.memcpy_d2h(tokens_gpu.data(), tokens_dev.data(), gpu_count * sizeof(int));
        for (int i = 0; i < gpu_count; ++i) {
            if (tokens_gpu[i] != tokens_ref[i]) {
                throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: clustered index mismatch at i=" +
                                         std::to_string(i) + " gpu=" + std::to_string(tokens_gpu[i]) +
                                         " ref=" + std::to_string(tokens_ref[i]));
            }
        }
        std::cout << "[PASS] QSA select top-k clustered/tied scores (600 blocks -> " << gpu_count << " active tokens, 0 mismatches vs host)\n";
    }

    // Case 4: All-zero scores (common when all QK dot products <= 0 after ReLU)
    {
        constexpr int num_blocks = 600;
        constexpr int block_topk = 512;
        constexpr int compress_ratio = 4;
        constexpr int cached_len = num_blocks * compress_ratio;

        std::vector<float> scores_h(num_blocks, 0.0f);
        std::vector<int> tokens_ref(block_topk * compress_ratio);
        int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_h.data(), num_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());

        Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
        Tensor tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), scores_h.size() * sizeof(float));

        int gpu_count = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_blocks, block_topk, cached_len, compress_ratio, tokens_dev);
        ctx.synchronize();

        if (gpu_count != ref_count) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: all-zero count mismatch");
        }
        std::vector<int> tokens_gpu(gpu_count);
        ctx.memcpy_d2h(tokens_gpu.data(), tokens_dev.data(), gpu_count * sizeof(int));
        for (int i = 0; i < gpu_count; ++i) {
            if (tokens_gpu[i] != tokens_ref[i]) {
                throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: all-zero index mismatch at i=" +
                                         std::to_string(i) + " gpu=" + std::to_string(tokens_gpu[i]) +
                                         " ref=" + std::to_string(tokens_ref[i]));
            }
        }
        std::cout << "[PASS] QSA select top-k all-zero scores (0 mismatches vs host)\n";
    }

    // Case 5: Large Top-K (block_topk = 1024 > 512, verifies dynamic SLM sizing)
    {
        constexpr int num_blocks = 1200;
        constexpr int block_topk = 1024;
        constexpr int compress_ratio = 4;
        constexpr int cached_len = num_blocks * compress_ratio + 2; // 4802 tokens

        std::vector<float> scores_h(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            scores_h[i] = static_cast<float>((i * 73) % 200);
        }

        std::vector<int> tokens_ref(block_topk * compress_ratio + 2);
        int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_h.data(), num_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());

        Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
        Tensor tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), scores_h.size() * sizeof(float));

        int gpu_count = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_blocks, block_topk, cached_len, compress_ratio, tokens_dev);
        ctx.synchronize();

        if (gpu_count != ref_count) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: large top-k count mismatch");
        }
        std::vector<int> tokens_gpu(gpu_count);
        ctx.memcpy_d2h(tokens_gpu.data(), tokens_dev.data(), gpu_count * sizeof(int));
        for (int i = 0; i < gpu_count; ++i) {
            if (tokens_gpu[i] != tokens_ref[i]) {
                throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: large top-k index mismatch at i=" +
                                         std::to_string(i) + " gpu=" + std::to_string(tokens_gpu[i]) +
                                         " ref=" + std::to_string(tokens_ref[i]));
            }
        }
        std::cout << "[PASS] QSA select top-k large top-k (block_topk=1024 > 512, " << gpu_count << " active tokens, 0 mismatches vs host)\n";
    }

    // Case 6: Edge parameter boundary conditions (non-positive cached_len or block_topk)
    {
        Tensor dummy_scores = Tensor::allocate(ctx, {10}, dnnl::memory::data_type::f32);
        Tensor dummy_tokens = Tensor::allocate(ctx, {10}, dnnl::memory::data_type::s32);
        int count_zero_len = ops::qsa_select_topk_and_expand_tokens(ctx, dummy_scores, 10, 5, 0, 4, dummy_tokens);
        int count_neg_len = ops::qsa_select_topk_and_expand_tokens(ctx, dummy_scores, 10, 5, -4, 4, dummy_tokens);
        int count_zero_topk = ops::qsa_select_topk_and_expand_tokens(ctx, dummy_scores, 10, 0, 16, 4, dummy_tokens);
        int count_neg_topk = ops::qsa_select_topk_and_expand_tokens(ctx, dummy_scores, 10, -1, 16, 4, dummy_tokens);
        if (count_zero_len != 0 || count_neg_len != 0 || count_zero_topk != 0 || count_neg_topk != 0) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: non-positive parameter guard failed");
        }
        std::cout << "[PASS] QSA select top-k non-positive parameter guards verified (clean zero returns)\n";
    }

    // Case 7: Distinct small scores inside the former floating-threshold interval.
    {
        constexpr int num_blocks = 514;
        constexpr int block_topk = 512;
        constexpr int compress_ratio = 4;
        constexpr int cached_len = num_blocks * compress_ratio;
        std::vector<float> scores_h(num_blocks, 1000000.0f);
        scores_h[0] = 0.0001f;
        scores_h[1] = 0.0002f;
        scores_h[2] = 0.0f;

        std::vector<int> tokens_ref(block_topk * compress_ratio);
        int ref_count = ops::qsa_select_topk_and_expand_tokens_host(
            scores_h.data(), num_blocks, block_topk, cached_len, compress_ratio, tokens_ref.data());
        Tensor scores_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
        Tensor tokens_dev = Tensor::allocate(ctx, {cached_len}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores_h.data(), scores_h.size() * sizeof(float));

        int gpu_count = ops::qsa_select_topk_and_expand_tokens(
            ctx, scores_dev, num_blocks, block_topk, cached_len, compress_ratio, tokens_dev);
        ctx.synchronize();
        if (gpu_count != ref_count) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: threshold regression count mismatch");
        }
        std::vector<int> tokens_gpu(gpu_count);
        ctx.memcpy_d2h(tokens_gpu.data(), tokens_dev.data(), gpu_count * sizeof(int));
        for (int i = 0; i < gpu_count; ++i) {
            if (tokens_gpu[i] != tokens_ref[i]) {
                throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: exact threshold mismatch at i=" +
                                         std::to_string(i) + " gpu=" + std::to_string(tokens_gpu[i]) +
                                         " ref=" + std::to_string(tokens_ref[i]));
            }
        }
        if (tokens_gpu[0] != 4 || tokens_gpu[4] != 12) {
            throw std::runtime_error("test_qsa_select_topk_and_expand_tokens: exact selection must keep block 1 and drop block 0/2");
        }
        std::cout << "[PASS] QSA exact top-k selects 0.0002 over 0.0001 in the former unresolved threshold interval\n";
    }
}

static void test_attention_decode_with_qsa_indices(Context& ctx) {
    std::cout << "[Test] Running Attention Decode with QSA Active Indices test...\n" << std::flush;
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int cached_len = 2400; // > 2048
    constexpr int num_active = 2051; // 512 blocks * 4 + 3 tail tokens

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> q_h(num_heads * head_dim);
    for (auto& v : q_h) v = bf16(dist(rng));

    // Active indices: select 512 distinct blocks out of 600 complete blocks, plus 3 tail tokens
    std::vector<int> active_indices_h(num_active);
    for (int b = 0; b < 512; ++b) {
        int blk = b;
        for (int c = 0; c < 4; ++c) {
            active_indices_h[b * 4 + c] = blk * 4 + c;
        }
    }
    for (int c = 0; c < 3; ++c) {
        active_indices_h[512 * 4 + c] = 2397 + c;
    }

    Tensor q_dev = Tensor::allocate(ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(ctx, {num_kv_heads, cached_len, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(ctx, {num_kv_heads, cached_len, head_dim}, dnnl::memory::data_type::bf16);
    Tensor out_dev = Tensor::allocate(ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor scores_buf = Tensor::allocate(ctx, {num_heads, cached_len}, dnnl::memory::data_type::f32);
    Tensor active_idx_dev = Tensor::allocate(ctx, {num_active}, dnnl::memory::data_type::s32);

    constexpr int kDecodeExactTile = 128;
    const int partial_stride = 272;
    const int max_tiles = (cached_len + kDecodeExactTile - 1) / kDecodeExactTile;
    Tensor partials_dev = Tensor::allocate(ctx, {(int64_t)num_heads, (int64_t)max_tiles, (int64_t)partial_stride}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(active_idx_dev.data(), active_indices_h.data(), active_indices_h.size() * sizeof(int));

    std::vector<bf16> kv_init(num_kv_heads * cached_len * head_dim);
    for (size_t i = 0; i < kv_init.size(); ++i) kv_init[i] = bf16(dist(rng) * 0.1f);
    ctx.memcpy_h2d(k_dev.data(), kv_init.data(), kv_init.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), kv_init.data(), kv_init.size() * sizeof(bf16));

    ops::attention_decode(
        ctx, q_dev, k_dev, v_dev, out_dev, scores_buf,
        num_heads, num_kv_heads, head_dim, cached_len,
        &partials_dev, /*gate_buf=*/nullptr,
        active_idx_dev.data_as<int>(), num_active);
    ctx.synchronize();

    std::vector<bf16> out_h(num_heads * head_dim);
    ctx.memcpy_d2h(out_h.data(), out_dev.data(), out_h.size() * sizeof(bf16));

    // Deep verification: compute CPU reference attention with active_indices and compare
    std::vector<float> ref_out(num_heads * head_dim, 0.0f);
    const int heads_per_kv = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    float max_attn_diff = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = h / heads_per_kv;
        std::vector<float> scores(num_active);
        float max_s = -1e30f;
        for (int t = 0; t < num_active; ++t) {
            int cache_t = active_indices_h[t];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<float>(q_h[h * head_dim + d]) *
                       static_cast<float>(kv_init[kv_head * cached_len * head_dim + cache_t * head_dim + d]);
            }
            float s = dot * scale;
            scores[t] = s;
            if (s > max_s) max_s = s;
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < num_active; ++t) {
            scores[t] = std::exp(scores[t] - max_s);
            sum_exp += scores[t];
        }
        const float inv_sum = 1.0f / sum_exp;
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int t = 0; t < num_active; ++t) {
                int cache_t = active_indices_h[t];
                acc += scores[t] * static_cast<float>(kv_init[kv_head * cached_len * head_dim + cache_t * head_dim + d]);
            }
            float expected_val = static_cast<float>(bf16(acc * inv_sum));
            float actual_val = static_cast<float>(out_h[h * head_dim + d]);
            if (std::isnan(actual_val) || std::isinf(actual_val)) {
                throw std::runtime_error("test_attention_decode_with_qsa_indices: NaN or Inf detected at h=" +
                                         std::to_string(h) + " d=" + std::to_string(d));
            }
            float diff = std::abs(actual_val - expected_val);
            if (diff > max_attn_diff) max_attn_diff = diff;
        }
    }

    if (max_attn_diff > 0.015f) {
        throw std::runtime_error("test_attention_decode_with_qsa_indices: attention output mismatch against CPU reference, max_diff=" +
                                 std::to_string(max_attn_diff));
    }

    std::cout << "[PASS] Attention decode with QSA active indices (cached_len=" << cached_len
              << ", num_active=" << num_active << ", max_attn_diff=" << max_attn_diff << " <= 0.015)\n";

    // Part 2: Non-contiguous / sparse active block indices with Epilogue Gate Fusion
    {
        constexpr int num_active_sparse = 512 * 4 + 2; // 2050 tokens
        std::vector<int> sparse_indices_h(num_active_sparse);
        // Select 512 sparse blocks out of 600 complete blocks (skipping every 7th block to create realistic non-contiguous gaps within cached_len=2400)
        int picked_blocks = 0;
        for (int b = 0; b < 600 && picked_blocks < 512; ++b) {
            if ((b % 7) == 3) continue; // gap
            for (int c = 0; c < 4; ++c) {
                sparse_indices_h[picked_blocks * 4 + c] = b * 4 + c;
            }
            picked_blocks++;
        }
        if (picked_blocks != 512) {
            throw std::runtime_error("test_attention_decode_with_qsa_indices: failed to pick 512 sparse blocks");
        }
        sparse_indices_h[512 * 4 + 0] = cached_len - 2;
        sparse_indices_h[512 * 4 + 1] = cached_len - 1;

        std::vector<bf16> gate_h(num_heads * head_dim);
        for (auto& v : gate_h) v = bf16(dist(rng) * 0.5f);

        Tensor sparse_idx_dev = Tensor::allocate(ctx, {num_active_sparse}, dnnl::memory::data_type::s32);
        Tensor gate_dev = Tensor::allocate(ctx, {1, num_heads * head_dim}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(sparse_idx_dev.data(), sparse_indices_h.data(), sparse_indices_h.size() * sizeof(int));
        ctx.memcpy_h2d(gate_dev.data(), gate_h.data(), gate_h.size() * sizeof(bf16));

        ops::attention_decode(
            ctx, q_dev, k_dev, v_dev, out_dev, scores_buf,
            num_heads, num_kv_heads, head_dim, cached_len,
            &partials_dev, &gate_dev,
            sparse_idx_dev.data_as<int>(), num_active_sparse);
        ctx.synchronize();

        ctx.memcpy_d2h(out_h.data(), out_dev.data(), out_h.size() * sizeof(bf16));

        float max_gate_diff = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            const int kv_head = h / heads_per_kv;
            std::vector<float> scores(num_active_sparse);
            float max_s = -1e30f;
            for (int t = 0; t < num_active_sparse; ++t) {
                int cache_t = sparse_indices_h[t];
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += static_cast<float>(q_h[h * head_dim + d]) *
                           static_cast<float>(kv_init[kv_head * cached_len * head_dim + cache_t * head_dim + d]);
                }
                float s = dot * scale;
                scores[t] = s;
                if (s > max_s) max_s = s;
            }
            float sum_exp = 0.0f;
            for (int t = 0; t < num_active_sparse; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                sum_exp += scores[t];
            }
            const float inv_sum = 1.0f / sum_exp;
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < num_active_sparse; ++t) {
                    int cache_t = sparse_indices_h[t];
                    acc += scores[t] * static_cast<float>(kv_init[kv_head * cached_len * head_dim + cache_t * head_dim + d]);
                }
                float val = static_cast<float>(bf16(acc * inv_sum));
                float g_v = static_cast<float>(gate_h[h * head_dim + d]);
                float s = 1.0f / (1.0f + std::exp(-g_v));
                val = static_cast<float>(bf16(val * s));
                float expected_val = val;
                float actual_val = static_cast<float>(out_h[h * head_dim + d]);
                if (std::isnan(actual_val) || std::isinf(actual_val)) {
                    throw std::runtime_error("test_attention_decode_with_qsa_indices (gate): NaN/Inf detected");
                }
                float diff = std::abs(actual_val - expected_val);
                if (diff > max_gate_diff) max_gate_diff = diff;
            }
        }
        if (max_gate_diff > 0.015f) {
            throw std::runtime_error("test_attention_decode_with_qsa_indices (gate): mismatch, diff=" + std::to_string(max_gate_diff));
        }
        std::cout << "[PASS] Attention decode with sparse non-contiguous QSA indices & Epilogue Gate Fusion (max_gate_diff="
                  << max_gate_diff << " <= 0.015)\n";
    }

    // Part 3: Baseline decode fallback with active indices (partials_buf == nullptr)
    {
        ops::attention_decode(
            ctx, q_dev, k_dev, v_dev, out_dev, scores_buf,
            num_heads, num_kv_heads, head_dim, cached_len,
            /*exact_partials_buf=*/nullptr, /*gate_buf=*/nullptr,
            active_idx_dev.data_as<int>(), num_active);
        ctx.synchronize();

        ctx.memcpy_d2h(out_h.data(), out_dev.data(), out_h.size() * sizeof(bf16));
        float max_base_diff = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float actual_val = static_cast<float>(out_h[h * head_dim + d]);
                float expected_val = ref_out[h * head_dim + d];
                float diff = std::abs(actual_val - expected_val);
                if (diff > max_base_diff) max_base_diff = diff;
            }
        }
        if (max_base_diff > 0.015f) {
            throw std::runtime_error("test_attention_decode_with_qsa_indices (baseline fallback): mismatch, diff=" + std::to_string(max_base_diff));
        }
        std::cout << "[PASS] Attention decode baseline fallback with QSA active indices (max_base_diff="
                  << max_base_diff << " <= 0.015)\n";
    }
}

static void test_qsa_compressed_k_scoring(Context& ctx) {
    std::cout << "[Test] Running QSA compressed K state (A1) and scoring tests...\n";
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int num_blocks = 64;
    constexpr int num_tokens = num_blocks * 4;
    constexpr float theta = 10000000.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> raw_k_h(num_tokens * kHeadDim);
    std::vector<bf16> kn_w_h(kHeadDim);
    std::vector<bf16> q_h(kNumHeads * kHeadDim);
    for (auto& v : raw_k_h) v = bf16(dist(rng) * 0.3f);
    for (auto& v : kn_w_h) v = bf16(1.0f + dist(rng) * 0.1f);
    for (auto& v : q_h) v = bf16(dist(rng) * 0.2f);

    // 1. Host compression
    std::vector<bf16> comp_k_host(num_blocks * kHeadDim);
    ops::qsa_compress_key_blocks_host(
        raw_k_h.data(), kn_w_h.data(), comp_k_host.data(), 0, num_blocks, theta);

    // 2. GPU compression
    Tensor raw_k_dev = Tensor::allocate(ctx, {num_tokens, kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor kn_w_dev = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor comp_k_dev = Tensor::allocate(ctx, {num_blocks, kHeadDim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(raw_k_dev.data(), raw_k_h.data(), raw_k_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(kn_w_dev.data(), kn_w_h.data(), kn_w_h.size() * sizeof(bf16));

    ops::qsa_compress_key_blocks(ctx, raw_k_dev, kn_w_dev, comp_k_dev, 0, num_blocks, theta);
    ctx.synchronize();

    std::vector<bf16> comp_k_gpu(num_blocks * kHeadDim);
    ctx.memcpy_d2h(comp_k_gpu.data(), comp_k_dev.data(), comp_k_gpu.size() * sizeof(bf16));

    float max_comp_diff = 0.0f;
    for (size_t i = 0; i < comp_k_host.size(); ++i) {
        float diff = std::abs(static_cast<float>(comp_k_gpu[i]) - static_cast<float>(comp_k_host[i]));
        if (diff > max_comp_diff) max_comp_diff = diff;
    }
    if (max_comp_diff > 0.01f) {
        throw std::runtime_error("test_qsa_compressed_k_scoring: compressed K GPU vs Host mismatch, diff=" +
                                 std::to_string(max_comp_diff));
    }
    std::cout << "[PASS] QSA key block compression verified (num_blocks=" << num_blocks
              << ", max_diff=" << max_comp_diff << " <= 0.01)\n";

    // 3. Compare scoring: GPU uncompressed vs GPU compressed
    Tensor q_dev = Tensor::allocate(ctx, {1, kNumHeads * kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor scores_uncomp_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
    Tensor scores_comp_dev = Tensor::allocate(ctx, {num_blocks}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));

    ops::qsa_compute_block_scores(ctx, q_dev, raw_k_dev, kn_w_dev, scores_uncomp_dev, num_blocks, theta);
    ops::qsa_compute_block_scores_compressed(ctx, q_dev, comp_k_dev, scores_comp_dev, num_blocks);
    ctx.synchronize();

    std::vector<float> scores_uncomp(num_blocks);
    std::vector<float> scores_comp(num_blocks);
    ctx.memcpy_d2h(scores_uncomp.data(), scores_uncomp_dev.data(), num_blocks * sizeof(float));
    ctx.memcpy_d2h(scores_comp.data(), scores_comp_dev.data(), num_blocks * sizeof(float));

    float max_score_diff = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        float diff = std::abs(scores_comp[b] - scores_uncomp[b]);
        if (diff > max_score_diff) max_score_diff = diff;
    }
    if (max_score_diff > 0.01f) {
        throw std::runtime_error("test_qsa_compressed_k_scoring: compressed vs uncompressed score mismatch, diff=" +
                                 std::to_string(max_score_diff));
    }
    std::cout << "[PASS] QSA compressed K scoring matches uncompressed scoring (max_score_diff="
              << max_score_diff << " <= 0.01)\n";

    // 4. Test incremental compression appending
    // Compress first 32 blocks, then incrementally compress next 32 blocks
    Tensor comp_k_incr = Tensor::allocate(ctx, {num_blocks, kHeadDim}, dnnl::memory::data_type::bf16);
    ops::qsa_compress_key_blocks(ctx, raw_k_dev, kn_w_dev, comp_k_incr, 0, 32, theta);
    ops::qsa_compress_key_blocks(ctx, raw_k_dev, kn_w_dev, comp_k_incr, 32, 32, theta);
    ctx.synchronize();

    std::vector<bf16> comp_k_incr_h(num_blocks * kHeadDim);
    ctx.memcpy_d2h(comp_k_incr_h.data(), comp_k_incr.data(), comp_k_incr_h.size() * sizeof(bf16));
    for (size_t i = 0; i < comp_k_incr_h.size(); ++i) {
        float diff = std::abs(static_cast<float>(comp_k_incr_h[i]) - static_cast<float>(comp_k_gpu[i]));
        if (diff > 1e-4f) {
            throw std::runtime_error("test_qsa_compressed_k_scoring: incremental compression slice mismatch at " +
                                     std::to_string(i));
        }
    }
    std::cout << "[PASS] QSA incremental slice compression verified (chunked [0..32) and [32..64) identical)\n";

    // 5. Test upstream reference function execution
    std::vector<bf16> kn_raw_h(kHeadDim);
    for (int d = 0; d < kHeadDim; ++d) {
        kn_raw_h[d] = bf16(static_cast<float>(kn_w_h[d]) - 1.0f);
    }
    std::vector<float> scores_upstream(num_blocks);
    ops::qsa_compute_block_scores_upstream_ref(
        q_h.data(), raw_k_h.data(), kn_raw_h.data(), scores_upstream.data(), num_blocks, theta);
    for (int b = 0; b < num_blocks; ++b) {
        if (std::isnan(scores_upstream[b]) || std::isinf(scores_upstream[b]) || scores_upstream[b] < 0.0f) {
            throw std::runtime_error("test_qsa_compressed_k_scoring: upstream ref produced invalid score at " +
                                     std::to_string(b));
        }
    }
    std::cout << "[PASS] QSA upstream precision reference sanity check verified (finite, non-negative scores)\n";

    // 6. Test qsa_prepare_causal_query_and_select_tokens with compressed_k
    {
        constexpr int kCapacity = 2056;
        constexpr int kBlockTopk = 512;
        constexpr int kCompressRatio = 4;
        constexpr int kMaxBlocks = kCapacity / kCompressRatio;
        Tensor k_cache = Tensor::allocate(ctx, {kCapacity, kHeadDim}, dnnl::memory::data_type::bf16);
        Tensor comp_k = Tensor::allocate(ctx, {kMaxBlocks, kHeadDim}, dnnl::memory::data_type::bf16);
        Tensor q_norm = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
        Tensor kn_w = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
        Tensor q_out = Tensor::allocate(ctx, {kNumHeads, kHeadDim}, dnnl::memory::data_type::bf16);
        Tensor scores = Tensor::allocate(ctx, {kMaxBlocks}, dnnl::memory::data_type::f32);
        Tensor active_tokens = Tensor::allocate(ctx, {kCapacity}, dnnl::memory::data_type::s32);
        Tensor qk_row = Tensor::allocate(ctx, {1, (kNumHeads + 1) * kHeadDim}, dnnl::memory::data_type::bf16);

        std::vector<bf16> ones(kHeadDim, bf16(1.0f));
        ctx.memcpy_h2d(q_norm.data(), ones.data(), kHeadDim * sizeof(bf16));
        ctx.memcpy_h2d(kn_w.data(), ones.data(), kHeadDim * sizeof(bf16));

        // Pre-fill k_cache up to 2050 tokens
        std::vector<bf16> raw_k_init(2050 * kHeadDim);
        for (size_t i = 0; i < raw_k_init.size(); ++i) raw_k_init[i] = bf16(0.05f * (i % 17));
        ctx.memcpy_h2d(k_cache.data(), raw_k_init.data(), raw_k_init.size() * sizeof(bf16));

        int num_comp = 0;
        // Step 1: pos = 2050 (visible_len = 2051, num_complete_blocks = 512 <= 512 -> dense)
        int active_count = ops::qsa_prepare_causal_query_and_select_tokens(
            ctx, qk_row, q_norm, kn_w, k_cache, q_out, scores, active_tokens,
            2050, kBlockTopk, kCompressRatio, theta, false,
            nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
            &comp_k, &num_comp);
        ctx.synchronize();
        if (active_count != 0) {
            throw std::runtime_error("test_qsa_compressed_k_scoring: pos=2050 must return 0 (dense)");
        }
        if (num_comp != 512) {
            throw std::runtime_error("test_qsa_compressed_k_scoring: pos=2050 must have compressed 512 blocks, got " +
                                     std::to_string(num_comp));
        }

        // Step 2: pos = 2051 (visible_len = 2052, num_complete_blocks = 513 > 512 -> sparse)
        active_count = ops::qsa_prepare_causal_query_and_select_tokens(
            ctx, qk_row, q_norm, kn_w, k_cache, q_out, scores, active_tokens,
            2051, kBlockTopk, kCompressRatio, theta, false,
            nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
            &comp_k, &num_comp);
        ctx.synchronize();
        if (active_count != 2048) {
            throw std::runtime_error("test_qsa_compressed_k_scoring: pos=2051 must return 2048 active tokens, got " +
                                     std::to_string(active_count));
        }
        if (num_comp != 513) {
            throw std::runtime_error("test_qsa_compressed_k_scoring: pos=2051 must advance to 513 compressed blocks, got " +
                                     std::to_string(num_comp));
        }
        std::cout << "[PASS] QSA causal query preparation with incremental compressed K verified across boundary.\n";
    }
}

static void test_qsa_select_topk_blocks(Context& ctx) {
    std::cout << "[Test] Running QSA Select Top-K Blocks (A3) tests...\n";
    constexpr int compress_ratio = 4;
    constexpr int block_topk = 512;

    // 1. Dense phase (num_complete_blocks <= block_topk)
    {
        constexpr int num_complete = 10;
        constexpr int cached_len = 42; // 10 blocks + 2 tail tokens
        std::vector<float> scores(num_complete, 0.5f);
        Tensor scores_dev = Tensor::allocate(ctx, {num_complete}, dnnl::memory::data_type::f32);
        Tensor active_dev = Tensor::allocate(ctx, {block_topk + 4}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores.data(), num_complete * sizeof(float));

        int gpu_blocks = ops::qsa_select_topk_blocks(
            ctx, scores_dev, num_complete, block_topk, cached_len, compress_ratio, active_dev);
        ctx.synchronize();

        std::vector<int> ref_out(block_topk + 4, -1);
        int ref_blocks = ops::qsa_select_topk_blocks_host(
            scores.data(), num_complete, block_topk, cached_len, compress_ratio, ref_out.data());

        if (gpu_blocks != ref_blocks || gpu_blocks != num_complete) {
            throw std::runtime_error("test_qsa_select_topk_blocks: dense phase block count mismatch");
        }
        std::vector<int> gpu_out(block_topk + 4);
        ctx.memcpy_d2h(gpu_out.data(), active_dev.data(), (num_complete + 2) * sizeof(int));
        for (int i = 0; i < num_complete + 2; ++i) {
            if (gpu_out[i] != ref_out[i]) {
                throw std::runtime_error("test_qsa_select_topk_blocks: dense phase index mismatch at " + std::to_string(i));
            }
        }
        std::cout << "[PASS] QSA select top-k blocks dense phase (num_complete=10, 2 tail tokens) verified.\n";
    }

    // 2. Sparse phase (num_complete_blocks = 600 > 512, cached_len = 2403)
    {
        constexpr int num_complete = 600;
        constexpr int cached_len = 2403; // 600 blocks + 3 tail tokens
        std::mt19937 rng(777);
        std::uniform_real_distribution<float> dist(0.01f, 10.0f);
        std::vector<float> scores(num_complete);
        for (auto& s : scores) s = dist(rng);

        Tensor scores_dev = Tensor::allocate(ctx, {num_complete}, dnnl::memory::data_type::f32);
        Tensor active_dev = Tensor::allocate(ctx, {block_topk + 4}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores.data(), num_complete * sizeof(float));

        int gpu_blocks = ops::qsa_select_topk_blocks(
            ctx, scores_dev, num_complete, block_topk, cached_len, compress_ratio, active_dev);
        ctx.synchronize();

        std::vector<int> ref_out(block_topk + 4, -1);
        int ref_blocks = ops::qsa_select_topk_blocks_host(
            scores.data(), num_complete, block_topk, cached_len, compress_ratio, ref_out.data());

        if (gpu_blocks != block_topk || ref_blocks != block_topk) {
            throw std::runtime_error("test_qsa_select_topk_blocks: sparse phase block count mismatch");
        }
        std::vector<int> gpu_out(block_topk + 4);
        ctx.memcpy_d2h(gpu_out.data(), active_dev.data(), (block_topk + 3) * sizeof(int));
        for (int i = 0; i < block_topk + 3; ++i) {
            if (gpu_out[i] != ref_out[i]) {
                throw std::runtime_error("test_qsa_select_topk_blocks: sparse phase mismatch at " + std::to_string(i) +
                                         " gpu=" + std::to_string(gpu_out[i]) + " ref=" + std::to_string(ref_out[i]));
            }
        }
        std::cout << "[PASS] QSA select top-k blocks sparse phase (600 blocks -> 512 + 3 tail) verified.\n";
    }

    // 3. Ties and exact cutoff
    {
        constexpr int num_complete = 520;
        constexpr int cached_len = 520 * compress_ratio;
        std::vector<float> scores(num_complete, 1.0f);
        scores[10] = 5.0f;
        scores[20] = 0.5f;

        Tensor scores_dev = Tensor::allocate(ctx, {num_complete}, dnnl::memory::data_type::f32);
        Tensor active_dev = Tensor::allocate(ctx, {block_topk + 4}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(scores_dev.data(), scores.data(), num_complete * sizeof(float));

        int gpu_blocks = ops::qsa_select_topk_blocks(
            ctx, scores_dev, num_complete, block_topk, cached_len, compress_ratio, active_dev);
        ctx.synchronize();

        std::vector<int> ref_out(block_topk + 4, -1);
        int ref_blocks = ops::qsa_select_topk_blocks_host(
            scores.data(), num_complete, block_topk, cached_len, compress_ratio, ref_out.data());

        if (gpu_blocks != block_topk || ref_blocks != block_topk) {
            throw std::runtime_error("test_qsa_select_topk_blocks: ties test block count mismatch");
        }
        std::vector<int> gpu_out(block_topk + 4);
        ctx.memcpy_d2h(gpu_out.data(), active_dev.data(), block_topk * sizeof(int));
        for (int i = 0; i < block_topk; ++i) {
            if (gpu_out[i] != ref_out[i]) {
                throw std::runtime_error("test_qsa_select_topk_blocks: ties mismatch at " + std::to_string(i));
            }
        }
        std::cout << "[PASS] QSA select top-k blocks ties & boundary verified.\n";
    }

    // 4. Edge cases
    {
        Tensor dummy_scores = Tensor::allocate(ctx, {10}, dnnl::memory::data_type::f32);
        Tensor dummy_blocks = Tensor::allocate(ctx, {10}, dnnl::memory::data_type::s32);
        int z1 = ops::qsa_select_topk_blocks(ctx, dummy_scores, 10, 5, 0, 4, dummy_blocks);
        int z2 = ops::qsa_select_topk_blocks(ctx, dummy_scores, 10, -1, 16, 4, dummy_blocks);
        if (z1 != 0 || z2 != 0) {
            throw std::runtime_error("test_qsa_select_topk_blocks: edge case guard failed");
        }
        std::cout << "[PASS] QSA select top-k blocks edge case guards verified.\n";
    }
}

static void test_qsa_graph_selector_radix(Context& ctx) {
    namespace ex = sycl::ext::oneapi::experimental;
    using ExecutableGraph = ex::command_graph<ex::graph_state::executable>;

    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int max_blocks = 8192;
    auto& queue = ctx.queue();
    if (!queue.get_device().has(sycl::aspect::ext_oneapi_limited_graph) &&
        !queue.get_device().has(sycl::aspect::ext_oneapi_graph)) {
        throw std::runtime_error(
            "QSA graph selector test requires SYCL graph support");
    }

    Tensor scores = Tensor::allocate(
        ctx, {max_blocks}, dnnl::memory::data_type::f32);
    Tensor control = Tensor::allocate(
        ctx, {ops::graph_control::words}, dnnl::memory::data_type::s32);
    Tensor active_legacy = Tensor::allocate(
        ctx, {block_topk + compress_ratio},
        dnnl::memory::data_type::s32);
    Tensor active_radix = Tensor::allocate(
        ctx, {block_topk + compress_ratio},
        dnnl::memory::data_type::s32);
    Tensor active_direct = Tensor::allocate(
        ctx, {block_topk + compress_ratio},
        dnnl::memory::data_type::s32);

    auto capture_selector = [&](Tensor& output, int selector_mode) {
        ex::command_graph<ex::graph_state::modifiable> recording(
            queue.get_context(), queue.get_device());
        recording.begin_recording(queue);
        try {
            ops::qsa_select_topk_blocks_graph(
                ctx, scores, output, control.data_as<int>(),
                block_topk, compress_ratio, max_blocks, 0,
                selector_mode);
            recording.end_recording(queue);
        } catch (...) {
            recording.end_recording(queue);
            throw;
        }
        return recording.finalize();
    };
    ExecutableGraph legacy_graph = capture_selector(active_legacy, 0);
    ExecutableGraph radix_graph = capture_selector(active_radix, 1);

    constexpr int sentinel = -777;
    auto run_case = [&](int blocks, int tail, int pattern) {
        const int visible_len = blocks * compress_ratio + tail;
        std::vector<float> host_scores(blocks);
        if (pattern == 0) {
            std::fill(host_scores.begin(), host_scores.end(), 1.0f);
            host_scores[100] = 0.25f;
        } else if (pattern == 1) {
            std::fill(host_scores.begin(), host_scores.end(), 1.0f);
            host_scores[0] = 2.0f;
            host_scores[1] = sycl::bit_cast<float>(0x7fc00001U);
        } else {
            std::mt19937 rng(81000 + blocks);
            for (auto& score : host_scores) {
                score = float(rng() % 100003) * 0.001f;
            }
            host_scores[3] = sycl::bit_cast<float>(0x7fc00002U);
            host_scores[blocks / 2] = host_scores[blocks / 2 + 1];
            host_scores[blocks - 1] = host_scores[blocks - 2];
        }

        std::array<int, ops::graph_control::words> control_host{};
        control_host[ops::graph_control::start_pos] = visible_len - 1;
        control_host[ops::graph_control::compressed_blocks] = blocks;
        control_host[ops::graph_control::abort] = 0;
        ctx.memcpy_h2d(
            scores.data(), host_scores.data(),
            host_scores.size() * sizeof(float));
        ctx.memcpy_h2d(
            control.data(), control_host.data(),
            control_host.size() * sizeof(int));

        std::vector<int> initial(block_topk + compress_ratio, sentinel);
        for (Tensor* output : {&active_legacy, &active_radix, &active_direct}) {
            ctx.memcpy_h2d(
                output->data(), initial.data(),
                initial.size() * sizeof(int));
        }

        auto legacy_event = queue.ext_oneapi_graph(legacy_graph);
        auto radix_event = queue.ext_oneapi_graph(radix_graph);
        const int direct_count = ops::qsa_select_topk_blocks(
            ctx, scores, blocks, block_topk, visible_len,
            compress_ratio, active_direct);
        legacy_event.wait_and_throw();
        radix_event.wait_and_throw();

        std::vector<int> legacy(initial);
        std::vector<int> radix(initial);
        std::vector<int> direct(initial);
        std::vector<int> expected(initial);
        ctx.memcpy_d2h(
            legacy.data(), active_legacy.data(),
            legacy.size() * sizeof(int));
        ctx.memcpy_d2h(
            radix.data(), active_radix.data(),
            radix.size() * sizeof(int));
        ctx.memcpy_d2h(
            direct.data(), active_direct.data(),
            direct.size() * sizeof(int));
        const int expected_count = ops::qsa_select_topk_blocks_host(
            host_scores.data(), blocks, block_topk, visible_len,
            compress_ratio, expected.data());
        const int used = block_topk + tail;
        if (direct_count != expected_count ||
            std::memcmp(
                direct.data(), expected.data(),
                used * sizeof(int)) != 0) {
            throw std::runtime_error(
                "QSA graph selector direct oracle mismatch at blocks=" +
                std::to_string(blocks) +
                " tail=" + std::to_string(tail));
        }
        if (std::memcmp(
                legacy.data(), expected.data(),
                used * sizeof(int)) != 0) {
            throw std::runtime_error(
                "QSA graph selector legacy mismatch at blocks=" +
                std::to_string(blocks) +
                " tail=" + std::to_string(tail));
        }
        if (std::memcmp(
                radix.data(), expected.data(),
                used * sizeof(int)) != 0) {
            throw std::runtime_error(
                "QSA graph selector radix mismatch at blocks=" +
                std::to_string(blocks) +
                " tail=" + std::to_string(tail));
        }
        std::cout << "[QSA GRAPH SELECTOR PASS] blocks=" << blocks
                  << " tail=" << tail << " selected=" << expected_count
                  << " legacy=exact radix=exact\n";
    };

    run_case(513, 0, 0);
    run_case(514, 3, 1);
    run_case(max_blocks, 2, 2);

    auto measure = [&](ExecutableGraph& graph) {
        const auto begin = std::chrono::steady_clock::now();
        auto event = queue.ext_oneapi_graph(graph);
        event.wait_and_throw();
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - begin)
            .count();
    };
    const double legacy_us = measure(legacy_graph);
    const double radix_us = measure(radix_graph);
    std::cout << "[QSA GRAPH SELECTOR COST] blocks=" << max_blocks
              << " legacy_us=" << legacy_us
              << " radix_us=" << radix_us
              << " speedup=" << (legacy_us / radix_us)
              << " runs=1 exact=1\n";
}

static void test_qsa_prepare_q_batched(Context& ctx) {
    std::cout << "[Test] Running QSA Batched Query Preparation (A4) tests...\n";
    constexpr int count = 8;
    constexpr int start_pos = 100;
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int kOutDim = 640;
    constexpr float theta = 1000000.0f;

    std::mt19937 rng(888);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> indexer_qk_h(count * kOutDim);
    std::vector<bf16> qn_w_h(kHeadDim);
    for (auto& v : indexer_qk_h) v = bf16(dist(rng));
    for (auto& v : qn_w_h) v = bf16(1.0f + 0.1f * dist(rng));

    Tensor qk_dev = Tensor::allocate(ctx, {count, kOutDim}, dnnl::memory::data_type::bf16);
    Tensor qn_dev = Tensor::allocate(ctx, {kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor q_dev = Tensor::allocate(ctx, {count, kNumHeads * kHeadDim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(qk_dev.data(), indexer_qk_h.data(), indexer_qk_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(qn_dev.data(), qn_w_h.data(), qn_w_h.size() * sizeof(bf16));

    ops::qsa_prepare_q_batched(ctx, qk_dev, qn_dev, q_dev, start_pos, count, theta);
    ctx.synchronize();

    std::vector<bf16> ref_out(count * kNumHeads * kHeadDim);
    ops::qsa_prepare_q_batched_host(indexer_qk_h.data(), qn_w_h.data(), ref_out.data(), start_pos, count, theta);

    std::vector<bf16> gpu_out(count * kNumHeads * kHeadDim);
    ctx.memcpy_d2h(gpu_out.data(), q_dev.data(), gpu_out.size() * sizeof(bf16));

    std::vector<float> ref_f32(ref_out.size());
    for (size_t i = 0; i < ref_out.size(); ++i) ref_f32[i] = static_cast<float>(ref_out[i]);

    check_close(gpu_out, ref_f32, "qsa_prepare_q_batched GPU vs Host", 2e-2f);

    // 2. Test with MRoPE positions (interleaved = true)
    {
        std::vector<int> pos_t_h(start_pos + count);
        std::vector<int> pos_h_h(start_pos + count);
        std::vector<int> pos_w_h(start_pos + count);
        for (int i = 0; i < start_pos + count; ++i) {
            pos_t_h[i] = i * 2;
            pos_h_h[i] = i * 3 + 1;
            pos_w_h[i] = i * 4 + 2;
        }
        Tensor pos_t_dev = Tensor::allocate(ctx, {start_pos + count}, dnnl::memory::data_type::s32);
        Tensor pos_h_dev = Tensor::allocate(ctx, {start_pos + count}, dnnl::memory::data_type::s32);
        Tensor pos_w_dev = Tensor::allocate(ctx, {start_pos + count}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(pos_t_dev.data(), pos_t_h.data(), pos_t_h.size() * sizeof(int));
        ctx.memcpy_h2d(pos_h_dev.data(), pos_h_h.data(), pos_h_h.size() * sizeof(int));
        ctx.memcpy_h2d(pos_w_dev.data(), pos_w_h.data(), pos_w_h.size() * sizeof(int));

        ops::qsa_prepare_q_batched(
            ctx, qk_dev, qn_dev, q_dev, start_pos, count, theta, true,
            pos_t_dev.data_as<int>(), pos_h_dev.data_as<int>(), pos_w_dev.data_as<int>(),
            start_pos + count, 0, 0, 8, 8);
        ctx.synchronize();

        ops::qsa_prepare_q_batched_host(
            indexer_qk_h.data(), qn_w_h.data(), ref_out.data(), start_pos, count, theta, true,
            pos_t_h.data(), pos_h_h.data(), pos_w_h.data(),
            start_pos + count, 0, 0, 8, 8);

        ctx.memcpy_d2h(gpu_out.data(), q_dev.data(), gpu_out.size() * sizeof(bf16));
        for (size_t i = 0; i < ref_out.size(); ++i) ref_f32[i] = static_cast<float>(ref_out[i]);
        check_close(gpu_out, ref_f32, "qsa_prepare_q_batched (MRoPE interleaved) GPU vs Host", 2e-2f);
    }
}

static void test_qsa_batched_scoring_and_selection(Context& ctx) {
    std::cout << "[Test] Running QSA Batched Scoring & Selection (A4) tests...\n";
    constexpr int query_count = 16;
    constexpr int max_complete_blocks = 550;
    constexpr int start_pos = 2040;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bf16> q_h(query_count * kNumHeads * kHeadDim);
    std::vector<bf16> comp_k_h(max_complete_blocks * kHeadDim);
    for (auto& v : q_h) v = bf16(dist(rng));
    for (auto& v : comp_k_h) v = bf16(dist(rng));

    Tensor q_dev = Tensor::allocate(ctx, {query_count, kNumHeads * kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor comp_k_dev = Tensor::allocate(ctx, {max_complete_blocks, kHeadDim}, dnnl::memory::data_type::bf16);
    Tensor scores_dev = Tensor::allocate(ctx, {query_count, max_complete_blocks}, dnnl::memory::data_type::f32);
    Tensor active_dev = Tensor::allocate(ctx, {query_count, block_topk + 4}, dnnl::memory::data_type::s32);

    ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(comp_k_dev.data(), comp_k_h.data(), comp_k_h.size() * sizeof(bf16));

    // 1. Batched scoring
    ops::qsa_compute_block_scores_compressed_batched(
        ctx, q_dev, comp_k_dev, scores_dev, query_count, max_complete_blocks, compress_ratio, start_pos);
    ctx.synchronize();

    std::vector<float> ref_scores(query_count * max_complete_blocks);
    ops::qsa_compute_block_scores_compressed_batched_host(
        q_h.data(), comp_k_h.data(), ref_scores.data(), query_count, max_complete_blocks, compress_ratio, start_pos);

    std::vector<float> gpu_scores(query_count * max_complete_blocks);
    ctx.memcpy_d2h(gpu_scores.data(), scores_dev.data(), gpu_scores.size() * sizeof(float));

    float max_diff = 0.0f;
    for (size_t i = 0; i < gpu_scores.size(); ++i) {
        if (ref_scores[i] < -1e20f && gpu_scores[i] < -1e20f) continue;
        float diff = std::abs(gpu_scores[i] - ref_scores[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 1e-3f) {
        throw std::runtime_error("test_qsa_batched_scoring_and_selection: scores mismatch, max_diff=" + std::to_string(max_diff));
    }
    std::cout << "[PASS] QSA batched scoring matches host reference (max_diff=" << max_diff << " < 1e-3)\n";

    // 2. Batched selection
    ops::qsa_select_topk_blocks_batched(
        ctx, scores_dev, active_dev, query_count, max_complete_blocks, start_pos, block_topk, compress_ratio);
    ctx.synchronize();

    std::vector<int> ref_active(query_count * (block_topk + 4));
    ops::qsa_select_topk_blocks_batched_host(
        ref_scores.data(), ref_active.data(), query_count, max_complete_blocks, start_pos, block_topk, compress_ratio);

    std::vector<int> gpu_active(query_count * (block_topk + 4));
    ctx.memcpy_d2h(gpu_active.data(), active_dev.data(), gpu_active.size() * sizeof(int));

    for (int q = 0; q < query_count; ++q) {
        const int pos = start_pos + q;
        const int visible_len = pos + 1;
        const int num_complete = visible_len / compress_ratio;
        const int is_sparse = (num_complete > block_topk);
        const int active_n = is_sparse ? block_topk : num_complete;
        const int tail_len = visible_len - num_complete * compress_ratio;
        const size_t row_stride = block_topk + 4;

        for (int i = 0; i < active_n + tail_len; ++i) {
            int g_val = gpu_active[static_cast<size_t>(q) * row_stride + i];
            int r_val = ref_active[static_cast<size_t>(q) * row_stride + i];
            if (g_val != r_val) {
                throw std::runtime_error("test_qsa_batched_scoring_and_selection: mismatch at query=" +
                                         std::to_string(q) + " idx=" + std::to_string(i) +
                                         " gpu=" + std::to_string(g_val) + " ref=" + std::to_string(r_val));
            }
        }
    }
    std::cout << "[PASS] QSA batched selection matches host reference for all queries across threshold.\n";
}

static void test_attention_prefill_qsa_batched(Context& ctx) {
    std::cout << "[Test] Running Batched QSA Prefill Attention (A4) tests...\n";
    constexpr int count = 8;
    constexpr int start_pos = 2050;
    constexpr int capacity = 2200;
    constexpr int num_heads = 24;
    constexpr int num_kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int block_topk = 512;
    constexpr int compress_ratio = 4;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<bf16> q_h(count * num_heads * head_dim);
    std::vector<bf16> k_cache_h(num_kv_heads * capacity * head_dim);
    std::vector<bf16> v_cache_h(num_kv_heads * capacity * head_dim);
    std::vector<int> active_blocks_h(count * (block_topk + 4));
    std::vector<bf16> gate_h(count * num_heads * head_dim);

    for (auto& v : q_h) v = bf16(dist(rng));
    for (auto& v : k_cache_h) v = bf16(dist(rng));
    for (auto& v : v_cache_h) v = bf16(dist(rng));
    for (auto& v : gate_h) v = bf16(dist(rng));

    // Setup active_blocks: 512 random blocks in [0, 512] for each query + tail
    for (int q = 0; q < count; ++q) {
        const int pos = start_pos + q;
        const int visible_len = pos + 1;
        const int num_complete = visible_len / compress_ratio;
        const int tail_start = num_complete * compress_ratio;
        const int tail_len = visible_len - tail_start;
        const size_t stride = block_topk + 4;

        std::vector<int> blocks(num_complete);
        for (int b = 0; b < num_complete; ++b) blocks[b] = b;
        std::shuffle(blocks.begin(), blocks.end(), rng);
        std::sort(blocks.begin(), blocks.begin() + std::min(block_topk, num_complete));

        for (int b = 0; b < std::min(block_topk, num_complete); ++b) {
            active_blocks_h[static_cast<size_t>(q) * stride + b] = blocks[b];
        }
        for (int t = 0; t < tail_len; ++t) {
            active_blocks_h[static_cast<size_t>(q) * stride + std::min(block_topk, num_complete) + t] = tail_start + t;
        }
    }

    Tensor q_dev = Tensor::allocate(ctx, {count, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, head_dim}, dnnl::memory::data_type::bf16);
    Tensor active_dev = Tensor::allocate(ctx, {count, block_topk + 4}, dnnl::memory::data_type::s32);
    Tensor out_dev = Tensor::allocate(ctx, {count, num_heads * head_dim}, dnnl::memory::data_type::bf16);
    Tensor gate_dev = Tensor::allocate(ctx, {count, num_heads * head_dim}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(q_dev.data(), q_h.data(), q_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_dev.data(), k_cache_h.data(), k_cache_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), v_cache_h.data(), v_cache_h.size() * sizeof(bf16));
    ctx.memcpy_h2d(active_dev.data(), active_blocks_h.data(), active_blocks_h.size() * sizeof(int));
    ctx.memcpy_h2d(gate_dev.data(), gate_h.data(), gate_h.size() * sizeof(bf16));

    // Test 1: Without gate
    ops::attention_prefill_qsa_batched(
        ctx, q_dev, k_dev, v_dev, active_dev, out_dev,
        count, start_pos, num_heads, num_kv_heads, head_dim,
        block_topk, compress_ratio, nullptr);
    ctx.synchronize();

    std::vector<bf16> ref_out(count * num_heads * head_dim);
    ops::attention_prefill_qsa_batched_host(
        q_h.data(), k_cache_h.data(), v_cache_h.data(), active_blocks_h.data(), ref_out.data(),
        count, start_pos, capacity, num_heads, num_kv_heads, head_dim,
        block_topk, compress_ratio, nullptr);

    std::vector<bf16> gpu_out(count * num_heads * head_dim);
    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), gpu_out.size() * sizeof(bf16));

    std::vector<float> ref_f32(ref_out.size());
    for (size_t i = 0; i < ref_out.size(); ++i) ref_f32[i] = static_cast<float>(ref_out[i]);

    check_close(gpu_out, ref_f32, "attention_prefill_qsa_batched (no gate)", 2e-2f);

    // Test 2: With gate
    ops::attention_prefill_qsa_batched(
        ctx, q_dev, k_dev, v_dev, active_dev, out_dev,
        count, start_pos, num_heads, num_kv_heads, head_dim,
        block_topk, compress_ratio, &gate_dev);
    ctx.synchronize();

    ops::attention_prefill_qsa_batched_host(
        q_h.data(), k_cache_h.data(), v_cache_h.data(), active_blocks_h.data(), ref_out.data(),
        count, start_pos, capacity, num_heads, num_kv_heads, head_dim,
        block_topk, compress_ratio, gate_h.data());

    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), gpu_out.size() * sizeof(bf16));
    for (size_t i = 0; i < ref_out.size(); ++i) ref_f32[i] = static_cast<float>(ref_out[i]);

    check_close(gpu_out, ref_f32, "attention_prefill_qsa_batched (with gate)", 2e-2f);

    // Test 2b: int8 KV cache with per-(head, token) symmetric scales. The host
    // reference runs on the dequantized int8 values, so this checks the scale
    // application rather than the quantization error itself.
    {
        std::vector<int8_t> k_i8_h(k_cache_h.size()), v_i8_h(v_cache_h.size());
        std::vector<float> k_scale_h(size_t(num_kv_heads) * capacity),
            v_scale_h(size_t(num_kv_heads) * capacity);
        std::vector<bf16> k_deq_h(k_cache_h.size()), v_deq_h(v_cache_h.size());
        auto quantize = [&](const std::vector<bf16>& src, std::vector<int8_t>& codes,
                            std::vector<float>& scales, std::vector<bf16>& deq) {
            for (int h = 0; h < num_kv_heads; ++h) {
                for (int t = 0; t < capacity; ++t) {
                    const size_t base = (size_t(h) * capacity + t) * head_dim;
                    float amax = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        amax = (std::max)(amax, std::fabs(static_cast<float>(src[base + d])));
                    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                    scales[size_t(h) * capacity + t] = s;
                    for (int d = 0; d < head_dim; ++d) {
                        const int code = static_cast<int>(sycl::rint(
                            static_cast<float>(src[base + d]) / s));
                        codes[base + d] = static_cast<int8_t>(
                            (std::max)(-127, (std::min)(127, code)));
                        deq[base + d] = bf16(static_cast<float>(codes[base + d]) * s);
                    }
                }
            }
        };
        quantize(k_cache_h, k_i8_h, k_scale_h, k_deq_h);
        quantize(v_cache_h, v_i8_h, v_scale_h, v_deq_h);

        Tensor k_i8_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, head_dim},
                                           dnnl::memory::data_type::s8);
        Tensor v_i8_dev = Tensor::allocate(ctx, {num_kv_heads, capacity, head_dim},
                                           dnnl::memory::data_type::s8);
        Tensor k_scale_dev = Tensor::allocate(ctx, {num_kv_heads, capacity},
                                              dnnl::memory::data_type::f32);
        Tensor v_scale_dev = Tensor::allocate(ctx, {num_kv_heads, capacity},
                                              dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(k_i8_dev.data(), k_i8_h.data(), k_i8_h.size());
        ctx.memcpy_h2d(v_i8_dev.data(), v_i8_h.data(), v_i8_h.size());
        ctx.memcpy_h2d(k_scale_dev.data(), k_scale_h.data(), k_scale_h.size() * sizeof(float));
        ctx.memcpy_h2d(v_scale_dev.data(), v_scale_h.data(), v_scale_h.size() * sizeof(float));

        ops::attention_prefill_qsa_batched(
            ctx, q_dev, k_i8_dev, v_i8_dev, active_dev, out_dev,
            count, start_pos, num_heads, num_kv_heads, head_dim,
            block_topk, compress_ratio, nullptr, nullptr,
            &k_scale_dev, &v_scale_dev);
        ctx.synchronize();

        std::vector<bf16> ref_i8_out(count * num_heads * head_dim);
        ops::attention_prefill_qsa_batched_host(
            q_h.data(), k_deq_h.data(), v_deq_h.data(), active_blocks_h.data(),
            ref_i8_out.data(), count, start_pos, capacity, num_heads, num_kv_heads,
            head_dim, block_topk, compress_ratio, nullptr);

        std::vector<bf16> gpu_i8_out(count * num_heads * head_dim);
        ctx.memcpy_d2h(gpu_i8_out.data(), out_dev.data(), gpu_i8_out.size() * sizeof(bf16));
        std::vector<float> ref_i8_f32(ref_i8_out.size());
        for (size_t i = 0; i < ref_i8_out.size(); ++i)
            ref_i8_f32[i] = static_cast<float>(ref_i8_out[i]);
        check_close(gpu_i8_out, ref_i8_f32, "attention_prefill_qsa_batched int8 KV", 2e-2f);
    }

    // Test 3: Guard checks (invalid head_dim)
    try {
        ops::attention_prefill_qsa_batched(
            ctx, q_dev, k_dev, v_dev, active_dev, out_dev,
            count, start_pos, num_heads, num_kv_heads, 128,
            block_topk, compress_ratio, nullptr);
        throw std::runtime_error("attention_prefill_qsa_batched failed to throw on invalid head_dim=128");
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] attention_prefill_qsa_batched guarded against invalid head_dim=128.\n";
    }

    // Test 4: Multi-tile prefill (count = 80)
    {
        constexpr int count_large = 80;
        std::vector<bf16> q_large_h(count_large * num_heads * head_dim);
        std::vector<int> active_large_h(count_large * (block_topk + 4));
        for (auto& v : q_large_h) v = bf16(dist(rng));
        for (int q = 0; q < count_large; ++q) {
            const int pos = start_pos + q;
            const int visible_len = pos + 1;
            const int num_complete = visible_len / compress_ratio;
            const int tail_start = num_complete * compress_ratio;
            const int tail_len = visible_len - tail_start;
            const size_t stride = block_topk + 4;
            std::vector<int> blocks(num_complete);
            for (int b = 0; b < num_complete; ++b) blocks[b] = b;
            std::shuffle(blocks.begin(), blocks.end(), rng);
            std::sort(blocks.begin(), blocks.begin() + std::min(block_topk, num_complete));
            for (int b = 0; b < std::min(block_topk, num_complete); ++b) {
                active_large_h[static_cast<size_t>(q) * stride + b] = blocks[b];
            }
            for (int t = 0; t < tail_len; ++t) {
                active_large_h[static_cast<size_t>(q) * stride + std::min(block_topk, num_complete) + t] = tail_start + t;
            }
        }
        Tensor q_large_dev = Tensor::allocate(ctx, {count_large, num_heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor active_large_dev = Tensor::allocate(ctx, {count_large, block_topk + 4}, dnnl::memory::data_type::s32);
        Tensor out_large_dev = Tensor::allocate(ctx, {count_large, num_heads * head_dim}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(q_large_dev.data(), q_large_h.data(), q_large_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(active_large_dev.data(), active_large_h.data(), active_large_h.size() * sizeof(int));

        ops::attention_prefill_qsa_batched(
            ctx, q_large_dev, k_dev, v_dev, active_large_dev, out_large_dev,
            count_large, start_pos, num_heads, num_kv_heads, head_dim,
            block_topk, compress_ratio, nullptr);
        ctx.synchronize();

        std::vector<bf16> ref_large_out(count_large * num_heads * head_dim);
        ops::attention_prefill_qsa_batched_host(
            q_large_h.data(), k_cache_h.data(), v_cache_h.data(), active_large_h.data(), ref_large_out.data(),
            count_large, start_pos, capacity, num_heads, num_kv_heads, head_dim,
            block_topk, compress_ratio, nullptr);

        std::vector<bf16> gpu_large_out(count_large * num_heads * head_dim);
        ctx.memcpy_d2h(gpu_large_out.data(), out_large_dev.data(), gpu_large_out.size() * sizeof(bf16));
        std::vector<float> ref_large_f32(ref_large_out.size());
        for (size_t i = 0; i < ref_large_out.size(); ++i) ref_large_f32[i] = static_cast<float>(ref_large_out[i]);
        check_close(gpu_large_out, ref_large_f32, "attention_prefill_qsa_batched multi-tile (count=80)", 2e-2f);
    }
}

void test_moe_prefill_weight_reuse_benchmark(Context& ctx) {
    constexpr int in_features = 2560;
    constexpr int intermediate_size = 640;
    constexpr int out_features = 2560;
    constexpr int blocksize = 64;

    std::cout << "\n======================================================================\n";
    std::cout << "  [Benchmark] W4 C1/C2 Expert-Major Prefill 2-Token Weight Reuse\n";
    std::cout << "======================================================================\n";

    for (int tokens : {16, 64}) {
        const int num_active_experts = 10;
        const int num_unique_experts = 32;
        const int total_acts = tokens * num_active_experts;

        std::mt19937 rng(999 + tokens);
        std::uniform_int_distribution<int> u8_dist(0, 255);
        std::uniform_real_distribution<float> f_dist(0.01f, 0.5f);
        std::uniform_real_distribution<float> in_dist(-0.2f, 0.2f);

        const size_t gu_packed_bytes = static_cast<size_t>(2 * intermediate_size) * (in_features / 2);
        const size_t gu_absmax_count = static_cast<size_t>(2 * intermediate_size) * (in_features / blocksize);
        const size_t down_packed_bytes = static_cast<size_t>(out_features) * (intermediate_size / 2);
        const size_t down_absmax_count = static_cast<size_t>(out_features) * (intermediate_size / blocksize);

        std::vector<Tensor> gu_packed_dev, gu_absmax_dev, down_packed_dev, down_absmax_dev;
        Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(qmap_dev.data(), kTestQuantMap, 16 * sizeof(float));

        std::vector<ops::MoeExpertSlotDev> expert_pool(num_unique_experts);
        for (int e = 0; e < num_unique_experts; ++e) {
            gu_packed_dev.push_back(Tensor::allocate(ctx, {gu_packed_bytes}, dnnl::memory::data_type::u8));
            gu_absmax_dev.push_back(Tensor::allocate(ctx, {gu_absmax_count}, dnnl::memory::data_type::f32));
            down_packed_dev.push_back(Tensor::allocate(ctx, {down_packed_bytes}, dnnl::memory::data_type::u8));
            down_absmax_dev.push_back(Tensor::allocate(ctx, {down_absmax_count}, dnnl::memory::data_type::f32));

            std::vector<uint8_t> gu_p(gu_packed_bytes), dn_p(down_packed_bytes);
            std::vector<float> gu_a(gu_absmax_count), dn_a(down_absmax_count);
            for (auto& v : gu_p) v = static_cast<uint8_t>(u8_dist(rng));
            for (auto& v : dn_p) v = static_cast<uint8_t>(u8_dist(rng));
            for (auto& v : gu_a) v = f_dist(rng);
            for (auto& v : dn_a) v = f_dist(rng);

            ctx.memcpy_h2d(gu_packed_dev[e].data(), gu_p.data(), gu_packed_bytes);
            ctx.memcpy_h2d(gu_absmax_dev[e].data(), gu_a.data(), gu_absmax_count * sizeof(float));
            ctx.memcpy_h2d(down_packed_dev[e].data(), dn_p.data(), down_packed_bytes);
            ctx.memcpy_h2d(down_absmax_dev[e].data(), dn_a.data(), down_absmax_count * sizeof(float));

            expert_pool[e].gate_up_packed = static_cast<const uint8_t*>(gu_packed_dev[e].data());
            expert_pool[e].gate_up_absmax = static_cast<const float*>(gu_absmax_dev[e].data());
            expert_pool[e].down_packed = static_cast<const uint8_t*>(down_packed_dev[e].data());
            expert_pool[e].down_absmax = static_cast<const float*>(down_absmax_dev[e].data());
        }

        std::vector<bf16> x_host(tokens * in_features);
        for (auto& v : x_host) v = bf16(in_dist(rng));
        Tensor x_dev = Tensor::allocate(ctx, {tokens, in_features}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));

        std::vector<float> weights_host(total_acts, 1.0f / num_active_experts);
        float* weights_dev = static_cast<float*>(ctx.alloc_device(total_acts * sizeof(float)));
        ctx.memcpy_h2d(weights_dev, weights_host.data(), total_acts * sizeof(float));

        std::vector<int> expert_counts(num_unique_experts, 0);
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < num_active_experts; ++k) {
                int e = (t * 7 + k) % num_unique_experts;
                expert_counts[e]++;
            }
        }

        // Layout 1: S4 baseline (num_pairs = 0)
        std::vector<int> s4_tokens(total_acts), s4_ranks(total_acts);
        std::vector<ops::MoeExpertSlotDev> s4_slots(total_acts);
        std::vector<int> s4_offsets(num_unique_experts + 1, 0);
        for (int e = 0; e < num_unique_experts; ++e) {
            s4_offsets[e + 1] = s4_offsets[e] + expert_counts[e];
        }
        std::vector<int> s4_pos = s4_offsets;
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < num_active_experts; ++k) {
                int e = (t * 7 + k) % num_unique_experts;
                int p = s4_pos[e]++;
                s4_tokens[p] = t;
                s4_ranks[p] = k;
                s4_slots[p] = expert_pool[e];
            }
        }
        Tensor s4_tok_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
        Tensor s4_rnk_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
        ops::MoeExpertSlotDev* s4_slt_dev = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(total_acts * sizeof(ops::MoeExpertSlotDev)));
        ctx.memcpy_h2d(s4_tok_dev.data(), s4_tokens.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(s4_rnk_dev.data(), s4_ranks.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(s4_slt_dev, s4_slots.data(), total_acts * sizeof(ops::MoeExpertSlotDev));

        // Layout 2: W4 C1/C2 (num_pairs = total_pairs)
        int total_pairs = 0;
        for (int e = 0; e < num_unique_experts; ++e) total_pairs += expert_counts[e] / 2;
        std::vector<int> wr_pair_pos(num_unique_experts, 0);
        std::vector<int> wr_single_pos(num_unique_experts, 0);
        int p_off = 0, s_off = 2 * total_pairs;
        for (int e = 0; e < num_unique_experts; ++e) {
            wr_pair_pos[e] = p_off;
            wr_single_pos[e] = s_off;
            p_off += (expert_counts[e] / 2) * 2;
            s_off += (expert_counts[e] % 2);
        }
        std::vector<int> wr_tokens(total_acts), wr_ranks(total_acts);
        std::vector<ops::MoeExpertSlotDev> wr_slots(total_acts);
        std::vector<int> wr_seen(num_unique_experts, 0);
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < num_active_experts; ++k) {
                int e = (t * 7 + k) % num_unique_experts;
                int seen = wr_seen[e]++;
                int pairs_limit = (expert_counts[e] / 2) * 2;
                int p = (seen < pairs_limit) ? (wr_pair_pos[e] + seen) : (wr_single_pos[e] + (seen - pairs_limit));
                wr_tokens[p] = t;
                wr_ranks[p] = k;
                wr_slots[p] = expert_pool[e];
            }
        }
        Tensor wr_tok_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
        Tensor wr_rnk_dev = Tensor::allocate(ctx, {total_acts}, dnnl::memory::data_type::s32);
        ops::MoeExpertSlotDev* wr_slt_dev = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(total_acts * sizeof(ops::MoeExpertSlotDev)));
        ctx.memcpy_h2d(wr_tok_dev.data(), wr_tokens.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(wr_rnk_dev.data(), wr_ranks.data(), total_acts * sizeof(int));
        ctx.memcpy_h2d(wr_slt_dev, wr_slots.data(), total_acts * sizeof(ops::MoeExpertSlotDev));

        Tensor h_s4_dev = Tensor::allocate(ctx, {total_acts * intermediate_size}, dnnl::memory::data_type::bf16);
        Tensor down_acc_s4_dev = Tensor::allocate(ctx, {total_acts * out_features}, dnnl::memory::data_type::f32);
        Tensor y_s4_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

        Tensor h_wr_dev = Tensor::allocate(ctx, {total_acts * intermediate_size}, dnnl::memory::data_type::bf16);
        Tensor down_acc_wr_dev = Tensor::allocate(ctx, {total_acts * out_features}, dnnl::memory::data_type::f32);
        Tensor y_wr_dev = Tensor::allocate(ctx, {tokens, out_features}, dnnl::memory::data_type::bf16);

        // Warmup
        for (int i = 0; i < 5; ++i) {
            sycl::event gu = ops::moe_expert_major_prefill_gate_up(
                ctx, s4_slt_dev, static_cast<const int*>(s4_tok_dev.data()), static_cast<const int*>(s4_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(x_dev.data()), static_cast<bf16*>(h_s4_dev.data()),
                total_acts, num_active_experts, in_features, intermediate_size, blocksize, nullptr, false, 0);
            sycl::event dn = ops::moe_expert_major_prefill_down(
                ctx, s4_slt_dev, static_cast<const int*>(s4_tok_dev.data()), static_cast<const int*>(s4_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(h_s4_dev.data()), static_cast<float*>(down_acc_s4_dev.data()),
                total_acts, num_active_experts, intermediate_size, out_features, blocksize, &gu, false, 0);
            ops::moe_ordered_accumulate_down_rows(
                ctx, static_cast<const float*>(down_acc_s4_dev.data()), weights_dev, static_cast<bf16*>(y_s4_dev.data()),
                tokens, num_active_experts, out_features, nullptr, nullptr, &dn);
        }
        ctx.synchronize();

        constexpr int iters = 40;
        // Benchmark S4 baseline
        auto t0_s4 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            sycl::event gu = ops::moe_expert_major_prefill_gate_up(
                ctx, s4_slt_dev, static_cast<const int*>(s4_tok_dev.data()), static_cast<const int*>(s4_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(x_dev.data()), static_cast<bf16*>(h_s4_dev.data()),
                total_acts, num_active_experts, in_features, intermediate_size, blocksize, nullptr, false, 0);
            sycl::event dn = ops::moe_expert_major_prefill_down(
                ctx, s4_slt_dev, static_cast<const int*>(s4_tok_dev.data()), static_cast<const int*>(s4_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(h_s4_dev.data()), static_cast<float*>(down_acc_s4_dev.data()),
                total_acts, num_active_experts, intermediate_size, out_features, blocksize, &gu, false, 0);
            ops::moe_ordered_accumulate_down_rows(
                ctx, static_cast<const float*>(down_acc_s4_dev.data()), weights_dev, static_cast<bf16*>(y_s4_dev.data()),
                tokens, num_active_experts, out_features, nullptr, nullptr, &dn);
        }
        ctx.synchronize();
        auto t1_s4 = std::chrono::high_resolution_clock::now();
        double ms_s4 = std::chrono::duration<double, std::milli>(t1_s4 - t0_s4).count() / iters;

        // Warmup W4 C1
        for (int i = 0; i < 5; ++i) {
            sycl::event gu = ops::moe_expert_major_prefill_gate_up(
                ctx, wr_slt_dev, static_cast<const int*>(wr_tok_dev.data()), static_cast<const int*>(wr_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(x_dev.data()), static_cast<bf16*>(h_wr_dev.data()),
                total_acts, num_active_experts, in_features, intermediate_size, blocksize, nullptr, false, total_pairs);
            sycl::event dn = ops::moe_expert_major_prefill_down(
                ctx, wr_slt_dev, static_cast<const int*>(wr_tok_dev.data()), static_cast<const int*>(wr_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(h_wr_dev.data()), static_cast<float*>(down_acc_wr_dev.data()),
                total_acts, num_active_experts, intermediate_size, out_features, blocksize, &gu, false, total_pairs);
            ops::moe_ordered_accumulate_down_rows(
                ctx, static_cast<const float*>(down_acc_wr_dev.data()), weights_dev, static_cast<bf16*>(y_wr_dev.data()),
                tokens, num_active_experts, out_features, nullptr, nullptr, &dn);
        }
        ctx.synchronize();

        // Benchmark W4 C1
        auto t0_w4 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            sycl::event gu = ops::moe_expert_major_prefill_gate_up(
                ctx, wr_slt_dev, static_cast<const int*>(wr_tok_dev.data()), static_cast<const int*>(wr_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(x_dev.data()), static_cast<bf16*>(h_wr_dev.data()),
                total_acts, num_active_experts, in_features, intermediate_size, blocksize, nullptr, false, total_pairs);
            sycl::event dn = ops::moe_expert_major_prefill_down(
                ctx, wr_slt_dev, static_cast<const int*>(wr_tok_dev.data()), static_cast<const int*>(wr_rnk_dev.data()),
                nullptr, static_cast<const bf16*>(h_wr_dev.data()), static_cast<float*>(down_acc_wr_dev.data()),
                total_acts, num_active_experts, intermediate_size, out_features, blocksize, &gu, false, total_pairs);
            ops::moe_ordered_accumulate_down_rows(
                ctx, static_cast<const float*>(down_acc_wr_dev.data()), weights_dev, static_cast<bf16*>(y_wr_dev.data()),
                tokens, num_active_experts, out_features, nullptr, nullptr, &dn);
        }
        ctx.synchronize();
        auto t1_w4 = std::chrono::high_resolution_clock::now();
        double ms_w4 = std::chrono::duration<double, std::milli>(t1_w4 - t0_w4).count() / iters;

        // Verify bit-exact equality between S4 baseline and W4 C1 reuse
        std::vector<bf16> y_s4_h(tokens * out_features), y_wr_h(tokens * out_features);
        ctx.memcpy_d2h(y_s4_h.data(), y_s4_dev.data(), y_s4_h.size() * sizeof(bf16));
        ctx.memcpy_d2h(y_wr_h.data(), y_wr_dev.data(), y_wr_h.size() * sizeof(bf16));
        for (size_t i = 0; i < y_s4_h.size(); ++i) {
            if (y_s4_h[i] != y_wr_h[i]) {
                throw std::runtime_error("Benchmark verification failed: bit mismatch between S4 and W4 C1 at " + std::to_string(i));
            }
        }

        double speedup = ms_s4 / ms_w4;
        std::cout << "  Tokens=" << tokens << ", Acts=" << total_acts << " (" << total_pairs << " pairs, "
                  << (total_acts - 2 * total_pairs) << " singles across " << num_unique_experts << " experts)\n";
        std::cout << "    S4 Baseline (num_pairs=0):       " << ms_s4 << " ms\n";
        std::cout << "    W4 C1 2-Token Reuse (num_pairs=" << total_pairs << "): " << ms_w4 << " ms\n";
        std::cout << "    Speedup: " << speedup << "x (" << ((speedup - 1.0) * 100.0) << "% faster, 0 bit mismatches)\n";

        // WG-share reference timing (opt-in AILA_Q4E_4TOKEN_WGSHARE mapping).
        // Purely informational; the bit-exact claims are asserted in the G3
        // section above and in test_moe_prefill_wgshare_mapping_edges.
        {
            int total_quads = 0;
            for (int e = 0; e < num_unique_experts; ++e) total_quads += expert_counts[e] / 4;
            if (total_quads > 0) {
                std::vector<int> q_tok(4 * total_quads), q_rnk(4 * total_quads);
                std::vector<ops::MoeExpertSlotDev> q_slots(4 * total_quads);
                std::vector<int> q_seen(num_unique_experts, 0), q_pos(num_unique_experts, 0);
                int q_off = 0;
                for (int e = 0; e < num_unique_experts; ++e) {
                    q_pos[e] = q_off;
                    q_off += (expert_counts[e] / 4) * 4;
                }
                for (int t = 0; t < tokens; ++t) {
                    for (int k = 0; k < num_active_experts; ++k) {
                        int e = (t * 7 + k) % num_unique_experts;
                        int seen = q_seen[e]++;
                        if (seen >= (expert_counts[e] / 4) * 4) continue;
                        int p = q_pos[e] + seen;
                        q_tok[p] = t;
                        q_rnk[p] = k;
                        q_slots[p] = expert_pool[e];
                    }
                }
                Tensor q_tok_dev = Tensor::allocate(ctx, {4 * total_quads}, dnnl::memory::data_type::s32);
                Tensor q_rnk_dev = Tensor::allocate(ctx, {4 * total_quads}, dnnl::memory::data_type::s32);
                auto* q_slt_dev = static_cast<ops::MoeExpertSlotDev*>(
                    ctx.alloc_device(4 * total_quads * sizeof(ops::MoeExpertSlotDev)));
                ctx.memcpy_h2d(q_tok_dev.data(), q_tok.data(), q_tok.size() * sizeof(int));
                ctx.memcpy_h2d(q_rnk_dev.data(), q_rnk.data(), q_rnk.size() * sizeof(int));
                ctx.memcpy_h2d(q_slt_dev, q_slots.data(), q_slots.size() * sizeof(ops::MoeExpertSlotDev));
                Tensor q_h_dev = Tensor::allocate(ctx, {4 * total_quads * intermediate_size}, dnnl::memory::data_type::bf16);
                Tensor q_dn_dev = Tensor::allocate(ctx, {4 * total_quads * out_features}, dnnl::memory::data_type::f32);

                const bf16* x_rows = static_cast<const bf16*>(x_dev.data());
                auto launch_quads = [&](int share) {
                    sycl::event gu = share
                        ? ops::moe_expert_major_prefill_gate_up_4token_wgshare(
                              ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()), static_cast<const int*>(q_rnk_dev.data()),
                              nullptr, x_rows, static_cast<bf16*>(q_h_dev.data()),
                              total_quads, num_active_experts, in_features, intermediate_size, blocksize, nullptr, false, share, false)
                        : ops::moe_expert_major_prefill_gate_up_4token(
                              ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()), static_cast<const int*>(q_rnk_dev.data()),
                              nullptr, x_rows, static_cast<bf16*>(q_h_dev.data()),
                              total_quads, num_active_experts, in_features, intermediate_size, blocksize, nullptr, false, false);
                    sycl::event dn = share
                        ? ops::moe_expert_major_prefill_down_4token_wgshare(
                              ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()), static_cast<const int*>(q_rnk_dev.data()),
                              nullptr, static_cast<const bf16*>(q_h_dev.data()), static_cast<float*>(q_dn_dev.data()),
                              total_quads, num_active_experts, intermediate_size, out_features, blocksize, &gu, false, share, false)
                        : ops::moe_expert_major_prefill_down_4token(
                              ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()), static_cast<const int*>(q_rnk_dev.data()),
                              nullptr, static_cast<const bf16*>(q_h_dev.data()), static_cast<float*>(q_dn_dev.data()),
                              total_quads, num_active_experts, intermediate_size, out_features, blocksize, &gu, false, false);
                    return dn;
                };
                constexpr int q_iters = 20;
                auto bench_quads = [&](int share) {
                    for (int i = 0; i < 3; ++i) launch_quads(share).wait();
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < q_iters; ++i) launch_quads(share);
                    ctx.synchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    return std::chrono::duration<double, std::milli>(t1 - t0).count() / q_iters;
                };
                double ms_4t = bench_quads(0);
                std::cout << "    WG-share quads=" << total_quads << " (tokens=" << tokens << ")\n";
                std::cout << "      4token share=0: " << ms_4t << " ms\n";
                // Diagnostic: point every quad at token 0 so the activation
                // rows are L1/L2 resident. If the time barely moves, the ALU
                // family is instruction/register bound rather than limited by
                // the activation stream (used to judge int8-activation ideas).
                if (aila::env::read_flag("AILA_Q4E_BENCH_XLOCALITY", false)) {
                    std::vector<int> zero_tok(q_tok.size(), 0);
                    ctx.memcpy_h2d(q_tok_dev.data(), zero_tok.data(),
                                   zero_tok.size() * sizeof(int));
                    const double ms_zero = bench_quads(4);
                    // Production-scale activations: 4096 distinct rows (20 MiB for
                    // H=2560) so the quad reads stream instead of hitting L2.
                    constexpr int kBigRows = 4096;
                    Tensor x_big = Tensor::allocate(
                        ctx, {kBigRows, in_features}, dnnl::memory::data_type::bf16);
                    {
                        std::vector<bf16> xh(size_t(kBigRows) * in_features);
                        for (size_t i = 0; i < xh.size(); ++i)
                            xh[i] = bf16(float(int((i * 2654435761u) % 2001) - 1000) / 1000.0f);
                        ctx.memcpy_h2d(x_big.data(), xh.data(), xh.size() * sizeof(bf16));
                    }
                    std::vector<int> spread_tok(q_tok.size());
                    for (size_t i = 0; i < spread_tok.size(); ++i)
                        spread_tok[i] = int((i * 2654435761u) % kBigRows);
                    ctx.memcpy_h2d(q_tok_dev.data(), spread_tok.data(),
                                   spread_tok.size() * sizeof(int));
                    x_rows = x_big.data_as<bf16>();
                    const double ms_big = bench_quads(4);
                    x_rows = static_cast<const bf16*>(x_dev.data());
                    ctx.memcpy_h2d(q_tok_dev.data(), q_tok.data(),
                                   q_tok.size() * sizeof(int));
                    std::cout << "      [x-locality] all-quads-token0 share=4: " << ms_zero
                              << " ms (real ids share=4 measured below)\n";
                    std::cout << "      [x-locality] 4096-row streaming x share=4: " << ms_big
                              << " ms\n";
                }
                for (int share : {2, 4, 8}) {
                    double ms = bench_quads(share);
                    std::cout << "      4token share=" << share << ": " << ms << " ms ("
                              << (ms_4t / ms) << "x vs 4token, informational)\n";
                }
                // P4 bounding probe (opt-in AILA_Q4E_NOUNPACK_PROBE): the
                // nounpack variant keeps the identical row mapping and every
                // FMA, but replaces the 16 NF4 LUT+scale multiplies with the
                // raw scale, so the packed-weight loads and nibble extraction
                // vanish as dead code. Its delta upper-bounds what any
                // same-math dequant-reuse scheme could ever recover.
                if (aila::env::read_flag("AILA_Q4E_NOUNPACK_PROBE", false)) {
                    auto launch_gu_only = [&](bool nounpack, int share) {
                        if (nounpack) {
                            return ops::moe_expert_major_prefill_gate_up_4token_wgshare_nounpack(
                                ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()),
                                static_cast<const int*>(q_rnk_dev.data()), nullptr, x_rows,
                                static_cast<bf16*>(q_h_dev.data()), total_quads,
                                num_active_experts, in_features, intermediate_size,
                                blocksize, nullptr, false, share, false);
                        }
                        return ops::moe_expert_major_prefill_gate_up_4token_wgshare(
                            ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()),
                            static_cast<const int*>(q_rnk_dev.data()), nullptr, x_rows,
                            static_cast<bf16*>(q_h_dev.data()), total_quads,
                            num_active_experts, in_features, intermediate_size,
                            blocksize, nullptr, false, share, false);
                    };
                    auto bench_gu = [&](bool nounpack, int share) {
                        for (int i = 0; i < 3; ++i) launch_gu_only(nounpack, share).wait();
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < q_iters; ++i) launch_gu_only(nounpack, share);
                        ctx.synchronize();
                        auto t1 = std::chrono::high_resolution_clock::now();
                        return std::chrono::duration<double, std::milli>(t1 - t0).count() / q_iters;
                    };
                    for (int share : {2, 4, 8}) {
                        std::vector<double> ms_orig, ms_nop;
                        for (int round = 0; round < 3; ++round) {
                            if ((round % 2) == 0) {
                                ms_orig.push_back(bench_gu(false, share));
                                ms_nop.push_back(bench_gu(true, share));
                            } else {
                                ms_nop.push_back(bench_gu(true, share));
                                ms_orig.push_back(bench_gu(false, share));
                            }
                        }
                        std::sort(ms_orig.begin(), ms_orig.end());
                        std::sort(ms_nop.begin(), ms_nop.end());
                        const double m_orig = ms_orig[1];
                        const double m_nop = ms_nop[1];
                        std::cout << "      [P4 probe] share=" << share
                                  << " gate/up only orig=" << m_orig
                                  << " ms nounpack=" << m_nop
                                  << " ms delta=" << ((m_orig - m_nop) / m_orig * 100.0)
                                  << "% (rounds orig=" << ms_orig[0] << "/" << ms_orig[1]
                                  << "/" << ms_orig[2] << ", nounpack=" << ms_nop[0] << "/"
                                  << ms_nop[1] << "/" << ms_nop[2] << ")\n";
                    }
                }
                // P4 diagnostic V2 (opt-in AILA_Q4E_NOLUT_PROBE): same kernel
                // with the SLM LUT gathers replaced by int-to-float converts.
                // Discriminates "LUT gather chain is the cost" (shareable,
                // SLM could pay) from "packed-load chain is the cost" (not
                // shareable, P4 cannot pay).
                if (aila::env::read_flag("AILA_Q4E_NOLUT_PROBE", false)) {
                    auto launch_gu_only = [&](bool nolut, int share) {
                        if (nolut) {
                            return ops::moe_expert_major_prefill_gate_up_4token_wgshare_nolut(
                                ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()),
                                static_cast<const int*>(q_rnk_dev.data()), nullptr, x_rows,
                                static_cast<bf16*>(q_h_dev.data()), total_quads,
                                num_active_experts, in_features, intermediate_size,
                                blocksize, nullptr, false, share, false);
                        }
                        return ops::moe_expert_major_prefill_gate_up_4token_wgshare(
                            ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()),
                            static_cast<const int*>(q_rnk_dev.data()), nullptr, x_rows,
                            static_cast<bf16*>(q_h_dev.data()), total_quads,
                            num_active_experts, in_features, intermediate_size,
                            blocksize, nullptr, false, share, false);
                    };
                    auto bench_gu = [&](bool nolut, int share) {
                        for (int i = 0; i < 3; ++i) launch_gu_only(nolut, share).wait();
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < q_iters; ++i) launch_gu_only(nolut, share);
                        ctx.synchronize();
                        auto t1 = std::chrono::high_resolution_clock::now();
                        return std::chrono::duration<double, std::milli>(t1 - t0).count() / q_iters;
                    };
                    for (int share : {4}) {
                        std::vector<double> ms_orig, ms_nolut;
                        for (int round = 0; round < 3; ++round) {
                            if ((round % 2) == 0) {
                                ms_orig.push_back(bench_gu(false, share));
                                ms_nolut.push_back(bench_gu(true, share));
                            } else {
                                ms_nolut.push_back(bench_gu(true, share));
                                ms_orig.push_back(bench_gu(false, share));
                            }
                        }
                        std::sort(ms_orig.begin(), ms_orig.end());
                        std::sort(ms_nolut.begin(), ms_nolut.end());
                        std::cout << "      [P4 nolut] share=" << share
                                  << " gate/up only orig=" << ms_orig[1]
                                  << " ms nolut=" << ms_nolut[1]
                                  << " ms delta=" << ((ms_orig[1] - ms_nolut[1]) / ms_orig[1] * 100.0)
                                  << "% (rounds orig=" << ms_orig[0] << "/" << ms_orig[1]
                                  << "/" << ms_orig[2] << ", nolut=" << ms_nolut[0] << "/"
                                  << ms_nolut[1] << "/" << ms_nolut[2] << ")\n";
                    }
                }
                // P4 step 2: the share-4 SLM dequant-reuse gate/up kernel must
                // be bit-exact against the wgshare share=4 kernel; timing is a
                // gate/up-only A/B/B/A against that same reference.
                if (aila::env::read_flag("AILA_Q4E_SLM_REUSE", false)) {
                    const size_t h_elems = size_t(4 * total_quads) * intermediate_size;
                    Tensor h_ref_dev = Tensor::allocate(
                        ctx, {static_cast<int64_t>(h_elems)}, dnnl::memory::data_type::bf16);
                    Tensor h_slm_dev = Tensor::allocate(
                        ctx, {static_cast<int64_t>(h_elems)}, dnnl::memory::data_type::bf16);
                    // The quad map only covers complete 4-groups, so leftover
                    // single-act slots are never written; pre-fill both buffers
                    // identically before comparing.
                    {
                        std::vector<bf16> h_init(h_elems, bf16(0.375f));
                        ctx.memcpy_h2d(h_ref_dev.data(), h_init.data(),
                                       h_elems * sizeof(bf16));
                        ctx.memcpy_h2d(h_slm_dev.data(), h_init.data(),
                                       h_elems * sizeof(bf16));
                    }
                    auto launch_ref = [&]() {
                        return ops::moe_expert_major_prefill_gate_up_4token_wgshare(
                            ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()),
                            static_cast<const int*>(q_rnk_dev.data()), nullptr, x_rows,
                            static_cast<bf16*>(h_ref_dev.data()), total_quads,
                            num_active_experts, in_features, intermediate_size,
                            blocksize, nullptr, false, 4, false);
                    };
                    auto launch_slm = [&]() {
                        return ops::moe_expert_major_prefill_gate_up_4token_wgshare_slm(
                            ctx, q_slt_dev, static_cast<const int*>(q_tok_dev.data()),
                            static_cast<const int*>(q_rnk_dev.data()), nullptr, x_rows,
                            static_cast<bf16*>(h_slm_dev.data()), total_quads,
                            num_active_experts, in_features, intermediate_size,
                            blocksize, nullptr, false, false);
                    };
                    launch_ref().wait();
                    launch_slm().wait();
                    std::vector<bf16> h_ref_host(h_elems), h_slm_host(h_elems);
                    ctx.memcpy_d2h(h_ref_host.data(), h_ref_dev.data(),
                                   h_elems * sizeof(bf16));
                    ctx.memcpy_d2h(h_slm_host.data(), h_slm_dev.data(),
                                   h_elems * sizeof(bf16));
                    size_t mismatch = 0;
                    size_t first_bad = 0;
                    for (size_t i = 0; i < h_elems; ++i) {
                        if (h_ref_host[i] != h_slm_host[i]) {
                            if (mismatch == 0) first_bad = i;
                            ++mismatch;
                        }
                    }
                    std::cout << "      [P4 slm] gate/up bit-exact vs wgshare share=4: ";
                    if (mismatch == 0) {
                        std::cout << "PASS (" << h_elems << " elems)\n";
                    } else {
                        std::cout << "FAIL " << mismatch << "/" << h_elems
                                  << " first=" << first_bad << "\n";
                    }
                    auto bench_lane = [&](bool slm_arm) {
                        for (int i = 0; i < 3; ++i) {
                            (slm_arm ? launch_slm() : launch_ref()).wait();
                        }
                        auto t0 = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < q_iters; ++i) {
                            slm_arm ? launch_slm() : launch_ref();
                        }
                        ctx.synchronize();
                        auto t1 = std::chrono::high_resolution_clock::now();
                        return std::chrono::duration<double, std::milli>(t1 - t0).count() / q_iters;
                    };
                    std::vector<double> ms_ref, ms_slm;
                    for (int round = 0; round < 3; ++round) {
                        if ((round % 2) == 0) {
                            ms_ref.push_back(bench_lane(false));
                            ms_slm.push_back(bench_lane(true));
                        } else {
                            ms_slm.push_back(bench_lane(true));
                            ms_ref.push_back(bench_lane(false));
                        }
                    }
                    std::sort(ms_ref.begin(), ms_ref.end());
                    std::sort(ms_slm.begin(), ms_slm.end());
                    std::cout << "      [P4 slm] gate/up only ref share=4=" << ms_ref[1]
                              << " ms slm=" << ms_slm[1] << " ms delta="
                              << ((ms_ref[1] - ms_slm[1]) / ms_ref[1] * 100.0)
                              << "% (rounds ref=" << ms_ref[0] << "/" << ms_ref[1] << "/"
                              << ms_ref[2] << ", slm=" << ms_slm[0] << "/" << ms_slm[1]
                              << "/" << ms_slm[2] << ")\n";
                }
                ctx.free_device(q_slt_dev);
            }
        }

        ctx.free_device(s4_slt_dev);
        ctx.free_device(wr_slt_dev);
        ctx.free_device(weights_dev);
    }
    std::cout << "======================================================================\n\n";
}

// Strata C1/C2 screening: can an H2D staging-style memcpy submitted to a
// separate in-order "copy" queue overlap a compute kernel running on the main
// queue on this device/driver? If the two serialize, no amount of prefill
// stream-ahead can hide the staging DMA and the fix has to change where the
// copies execute (copy-only queue / engine).
void test_queue_overlap_probe(Context& ctx, bool json_output = false) {
    constexpr size_t kCopyBytes = 32ull << 20; // ~one 16-expert staging batch
    constexpr int kRounds = 16;
    constexpr int kBurnItems = 1 << 18;
    constexpr int kBurnIters = 8192;

    void* dev = ctx.alloc_device(kCopyBytes);
    void* host = ctx.alloc_host(kCopyBytes);
    std::memset(host, 0x5A, kCopyBytes);
    float* sink = static_cast<float*>(ctx.alloc_device(4 * sizeof(float)));
    sycl::queue copy_q(ctx.queue().get_context(), ctx.queue().get_device(),
                       sycl::property_list{sycl::property::queue::in_order()});

    auto burn_with_dep = [&](const sycl::event& dep) {
        return ctx.queue().submit([&](sycl::handler& cgh) {
            if (dep != sycl::event{}) cgh.depends_on(dep);
            cgh.parallel_for(sycl::nd_range<1>(kBurnItems, 256),
                             [=](sycl::nd_item<1> item) {
                                 const int gid = static_cast<int>(item.get_global_id(0));
                                 const float base = float(gid & 255) * 1e-3f;
                                 float a0 = base;
                                 float a1 = base + 1.0f;
                                 float a2 = base + 2.0f;
                                 float a3 = base + 3.0f;
                                 float b0 = base + 4.0f;
                                 float b1 = base + 5.0f;
                                 float b2 = base + 6.0f;
                                 float b3 = base + 7.0f;
                                 for (int i = 0; i < kBurnIters; ++i) {
                                     a0 = sycl::fma(a0, 1.000001f, 1e-6f);
                                     a1 = sycl::fma(a1, 1.000001f, 1e-6f);
                                     a2 = sycl::fma(a2, 1.000001f, 1e-6f);
                                     a3 = sycl::fma(a3, 1.000001f, 1e-6f);
                                     b0 = sycl::fma(b0, 1.000001f, 1e-6f);
                                     b1 = sycl::fma(b1, 1.000001f, 1e-6f);
                                     b2 = sycl::fma(b2, 1.000001f, 1e-6f);
                                     b3 = sycl::fma(b3, 1.000001f, 1e-6f);
                                 }
                                 const float v = a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3;
                                 if (v == 123456.789f) {
                                     sink[gid & 3] = v;
                                 }
                             });
        });
    };
    auto burn = [&]() { return burn_with_dep(sycl::event{}); };
    auto copy = [&]() { return copy_q.submit([&](sycl::handler& cgh) {
                            cgh.memcpy(dev, host, kCopyBytes);
                        }); };
    auto wall_ms = [](const std::chrono::high_resolution_clock::time_point& a,
                      const std::chrono::high_resolution_clock::time_point& b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    for (int i = 0; i < 3; ++i) burn();          // warmup / JIT
    copy(); copy_q.wait(); ctx.queue().wait();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) burn();
    ctx.queue().wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double burn_ms = wall_ms(t0, t1) / kRounds;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) copy();
    copy_q.wait();
    t1 = std::chrono::high_resolution_clock::now();
    const double copy_ms = wall_ms(t0, t1) / kRounds;

    // Dependent pipeline: kernel k waits for copy k, exactly like staging.
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) {
        const sycl::event cp = copy();
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.depends_on(cp);
            cgh.single_task([] {});
        });
        burn();
    }
    ctx.queue().wait(); copy_q.wait();
    t1 = std::chrono::high_resolution_clock::now();
    const double dep_ms = wall_ms(t0, t1) / kRounds;

    // Independent pipeline: no dependency between copy k and kernel k.
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) {
        copy();
        burn();
    }
    ctx.queue().wait(); copy_q.wait();
    t1 = std::chrono::high_resolution_clock::now();
    const double indep_ms = wall_ms(t0, t1) / kRounds;

    // Full two-bank ping-pong exactly like the staging pipeline: copy k waits
    // for the previous consumer of its bank (compute k-2), compute k waits for
    // copy k. This reverse dependency is what the real prefill inserts.
    void* bank_dev[2] = {ctx.alloc_device(kCopyBytes), ctx.alloc_device(kCopyBytes)};
    sycl::event bank_reader[2];
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) {
        const int bank = i & 1;
        const sycl::event reader = bank_reader[bank];
        const sycl::event cp = copy_q.submit([&](sycl::handler& cgh) {
            if (reader != sycl::event{}) cgh.depends_on(reader);
            cgh.memcpy(bank_dev[bank], host, kCopyBytes);
        });
        bank_reader[bank] = burn_with_dep(cp);
    }
    ctx.queue().wait(); copy_q.wait();
    t1 = std::chrono::high_resolution_clock::now();
    const double pingpong_ms = wall_ms(t0, t1) / kRounds;

    // Memory-heavy compute variant: the real expert kernels stream the staged
    // weights from VRAM, so screen whether the DMA stays hidden behind SM
    // memory traffic (which a pure-ALU burn kernel cannot answer).
    constexpr size_t kStreamBytes = 256ull << 20;
    float* stream = static_cast<float*>(ctx.alloc_device(kStreamBytes));
    const int64_t stream_elems = static_cast<int64_t>(kStreamBytes / sizeof(float));
    constexpr int kStreamPasses = 6;
    auto burn_mem_with_dep = [&](const sycl::event& dep) {
        return ctx.queue().submit([&](sycl::handler& cgh) {
            if (dep != sycl::event{}) cgh.depends_on(dep);
            cgh.parallel_for(sycl::nd_range<1>(kBurnItems, 256),
                             [=](sycl::nd_item<1> item) {
                                 const int gid = static_cast<int>(item.get_global_id(0));
                                 float acc = 0.0f;
                                 for (int pass = 0; pass < kStreamPasses; ++pass) {
                                     for (int64_t i = gid; i < stream_elems; i += kBurnItems) {
                                         acc = sycl::fma(stream[i], 1.000001f, acc);
                                     }
                                 }
                                 if (acc == 123456.789f) sink[gid & 3] = acc;
                             });
        });
    };
    for (int i = 0; i < 2; ++i) burn_mem_with_dep(sycl::event{}).wait();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) burn_mem_with_dep(sycl::event{});
    ctx.queue().wait();
    t1 = std::chrono::high_resolution_clock::now();
    const double burn_mem_ms = wall_ms(t0, t1) / kRounds;
    sycl::event mem_reader[2];
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) {
        const int bank = i & 1;
        const sycl::event reader = mem_reader[bank];
        const sycl::event cp = copy_q.submit([&](sycl::handler& cgh) {
            if (reader != sycl::event{}) cgh.depends_on(reader);
            cgh.memcpy(bank_dev[bank], host, kCopyBytes);
        });
        mem_reader[bank] = burn_mem_with_dep(cp);
    }
    ctx.queue().wait(); copy_q.wait();
    t1 = std::chrono::high_resolution_clock::now();
    const double pingpong_mem_ms = wall_ms(t0, t1) / kRounds;

    // (output is emitted after all measurements; see the consolidated block)

    // Everything on the main in-order queue: the strictly serial reference.
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kRounds; ++i) {
        ctx.queue().memcpy(dev, host, kCopyBytes);
        burn();
    }
    ctx.queue().wait();
    t1 = std::chrono::high_resolution_clock::now();
    const double same_ms = wall_ms(t0, t1) / kRounds;

    // Source-type screen: can the staging H2D read straight from the large-page
    // arena memory instead of the pinned bounce buffer (which costs a second
    // full host copy per byte)?
    auto time_copy_src = [&](void* src) {
        copy_q.wait();
        auto a = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kRounds; ++i) {
            copy_q.submit([&](sycl::handler& cgh) { cgh.memcpy(dev, src, kCopyBytes); });
        }
        copy_q.wait();
        auto b = std::chrono::high_resolution_clock::now();
        return wall_ms(a, b) / kRounds;
    };
    void* large = ctx.alloc_large_host(kCopyBytes);
    std::memset(large, 0x3C, kCopyBytes);
    const double pinned_ms = time_copy_src(host);
    const double large_ms = time_copy_src(large);
    void* dev_src = ctx.alloc_device(kCopyBytes);
    const double d2d_ms = time_copy_src(dev_src);
    if (json_output) {
        std::cout << "{\"probe\":\"queue-overlap\","
                  << "\"burn_ms\":" << burn_ms << ","
                  << "\"copy_ms\":" << copy_ms << ","
                  << "\"dep_ms\":" << dep_ms << ","
                  << "\"pingpong_ms\":" << pingpong_ms << ","
                  << "\"indep_ms\":" << indep_ms << ","
                  << "\"same_queue_ms\":" << same_ms << ","
                  << "\"burn_mem_ms\":" << burn_mem_ms << ","
                  << "\"pingpong_mem_ms\":" << pingpong_mem_ms << ","
                  << "\"serial_sum_ms\":" << (burn_ms + copy_ms) << ","
                  << "\"h2d_pinned_ms\":" << pinned_ms << ","
                  << "\"h2d_large_page_ms\":" << large_ms << ","
                  << "\"d2d_ms\":" << d2d_ms << ","
                  << "\"copy_bytes\":" << kCopyBytes << "}\n";
    } else {
        std::cout << "  [queue-overlap] burn=" << burn_ms << " ms copy=" << copy_ms
                  << " ms dep=" << dep_ms << " ms pingpong=" << pingpong_ms
                  << " ms indep=" << indep_ms
                  << " ms same-queue=" << same_ms << " ms"
                  << " (serial_sum=" << (burn_ms + copy_ms) << ")\n";
        std::cout << "  [queue-overlap-mem] burn_mem=" << burn_mem_ms
                  << " ms pingpong_mem=" << pingpong_mem_ms
                  << " ms (serial_sum=" << (burn_mem_ms + copy_ms)
                  << ", max=" << std::max(burn_mem_ms, copy_ms) << ")\n";
        std::cout << "  [stage-src] 32MiB H2D pinned=" << pinned_ms
                  << " ms (" << (kCopyBytes / (pinned_ms * 1e-3) / 1e9) << " GB/s)"
                  << " large-page=" << large_ms << " ms ("
                  << (kCopyBytes / (large_ms * 1e-3) / 1e9) << " GB/s)"
                  << " d2d=" << d2d_ms << " ms ("
                  << (kCopyBytes / (d2d_ms * 1e-3) / 1e9) << " GB/s)\n";
    }

    ctx.free_device(dev_src);
    ctx.free_host(large);
    ctx.free_device(stream);
    ctx.free_device(bank_dev[1]);
    ctx.free_device(bank_dev[0]);
    ctx.free_device(dev);
    ctx.free_host(host);
    ctx.free_device(sink);
}

// Strata C3 screen: time the production decode gate/up GEMV at the real
// per-token shape (10 experts) with and without the split activation load
// (AILA_Q4E_DECODE_XSPLIT). Bit-identical math; two processes, one flag.
void test_decode_gate_up_bench(Context& ctx) {
    constexpr int in_features = 2560;
    constexpr int intermediate = 640;
    constexpr int blocksize = 64;
    constexpr int experts = 10;
    const int packed_bytes = 2 * intermediate * (in_features / 2);
    const int absmax_count = 2 * intermediate * (in_features / blocksize);

    std::mt19937 rng(777);
    std::uniform_int_distribution<int> u8_dist(0, 255);
    std::uniform_real_distribution<float> f_dist(0.01f, 0.5f);
    std::vector<bf16> x_host(in_features);
    std::uniform_real_distribution<float> in_dist(-0.2f, 0.2f);
    for (auto& v : x_host) v = bf16(in_dist(rng));
    Tensor x_dev = Tensor::allocate(ctx, {in_features}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));

    std::vector<Tensor> packed_dev, absmax_dev;
    std::vector<ops::MoeExpertSlotDev> slots(experts);
    std::vector<uint8_t> packed_host(packed_bytes);
    std::vector<float> absmax_host(absmax_count);
    for (int e = 0; e < experts; ++e) {
        for (auto& v : packed_host) v = static_cast<uint8_t>(u8_dist(rng));
        for (auto& v : absmax_host) v = f_dist(rng);
        packed_dev.push_back(Tensor::allocate(ctx, {packed_bytes}, dnnl::memory::data_type::u8));
        absmax_dev.push_back(Tensor::allocate(ctx, {absmax_count}, dnnl::memory::data_type::f32));
        ctx.memcpy_h2d(packed_dev.back().data(), packed_host.data(), packed_bytes);
        ctx.memcpy_h2d(absmax_dev.back().data(), absmax_host.data(),
                       absmax_count * sizeof(float));
        slots[e].gate_up_packed = static_cast<const uint8_t*>(packed_dev.back().data());
        slots[e].gate_up_absmax = static_cast<const float*>(absmax_dev.back().data());
    }
    auto* slots_dev = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(experts * sizeof(ops::MoeExpertSlotDev)));
    ctx.memcpy_h2d(slots_dev, slots.data(), experts * sizeof(ops::MoeExpertSlotDev));
    Tensor h_dev = Tensor::allocate(ctx, {experts * intermediate}, dnnl::memory::data_type::bf16);

    auto run = [&]() {
        return ops::moe_decode_gate_up_swiglu_grouped(
            ctx, slots_dev, nullptr, static_cast<const bf16*>(x_dev.data()),
            static_cast<bf16*>(h_dev.data()), experts, in_features, intermediate, blocksize);
    };
    for (int i = 0; i < 5; ++i) run();
    ctx.synchronize();
    constexpr int iters = 200;
    std::vector<double> samples;
    for (int r = 0; r < 3; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) run();
        ctx.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    std::sort(samples.begin(), samples.end());
    const bool xsplit = aila::env::read_flag("AILA_Q4E_DECODE_XSPLIT", false);
    std::cout << "  [decode-gu-bench] experts=" << experts << " iters=" << iters
              << " xsplit=" << int(xsplit)
              << " median_ms=" << samples[1]
              << " rounds=" << samples[0] << "/" << samples[1] << "/" << samples[2]
              << " rows=" << (experts * intermediate) << "\n";
    ctx.free_device(slots_dev);
}

void test_direct_prefill_io(Context& ctx) {
    constexpr int dim = 2560;
    constexpr int block = 64;
    std::mt19937 rng(20260928);
    std::vector<uint8_t> packed(dim * dim / 2);
    std::vector<float> scales(dim * dim / block);
    for (auto& value : packed) value = static_cast<uint8_t>(rng());
    for (auto& value : scales) value = 0.01f + static_cast<float>(rng() % 100) * 0.002f;
    Tensor weight = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())}, dnnl::memory::data_type::u8);
    Tensor absmax = Tensor::allocate(ctx, {static_cast<int64_t>(scales.size())}, dnnl::memory::data_type::f32);
    Tensor qmap = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(weight.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(absmax.data(), scales.data(), scales.size() * sizeof(float));
    ctx.memcpy_h2d(qmap.data(), kTestQuantMap, sizeof(kTestQuantMap));
    Bnb4BitWeightRef ref;
    ref.name = "direct_prefill_test";
    ref.packed_weight = &weight;
    ref.absmax = &absmax;
    ref.quant_map = &qmap;
    ref.quant_state.shape = {dim, dim};
    ref.quant_state.blocksize = block;
    ref.quant_state.quant_type = "nf4";
    Bnb4BitLinear linear;
    std::string error;
    if (!linear.init(ctx, ref, &error)) throw std::runtime_error(error);
    linear.set_allow_async_dequant(true);
    Bnb4BitLinearScratch scratch;
    for (bool allow_xmx : {false, true}) {
        linear.set_allow_dense_xmx(allow_xmx);
        for (int rows : {1, 2, 4, 26, 64}) {
            const size_t elements = static_cast<size_t>(rows) * dim;
            const size_t bytes = elements * sizeof(bf16);
            std::vector<bf16> input(elements), expected(elements), actual(elements);
            for (auto& value : input) value = bf16((static_cast<int>(rng() % 101) - 50) * 0.01f);
            Tensor x = Tensor::allocate(ctx, {rows + 1, dim}, dnnl::memory::data_type::bf16);
            Tensor x2 = Tensor::allocate(ctx, {rows + 1, dim}, dnnl::memory::data_type::bf16);
            Tensor y = Tensor::allocate(ctx, {rows + 1, dim}, dnnl::memory::data_type::bf16);
            Tensor y2 = Tensor::allocate(ctx, {rows + 1, dim}, dnnl::memory::data_type::bf16);
            ctx.memcpy_h2d(x.data(), input.data(), bytes);
            ctx.memcpy_h2d(x2.data(), input.data(), bytes);
            linear.set_direct_prefill_io(false);
            linear.forward(ctx, scratch, x, y, rows);
            ctx.memcpy_d2h(expected.data(), y.data(), bytes);
            linear.set_direct_prefill_io(true);
            const auto before = Bnb4BitLinear::get_prefill_dispatch_stats().direct_io;
            linear.forward(ctx, scratch, x, y2, rows);
            ctx.memcpy_d2h(actual.data(), y2.data(), bytes);
            if (std::memcmp(expected.data(), actual.data(), bytes) != 0) {
                throw std::runtime_error(std::string("direct prefill output differs from staged output (xmx=") + (allow_xmx ? "1" : "0") + ")");
            }
            const auto after = Bnb4BitLinear::get_prefill_dispatch_stats().direct_io;
            if (after - before != static_cast<uint64_t>(rows > 1)) {
                throw std::runtime_error("direct I/O must bypass only multi-row prefill staging");
            }
            linear.forward(ctx, scratch, x2, y, rows);
            ctx.memcpy_d2h(actual.data(), y.data(), bytes);
            if (std::memcmp(expected.data(), actual.data(), bytes) != 0) {
                throw std::runtime_error("direct prefill cached bindings were not refreshed");
            }
            if (rows > 1) {
                Tensor arena = Tensor::allocate(ctx, {static_cast<int64_t>(elements + 1)}, dnnl::memory::data_type::bf16);
                auto source = Tensor::view(ctx, arena.data(), {rows, dim}, dnnl::memory::data_type::bf16);
                const auto before_alias = Bnb4BitLinear::get_prefill_dispatch_stats().direct_io;
                for (int offset : {0, 1}) {
                    auto target = Tensor::view(ctx, arena.data_as<bf16>() + offset, {rows, dim}, dnnl::memory::data_type::bf16);
                    ctx.memcpy_h2d(source.data(), input.data(), bytes);
                    linear.forward(ctx, scratch, source, target, rows);
                    ctx.memcpy_d2h(actual.data(), target.data(), bytes);
                    if (std::memcmp(expected.data(), actual.data(), bytes) != 0) {
                        throw std::runtime_error("overlapping prefill buffers must retain staged semantics");
                    }
                }
                if (Bnb4BitLinear::get_prefill_dispatch_stats().direct_io != before_alias) {
                    throw std::runtime_error("overlapping prefill must not take direct I/O path");
                }
            }
        }
    }
    std::cout << "[PASS] Direct/staged prefill (oneDNN + XMX), pointer rebinding and alias fallback: bit-exact\n";
}

void evaluate_dense_prefill_shape(Context& ctx, int K, int N, const std::string& desc) {
    constexpr int blocksize = 64;

    std::mt19937 rng(777 + K + N);
    std::uniform_int_distribution<int> u8_dist(0, 255);
    std::uniform_real_distribution<float> f_dist(0.01f, 0.5f);
    std::uniform_real_distribution<float> in_dist(-0.2f, 0.2f);

    const size_t packed_bytes = static_cast<size_t>(N) * (K / 2);
    const size_t absmax_count = static_cast<size_t>(N) * (K / blocksize);

    std::vector<uint8_t> packed_host(packed_bytes);
    std::vector<float> absmax_host(absmax_count);
    for (auto& v : packed_host) v = static_cast<uint8_t>(u8_dist(rng));
    for (auto& v : absmax_host) v = f_dist(rng);

    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed_bytes)}, dnnl::memory::data_type::u8);
    Tensor absmax_dev = Tensor::allocate(ctx, {static_cast<int64_t>(absmax_count)}, dnnl::memory::data_type::f32);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(packed_dev.data(), packed_host.data(), packed_bytes);
    ctx.memcpy_h2d(absmax_dev.data(), absmax_host.data(), absmax_count * sizeof(float));
    ctx.memcpy_h2d(qmap_dev.data(), kTestQuantMap, 16 * sizeof(float));

    Bnb4BitWeightRef wref;
    wref.name = "test_dense_weight";
    wref.packed_weight = &packed_dev;
    wref.absmax = &absmax_dev;
    wref.quant_map = &qmap_dev;
    wref.quant_state.shape = {N, K};
    wref.quant_state.blocksize = blocksize;
    wref.quant_state.quant_type = "nf4";

    Bnb4BitLinear linear;
    std::string err;
    if (!linear.init(ctx, wref, &err)) {
        throw std::runtime_error("linear.init failed for " + desc + ": " + err);
    }
    linear.set_allow_async_dequant(true);
    linear.set_direct_prefill_io(true);

    std::cout << "  --- Shape: " << desc << " (K=" << K << ", N=" << N << ") ---\n";
    for (int M : {4, 26, 30, 64}) {
        std::vector<bf16> x_host(static_cast<size_t>(M) * K);
        for (auto& v : x_host) v = bf16(in_dist(rng));
        Tensor x_dev = Tensor::allocate(ctx, {M, K}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));

        Tensor out_fallback = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
        Tensor out_xmx = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);

        Bnb4BitLinearScratch scratch_fb;
        linear.set_allow_dense_xmx(false);
        linear.set_force_dense_xmx(false);
        // Warmup
        for (int i = 0; i < 3; ++i) {
            linear.forward(ctx, scratch_fb, x_dev, out_fallback, M);
        }
        ctx.synchronize();

        constexpr int iters = 15;
        auto t0_fb = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            linear.forward(ctx, scratch_fb, x_dev, out_fallback, M);
        }
        ctx.synchronize();
        auto t1_fb = std::chrono::high_resolution_clock::now();
        double ms_fb = std::chrono::duration<double, std::milli>(t1_fb - t0_fb).count() / iters;

        Bnb4BitLinearScratch scratch_xmx;
        linear.set_allow_dense_xmx(true);
        linear.set_force_dense_xmx(true);
        // Warmup
        for (int i = 0; i < 3; ++i) {
            linear.forward(ctx, scratch_xmx, x_dev, out_xmx, M);
        }
        ctx.synchronize();

        auto t0_xmx = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            linear.forward(ctx, scratch_xmx, x_dev, out_xmx, M);
        }
        ctx.synchronize();
        auto t1_xmx = std::chrono::high_resolution_clock::now();
        double ms_xmx = std::chrono::duration<double, std::milli>(t1_xmx - t0_xmx).count() / iters;

        std::vector<bf16> y_fb_host(static_cast<size_t>(M) * N);
        std::vector<bf16> y_xmx_host(static_cast<size_t>(M) * N);
        ctx.memcpy_d2h(y_fb_host.data(), out_fallback.data(), y_fb_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(y_xmx_host.data(), out_xmx.data(), y_xmx_host.size() * sizeof(bf16));

        float max_abs_diff = 0.0f;
        double sum_sq_diff = 0.0;
        double sum_sq_fb = 0.0;
        double sum_sq_xmx = 0.0;
        double dot_prod = 0.0;
        for (size_t i = 0; i < y_fb_host.size(); ++i) {
            float v_fb = static_cast<float>(y_fb_host[i]);
            float v_xmx = static_cast<float>(y_xmx_host[i]);
            float diff = std::abs(v_fb - v_xmx);
            if (diff > max_abs_diff) max_abs_diff = diff;
            sum_sq_diff += diff * diff;
            sum_sq_fb += v_fb * v_fb;
            sum_sq_xmx += v_xmx * v_xmx;
            dot_prod += v_fb * v_xmx;
        }
        const size_t total_elems = y_fb_host.size();
        double rms_fb = std::sqrt(sum_sq_fb / total_elems);
        double rms_diff = std::sqrt(sum_sq_diff / total_elems);
        double nrmse = rms_diff / std::max(rms_fb, 1e-6);
        double cos_sim = dot_prod / std::max(std::sqrt(sum_sq_fb * sum_sq_xmx), 1e-12);

        std::cout << "    M=" << M << ": oneDNN=" << ms_fb << " ms | XMX=" << ms_xmx
                  << " ms | Speedup: " << (ms_fb / ms_xmx) << "x | max_diff=" << max_abs_diff
                  << " | NRMSE=" << (nrmse * 100.0) << "% | CosSim=" << cos_sim << "\n";

        if (nrmse > 0.015) {
            throw std::runtime_error("NRMSE exceeded tolerance threshold (1.5%) for " + desc + " M=" + std::to_string(M));
        }
        if (cos_sim < 0.998) {
            throw std::runtime_error("Cosine similarity dropped below tolerance (0.998) for " + desc + " M=" + std::to_string(M));
        }
    }
}

void test_non_moe_prefill_dense_xmx_evaluation(Context& ctx) {
    std::cout << "\n======================================================================\n";
    std::cout << "  [Evaluation] W4 C3 Non-MoE Dense Prefill: oneDNN vs XMX Joint Matrix\n";
    std::cout << "======================================================================\n";

    evaluate_dense_prefill_shape(ctx, 2560, 2560, "Standard Projection");
    evaluate_dense_prefill_shape(ctx, 2560, 16480, "Large FFN Gate/Up Projection");
    evaluate_dense_prefill_shape(ctx, 6144, 2560, "Linear Attention Projection");
    evaluate_dense_prefill_shape(ctx, 2560, 1280, "Shared Expert Gate/Up (Unfused, N=1280)");
    evaluate_dense_prefill_shape(ctx, 1280, 2560, "Shared Expert Down Projection (K=1280, N=2560)");

    std::cout << "======================================================================\n\n";
}

#include "Qwen4ExpQsaProbe.hpp"
#include "Qwen4ExpGdnDecodeOrder.hpp"
#include "Qwen4ExpGreedyTests.hpp"
#include "Qwen4ExpKvFp8Tests.hpp"
#include "Qwen4ExpDpasProbe.hpp"
#include "Qwen4ExpAttentionPairTests.hpp"
#include "Qwen4ExpW4A8Tests.hpp"

// R1-QSA route cost probe (2026-10-15). Hardened replacement for the
// 2026-09-30 header probe, which printed nothing between the device banner
// and the first per-length verification line, so a stall or a silent process
// death looked like "starts and immediately exits". Every GPU phase now
// prints a flushed [QSA PROBE] breadcrumb before it runs and the failure
// path reports stage + length. Emits [QSA COST] rows for the 2048/2051 dense
// controls (route keeps every block: pure selector overhead) and the
// 2052/4096/8192 sparse route (512-block budget, 2052 = first sparse length).
void run_qsa_route_cost_probe(Context& ctx) {
    std::cout << std::unitbuf;
    struct TerminateGuard {
        std::terminate_handler previous = std::get_terminate();
        TerminateGuard() {
            std::set_terminate([] {
                std::cout << "[QSA PROBE TERMINATE] unhandled termination inside the QSA probe; "
                             "last [QSA PROBE] breadcrumb above names the stalled stage\n";
                std::abort();
            });
        }
        ~TerminateGuard() { std::set_terminate(previous); }
    } terminate_guard;

    constexpr int heads = 24;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int topk = 512;
    constexpr int max_len = 8192;
    constexpr int tile = 128;
    constexpr int partial_stride = 272;
    constexpr int reps = 15;

    const char* stage = "host_fill";
    int length = 0;
    try {
        std::cout << "[QSA PROBE] stage=host_fill lengths=2048,2051,2052,4096,8192 seed=20261001\n";
        std::mt19937 rng(20261001);
        std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
        std::vector<bf16> q_h(heads * head_dim);
        std::vector<bf16> k_h(kv_heads * max_len * head_dim);
        std::vector<bf16> v_h(k_h.size());
        std::vector<bf16> iq_h(4 * 128);
        std::vector<bf16> ck_h((max_len / 4) * 128);
        for (auto* values : {&q_h, &k_h, &v_h, &iq_h, &ck_h})
            for (auto& x : *values) x = bf16(dist(rng));
        std::cout << "[QSA PROBE] stage=host_fill_done\n";

        stage = "allocate";
        Tensor q = Tensor::allocate(ctx, {1, heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor k = Tensor::allocate(ctx, {kv_heads, max_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor v = Tensor::allocate(ctx, {kv_heads, max_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor out = Tensor::allocate(ctx, {1, heads * head_dim}, dnnl::memory::data_type::bf16);
        Tensor iq = Tensor::allocate(ctx, {4, 128}, dnnl::memory::data_type::bf16);
        Tensor ck = Tensor::allocate(ctx, {max_len / 4, 128}, dnnl::memory::data_type::bf16);
        Tensor scores = Tensor::allocate(ctx, {max_len / 4}, dnnl::memory::data_type::f32);
        Tensor ids = Tensor::allocate(ctx, {topk + 4}, dnnl::memory::data_type::s32);
        Tensor attn_scores = Tensor::allocate(ctx, {heads, max_len}, dnnl::memory::data_type::f32);
        Tensor partials = Tensor::allocate(ctx, {heads, (max_len + tile - 1) / tile, partial_stride},
                                           dnnl::memory::data_type::f32);
        std::cout << "[QSA PROBE] stage=allocate_done\n";

        stage = "upload";
        ctx.memcpy_h2d(q.data(), q_h.data(), q_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(k.data(), k_h.data(), k_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(v.data(), v_h.data(), v_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(iq.data(), iq_h.data(), iq_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(ck.data(), ck_h.data(), ck_h.size() * sizeof(bf16));
        const auto live = ctx.device_allocation_count();
        std::cout << "[QSA PROBE] stage=upload_done\n";

        auto now = [] { return std::chrono::steady_clock::now(); };
        auto elapsed_ms = [&](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        // Warm + drain first, then report the warm wall time so a first-use
        // kernel build (seconds to tens of seconds) is visible in the log
        // instead of looking like a hang.
        auto measure_ms = [&](const char* what, auto&& fn) {
            auto t_warm = now();
            for (int i = 0; i < 3; ++i) fn();
            ctx.synchronize();
            std::cout << "[QSA PROBE] stage=warm_done what=" << what << " length=" << length
                      << " warm_ms=" << elapsed_ms(t_warm) << "\n";
            std::vector<double> us;
            us.reserve(reps);
            for (int i = 0; i < reps; ++i) {
                auto t0 = now();
                fn();
                ctx.synchronize();
                us.push_back(std::chrono::duration<double, std::micro>(now() - t0).count());
            }
            std::sort(us.begin(), us.end());
            return us[reps / 2] / 1000.0;
        };

        for (int len : {2048, 2051, 2052, 4096, 8192}) {
            length = len;
            const int blocks = len / 4;
            const int tail = len % 4;

            stage = "score";
            std::cout << "[QSA PROBE] stage=score length=" << len << " blocks=" << blocks << "\n";
            auto score = [&] { ops::qsa_compute_block_scores_compressed(ctx, iq, ck, scores, blocks); };
            auto t_score = now();
            score();
            ctx.synchronize();
            std::cout << "[QSA PROBE] stage=score_done length=" << len
                      << " ms=" << elapsed_ms(t_score) << "\n";

            stage = "select";
            std::cout << "[QSA PROBE] stage=select length=" << len << "\n";
            auto select = [&] {
                return ops::qsa_select_topk_blocks(ctx, scores, blocks, topk, len, 4, ids);
            };
            const int selected = select();
            ctx.synchronize();
            if (selected <= 0 || selected > topk || selected > blocks)
                throw std::runtime_error("QSA probe: invalid selector block count " + std::to_string(selected) +
                                         " at length " + std::to_string(len));
            std::vector<float> scores_h(blocks);
            std::vector<int> ids_h(selected + tail), ref_h(ids_h.size());
            ctx.memcpy_d2h(scores_h.data(), scores.data(), scores_h.size() * sizeof(float));
            ctx.memcpy_d2h(ids_h.data(), ids.data(), ids_h.size() * sizeof(int));
            const int ref_selected =
                ops::qsa_select_topk_blocks_host(scores_h.data(), blocks, topk, len, 4, ref_h.data());
            if (ref_selected != selected || ids_h != ref_h)
                throw std::runtime_error("QSA probe: selector IDs differ from the CPU oracle at length " +
                                         std::to_string(len));
            const bool drops_blocks = selected < blocks;
            std::cout << "[QSA PROBE] stage=select_done length=" << len << " selected=" << selected
                      << " tail=" << tail << " drops_blocks=" << (drops_blocks ? 1 : 0) << "\n";

            auto attention = [&](bool routed) {
                ops::attention_decode(ctx, q, k, v, out, attn_scores, heads, kv_heads, head_dim, len,
                                      &partials, nullptr, nullptr, 0,
                                      routed ? ids.data_as<int>() : nullptr, routed ? selected : 0);
            };
            auto verify = [&](bool routed) {
                stage = routed ? "verify_routed" : "verify_dense";
                std::cout << "[QSA PROBE] stage=" << stage << " length=" << len << "\n";
                attention(routed);
                ctx.synchronize();
                std::vector<bf16> got(heads * head_dim);
                ctx.memcpy_d2h(got.data(), out.data(), got.size() * sizeof(bf16));
                for (auto x : got)
                    if (!std::isfinite(float(x)))
                        throw std::runtime_error("QSA probe: non-finite attention output at length " +
                                                 std::to_string(len) + " routed=" + std::to_string(routed ? 1 : 0));
                std::vector<int> active;
                if (routed && drops_blocks) {
                    for (int b = 0; b < selected; ++b)
                        for (int j = 0; j < 4; ++j) active.push_back(ids_h[b] * 4 + j);
                    for (int j = 0; j < tail; ++j) active.push_back(ids_h[selected + j]);
                } else {
                    for (int t = 0; t < len; ++t) active.push_back(t);
                }
                double max_err = 0.0;
                for (int h : {0, 12}) {
                    const int kv = h / (heads / kv_heads);
                    std::vector<double> logits(active.size());
                    double mx = -1e100, sum = 0.0;
                    for (size_t t = 0; t < active.size(); ++t) {
                        double dot = 0.0;
                        for (int d = 0; d < head_dim; ++d)
                            dot += double(q_h[h * head_dim + d]) *
                                   double(k_h[(size_t(kv) * max_len + active[t]) * head_dim + d]);
                        logits[t] = dot / 16.0;
                        mx = std::max(mx, logits[t]);
                    }
                    for (auto& x : logits) { x = std::exp(x - mx); sum += x; }
                    for (int d = 0; d < head_dim; ++d) {
                        double value = 0.0;
                        for (size_t t = 0; t < active.size(); ++t)
                            value += logits[t] * double(v_h[(size_t(kv) * max_len + active[t]) * head_dim + d]);
                        max_err = std::max(max_err, std::abs(double(got[h * head_dim + d]) - value / sum));
                    }
                }
                if (max_err > 2e-4)
                    throw std::runtime_error("QSA probe: CPU f64 reference mismatch at length " +
                                             std::to_string(len) + " routed=" + std::to_string(routed ? 1 : 0) +
                                             " max_abs=" + std::to_string(max_err));
                std::cout << "[QSA CHECK] length=" << len << " routed=" << (routed ? 1 : 0)
                          << " selected=" << selected << " max_abs=" << max_err
                          << " reference_heads=2 finite_heads=" << heads << "\n";
            };
            verify(false);
            verify(true);

            auto routed = [&] {
                score();
                (void)select();
                attention(true);
            };
            for (int round = 0; round < 2; ++round) {
                const double dense_ms = measure_ms("dense", [&] { attention(false); });
                const double routed_ms = measure_ms("routed_post_projection", routed);
                const double net_pct = 100.0 * (dense_ms - routed_ms) / dense_ms;
                std::cout << "[QSA COST] {\"length\":" << len << ",\"sparse\":" << (drops_blocks ? 1 : 0)
                          << ",\"selected\":" << selected << ",\"round\":" << round
                          << ",\"dense_attention_ms\":" << dense_ms
                          << ",\"routed_post_projection_ms\":" << routed_ms
                          << ",\"route_net_pct\":" << net_pct << ",\"reps\":" << reps << "}\n";
            }
        }
        if (ctx.device_allocation_count() != live)
            throw std::runtime_error("QSA probe: live USM changed during the run");
        std::cout << "[QSA COST END] lengths=5 rounds=2 reps=" << reps << " model_free=1\n";
    } catch (const std::exception& e) {
        std::cout << "[QSA PROBE FAIL] stage=" << stage << " length=" << length << " what=" << e.what() << "\n";
        throw;
    }
}

// R1 decode-attention K/V re-read bench (2026-10-15, role K3). Times the
// production decode-attention entry point (ops::attention_decode_rows, which
// dispatches to the D=256 exact-partial-merge kernels at these lengths) on the
// audit shape Hq24/Hkv2/D256/bf16 KV for lengths {4096, 8192, 14336} and rows
// {1, 2}. Both rows read one shared K/V cache image, which is the retained
// pair-arm semantics; the per-row K/V traffic -- not the per-session
// footprint -- is the quantity under test. Emits one parseable
// [DECODE ATTN COST] JSON row per (length, rows) with warm_ms and per-row
// microsecond median/min/max, so a K/V-sharing phase-1 kernel can be A/B'd
// against the pre-change binary (adoption gate: >= 2x on median_us_per_row).
// Every GPU phase prints a flushed [DECODE ATTN] breadcrumb before it runs
// (same hardening as the QSA probe) so a stall names its stage instead of
// looking like a silent exit.
void run_decode_attn_bench(Context& ctx) {
    std::cout << std::unitbuf;
    struct TerminateGuard {
        std::terminate_handler previous = std::get_terminate();
        TerminateGuard() {
            std::set_terminate([] {
                std::cout << "[DECODE ATTN TERMINATE] unhandled termination inside the decode-attn bench; "
                             "last [DECODE ATTN] breadcrumb above names the stalled stage\n";
                std::abort();
            });
        }
        ~TerminateGuard() { std::set_terminate(previous); }
    } terminate_guard;

    constexpr int heads = 24;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int tile = 128;
    constexpr int partial_stride = 272;
    constexpr int max_len = 14336;
    constexpr int max_rows = 2;
    constexpr int reps = 20;
    const int max_tiles = (max_len + tile - 1) / tile;
    const int q_row_stride = heads * head_dim;
    const int row_bytes = q_row_stride * static_cast<int>(sizeof(bf16));

    const char* stage = "host_fill";
    int length = 0;
    int rows_now = 0;
    try {
        std::cout << "[DECODE ATTN] stage=host_fill lengths=4096,8192,14336 heads=24 kv_heads=2 "
                     "head_dim=256 dtype=bf16 rows=1,2 reps=" << reps << " seed=20261015 shared_kv_cache=1\n";
        std::mt19937 rng(20261015);
        std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
        std::vector<bf16> q_h(static_cast<size_t>(max_rows) * q_row_stride);
        std::vector<bf16> k_h(static_cast<size_t>(kv_heads) * max_len * head_dim);
        std::vector<bf16> v_h(k_h.size());
        for (auto* values : {&q_h, &k_h, &v_h})
            for (auto& x : *values) x = bf16(dist(rng));
        // Row 1 is an exact copy of row 0, so the rows=2 arm is a duplicate
        // row: any row-mapping bug shows up as a bitwise mismatch, not as a
        // timing artifact.
        for (int i = 0; i < q_row_stride; ++i) q_h[q_row_stride + i] = q_h[i];
        std::cout << "[DECODE ATTN] stage=host_fill_done\n";

        stage = "allocate";
        Tensor q = Tensor::allocate(ctx, {max_rows, q_row_stride}, dnnl::memory::data_type::bf16);
        Tensor k = Tensor::allocate(ctx, {kv_heads, max_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor v = Tensor::allocate(ctx, {kv_heads, max_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor out = Tensor::allocate(ctx, {max_rows, q_row_stride}, dnnl::memory::data_type::bf16);
        Tensor scores = Tensor::allocate(ctx, {heads, max_len}, dnnl::memory::data_type::f32);
        // [row, head, tile, stride] so the multi-row dispatcher takes its
        // private-partials-region path for rows=2, while the per-row path
        // slices a 3-D region out of the same buffer for rows=1.
        Tensor partials = Tensor::allocate(ctx, {max_rows, heads, max_tiles, partial_stride},
                                           dnnl::memory::data_type::f32);
        const auto live = ctx.device_allocation_count();
        std::cout << "[DECODE ATTN] stage=allocate_done partials_bytes="
                  << partials.numel() * static_cast<int>(sizeof(float)) << "\n";

        stage = "upload";
        ctx.memcpy_h2d(q.data(), q_h.data(), q_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(k.data(), k_h.data(), k_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(v.data(), v_h.data(), v_h.size() * sizeof(bf16));
        ctx.synchronize();
        std::cout << "[DECODE ATTN] stage=upload_done\n";

        auto now = [] { return std::chrono::steady_clock::now(); };
        auto elapsed_ms = [&](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };

        ops::DecodeRowGeometry rows[max_rows];
        auto run_rows = [&](int cached_len, int row_count) {
            for (int r = 0; r < row_count; ++r) {
                rows[r].k_cache = &k;
                rows[r].v_cache = &v;
                rows[r].cached_len = cached_len;
            }
            ops::attention_decode_rows(ctx, q, q_row_stride, out, scores, rows, row_count,
                                       heads, kv_heads, head_dim, cached_len, &partials);
        };

        // CPU f64 reference over two heads (0 and 12, one per kv head) plus
        // finite checks over every output. This is a coarse garbage guard; the
        // strict bit-equality evidence stays with the pair and rows suites.
        auto verify_output = [&](int cached_len) {
            stage = "verify";
            std::cout << "[DECODE ATTN] stage=verify length=" << cached_len << " rows=" << rows_now << "\n";
            ctx.synchronize();
            std::vector<bf16> got(out.numel());
            ctx.memcpy_d2h(got.data(), out.data(), got.size() * sizeof(bf16));
            for (auto x : got)
                if (!std::isfinite(static_cast<float>(x)))
                    throw std::runtime_error("decode-attn bench: non-finite attention output at length " +
                                             std::to_string(cached_len));
            if (rows_now == 2 &&
                std::memcmp(got.data(), got.data() + q_row_stride, static_cast<size_t>(row_bytes)) != 0)
                throw std::runtime_error("decode-attn bench: duplicate rows differ at length " +
                                         std::to_string(cached_len));
            double max_err = 0.0;
            for (int h : {0, 12}) {
                const int kv = h / (heads / kv_heads);
                std::vector<double> logits(cached_len);
                double mx = -1e100, sum = 0.0;
                for (int t = 0; t < cached_len; ++t) {
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += double(q_h[h * head_dim + d]) *
                               double(k_h[(size_t(kv) * max_len + t) * head_dim + d]);
                    logits[t] = dot / 16.0;
                    mx = std::max(mx, logits[t]);
                }
                for (auto& x : logits) { x = std::exp(x - mx); sum += x; }
                for (int d = 0; d < head_dim; ++d) {
                    double value = 0.0;
                    for (int t = 0; t < cached_len; ++t)
                        value += logits[t] * double(v_h[(size_t(kv) * max_len + t) * head_dim + d]);
                    max_err = std::max(max_err, std::abs(double(got[h * head_dim + d]) - value / sum));
                }
            }
            if (max_err > 2e-4)
                throw std::runtime_error("decode-attn bench: CPU f64 reference mismatch at length " +
                                         std::to_string(cached_len) + " max_abs=" + std::to_string(max_err));
            std::cout << "[DECODE ATTN CHECK] length=" << cached_len << " rows=" << rows_now
                      << " max_abs=" << max_err << " reference_heads=2 duplicate_rows="
                      << (rows_now == 2 ? 1 : 0) << "\n";
        };

        for (int cached_len : {4096, 8192, 14336}) {
            length = cached_len;
            for (int row_count : {1, 2}) {
                rows_now = row_count;
                stage = "warm";
                std::cout << "[DECODE ATTN] stage=warm length=" << cached_len << " rows=" << row_count << "\n";
                const auto t_warm = now();
                for (int i = 0; i < 3; ++i) run_rows(cached_len, row_count);
                ctx.synchronize();
                const double warm_ms = elapsed_ms(t_warm);
                std::cout << "[DECODE ATTN] stage=warm_done length=" << cached_len << " rows=" << row_count
                          << " warm_ms=" << warm_ms << "\n";
                verify_output(cached_len);

                stage = "measure";
                std::vector<double> us_per_row;
                us_per_row.reserve(reps);
                for (int i = 0; i < reps; ++i) {
                    const auto t0 = now();
                    run_rows(cached_len, row_count);
                    ctx.synchronize();
                    us_per_row.push_back(
                        std::chrono::duration<double, std::micro>(now() - t0).count() / row_count);
                }
                std::sort(us_per_row.begin(), us_per_row.end());
                const double median_us_per_row = us_per_row[reps / 2];
                std::cout << "[DECODE ATTN COST] {\"length\":" << cached_len
                          << ",\"rows\":" << row_count
                          << ",\"reps\":" << reps
                          << ",\"hq\":24,\"hkv\":2,\"head_dim\":256,\"dtype\":\"bf16\""
                          << ",\"warm_ms\":" << warm_ms
                          << ",\"median_ms_total\":" << median_us_per_row * row_count / 1000.0
                          << ",\"median_us_per_row\":" << median_us_per_row
                          << ",\"min_us_per_row\":" << us_per_row.front()
                          << ",\"max_us_per_row\":" << us_per_row.back()
                          << "}\n";
            }
        }
        if (ctx.device_allocation_count() != live)
            throw std::runtime_error("decode-attn bench: live USM changed during the run");
        std::cout << "[DECODE ATTN END] lengths=3 rows=2 reps=" << reps << " model_free=1\n";
    } catch (const std::exception& e) {
        std::cout << "[DECODE ATTN FAIL] stage=" << stage << " length=" << length
                  << " rows=" << rows_now << " what=" << e.what() << "\n";
        throw;
    }
}

// R3 decode-attention K/V reuse A/B arm (2026-10-15, role K5). Times the
// production decode entry point (ops::attention_decode_rows) on the audit shape
// Hq24/Hkv2/D256/bf16 KV at lengths {333, 1834, 4096, 8192, 14336} for rows
// {1, 2}. The variant is selected by the process-wide AILA_ATTN_KVREUSE env
// (read once inside AttentionOps.cpp; "pair2" = head-pair K/V reuse, anything
// else = the unchanged default path), so one process measures one variant and
// the A/B is a two-process comparison: run the arm once with
// AILA_ATTN_KVREUSE=0 and once with AILA_ATTN_KVREUSE=pair2.
//
// Cross-process bit-exactness: AILA_ATTN_KVREUSE_REF=<path> points at a small
// text ledger of FNV-1a 64 hashes over the raw bf16 bytes of output row 0. The
// first process (ledger absent) writes it and reports bitwise_eq=-1; a later
// process compares and reports bitwise_eq=1/0 per shape. Hashes are compared
// only for rows=1, because at rows=2 the D256 dispatcher selects the multi-row
// exact-merge kernel, which this round's variant does not modify: that row is a
// timing control and its hash is printed for cross-run eyeballing only.
// Pair2 is also inert below kDecodeExact256MinLen=1024 (len 333 keeps the
// short-context baseline kernel), which makes 333 the no-change control and
// 1834 the first shape the variant can move. Compare the two processes'
// median_us_per_row per (length, rows): >=3% faster at 4096 and 8192 with an
// A/A spread within +-1% adopts pair2; >2% slower at 333 or 1834 rejects it.
// Every GPU stage prints a flushed [DECODE ATTN REUSE] breadcrumb first, so a
// stall names its stage instead of looking like a silent exit.
uint64_t reuse_bench_fnv1a64(const void* data, size_t bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

void run_decode_attn_reuse_bench(Context& ctx) {
    std::cout << std::unitbuf;
    struct TerminateGuard {
        std::terminate_handler previous = std::get_terminate();
        TerminateGuard() {
            std::set_terminate([] {
                std::cout << "[DECODE ATTN REUSE TERMINATE] unhandled termination inside the "
                             "decode-attn reuse bench; last [DECODE ATTN REUSE] breadcrumb above "
                             "names the stalled stage\n";
                std::abort();
            });
        }
        ~TerminateGuard() { std::set_terminate(previous); }
    } terminate_guard;

    constexpr int heads = 24;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 256;
    constexpr int tile = 128;
    constexpr int partial_stride = 272;
    constexpr int max_len = 14336;
    constexpr int max_rows = 2;
    constexpr int reps = 20;
    constexpr int shape_count = 5;
    const int lens[shape_count] = {333, 1834, 4096, 8192, 14336};
    const int max_tiles = (max_len + tile - 1) / tile;
    const int q_row_stride = heads * head_dim;
    const int row_bytes = q_row_stride * static_cast<int>(sizeof(bf16));

    const char* env_mode = std::getenv("AILA_ATTN_KVREUSE");
    const std::string variant = env_mode ? std::string(env_mode) : std::string("<unset>");
    const char* ref_path_env = std::getenv("AILA_ATTN_KVREUSE_REF");
    const std::string ref_path = ref_path_env ? std::string(ref_path_env) : std::string();

    // Reference ledger, one "length rows hex_hash" line per (length, rows).
    std::vector<int> ref_len;
    std::vector<int> ref_rows;
    std::vector<uint64_t> ref_hash;
    bool ref_compare = false;

    const char* stage = "host_fill";
    int length = 0;
    int rows_now = 0;
    try {
        if (!ref_path.empty()) {
            stage = "ref_load";
            std::cout << "[DECODE ATTN REUSE] stage=ref_load path=" << ref_path << "\n";
            std::ifstream in(ref_path);
            if (in.good()) {
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty() || line[0] == '#') continue;
                    const size_t p1 = line.find(' ');
                    if (p1 == std::string::npos) continue;
                    const size_t p2 = line.find(' ', p1 + 1);
                    if (p2 == std::string::npos) continue;
                    ref_len.push_back(std::atoi(line.substr(0, p1).c_str()));
                    ref_rows.push_back(std::atoi(line.substr(p1 + 1, p2 - p1 - 1).c_str()));
                    ref_hash.push_back(std::strtoull(line.substr(p2 + 1).c_str(), nullptr, 16));
                }
                ref_compare = true;
            }
            std::cout << "[DECODE ATTN REUSE] stage=ref_load_done mode="
                      << (ref_compare ? "compare" : "write")
                      << " entries=" << ref_len.size() << "\n";
        }
        std::cout << "[DECODE ATTN REUSE META] {\"kvreuse\":\"" << variant
                  << "\",\"head_dim\":256,\"hq\":24,\"hkv\":2,\"dtype\":\"bf16\""
                  << ",\"lengths\":[333,1834,4096,8192,14336],\"rows\":[1,2]"
                  << ",\"reps\":" << reps
                  << ",\"ref_mode\":\"" << (ref_compare ? "compare" : (ref_path.empty() ? "off" : "write"))
                  << "\",\"pair2_len_floor\":1024,\"rows2_pair2_scope\":0,\"seed\":20261015}\n";

        stage = "host_fill";
        std::mt19937 rng(20261015);
        std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
        std::vector<bf16> q_h(static_cast<size_t>(max_rows) * q_row_stride);
        std::vector<bf16> k_h(static_cast<size_t>(kv_heads) * max_len * head_dim);
        std::vector<bf16> v_h(k_h.size());
        for (auto* values : {&q_h, &k_h, &v_h})
            for (auto& x : *values) x = bf16(dist(rng));
        // Row 1 duplicates row 0, so a row-mapping bug is a bitwise mismatch
        // instead of a timing artifact.
        for (int i = 0; i < q_row_stride; ++i) q_h[q_row_stride + i] = q_h[i];
        std::cout << "[DECODE ATTN REUSE] stage=host_fill_done\n";

        stage = "allocate";
        Tensor q = Tensor::allocate(ctx, {max_rows, q_row_stride}, dnnl::memory::data_type::bf16);
        Tensor k = Tensor::allocate(ctx, {kv_heads, max_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor v = Tensor::allocate(ctx, {kv_heads, max_len, head_dim}, dnnl::memory::data_type::bf16);
        Tensor out = Tensor::allocate(ctx, {max_rows, q_row_stride}, dnnl::memory::data_type::bf16);
        Tensor scores = Tensor::allocate(ctx, {heads, max_len}, dnnl::memory::data_type::f32);
        Tensor partials = Tensor::allocate(ctx, {max_rows, heads, max_tiles, partial_stride},
                                           dnnl::memory::data_type::f32);
        const auto live = ctx.device_allocation_count();
        std::cout << "[DECODE ATTN REUSE] stage=allocate_done partials_bytes="
                  << partials.numel() * static_cast<int>(sizeof(float)) << "\n";

        stage = "upload";
        ctx.memcpy_h2d(q.data(), q_h.data(), q_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(k.data(), k_h.data(), k_h.size() * sizeof(bf16));
        ctx.memcpy_h2d(v.data(), v_h.data(), v_h.size() * sizeof(bf16));
        ctx.synchronize();
        std::cout << "[DECODE ATTN REUSE] stage=upload_done\n";

        auto now = [] { return std::chrono::steady_clock::now(); };
        auto elapsed_ms = [&](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };

        ops::DecodeRowGeometry rows[max_rows];
        auto run_rows = [&](int cached_len, int row_count) {
            for (int r = 0; r < row_count; ++r) {
                rows[r].k_cache = &k;
                rows[r].v_cache = &v;
                rows[r].cached_len = cached_len;
            }
            ops::attention_decode_rows(ctx, q, q_row_stride, out, scores, rows, row_count,
                                       heads, kv_heads, head_dim, cached_len, &partials);
        };

        std::ofstream ref_out;
        if (!ref_path.empty() && !ref_compare) {
            ref_out.open(ref_path, std::ios::out | std::ios::trunc);
            if (!ref_out.good())
                throw std::runtime_error("decode-attn reuse bench: cannot write reference ledger " + ref_path);
            ref_out << "# Aila decode-attn K/V reuse reference, FNV-1a 64 over the raw bf16 bytes of"
                       " output row 0 (length rows hash)\n";
        }
        int missing_refs = 0;

        for (int li = 0; li < shape_count; ++li) {
            const int cached_len = lens[li];
            length = cached_len;
            stage = "reference";
            std::cout << "[DECODE ATTN REUSE] stage=reference length=" << cached_len << "\n";
            // f64 CPU reference for heads 0 and 12 (one per kv-head) on q row 0.
            std::vector<double> ref_probs[2];
            for (int hi = 0; hi < 2; ++hi) {
                const int h = hi * 12;
                const int kv = h / (heads / kv_heads);
                std::vector<double> logits(cached_len);
                double mx = -1e100, sum = 0.0;
                for (int t = 0; t < cached_len; ++t) {
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; ++d)
                        dot += double(q_h[h * head_dim + d]) *
                               double(k_h[(size_t(kv) * max_len + t) * head_dim + d]);
                    logits[t] = dot / 16.0;
                    mx = std::max(mx, logits[t]);
                }
                for (auto& x : logits) { x = std::exp(x - mx); sum += x; }
                for (auto& x : logits) x /= sum;
                ref_probs[hi] = logits;
            }
            std::cout << "[DECODE ATTN REUSE] stage=reference_done length=" << cached_len << "\n";

            for (int row_count : {1, 2}) {
                rows_now = row_count;
                stage = "warm";
                std::cout << "[DECODE ATTN REUSE] stage=warm length=" << cached_len
                          << " rows=" << row_count << "\n";
                const auto t_warm = now();
                for (int i = 0; i < 3; ++i) run_rows(cached_len, row_count);
                ctx.synchronize();
                const double warm_ms = elapsed_ms(t_warm);
                std::cout << "[DECODE ATTN REUSE] stage=warm_done length=" << cached_len
                          << " rows=" << row_count << " warm_ms=" << warm_ms << "\n";

                stage = "verify";
                std::vector<bf16> got(out.numel());
                ctx.memcpy_d2h(got.data(), out.data(), got.size() * sizeof(bf16));
                for (auto x : got)
                    if (!std::isfinite(static_cast<float>(x)))
                        throw std::runtime_error("decode-attn reuse bench: non-finite output at length " +
                                                 std::to_string(cached_len));
                const bool dup_eq = row_count == 1 ||
                    std::memcmp(got.data(), got.data() + q_row_stride,
                                static_cast<size_t>(row_bytes)) == 0;
                if (!dup_eq)
                    throw std::runtime_error("decode-attn reuse bench: duplicate rows differ at length " +
                                             std::to_string(cached_len));
                double max_err = 0.0;
                for (int hi = 0; hi < 2; ++hi) {
                    const int h = hi * 12;
                    const int kv = h / (heads / kv_heads);
                    for (int d = 0; d < head_dim; ++d) {
                        double value = 0.0;
                        for (int t = 0; t < cached_len; ++t)
                            value += ref_probs[hi][t] *
                                     double(v_h[(size_t(kv) * max_len + t) * head_dim + d]);
                        max_err = std::max(max_err,
                                           std::abs(double(got[h * head_dim + d]) - value));
                    }
                }
                if (max_err > 2e-4)
                    throw std::runtime_error("decode-attn reuse bench: CPU f64 reference mismatch at length " +
                                             std::to_string(cached_len) + " max_abs=" + std::to_string(max_err));
                // Row 0 of the output is the q row 0 result in both arms.
                const uint64_t row_hash = reuse_bench_fnv1a64(got.data(), static_cast<size_t>(row_bytes));
                std::cout << "[DECODE ATTN REUSE CHECK] length=" << cached_len
                          << " rows=" << row_count << " max_abs=" << max_err
                          << " duplicate_rows=" << (row_count == 2 ? 1 : 0)
                          << " row0_hash=" << std::hex << row_hash << std::dec << "\n";

                long long bitwise_eq = -1;
                const char* ref_mode = ref_path.empty() ? "off" : (ref_compare ? "compare" : "write");
                if (!ref_path.empty()) {
                    stage = "ref_compare";
                    if (ref_compare) {
                        int found = -1;
                        for (size_t i = 0; i < ref_len.size(); ++i)
                            if (ref_len[i] == cached_len && ref_rows[i] == row_count) { found = static_cast<int>(i); break; }
                        if (found < 0) {
                            // Only rows=1 is pair2 scope, so a missing rows=2
                            // entry is reported as absent, not as a failure.
                            if (row_count == 1) ++missing_refs;
                            bitwise_eq = -1;
                            ref_mode = "compare_missing";
                        } else {
                            bitwise_eq = (ref_hash[found] == row_hash) ? 1 : 0;
                        }
                    } else if (ref_out.good()) {
                        ref_out << cached_len << " " << row_count << " " << std::hex << row_hash
                                << std::dec << "\n";
                        ref_out.flush();
                    }
                }
                if (bitwise_eq == 0) {
                    std::cout << "[DECODE ATTN REUSE MISMATCH] length=" << cached_len
                              << " rows=" << row_count << " variant=" << variant
                              << " reference differs from this run's row-0 bytes\n";
                    if (variant == "pair2" && row_count == 1)
                        throw std::runtime_error("decode-attn reuse bench: pair2 row-0 output is not "
                                                 "bit-identical to the reference at length " +
                                                 std::to_string(cached_len));
                }

                stage = "measure";
                std::vector<double> us_per_row;
                us_per_row.reserve(reps);
                for (int i = 0; i < reps; ++i) {
                    const auto t0 = now();
                    run_rows(cached_len, row_count);
                    ctx.synchronize();
                    us_per_row.push_back(
                        std::chrono::duration<double, std::micro>(now() - t0).count() / row_count);
                }
                std::sort(us_per_row.begin(), us_per_row.end());
                const double median_us_per_row = us_per_row[reps / 2];
                std::cout << "[DECODE ATTN REUSE] {\"length\":" << cached_len
                          << ",\"rows\":" << row_count
                          << ",\"variant\":\"" << variant << "\""
                          << ",\"reps\":" << reps
                          << ",\"warm_ms\":" << warm_ms
                          << ",\"median_ms_total\":" << median_us_per_row * row_count / 1000.0
                          << ",\"median_us_per_row\":" << median_us_per_row
                          << ",\"min_us_per_row\":" << us_per_row.front()
                          << ",\"max_us_per_row\":" << us_per_row.back()
                          << ",\"finite\":1"
                          << ",\"rows_duplicate_eq\":" << (dup_eq ? 1 : 0)
                          << ",\"max_abs\":" << max_err
                          << ",\"row_hash\":\"" << std::hex << row_hash << std::dec << "\""
                          << ",\"bitwise_eq\":" << bitwise_eq
                          << ",\"ref_mode\":\"" << ref_mode << "\""
                          << ",\"pair2_scope\":" << (row_count == 1 ? 1 : 0)
                          << "}\n";
            }
        }
        if (ref_out.is_open()) ref_out.close();
        if (missing_refs > 0)
            throw std::runtime_error("decode-attn reuse bench: reference ledger is missing " +
                                     std::to_string(missing_refs) + " rows=1 entries");
        if (ctx.device_allocation_count() != live)
            throw std::runtime_error("decode-attn reuse bench: live USM changed during the run");
        std::cout << "[DECODE ATTN REUSE END] shapes=" << shape_count << " rows=2 reps=" << reps
                  << " variant=" << variant
                  << " ref_mode=" << (ref_path.empty() ? "off" : (ref_compare ? "compare" : "write"))
                  << " model_free=1\n";
    } catch (const std::exception& e) {
        std::cout << "[DECODE ATTN REUSE FAIL] stage=" << stage << " length=" << length
                  << " rows=" << rows_now << " what=" << e.what() << "\n";
        throw;
    }
}

} // namespace

int main(int argc, char** argv) {
    aila::env::enable_l0_copy_engine();
    try {
        Context ctx;
        std::cout << "Starting Qwen4Exp Kernel Tests on device: "
                  << ctx.queue().get_device().get_info<sycl::info::device::name>() << "\n";

        if (argc == 2 && std::string(argv[1]) == "--hc-only") {
            test_hc_decode_fused(ctx,true); test_hc_decode_fused(ctx,false);
            test_hc_rows_suite(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("HC test teardown USM leak");
            std::cout << "HC-only tests PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--ple-device-head-only") {
            test_ple_dilated_conv_silu_add_device_head_graph(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("PLE device-head test teardown USM leak");
            std::cout << "PLE device-head graph test PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qsa-graph-only") {
            test_qsa_graph_position_and_block_advance(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA graph test teardown USM leak");
            std::cout << "QSA graph device-state test PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qsa-selector-graph-only") {
            test_qsa_graph_selector_radix(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA graph selector test teardown USM leak");
            std::cout << "QSA graph selector test PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qsa-per-layer-only") {
            test_qsa_graph_per_layer_block_advance(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA per-layer test teardown USM leak");
            std::cout << "QSA per-layer graph test PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--attention-graph-only") {
            test_attention_decode_block_native_graph(ctx);
            test_attention_decode_block_native_graph_long_capacity(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("Attention graph test teardown USM leak");
            std::cout << "Attention graph test PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qgkv-graph-only") {
            test_decode_prepare_qgkv_graph(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QGKV graph test teardown USM leak");
            std::cout << "QGKV graph test PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--gdn-order-only") {
            test_gated_delta_net_sigmoid(ctx);
            test_gdn_device_conv_head(ctx);
            test_gdn_decode_order(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("GDN test teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--mtp-greedy-only") {
            test_mtp_greedy(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if(ctx.device_allocation_count() || ctx.pinned_host_allocation_count())throw std::runtime_error("Greedy teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--dpas-probe") {
            test_int8_dpas_probe(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("DPAS probe teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--queue-overlap-probe") {
            test_queue_overlap_probe(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("Queue overlap probe teardown USM leak");
            std::cout << "Queue overlap probe completed; teardown USM zero\n";
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--queue-overlap-probe" &&
            std::string(argv[2]) == "--json") {
            test_queue_overlap_probe(ctx, true);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("Queue overlap probe teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--decode-gu-bench") {
            test_decode_gate_up_bench(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("Decode GU bench teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--kv-int8-dpas-only") {
            test_int8_dpas_probe(ctx);
            test_int8_dpas_attention(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("INT8 DPAS attention teardown USM leak");
            return 0;
        }

        if (argc == 2 && std::string(argv[1]) == "--kv-fp8-only") {
            test_q4e_kv_fp8(ctx);
            ctx.synchronize();ctx.release_execution_resources();
            if(ctx.device_allocation_count() || ctx.pinned_host_allocation_count())throw std::runtime_error("FP8 teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--attention-pair-only") {
            test_q4e_attention_pair(ctx);
            ctx.synchronize();ctx.release_execution_resources();
            if(ctx.device_allocation_count() || ctx.pinned_host_allocation_count())throw std::runtime_error("Pair teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qsa-probe") {
            run_qsa_route_cost_probe(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if(ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA probe teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--decode-attn-bench") {
            run_decode_attn_bench(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if(ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("Decode attention bench teardown USM leak");
            return 0;
        }
        if ((argc == 2 || argc == 3) && std::string(argv[1]) == "--decode-attn-reuse-bench" &&
            (argc == 2 || std::string(argv[2]) == "--json")) {
            run_decode_attn_reuse_bench(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if(ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("Decode attention reuse bench teardown USM leak");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--w4a8-only") {
            test_q4e_w4a8_adaptive_decode(ctx, true);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("W4A8 teardown USM leak");
            std::cout << "W4A8 adaptive decode checks PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--prefill-bench-only") {
            test_moe_prefill_gate_up_and_down(ctx);
            test_moe_prefill_wgshare_mapping_edges(ctx);
            test_moe_prefill_weight_reuse_benchmark(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("Prefill bench teardown USM leak");
            std::cout << "Prefill benchmark completed; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qsa-prefill-only") {
            test_attention_prefill_qsa_batched(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA prefill teardown USM leak");
            std::cout << "QSA prefill checks PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qsa-tests") {
            test_qsa_graph_position_and_block_advance(ctx);
            test_qsa_graph_per_layer_block_advance(ctx);
            test_attention_decode_block_native_graph(ctx);
            test_attention_decode_block_native_graph_long_capacity(ctx);
            test_decode_prepare_qgkv_graph(ctx);
            test_qsa_layer_graph_composite(ctx);
            test_qsa_selector_adversarial(ctx);
            test_qsa_select_topk_blocks(ctx);
            test_qsa_batched_scoring_and_selection(ctx);
            test_attention_prefill_qsa_batched(ctx);
            test_attention_decode_with_qsa_indices(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA tests teardown USM leak");
            std::cout << "QSA tests PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--qsa-layer-graph-only") {
            test_qsa_layer_graph_composite(ctx);
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA layer graph teardown USM leak");
            std::cout << "QSA layer graph checks PASSED; teardown USM zero\n";
            return 0;
        }
        if (argc == 2 && (std::string(argv[1]) == "--qsa-batches" || std::string(argv[1]) == "--qsa-profile")) {
            test_qsa_selector_batches(ctx, std::string(argv[1]) == "--qsa-profile");
            ctx.synchronize(); ctx.release_execution_resources();
            if (ctx.device_allocation_count() || ctx.pinned_host_allocation_count())
                throw std::runtime_error("QSA batches teardown USM leak");
            return 0;
        }
        test_grouped_rms_norm(ctx);
        test_hc_mix_reduce(ctx);
        test_hc_inject_add(ctx);
        test_hc_inject_add_and_grouped_rms_norm(ctx);
        test_hc_decode_fused(ctx, true);
        test_hc_decode_fused(ctx, false);
        test_hc_rows_suite(ctx);
        test_ple_gate_value(ctx);
        test_ple_dilated_conv_silu_add(ctx);
        test_ple_dilated_conv_silu_add_residual(ctx);
        test_ple_dilated_conv_silu_add_device_head_graph(ctx);
        test_router_softmax_topk(ctx);
        test_moe_decode_gate_up_and_down(ctx);
        test_q4e_w4a8_adaptive_decode(ctx, false);
        test_moe_prefill_gate_up_and_down(ctx);
        test_moe_prefill_wgshare_mapping_edges(ctx);
        test_moe_prefill_weight_reuse_benchmark(ctx);
        test_non_moe_prefill_dense_xmx_evaluation(ctx);
        test_direct_prefill_io(ctx);
        test_gated_delta_net_sigmoid(ctx);
        test_gdn_device_conv_head(ctx);
        test_attention_decode_gate_fusion(ctx);
        test_attention_prefill_gate_fusion(ctx);
        test_attention_prefill_cached_gate_fusion(ctx);
        test_split_qgkv(ctx);

    // QSA Indexer (M2) tests
        test_qsa_prepare_q_and_cache_k(ctx);
        test_qsa_cache_k_batched(ctx);
        test_qsa_compute_block_scores(ctx);
        test_qsa_select_topk_and_expand_tokens(ctx);
        test_attention_decode_with_qsa_indices(ctx);
        test_qsa_compressed_k_scoring(ctx);
        test_qsa_select_topk_blocks(ctx);
        test_qsa_selector_adversarial(ctx);
        test_qsa_prepare_q_batched(ctx);
        test_qsa_batched_scoring_and_selection(ctx);
        test_attention_prefill_qsa_batched(ctx);

        std::cout << "All Qwen4Exp Kernel Tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Test failed: " << e.what() << "\n" << std::flush;
        return 1;
    } catch (...) {
        std::cout << "Test failed: unknown exception\n" << std::flush;
        return 1;
    }
}
