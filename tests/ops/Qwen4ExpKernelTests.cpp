#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/HyperConnectionOps.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/GatedDeltaNetOps.hpp"
#include "ops/QsaIndexerOps.hpp"
#include "ops/Ops.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

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

void test_hc_decode_fused(Context& ctx, bool with_inject) {
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    constexpr int lowrank = 320;
    constexpr int k_dim = groups * group_dim;
    auto round_bf16 = [](float v) { return static_cast<float>(bf16(v)); };
    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };

    std::mt19937 rng(with_inject ? 789 : 790);
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
    const std::string tag = with_inject ? " (inject)" : " (mixer)";
    check_close(act, ref_act, "hc_decode_down_inject act" + tag, 1e-2f);
    check_close(x, ref_x, "hc_decode_up_mix x" + tag, 1e-2f);
    check_close_f32(inj, with_inject ? ref_inj : inj_sentinel, "hc_decode_down_inject inj" + tag, 1e-2f);
}

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

    std::cout << "[PASS] moe_prefill tokens=" << tokens
              << " k=" << num_active_experts
              << " nested=" << (use_nested ? "true" : "false")
              << " bit-exact match (0 mismatches across " << ref_y.size() << " values, both S1 and S4)\n";

    ctx.free_device(slots_dev_buf);
    ctx.free_device(weights_dev);
}

void test_moe_prefill_gate_up_and_down(Context& ctx) {
    run_moe_prefill_case(ctx, 2, 4, false);
    run_moe_prefill_case(ctx, 2, 10, true);
    run_moe_prefill_case(ctx, 26, 10, false);
    run_moe_prefill_case(ctx, 26, 10, true);
    run_moe_prefill_case(ctx, 64, 10, true);
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

} // namespace

int main() {
    aila::env::enable_l0_copy_engine();
    try {
        Context ctx;
        std::cout << "Starting Qwen4Exp Kernel Tests on device: "
                  << ctx.queue().get_device().get_info<sycl::info::device::name>() << "\n";

        test_grouped_rms_norm(ctx);
        test_hc_mix_reduce(ctx);
        test_hc_inject_add(ctx);
        test_hc_decode_fused(ctx, true);
        test_hc_decode_fused(ctx, false);
        test_ple_gate_value(ctx);
        test_ple_dilated_conv_silu_add(ctx);
        test_ple_dilated_conv_silu_add_residual(ctx);
        test_router_softmax_topk(ctx);
        test_moe_decode_gate_up_and_down(ctx);
        test_moe_prefill_gate_up_and_down(ctx);
        test_gated_delta_net_sigmoid(ctx);
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
