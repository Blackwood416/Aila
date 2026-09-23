#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/HyperConnectionOps.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/GatedDeltaNetOps.hpp"

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

void test_grouped_rms_norm(Context& ctx) {
    constexpr int rows = 2;
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    constexpr float eps = 1e-6f;
    const size_t total = rows * groups * group_dim;

    std::mt19937 rng(42);
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
    check_close(gpu_out, ref_out, "grouped_rms_norm (plus_one=true)", 1e-2f);
}

void test_hc_mix_reduce(Context& ctx) {
    constexpr int rows = 2;
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    const size_t total_in = rows * groups * group_dim;
    const size_t total_out = rows * group_dim;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bf16> N_host(total_in);
    std::vector<bf16> mix_host(total_in);
    for (size_t i = 0; i < total_in; ++i) {
        N_host[i] = bf16(dist(rng));
        mix_host[i] = bf16(1.0f / (1.0f + std::exp(-dist(rng)))); // sigmoid-like
    }

    std::vector<float> ref_out(total_out);
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < group_dim; ++d) {
            float sum = 0.0f;
            for (int g = 0; g < groups; ++g) {
                size_t off = static_cast<size_t>(r) * (groups * group_dim) + g * group_dim + d;
                sum += static_cast<float>(mix_host[off]) * static_cast<float>(N_host[off]);
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
                       rows, group_dim, groups);
    ctx.synchronize();

    std::vector<bf16> gpu_out(total_out);
    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), total_out * sizeof(bf16));
    check_close(gpu_out, ref_out, "hc_mix_reduce", 1e-2f);
}

void test_hc_inject_add(Context& ctx) {
    constexpr int rows = 2;
    constexpr int groups = 4;
    constexpr int group_dim = 2560;
    const size_t total_H = rows * groups * group_dim;
    const size_t total_y = rows * group_dim;
    const size_t total_inj = rows * groups;

    std::mt19937 rng(456);
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
    check_close(gpu_out, ref_out, "hc_inject_add", 1e-2f);
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

void test_router_softmax_topk(Context& ctx) {
    constexpr int tokens = 4;
    constexpr int num_experts = 512;
    constexpr int top_k = 10;

    std::mt19937 rng(2026);
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
    check_close_f32(gpu_weights, ref_weights, "router_softmax_topk_device weights", 1e-4f);
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

    ops::moe_decode_down_weighted_accumulate(
        ctx, slots.data(), nullptr, weights_host.data(),
        static_cast<const bf16*>(h_dev.data()), nullptr,
        static_cast<bf16*>(out_dev.data()),
        num_active_experts, intermediate_size, out_features, blocksize);
    ctx.synchronize();

    std::vector<bf16> gpu_out(out_features);
    ctx.memcpy_d2h(gpu_out.data(), out_dev.data(), out_features * sizeof(bf16));
    check_close(gpu_out, ref_out, "moe_decode_down_weighted_accumulate", 2e-2f);
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

} // namespace

int main() {
    try {
        Context ctx;
        std::cout << "Starting Qwen4Exp Kernel Tests on device: "
                  << ctx.queue().get_device().get_info<sycl::info::device::name>() << "\n";

        test_grouped_rms_norm(ctx);
        test_hc_mix_reduce(ctx);
        test_hc_inject_add(ctx);
        test_ple_gate_value(ctx);
        test_ple_dilated_conv_silu_add(ctx);
        test_router_softmax_topk(ctx);
        test_moe_decode_gate_up_and_down(ctx);
        test_gated_delta_net_sigmoid(ctx);

        std::cout << "All Qwen4Exp Kernel Tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}
