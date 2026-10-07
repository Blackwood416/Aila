#include "GatedDeltaNetOps.hpp"
#include "utils/EnvUtils.hpp"

namespace ops {

bool supports_linear_delta_gqa_fastpath(int q_heads, int kv_heads,
                                        int head_k_dim, int head_v_dim,
                                        int kernel, int conv_rows,
                                        bool allow_grouped_fastpath) {
    return q_heads > 0 &&
           kv_heads > 0 &&
           (kv_heads % q_heads) == 0 &&
           (allow_grouped_fastpath || q_heads == kv_heads) &&
           head_k_dim == head_v_dim &&
           head_k_dim == 128 &&
           kernel == 4 &&
           conv_rows == 3;
}

bool supports_default_grouped_linear_delta_fastpath(int q_heads, int kv_heads,
                                                    int head_k_dim, int head_v_dim,
                                                    int kernel, int conv_rows) {
    return kv_heads > q_heads &&
           supports_linear_delta_gqa_fastpath(q_heads, kv_heads,
                                              head_k_dim, head_v_dim,
                                              kernel, conv_rows,
                                              true);
}

void commit_linear_conv_history(Context& ctx, const bf16* input, int input_stride,
                                float* history, int channels, int history_rows,
                                int initial_head, int token_count) {
    const int rows_to_copy = std::min(token_count, history_rows);
    const int first_token = token_count - rows_to_copy;
    ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(rows_to_copy) * channels),
        [=](sycl::id<1> index) {
            const size_t channel = index[0] % static_cast<size_t>(channels);
            const int token = first_token + static_cast<int>(index[0] / channels);
            const int row = (initial_head + token % history_rows) % history_rows;
            history[static_cast<size_t>(row) * channels + channel] =
                static_cast<float>(input[static_cast<size_t>(token) * input_stride + channel]);
        });
}

int linear_delta_gqa_group_size(int q_heads, int kv_heads) {
    if (q_heads <= 0 || kv_heads <= 0 || (kv_heads % q_heads) != 0) {
        return 0;
    }
    return kv_heads / q_heads;
}

int linear_delta_key_head_for_value_head(int value_head, int q_heads, int kv_heads) {
    int group_size = linear_delta_gqa_group_size(q_heads, kv_heads);
    if (group_size <= 0) {
        return std::clamp(value_head, 0, std::max(0, q_heads - 1));
    }
    return std::min(value_head / group_size, q_heads - 1);
}

bool linear_delta_is_group_leader(int value_head, int q_heads, int kv_heads) {
    int group_size = linear_delta_gqa_group_size(q_heads, kv_heads);
    if (group_size <= 0) {
        return value_head < q_heads;
    }
    return (value_head % group_size) == 0;
}

template <GateActivation Act>
void run_linear_delta_decode_gpu_impl(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* qkv_ptr,
    const bf16* z_ptr,
    const bf16* a_ptr,
    const bf16* b_ptr,
    bf16* out_ptr, bool defer_grouped_history = false) {
    const int qkv_dim = geom.qkv_dim;
    const int z_dim = geom.z_dim;
    const int q_dim = geom.q_dim;
    const int kv_dim = geom.kv_dim;
    const int num_heads = geom.v_heads;
    const int q_heads = geom.q_heads;
    const int head_k_dim = geom.head_k_dim;
    const int head_v_dim = geom.head_v_dim;
    const int kernel = geom.kernel;
    const int conv_rows = geom.conv_rows;
    const int conv_head = (conv_rows > 0) ? (state.conv_head % conv_rows) : 0;
    const float eps = geom.eps;
    const float scale = geom.scale;
    constexpr int wg = 256;

    float* state_ptr = state.state;
    float* conv_state_ptr = state.conv_state;
    const bf16* conv_w_ptr = weights.conv_weight;
    const float* norm_w_ptr = weights.norm_weight;
    const float* a_log_ptr = weights.a_log;
    const bf16* dt_bias_ptr = weights.dt_bias;

    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool default_grouped_fastpath = supports_default_grouped_linear_delta_fastpath(
        q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows);
    bool allow_grouped_fastpath = experimental_gqa_fastpath || default_grouped_fastpath;

    if (supports_linear_delta_gqa_fastpath(q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows,
                                           allow_grouped_fastpath)) {
        constexpr int head_dim = 128;
        constexpr int head_wg = 128;
        const bool grouped_heads = num_heads > q_heads;
        const int row0 = conv_head;
        const int row1 = (conv_head + 1) % conv_rows;
        const int row2 = (conv_head + 2) % conv_rows;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> k_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> out_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> head_params(sycl::range<1>(2), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_heads * head_wg), sycl::range<1>(head_wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    auto softplus_dev = [](float x) {
                        if (x > 20.0f) return x;
                        if (x < -20.0f) return sycl::native::exp(x);
                        return sycl::log1p(sycl::native::exp(x));
                    };
                    auto silu_dev = [](float x) {
                        return x / (1.0f + sycl::native::exp(-x));
                    };
                    auto sigmoid_dev = [](float x) {
                        return 1.0f / (1.0f + sycl::native::exp(-x));
                    };

                    int hv = static_cast<int>(item.get_group(0));
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    bool write_shared_qk = linear_delta_is_group_leader(hv, q_heads, num_heads);
                    int lid = static_cast<int>(item.get_local_id(0));
                    int q_base = hk * head_dim;
                    int k_base = q_dim + hk * head_dim;
                    int v_base = q_dim + q_dim + hv * head_dim;
                    int z_base = hv * head_dim;
                    float* S = state_ptr + static_cast<size_t>(hv) * head_dim * head_dim;
                    float v_reg = 0.0f;
                    float z_reg = 0.0f;

                    if (lid < head_dim) {
                        auto conv_channel = [&](int channel) {
                            const size_t state0 = static_cast<size_t>(row0) * qkv_dim + channel;
                            const size_t state1 = static_cast<size_t>(row1) * qkv_dim + channel;
                            const size_t state2 = static_cast<size_t>(row2) * qkv_dim + channel;
                            const bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                            float v = conv_state_ptr[state0] * static_cast<float>(w[0]) +
                                      conv_state_ptr[state1] * static_cast<float>(w[1]) +
                                      conv_state_ptr[state2] * static_cast<float>(w[2]) +
                                      static_cast<float>(qkv_ptr[channel]) * static_cast<float>(w[3]);
                            return silu_dev(v);
                        };

                        q_local[lid] = conv_channel(q_base + lid);
                        k_local[lid] = conv_channel(k_base + lid);
                        v_reg = conv_channel(v_base + lid);
                        z_reg = static_cast<float>(z_ptr[z_base + lid]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (!grouped_heads && lid < head_dim) {
                        size_t dst = static_cast<size_t>(conv_head) * qkv_dim;
                        if (write_shared_qk) {
                            conv_state_ptr[dst + q_base + lid] = static_cast<float>(qkv_ptr[q_base + lid]);
                            conv_state_ptr[dst + k_base + lid] = static_cast<float>(qkv_ptr[k_base + lid]);
                        }
                        conv_state_ptr[dst + v_base + lid] = static_cast<float>(qkv_ptr[v_base + lid]);
                    }

                    float q_sq = (lid < head_dim) ? (q_local[lid] * q_local[lid]) : 0.0f;
                    float k_sq = (lid < head_dim) ? (k_local[lid] * k_local[lid]) : 0.0f;
                    q_sq = sycl::reduce_over_group(item.get_group(), q_sq, sycl::plus<float>());
                    k_sq = sycl::reduce_over_group(item.get_group(), k_sq, sycl::plus<float>());
                    float q_scale = 1.0f / sycl::sqrt(q_sq + eps);
                    float k_scale = 1.0f / sycl::sqrt(k_sq + eps);

                    if (lid < head_dim) {
                        q_local[lid] *= q_scale;
                        k_local[lid] *= k_scale;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        float beta = 1.0f / (1.0f + sycl::native::exp(-static_cast<float>(b_ptr[hv])));
                        float alpha = static_cast<float>(a_ptr[hv]);
                        float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                    * (-sycl::native::exp(a_log_ptr[hv]));
                        head_params[0] = beta;
                        head_params[1] = sycl::native::exp(g_pre);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float beta = head_params[0];
                    float decay = head_params[1];

                    if (lid < head_dim) {
                        const int dv = lid;
                        float acc = 0.0f;
                        for (int j = 0; j < head_dim; j += 8) {
                            const size_t b = static_cast<size_t>(j) * head_dim + dv;
                            float sv0 = S[b + 0 * head_dim] * decay;
                            float sv1 = S[b + 1 * head_dim] * decay;
                            float sv2 = S[b + 2 * head_dim] * decay;
                            float sv3 = S[b + 3 * head_dim] * decay;
                            float sv4 = S[b + 4 * head_dim] * decay;
                            float sv5 = S[b + 5 * head_dim] * decay;
                            float sv6 = S[b + 6 * head_dim] * decay;
                            float sv7 = S[b + 7 * head_dim] * decay;
                            acc += sv0 * k_local[j + 0] + sv1 * k_local[j + 1] +
                                   sv2 * k_local[j + 2] + sv3 * k_local[j + 3] +
                                   sv4 * k_local[j + 4] + sv5 * k_local[j + 5] +
                                   sv6 * k_local[j + 6] + sv7 * k_local[j + 7];
                        }

                        float dval = (v_reg - acc) * beta;
                        float out = 0.0f;
                        for (int j = 0; j < head_dim; j += 8) {
                            const size_t b = static_cast<size_t>(j) * head_dim + dv;
                            float sv0 = S[b + 0 * head_dim] * decay + k_local[j + 0] * dval;
                            float sv1 = S[b + 1 * head_dim] * decay + k_local[j + 1] * dval;
                            float sv2 = S[b + 2 * head_dim] * decay + k_local[j + 2] * dval;
                            float sv3 = S[b + 3 * head_dim] * decay + k_local[j + 3] * dval;
                            float sv4 = S[b + 4 * head_dim] * decay + k_local[j + 4] * dval;
                            float sv5 = S[b + 5 * head_dim] * decay + k_local[j + 5] * dval;
                            float sv6 = S[b + 6 * head_dim] * decay + k_local[j + 6] * dval;
                            float sv7 = S[b + 7 * head_dim] * decay + k_local[j + 7] * dval;
                            S[b + 0 * head_dim] = sv0;
                            S[b + 1 * head_dim] = sv1;
                            S[b + 2 * head_dim] = sv2;
                            S[b + 3 * head_dim] = sv3;
                            S[b + 4 * head_dim] = sv4;
                            S[b + 5 * head_dim] = sv5;
                            S[b + 6 * head_dim] = sv6;
                            S[b + 7 * head_dim] = sv7;
                            out += sv0 * q_local[j + 0] + sv1 * q_local[j + 1] +
                                   sv2 * q_local[j + 2] + sv3 * q_local[j + 3] +
                                   sv4 * q_local[j + 4] + sv5 * q_local[j + 5] +
                                   sv6 * q_local[j + 6] + sv7 * q_local[j + 7];
                        }
                        out_local[dv] = out * scale;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float out_sq = (lid < head_dim) ? (out_local[lid] * out_local[lid]) : 0.0f;
                    out_sq = sycl::reduce_over_group(item.get_group(), out_sq, sycl::plus<float>());
                    float out_scale = 1.0f / sycl::sqrt(out_sq / static_cast<float>(head_dim) + eps);

                    if (lid < head_dim) {
                        float n = out_local[lid] * out_scale * norm_w_ptr[lid];
                        float gate_val = (Act == GateActivation::Silu) ? silu_dev(z_reg) : sigmoid_dev(z_reg);
                        out_ptr[hv * head_dim + lid] = bf16(n * gate_val);
                    }
                });
        });
        if (grouped_heads && !defer_grouped_history) {
            commit_linear_conv_history(ctx, qkv_ptr, qkv_dim, conv_state_ptr,
                                       qkv_dim, conv_rows, conv_head, 1);
        }
        state.device_state_dirty = true;
        if (conv_rows > 0) {
            state.conv_head = (conv_head + 1) % conv_rows;
        }
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> q_local(sycl::range<1>(q_dim), cgh);
        sycl::local_accessor<float, 1> k_local(sycl::range<1>(q_dim), cgh);
        sycl::local_accessor<float, 1> v_local(sycl::range<1>(kv_dim), cgh);
        sycl::local_accessor<float, 1> z_local(sycl::range<1>(z_dim), cgh);
        sycl::local_accessor<float, 1> out_local(sycl::range<1>(kv_dim), cgh);
        sycl::local_accessor<float, 1> q_scale_local(sycl::range<1>(q_heads), cgh);
        sycl::local_accessor<float, 1> k_scale_local(sycl::range<1>(q_heads), cgh);
        sycl::local_accessor<float, 1> out_scale_local(sycl::range<1>(num_heads), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                auto softplus_dev = [](float x) {
                    if (x > 20.0f) return x;
                    if (x < -20.0f) return sycl::exp(x);
                    return sycl::log1p(sycl::exp(x));
                };
                auto silu_dev = [](float x) {
                    return x / (1.0f + sycl::exp(-x));
                };
                auto sigmoid_dev = [](float x) {
                    return 1.0f / (1.0f + sycl::exp(-x));
                };

                int lid = static_cast<int>(item.get_local_id(0));

                for (int i = lid; i < z_dim; i += wg) {
                    z_local[i] = static_cast<float>(z_ptr[i]);
                }

                for (int c = lid; c < qkv_dim; c += wg) {
                    float v = 0.0f;
                    const bf16* w = conv_w_ptr + static_cast<size_t>(c) * kernel;
                    if (conv_rows > 0 && conv_state_ptr) {
                        for (int j = 0; j < kernel - 1; ++j) {
                            int row = (conv_head + j) % conv_rows;
                            v += conv_state_ptr[static_cast<size_t>(row) * qkv_dim + c]
                                * static_cast<float>(w[j]);
                        }
                    }
                    v += static_cast<float>(qkv_ptr[c]) * static_cast<float>(w[kernel - 1]);
                    float y = silu_dev(v);
                    if (c < q_dim) {
                        q_local[c] = y;
                    } else if (c < q_dim + q_dim) {
                        k_local[c - q_dim] = y;
                    } else {
                        v_local[c - q_dim - q_dim] = y;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (conv_rows > 0 && conv_state_ptr) {
                    size_t dst_base = static_cast<size_t>(conv_head) * qkv_dim;
                    for (int idx = lid; idx < qkv_dim; idx += wg) {
                        conv_state_ptr[dst_base + idx] = static_cast<float>(qkv_ptr[idx]);
                    }
                }

                for (int h = lid; h < q_heads; h += wg) {
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    int base = h * head_k_dim;
                    for (int d = 0; d < head_k_dim; ++d) {
                        float qv = q_local[base + d];
                        float kv = k_local[base + d];
                        q_sum += qv * qv;
                        k_sum += kv * kv;
                    }
                    q_scale_local[h] = 1.0f / sycl::sqrt(q_sum + eps);
                    k_scale_local[h] = 1.0f / sycl::sqrt(k_sum + eps);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < q_dim; idx += wg) {
                    int h = idx / head_k_dim;
                    q_local[idx] *= q_scale_local[h];
                    k_local[idx] *= k_scale_local[h];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < kv_dim; idx += wg) {
                    int hv = idx / head_v_dim;
                    int dv = idx % head_v_dim;
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    float* S = state_ptr + static_cast<size_t>(hv) * head_k_dim * head_v_dim;

                    float beta = 1.0f / (1.0f + sycl::exp(-static_cast<float>(b_ptr[hv])));
                    float alpha = static_cast<float>(a_ptr[hv]);
                    float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                * (-sycl::exp(a_log_ptr[hv]));
                    float decay = sycl::exp(g_pre);

                    float acc = 0.0f;
                    for (int j = 0; j < head_k_dim; ++j) {
                        size_t off = static_cast<size_t>(j) * head_v_dim + dv;
                        float sv = S[off] * decay;
                        acc += sv * k_local[hk * head_k_dim + j];
                    }

                    float d = (v_local[idx] - acc) * beta;
                    float out = 0.0f;
                    for (int j = 0; j < head_k_dim; ++j) {
                        size_t off = static_cast<size_t>(j) * head_v_dim + dv;
                        float sv = S[off] * decay + k_local[hk * head_k_dim + j] * d;
                        S[off] = sv;
                        out += sv * q_local[hk * head_k_dim + j];
                    }
                    out_local[idx] = out * scale;
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int h = lid; h < num_heads; h += wg) {
                    float sum_sq = 0.0f;
                    int base = h * head_v_dim;
                    for (int d = 0; d < head_v_dim; ++d) {
                        float v = out_local[base + d];
                        sum_sq += v * v;
                    }
                    out_scale_local[h] = 1.0f / sycl::sqrt(
                        sum_sq / static_cast<float>(head_v_dim) + eps);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < kv_dim; idx += wg) {
                    int hv = idx / head_v_dim;
                    int dv = idx % head_v_dim;
                    float n = out_local[idx] * out_scale_local[hv] * norm_w_ptr[dv];
                    float zg = z_local[idx];
                    float gate_val = (Act == GateActivation::Silu) ? silu_dev(zg) : sigmoid_dev(zg);
                    out_ptr[idx] = bf16(n * gate_val);
                }
            });
    });

    state.device_state_dirty = true;
    if (conv_rows > 0) {
        state.conv_head = (conv_head + 1) % conv_rows;
    }
}

void run_linear_delta_decode_gpu(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* qkv_ptr,
    const bf16* z_ptr,
    const bf16* a_ptr,
    const bf16* b_ptr,
    bf16* out_ptr, bool defer_grouped_history) {
    if (geom.gate_act == GateActivation::Sigmoid) {
        run_linear_delta_decode_gpu_impl<GateActivation::Sigmoid>(
            ctx, geom, weights, state, qkv_ptr, z_ptr, a_ptr, b_ptr, out_ptr, defer_grouped_history);
    } else {
        run_linear_delta_decode_gpu_impl<GateActivation::Silu>(
            ctx, geom, weights, state, qkv_ptr, z_ptr, a_ptr, b_ptr, out_ptr, defer_grouped_history);
    }
}

template <GateActivation Act>
void run_linear_delta_prefill_gpu_batched_impl(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* fused_ptr,
    bf16* out_ptr,
    int seq_len) {

    const int qkv_dim = geom.qkv_dim;
    const int z_dim = geom.z_dim;
    const int q_dim = geom.q_dim;
    const int kv_dim = geom.kv_dim;
    const int num_heads = geom.v_heads;
    const int q_heads = geom.q_heads;
    const int head_k_dim = geom.head_k_dim;
    const int head_v_dim = geom.head_v_dim;
    const int kernel = geom.kernel;
    const int conv_rows = geom.conv_rows;
    int conv_head = (conv_rows > 0) ? (state.conv_head % conv_rows) : 0;
    const float eps = geom.eps;
    const float scale = geom.scale;
    const int all_dim = geom.all_dim;

    float* state_ptr = state.state;
    float* conv_state_ptr = state.conv_state;
    const bf16* conv_w_ptr = weights.conv_weight;
    const float* norm_w_ptr = weights.norm_weight;
    const float* a_log_ptr = weights.a_log;
    const bf16* dt_bias_ptr = weights.dt_bias;

    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool default_grouped_fastpath = supports_default_grouped_linear_delta_fastpath(
        q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows);
    bool allow_grouped_fastpath = experimental_gqa_fastpath || default_grouped_fastpath;

    if (supports_linear_delta_gqa_fastpath(q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows,
                                           allow_grouped_fastpath)) {
        constexpr int head_dim = 128;
        constexpr int head_wg = 128;
        const bool grouped_heads = num_heads > q_heads;

        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> k_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> out_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> head_params(sycl::range<1>(2), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_heads * head_wg), sycl::range<1>(head_wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    auto softplus_dev = [](float x) {
                        if (x > 20.0f) return x;
                        if (x < -20.0f) return sycl::native::exp(x);
                        return sycl::log1p(sycl::native::exp(x));
                    };
                    auto silu_dev = [](float x) {
                        return x / (1.0f + sycl::native::exp(-x));
                    };
                    auto sigmoid_dev = [](float x) {
                        return 1.0f / (1.0f + sycl::native::exp(-x));
                    };

                    int hv = static_cast<int>(item.get_group(0));
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    bool write_shared_qk = linear_delta_is_group_leader(hv, q_heads, num_heads);
                    int lid = static_cast<int>(item.get_local_id(0));
                    int q_base = hk * head_dim;
                    int k_base = q_dim + hk * head_dim;
                    int v_base = q_dim + q_dim + hv * head_dim;
                    int z_base_offset = qkv_dim + hv * head_dim;
                    int a_offset = qkv_dim + z_dim;
                    int b_offset = qkv_dim + z_dim + num_heads;
                    float* S = state_ptr + static_cast<size_t>(hv) * head_dim * head_dim;

                    int local_conv_head = conv_head;

                    for (int t = 0; t < seq_len; ++t) {
                        const bf16* row = fused_ptr + static_cast<size_t>(t) * all_dim;
                        bf16* out_row = out_ptr + static_cast<size_t>(t) * kv_dim;

                        float v_reg = 0.0f;
                        float z_reg = 0.0f;

                        const int row0 = local_conv_head;
                        const int row1 = (local_conv_head + 1) % conv_rows;
                        const int row2 = (local_conv_head + 2) % conv_rows;

                        if (lid < head_dim) {
                            auto conv_channel = [&](int channel) {
                                if (grouped_heads) {
                                    auto history_value = [&](int offset) {
                                        const int prior_token = t - conv_rows + offset;
                                        if (prior_token >= 0) {
                                            return static_cast<float>(fused_ptr[
                                                static_cast<size_t>(prior_token) * all_dim + channel]);
                                        }
                                        const int history_row =
                                            (conv_head + prior_token + conv_rows) % conv_rows;
                                        return conv_state_ptr[
                                            static_cast<size_t>(history_row) * qkv_dim + channel];
                                    };
                                    const bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                                    const float v = history_value(0) * static_cast<float>(w[0]) +
                                                    history_value(1) * static_cast<float>(w[1]) +
                                                    history_value(2) * static_cast<float>(w[2]) +
                                                    static_cast<float>(row[channel]) * static_cast<float>(w[3]);
                                    return silu_dev(v);
                                }
                                const size_t state0 = static_cast<size_t>(row0) * qkv_dim + channel;
                                const size_t state1 = static_cast<size_t>(row1) * qkv_dim + channel;
                                const size_t state2 = static_cast<size_t>(row2) * qkv_dim + channel;
                                const bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                                float v = conv_state_ptr[state0] * static_cast<float>(w[0]) +
                                          conv_state_ptr[state1] * static_cast<float>(w[1]) +
                                          conv_state_ptr[state2] * static_cast<float>(w[2]) +
                                          static_cast<float>(row[channel]) * static_cast<float>(w[3]);
                                return silu_dev(v);
                            };

                            q_local[lid] = conv_channel(q_base + lid);
                            k_local[lid] = conv_channel(k_base + lid);
                            v_reg = conv_channel(v_base + lid);
                            z_reg = static_cast<float>(row[z_base_offset + lid]);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        if (!grouped_heads && lid < head_dim) {
                            size_t dst = static_cast<size_t>(local_conv_head) * qkv_dim;
                            if (write_shared_qk) {
                                conv_state_ptr[dst + q_base + lid] = static_cast<float>(row[q_base + lid]);
                                conv_state_ptr[dst + k_base + lid] = static_cast<float>(row[k_base + lid]);
                            }
                            conv_state_ptr[dst + v_base + lid] = static_cast<float>(row[v_base + lid]);
                        }

                        float q_sq = (lid < head_dim) ? (q_local[lid] * q_local[lid]) : 0.0f;
                        float k_sq = (lid < head_dim) ? (k_local[lid] * k_local[lid]) : 0.0f;
                        q_sq = sycl::reduce_over_group(item.get_group(), q_sq, sycl::plus<float>());
                        k_sq = sycl::reduce_over_group(item.get_group(), k_sq, sycl::plus<float>());
                        float q_scale = 1.0f / sycl::sqrt(q_sq + eps);
                        float k_scale = 1.0f / sycl::sqrt(k_sq + eps);

                        if (lid < head_dim) {
                            q_local[lid] *= q_scale;
                            k_local[lid] *= k_scale;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        if (lid == 0) {
                            float beta = 1.0f / (1.0f + sycl::native::exp(-static_cast<float>(row[b_offset + hv])));
                            float alpha = static_cast<float>(row[a_offset + hv]);
                            float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                        * (-sycl::native::exp(a_log_ptr[hv]));
                            head_params[0] = beta;
                            head_params[1] = sycl::native::exp(g_pre);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        float beta = head_params[0];
                        float decay = head_params[1];

                        if (lid < head_dim) {
                            const int dv = lid;
                            float acc = 0.0f;
                            for (int j = 0; j < head_dim; j += 4) {
                                const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                                const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                                const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                                const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                                float sv0 = S[off0] * decay;
                                float sv1 = S[off1] * decay;
                                float sv2 = S[off2] * decay;
                                float sv3 = S[off3] * decay;
                                acc += sv0 * k_local[j + 0] +
                                       sv1 * k_local[j + 1] +
                                       sv2 * k_local[j + 2] +
                                       sv3 * k_local[j + 3];
                            }

                            float dval = (v_reg - acc) * beta;
                            float out = 0.0f;
                            for (int j = 0; j < head_dim; j += 4) {
                                const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                                const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                                const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                                const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                                float sv0 = S[off0] * decay + k_local[j + 0] * dval;
                                float sv1 = S[off1] * decay + k_local[j + 1] * dval;
                                float sv2 = S[off2] * decay + k_local[j + 2] * dval;
                                float sv3 = S[off3] * decay + k_local[j + 3] * dval;
                                S[off0] = sv0;
                                S[off1] = sv1;
                                S[off2] = sv2;
                                S[off3] = sv3;
                                out += sv0 * q_local[j + 0] +
                                       sv1 * q_local[j + 1] +
                                       sv2 * q_local[j + 2] +
                                       sv3 * q_local[j + 3];
                            }
                            out_local[dv] = out * scale;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        float out_sq = (lid < head_dim) ? (out_local[lid] * out_local[lid]) : 0.0f;
                        out_sq = sycl::reduce_over_group(item.get_group(), out_sq, sycl::plus<float>());
                        float out_scale = 1.0f / sycl::sqrt(out_sq / static_cast<float>(head_dim) + eps);

                        if (lid < head_dim) {
                            float n = out_local[lid] * out_scale * norm_w_ptr[lid];
                            float gate_val = (Act == GateActivation::Silu) ? silu_dev(z_reg) : sigmoid_dev(z_reg);
                            out_row[hv * head_dim + lid] = bf16(n * gate_val);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        local_conv_head = (local_conv_head + 1) % conv_rows;
                    }
                });
        });

        if (grouped_heads) {
            commit_linear_conv_history(ctx, fused_ptr, all_dim, conv_state_ptr,
                                       qkv_dim, conv_rows, conv_head, seq_len);
        }
        state.device_state_dirty = true;
        state.conv_head = (conv_head + seq_len) % conv_rows;
        return;
    }

    // Fallback: per-token iteration for non-standard shapes
    for (int t = 0; t < seq_len; ++t) {
        const bf16* row_ptr = fused_ptr + static_cast<size_t>(t) * all_dim;
        const bf16* qkv_view = row_ptr;
        const bf16* z_view = row_ptr + qkv_dim;
        const bf16* a_view = row_ptr + qkv_dim + z_dim;
        const bf16* b_view = row_ptr + qkv_dim + z_dim + num_heads;
        bf16* out_view = out_ptr + static_cast<size_t>(t) * kv_dim;
        run_linear_delta_decode_gpu_impl<Act>(ctx, geom, weights, state,
                                             qkv_view, z_view, a_view, b_view, out_view);
    }
}

void run_linear_delta_prefill_gpu_batched(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* fused_ptr,
    bf16* out_ptr,
    int seq_len, bool decode_order) {
    if (decode_order && seq_len >= 1 && seq_len <= 4) {
        // Reuse the actual decode kernel: matching the source-level sum grouping
        // alone did not produce bitwise recurrent trajectories on DG2.
        for (int r = 0; r < seq_len; ++r) {
            const bf16* row = fused_ptr + size_t(r) * geom.all_dim;
            run_linear_delta_decode_gpu(ctx, geom, weights, state, row, row + geom.qkv_dim,
                row + geom.qkv_dim + geom.z_dim, row + geom.qkv_dim + geom.z_dim + geom.v_heads,
                out_ptr + size_t(r) * geom.kv_dim);
        }
        return;
    }
    if (geom.gate_act == GateActivation::Sigmoid) {
        run_linear_delta_prefill_gpu_batched_impl<GateActivation::Sigmoid>(
            ctx, geom, weights, state, fused_ptr, out_ptr, seq_len);
    } else {
        run_linear_delta_prefill_gpu_batched_impl<GateActivation::Silu>(
            ctx, geom, weights, state, fused_ptr, out_ptr, seq_len);
    }
}

} // namespace ops

