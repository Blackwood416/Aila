#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <cmath>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// ============================================================
// SYCL Kernel: RMSNorm
// output[s][h] = input[s][h] / sqrt(mean(input[s][:]^2) + eps) * weight[h]
// ============================================================

void rms_norm(Context& ctx, Tensor& input, Tensor& weight,
               float eps, Tensor& output, int seq_len, int hidden_size) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* w_ptr = static_cast<bf16*>(weight.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    int wg_size = 256; // Intel A770 倾向于较大的 WG 以掩盖延迟

    if (seq_len == 1) {
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int lid = item.get_local_id(0);

                    float local_sum_sq = 0.0f;
                    for (int h = lid; h < hidden_size; h += wg_size) {
                        float val = static_cast<float>(in_ptr[h]);
                        local_sum_sq += val * val;
                    }

                    float group_sum = sycl::reduce_over_group(
                        item.get_group(), local_sum_sq, sycl::plus<float>());
                    float inv_rms = sycl::rsqrt(group_sum / static_cast<float>(hidden_size) + eps);

                    for (int h = lid; h < hidden_size; h += wg_size) {
                        float val = static_cast<float>(in_ptr[h]);
                        float w = static_cast<float>(w_ptr[h]);
                        out_ptr[h] = bf16(val * inv_rms * w);
                    }
                });
        });
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> row_cache(sycl::range<1>(hidden_size), cgh);
        cgh.parallel_for(sycl::nd_range<1>(seq_len * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                int wg = item.get_local_range(0);

                float local_sum_sq = 0.0f;
                for (int h = lid; h < hidden_size; h += wg) {
                    float val = static_cast<float>(in_ptr[row * hidden_size + h]);
                    row_cache[h] = val;
                    local_sum_sq += val * val;
                }
                item.barrier(sycl::access::fence_space::local_space);

                float group_sum = sycl::reduce_over_group(item.get_group(), local_sum_sq, sycl::plus<>());
                float inv_rms = sycl::rsqrt(group_sum / static_cast<float>(hidden_size) + eps);

                for (int h = lid; h < hidden_size; h += wg) {
                    float val = row_cache[h];
                    float w = static_cast<float>(w_ptr[h]);
                    out_ptr[row * hidden_size + h] = bf16(val * inv_rms * w);
                }
            });
    });
}

void rms_norm_unweighted(Context& ctx, Tensor& input, float eps,
                         Tensor& output, int seq_len, int hidden_size) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    constexpr int wg_size = 256;
    if (seq_len == 1) {
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    const int lid = item.get_local_id(0);
                    float local_sum_sq = 0.0f;
                    for (int h = lid; h < hidden_size; h += wg_size) {
                        const float val = static_cast<float>(in_ptr[h]);
                        local_sum_sq += val * val;
                    }
                    const float group_sum = sycl::reduce_over_group(
                        item.get_group(), local_sum_sq, sycl::plus<float>());
                    const float inv_rms =
                        sycl::rsqrt(group_sum / static_cast<float>(hidden_size) + eps);
                    for (int h = lid; h < hidden_size; h += wg_size) {
                        out_ptr[h] = bf16(static_cast<float>(in_ptr[h]) * inv_rms);
                    }
                });
        });
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> row_cache(sycl::range<1>(hidden_size), cgh);
        cgh.parallel_for(sycl::nd_range<1>(seq_len * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const int row = item.get_group(0);
                const int lid = item.get_local_id(0);
                const int wg = item.get_local_range(0);

                float local_sum_sq = 0.0f;
                for (int h = lid; h < hidden_size; h += wg) {
                    const float val = static_cast<float>(in_ptr[row * hidden_size + h]);
                    row_cache[h] = val;
                    local_sum_sq += val * val;
                }
                item.barrier(sycl::access::fence_space::local_space);

                const float group_sum = sycl::reduce_over_group(
                    item.get_group(), local_sum_sq, sycl::plus<float>());
                const float inv_rms =
                    sycl::rsqrt(group_sum / static_cast<float>(hidden_size) + eps);

                for (int h = lid; h < hidden_size; h += wg) {
                    out_ptr[row * hidden_size + h] = bf16(row_cache[h] * inv_rms);
                }
            });
    });
}

void layer_norm(Context& ctx, Tensor& input, Tensor& weight, Tensor& bias,
                float eps, Tensor& output, int rows, int cols) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    const bool has_weight = weight.valid();
    const bool has_bias = bias.valid();
    bf16* w_bf16_ptr = nullptr;
    float* w_f32_ptr = nullptr;
    bf16* b_bf16_ptr = nullptr;
    float* b_f32_ptr = nullptr;
    if (has_weight) {
        if (weight.dtype() == dnnl::memory::data_type::f32) {
            w_f32_ptr = static_cast<float*>(weight.data());
        } else {
            w_bf16_ptr = static_cast<bf16*>(weight.data());
        }
    }
    if (has_bias) {
        if (bias.dtype() == dnnl::memory::data_type::f32) {
            b_f32_ptr = static_cast<float*>(bias.data());
        } else {
            b_bf16_ptr = static_cast<bf16*>(bias.data());
        }
    }

    int wg_size = (cols > 128 ? 256 : 128);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(rows * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                int wg = item.get_local_range(0);

                float local_sum = 0.0f;
                for (int c = lid; c < cols; c += wg) {
                    local_sum += static_cast<float>(in_ptr[row * cols + c]);
                }
                float mean = sycl::reduce_over_group(item.get_group(), local_sum, sycl::plus<float>()) /
                             static_cast<float>(cols);

                float local_var = 0.0f;
                for (int c = lid; c < cols; c += wg) {
                    float diff = static_cast<float>(in_ptr[row * cols + c]) - mean;
                    local_var += diff * diff;
                }
                float var = sycl::reduce_over_group(item.get_group(), local_var, sycl::plus<float>()) /
                            static_cast<float>(cols);
                float inv_std = sycl::rsqrt(var + eps);

                for (int c = lid; c < cols; c += wg) {
                    float x = static_cast<float>(in_ptr[row * cols + c]);
                    float w = 1.0f;
                    float b = 0.0f;
                    if (w_f32_ptr) w = w_f32_ptr[c];
                    else if (w_bf16_ptr) w = static_cast<float>(w_bf16_ptr[c]);
                    if (b_f32_ptr) b = b_f32_ptr[c];
                    else if (b_bf16_ptr) b = static_cast<float>(b_bf16_ptr[c]);
                    out_ptr[row * cols + c] = bf16((x - mean) * inv_std * w + b);
                }
            });
    });
}

void fused_add_rms_norm(Context& ctx, Tensor& input, Tensor& residual,
                         Tensor& weight, float eps, Tensor& output,
                         int seq_len, int hidden_size) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* res_ptr = static_cast<bf16*>(residual.data());
    bf16* w_ptr = static_cast<bf16*>(weight.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    int wg_size = 256;

    if (seq_len == 1) {
        if (hidden_size == 1024) {
            constexpr int hidden = 1024;
            constexpr int wg_1024 = 128;
            ctx.queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(wg_1024, wg_1024),
                    [=](sycl::nd_item<1> item) {
                        using vec8 = sycl::vec<bf16, 8>;
                        int lid = static_cast<int>(item.get_local_id(0));
                        int base = lid * 8;
                        const vec8* in_v8 = reinterpret_cast<const vec8*>(in_ptr + base);
                        const vec8* res_v8 = reinterpret_cast<const vec8*>(res_ptr + base);
                        const vec8* w_v8 = reinterpret_cast<const vec8*>(w_ptr + base);
                        vec8* out_v8 = reinterpret_cast<vec8*>(out_ptr + base);

                        vec8 in_vec = *in_v8;
                        vec8 res_vec = *res_v8;
                        float v[8];
                        for (int k = 0; k < 8; ++k)
                            v[k] = static_cast<float>(in_vec[k]) + static_cast<float>(res_vec[k]);

                        float local_sum_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] +
                                             v[4]*v[4] + v[5]*v[5] + v[6]*v[6] + v[7]*v[7];
                        float group_sum = sycl::reduce_over_group(
                            item.get_group(), local_sum_sq, sycl::plus<float>());
                        float inv_rms = sycl::rsqrt(group_sum / static_cast<float>(hidden) + eps);

                        vec8 w_vec = *w_v8;
                        for (int k = 0; k < 8; ++k) in_ptr[base + k] = bf16(v[k]);
                        vec8 out_vec;
                        for (int k = 0; k < 8; ++k)
                            out_vec[k] = bf16(v[k] * inv_rms * static_cast<float>(w_vec[k]));
                        *out_v8 = out_vec;
                    });
            });
            return;
        }

        if (hidden_size == 2048) {
            constexpr int hidden = 2048;
            constexpr int wg_2048 = 256;
            ctx.queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(wg_2048, wg_2048),
                    [=](sycl::nd_item<1> item) {
                        using vec8 = sycl::vec<bf16, 8>;
                        int lid = static_cast<int>(item.get_local_id(0));
                        int base = lid * 8;
                        const vec8* in_v8 = reinterpret_cast<const vec8*>(in_ptr + base);
                        const vec8* res_v8 = reinterpret_cast<const vec8*>(res_ptr + base);
                        const vec8* w_v8 = reinterpret_cast<const vec8*>(w_ptr + base);
                        vec8* out_v8 = reinterpret_cast<vec8*>(out_ptr + base);

                        vec8 in_vec = *in_v8;
                        vec8 res_vec = *res_v8;
                        float v[8];
                        for (int k = 0; k < 8; ++k)
                            v[k] = static_cast<float>(in_vec[k]) + static_cast<float>(res_vec[k]);

                        float local_sum_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] +
                                             v[4]*v[4] + v[5]*v[5] + v[6]*v[6] + v[7]*v[7];
                        float group_sum = sycl::reduce_over_group(
                            item.get_group(), local_sum_sq, sycl::plus<float>());
                        float inv_rms = sycl::rsqrt(group_sum / static_cast<float>(hidden) + eps);

                        vec8 w_vec = *w_v8;
                        for (int k = 0; k < 8; ++k) in_ptr[base + k] = bf16(v[k]);
                        vec8 out_vec;
                        for (int k = 0; k < 8; ++k)
                            out_vec[k] = bf16(v[k] * inv_rms * static_cast<float>(w_vec[k]));
                        *out_v8 = out_vec;
                    });
            });
            return;
        }

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    using vec8 = sycl::vec<bf16, 8>;
                    int lid = item.get_local_id(0);
                    const int vec_chunks = hidden_size / 8;
                    const int vec_tail = vec_chunks * 8;
                    const vec8* in_vec = reinterpret_cast<const vec8*>(in_ptr);
                    const vec8* res_vec = reinterpret_cast<const vec8*>(res_ptr);
                    const vec8* w_vec = reinterpret_cast<const vec8*>(w_ptr);
                    vec8* out_vec = reinterpret_cast<vec8*>(out_ptr);

                    float local_sum_sq = 0.0f;
                    for (int vc = lid; vc < vec_chunks; vc += wg_size) {
                        vec8 in_v = in_vec[vc];
                        vec8 res_v = res_vec[vc];
                        int base = vc * 8;
                        for (int k = 0; k < 8; ++k) {
                            float val = static_cast<float>(in_v[k]) +
                                        static_cast<float>(res_v[k]);
                            in_ptr[base + k] = bf16(val);
                            local_sum_sq += val * val;
                        }
                    }
                    for (int h = vec_tail + lid; h < hidden_size; h += wg_size) {
                        float val = static_cast<float>(in_ptr[h]) +
                                    static_cast<float>(res_ptr[h]);
                        in_ptr[h] = bf16(val);
                        local_sum_sq += val * val;
                    }

                    float group_sum = sycl::reduce_over_group(
                        item.get_group(), local_sum_sq, sycl::plus<float>());
                    float inv_rms = sycl::rsqrt(group_sum / static_cast<float>(hidden_size) + eps);

                    for (int vc = lid; vc < vec_chunks; vc += wg_size) {
                        vec8 in_v = in_vec[vc]; // already residual-added from Phase 1
                        vec8 w_v = w_vec[vc];
                        vec8 out_v;
                        for (int k = 0; k < 8; ++k) {
                            float val = static_cast<float>(in_v[k]);
                            out_v[k] = bf16(val * inv_rms * static_cast<float>(w_v[k]));
                        }
                        out_vec[vc] = out_v;
                    }
                    for (int h = vec_tail + lid; h < hidden_size; h += wg_size) {
                        float val = static_cast<float>(in_ptr[h]);
                        out_ptr[h] = bf16(val * inv_rms * static_cast<float>(w_ptr[h]));
                    }
                });
        });
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> row_cache(sycl::range<1>(hidden_size), cgh);
        cgh.parallel_for(sycl::nd_range<1>(seq_len * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                int wg = item.get_local_range(0);

                float local_sum_sq = 0.0f;
                for (int h = lid; h < hidden_size; h += wg) {
                    float val = static_cast<float>(in_ptr[row * hidden_size + h]) +
                                static_cast<float>(res_ptr[row * hidden_size + h]);
                    row_cache[h] = val;
                    local_sum_sq += val * val;
                }
                item.barrier(sycl::access::fence_space::local_space);

                float group_sum = sycl::reduce_over_group(item.get_group(), local_sum_sq, sycl::plus<>());
                float inv_rms = sycl::rsqrt(group_sum / static_cast<float>(hidden_size) + eps);

                for (int h = lid; h < hidden_size; h += wg) {
                    float val = row_cache[h];
                    float w = static_cast<float>(w_ptr[h]);
                    in_ptr[row * hidden_size + h] = bf16(val);
                    out_ptr[row * hidden_size + h] = bf16(val * inv_rms * w);
                }
            });
    });
}

// ============================================================
// SYCL Kernel: Per-head RMSNorm (Qwen3 QK-norm)
// ============================================================

void head_rms_norm(Context& ctx, Tensor& x, Tensor& weight,
                    float eps, int seq_len, int num_heads, int head_dim) {
    bf16* x_ptr = static_cast<bf16*>(x.data());
    bf16* w_bf16_ptr = nullptr;
    float* w_f32_ptr = nullptr;
    if (weight.dtype() == dnnl::memory::data_type::f32) {
        w_f32_ptr = static_cast<float*>(weight.data());
    } else {
        w_bf16_ptr = static_cast<bf16*>(weight.data());
    }

    int wg_size = (head_dim > 128 ? 256 : 128);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(seq_len * num_heads * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int group_idx = item.get_group(0);
                int lid = item.get_local_id(0);
                int base = group_idx * head_dim;

                float sum_sq = 0.0f;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float val = static_cast<float>(x_ptr[base + d]);
                    sum_sq += val * val;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) local_sum[lid] += local_sum[lid + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

                for (int d = lid; d < head_dim; d += wg_size) {
                    float val = static_cast<float>(x_ptr[base + d]);
                    float w = w_f32_ptr ? w_f32_ptr[d] : static_cast<float>(w_bf16_ptr[d]);
                    x_ptr[base + d] = bf16(val / rms * w);
                }
            });
    });
}

void fused_head_rms_norm_rope_qkv(
    Context& ctx, Tensor& q, Tensor& k, Tensor& v,
    Tensor& q_norm, Tensor& k_norm, Tensor& v_norm,
    int seq_len, int start_pos, int num_heads, int num_kv_heads, int head_dim,
    int rotary_dim, float theta, float eps, bool proportional) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = v.valid() ? static_cast<bf16*>(v.data()) : nullptr;
    bf16* qn_ptr = static_cast<bf16*>(q_norm.data());
    bf16* kn_ptr = static_cast<bf16*>(k_norm.data());
    bf16* vn_ptr = v_norm.valid() ? static_cast<bf16*>(v_norm.data()) : nullptr;

    const int wg_size = (head_dim > 128 ? 256 : 128);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(seq_len) * num_heads * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const int group_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int s = group_idx / num_heads;
                const int h = group_idx % num_heads;
                const int q_dim = num_heads * head_dim;
                const int kv_dim = num_kv_heads * head_dim;
                const int q_base = s * q_dim + h * head_dim;

                int rot = std::max(2, std::min(head_dim, rotary_dim));
                if (rot & 1) --rot;
                const int active_half_dim = rot / 2;
                const int pair_offset = proportional ? head_dim / 2 : active_half_dim;
                const int frequency_dim = proportional ? head_dim : rot;

                // 1. Q RMSNorm
                float q_sum_sq = 0.0f;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float val = static_cast<float>(q_ptr[q_base + d]);
                    q_sum_sq += val * val;
                }
                local_sum[lid] = q_sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) local_sum[lid] += local_sum[lid + stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const float q_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

                for (int d = lid; d < head_dim; d += wg_size) {
                    float val = static_cast<float>(q_ptr[q_base + d]);
                    float w = static_cast<float>(qn_ptr[d]);
                    q_ptr[q_base + d] = bf16((val / q_rms) * w);
                }
                item.barrier(sycl::access::fence_space::local_space);

                // 2. Q RoPE
                for (int d = lid; d < active_half_dim; d += wg_size) {
                    float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(frequency_dim));
                    float angle = static_cast<float>(start_pos + s) * freq;
                    float cos_val = sycl::cos(angle);
                    float sin_val = sycl::sin(angle);
                    int q_i0 = d;
                    int q_i1 = d + pair_offset;
                    float q0 = static_cast<float>(q_ptr[q_base + q_i0]);
                    float q1 = static_cast<float>(q_ptr[q_base + q_i1]);
                    q_ptr[q_base + q_i0] = bf16(q0 * cos_val - q1 * sin_val);
                    q_ptr[q_base + q_i1] = bf16(q1 * cos_val + q0 * sin_val);
                }

                // 3. K / V (only for h < num_kv_heads)
                if (h < num_kv_heads) {
                    const int k_base = s * kv_dim + h * head_dim;

                    // K RMSNorm
                    item.barrier(sycl::access::fence_space::local_space);
                    float k_sum_sq = 0.0f;
                    for (int d = lid; d < head_dim; d += wg_size) {
                        float val = static_cast<float>(k_ptr[k_base + d]);
                        k_sum_sq += val * val;
                    }
                    local_sum[lid] = k_sum_sq;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) local_sum[lid] += local_sum[lid + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    const float k_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

                    for (int d = lid; d < head_dim; d += wg_size) {
                        float val = static_cast<float>(k_ptr[k_base + d]);
                        float w = static_cast<float>(kn_ptr[d]);
                        k_ptr[k_base + d] = bf16((val / k_rms) * w);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // K RoPE
                    for (int d = lid; d < active_half_dim; d += wg_size) {
                        float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(frequency_dim));
                        float angle = static_cast<float>(start_pos + s) * freq;
                        float cos_val = sycl::cos(angle);
                        float sin_val = sycl::sin(angle);
                        int k_i0 = d;
                        int k_i1 = d + pair_offset;
                        float k0 = static_cast<float>(k_ptr[k_base + k_i0]);
                        float k1 = static_cast<float>(k_ptr[k_base + k_i1]);
                        k_ptr[k_base + k_i0] = bf16(k0 * cos_val - k1 * sin_val);
                        k_ptr[k_base + k_i1] = bf16(k1 * cos_val + k0 * sin_val);
                    }

                    // V RMSNorm (if v_norm is present)
                    if (v_ptr != nullptr && vn_ptr != nullptr) {
                        item.barrier(sycl::access::fence_space::local_space);
                        float v_sum_sq = 0.0f;
                        for (int d = lid; d < head_dim; d += wg_size) {
                            float val = static_cast<float>(v_ptr[k_base + d]);
                            v_sum_sq += val * val;
                        }
                        local_sum[lid] = v_sum_sq;
                        item.barrier(sycl::access::fence_space::local_space);
                        for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                            if (lid < stride) local_sum[lid] += local_sum[lid + stride];
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                        const float v_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

                        for (int d = lid; d < head_dim; d += wg_size) {
                            float val = static_cast<float>(v_ptr[k_base + d]);
                            float w = static_cast<float>(vn_ptr[d]);
                            v_ptr[k_base + d] = bf16((val / v_rms) * w);
                        }
                    }
                }
            });
    });
}

void fused_post_attn_residual_norms(
    Context& ctx, Tensor& hidden, Tensor& attn_o, Tensor& post_attn_ln,
    Tensor& pre_ff_ln, Tensor& pre_ff_ln_2, Tensor& router_scale,
    Tensor& mlp_normed, Tensor& moe_normed, Tensor& router_scaled,
    int seq_len, int hidden_size, float eps) {
    bf16* hidden_ptr = static_cast<bf16*>(hidden.data());
    const bf16* attn_o_ptr = static_cast<const bf16*>(attn_o.data());
    const bf16* post_attn_w_ptr = static_cast<const bf16*>(post_attn_ln.data());
    const bf16* pre_ff_w_ptr = static_cast<const bf16*>(pre_ff_ln.data());
    const bf16* pre_ff2_w_ptr = static_cast<const bf16*>(pre_ff_ln_2.data());
    const bf16* router_scale_ptr = static_cast<const bf16*>(router_scale.data());
    bf16* mlp_normed_ptr = static_cast<bf16*>(mlp_normed.data());
    bf16* moe_normed_ptr = static_cast<bf16*>(moe_normed.data());
    bf16* router_scaled_ptr = static_cast<bf16*>(router_scaled.data());

    constexpr int wg_size = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> row_cache(sycl::range<1>(hidden_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(seq_len) * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const int row = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));

                // 1. Post-attention RMSNorm of attn_o
                float sum_sq_post = 0.0f;
                for (int h = lid; h < hidden_size; h += wg_size) {
                    float val = static_cast<float>(attn_o_ptr[row * hidden_size + h]);
                    sum_sq_post += val * val;
                }
                float grp_sum_post = sycl::reduce_over_group(item.get_group(), sum_sq_post, sycl::plus<float>());
                float inv_rms_post = sycl::rsqrt(grp_sum_post / static_cast<float>(hidden_size) + eps);

                // 2. Add to hidden: h_new = hidden + rms_norm(attn_o)
                float sum_sq_h = 0.0f;
                for (int h = lid; h < hidden_size; h += wg_size) {
                    float ao = static_cast<float>(attn_o_ptr[row * hidden_size + h]);
                    float w_post = static_cast<float>(post_attn_w_ptr[h]);
                    float norm_ao = ao * inv_rms_post * w_post;
                    float orig_h = static_cast<float>(hidden_ptr[row * hidden_size + h]);
                    float h_new = orig_h + norm_ao;
                    hidden_ptr[row * hidden_size + h] = bf16(h_new);
                    row_cache[h] = h_new;
                    sum_sq_h += h_new * h_new;
                }
                item.barrier(sycl::access::fence_space::local_space);
                float grp_sum_h = sycl::reduce_over_group(item.get_group(), sum_sq_h, sycl::plus<float>());
                float inv_rms_h = sycl::rsqrt(grp_sum_h / static_cast<float>(hidden_size) + eps);

                // 3. Emit mlp_normed, moe_normed, and router_scaled
                const float root_size = 1.0f / sycl::sqrt(static_cast<float>(hidden_size));
                for (int h = lid; h < hidden_size; h += wg_size) {
                    float val = row_cache[h];
                    float normalized = val * inv_rms_h;
                    mlp_normed_ptr[row * hidden_size + h] = bf16(normalized * static_cast<float>(pre_ff_w_ptr[h]));
                    moe_normed_ptr[row * hidden_size + h] = bf16(normalized * static_cast<float>(pre_ff2_w_ptr[h]));
                    bf16 scaled = bf16(normalized * static_cast<float>(router_scale_ptr[h]));
                    router_scaled_ptr[row * hidden_size + h] = bf16(static_cast<float>(scaled) * root_size);
                }
            });
    });
}

void fused_post_ff_norm_residual_scale(
    Context& ctx, Tensor& hidden, Tensor& ff1, Tensor& ff2,
    Tensor& post_ff_ln, float layer_scalar, int seq_len, int hidden_size, float eps) {
    bf16* hidden_ptr = static_cast<bf16*>(hidden.data());
    const bf16* ff1_ptr = static_cast<const bf16*>(ff1.data());
    const bf16* ff2_ptr = static_cast<const bf16*>(ff2.data());
    const bf16* post_ff_w_ptr = static_cast<const bf16*>(post_ff_ln.data());

    constexpr int wg_size = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> row_cache(sycl::range<1>(hidden_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(seq_len) * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const int row = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));

                // 1. Sum ff1 + ff2 and compute RMS
                float sum_sq = 0.0f;
                for (int h = lid; h < hidden_size; h += wg_size) {
                    float f1 = static_cast<float>(ff1_ptr[row * hidden_size + h]);
                    float f2 = static_cast<float>(ff2_ptr[row * hidden_size + h]);
                    float sum = f1 + f2;
                    row_cache[h] = sum;
                    sum_sq += sum * sum;
                }
                item.barrier(sycl::access::fence_space::local_space);
                float grp_sum = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());
                float inv_rms = sycl::rsqrt(grp_sum / static_cast<float>(hidden_size) + eps);

                // 2. RMSNorm of ff_sum, add to hidden, multiply by layer_scalar
                for (int h = lid; h < hidden_size; h += wg_size) {
                    float sum = row_cache[h];
                    float w = static_cast<float>(post_ff_w_ptr[h]);
                    float normed = sum * inv_rms * w;
                    float orig_h = static_cast<float>(hidden_ptr[row * hidden_size + h]);
                    float final_h = (orig_h + normed) * layer_scalar;
                    hidden_ptr[row * hidden_size + h] = bf16(final_h);
                }
            });
    });
}

void logits_row_max_inv_sum(
    Context& ctx, const float* logits_ptr, float* row_max_ptr,
    float* row_inv_sum_ptr, int rows, int vocab) {
    constexpr int wg_size = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> shared_max(sycl::range<1>(wg_size), cgh);
        sycl::local_accessor<float, 1> shared_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(rows) * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int row = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const float* src = logits_ptr + static_cast<size_t>(row) * vocab;

                // Phase 1: Local Max
                float local_max = -1.0e30f;
                for (int c = lid; c < vocab; c += wg_size) {
                    float val = static_cast<float>(bf16(src[c]));
                    local_max = sycl::max(local_max, val);
                }
                shared_max[lid] = local_max;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        shared_max[lid] = sycl::max(shared_max[lid], shared_max[lid + stride]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const float row_max = shared_max[0];
                if (lid == 0) {
                    row_max_ptr[row] = row_max;
                }

                // Phase 2: Local Sum
                float local_sum = 0.0f;
                for (int c = lid; c < vocab; c += wg_size) {
                    float val = static_cast<float>(bf16(src[c]));
                    local_sum += sycl::exp(val - row_max);
                }
                shared_sum[lid] = local_sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        shared_sum[lid] += shared_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (lid == 0) {
                    row_inv_sum_ptr[row] = 1.0f / shared_sum[0];
                }
            });
    });
}

void router_topk_device(
    Context& ctx, const bf16* router_scores_ptr, const bf16* per_expert_scale_ptr,
    int* expert_ids_dev, float* expert_weights_dev,
    int tokens, int num_experts, int top_k) {
    // Each workgroup handles one token, workgroup size = num_experts (128).
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scores_slm(sycl::range<1>(128), cgh);
        sycl::local_accessor<int, 1> indices_slm(sycl::range<1>(128), cgh);
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(128), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(tokens) * 128, 128),
            [=](sycl::nd_item<1> item) {
                const int token = static_cast<int>(item.get_group(0));
                const int expert = static_cast<int>(item.get_local_id(0));
                const bf16* row = router_scores_ptr + static_cast<size_t>(token) * num_experts;

                float s = (expert < num_experts) ? static_cast<float>(row[expert]) : -1.0e30f;
                scores_slm[expert] = s;
                indices_slm[expert] = expert;
                red_slm[expert] = s;
                item.barrier(sycl::access::fence_space::local_space);

                // Max score reduction
                for (int stride = 64; stride > 0; stride >>= 1) {
                    if (expert < stride) {
                        red_slm[expert] = sycl::max(red_slm[expert], red_slm[expert + stride]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const float max_score = red_slm[0];

                // Softmax exp & sum
                float exp_val = (expert < num_experts) ? sycl::exp(s - max_score) : 0.0f;
                scores_slm[expert] = exp_val;
                red_slm[expert] = exp_val;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = 64; stride > 0; stride >>= 1) {
                    if (expert < stride) {
                        red_slm[expert] += red_slm[expert + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const float inv_prob_sum = 1.0f / red_slm[0];
                float prob = exp_val * inv_prob_sum;
                scores_slm[expert] = prob;
                item.barrier(sycl::access::fence_space::local_space);

                // Top-k selection in SLM
                for (int k = 0; k < top_k; ++k) {
                    float cand = (expert >= k && expert < num_experts) ? scores_slm[expert] : -1.0e30f;
                    red_slm[expert] = cand;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int stride = 64; stride > 0; stride >>= 1) {
                        if (expert < stride) {
                            red_slm[expert] = sycl::max(red_slm[expert], red_slm[expert + stride]);
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    const float best_val = red_slm[0];
                    item.barrier(sycl::access::fence_space::local_space);

                    if (expert == k) {
                        int best_pos = k;
                        for (int i = k; i < num_experts; ++i) {
                            if (scores_slm[i] == best_val) {
                                best_pos = i;
                                break;
                            }
                        }
                        if (best_pos != k) {
                            float tmp_v = scores_slm[k];
                            int tmp_i = indices_slm[k];
                            scores_slm[k] = scores_slm[best_pos];
                            indices_slm[k] = indices_slm[best_pos];
                            scores_slm[best_pos] = tmp_v;
                            indices_slm[best_pos] = tmp_i;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Selected sum and output weights
                if (expert == 0) {
                    float selected_sum = 0.0f;
                    for (int k = 0; k < top_k; ++k) {
                        selected_sum += scores_slm[k];
                    }
                    const float inv_selected_sum = (selected_sum > 0.0f) ? (1.0f / selected_sum) : 1.0f;
                    for (int k = 0; k < top_k; ++k) {
                        const int sel_expert = indices_slm[k];
                        const float sel_prob = scores_slm[k];
                        const float expert_scale = (per_expert_scale_ptr != nullptr)
                                                       ? static_cast<float>(per_expert_scale_ptr[sel_expert])
                                                       : 1.0f;
                        const float w = sel_prob * inv_selected_sum * expert_scale;
                        expert_ids_dev[token * top_k + k] = sel_expert;
                        expert_weights_dev[token * top_k + k] = w;
                    }
                }
            });
    });
}

} // namespace ops
