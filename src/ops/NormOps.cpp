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

} // namespace ops
