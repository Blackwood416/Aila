#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// ============================================================
// SYCL Kernel: Embedding Lookup
// ============================================================

void embedding_lookup(Context& ctx, Tensor& table,
                       const int* token_ids_device, int seq_len,
                       Tensor& output, int hidden_size) {
    bf16* table_ptr = static_cast<bf16*>(table.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    ctx.queue().parallel_for(sycl::range<2>(seq_len, hidden_size),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int h = idx[1];
            int token_id = token_ids_device[s];
            out_ptr[s * hidden_size + h] = table_ptr[token_id * hidden_size + h];
        });
}

// ============================================================
// SYCL Kernel: RoPE (Rotary Position Embedding)
// ============================================================

void apply_rope_partial(Context& ctx, Tensor& q, Tensor& k,
                        int seq_len, int start_pos,
                        int num_heads_q, int num_kv_heads, int head_dim,
                        int rotary_dim, float theta, bool interleaved,
                        const int* pos_t, const int* pos_h, const int* pos_w,
                        int prompt_pos_len, int text_pos_delta,
                        int mrope_section_t, int mrope_section_h, int mrope_section_w,
                        bool proportional) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    int q_dim = num_heads_q * head_dim;
    int k_dim = num_kv_heads * head_dim;
    int rot = std::max(2, std::min(head_dim, rotary_dim));
    if (rot & 1) --rot;
    const int active_half_dim = rot / 2;
    // Proportional RoPE (Gemma 4 full attention) keeps a full-head cos/sin
    // vector with zero frequencies in the non-rotary portion. rotate_half()
    // therefore pairs the active prefix with the second half of the complete
    // head, not with the second half of the rotary prefix.
    const int pair_offset = proportional ? head_dim / 2 : active_half_dim;
    const int frequency_dim = proportional ? head_dim : rot;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, num_heads_q * active_half_dim),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int flat = idx[1];
            int h = flat / active_half_dim;
            int d = flat % active_half_dim;
            int global_pos = start_pos + s;
            int pos_scalar = global_pos;
            if (global_pos >= prompt_pos_len) {
                pos_scalar = global_pos + text_pos_delta;
            }

            float position_value = static_cast<float>(pos_scalar);
            if (interleaved && pos_t && pos_h && pos_w) {
                int stream = 0;
                if ((d % 3) == 1 && (d / 3) < mrope_section_h) {
                    stream = 1;
                } else if ((d % 3) == 2 && (d / 3) < mrope_section_w) {
                    stream = 2;
                } else {
                    stream = 0;
                }

                int pos_index = global_pos;
                int t_val = (pos_index < prompt_pos_len) ? pos_t[pos_index] : pos_scalar;
                int h_val = (pos_index < prompt_pos_len) ? pos_h[pos_index] : pos_scalar;
                int w_val = (pos_index < prompt_pos_len) ? pos_w[pos_index] : pos_scalar;
                if (stream == 1) position_value = static_cast<float>(h_val);
                else if (stream == 2) position_value = static_cast<float>(w_val);
                else position_value = static_cast<float>(t_val);
            }

            float freq = 1.0f / sycl::pow(
                theta, (2.0f * d) / static_cast<float>(frequency_dim));
            float angle = position_value * freq;
            float cos_val = sycl::cos(angle);
            float sin_val = sycl::sin(angle);

            int q_base = s * q_dim + h * head_dim;
            int q_i0 = d;
            int q_i1 = d + pair_offset;
            float q0 = static_cast<float>(q_ptr[q_base + q_i0]);
            float q1 = static_cast<float>(q_ptr[q_base + q_i1]);
            q_ptr[q_base + q_i0] = bf16(q0 * cos_val - q1 * sin_val);
            q_ptr[q_base + q_i1] = bf16(q1 * cos_val + q0 * sin_val);

            if (h < num_kv_heads) {
                int k_base = s * k_dim + h * head_dim;
                int k_i0 = d;
                int k_i1 = d + pair_offset;
                float k0 = static_cast<float>(k_ptr[k_base + k_i0]);
                float k1 = static_cast<float>(k_ptr[k_base + k_i1]);
                k_ptr[k_base + k_i0] = bf16(k0 * cos_val - k1 * sin_val);
                k_ptr[k_base + k_i1] = bf16(k1 * cos_val + k0 * sin_val);
            }
        });
}

template <bool Quantized>
void decode_prepare_qkv_impl(Context& ctx,
                             Tensor& q, Tensor& k, Tensor& v,
                             Tensor& rope_freq,
                             Tensor& q_norm_weight, Tensor& k_norm_weight,
                             Tensor& k_cache, Tensor& v_cache,
                             int start_pos,
                             int num_heads_q, int num_kv_heads, int head_dim,
                             float eps, float theta) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    float* rope_freq_ptr = static_cast<float*>(rope_freq.data());
    bf16* qn_ptr = static_cast<bf16*>(q_norm_weight.data());
    bf16* kn_ptr = static_cast<bf16*>(k_norm_weight.data());
    void* k_cache_ptr = k_cache.data();
    void* v_cache_ptr = v_cache.data();
    int max_seq_len = static_cast<int>(k_cache.shape(1));
    int half_dim = head_dim / 2;
    int wg_size = 128;

    (void)theta;

    if (head_dim == 128) {
        constexpr int head_dim_const = 128;
        constexpr int half_dim_const = 64;
        constexpr int wg_size_const = 16;

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size_const, wg_size_const),
                [=](sycl::nd_item<1> item) {
                    using vec4 = sycl::vec<bf16, 4>;
                    int head = item.get_group(0);
                    int lid = item.get_local_id(0);
                    int q_base = head * head_dim_const;

                    vec4 q0_vec = reinterpret_cast<const vec4*>(q_ptr + q_base)[lid];
                    vec4 q1_vec = reinterpret_cast<const vec4*>(q_ptr + q_base + half_dim_const)[lid];

                    float sum_sq = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        float x0 = static_cast<float>(q0_vec[k]);
                        float x1 = static_cast<float>(q1_vec[k]);
                        sum_sq += x0 * x0 + x1 * x1;
                    }

                    float group_sum = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());
                    float q_inv_rms = sycl::rsqrt(group_sum / static_cast<float>(head_dim_const) + eps);

                    vec4 qn0_vec = reinterpret_cast<const vec4*>(qn_ptr)[lid];
                    vec4 qn1_vec = reinterpret_cast<const vec4*>(qn_ptr + half_dim_const)[lid];

                    for (int k = 0; k < 4; ++k) {
                        q0_vec[k] = bf16(static_cast<float>(q0_vec[k]) * q_inv_rms * static_cast<float>(qn0_vec[k]));
                        q1_vec[k] = bf16(static_cast<float>(q1_vec[k]) * q_inv_rms * static_cast<float>(qn1_vec[k]));
                    }

                    for (int k = 0; k < 4; ++k) {
                        int d = lid * 4 + k;
                        float angle = static_cast<float>(start_pos) * rope_freq_ptr[d];
                        float c = sycl::native::cos(angle);
                        float s = sycl::native::sin(angle);

                        float v0 = static_cast<float>(q0_vec[k]);
                        float v1 = static_cast<float>(q1_vec[k]);
                        q0_vec[k] = bf16(v0 * c - v1 * s);
                        q1_vec[k] = bf16(v1 * c + v0 * s);
                    }

                    reinterpret_cast<vec4*>(q_ptr + q_base)[lid] = q0_vec;
                    reinterpret_cast<vec4*>(q_ptr + q_base + half_dim_const)[lid] = q1_vec;

                    if (head >= num_kv_heads) {
                        return;
                    }

                    int kv_base = head * head_dim_const;
                    vec4 k0_vec = reinterpret_cast<const vec4*>(k_ptr + kv_base)[lid];
                    vec4 k1_vec = reinterpret_cast<const vec4*>(k_ptr + kv_base + half_dim_const)[lid];

                    sum_sq = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        float x0 = static_cast<float>(k0_vec[k]);
                        float x1 = static_cast<float>(k1_vec[k]);
                        sum_sq += x0 * x0 + x1 * x1;
                    }
                    group_sum = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());
                    float k_inv_rms = sycl::rsqrt(group_sum / static_cast<float>(head_dim_const) + eps);

                    vec4 kn0_vec = reinterpret_cast<const vec4*>(kn_ptr)[lid];
                    vec4 kn1_vec = reinterpret_cast<const vec4*>(kn_ptr + half_dim_const)[lid];

                    for (int k = 0; k < 4; ++k) {
                        k0_vec[k] = bf16(static_cast<float>(k0_vec[k]) * k_inv_rms * static_cast<float>(kn0_vec[k]));
                        k1_vec[k] = bf16(static_cast<float>(k1_vec[k]) * k_inv_rms * static_cast<float>(kn1_vec[k]));
                    }

                    for (int k = 0; k < 4; ++k) {
                        int d = lid * 4 + k;
                        float angle = static_cast<float>(start_pos) * rope_freq_ptr[d];
                        float c = sycl::native::cos(angle);
                        float s = sycl::native::sin(angle);

                        float v0 = static_cast<float>(k0_vec[k]);
                        float v1 = static_cast<float>(k1_vec[k]);
                        k0_vec[k] = bf16(v0 * c - v1 * s);
                        k1_vec[k] = bf16(v1 * c + v0 * s);
                    }

                    reinterpret_cast<vec4*>(k_ptr + kv_base)[lid] = k0_vec;
                    reinterpret_cast<vec4*>(k_ptr + kv_base + half_dim_const)[lid] = k1_vec;

                    vec4 v0_vec = reinterpret_cast<const vec4*>(v_ptr + kv_base)[lid];
                    vec4 v1_vec = reinterpret_cast<const vec4*>(v_ptr + kv_base + half_dim_const)[lid];

                    int cache_base = head * max_seq_len * head_dim_const + start_pos * head_dim_const;
                    if constexpr (Quantized) {
                        uint8_t* k_cache_fp8 = static_cast<uint8_t*>(k_cache_ptr);
                        uint8_t* v_cache_fp8 = static_cast<uint8_t*>(v_cache_ptr);
                        for (int k = 0; k < 4; ++k) {
                            k_cache_fp8[cache_base + lid * 4 + k] = float_to_fp8_e4m3fn(static_cast<float>(k0_vec[k]));
                            v_cache_fp8[cache_base + lid * 4 + k] = float_to_fp8_e4m3fn(static_cast<float>(v0_vec[k]));
                            // S0: the fast path holds the full head in two
                            // vec4 halves; the second half must be encoded too.
                            k_cache_fp8[cache_base + half_dim_const + lid * 4 + k] =
                                float_to_fp8_e4m3fn(static_cast<float>(k1_vec[k]));
                            v_cache_fp8[cache_base + half_dim_const + lid * 4 + k] =
                                float_to_fp8_e4m3fn(static_cast<float>(v1_vec[k]));
                        }
                    } else {
                        bf16* k_cache_bf16 = static_cast<bf16*>(k_cache_ptr);
                        bf16* v_cache_bf16 = static_cast<bf16*>(v_cache_ptr);
                        reinterpret_cast<vec4*>(k_cache_bf16 + cache_base)[lid] = k0_vec;
                        reinterpret_cast<vec4*>(k_cache_bf16 + cache_base + half_dim_const)[lid] = k1_vec;
                        reinterpret_cast<vec4*>(v_cache_bf16 + cache_base)[lid] = v0_vec;
                        reinterpret_cast<vec4*>(v_cache_bf16 + cache_base + half_dim_const)[lid] = v1_vec;
                    }
                });
        });
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int lid = item.get_local_id(0);

                float sum_sq = 0.0f;
                int q_base = head * head_dim;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(q_ptr[q_base + d]);
                    sum_sq += x * x;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float q_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(q_ptr[q_base + d]);
                    float w = static_cast<float>(qn_ptr[d]);
                    q_ptr[q_base + d] = bf16((x / q_rms) * w);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int d = lid; d < half_dim; d += wg_size) {
                    float angle = static_cast<float>(start_pos) * rope_freq_ptr[d];
                    float c = sycl::native::cos(angle);
                    float s = sycl::native::sin(angle);

                    float q0 = static_cast<float>(q_ptr[q_base + d]);
                    float q1 = static_cast<float>(q_ptr[q_base + d + half_dim]);
                    q_ptr[q_base + d] = bf16(q0 * c - q1 * s);
                    q_ptr[q_base + d + half_dim] = bf16(q1 * c + q0 * s);
                }

                if (head >= num_kv_heads) {
                    return;
                }

                item.barrier(sycl::access::fence_space::local_space);
                sum_sq = 0.0f;
                int kv_base = head * head_dim;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(k_ptr[kv_base + d]);
                    sum_sq += x * x;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float k_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(k_ptr[kv_base + d]);
                    float w = static_cast<float>(kn_ptr[d]);
                    k_ptr[kv_base + d] = bf16((x / k_rms) * w);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int d = lid; d < half_dim; d += wg_size) {
                    float angle = static_cast<float>(start_pos) * rope_freq_ptr[d];
                    float c = sycl::native::cos(angle);
                    float s = sycl::native::sin(angle);

                    float k0 = static_cast<float>(k_ptr[kv_base + d]);
                    float k1 = static_cast<float>(k_ptr[kv_base + d + half_dim]);
                    k_ptr[kv_base + d] = bf16(k0 * c - k1 * s);
                    k_ptr[kv_base + d + half_dim] = bf16(k1 * c + k0 * s);
                }
                item.barrier(sycl::access::fence_space::local_space);

                int cache_base = head * max_seq_len * head_dim + start_pos * head_dim;
                for (int d = lid; d < head_dim; d += wg_size) {
                    if constexpr (Quantized) {
                        uint8_t* k_cache_fp8 = static_cast<uint8_t*>(k_cache_ptr);
                        uint8_t* v_cache_fp8 = static_cast<uint8_t*>(v_cache_ptr);
                        k_cache_fp8[cache_base + d] = float_to_fp8_e4m3fn(static_cast<float>(k_ptr[kv_base + d]));
                        v_cache_fp8[cache_base + d] = float_to_fp8_e4m3fn(static_cast<float>(v_ptr[kv_base + d]));
                    } else {
                        bf16* k_cache_bf16 = static_cast<bf16*>(k_cache_ptr);
                        bf16* v_cache_bf16 = static_cast<bf16*>(v_cache_ptr);
                        k_cache_bf16[cache_base + d] = k_ptr[kv_base + d];
                        v_cache_bf16[cache_base + d] = v_ptr[kv_base + d];
                    }
                }
            });
    });
}

void decode_prepare_qkv(Context& ctx,
                        Tensor& q, Tensor& k, Tensor& v,
                        Tensor& rope_freq,
                        Tensor& q_norm_weight, Tensor& k_norm_weight,
                        Tensor& k_cache, Tensor& v_cache,
                        int start_pos,
                        int num_heads_q, int num_kv_heads, int head_dim,
                        float eps, float theta) {
    if (k_cache.dtype() == dnnl::memory::data_type::s8) {
        throw std::invalid_argument(
            "decode_prepare_qkv: int8 KV cache is not supported on this path "
            "(only decode_prepare_qgkv_packed_partial implements INT8 KV)");
    } else if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
        decode_prepare_qkv_impl<true>(ctx, q, k, v, rope_freq, q_norm_weight, k_norm_weight, k_cache, v_cache, start_pos, num_heads_q, num_kv_heads, head_dim, eps, theta);
    } else {
        decode_prepare_qkv_impl<false>(ctx, q, k, v, rope_freq, q_norm_weight, k_norm_weight, k_cache, v_cache, start_pos, num_heads_q, num_kv_heads, head_dim, eps, theta);
    }
}

// ============================================================
// SYCL Kernel: SwiGLU
// output = silu(gate) * up = (gate * sigmoid(gate)) * up
// ============================================================

void swiglu(Context& ctx, Tensor& gate, Tensor& up, Tensor& output, int n) {
    using vec8 = sycl::vec<bf16, 8>;
    vec8* g_ptr = reinterpret_cast<vec8*>(gate.data());
    vec8* u_ptr = reinterpret_cast<vec8*>(up.data());
    vec8* o_ptr = reinterpret_cast<vec8*>(output.data());

    int n8 = n / 8; // Assumes perfectly aligned, which is true for Qwen3 0.6B hidden sizes

    ctx.queue().parallel_for(sycl::range<1>(n8),
        [=](sycl::id<1> i) {
            vec8 g_vec = g_ptr[i];
            vec8 u_vec = u_ptr[i];
            vec8 o_vec;

            for (int k = 0; k < 8; ++k) {
                float g = static_cast<float>(g_vec[k]);
                float u = static_cast<float>(u_vec[k]);
                float silu_g = g / (1.0f + sycl::native::exp(-g));
                o_vec[k] = bf16(silu_g * u);
            }
            o_ptr[i] = o_vec;
        });
}

void fused_gate_up_swiglu(Context& ctx, Tensor& gate_up, Tensor& output, int ff_dim) {
    using vec8 = sycl::vec<bf16, 8>;
    bf16* src_ptr = static_cast<bf16*>(gate_up.data());
    vec8* o_ptr = reinterpret_cast<vec8*>(output.data());

    int n8 = ff_dim / 8;
    constexpr int wg_size = 256;
    int num_groups = (n8 + wg_size - 1) / wg_size;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int i = static_cast<int>(item.get_global_id(0));
                if (i >= n8) return;

                const vec8* gate_ptr = reinterpret_cast<const vec8*>(src_ptr);
                const vec8* up_ptr = reinterpret_cast<const vec8*>(src_ptr + ff_dim);

                vec8 g_vec = gate_ptr[i];
                vec8 u_vec = up_ptr[i];
                vec8 o_vec;
                for (int k = 0; k < 8; ++k) {
                    float g = static_cast<float>(g_vec[k]);
                    float u = static_cast<float>(u_vec[k]);
                    float silu_g = g / (1.0f + sycl::native::exp(-g));
                    o_vec[k] = bf16(silu_g * u);
                }
                o_ptr[i] = o_vec;
            });
    });
}

void snake_beta(Context& ctx, Tensor& input, Tensor& alpha, Tensor& beta,
                Tensor& output, int n, int channels, int seq_len) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());
    const void* alpha_ptr = alpha.data();
    const void* beta_ptr = beta.data();
    bool alpha_is_f32 = (alpha.dtype() == dnnl::memory::data_type::f32);
    bool beta_is_f32 = (beta.dtype() == dnnl::memory::data_type::f32);

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            if (i >= n) return;

            // Channel index c: for row-major layout [seq_len, channels], c = i % channels
            int c = i % channels;

            float a = alpha_is_f32 ? static_cast<const float*>(alpha_ptr)[c] : static_cast<float>(
                static_cast<const bf16*>(alpha_ptr)[c]);
            float b = beta_is_f32 ? static_cast<const float*>(beta_ptr)[c] : static_cast<float>(
                static_cast<const bf16*>(beta_ptr)[c]);

            float exp_a = sycl::exp(a);
            float exp_b = sycl::exp(b);

            float x = static_cast<float>(in_ptr[i]);
            float sin_val = sycl::sin(x * exp_a);
            float res = x + (1.0f / (exp_b + 1e-9f)) * (sin_val * sin_val);

            out_ptr[i] = bf16(res);
        });
    });
}

void vq_lookup_add(Context& ctx, Tensor& codes, Tensor& table,
                   int cb_idx, Tensor& output, int n_frames, int codebook_dim, bool accum) {
    const int32_t* codes_ptr = static_cast<const int32_t*>(codes.data());
    const void* table_raw = table.data();
    bool table_is_f32 = (table.dtype() == dnnl::memory::data_type::f32);
    bf16* out_ptr = static_cast<bf16*>(output.data());

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<2>(n_frames, codebook_dim), [=](sycl::id<2> idx) {
            int f = static_cast<int>(idx[0]);
            int d = static_cast<int>(idx[1]);

            int token_id = codes_ptr[f * 16 + cb_idx];
            if (token_id < 0) token_id = 0; // fallback clamp to prevent OOB
            
            int out_idx = f * codebook_dim + d;
            int tbl_idx = token_id * codebook_dim + d;

            float val = table_is_f32 ? 
                static_cast<const float*>(table_raw)[tbl_idx] : 
                static_cast<float>(static_cast<const bf16*>(table_raw)[tbl_idx]);

            if (accum) {
                out_ptr[out_idx] = bf16(static_cast<float>(out_ptr[out_idx]) + val);
            } else {
                out_ptr[out_idx] = bf16(val);
            }
        });
    });
}

void gelu_tanh_inplace(Context& ctx, Tensor& input, int n) {
    bf16* ptr = static_cast<bf16*>(input.data());
    constexpr float kAlpha = 0.7978845608028654f;
    ctx.queue().parallel_for(sycl::range<1>(n),
        [=](sycl::id<1> idx) {
            int i = idx[0];
            float x = static_cast<float>(ptr[i]);
            float u = kAlpha * (x + 0.044715f * x * x * x);
            ptr[i] = bf16(0.5f * x * (1.0f + sycl::tanh(u)));
        });
}

void gelu_tanh_gate_mul(Context& ctx, Tensor& gate, Tensor& up,
                        Tensor& output, int n) {
    bf16* gate_ptr = static_cast<bf16*>(gate.data());
    bf16* up_ptr = static_cast<bf16*>(up.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());
    constexpr float kAlpha = 0.7978845608028654f;
    ctx.queue().parallel_for(sycl::range<1>(n),
        [=](sycl::id<1> idx) {
            const int i = idx[0];
            const float g = static_cast<float>(gate_ptr[i]);
            const float u = static_cast<float>(up_ptr[i]);
            const float v = kAlpha * (g + 0.044715f * g * g * g);
            out_ptr[i] = bf16(0.5f * g * (1.0f + sycl::tanh(v)) * u);
        });
}

void gelu_tanh_gate_up_mul_fused(Context& ctx, Tensor& gate_up,
                                 Tensor& output, int rows, int inter_dim) {
    const bf16* gu_ptr = static_cast<const bf16*>(gate_up.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());
    constexpr float kAlpha = 0.7978845608028654f;
    if (inter_dim % 8 == 0) {
        const int inter_vec = inter_dim / 8;
        const int total_vec = rows * inter_vec;
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(total_vec)),
            [=](sycl::id<1> idx) {
                const int i = static_cast<int>(idx[0]);
                const int r = i / inter_vec;
                const int c_vec = i % inter_vec;
                const size_t gu_offset = static_cast<size_t>(r) * (2 * inter_dim);
                auto g_v = *reinterpret_cast<const sycl::vec<bf16, 8>*>(gu_ptr + gu_offset + c_vec * 8);
                auto u_v = *reinterpret_cast<const sycl::vec<bf16, 8>*>(gu_ptr + gu_offset + inter_dim + c_vec * 8);
                sycl::vec<bf16, 8> out_v;
                #pragma unroll
                for (int v_idx = 0; v_idx < 8; ++v_idx) {
                    const float g = static_cast<float>(g_v[v_idx]);
                    const float u = static_cast<float>(u_v[v_idx]);
                    const float v = kAlpha * (g + 0.044715f * g * g * g);
                    out_v[v_idx] = bf16(0.5f * g * (1.0f + sycl::tanh(v)) * u);
                }
                *reinterpret_cast<sycl::vec<bf16, 8>*>(out_ptr + static_cast<size_t>(i) * 8) = out_v;
            });
        return;
    }
    const int total = rows * inter_dim;
    ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(total)),
        [=](sycl::id<1> idx) {
            const int i = static_cast<int>(idx[0]);
            const int r = i / inter_dim;
            const int c = i % inter_dim;
            const size_t gu_offset = static_cast<size_t>(r) * (2 * inter_dim);
            const float g = static_cast<float>(gu_ptr[gu_offset + c]);
            const float u = static_cast<float>(gu_ptr[gu_offset + inter_dim + c]);
            const float v = kAlpha * (g + 0.044715f * g * g * g);
            out_ptr[i] = bf16(0.5f * g * (1.0f + sycl::tanh(v)) * u);
        });
}

void scale_inplace(Context& ctx, Tensor& input, float scale, int n) {
    bf16* ptr = static_cast<bf16*>(input.data());
    ctx.queue().parallel_for(sycl::range<1>(n),
        [=](sycl::id<1> idx) {
            const int i = idx[0];
            ptr[i] = bf16(static_cast<float>(ptr[i]) * scale);
        });
}

void logit_softcap_inplace(Context& ctx, Tensor& input, float cap, int n) {
    float* ptr = static_cast<float*>(input.data());
    ctx.queue().parallel_for(sycl::range<1>(n),
        [=](sycl::id<1> idx) {
            const int i = idx[0];
            ptr[i] = sycl::tanh(ptr[i] / cap) * cap;
        });
}

void bias_add_inplace(Context& ctx, Tensor& input, Tensor& bias, int rows, int cols) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* bias_bf16_ptr = nullptr;
    float* bias_f32_ptr = nullptr;
    if (bias.dtype() == dnnl::memory::data_type::f32) {
        bias_f32_ptr = static_cast<float*>(bias.data());
    } else {
        bias_bf16_ptr = static_cast<bf16*>(bias.data());
    }

    ctx.queue().parallel_for(sycl::range<2>(rows, cols),
        [=](sycl::id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            float b = bias_f32_ptr ? bias_f32_ptr[col] : static_cast<float>(bias_bf16_ptr[col]);
            int offset = row * cols + col;
            in_ptr[offset] = bf16(static_cast<float>(in_ptr[offset]) + b);
        });
}

// ============================================================
// SYCL Kernel: Residual Add (a += b, in-place)
// ============================================================

void residual_add(Context& ctx, Tensor& a, Tensor& b, int n) {
    if (n <= 0) return;
    using vec8 = sycl::vec<bf16, 8>;
    vec8* a_ptr = reinterpret_cast<vec8*>(a.data());
    const vec8* b_ptr = reinterpret_cast<const vec8*>(b.data());
    
    int n8 = n / 8;
    constexpr int wg = 256;
    size_t num_wgs = (static_cast<size_t>(n8) + wg - 1) / wg;

    if (num_wgs > 0) {
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    size_t i = item.get_global_id(0);
                    if (i < static_cast<size_t>(n8)) {
                        vec8 va = a_ptr[i];
                        const vec8 vb = b_ptr[i];
                        vec8 vo;
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            float f_a = static_cast<float>(va[k]);
                            float f_b = static_cast<float>(vb[k]);
                            vo[k] = bf16(f_a + f_b);
                        }
                        a_ptr[i] = vo;
                    }
                });
        });
    }

    int tail_start = n8 * 8;
    int tail_count = n - tail_start;
    if (tail_count > 0) {
        bf16* raw_a = static_cast<bf16*>(a.data()) + tail_start;
        const bf16* raw_b = static_cast<const bf16*>(b.data()) + tail_start;
        ctx.queue().parallel_for(sycl::range<1>(tail_count), [=](sycl::id<1> idx) {
            float f_a = static_cast<float>(raw_a[idx[0]]);
            float f_b = static_cast<float>(raw_b[idx[0]]);
            raw_a[idx[0]] = bf16(f_a + f_b);
        });
    }
}

// ============================================================
// SYCL Kernel: Copy Tensor
// ============================================================

void copy_tensor(Context& ctx, Tensor& src, Tensor& dst, int n) {
    using vec8 = sycl::vec<bf16, 8>;
    vec8* s_ptr = reinterpret_cast<vec8*>(src.data());
    vec8* d_ptr = reinterpret_cast<vec8*>(dst.data());
    
    int n8 = n / 8;

    ctx.queue().parallel_for(sycl::range<1>(n8),
        [=](sycl::id<1> i) {
            d_ptr[i] = s_ptr[i];
        });
}

void split_qkv(Context& ctx, Tensor& qkv, Tensor& q, Tensor& k, Tensor& v,
               int seq_len, int q_dim, int kv_dim) {
    bf16* src = static_cast<bf16*>(qkv.data());
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    int total = q_dim + kv_dim + kv_dim;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, total),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int c = idx[1];
            int src_idx = s * total + c;

            if (c < q_dim) {
                q_ptr[s * q_dim + c] = src[src_idx];
            } else if (c < q_dim + kv_dim) {
                int kc = c - q_dim;
                k_ptr[s * kv_dim + kc] = src[src_idx];
            } else {
                int vc = c - q_dim - kv_dim;
                v_ptr[s * kv_dim + vc] = src[src_idx];
            }
        });
}

void split_qkv_bias(Context& ctx, Tensor& qkv, Tensor& bias,
                    Tensor& q, Tensor& k, Tensor& v,
                    int seq_len, int q_dim, int kv_dim) {
    bf16* src = static_cast<bf16*>(qkv.data());
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    bf16* bias_bf16_ptr = nullptr;
    float* bias_f32_ptr = nullptr;
    if (bias.dtype() == dnnl::memory::data_type::f32) {
        bias_f32_ptr = static_cast<float*>(bias.data());
    } else {
        bias_bf16_ptr = static_cast<bf16*>(bias.data());
    }
    int total = q_dim + kv_dim + kv_dim;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, total),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int c = idx[1];
            int src_idx = s * total + c;
            float value = static_cast<float>(src[src_idx]);
            float b = bias_f32_ptr ? bias_f32_ptr[c] : static_cast<float>(bias_bf16_ptr[c]);
            bf16 out = bf16(value + b);

            if (c < q_dim) {
                q_ptr[s * q_dim + c] = out;
            } else if (c < q_dim + kv_dim) {
                int kc = c - q_dim;
                k_ptr[s * kv_dim + kc] = out;
            } else {
                int vc = c - q_dim - kv_dim;
                v_ptr[s * kv_dim + vc] = out;
            }
        });
}

void split_gate_up(Context& ctx, Tensor& gate_up, Tensor& gate, Tensor& up,
                   int seq_len, int ff_dim) {
    bf16* src = static_cast<bf16*>(gate_up.data());
    bf16* gate_ptr = static_cast<bf16*>(gate.data());
    bf16* up_ptr = static_cast<bf16*>(up.data());
    int total = ff_dim * 2;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, total),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int c = idx[1];
            int src_idx = s * total + c;

            if (c < ff_dim) {
                gate_ptr[s * ff_dim + c] = src[src_idx];
            } else {
                int uc = c - ff_dim;
                up_ptr[s * ff_dim + uc] = src[src_idx];
            }
        });
}

void split_linear_all(Context& ctx, Tensor& linear_all, Tensor& qkv,
                      Tensor& z, Tensor& a, Tensor& b,
                      int seq_len, int qkv_dim, int z_dim,
                      int a_dim, int b_dim) {
    bf16* src = static_cast<bf16*>(linear_all.data());
    bf16* qkv_ptr = static_cast<bf16*>(qkv.data());
    bf16* z_ptr = static_cast<bf16*>(z.data());
    bf16* a_ptr = static_cast<bf16*>(a.data());
    bf16* b_ptr = static_cast<bf16*>(b.data());
    int total = qkv_dim + z_dim + a_dim + b_dim;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, total),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int c = idx[1];
            int src_idx = s * total + c;

            if (c < qkv_dim) {
                qkv_ptr[s * qkv_dim + c] = src[src_idx];
            } else if (c < qkv_dim + z_dim) {
                int zc = c - qkv_dim;
                z_ptr[s * z_dim + zc] = src[src_idx];
            } else if (c < qkv_dim + z_dim + a_dim) {
                int ac = c - qkv_dim - z_dim;
                a_ptr[s * a_dim + ac] = src[src_idx];
            } else {
                int bc = c - qkv_dim - z_dim - a_dim;
                b_ptr[s * b_dim + bc] = src[src_idx];
            }
        });
}

void split_q_gate(Context& ctx, Tensor& q_gate, Tensor& q, Tensor& gate,
                  int seq_len, int num_heads, int head_dim) {
    bf16* src = static_cast<bf16*>(q_gate.data());
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* g_ptr = static_cast<bf16*>(gate.data());

    int q_dim = num_heads * head_dim;
    int packed_per_head = head_dim * 2;
    int packed_total = num_heads * packed_per_head;

    ctx.queue().parallel_for(sycl::range<3>(seq_len, num_heads, head_dim),
        [=](sycl::id<3> idx) {
            int s = idx[0];
            int h = idx[1];
            int d = idx[2];

            int src_base = s * packed_total + h * packed_per_head;
            int dst_base = s * q_dim + h * head_dim;

            q_ptr[dst_base + d] = src[src_base + d];
            g_ptr[dst_base + d] = src[src_base + head_dim + d];
        });
}

void split_qgkv(Context& ctx, Tensor& src, Tensor& q, Tensor& gate,
                Tensor& k, Tensor& v, int seq_len,
                int num_heads, int num_kv_heads, int head_dim) {
    if (seq_len <= 0) return;
    const bf16* src_ptr = static_cast<const bf16*>(src.data());
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* g_ptr = static_cast<bf16*>(gate.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());

    const size_t q_dim = static_cast<size_t>(num_heads) * head_dim;
    const size_t kv_dim = static_cast<size_t>(num_kv_heads) * head_dim;
    const size_t q_proj_dim = static_cast<size_t>(num_heads) * 2 * head_dim;
    const size_t total_row = q_proj_dim + 2 * kv_dim;

    if (head_dim % 8 == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        const size_t dim8 = static_cast<size_t>(head_dim / 8);
        const size_t q_blocks = static_cast<size_t>(num_heads) * dim8;
        const size_t kv_blocks = static_cast<size_t>(num_kv_heads) * dim8;
        const size_t total_blocks_per_token = q_blocks + kv_blocks;

        const bool is_pow2 = (dim8 & (dim8 - 1)) == 0;
        int shift_calc = 0;
        if (is_pow2) {
            while ((size_t(1) << shift_calc) < dim8) ++shift_calc;
        }
        const int shift = shift_calc;
        const size_t dim8_mask = is_pow2 ? (dim8 - 1) : 0;

        constexpr size_t wg = 256;
        const size_t num_wgs_per_token = (total_blocks_per_token + wg - 1) / wg;
        const size_t grid_blocks = num_wgs_per_token * wg;

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(seq_len), grid_blocks),
                                  sycl::range<2>(1, wg)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t s = item.get_group(0);
                    const size_t b = item.get_global_id(1);
                    if (b < total_blocks_per_token) {
                        const size_t src_row_base = s * total_row;

                        if (b < q_blocks) {
                            const size_t h = is_pow2 ? (b >> shift) : (b / dim8);
                            const size_t d8 = is_pow2 ? (b & dim8_mask) : (b % dim8);
                            const size_t src_q_off = src_row_base + h * (2 * head_dim) + d8 * 8;
                            const size_t src_g_off = src_q_off + head_dim;
                            const size_t dst_off = s * q_dim + h * head_dim + d8 * 8;

                            const vec8 v_q = *reinterpret_cast<const vec8*>(src_ptr + src_q_off);
                            const vec8 v_g = *reinterpret_cast<const vec8*>(src_ptr + src_g_off);
                            *reinterpret_cast<vec8*>(q_ptr + dst_off) = v_q;
                            *reinterpret_cast<vec8*>(g_ptr + dst_off) = v_g;
                        } else {
                            const size_t kv_b = b - q_blocks;
                            const size_t kv_h = is_pow2 ? (kv_b >> shift) : (kv_b / dim8);
                            const size_t d8 = is_pow2 ? (kv_b & dim8_mask) : (kv_b % dim8);
                            const size_t src_k_off = src_row_base + q_proj_dim + kv_h * head_dim + d8 * 8;
                            const size_t src_v_off = src_row_base + q_proj_dim + kv_dim + kv_h * head_dim + d8 * 8;
                            const size_t dst_off = s * kv_dim + kv_h * head_dim + d8 * 8;

                            const vec8 v_k = *reinterpret_cast<const vec8*>(src_ptr + src_k_off);
                            const vec8 v_v = *reinterpret_cast<const vec8*>(src_ptr + src_v_off);
                            *reinterpret_cast<vec8*>(k_ptr + dst_off) = v_k;
                            *reinterpret_cast<vec8*>(v_ptr + dst_off) = v_v;
                        }
                    }
                });
        });
        return;
    }

    // Fallback for non-multiple-of-8 head_dim
    ctx.queue().parallel_for(sycl::range<2>(static_cast<size_t>(seq_len), q_dim),
        [=](sycl::id<2> idx) {
            size_t s = idx[0];
            size_t i = idx[1];
            size_t h = i / head_dim;
            size_t d = i % head_dim;
            size_t src_q_off = s * total_row + h * 2 * head_dim + d;
            size_t src_g_off = src_q_off + head_dim;
            q_ptr[s * q_dim + i] = src_ptr[src_q_off];
            g_ptr[s * q_dim + i] = src_ptr[src_g_off];
        });

    ctx.queue().parallel_for(sycl::range<2>(static_cast<size_t>(seq_len), kv_dim),
        [=](sycl::id<2> idx) {
            size_t s = idx[0];
            size_t i = idx[1];
            k_ptr[s * kv_dim + i] = src_ptr[s * total_row + q_proj_dim + i];
            v_ptr[s * kv_dim + i] = src_ptr[s * total_row + q_proj_dim + kv_dim + i];
        });
}

void apply_rope(Context& ctx, Tensor& q, Tensor& k,
                int seq_len, int start_pos,
                int num_heads_q, int num_kv_heads, int head_dim,
                float theta) {
    apply_rope_partial(ctx, q, k, seq_len, start_pos,
                       num_heads_q, num_kv_heads, head_dim, head_dim, theta, false);
}

void sigmoid_mul(Context& ctx, Tensor& input, Tensor& gate, Tensor& output, int n) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    bf16* g_ptr = static_cast<bf16*>(gate.data());
    bf16* o_ptr = static_cast<bf16*>(output.data());

    if ((n & 7) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        vec8* in8_ptr = reinterpret_cast<vec8*>(input.data());
        vec8* g8_ptr = reinterpret_cast<vec8*>(gate.data());
        vec8* o8_ptr = reinterpret_cast<vec8*>(output.data());
        int n8 = n / 8;

        ctx.queue().parallel_for(sycl::range<1>(n8), [=](sycl::id<1> idx) {
            vec8 in_vec = in8_ptr[idx];
            vec8 g_vec = g8_ptr[idx];
            vec8 o_vec;
            for (int k = 0; k < 8; ++k) {
                float in_v = static_cast<float>(in_vec[k]);
                float g_v = static_cast<float>(g_vec[k]);
                float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
                o_vec[k] = bf16(in_v * s);
            }
            o8_ptr[idx] = o_vec;
        });
        return;
    }

    ctx.queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        int i = idx[0];
        float in_v = static_cast<float>(in_ptr[i]);
        float g_v = static_cast<float>(g_ptr[i]);
        float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
        o_ptr[i] = bf16(in_v * s);
    });
}

template <bool Quantized>
void decode_prepare_qkv_partial_impl(Context& ctx,
                                     Tensor& q, Tensor& k, Tensor& v,
                                     Tensor& q_norm_weight, Tensor& k_norm_weight,
                                     Tensor& k_cache, Tensor& v_cache,
                                     int start_pos,
                                     int num_heads_q, int num_kv_heads, int head_dim,
                                     float eps, int rotary_dim, float theta,
                                     bool interleaved,
                                     const int* pos_t,
                                     const int* pos_h,
                                     const int* pos_w,
                                     int prompt_pos_len,
                                     int text_pos_delta,
                                     int mrope_section_t,
                                     int mrope_section_h,
                                     int mrope_section_w) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    bf16* qn_ptr = static_cast<bf16*>(q_norm_weight.data());
    bf16* kn_ptr = static_cast<bf16*>(k_norm_weight.data());
    void* k_cache_ptr = k_cache.data();
    void* v_cache_ptr = v_cache.data();
    int max_seq_len = static_cast<int>(k_cache.shape(1));
    int q_dim = num_heads_q * head_dim;
    int k_dim = num_kv_heads * head_dim;
    int rot = std::max(2, std::min(head_dim, rotary_dim));
    if (rot & 1) --rot;
    int half_dim = rot / 2;
    int wg_size = (head_dim > 128 ? 256 : 128);

    if (head_dim == 128 && rotary_dim == 128 && !interleaved) {
        constexpr int head_dim_const = 128;
        constexpr int half_dim_const = 64;
        constexpr int wg_size_const = 16;
        float pos_val = static_cast<float>(start_pos >= prompt_pos_len ? start_pos + text_pos_delta : start_pos);

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size_const, wg_size_const),
                [=](sycl::nd_item<1> item) {
                    using vec4 = sycl::vec<bf16, 4>;
                    int head = item.get_group(0);
                    int lid = item.get_local_id(0);
                    int q_base = head * head_dim_const;

                    vec4 q0_vec = reinterpret_cast<const vec4*>(q_ptr + q_base)[lid];
                    vec4 q1_vec = reinterpret_cast<const vec4*>(q_ptr + q_base + half_dim_const)[lid];

                    float sum_sq = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        float x0 = static_cast<float>(q0_vec[k]);
                        float x1 = static_cast<float>(q1_vec[k]);
                        sum_sq += x0 * x0 + x1 * x1;
                    }

                    float group_sum = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());
                    float q_inv_rms = sycl::rsqrt(group_sum / static_cast<float>(head_dim_const) + eps);

                    vec4 qn0_vec = reinterpret_cast<const vec4*>(qn_ptr)[lid];
                    vec4 qn1_vec = reinterpret_cast<const vec4*>(qn_ptr + half_dim_const)[lid];

                    for (int k = 0; k < 4; ++k) {
                        q0_vec[k] = bf16(static_cast<float>(q0_vec[k]) * q_inv_rms * static_cast<float>(qn0_vec[k]));
                        q1_vec[k] = bf16(static_cast<float>(q1_vec[k]) * q_inv_rms * static_cast<float>(qn1_vec[k]));
                    }

                    for (int k = 0; k < 4; ++k) {
                        int d = lid * 4 + k;
                        float freq = 1.0f / sycl::pow(theta, (2.0f * d) / 128.0f);
                        float angle = pos_val * freq;
                        float c = sycl::native::cos(angle);
                        float s = sycl::native::sin(angle);

                        float v0 = static_cast<float>(q0_vec[k]);
                        float v1 = static_cast<float>(q1_vec[k]);
                        q0_vec[k] = bf16(v0 * c - v1 * s);
                        q1_vec[k] = bf16(v1 * c + v0 * s);
                    }

                    reinterpret_cast<vec4*>(q_ptr + q_base)[lid] = q0_vec;
                    reinterpret_cast<vec4*>(q_ptr + q_base + half_dim_const)[lid] = q1_vec;

                    if (head >= num_kv_heads) {
                        return;
                    }

                    int kv_base = head * head_dim_const;
                    vec4 k0_vec = reinterpret_cast<const vec4*>(k_ptr + kv_base)[lid];
                    vec4 k1_vec = reinterpret_cast<const vec4*>(k_ptr + kv_base + half_dim_const)[lid];

                    sum_sq = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        float x0 = static_cast<float>(k0_vec[k]);
                        float x1 = static_cast<float>(k1_vec[k]);
                        sum_sq += x0 * x0 + x1 * x1;
                    }
                    group_sum = sycl::reduce_over_group(item.get_group(), sum_sq, sycl::plus<float>());
                    float k_inv_rms = sycl::rsqrt(group_sum / static_cast<float>(head_dim_const) + eps);

                    vec4 kn0_vec = reinterpret_cast<const vec4*>(kn_ptr)[lid];
                    vec4 kn1_vec = reinterpret_cast<const vec4*>(kn_ptr + half_dim_const)[lid];

                    for (int k = 0; k < 4; ++k) {
                        k0_vec[k] = bf16(static_cast<float>(k0_vec[k]) * k_inv_rms * static_cast<float>(kn0_vec[k]));
                        k1_vec[k] = bf16(static_cast<float>(k1_vec[k]) * k_inv_rms * static_cast<float>(kn1_vec[k]));
                    }

                    for (int k = 0; k < 4; ++k) {
                        int d = lid * 4 + k;
                        float freq = 1.0f / sycl::pow(theta, (2.0f * d) / 128.0f);
                        float angle = pos_val * freq;
                        float c = sycl::native::cos(angle);
                        float s = sycl::native::sin(angle);

                        float v0 = static_cast<float>(k0_vec[k]);
                        float v1 = static_cast<float>(k1_vec[k]);
                        k0_vec[k] = bf16(v0 * c - v1 * s);
                        k1_vec[k] = bf16(v1 * c + v0 * s);
                    }

                    reinterpret_cast<vec4*>(k_ptr + kv_base)[lid] = k0_vec;
                    reinterpret_cast<vec4*>(k_ptr + kv_base + half_dim_const)[lid] = k1_vec;

                    vec4 v0_vec = reinterpret_cast<const vec4*>(v_ptr + kv_base)[lid];
                    vec4 v1_vec = reinterpret_cast<const vec4*>(v_ptr + kv_base + half_dim_const)[lid];

                    int cache_base = head * max_seq_len * head_dim_const + start_pos * head_dim_const;
                    if constexpr (Quantized == kKvFmtFp8) {
                        uint8_t* k_cache_fp8 = static_cast<uint8_t*>(k_cache_ptr);
                        uint8_t* v_cache_fp8 = static_cast<uint8_t*>(v_cache_ptr);
                        for (int k = 0; k < 4; ++k) {
                            k_cache_fp8[cache_base + lid * 4 + k] = float_to_fp8_e4m3fn(static_cast<float>(k0_vec[k]));
                            v_cache_fp8[cache_base + lid * 4 + k] = float_to_fp8_e4m3fn(static_cast<float>(v0_vec[k]));
                            // S0: encode the second vec4 half as well; the
                            // fast path never wrote it, so heads 64..127 of
                            // every cached token were left uninitialized.
                            k_cache_fp8[cache_base + half_dim_const + lid * 4 + k] =
                                float_to_fp8_e4m3fn(static_cast<float>(k1_vec[k]));
                            v_cache_fp8[cache_base + half_dim_const + lid * 4 + k] =
                                float_to_fp8_e4m3fn(static_cast<float>(v1_vec[k]));
                        }
                    } else {
                        bf16* k_cache_bf16 = static_cast<bf16*>(k_cache_ptr);
                        bf16* v_cache_bf16 = static_cast<bf16*>(v_cache_ptr);
                        reinterpret_cast<vec4*>(k_cache_bf16 + cache_base)[lid] = k0_vec;
                        reinterpret_cast<vec4*>(k_cache_bf16 + cache_base + half_dim_const)[lid] = k1_vec;
                        reinterpret_cast<vec4*>(v_cache_bf16 + cache_base)[lid] = v0_vec;
                        reinterpret_cast<vec4*>(v_cache_bf16 + cache_base + half_dim_const)[lid] = v1_vec;
                    }
                });
        });
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int lid = item.get_local_id(0);
                int q_base = head * head_dim;

                auto rotary_position = [&](int d) {
                    int global_pos = start_pos;
                    int pos_scalar = global_pos;
                    if (global_pos >= prompt_pos_len) {
                        pos_scalar = global_pos + text_pos_delta;
                    }

                    float position_value = static_cast<float>(pos_scalar);
                    if (interleaved && pos_t && pos_h && pos_w) {
                        int stream = 0;
                        if ((d % 3) == 1 && (d / 3) < mrope_section_h) {
                            stream = 1;
                        } else if ((d % 3) == 2 && (d / 3) < mrope_section_w) {
                            stream = 2;
                        } else {
                            stream = 0;
                        }

                        int t_val = (global_pos < prompt_pos_len) ? pos_t[global_pos] : pos_scalar;
                        int h_val = (global_pos < prompt_pos_len) ? pos_h[global_pos] : pos_scalar;
                        int w_val = (global_pos < prompt_pos_len) ? pos_w[global_pos] : pos_scalar;
                        if (stream == 1) position_value = static_cast<float>(h_val);
                        else if (stream == 2) position_value = static_cast<float>(w_val);
                        else position_value = static_cast<float>(t_val);
                    }
                    return position_value;
                };

                float sum_sq = 0.0f;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(q_ptr[q_base + d]);
                    sum_sq += x * x;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float q_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(q_ptr[q_base + d]);
                    float w = static_cast<float>(qn_ptr[d]);
                    q_ptr[q_base + d] = bf16((x / q_rms) * w);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int d = lid; d < half_dim; d += wg_size) {
                    float pos = rotary_position(d);
                    float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(rot));
                    float angle = pos * freq;
                    float c = sycl::native::cos(angle);
                    float s = sycl::native::sin(angle);
                    float q0 = static_cast<float>(q_ptr[q_base + d]);
                    float q1 = static_cast<float>(q_ptr[q_base + d + half_dim]);
                    q_ptr[q_base + d] = bf16(q0 * c - q1 * s);
                    q_ptr[q_base + d + half_dim] = bf16(q1 * c + q0 * s);
                }

                if (head >= num_kv_heads) {
                    return;
                }

                int k_base = head * head_dim;
                item.barrier(sycl::access::fence_space::local_space);
                sum_sq = 0.0f;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(k_ptr[k_base + d]);
                    sum_sq += x * x;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float k_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(k_ptr[k_base + d]);
                    float w = static_cast<float>(kn_ptr[d]);
                    k_ptr[k_base + d] = bf16((x / k_rms) * w);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int d = lid; d < half_dim; d += wg_size) {
                    float pos = rotary_position(d);
                    float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(rot));
                    float angle = pos * freq;
                    float c = sycl::native::cos(angle);
                    float s = sycl::native::sin(angle);
                    float k0 = static_cast<float>(k_ptr[k_base + d]);
                    float k1 = static_cast<float>(k_ptr[k_base + d + half_dim]);
                    k_ptr[k_base + d] = bf16(k0 * c - k1 * s);
                    k_ptr[k_base + d + half_dim] = bf16(k1 * c + k0 * s);
                }
                item.barrier(sycl::access::fence_space::local_space);

                int cache_base = head * max_seq_len * head_dim + start_pos * head_dim;
                for (int d = lid; d < head_dim; d += wg_size) {
                    if constexpr (Quantized == kKvFmtFp8) {
                        uint8_t* k_cache_fp8 = static_cast<uint8_t*>(k_cache_ptr);
                        uint8_t* v_cache_fp8 = static_cast<uint8_t*>(v_cache_ptr);
                        k_cache_fp8[cache_base + d] = float_to_fp8_e4m3fn(static_cast<float>(k_ptr[k_base + d]));
                        v_cache_fp8[cache_base + d] = float_to_fp8_e4m3fn(static_cast<float>(v_ptr[k_base + d]));
                    } else {
                        bf16* k_cache_bf16 = static_cast<bf16*>(k_cache_ptr);
                        bf16* v_cache_bf16 = static_cast<bf16*>(v_cache_ptr);
                        k_cache_bf16[cache_base + d] = k_ptr[k_base + d];
                        v_cache_bf16[cache_base + d] = v_ptr[k_base + d];
                    }
                }
            });
    });
}

void decode_prepare_qkv_partial(Context& ctx,
                                Tensor& q, Tensor& k, Tensor& v,
                                Tensor& q_norm_weight, Tensor& k_norm_weight,
                                Tensor& k_cache, Tensor& v_cache,
                                int start_pos,
                                int num_heads_q, int num_kv_heads, int head_dim,
                                float eps, int rotary_dim, float theta,
                                bool interleaved,
                                const int* pos_t,
                                const int* pos_h,
                                const int* pos_w,
                                int prompt_pos_len,
                                int text_pos_delta,
                                int mrope_section_t,
                                int mrope_section_h,
                                int mrope_section_w) {
    if (k_cache.dtype() == dnnl::memory::data_type::s8) {
        throw std::invalid_argument(
            "decode_prepare_qkv_partial: int8 KV cache is not supported on this path "
            "(only decode_prepare_qgkv_packed_partial implements INT8 KV)");
    } else if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
        decode_prepare_qkv_partial_impl<true>(ctx, q, k, v, q_norm_weight, k_norm_weight, k_cache, v_cache, start_pos, num_heads_q, num_kv_heads, head_dim, eps, rotary_dim, theta, interleaved, pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w);
    } else {
        decode_prepare_qkv_partial_impl<false>(ctx, q, k, v, q_norm_weight, k_norm_weight, k_cache, v_cache, start_pos, num_heads_q, num_kv_heads, head_dim, eps, rotary_dim, theta, interleaved, pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w);
    }
}

template <int Quantized, bool Streamed = false>
void decode_prepare_qgkv_packed_partial_impl(Context& ctx,
                                             Tensor& q_gate_packed, Tensor& k, Tensor& v,
                                             Tensor& q_out, Tensor& gate_out,
                                             Tensor& q_norm_weight, Tensor& k_norm_weight,
                                             Tensor& k_cache, Tensor& v_cache,
                                             int start_pos,
                                             GraphControl control,
                                             int layer,
                                             int num_heads_q, int num_kv_heads, int head_dim,
                                             float eps, int rotary_dim, float theta,
                                             bool interleaved,
                                             const int* pos_t,
                                             const int* pos_h,
                                             const int* pos_w,
                                             int prompt_pos_len,
                                             int text_pos_delta,
                                             int mrope_section_t,
                                             int mrope_section_h,
                                             int mrope_section_w,
                                             float* k_scale_ptr = nullptr,
                                             float* v_scale_ptr = nullptr,
                                             int rows = 1,
                                             int qkv_row_stride = 0,
                                             const QsaKvStreamWriteTarget* stream_target = nullptr) {
    bf16* packed_base_ptr = static_cast<bf16*>(q_gate_packed.data());
    bf16* q_base_ptr = static_cast<bf16*>(q_out.data());
    bf16* gate_base_ptr = static_cast<bf16*>(gate_out.data());
    bf16* k_base_ptr = static_cast<bf16*>(k.data());
    bf16* v_base_ptr = static_cast<bf16*>(v.data());
    bf16* qn_ptr = static_cast<bf16*>(q_norm_weight.data());
    bf16* kn_ptr = static_cast<bf16*>(k_norm_weight.data());
    void* k_cache_ptr = k_cache.data();
    void* v_cache_ptr = v_cache.data();
    if constexpr (Quantized == kKvFmtInt8) {
        if (k_scale_ptr == nullptr || v_scale_ptr == nullptr) {
            throw std::invalid_argument(
                "decode_prepare_qgkv_packed_partial: int8 KV cache requires k_scale/v_scale");
        }
    }
    QsaKvHostSegments host_segs{};
    const int32_t* stream_pt = nullptr;
    int host_capacity = 0;
    if constexpr (Streamed) {
        if (!stream_target || !stream_target->page_table || stream_target->host_capacity <= 0) {
            throw std::invalid_argument(
                "decode_prepare_qgkv_packed_partial: invalid KV stream write target");
        }
        stream_pt = stream_target->page_table;
        host_capacity = stream_target->host_capacity;
        if (stream_target->segments.num_segments > 0) {
            host_segs = stream_target->segments;
            for (int s = 0; s < host_segs.num_segments; ++s) {
                if (!host_segs.k[s] || !host_segs.v[s]) {
                    throw std::invalid_argument(
                        "decode_prepare_qgkv_packed_partial: unallocated host KV segment");
                }
                if constexpr (Quantized == kKvFmtInt8) {
                    if (!host_segs.k_scale[s] || !host_segs.v_scale[s]) {
                        throw std::invalid_argument(
                            "decode_prepare_qgkv_packed_partial: int8 KV stream requires host scales");
                    }
                }
            }
        } else {
            if (!stream_target->host_k || !stream_target->host_v) {
                throw std::invalid_argument(
                    "decode_prepare_qgkv_packed_partial: invalid KV stream write target");
            }
            if constexpr (Quantized == kKvFmtInt8) {
                if (!stream_target->host_k_scale || !stream_target->host_v_scale) {
                    throw std::invalid_argument(
                        "decode_prepare_qgkv_packed_partial: int8 KV stream requires host scales");
                }
            }
            host_segs.host_capacity = host_capacity;
            host_segs.segment_tokens = host_capacity;
            host_segs.last_segment_tokens = host_capacity;
            host_segs.num_segments = 1;
            host_segs.k[0] = stream_target->host_k;
            host_segs.v[0] = stream_target->host_v;
            host_segs.k_scale[0] = stream_target->host_k_scale;
            host_segs.v_scale[0] = stream_target->host_v_scale;
        }
    }
    int max_seq_len = static_cast<int>(k_cache.shape(1));
    int logical_limit = Streamed ? host_capacity : max_seq_len;
    int packed_per_head = head_dim * 2;
    int q_row_dim = num_heads_q * head_dim;
    int k_dim = num_kv_heads * head_dim;
    int effective_qkv_stride = qkv_row_stride > 0
        ? qkv_row_stride : (num_heads_q * packed_per_head + 2 * k_dim);
    int effective_rows = std::max(1, rows);
    int rot = std::max(2, std::min(head_dim, rotary_dim));
    if (rot & 1) --rot;
    int half_dim = rot / 2;
    int wg_size = (head_dim > 128 ? 256 : 128);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(effective_rows) * num_heads_q * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int group = static_cast<int>(item.get_group(0));
                int row = group / num_heads_q;
                int head = group % num_heads_q;
                int lid = item.get_local_id(0);
                if (control && graph_control_aborted(control)) return;
                const int active_pos =
                    (control ? control[graph_control::start_pos] : start_pos) + row;
                if (active_pos < 0 || active_pos >= logical_limit) {
                    if (control) graph_control_mark_abort(control);
                    return;
                }
                if (control) {
                    int visible_len = 0;
                    int compressed_blocks = 0;
                    if (!graph_control_pending_blocks(
                            control, max_seq_len / 4, visible_len,
                            compressed_blocks, layer))
                        return;
                }
                bf16* packed_ptr = packed_base_ptr + static_cast<size_t>(row) * effective_qkv_stride;
                bf16* k_ptr = k_base_ptr + static_cast<size_t>(row) * effective_qkv_stride;
                bf16* v_ptr = v_base_ptr + static_cast<size_t>(row) * effective_qkv_stride;
                bf16* q_ptr = q_base_ptr + static_cast<size_t>(row) * q_row_dim;
                bf16* gate_ptr = gate_base_ptr + static_cast<size_t>(row) * q_row_dim;
                int packed_base = head * packed_per_head;
                int q_base = head * head_dim;

                auto rotary_position = [&](int d) {
                    int global_pos = active_pos;
                    int pos_scalar = global_pos;
                    if (global_pos >= prompt_pos_len) {
                        pos_scalar = global_pos + text_pos_delta;
                    }

                    float position_value = static_cast<float>(pos_scalar);
                    if (interleaved && pos_t && pos_h && pos_w) {
                        int stream = 0;
                        if ((d % 3) == 1 && (d / 3) < mrope_section_h) {
                            stream = 1;
                        } else if ((d % 3) == 2 && (d / 3) < mrope_section_w) {
                            stream = 2;
                        } else {
                            stream = 0;
                        }

                        int t_val = (global_pos < prompt_pos_len) ? pos_t[global_pos] : pos_scalar;
                        int h_val = (global_pos < prompt_pos_len) ? pos_h[global_pos] : pos_scalar;
                        int w_val = (global_pos < prompt_pos_len) ? pos_w[global_pos] : pos_scalar;
                        if (stream == 1) position_value = static_cast<float>(h_val);
                        else if (stream == 2) position_value = static_cast<float>(w_val);
                        else position_value = static_cast<float>(t_val);
                    }
                    return position_value;
                };

                float sum_sq = 0.0f;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float qv = static_cast<float>(packed_ptr[packed_base + d]);
                    q_ptr[q_base + d] = bf16(qv);
                    gate_ptr[q_base + d] = packed_ptr[packed_base + head_dim + d];
                    sum_sq += qv * qv;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float q_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(q_ptr[q_base + d]);
                    float w = static_cast<float>(qn_ptr[d]);
                    q_ptr[q_base + d] = bf16((x / q_rms) * w);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int d = lid; d < half_dim; d += wg_size) {
                    float pos = rotary_position(d);
                    float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(rot));
                    float angle = pos * freq;
                    float c = sycl::native::cos(angle);
                    float s = sycl::native::sin(angle);
                    float q0 = static_cast<float>(q_ptr[q_base + d]);
                    float q1 = static_cast<float>(q_ptr[q_base + d + half_dim]);
                    q_ptr[q_base + d] = bf16(q0 * c - q1 * s);
                    q_ptr[q_base + d + half_dim] = bf16(q1 * c + q0 * s);
                }

                if (head >= num_kv_heads) {
                    return;
                }

                int k_base = head * head_dim;
                item.barrier(sycl::access::fence_space::local_space);
                sum_sq = 0.0f;
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(k_ptr[k_base + d]);
                    sum_sq += x * x;
                }
                local_sum[lid] = sum_sq;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float k_rms = sycl::sqrt(local_sum[0] / static_cast<float>(head_dim) + eps);
                for (int d = lid; d < head_dim; d += wg_size) {
                    float x = static_cast<float>(k_ptr[k_base + d]);
                    float w = static_cast<float>(kn_ptr[d]);
                    k_ptr[k_base + d] = bf16((x / k_rms) * w);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int d = lid; d < half_dim; d += wg_size) {
                    float pos = rotary_position(d);
                    float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(rot));
                    float angle = pos * freq;
                    float c = sycl::native::cos(angle);
                    float s = sycl::native::sin(angle);
                    float k0 = static_cast<float>(k_ptr[k_base + d]);
                    float k1 = static_cast<float>(k_ptr[k_base + d + half_dim]);
                    k_ptr[k_base + d] = bf16(k0 * c - k1 * s);
                    k_ptr[k_base + d + half_dim] = bf16(k1 * c + k0 * s);
                }
                item.barrier(sycl::access::fence_space::local_space);

                int phys_slot = -1;
                int phys_pos = active_pos;
                if constexpr (Streamed) {
                    phys_slot = stream_pt[active_pos >> 2];
                    phys_pos = (phys_slot << 2) | (active_pos & 3);
                }
                const size_t cache_base =
                    static_cast<size_t>(head) * max_seq_len * head_dim + static_cast<size_t>(phys_pos) * head_dim;
                int host_seg = 0;
                int host_local_pos = active_pos;
                int host_seg_stride = host_capacity;
                void* host_k_ptr = nullptr;
                void* host_v_ptr = nullptr;
                float* host_k_scale_ptr = nullptr;
                float* host_v_scale_ptr = nullptr;
                size_t host_base = 0u;
                if constexpr (Streamed) {
                    host_seg = active_pos / host_segs.segment_tokens;
                    host_local_pos = active_pos - host_seg * host_segs.segment_tokens;
                    host_seg_stride = (host_seg + 1 < host_segs.num_segments)
                        ? host_segs.segment_tokens : host_segs.last_segment_tokens;
                    host_k_ptr = host_segs.k[host_seg];
                    host_v_ptr = host_segs.v[host_seg];
                    host_k_scale_ptr = host_segs.k_scale[host_seg];
                    host_v_scale_ptr = host_segs.v_scale[host_seg];
                    host_base = static_cast<size_t>(head) * host_seg_stride * head_dim +
                                static_cast<size_t>(host_local_pos) * head_dim;
                }
                const bool write_slot = (!Streamed) || (phys_slot >= 0);
                if constexpr (Quantized == kKvFmtInt8) {
                    // Opt-in INT8/INT8 KV: K is quantized after RoPE, V as-is.
                    // One subgroup-wide amax per (head, token) yields the
                    // per-token scale in the same pass (no extra global pass).
                    float kamax = 0.0f;
                    for (int d = lid; d < head_dim; d += wg_size) {
                        kamax = sycl::fmax(kamax, sycl::fabs(static_cast<float>(k_ptr[k_base + d])));
                    }
                    local_sum[lid] = kamax;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) local_sum[lid] = sycl::fmax(local_sum[lid], local_sum[lid + stride]);
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    const float k_scale = local_sum[0] / 127.0f;
                    if (lid == 0) {
                        if constexpr (Streamed) {
                            host_k_scale_ptr[static_cast<size_t>(head) * host_seg_stride + host_local_pos] = k_scale;
                        }
                        if (write_slot) {
                            k_scale_ptr[static_cast<size_t>(head) * max_seq_len + phys_pos] = k_scale;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float vamax = 0.0f;
                    for (int d = lid; d < head_dim; d += wg_size) {
                        vamax = sycl::fmax(vamax, sycl::fabs(static_cast<float>(v_ptr[k_base + d])));
                    }
                    local_sum[lid] = vamax;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) local_sum[lid] = sycl::fmax(local_sum[lid], local_sum[lid + stride]);
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    const float v_scale = local_sum[0] / 127.0f;
                    if (lid == 0) {
                        if constexpr (Streamed) {
                            host_v_scale_ptr[static_cast<size_t>(head) * host_seg_stride + host_local_pos] = v_scale;
                        }
                        if (write_slot) {
                            v_scale_ptr[static_cast<size_t>(head) * max_seq_len + phys_pos] = v_scale;
                        }
                    }

                    int8_t* k_cache_i8 = static_cast<int8_t*>(k_cache_ptr);
                    int8_t* v_cache_i8 = static_cast<int8_t*>(v_cache_ptr);
                    int8_t* host_k_i8 = static_cast<int8_t*>(host_k_ptr);
                    int8_t* host_v_i8 = static_cast<int8_t*>(host_v_ptr);
                    for (int d = lid; d < head_dim; d += wg_size) {
                        const int8_t kq = float_to_int8_kv(static_cast<float>(k_ptr[k_base + d]), k_scale);
                        const int8_t vq = float_to_int8_kv(static_cast<float>(v_ptr[k_base + d]), v_scale);
                        if constexpr (Streamed) {
                            host_k_i8[host_base + d] = kq;
                            host_v_i8[host_base + d] = vq;
                        }
                        if (write_slot) {
                            k_cache_i8[cache_base + d] = kq;
                            v_cache_i8[cache_base + d] = vq;
                        }
                    }
                } else {
                    for (int d = lid; d < head_dim; d += wg_size) {
                        if constexpr (Quantized == kKvFmtFp8) {
                            const uint8_t kf = float_to_fp8_e4m3fn(static_cast<float>(k_ptr[k_base + d]));
                            const uint8_t vf = float_to_fp8_e4m3fn(static_cast<float>(v_ptr[k_base + d]));
                            if constexpr (Streamed) {
                                static_cast<uint8_t*>(host_k_ptr)[host_base + d] = kf;
                                static_cast<uint8_t*>(host_v_ptr)[host_base + d] = vf;
                            }
                            if (write_slot) {
                                static_cast<uint8_t*>(k_cache_ptr)[cache_base + d] = kf;
                                static_cast<uint8_t*>(v_cache_ptr)[cache_base + d] = vf;
                            }
                        } else {
                            const bf16 kb = k_ptr[k_base + d];
                            const bf16 vb = v_ptr[k_base + d];
                            if constexpr (Streamed) {
                                static_cast<bf16*>(host_k_ptr)[host_base + d] = kb;
                                static_cast<bf16*>(host_v_ptr)[host_base + d] = vb;
                            }
                            if (write_slot) {
                                static_cast<bf16*>(k_cache_ptr)[cache_base + d] = kb;
                                static_cast<bf16*>(v_cache_ptr)[cache_base + d] = vb;
                            }
                        }
                    }
                }
            });
    });
}

void decode_prepare_qgkv_packed_partial(Context& ctx,
                                        Tensor& q_gate_packed, Tensor& k, Tensor& v,
                                        Tensor& q_out, Tensor& gate_out,
                                        Tensor& q_norm_weight, Tensor& k_norm_weight,
                                        Tensor& k_cache, Tensor& v_cache,
                                        int start_pos,
                                        int num_heads_q, int num_kv_heads, int head_dim,
                                        float eps, int rotary_dim, float theta,
                                        bool interleaved,
                                        const int* pos_t,
                                        const int* pos_h,
                                        const int* pos_w,
                                        int prompt_pos_len,
                                        int text_pos_delta,
                                        int mrope_section_t,
                                        int mrope_section_h,
                                        int mrope_section_w,
                                        Tensor* k_scale,
                                        Tensor* v_scale,
                                        int rows,
                                        int qkv_row_stride,
                                        const QsaKvStreamWriteTarget* stream_target) {
    float* k_scale_ptr = (k_scale && k_scale->valid()) ? k_scale->data_as<float>() : nullptr;
    float* v_scale_ptr = (v_scale && v_scale->valid()) ? v_scale->data_as<float>() : nullptr;
    const bool streamed = (stream_target != nullptr && stream_target->page_table != nullptr);
    auto dispatch = [&](auto tag) {
        if (streamed) {
            decode_prepare_qgkv_packed_partial_impl<decltype(tag)::value, true>(ctx, q_gate_packed, k, v, q_out, gate_out, q_norm_weight, k_norm_weight, k_cache, v_cache, start_pos, nullptr, 0, num_heads_q, num_kv_heads, head_dim, eps, rotary_dim, theta, interleaved, pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w, k_scale_ptr, v_scale_ptr, rows, qkv_row_stride, stream_target);
        } else {
            decode_prepare_qgkv_packed_partial_impl<decltype(tag)::value, false>(ctx, q_gate_packed, k, v, q_out, gate_out, q_norm_weight, k_norm_weight, k_cache, v_cache, start_pos, nullptr, 0, num_heads_q, num_kv_heads, head_dim, eps, rotary_dim, theta, interleaved, pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w, k_scale_ptr, v_scale_ptr, rows, qkv_row_stride, nullptr);
        }
    };
    if (k_cache.dtype() == dnnl::memory::data_type::s8) {
        dispatch(std::integral_constant<int, kKvFmtInt8>{});
    } else if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
        dispatch(std::integral_constant<int, kKvFmtFp8>{});
    } else {
        dispatch(std::integral_constant<int, kKvFmtBf16>{});
    }
}

void decode_prepare_qgkv_packed_partial_graph(
    Context& ctx, Tensor& q_gate_packed, Tensor& k, Tensor& v,
    Tensor& q_out, Tensor& gate_out, Tensor& q_norm_weight,
    Tensor& k_norm_weight, Tensor& k_cache, Tensor& v_cache,
    GraphControl control, int num_heads_q, int num_kv_heads,
    int head_dim, float eps, int rotary_dim, float theta,
    bool interleaved, const int* pos_t, const int* pos_h,
    const int* pos_w, int prompt_pos_len, int text_pos_delta,
    int mrope_section_t, int mrope_section_h, int mrope_section_w,
    Tensor* k_scale, Tensor* v_scale, int layer) {
    if (!control) {
        throw std::invalid_argument(
            "decode_prepare_qgkv_packed_partial_graph requires control");
    }
    if (graph_control_layer_blocks_slot(layer) < 0) {
        throw std::invalid_argument(
            "decode_prepare_qgkv_packed_partial_graph layer out of range");
    }
    float* k_scale_ptr =
        (k_scale && k_scale->valid()) ? k_scale->data_as<float>() : nullptr;
    float* v_scale_ptr =
        (v_scale && v_scale->valid()) ? v_scale->data_as<float>() : nullptr;
    auto dispatch = [&](auto tag) {
        decode_prepare_qgkv_packed_partial_impl<decltype(tag)::value>(
            ctx, q_gate_packed, k, v, q_out, gate_out, q_norm_weight,
            k_norm_weight, k_cache, v_cache, -1, control, layer, num_heads_q,
            num_kv_heads, head_dim, eps, rotary_dim, theta, interleaved,
            pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
            mrope_section_t, mrope_section_h, mrope_section_w,
            k_scale_ptr, v_scale_ptr);
    };
    if (k_cache.dtype() == dnnl::memory::data_type::s8) {
        dispatch(std::integral_constant<int, kKvFmtInt8>{});
    } else if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
        dispatch(std::integral_constant<int, kKvFmtFp8>{});
    } else {
        dispatch(std::integral_constant<int, kKvFmtBf16>{});
    }
}

// ============================================================
// SYCL Kernel: Sinusoidal Position Embedding (audio encoder)
// ============================================================

void sinusoidal_position_embedding(Context& ctx, Tensor& input,
                                   int seq_len, int d_model,
                                   float max_timescale) {
    bf16* in_ptr = static_cast<bf16*>(input.data());
    // Python SinusoidsPositionEmbedding formula:
    // log_inc = log(max_ts) / (C//2 - 1)
    // inv_timescales[j] = exp(-j * log_inc)
    // pe[pos, j] = sin(pos * inv_timescales[j])      for j=0..C//2-1
    // pe[pos, C//2+j] = cos(pos * inv_timescales[j])
    int half = d_model / 2;
    float log_timescale_increment = std::log(max_timescale) / static_cast<float>(half - 1);

    ctx.queue().parallel_for(sycl::range<1>(seq_len * half),
        [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            int pos = i / half;
            int j = i % half;

            float inv_timescale = sycl::exp(-log_timescale_increment * static_cast<float>(j));
            float angle = static_cast<float>(pos) * inv_timescale;
            float sin_val = sycl::sin(angle);
            float cos_val = sycl::cos(angle);

            int base = pos * d_model;
            float old_sin_pos = static_cast<float>(in_ptr[base + j]);
            float old_cos_pos = static_cast<float>(in_ptr[base + half + j]);
            in_ptr[base + j]          = bf16(old_sin_pos + sin_val);
            in_ptr[base + half + j]   = bf16(old_cos_pos + cos_val);
        });
}

} // namespace ops
