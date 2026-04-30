#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>

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
                        int mrope_section_t, int mrope_section_h, int mrope_section_w) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    int q_dim = num_heads_q * head_dim;
    int k_dim = num_kv_heads * head_dim;
    int rot = std::max(2, std::min(head_dim, rotary_dim));
    if (rot & 1) --rot;
    int half_dim = rot / 2;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, num_heads_q * half_dim),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int flat = idx[1];
            int h = flat / half_dim;   
            int d = flat % half_dim;   
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

            float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(rot));
            float angle = position_value * freq;
            float cos_val = sycl::cos(angle);
            float sin_val = sycl::sin(angle);

            int q_base = s * q_dim + h * head_dim;
            int q_i0 = d;
            int q_i1 = d + half_dim;
            float q0 = static_cast<float>(q_ptr[q_base + q_i0]);
            float q1 = static_cast<float>(q_ptr[q_base + q_i1]);
            q_ptr[q_base + q_i0] = bf16(q0 * cos_val - q1 * sin_val);
            q_ptr[q_base + q_i1] = bf16(q1 * cos_val + q0 * sin_val);

            if (h < num_kv_heads) {
                int k_base = s * k_dim + h * head_dim;
                int k_i0 = d;
                int k_i1 = d + half_dim;
                float k0 = static_cast<float>(k_ptr[k_base + k_i0]);
                float k1 = static_cast<float>(k_ptr[k_base + k_i1]);
                k_ptr[k_base + k_i0] = bf16(k0 * cos_val - k1 * sin_val);
                k_ptr[k_base + k_i1] = bf16(k1 * cos_val + k0 * sin_val);
            }
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
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    float* rope_freq_ptr = static_cast<float*>(rope_freq.data());
    bf16* qn_ptr = static_cast<bf16*>(q_norm_weight.data());
    bf16* kn_ptr = static_cast<bf16*>(k_norm_weight.data());
    bf16* k_cache_ptr = static_cast<bf16*>(k_cache.data());
    bf16* v_cache_ptr = static_cast<bf16*>(v_cache.data());
    int max_seq_len = static_cast<int>(k_cache.shape(1));
    int half_dim = head_dim / 2;
    int wg_size = 128;

    (void)theta;

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
                    k_cache_ptr[cache_base + d] = k_ptr[kv_base + d];
                    v_cache_ptr[cache_base + d] = v_ptr[kv_base + d];
                }
            });
    });
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

    if (ff_dim == 3584) {
        constexpr int wg_size = 256;
        constexpr int chunk_count = 3584 / 8;
        vec8* gate_ptr = reinterpret_cast<vec8*>(src_ptr);
        vec8* up_ptr = reinterpret_cast<vec8*>(src_ptr + 3584);

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                    int lid = static_cast<int>(item.get_local_id(0));
                    for (int chunk = lid; chunk < chunk_count; chunk += wg_size) {
                        vec8 g_vec = gate_ptr[chunk];
                        vec8 u_vec = up_ptr[chunk];
                        vec8 o_vec;
                        for (int k = 0; k < 8; ++k) {
                            float g = static_cast<float>(g_vec[k]);
                            float u = static_cast<float>(u_vec[k]);
                            float silu_g = g / (1.0f + sycl::native::exp(-g));
                            o_vec[k] = bf16(silu_g * u);
                        }
                        o_ptr[chunk] = o_vec;
                    }
                });
        });
        return;
    }

    int n8 = ff_dim / 8;

    ctx.queue().parallel_for(sycl::range<1>(n8),
        [=](sycl::id<1> i) {
            int base = static_cast<int>(i) * 8;
            // Gate is in first half, Up is in second half
            vec8 o_vec;
            for (int k = 0; k < 8; ++k) {
                float g = static_cast<float>(src_ptr[base + k]);
                float u = static_cast<float>(src_ptr[ff_dim + base + k]);
                float silu_g = g / (1.0f + sycl::native::exp(-g));
                o_vec[k] = bf16(silu_g * u);
            }
            o_ptr[i] = o_vec;
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
    using vec8 = sycl::vec<bf16, 8>;
    vec8* a_ptr = reinterpret_cast<vec8*>(a.data());
    vec8* b_ptr = reinterpret_cast<vec8*>(b.data());
    
    int n8 = n / 8;

    ctx.queue().parallel_for(sycl::range<1>(n8),
        [=](sycl::id<1> i) {
            vec8 va = a_ptr[i];
            vec8 vb = b_ptr[i];
            vec8 vo;
            for (int k = 0; k < 8; ++k) {
                float f_a = static_cast<float>(va[k]);
                float f_b = static_cast<float>(vb[k]);
                vo[k] = bf16(f_a + f_b);
            }
            a_ptr[i] = vo;
        });
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
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    bf16* qn_ptr = static_cast<bf16*>(q_norm_weight.data());
    bf16* kn_ptr = static_cast<bf16*>(k_norm_weight.data());
    bf16* k_cache_ptr = static_cast<bf16*>(k_cache.data());
    bf16* v_cache_ptr = static_cast<bf16*>(v_cache.data());
    int max_seq_len = static_cast<int>(k_cache.shape(1));
    int q_dim = num_heads_q * head_dim;
    int k_dim = num_kv_heads * head_dim;
    int rot = std::max(2, std::min(head_dim, rotary_dim));
    if (rot & 1) --rot;
    int half_dim = rot / 2;
    int wg_size = (head_dim > 128 ? 256 : 128);

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
                    k_cache_ptr[cache_base + d] = k_ptr[k_base + d];
                    v_cache_ptr[cache_base + d] = v_ptr[k_base + d];
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
                                        int mrope_section_w) {
    bf16* packed_ptr = static_cast<bf16*>(q_gate_packed.data());
    bf16* q_ptr = static_cast<bf16*>(q_out.data());
    bf16* gate_ptr = static_cast<bf16*>(gate_out.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    bf16* qn_ptr = static_cast<bf16*>(q_norm_weight.data());
    bf16* kn_ptr = static_cast<bf16*>(k_norm_weight.data());
    bf16* k_cache_ptr = static_cast<bf16*>(k_cache.data());
    bf16* v_cache_ptr = static_cast<bf16*>(v_cache.data());
    int max_seq_len = static_cast<int>(k_cache.shape(1));
    int packed_per_head = head_dim * 2;
    int k_dim = num_kv_heads * head_dim;
    int rot = std::max(2, std::min(head_dim, rotary_dim));
    if (rot & 1) --rot;
    int half_dim = rot / 2;
    int wg_size = (head_dim > 128 ? 256 : 128);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(num_heads_q * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int lid = item.get_local_id(0);
                int packed_base = head * packed_per_head;
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

                int cache_base = head * max_seq_len * head_dim + start_pos * head_dim;
                for (int d = lid; d < head_dim; d += wg_size) {
                    k_cache_ptr[cache_base + d] = k_ptr[k_base + d];
                    v_cache_ptr[cache_base + d] = v_ptr[k_base + d];
                }
            });
    });
}

} // namespace ops
