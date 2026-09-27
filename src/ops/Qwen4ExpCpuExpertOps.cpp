#include "Qwen4ExpCpuExpertOps.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace ops {
namespace cpu {

static constexpr float kDefaultNf4QuantMap[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

namespace {

inline const float* resolve_quant_map(const float* qmap) {
    return qmap ? qmap : kDefaultNf4QuantMap;
}

inline float load_host_absmax(const float* absmax_f32,
                              const MoeNestedScaleHost& nested,
                              size_t index) {
    if (nested.codes != nullptr) {
        const size_t bs = nested.blocksize > 0 ? static_cast<size_t>(nested.blocksize) : 256;
        const float product = nested.nested_quant_map[nested.codes[index]] *
                              nested.nested_absmax[index / bs];
        return product + nested.offset;
    }
    return absmax_f32 ? absmax_f32[index] : 1.0f;
}

inline void convert_bf16_to_f32(const bf16* src, float* dst, int count) {
#if defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256i f32_bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw16), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(f32_bits));
    }
    for (; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
#else
    for (int i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
#endif
}

#if defined(__AVX2__)
inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    return _mm_cvtss_f32(sum128);
}

inline void unpack_nf4_16bytes_avx2(
    __m128i raw16,
    const float* qmap_scaled,
    __m256& w0, __m256& w1, __m256& w2, __m256& w3) {
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(raw16, mask0f);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f);

    const __m128i nibbles_lo = _mm_unpacklo_epi8(hi, lo);
    const __m128i nibbles_hi = _mm_unpackhi_epi8(hi, lo);

    const __m256i idx0 = _mm256_cvtepu8_epi32(nibbles_lo);
    const __m256i idx1 = _mm256_cvtepu8_epi32(_mm_srli_si128(nibbles_lo, 8));
    const __m256i idx2 = _mm256_cvtepu8_epi32(nibbles_hi);
    const __m256i idx3 = _mm256_cvtepu8_epi32(_mm_srli_si128(nibbles_hi, 8));

    w0 = _mm256_i32gather_ps(qmap_scaled, idx0, 4);
    w1 = _mm256_i32gather_ps(qmap_scaled, idx1, 4);
    w2 = _mm256_i32gather_ps(qmap_scaled, idx2, 4);
    w3 = _mm256_i32gather_ps(qmap_scaled, idx3, 4);
}
#endif

constexpr int kMaxStackTokens = 4;

} // namespace

// =========================================================================
// Scalar Implementations
// =========================================================================

void cpu_expert_gate_up_swiglu_scalar(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map,
    int in_features,
    int intermediate_size,
    int blocksize) {
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || blocksize <= 0) return;

    const float* qmap = resolve_quant_map(quant_map);
    const int num_blocks = in_features / blocksize;
    const int bytes_per_block = blocksize / 2;

    for (int t = 0; t < tokens; ++t) {
        const bf16* x_tok = x + static_cast<size_t>(t) * in_features;
        bf16* h_tok = h + static_cast<size_t>(t) * intermediate_size;

        for (int r = 0; r < intermediate_size; ++r) {
            float gate_acc = 0.0f;
            float up_acc = 0.0f;
            const size_t gate_byte_base = static_cast<size_t>(r) * (in_features / 2);
            const size_t gate_block_base = static_cast<size_t>(r) * num_blocks;
            const size_t up_byte_base = static_cast<size_t>(r + intermediate_size) * (in_features / 2);
            const size_t up_block_base = static_cast<size_t>(r + intermediate_size) * num_blocks;

            for (int b = 0; b < num_blocks; ++b) {
                const float ga = load_host_absmax(slot.gate_up_absmax, slot.gate_up_nested, gate_block_base + b);
                const float ua = load_host_absmax(slot.gate_up_absmax, slot.gate_up_nested, up_block_base + b);
                const size_t block_byte_start = static_cast<size_t>(b) * bytes_per_block;

                for (int byte_i = 0; byte_i < bytes_per_block; ++byte_i) {
                    const size_t byte_idx = block_byte_start + byte_i;
                    const uint8_t gb = slot.gate_up_packed[gate_byte_base + byte_idx];
                    const uint8_t ub = slot.gate_up_packed[up_byte_base + byte_idx];
                    const size_t d = byte_idx * 2;
                    const float x0 = static_cast<float>(x_tok[d]);
                    const float x1 = static_cast<float>(x_tok[d + 1]);

                    gate_acc += x0 * (qmap[gb >> 4] * ga) + x1 * (qmap[gb & 0xF] * ga);
                    up_acc   += x0 * (qmap[ub >> 4] * ua) + x1 * (qmap[ub & 0xF] * ua);
                }
            }

            const float silu_g = gate_acc / (1.0f + std::exp(-gate_acc));
            h_tok[r] = bf16(silu_g * up_acc);
        }
    }
}

void cpu_expert_down_scalar(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map,
    float weight,
    bool accumulate,
    int intermediate_size,
    int out_features,
    int blocksize) {
    if (tokens <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;

    const float* qmap = resolve_quant_map(quant_map);
    const int num_blocks = intermediate_size / blocksize;
    const int bytes_per_block = blocksize / 2;

    for (int t = 0; t < tokens; ++t) {
        const bf16* h_tok = h + static_cast<size_t>(t) * intermediate_size;
        float* out_tok = out + static_cast<size_t>(t) * out_features;

        for (int row = 0; row < out_features; ++row) {
            float down_acc = 0.0f;
            const size_t byte_base = static_cast<size_t>(row) * (intermediate_size / 2);
            const size_t block_base = static_cast<size_t>(row) * num_blocks;

            for (int b = 0; b < num_blocks; ++b) {
                const float da = load_host_absmax(slot.down_absmax, slot.down_nested, block_base + b);
                const size_t block_byte_start = static_cast<size_t>(b) * bytes_per_block;

                for (int byte_i = 0; byte_i < bytes_per_block; ++byte_i) {
                    const size_t byte_idx = block_byte_start + byte_i;
                    const uint8_t db = slot.down_packed[byte_base + byte_idx];
                    const size_t d = byte_idx * 2;
                    const float h0 = static_cast<float>(h_tok[d]);
                    const float h1 = static_cast<float>(h_tok[d + 1]);

                    down_acc += h0 * (qmap[db >> 4] * da) + h1 * (qmap[db & 0xF] * da);
                }
            }

            if (accumulate) {
                out_tok[row] += weight * down_acc;
            } else {
                out_tok[row] = weight * down_acc;
            }
        }
    }
}

void cpu_expert_forward_scalar(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h,
    const float* quant_map,
    float weight,
    bool accumulate,
    int in_features,
    int intermediate_size,
    int out_features,
    int blocksize) {
    alignas(64) bf16 local_h_stack[kMaxStackTokens * kIntermediateDim];
    std::vector<bf16> local_h_heap;
    bf16* effective_h = scratch_h;
    if (!effective_h) {
        if (tokens <= kMaxStackTokens && intermediate_size <= kIntermediateDim) {
            effective_h = local_h_stack;
        } else {
            local_h_heap.resize(static_cast<size_t>(tokens) * intermediate_size);
            effective_h = local_h_heap.data();
        }
    }

    cpu_expert_gate_up_swiglu_scalar(slot, x, effective_h, tokens, quant_map,
                                     in_features, intermediate_size, blocksize);
    cpu_expert_down_scalar(slot, effective_h, out, tokens, quant_map,
                           weight, accumulate, intermediate_size, out_features, blocksize);
}

// =========================================================================
// AVX2 Implementations
// =========================================================================

#if defined(__AVX2__)

static void cpu_expert_gate_up_swiglu_avx2_impl(
    const MoeExpertSlotHost& slot,
    const float* x_f32,
    bf16* h,
    int r_begin,
    int r_end,
    int tokens,
    const float* quant_map,
    int in_features,
    int intermediate_size,
    int blocksize) {
    const float* qmap = resolve_quant_map(quant_map);
    const int num_blocks = in_features / blocksize;
    const int bytes_per_block = blocksize / 2;

    alignas(32) float qmap_scaled_g[16];
    alignas(32) float qmap_scaled_u[16];

    for (int r = r_begin; r < r_end; ++r) {
        const size_t gate_byte_base = static_cast<size_t>(r) * (in_features / 2);
        const size_t gate_block_base = static_cast<size_t>(r) * num_blocks;
        const size_t up_byte_base = static_cast<size_t>(r + intermediate_size) * (in_features / 2);
        const size_t up_block_base = static_cast<size_t>(r + intermediate_size) * num_blocks;

        // Accumulators for tokens (support up to kMaxStackTokens directly in registers)
        __m256 gate_acc[kMaxStackTokens][4];
        __m256 up_acc[kMaxStackTokens][4];
        for (int t = 0; t < tokens && t < kMaxStackTokens; ++t) {
            for (int k = 0; k < 4; ++k) {
                gate_acc[t][k] = _mm256_setzero_ps();
                up_acc[t][k]   = _mm256_setzero_ps();
            }
        }

        for (int b = 0; b < num_blocks; ++b) {
            const float ga = load_host_absmax(slot.gate_up_absmax, slot.gate_up_nested, gate_block_base + b);
            const float ua = load_host_absmax(slot.gate_up_absmax, slot.gate_up_nested, up_block_base + b);

            const __m256 ga_v = _mm256_set1_ps(ga);
            const __m256 ua_v = _mm256_set1_ps(ua);
            _mm256_store_ps(qmap_scaled_g,     _mm256_mul_ps(_mm256_loadu_ps(qmap), ga_v));
            _mm256_store_ps(qmap_scaled_g + 8, _mm256_mul_ps(_mm256_loadu_ps(qmap + 8), ga_v));
            _mm256_store_ps(qmap_scaled_u,     _mm256_mul_ps(_mm256_loadu_ps(qmap), ua_v));
            _mm256_store_ps(qmap_scaled_u + 8, _mm256_mul_ps(_mm256_loadu_ps(qmap + 8), ua_v));

            const uint8_t* g_packed = slot.gate_up_packed + gate_byte_base + b * bytes_per_block;
            const uint8_t* u_packed = slot.gate_up_packed + up_byte_base + b * bytes_per_block;

            // First 16 bytes of block (32 weights = 4 vectors of 8 floats)
            const __m128i g_raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(g_packed));
            const __m128i u_raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u_packed));
            __m256 gw0, gw1, gw2, gw3;
            __m256 uw0, uw1, uw2, uw3;
            unpack_nf4_16bytes_avx2(g_raw0, qmap_scaled_g, gw0, gw1, gw2, gw3);
            unpack_nf4_16bytes_avx2(u_raw0, qmap_scaled_u, uw0, uw1, uw2, uw3);

            // Second 16 bytes of block (weights 32..63 = 4 vectors of 8 floats)
            const __m128i g_raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(g_packed + 16));
            const __m128i u_raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(u_packed + 16));
            __m256 gw4, gw5, gw6, gw7;
            __m256 uw4, uw5, uw6, uw7;
            unpack_nf4_16bytes_avx2(g_raw1, qmap_scaled_g, gw4, gw5, gw6, gw7);
            unpack_nf4_16bytes_avx2(u_raw1, qmap_scaled_u, uw4, uw5, uw6, uw7);

            for (int t = 0; t < tokens && t < kMaxStackTokens; ++t) {
                const float* x_tok_b = x_f32 + static_cast<size_t>(t) * in_features + b * blocksize;
                const __m256 x0 = _mm256_loadu_ps(x_tok_b);
                const __m256 x1 = _mm256_loadu_ps(x_tok_b + 8);
                const __m256 x2 = _mm256_loadu_ps(x_tok_b + 16);
                const __m256 x3 = _mm256_loadu_ps(x_tok_b + 24);
                const __m256 x4 = _mm256_loadu_ps(x_tok_b + 32);
                const __m256 x5 = _mm256_loadu_ps(x_tok_b + 40);
                const __m256 x6 = _mm256_loadu_ps(x_tok_b + 48);
                const __m256 x7 = _mm256_loadu_ps(x_tok_b + 56);

                gate_acc[t][0] = _mm256_fmadd_ps(x0, gw0, gate_acc[t][0]);
                gate_acc[t][1] = _mm256_fmadd_ps(x1, gw1, gate_acc[t][1]);
                gate_acc[t][2] = _mm256_fmadd_ps(x2, gw2, gate_acc[t][2]);
                gate_acc[t][3] = _mm256_fmadd_ps(x3, gw3, gate_acc[t][3]);
                gate_acc[t][0] = _mm256_fmadd_ps(x4, gw4, gate_acc[t][0]);
                gate_acc[t][1] = _mm256_fmadd_ps(x5, gw5, gate_acc[t][1]);
                gate_acc[t][2] = _mm256_fmadd_ps(x6, gw6, gate_acc[t][2]);
                gate_acc[t][3] = _mm256_fmadd_ps(x7, gw7, gate_acc[t][3]);

                up_acc[t][0] = _mm256_fmadd_ps(x0, uw0, up_acc[t][0]);
                up_acc[t][1] = _mm256_fmadd_ps(x1, uw1, up_acc[t][1]);
                up_acc[t][2] = _mm256_fmadd_ps(x2, uw2, up_acc[t][2]);
                up_acc[t][3] = _mm256_fmadd_ps(x3, uw3, up_acc[t][3]);
                up_acc[t][0] = _mm256_fmadd_ps(x4, uw4, up_acc[t][0]);
                up_acc[t][1] = _mm256_fmadd_ps(x5, uw5, up_acc[t][1]);
                up_acc[t][2] = _mm256_fmadd_ps(x6, uw6, up_acc[t][2]);
                up_acc[t][3] = _mm256_fmadd_ps(x7, uw7, up_acc[t][3]);
            }
        }

        for (int t = 0; t < tokens && t < kMaxStackTokens; ++t) {
            __m256 g_sum256 = _mm256_add_ps(_mm256_add_ps(gate_acc[t][0], gate_acc[t][1]),
                                            _mm256_add_ps(gate_acc[t][2], gate_acc[t][3]));
            __m256 u_sum256 = _mm256_add_ps(_mm256_add_ps(up_acc[t][0], up_acc[t][1]),
                                            _mm256_add_ps(up_acc[t][2], up_acc[t][3]));
            const float gate_sum = hsum256_ps(g_sum256);
            const float up_sum   = hsum256_ps(u_sum256);

            const float silu_g = gate_sum / (1.0f + std::exp(-gate_sum));
            h[static_cast<size_t>(t) * intermediate_size + r] = bf16(silu_g * up_sum);
        }
    }
}

static void cpu_expert_down_avx2_impl(
    const MoeExpertSlotHost& slot,
    const float* h_f32,
    float* out,
    int row_begin,
    int row_end,
    int tokens,
    const float* quant_map,
    float weight,
    bool accumulate,
    int intermediate_size,
    int out_features,
    int blocksize) {
    const float* qmap = resolve_quant_map(quant_map);
    const int num_blocks = intermediate_size / blocksize;
    const int bytes_per_block = blocksize / 2;

    alignas(32) float qmap_scaled_d[16];

    for (int row = row_begin; row < row_end; ++row) {
        const size_t byte_base = static_cast<size_t>(row) * (intermediate_size / 2);
        const size_t block_base = static_cast<size_t>(row) * num_blocks;

        __m256 down_acc[kMaxStackTokens][4];
        for (int t = 0; t < tokens && t < kMaxStackTokens; ++t) {
            for (int k = 0; k < 4; ++k) {
                down_acc[t][k] = _mm256_setzero_ps();
            }
        }

        for (int b = 0; b < num_blocks; ++b) {
            const float da = load_host_absmax(slot.down_absmax, slot.down_nested, block_base + b);
            const __m256 da_v = _mm256_set1_ps(da);
            _mm256_store_ps(qmap_scaled_d,     _mm256_mul_ps(_mm256_loadu_ps(qmap), da_v));
            _mm256_store_ps(qmap_scaled_d + 8, _mm256_mul_ps(_mm256_loadu_ps(qmap + 8), da_v));

            const uint8_t* d_packed = slot.down_packed + byte_base + b * bytes_per_block;

            const __m128i d_raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d_packed));
            __m256 dw0, dw1, dw2, dw3;
            unpack_nf4_16bytes_avx2(d_raw0, qmap_scaled_d, dw0, dw1, dw2, dw3);

            const __m128i d_raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d_packed + 16));
            __m256 dw4, dw5, dw6, dw7;
            unpack_nf4_16bytes_avx2(d_raw1, qmap_scaled_d, dw4, dw5, dw6, dw7);

            for (int t = 0; t < tokens && t < kMaxStackTokens; ++t) {
                const float* h_tok_b = h_f32 + static_cast<size_t>(t) * intermediate_size + b * blocksize;
                const __m256 h0 = _mm256_loadu_ps(h_tok_b);
                const __m256 h1 = _mm256_loadu_ps(h_tok_b + 8);
                const __m256 h2 = _mm256_loadu_ps(h_tok_b + 16);
                const __m256 h3 = _mm256_loadu_ps(h_tok_b + 24);
                const __m256 h4 = _mm256_loadu_ps(h_tok_b + 32);
                const __m256 h5 = _mm256_loadu_ps(h_tok_b + 40);
                const __m256 h6 = _mm256_loadu_ps(h_tok_b + 48);
                const __m256 h7 = _mm256_loadu_ps(h_tok_b + 56);

                down_acc[t][0] = _mm256_fmadd_ps(h0, dw0, down_acc[t][0]);
                down_acc[t][1] = _mm256_fmadd_ps(h1, dw1, down_acc[t][1]);
                down_acc[t][2] = _mm256_fmadd_ps(h2, dw2, down_acc[t][2]);
                down_acc[t][3] = _mm256_fmadd_ps(h3, dw3, down_acc[t][3]);
                down_acc[t][0] = _mm256_fmadd_ps(h4, dw4, down_acc[t][0]);
                down_acc[t][1] = _mm256_fmadd_ps(h5, dw5, down_acc[t][1]);
                down_acc[t][2] = _mm256_fmadd_ps(h6, dw6, down_acc[t][2]);
                down_acc[t][3] = _mm256_fmadd_ps(h7, dw7, down_acc[t][3]);
            }
        }

        for (int t = 0; t < tokens && t < kMaxStackTokens; ++t) {
            __m256 d_sum256 = _mm256_add_ps(_mm256_add_ps(down_acc[t][0], down_acc[t][1]),
                                            _mm256_add_ps(down_acc[t][2], down_acc[t][3]));
            const float down_sum = hsum256_ps(d_sum256);
            const size_t out_idx = static_cast<size_t>(t) * out_features + row;
            if (accumulate) {
                out[out_idx] += weight * down_sum;
            } else {
                out[out_idx] = weight * down_sum;
            }
        }
    }
}

#endif // __AVX2__

void cpu_expert_gate_up_swiglu_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map,
    int in_features,
    int intermediate_size,
    int blocksize) {
#if defined(__AVX2__)
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || blocksize <= 0) return;
    if (tokens > kMaxStackTokens) {
        // Fallback to scalar for large batch sizes beyond stack allocation
        cpu_expert_gate_up_swiglu_scalar(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
        return;
    }

    alignas(64) float x_f32_stack[kMaxStackTokens * kHiddenDim];
    convert_bf16_to_f32(x, x_f32_stack, tokens * in_features);

    cpu_expert_gate_up_swiglu_avx2_impl(slot, x_f32_stack, h, 0, intermediate_size, tokens,
                                        quant_map, in_features, intermediate_size, blocksize);
#else
    cpu_expert_gate_up_swiglu_scalar(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
#endif
}

void cpu_expert_down_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map,
    float weight,
    bool accumulate,
    int intermediate_size,
    int out_features,
    int blocksize) {
#if defined(__AVX2__)
    if (tokens <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (tokens > kMaxStackTokens) {
        cpu_expert_down_scalar(slot, h, out, tokens, quant_map, weight, accumulate,
                               intermediate_size, out_features, blocksize);
        return;
    }

    alignas(64) float h_f32_stack[kMaxStackTokens * kIntermediateDim];
    convert_bf16_to_f32(h, h_f32_stack, tokens * intermediate_size);

    cpu_expert_down_avx2_impl(slot, h_f32_stack, out, 0, out_features, tokens,
                              quant_map, weight, accumulate, intermediate_size, out_features, blocksize);
#else
    cpu_expert_down_scalar(slot, h, out, tokens, quant_map, weight, accumulate,
                           intermediate_size, out_features, blocksize);
#endif
}

void cpu_expert_forward_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h,
    const float* quant_map,
    float weight,
    bool accumulate,
    int in_features,
    int intermediate_size,
    int out_features,
    int blocksize) {
    alignas(64) bf16 local_h_stack[kMaxStackTokens * kIntermediateDim];
    std::vector<bf16> local_h_heap;
    bf16* effective_h = scratch_h;
    if (!effective_h) {
        if (tokens <= kMaxStackTokens && intermediate_size <= kIntermediateDim) {
            effective_h = local_h_stack;
        } else {
            local_h_heap.resize(static_cast<size_t>(tokens) * intermediate_size);
            effective_h = local_h_heap.data();
        }
    }

    cpu_expert_gate_up_swiglu_avx2(slot, x, effective_h, tokens, quant_map,
                                   in_features, intermediate_size, blocksize);
    cpu_expert_down_avx2(slot, effective_h, out, tokens, quant_map,
                         weight, accumulate, intermediate_size, out_features, blocksize);
}

// =========================================================================
// oneTBB + AVX2 Parallel Implementations
// =========================================================================

void cpu_expert_gate_up_swiglu_tbb_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map,
    int in_features,
    int intermediate_size,
    int blocksize) {
#if defined(__AVX2__)
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || blocksize <= 0) return;
    if (tokens > kMaxStackTokens) {
        cpu_expert_gate_up_swiglu_scalar(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
        return;
    }

    alignas(64) float x_f32_stack[kMaxStackTokens * kHiddenDim];
    convert_bf16_to_f32(x, x_f32_stack, tokens * in_features);

    tbb::parallel_for(
        tbb::blocked_range<int>(0, intermediate_size, 16),
        [&](const tbb::blocked_range<int>& range) {
            cpu_expert_gate_up_swiglu_avx2_impl(
                slot, x_f32_stack, h, range.begin(), range.end(), tokens,
                quant_map, in_features, intermediate_size, blocksize);
        });
#else
    cpu_expert_gate_up_swiglu_scalar(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
#endif
}

void cpu_expert_down_tbb_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map,
    float weight,
    bool accumulate,
    int intermediate_size,
    int out_features,
    int blocksize) {
#if defined(__AVX2__)
    if (tokens <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (tokens > kMaxStackTokens) {
        cpu_expert_down_scalar(slot, h, out, tokens, quant_map, weight, accumulate,
                               intermediate_size, out_features, blocksize);
        return;
    }

    alignas(64) float h_f32_stack[kMaxStackTokens * kIntermediateDim];
    convert_bf16_to_f32(h, h_f32_stack, tokens * intermediate_size);

    tbb::parallel_for(
        tbb::blocked_range<int>(0, out_features, 32),
        [&](const tbb::blocked_range<int>& range) {
            cpu_expert_down_avx2_impl(
                slot, h_f32_stack, out, range.begin(), range.end(), tokens,
                quant_map, weight, accumulate, intermediate_size, out_features, blocksize);
        });
#else
    cpu_expert_down_scalar(slot, h, out, tokens, quant_map, weight, accumulate,
                           intermediate_size, out_features, blocksize);
#endif
}

void cpu_expert_forward_tbb_avx2(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h,
    const float* quant_map,
    float weight,
    bool accumulate,
    int in_features,
    int intermediate_size,
    int out_features,
    int blocksize) {
    alignas(64) bf16 local_h_stack[kMaxStackTokens * kIntermediateDim];
    std::vector<bf16> local_h_heap;
    bf16* effective_h = scratch_h;
    if (!effective_h) {
        if (tokens <= kMaxStackTokens && intermediate_size <= kIntermediateDim) {
            effective_h = local_h_stack;
        } else {
            local_h_heap.resize(static_cast<size_t>(tokens) * intermediate_size);
            effective_h = local_h_heap.data();
        }
    }

    cpu_expert_gate_up_swiglu_tbb_avx2(slot, x, effective_h, tokens, quant_map,
                                       in_features, intermediate_size, blocksize);
    cpu_expert_down_tbb_avx2(slot, effective_h, out, tokens, quant_map,
                             weight, accumulate, intermediate_size, out_features, blocksize);
}

// =========================================================================
// General Dispatch Entry Points
// =========================================================================

void cpu_expert_gate_up_swiglu(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    bf16* h,
    int tokens,
    const float* quant_map,
    int in_features,
    int intermediate_size,
    int blocksize,
    CpuKernelBackend backend) {
    switch (backend) {
        case CpuKernelBackend::Scalar:
            cpu_expert_gate_up_swiglu_scalar(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
            break;
        case CpuKernelBackend::Avx2:
            cpu_expert_gate_up_swiglu_avx2(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
            break;
        case CpuKernelBackend::TbbAvx2:
        case CpuKernelBackend::Auto:
        default:
            cpu_expert_gate_up_swiglu_tbb_avx2(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
            break;
    }
}

void cpu_expert_down(
    const MoeExpertSlotHost& slot,
    const bf16* h,
    float* out,
    int tokens,
    const float* quant_map,
    float weight,
    bool accumulate,
    int intermediate_size,
    int out_features,
    int blocksize,
    CpuKernelBackend backend) {
    switch (backend) {
        case CpuKernelBackend::Scalar:
            cpu_expert_down_scalar(slot, h, out, tokens, quant_map, weight, accumulate,
                                   intermediate_size, out_features, blocksize);
            break;
        case CpuKernelBackend::Avx2:
            cpu_expert_down_avx2(slot, h, out, tokens, quant_map, weight, accumulate,
                                 intermediate_size, out_features, blocksize);
            break;
        case CpuKernelBackend::TbbAvx2:
        case CpuKernelBackend::Auto:
        default:
            cpu_expert_down_tbb_avx2(slot, h, out, tokens, quant_map, weight, accumulate,
                                     intermediate_size, out_features, blocksize);
            break;
    }
}

void cpu_expert_forward(
    const MoeExpertSlotHost& slot,
    const bf16* x,
    float* out,
    int tokens,
    bf16* scratch_h,
    const float* quant_map,
    float weight,
    bool accumulate,
    int in_features,
    int intermediate_size,
    int out_features,
    int blocksize,
    CpuKernelBackend backend) {
    switch (backend) {
        case CpuKernelBackend::Scalar:
            cpu_expert_forward_scalar(slot, x, out, tokens, scratch_h, quant_map, weight, accumulate,
                                      in_features, intermediate_size, out_features, blocksize);
            break;
        case CpuKernelBackend::Avx2:
            cpu_expert_forward_avx2(slot, x, out, tokens, scratch_h, quant_map, weight, accumulate,
                                    in_features, intermediate_size, out_features, blocksize);
            break;
        case CpuKernelBackend::TbbAvx2:
        case CpuKernelBackend::Auto:
        default:
            cpu_expert_forward_tbb_avx2(slot, x, out, tokens, scratch_h, quant_map, weight, accumulate,
                                        in_features, intermediate_size, out_features, blocksize);
            break;
    }
}

} // namespace cpu
} // namespace ops
