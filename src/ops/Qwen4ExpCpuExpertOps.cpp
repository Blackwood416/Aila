#include "Qwen4ExpCpuExpertOps.hpp"
#include "utils/EnvUtils.hpp"
#include "ops/CpuTilePool.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <atomic>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>

#if defined(_MSC_VER)
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

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
                              size_t index, bool nf4_scale_shift = false) {
    if (nested.codes != nullptr) {
        const size_t bs = nested.blocksize > 0 ? static_cast<size_t>(nested.blocksize) : 256;
        const size_t nested_index = nf4_scale_shift && bs == 256 ? index >> 8 : index / bs;
        // Match GPU/fill two-step f32 rounding; contraction changes real scales.
        volatile float product = nested.nested_quant_map[nested.codes[index]] *
                                 nested.nested_absmax[nested_index];
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
    if (!slot.gate_up_packed || !x || !h) return;
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
    if (!slot.down_packed || !h || !out) return;
    if (tokens <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (accumulate && weight == 0.0f) return;
    if (!accumulate && weight == 0.0f) {
        std::memset(out, 0, static_cast<size_t>(tokens) * out_features * sizeof(float));
        return;
    }

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
    if (!slot.gate_up_packed || !slot.down_packed || !x || !out) return;
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (accumulate && weight == 0.0f) return;

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

static float gpu_order_dot(const uint8_t* packed, const float* absmax,
                           const MoeNestedScaleHost& nested, size_t row,
                           const float* x, int columns, const float* qmap) {
    const __m256i byte_offsets = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i input_offsets = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);
    __m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int column = 0; column < columns; column += 128) {
        for (int half = 0; half < 2; ++half) {
            const int start = column + half * 64;
            const float scale = load_host_absmax(absmax, nested, row * (columns / 64) + start / 64);
            const auto raw = _mm256_i32gather_epi32(
                reinterpret_cast<const int*>(packed + row * (columns / 2) + start / 2), byte_offsets, 1);
            for (int j = 0; j < 8; ++j) {
                const int shift = (j / 2) * 8 + ((j & 1) ? 0 : 4);
                const auto codes = _mm256_and_si256(_mm256_srlv_epi32(raw, _mm256_set1_epi32(shift)), _mm256_set1_epi32(15));
                const auto w = _mm256_mul_ps(_mm256_i32gather_ps(qmap, codes, 4), _mm256_set1_ps(scale));
                const auto values = _mm256_i32gather_ps(x + start + j, input_offsets, 4);
                acc[half] = _mm256_fmadd_ps(values, w, acc[half]);
            }
        }
    }
    alignas(32) float parts[16];
    _mm256_store_ps(parts, acc[0]);
    _mm256_store_ps(parts + 8, acc[1]);
    volatile float sums[16];
    for (int i = 0; i < 16; ++i) sums[i] = parts[i];
    for (int stride = 8; stride; stride /= 2)
        for (int i = 0; i < stride; ++i) sums[i] = sums[i] + sums[i + stride];
    return sums[0];
}

static int lane_input_index(int column) {
    return (column / 64) * 64 + (column % 8) * 8 + (column % 64) / 8;
}

static float gpu_order_dot_prepared(const uint8_t* packed, const float* absmax,
                                   const MoeNestedScaleHost& nested, size_t row,
                                   const float* x, int columns, const float* qmap) {
    const auto qlo = _mm256_loadu_ps(qmap);
    const auto qhi = _mm256_loadu_ps(qmap + 8);
    __m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int column = 0; column < columns; column += 128) {
        for (int half = 0; half < 2; ++half) {
            const int start = column + half * 64;
            const auto scale = _mm256_set1_ps(load_host_absmax(absmax, nested, row * (columns / 64) + start / 64));
            const auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + row * (columns / 2) + start / 2));
            for (int j = 0; j < 8; ++j) {
                const int shift = (j / 2) * 8 + ((j & 1) ? 0 : 4);
                const auto codes = _mm256_and_si256(_mm256_srlv_epi32(raw, _mm256_set1_epi32(shift)), _mm256_set1_epi32(15));
                const auto low = _mm256_permutevar8x32_ps(qlo, codes);
                const auto high = _mm256_permutevar8x32_ps(qhi, codes);
                const auto values = _mm256_blendv_ps(low, high, _mm256_castsi256_ps(_mm256_slli_epi32(codes, 28)));
                const auto w = _mm256_mul_ps(values, scale);
                acc[half] = _mm256_fmadd_ps(_mm256_loadu_ps(x + start + j * 8), w, acc[half]);
            }
        }
    }
    // Same 16-lane reduction and rounding as the unprepared reference.
    alignas(32) float parts[16];
    _mm256_store_ps(parts, acc[0]);
    _mm256_store_ps(parts + 8, acc[1]);
    volatile float sums[16];
    for (int i = 0; i < 16; ++i) sums[i] = parts[i];
    for (int stride = 8; stride; stride /= 2)
        for (int i = 0; i < stride; ++i) sums[i] = sums[i] + sums[i + stride];
    return sums[0];
}

// Experimental CPU-only arithmetic. NF4 bytes/scales stay in their original
// layout; an int8 lookup and per-32 activation quantization replace FP32 dots.
// This borrows the integer-dot strategy used by Strata, with a separate NF4
// contract. It is not the GPU-order reference and requires model quality gates.
static void quantize_bf16_q8(const bf16* input, int columns, int8_t* codes, float* scales) {
    const auto abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int begin = 0; begin < columns; begin += 32) {
        __m256 values[4];
        auto maximum = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            const auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + begin + i * 8));
            values[i] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
            maximum = _mm256_max_ps(maximum, _mm256_and_ps(values[i], abs_mask));
        }
        auto max4 = _mm_max_ps(_mm256_castps256_ps128(maximum), _mm256_extractf128_ps(maximum, 1));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        const float amax = _mm_cvtss_f32(_mm_max_ss(max4, _mm_movehdup_ps(max4)));
        scales[begin / 32] = amax / 127.0f;
        const auto inverse = _mm256_set1_ps(amax > 0.0f ? 127.0f / amax : 0.0f);
        __m256i q[4];
        for (int i = 0; i < 4; ++i) {
            // Fixed nearest-even rule, independent of the caller's MXCSR mode.
            const auto rounded = _mm256_round_ps(_mm256_mul_ps(values[i], inverse), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            q[i] = _mm256_cvttps_epi32(rounded);
            q[i] = _mm256_min_epi32(_mm256_max_epi32(q[i], _mm256_set1_epi32(-127)), _mm256_set1_epi32(127));
        }
        const auto q01 = _mm256_packs_epi32(q[0], q[1]);
        const auto q23 = _mm256_packs_epi32(q[2], q[3]);
        const auto packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(q01, q23),
            _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(codes + begin), packed);
    }
}

// A separate arithmetic contract: both halves share one activation scale.
// Keep the old per-32 quantizer unchanged for the H8 and bitwise candidates.
static void quantize_bf16_q8_activation64(const bf16* input, int columns, int8_t* codes, float* scales) {
    const auto abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int begin = 0; begin < columns; begin += 64) {
        __m256 values[8];
        auto maximum = _mm256_setzero_ps();
        for (int i = 0; i < 8; ++i) {
            const auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + begin + i * 8));
            values[i] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
            maximum = _mm256_max_ps(maximum, _mm256_and_ps(values[i], abs_mask));
        }
        auto max4 = _mm_max_ps(_mm256_castps256_ps128(maximum), _mm256_extractf128_ps(maximum, 1));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        const float amax = _mm_cvtss_f32(_mm_max_ss(max4, _mm_movehdup_ps(max4)));
        scales[begin / 64] = amax / 127.0f;
        const auto inverse = _mm256_set1_ps(amax > 0.0f ? 127.0f / amax : 0.0f);
        for (int half = 0; half < 2; ++half) {
            __m256i q[4];
            for (int i = 0; i < 4; ++i) {
                const auto rounded = _mm256_round_ps(_mm256_mul_ps(values[half * 4 + i], inverse),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                q[i] = _mm256_cvttps_epi32(rounded);
                q[i] = _mm256_min_epi32(_mm256_max_epi32(q[i], _mm256_set1_epi32(-127)), _mm256_set1_epi32(127));
            }
            const auto q01 = _mm256_packs_epi32(q[0], q[1]);
            const auto q23 = _mm256_packs_epi32(q[2], q[3]);
            const auto packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(q01, q23),
                _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(codes + begin + half * 32), packed);
        }
    }
}

static float nf4_q8_dot(const uint8_t* packed, const float* absmax,
                         const MoeNestedScaleHost& nested, size_t row,
                         const int8_t* input, const float* input_scales,
                         int columns, __m128i lut, float lut_scale, bool nf4_scale_shift) {
    const auto mask = _mm_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    auto accumulator = _mm256_setzero_ps();
    for (int b = 0; b < columns / 64; ++b) {
        const float scale = load_host_absmax(absmax, nested, row * (columns / 64) + b, nf4_scale_shift);
        for (int half = 0; half < 2; ++half) {
            const auto bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + row * (columns / 2) + b * 32 + half * 16));
            const auto high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            const auto low = _mm_and_si128(bytes, mask);
            // BNB NF4 stores the earlier element in the high nibble.
            const auto w0 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(high, low));
            const auto w1 = _mm_shuffle_epi8(lut, _mm_unpackhi_epi8(high, low));
            const auto weights = _mm256_set_m128i(w1, w0);
            const auto activation = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64 + half * 32));
            // abs(weight)<=127 and activation excludes -128, so each adjacent
            // product pair fits int16 (2*127*127=32258): maddubs cannot saturate.
            const auto pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(weights), _mm256_sign_epi8(activation, weights));
            const auto dot = _mm256_madd_epi16(pairs, ones);
            accumulator = _mm256_fmadd_ps(_mm256_set1_ps(scale * input_scales[b * 2 + half] * lut_scale),
                                          _mm256_cvtepi32_ps(dot), accumulator);
        }
    }
    return hsum256_ps(accumulator);
}

// One 32-byte read decodes all 64 NF4 weights. AVX2 byte unpack/shuffle work
// independently within 128-bit lanes, so the two decoded vectors initially
// contain [0..15,32..47] and [16..31,48..63]. Recombine those lanes into the
// original activation halves before performing H8's half0 then half1 FMA.
template <int Columns>
#if defined(_MSC_VER)
static __forceinline float nf4_q8_dot_block64(const uint8_t* packed, const float* absmax,
#else
static inline __attribute__((always_inline)) float nf4_q8_dot_block64(const uint8_t* packed, const float* absmax,
#endif
    const MoeNestedScaleHost& nested, size_t row, const int8_t* input,
    const float* input_scales, __m128i lut, float lut_scale, bool nf4_scale_shift) {
    static_assert(Columns > 0 && Columns % 64 == 0, "NF4Q8 block64 requires complete 64-weight blocks");
    const auto mask = _mm256_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    const auto table = _mm256_broadcastsi128_si256(lut);
    auto accumulator = _mm256_setzero_ps();
    for (int b = 0; b < Columns / 64; ++b) {
        const float scale = load_host_absmax(absmax, nested, row * (Columns / 64) + b, nf4_scale_shift);
        const auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + row * (Columns / 2) + b * 32));
        const auto high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);
        const auto low = _mm256_and_si256(bytes, mask);
        const auto wlo = _mm256_shuffle_epi8(table, _mm256_unpacklo_epi8(high, low));
        const auto whi = _mm256_shuffle_epi8(table, _mm256_unpackhi_epi8(high, low));
        const auto weights0 = _mm256_permute2x128_si256(wlo, whi, 0x20);
        const auto weights1 = _mm256_permute2x128_si256(wlo, whi, 0x31);
        const auto activation0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64));
        const auto pairs0 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights0), _mm256_sign_epi8(activation0, weights0));
        const auto dot0 = _mm256_madd_epi16(pairs0, ones);
        accumulator = _mm256_fmadd_ps(_mm256_set1_ps(scale * input_scales[b * 2] * lut_scale),
            _mm256_cvtepi32_ps(dot0), accumulator);
        const auto activation1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64 + 32));
        const auto pairs1 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights1), _mm256_sign_epi8(activation1, weights1));
        const auto dot1 = _mm256_madd_epi16(pairs1, ones);
        accumulator = _mm256_fmadd_ps(_mm256_set1_ps(scale * input_scales[b * 2 + 1] * lut_scale),
            _mm256_cvtepi32_ps(dot1), accumulator);
    }
    return hsum256_ps(accumulator);
}

// A64 source specialization is opt-in. Select once per tile, so the
// block loop has neither source-kind guards nor nested blocksize division.
enum class A64ScaleMode { Generic, FlatF32, Nested256, Nested256Register, Nested256ByteShuffle };

inline float a64_nested_scale_register(const MoeNestedScaleHost& nested, size_t index) {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    // Two scalar SSE operations retain the original separately rounded f32
    // multiply/add while avoiding volatile stack round trips in every block.
    auto product = _mm_mul_ss(
        _mm_set_ss(nested.nested_quant_map[nested.codes[index]]),
        _mm_set_ss(nested.nested_absmax[index >> 8]));
#if defined(__clang__)
    // Intrinsic mul/add can still contract after inlining despite local fp
    // pragmas. An opaque register value blocks FMA without stack traffic.
    __asm__ __volatile__("" : "+x"(product));
#else
    volatile float rounded = _mm_cvtss_f32(product);
    product = _mm_set_ss(rounded);
#endif
    return _mm_cvtss_f32(_mm_add_ss(product, _mm_set_ss(nested.offset)));
}

inline A64ScaleMode a64_scale_mode(const float* absmax,
    const MoeNestedScaleHost& nested, bool nf4_scale_shift) {
    static const bool registerScale = aila::env::read_flag("AILA_Q4E_CPU_SCALE_REGISTER", true);
    static const bool byteShuffle = aila::env::read_flag("AILA_Q4E_CPU_BYTE_SHUFFLE", false);
    if (nested.codes == nullptr && absmax != nullptr) return A64ScaleMode::FlatF32;
    if (nf4_scale_shift && nested.codes != nullptr && nested.blocksize == 256 &&
        nested.nested_absmax != nullptr && nested.nested_quant_map != nullptr)
        return byteShuffle ? A64ScaleMode::Nested256ByteShuffle : registerScale
            ? A64ScaleMode::Nested256Register : A64ScaleMode::Nested256;
    return A64ScaleMode::Generic;
}

inline void mark_a64_specialize_enabled(A64ScaleMode mode) {
    // Report only a source-specialized tile that actually completed; generic
    // fallback alone must not claim the guarded inner loop was eliminated.
    if (mode == A64ScaleMode::Generic) return;
    static std::atomic<bool> reported{false};
    if (!reported.load(std::memory_order_relaxed) &&
        !reported.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[Q4E A64SpecializeEnabled] source=%s\n",
            mode == A64ScaleMode::FlatF32 ? "flat_f32" : "nested256");
}

// Independent A64 approximation. Each lane merges two four-product integer
// sums before conversion and one FMA. 8*127*127=129032 fits int32; each
// maddubs pair is at most 32258, so signed saturation cannot occur.
template <int Columns, A64ScaleMode Mode = A64ScaleMode::Generic>
#if defined(_MSC_VER)
static __forceinline float nf4_q8_dot_activation64(const uint8_t* packed, const float* absmax,
#else
static inline __attribute__((always_inline)) float nf4_q8_dot_activation64(const uint8_t* packed, const float* absmax,
#endif
    const MoeNestedScaleHost& nested, size_t row, const int8_t* input,
    const float* input_scales, __m128i lut, float lut_scale, bool nf4_scale_shift) {
    static_assert(Columns > 0 && Columns % 64 == 0, "NF4Q8 A64 requires complete 64-weight blocks");
    const auto mask = _mm256_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    const auto table = _mm256_broadcastsi128_si256(lut);
    auto accumulator = _mm256_setzero_ps();
    for (int b = 0; b < Columns / 64; ++b) {
        const size_t scale_index = row * (Columns / 64) + b;
        float scale;
        if constexpr (Mode == A64ScaleMode::FlatF32) {
            scale = absmax[scale_index];
        } else if constexpr ((Mode == A64ScaleMode::Nested256 || Mode == A64ScaleMode::Nested256ByteShuffle)) {
            // Keep the original separately rounded f32 multiply then add.
            volatile float product = nested.nested_quant_map[nested.codes[scale_index]] *
                                     nested.nested_absmax[scale_index >> 8];
            scale = product + nested.offset;
        } else if constexpr (Mode == A64ScaleMode::Nested256Register) {
            scale = a64_nested_scale_register(nested, scale_index);
        } else {
            scale = load_host_absmax(absmax, nested, scale_index, nf4_scale_shift);
        }
        auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + row * (Columns / 2) + b * 32));
        if constexpr (Mode == A64ScaleMode::Nested256ByteShuffle) bytes = _mm256_permute4x64_epi64(bytes, _MM_SHUFFLE(3, 1, 2, 0));
        const auto high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);
        const auto low = _mm256_and_si256(bytes, mask);
        const auto wlo = _mm256_shuffle_epi8(table, _mm256_unpacklo_epi8(high, low));
        const auto whi = _mm256_shuffle_epi8(table, _mm256_unpackhi_epi8(high, low));
        const auto weights0 = Mode == A64ScaleMode::Nested256ByteShuffle ? wlo : _mm256_permute2x128_si256(wlo, whi, 0x20);
        const auto weights1 = Mode == A64ScaleMode::Nested256ByteShuffle ? whi : _mm256_permute2x128_si256(wlo, whi, 0x31);
        const auto activation0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64));
        const auto activation1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64 + 32));
        const auto pairs0 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights0), _mm256_sign_epi8(activation0, weights0));
        const auto pairs1 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights1), _mm256_sign_epi8(activation1, weights1));
        const auto dot = _mm256_add_epi32(_mm256_madd_epi16(pairs0, ones), _mm256_madd_epi16(pairs1, ones));
        // Spell out left-associated FP32 multiplies; the scalar reference
        // checks this contract without relying on fast-math reassociation.
        const auto scale_input = _mm_mul_ss(_mm_set_ss(scale), _mm_set_ss(input_scales[b]));
        const float factor = _mm_cvtss_f32(_mm_mul_ss(scale_input, _mm_set_ss(lut_scale)));
        accumulator = _mm256_fmadd_ps(_mm256_set1_ps(factor), _mm256_cvtepi32_ps(dot), accumulator);
    }
    return hsum256_ps(accumulator);
}

template <A64ScaleMode Mode>
static void nf4_q8_activation64_gate_tile(const MoeExpertSlotHost& slot,
    int begin, const int8_t* values, const float* scales, __m128i lut,
    float lut_scale, bool nf4_scale_shift, bf16* h) {
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int row = begin; row < begin + 16; ++row) {
        const float gate = nf4_q8_dot_activation64<kHiddenDim, Mode>(slot.gate_up_packed,
            slot.gate_up_absmax, slot.gate_up_nested, row, values, scales, lut, lut_scale, nf4_scale_shift);
        const float up = nf4_q8_dot_activation64<kHiddenDim, Mode>(slot.gate_up_packed,
            slot.gate_up_absmax, slot.gate_up_nested, row + kIntermediateDim,
            values, scales, lut, lut_scale, nf4_scale_shift);
        const float denominator = 1.0f + std::exp(-gate);
        const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
        h[row] = bf16(silu * up);
    }
}

template <A64ScaleMode Mode>
static void nf4_q8_activation64_down_tile(const MoeExpertSlotHost& slot,
    int begin, const int8_t* values, const float* scales, __m128i lut,
    float lut_scale, bool nf4_scale_shift, float* out) {
    for (int row = begin; row < begin + 32; ++row)
        out[row] = nf4_q8_dot_activation64<kIntermediateDim, Mode>(slot.down_packed,
            slot.down_absmax, slot.down_nested, row, values, scales, lut, lut_scale, nf4_scale_shift);
}

// A64 row2 shares only immutable activation loads. Independent accumulators
// keep each row's original increasing-block FMA and horizontal reduction.
template <int Columns, A64ScaleMode Mode>
#if defined(_MSC_VER)
static __forceinline void nf4_q8_dot_activation64_rows2(const uint8_t* packed, const float* absmax,
#else
static inline __attribute__((always_inline)) void nf4_q8_dot_activation64_rows2(const uint8_t* packed, const float* absmax,
#endif
    const MoeNestedScaleHost& nested, size_t row0, size_t row1, const int8_t* input,
    const float* input_scales, __m128i lut, float lut_scale, bool nf4_scale_shift, float* output) {
    static_assert(Columns > 0 && Columns % 64 == 0, "NF4Q8 A64 requires complete 64-weight blocks");
    const auto mask = _mm256_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    const auto table = _mm256_broadcastsi128_si256(lut);
    auto accumulator0 = _mm256_setzero_ps();
    auto accumulator1 = _mm256_setzero_ps();
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int b = 0; b < Columns / 64; ++b) {
        const auto activation0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64));
        const auto activation1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64 + 32));
        {
            const size_t scale_index = row0 * (Columns / 64) + b;
            float scale;
            if constexpr (Mode == A64ScaleMode::FlatF32) {
                scale = absmax[scale_index];
            } else if constexpr ((Mode == A64ScaleMode::Nested256 || Mode == A64ScaleMode::Nested256ByteShuffle)) {
                // Keep the original separately rounded f32 multiply then add.
                volatile float product = nested.nested_quant_map[nested.codes[scale_index]] *
                                         nested.nested_absmax[scale_index >> 8];
                scale = product + nested.offset;
            } else if constexpr (Mode == A64ScaleMode::Nested256Register) {
                scale = a64_nested_scale_register(nested, scale_index);
            } else {
                scale = load_host_absmax(absmax, nested, scale_index, nf4_scale_shift);
            }
            auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + row0 * (Columns / 2) + b * 32));
            if constexpr (Mode == A64ScaleMode::Nested256ByteShuffle) bytes = _mm256_permute4x64_epi64(bytes, _MM_SHUFFLE(3, 1, 2, 0));
            const auto high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);
            const auto low = _mm256_and_si256(bytes, mask);
            const auto wlo = _mm256_shuffle_epi8(table, _mm256_unpacklo_epi8(high, low));
            const auto whi = _mm256_shuffle_epi8(table, _mm256_unpackhi_epi8(high, low));
            const auto weights0 = Mode == A64ScaleMode::Nested256ByteShuffle ? wlo : _mm256_permute2x128_si256(wlo, whi, 0x20);
            const auto weights1 = Mode == A64ScaleMode::Nested256ByteShuffle ? whi : _mm256_permute2x128_si256(wlo, whi, 0x31);
            const auto pairs0 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights0), _mm256_sign_epi8(activation0, weights0));
            const auto pairs1 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights1), _mm256_sign_epi8(activation1, weights1));
            const auto dot = _mm256_add_epi32(_mm256_madd_epi16(pairs0, ones), _mm256_madd_epi16(pairs1, ones));
            // Spell out left-associated FP32 multiplies; the scalar reference
            // checks this contract without relying on fast-math reassociation.
            const auto scale_input = _mm_mul_ss(_mm_set_ss(scale), _mm_set_ss(input_scales[b]));
            const float factor = _mm_cvtss_f32(_mm_mul_ss(scale_input, _mm_set_ss(lut_scale)));
            accumulator0 = _mm256_fmadd_ps(_mm256_set1_ps(factor), _mm256_cvtepi32_ps(dot), accumulator0);
        }
        {
            const size_t scale_index = row1 * (Columns / 64) + b;
            float scale;
            if constexpr (Mode == A64ScaleMode::FlatF32) {
                scale = absmax[scale_index];
            } else if constexpr ((Mode == A64ScaleMode::Nested256 || Mode == A64ScaleMode::Nested256ByteShuffle)) {
                // Keep the original separately rounded f32 multiply then add.
                volatile float product = nested.nested_quant_map[nested.codes[scale_index]] *
                                         nested.nested_absmax[scale_index >> 8];
                scale = product + nested.offset;
            } else if constexpr (Mode == A64ScaleMode::Nested256Register) {
                scale = a64_nested_scale_register(nested, scale_index);
            } else {
                scale = load_host_absmax(absmax, nested, scale_index, nf4_scale_shift);
            }
            auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + row1 * (Columns / 2) + b * 32));
            if constexpr (Mode == A64ScaleMode::Nested256ByteShuffle) bytes = _mm256_permute4x64_epi64(bytes, _MM_SHUFFLE(3, 1, 2, 0));
            const auto high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);
            const auto low = _mm256_and_si256(bytes, mask);
            const auto wlo = _mm256_shuffle_epi8(table, _mm256_unpacklo_epi8(high, low));
            const auto whi = _mm256_shuffle_epi8(table, _mm256_unpackhi_epi8(high, low));
            const auto weights0 = Mode == A64ScaleMode::Nested256ByteShuffle ? wlo : _mm256_permute2x128_si256(wlo, whi, 0x20);
            const auto weights1 = Mode == A64ScaleMode::Nested256ByteShuffle ? whi : _mm256_permute2x128_si256(wlo, whi, 0x31);
            const auto pairs0 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights0), _mm256_sign_epi8(activation0, weights0));
            const auto pairs1 = _mm256_maddubs_epi16(_mm256_abs_epi8(weights1), _mm256_sign_epi8(activation1, weights1));
            const auto dot = _mm256_add_epi32(_mm256_madd_epi16(pairs0, ones), _mm256_madd_epi16(pairs1, ones));
            // Spell out left-associated FP32 multiplies; the scalar reference
            // checks this contract without relying on fast-math reassociation.
            const auto scale_input = _mm_mul_ss(_mm_set_ss(scale), _mm_set_ss(input_scales[b]));
            const float factor = _mm_cvtss_f32(_mm_mul_ss(scale_input, _mm_set_ss(lut_scale)));
            accumulator1 = _mm256_fmadd_ps(_mm256_set1_ps(factor), _mm256_cvtepi32_ps(dot), accumulator1);
        }
    }
    output[0] = hsum256_ps(accumulator0);
    output[1] = hsum256_ps(accumulator1);
}

inline void mark_a64_row2_enabled() {
    static std::atomic<bool> reported{false};
    if (!reported.load(std::memory_order_relaxed) &&
        !reported.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[Q4E A64Row2Enabled] rows=2\n");
}

template <A64ScaleMode Mode>
static void dispatch_a64_gate_tile(const MoeExpertSlotHost& slot,
    int begin, const int8_t* values, const float* scales, __m128i lut,
    float lut_scale, bool nf4_scale_shift, bf16* h, bool row2) {
    if (!row2) {
        nf4_q8_activation64_gate_tile<Mode>(slot, begin, values, scales,
            lut, lut_scale, nf4_scale_shift, h);
        return;
    }
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int row = begin; row < begin + 16; ++row) {
        float dots[2];
        nf4_q8_dot_activation64_rows2<kHiddenDim, Mode>(slot.gate_up_packed,
            slot.gate_up_absmax, slot.gate_up_nested, row, row + kIntermediateDim,
            values, scales, lut, lut_scale, nf4_scale_shift, dots);
        const float gate = dots[0], up = dots[1];
        const float denominator = 1.0f + std::exp(-gate);
        const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
        h[row] = bf16(silu * up);
    }
    mark_a64_row2_enabled();
}

template <A64ScaleMode Mode>
static void dispatch_a64_down_tile(const MoeExpertSlotHost& slot,
    int begin, const int8_t* values, const float* scales, __m128i lut,
    float lut_scale, bool nf4_scale_shift, float* out, bool row2) {
    if (!row2) {
        nf4_q8_activation64_down_tile<Mode>(slot, begin, values, scales,
            lut, lut_scale, nf4_scale_shift, out);
        return;
    }
    for (int row = begin; row < begin + 32; row += 2)
        nf4_q8_dot_activation64_rows2<kIntermediateDim, Mode>(slot.down_packed,
            slot.down_absmax, slot.down_nested, row, row + 1, values, scales,
            lut, lut_scale, nf4_scale_shift, out + row);
    mark_a64_row2_enabled();
}

// Keep the single-row reference above intact. Four independent accumulators
// share activation loads, but each row sees exactly its original block/half
// FMA order, scale expression and final horizontal reduction.
template <int Columns>
#if defined(_MSC_VER)
static __forceinline void nf4_q8_dot_rows4(const uint8_t* packed, const float* absmax,
#else
static inline __attribute__((always_inline)) void nf4_q8_dot_rows4(const uint8_t* packed, const float* absmax,
#endif
    const MoeNestedScaleHost& nested, size_t row0, size_t row1, size_t row2, size_t row3,
    const int8_t* input, const float* input_scales,
    __m128i lut, float lut_scale, float* result, bool nf4_scale_shift) {
    static_assert(Columns > 0 && Columns % 64 == 0, "NF4Q8 row4 requires complete 64-weight blocks");
    const auto mask = _mm_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    auto acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    auto acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    constexpr size_t stride = Columns / 2;
    for (int b = 0; b < Columns / 64; ++b) {
        const float scale0 = load_host_absmax(absmax, nested, row0 * (Columns / 64) + b, nf4_scale_shift);
        const float scale1 = load_host_absmax(absmax, nested, row1 * (Columns / 64) + b, nf4_scale_shift);
        const float scale2 = load_host_absmax(absmax, nested, row2 * (Columns / 64) + b, nf4_scale_shift);
        const float scale3 = load_host_absmax(absmax, nested, row3 * (Columns / 64) + b, nf4_scale_shift);
        for (int half = 0; half < 2; ++half) {
            const auto activation = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + b * 64 + half * 32));
            const float input_scale = input_scales[b * 2 + half];
            const size_t offset = b * 32 + half * 16;
            auto update = [&](size_t row, float scale, __m256 accumulator) {
                const auto bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + row * stride + offset));
                const auto high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
                const auto low = _mm_and_si128(bytes, mask);
                const auto w0 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(high, low));
                const auto w1 = _mm_shuffle_epi8(lut, _mm_unpackhi_epi8(high, low));
                const auto weights = _mm256_set_m128i(w1, w0);
                const auto pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(weights), _mm256_sign_epi8(activation, weights));
                const auto dot = _mm256_madd_epi16(pairs, ones);
                return _mm256_fmadd_ps(_mm256_set1_ps(scale * input_scale * lut_scale),
                    _mm256_cvtepi32_ps(dot), accumulator);
            };
            acc0 = update(row0, scale0, acc0);
            acc1 = update(row1, scale1, acc1);
            acc2 = update(row2, scale2, acc2);
            acc3 = update(row3, scale3, acc3);
        }
    }
    result[0] = hsum256_ps(acc0); result[1] = hsum256_ps(acc1);
    result[2] = hsum256_ps(acc2); result[3] = hsum256_ps(acc3);
}

int nf4_q8_chunk_factor() {
    static const int factor = std::clamp(aila::env::read_int_raw("AILA_Q4E_CPU_Q8_CHUNKS", 0), 0, 8);
    return factor;
}

template <class Body>
int nf4_q8_parallel_tiles(int tile_count, const Body& body) {
    if (currentTilePool) {
        currentTilePool->ParallelFor(tile_count, [&](int first, int last) {
            body(tbb::blocked_range<int>(first, last, 1));
        });
        return 0;
    }
    const int factor = nf4_q8_chunk_factor();
    if (!factor) {
        tbb::parallel_for(tbb::blocked_range<int>(0, tile_count, 1), body);
        return 0;
    }
    const int chunks = std::min(tile_count, factor * std::max(1, tbb::this_task_arena::max_concurrency()));
    tbb::parallel_for(tbb::blocked_range<int>(0, chunks, 1), [&](const tbb::blocked_range<int>& range) {
        for (int chunk = range.begin(); chunk < range.end(); ++chunk) {
            // Integral floor boundaries cover every complete original tile once.
            body(tbb::blocked_range<int>(chunk * tile_count / chunks, (chunk + 1) * tile_count / chunks, 1));
        }
    });
    return chunks;
}

uint64_t nf4_q8_phase_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void nf4_q8_activation64_dag_tiles(const CpuExpertTask* tasks, int count,
    const int8_t* input, const float* input_scales, bf16* h, float* out,
    int8_t* intermediate, float* intermediate_scales, __m128i lut, float lut_scale,
    bool nf4_scale_shift, bool a64_specialize, bool a64_row2,
    const CpuExpertDagTestHooks* test_hooks) {
    constexpr int GateTile = 16, DownTile = 32;
    constexpr int GateTiles = kIntermediateDim / GateTile;
    struct alignas(64) GateCountdown { std::atomic<int> remaining{GateTiles}; };
    GateCountdown remaining[kMaxExpertBatchTasks];
    struct ActiveTile {
        std::atomic<int>* counter;
        explicit ActiveTile(std::atomic<int>* value) : counter(value) {
            if(counter)counter->fetch_add(1,std::memory_order_relaxed);
        }
        ~ActiveTile() {if(counter)counter->fetch_sub(1,std::memory_order_relaxed);}
    };
    // A fault in one concurrent invocation must not cancel a sibling batch.
    tbb::task_group_context context(tbb::task_group_context::isolated);
    tbb::task_group group(context);
    auto down_tile = [&](int c, int begin) {
        ActiveTile active(test_hooks ? test_hooks->active_tiles : nullptr);
        const auto& slot=tasks[c].slot;
        const auto mode=a64_specialize ? a64_scale_mode(slot.down_absmax,slot.down_nested,nf4_scale_shift) : A64ScaleMode::Generic;
        const auto* values=intermediate+c*kIntermediateDim;
        const auto* scales=intermediate_scales+c*(kIntermediateDim/64);
        auto* output=out+c*kHiddenDim;
        switch(mode) {
        case A64ScaleMode::FlatF32:
            dispatch_a64_down_tile<A64ScaleMode::FlatF32>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        case A64ScaleMode::Nested256:
            dispatch_a64_down_tile<A64ScaleMode::Nested256>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        case A64ScaleMode::Nested256Register:
            dispatch_a64_down_tile<A64ScaleMode::Nested256Register>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        case A64ScaleMode::Nested256ByteShuffle:
            dispatch_a64_down_tile<A64ScaleMode::Nested256ByteShuffle>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        default:
            dispatch_a64_down_tile<A64ScaleMode::Generic>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        }
        if(a64_specialize || a64_row2)mark_a64_specialize_enabled(mode);
    };
    auto gate_tile = [&](int c, int begin) {
        ActiveTile active(test_hooks ? test_hooks->active_tiles : nullptr);
        if(test_hooks && test_hooks->fail_gate_tile==c*GateTiles+begin/GateTile)
            throw std::runtime_error("Injected CPU expert DAG gate failure");
        const auto& slot=tasks[c].slot;
        const auto mode=a64_specialize ? a64_scale_mode(slot.gate_up_absmax,slot.gate_up_nested,nf4_scale_shift) : A64ScaleMode::Generic;
        const auto* values=input+tasks[c].token*kHiddenDim;
        const auto* scales=input_scales+tasks[c].token*(kHiddenDim/64);
        auto* output=h+c*kIntermediateDim;
        switch(mode) {
        case A64ScaleMode::FlatF32:
            dispatch_a64_gate_tile<A64ScaleMode::FlatF32>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        case A64ScaleMode::Nested256:
            dispatch_a64_gate_tile<A64ScaleMode::Nested256>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        case A64ScaleMode::Nested256Register:
            dispatch_a64_gate_tile<A64ScaleMode::Nested256Register>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        case A64ScaleMode::Nested256ByteShuffle:
            dispatch_a64_gate_tile<A64ScaleMode::Nested256ByteShuffle>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        default:
            dispatch_a64_gate_tile<A64ScaleMode::Generic>(slot,begin,values,scales,lut,lut_scale,nf4_scale_shift,output,a64_row2);break;
        }
        if(a64_specialize || a64_row2)mark_a64_specialize_enabled(mode);
        // Every gate tile releases its BF16 stores. The final acq_rel RMW
        // acquires the complete expert h before quantization reads all 640 rows.
        if(remaining[c].remaining.fetch_sub(1,std::memory_order_acq_rel)==1) {
            quantize_bf16_q8_activation64(output,kIntermediateDim,
                intermediate+c*kIntermediateDim,intermediate_scales+c*(kIntermediateDim/64));
            // Publish complete quantization to independently scheduled row
            // tasks. Other experts' gate tiles need not reach a batch barrier.
            for(int begin_down=0;begin_down<kHiddenDim;begin_down+=DownTile)
                group.run([&,c,begin_down] {down_tile(c,begin_down);});
        }
    };
    try {
        // Keep row-level parallelism even when there is only one cold expert.
        for(int c=0;c<count;++c)for(int begin=0;begin<kIntermediateDim;begin+=GateTile)
            group.run([&,c,begin] {gate_tile(c,begin);});
        group.wait();
    } catch(...) {
        // A scheduling allocation or descendant error must not outlive the
        // stack scratch, borrowed sources, or caller-owned output buffers.
        group.cancel();
        try {group.wait();}catch(...) {}
        throw;
    }
}

static void nf4_q8_batch_forward(const CpuExpertTask* tasks, int count,
    const bf16* x, int tokens, bf16* h, float* out, const float* qmap, bool row4,
    bool block64, bool activation64, CpuExpertPhaseTimings* timings, bool nf4_scale_shift,
    bool a64_specialize, bool a64_row2, bool per_expert_dag,
    const CpuExpertDagTestHooks* dag_test_hooks,
    // Optional (AILA_Q4E_VERIFY_CPU_MERGE) grouped schedule: members of one
    // group iterate the same tile body back to back, so the packed weights of
    // that tile stay in L1/L2 across the group's tasks. Null keeps the exact
    // ungrouped schedule. `groups` must be non-null when group_count > 0.
    const CpuExpertGroup* groups = nullptr, const int* members = nullptr, int group_count = 0) {
    alignas(64) int8_t input[4 * kHiddenDim];
    alignas(64) float input_scales[4 * kHiddenDim / 32];
    alignas(64) int8_t intermediate[kMaxExpertBatchTasks * kIntermediateDim];
    alignas(64) float intermediate_scales[kMaxExpertBatchTasks * kIntermediateDim / 32];
    alignas(16) int8_t table[16];
    float maximum = 0.0f;
    for (int i = 0; i < 16; ++i) maximum = std::max(maximum, std::abs(qmap[i]));
    const float lut_scale = maximum / 127.0f;
    for (int i = 0; i < 16; ++i) {
        const float normalized = maximum > 0.0f ? qmap[i] * (127.0f / maximum) : 0.0f;
        table[i] = static_cast<int8_t>(_mm_cvtss_si32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(normalized),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
    }
    const auto lut = _mm_load_si128(reinterpret_cast<const __m128i*>(table));
    uint64_t phase_begin = timings ? nf4_q8_phase_ns() : 0;
    const int activation_block = activation64 ? 64 : 32;
    for (int t = 0; t < tokens; ++t) {
        if (activation64) quantize_bf16_q8_activation64(x + t * kHiddenDim, kHiddenDim,
            input + t * kHiddenDim, input_scales + t * (kHiddenDim / activation_block));
        else quantize_bf16_q8(x + t * kHiddenDim, kHiddenDim, input + t * kHiddenDim,
            input_scales + t * (kHiddenDim / activation_block));
    }
    if (timings) {
        const uint64_t now = nf4_q8_phase_ns();
        timings->quantize_ns = now - phase_begin;
        phase_begin = now;
    }
    if(per_expert_dag && activation64 && !timings && !currentTilePool) {
        if (group_count > 0) {
            // The per-expert DAG schedules one task at a time and cannot carry
            // the group map; both switches are opt-in, so keep the exact
            // ungrouped DAG bytes instead of silently rescheduling.
            nf4_q8_batch_forward(tasks, count, x, tokens, h, out, qmap, row4, block64, activation64, timings,
                nf4_scale_shift, a64_specialize, a64_row2, per_expert_dag, dag_test_hooks);
            return;
        }
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))
            std::fprintf(stderr,"[Q4E CpuExpertDagEnabled] actual=1 gate_tile=16 down_tile=32 expert_dependency=1\n");
        nf4_q8_activation64_dag_tiles(tasks,count,input,input_scales,h,out,
            intermediate,intermediate_scales,lut,lut_scale,nf4_scale_shift,a64_specialize,a64_row2,dag_test_hooks);
        return;
    }
    constexpr int gate_tile = 16, down_tile = 32;
    const int gate_tile_cols = kIntermediateDim / gate_tile;
    // Single (task, tile) body shared by the ungrouped loop and the grouped
    // (AILA_Q4E_VERIFY_CPU_MERGE) loop below. `c` owns both the activation row
    // and the output row, so every call is byte-identical to the ungrouped
    // body; only the interleaving across independent tasks changes.
    auto nf4_q8_gate_tile_body = [&](int c, int begin) {
                const auto& slot = tasks[c].slot;
                const auto* values = input + tasks[c].token * kHiddenDim;
                if (activation64 && (a64_specialize || a64_row2)) {
                    const auto* scales64 = input_scales + tasks[c].token * (kHiddenDim / 64);
                    auto* output = h + c * kIntermediateDim;
                    const auto mode = a64_specialize
                        ? a64_scale_mode(slot.gate_up_absmax, slot.gate_up_nested, nf4_scale_shift)
                        : A64ScaleMode::Generic;
                    switch (mode) {
                    case A64ScaleMode::FlatF32:
                        dispatch_a64_gate_tile<A64ScaleMode::FlatF32>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    case A64ScaleMode::Nested256:
                        dispatch_a64_gate_tile<A64ScaleMode::Nested256>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    case A64ScaleMode::Nested256Register:
                        dispatch_a64_gate_tile<A64ScaleMode::Nested256Register>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    case A64ScaleMode::Nested256ByteShuffle:
                        dispatch_a64_gate_tile<A64ScaleMode::Nested256ByteShuffle>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    default:
                        dispatch_a64_gate_tile<A64ScaleMode::Generic>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    }
                    mark_a64_specialize_enabled(mode);
                    return;
                }
                const auto* scales = input_scales + tasks[c].token * (kHiddenDim / activation_block);
                if (activation64) {
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
                    for (int row = begin; row < begin + gate_tile; ++row) {
                        const float gate = nf4_q8_dot_activation64<kHiddenDim>(slot.gate_up_packed, slot.gate_up_absmax,
                            slot.gate_up_nested, row, values, scales, lut, lut_scale, nf4_scale_shift);
                        const float up = nf4_q8_dot_activation64<kHiddenDim>(slot.gate_up_packed, slot.gate_up_absmax,
                            slot.gate_up_nested, row + kIntermediateDim, values, scales, lut, lut_scale, nf4_scale_shift);
                        const float denominator = 1.0f + std::exp(-gate);
                        const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
                        h[c * kIntermediateDim + row] = bf16(silu * up);
                    }
                } else if (row4) {
                    for (int row = begin; row < begin + gate_tile; row += 2) {
                        float dots[4];
                        nf4_q8_dot_rows4<kHiddenDim>(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested,
                            row, row + kIntermediateDim, row + 1, row + 1 + kIntermediateDim,
                            values, scales, lut, lut_scale, dots, nf4_scale_shift);
#if defined(__clang__)
                        // Preserve H8's scalar expf. With fast-math, merging
                        // these two rows selects SVML vector exp and can cross
                        // a BF16 rounding boundary on real model weights.
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
                        for (int r = 0; r < 2; ++r) {
                            const float gate = dots[r * 2], up = dots[r * 2 + 1];
                            const float denominator = 1.0f + std::exp(-gate);
                            const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
                            h[c * kIntermediateDim + row + r] = bf16(silu * up);
                        }
                    }
                } else if (block64) {
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
                    for (int row = begin; row < begin + gate_tile; ++row) {
                        const float gate = nf4_q8_dot_block64<kHiddenDim>(slot.gate_up_packed, slot.gate_up_absmax,
                            slot.gate_up_nested, row, values, scales, lut, lut_scale, nf4_scale_shift);
                        const float up = nf4_q8_dot_block64<kHiddenDim>(slot.gate_up_packed, slot.gate_up_absmax,
                            slot.gate_up_nested, row + kIntermediateDim, values, scales, lut, lut_scale, nf4_scale_shift);
                        const float denominator = 1.0f + std::exp(-gate);
                        const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
                        h[c * kIntermediateDim + row] = bf16(silu * up);
                    }
                } else for (int row = begin; row < begin + gate_tile; ++row) {
                    const float gate = nf4_q8_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested,
                        row, values, scales, kHiddenDim, lut, lut_scale, nf4_scale_shift);
                    const float up = nf4_q8_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested,
                        row + kIntermediateDim, values, scales, kHiddenDim, lut, lut_scale, nf4_scale_shift);
                    const float denominator = 1.0f + std::exp(-gate);
                    const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
                    h[c * kIntermediateDim + row] = bf16(silu * up);
                }
    };
    const int gate_chunks = nf4_q8_parallel_tiles((group_count > 0 ? group_count : count) * gate_tile_cols,
        [&](const tbb::blocked_range<int>& range) {
            for (int tile = range.begin(); tile < range.end(); ++tile) {
                const int unit = tile / gate_tile_cols;
                const int begin = tile % gate_tile_cols * gate_tile;
                if (group_count <= 0) { nf4_q8_gate_tile_body(unit, begin); continue; }
                const int member_begin = groups[unit].member_begin;
                const int member_count = groups[unit].member_count;
                for (int mi = 0; mi < member_count; ++mi)
                    nf4_q8_gate_tile_body(members[member_begin + mi], begin);
            }
        });
    if (timings) {
        const uint64_t now = nf4_q8_phase_ns();
        timings->gate_up_ns = now - phase_begin;
        timings->gate_chunks = gate_chunks;
        phase_begin = now;
    }
    // Gate tiles are 16 rows, while quantization scales span 32. Quantize only
    // after the stage barrier, so no worker reads another tile's unfinished h.
    for (int c = 0; c < count; ++c) {
        if (activation64) quantize_bf16_q8_activation64(h + c * kIntermediateDim, kIntermediateDim,
            intermediate + c * kIntermediateDim, intermediate_scales + c * (kIntermediateDim / activation_block));
        else quantize_bf16_q8(h + c * kIntermediateDim, kIntermediateDim,
            intermediate + c * kIntermediateDim, intermediate_scales + c * (kIntermediateDim / activation_block));
    }
    if (timings) {
        const uint64_t now = nf4_q8_phase_ns();
        timings->quantize_ns += now - phase_begin;
        phase_begin = now;
    }
    const int down_tile_cols = kHiddenDim / down_tile;
    auto nf4_q8_down_tile_body = [&](int c, int begin) {
                const auto& slot = tasks[c].slot;
                if (activation64 && (a64_specialize || a64_row2)) {
                    const auto* values = intermediate + c * kIntermediateDim;
                    const auto* scales64 = intermediate_scales + c * (kIntermediateDim / 64);
                    auto* output = out + c * kHiddenDim;
                    const auto mode = a64_specialize
                        ? a64_scale_mode(slot.down_absmax, slot.down_nested, nf4_scale_shift)
                        : A64ScaleMode::Generic;
                    switch (mode) {
                    case A64ScaleMode::FlatF32:
                        dispatch_a64_down_tile<A64ScaleMode::FlatF32>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    case A64ScaleMode::Nested256:
                        dispatch_a64_down_tile<A64ScaleMode::Nested256>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    case A64ScaleMode::Nested256Register:
                        dispatch_a64_down_tile<A64ScaleMode::Nested256Register>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    case A64ScaleMode::Nested256ByteShuffle:
                        dispatch_a64_down_tile<A64ScaleMode::Nested256ByteShuffle>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    default:
                        dispatch_a64_down_tile<A64ScaleMode::Generic>(slot, begin,
                            values, scales64, lut, lut_scale, nf4_scale_shift, output, a64_row2); break;
                    }
                    mark_a64_specialize_enabled(mode);
                    return;
                }
                if (activation64) {
                    for (int row = begin; row < begin + down_tile; ++row)
                        out[c * kHiddenDim + row] = nf4_q8_dot_activation64<kIntermediateDim>(slot.down_packed,
                            slot.down_absmax, slot.down_nested, row, intermediate + c * kIntermediateDim,
                            intermediate_scales + c * (kIntermediateDim / activation_block), lut, lut_scale, nf4_scale_shift);
                } else if (row4) {
                    for (int row = begin; row < begin + down_tile; row += 4)
                        nf4_q8_dot_rows4<kIntermediateDim>(slot.down_packed, slot.down_absmax, slot.down_nested,
                            row, row + 1, row + 2, row + 3,
                            intermediate + c * kIntermediateDim, intermediate_scales + c * (kIntermediateDim / 32),
                            lut, lut_scale, out + c * kHiddenDim + row, nf4_scale_shift);
                } else if (block64) {
                    for (int row = begin; row < begin + down_tile; ++row)
                        out[c * kHiddenDim + row] = nf4_q8_dot_block64<kIntermediateDim>(slot.down_packed,
                            slot.down_absmax, slot.down_nested, row, intermediate + c * kIntermediateDim,
                            intermediate_scales + c * (kIntermediateDim / 32), lut, lut_scale, nf4_scale_shift);
                } else for (int row = begin; row < begin + down_tile; ++row)
                    out[c * kHiddenDim + row] = nf4_q8_dot(slot.down_packed, slot.down_absmax, slot.down_nested,
                        row, intermediate + c * kIntermediateDim, intermediate_scales + c * (kIntermediateDim / 32),
                        kIntermediateDim, lut, lut_scale, nf4_scale_shift);
    };
    const int down_chunks = nf4_q8_parallel_tiles((group_count > 0 ? group_count : count) * down_tile_cols,
        [&](const tbb::blocked_range<int>& range) {
            for (int tile = range.begin(); tile < range.end(); ++tile) {
                const int unit = tile / down_tile_cols;
                const int begin = tile % down_tile_cols * down_tile;
                if (group_count <= 0) { nf4_q8_down_tile_body(unit, begin); continue; }
                const int member_begin = groups[unit].member_begin;
                const int member_count = groups[unit].member_count;
                for (int mi = 0; mi < member_count; ++mi)
                    nf4_q8_down_tile_body(members[member_begin + mi], begin);
            }
        });
    if (timings) {
        timings->down_ns = nf4_q8_phase_ns() - phase_begin;
        timings->down_chunks = down_chunks;
    }
}

static void gpu_order_gate_up(const MoeExpertSlotHost& slot, const bf16* x, bf16* h,
                              int tokens, const float* quant_map, int in_features, int intermediate_size, int blocksize) {
    if (!slot.gate_up_packed || !x || !h || tokens <= 0) return;
    if (tokens < 1 || tokens > 4 || in_features != kHiddenDim || intermediate_size != kIntermediateDim || blocksize != 64)
        throw std::invalid_argument("GpuOrderAvx2 supports only H=2560 I=640 block=64 m=1..4");
    alignas(64) float input[4 * kHiddenDim];
    convert_bf16_to_f32(x, input, tokens * in_features);
    const auto* qmap = resolve_quant_map(quant_map);
    tbb::parallel_for(tbb::blocked_range<int>(0, intermediate_size, 16), [&](const tbb::blocked_range<int>& range) {
        for (int row = range.begin(); row < range.end(); ++row) {
            for (int t = 0; t < tokens; ++t) {
                const float* values = input + t * in_features;
                const float gate = gpu_order_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested,
                                                 row, values, in_features, qmap);
                const float up = gpu_order_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested,
                                               row + intermediate_size, values, in_features, qmap);
                // Explicit division avoids host fast-math reciprocal reassociation.
                const float denominator = 1.0f + std::exp(-gate);
                const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
                h[t * intermediate_size + row] = bf16(silu * up);
            }
        }
    });
}

static void gpu_order_down(const MoeExpertSlotHost& slot, const bf16* h, float* out, int tokens,
                           const float* quant_map, float weight, bool accumulate,
                           int intermediate_size, int out_features, int blocksize) {
    if (!slot.down_packed || !h || !out || tokens <= 0) return;
    if (tokens < 1 || tokens > 4 || out_features != kHiddenDim || intermediate_size != kIntermediateDim || blocksize != 64)
        throw std::invalid_argument("GpuOrderAvx2 supports only H=2560 I=640 block=64 m=1..4");
    if (weight == 0.0f) {
        if (!accumulate) std::memset(out, 0, static_cast<size_t>(tokens) * out_features * sizeof(float));
        return;
    }
    alignas(64) float input[4 * kIntermediateDim];
    convert_bf16_to_f32(h, input, tokens * intermediate_size);
    const auto* qmap = resolve_quant_map(quant_map);
    tbb::parallel_for(tbb::blocked_range<int>(0, out_features, 32), [&](const tbb::blocked_range<int>& range) {
        for (int row = range.begin(); row < range.end(); ++row) {
            for (int t = 0; t < tokens; ++t) {
                const float sum = gpu_order_dot(slot.down_packed, slot.down_absmax, slot.down_nested,
                                                row, input + t * intermediate_size, intermediate_size, qmap);
                const int idx = t * out_features + row;
                if (accumulate) out[idx] += weight * sum;
                else out[idx] = weight * sum;
            }
        }
    });
}

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
    if (!slot.gate_up_packed || !x || !h) return;
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || blocksize <= 0) return;
    if (tokens > kMaxStackTokens || in_features != kHiddenDim || intermediate_size != kIntermediateDim || blocksize != kBlockSize) {
        // Fallback to scalar for large batch sizes beyond stack allocation or non-standard dimensions
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
    if (!slot.down_packed || !h || !out) return;
    if (tokens <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (accumulate && weight == 0.0f) return;
    if (!accumulate && weight == 0.0f) {
        std::memset(out, 0, static_cast<size_t>(tokens) * out_features * sizeof(float));
        return;
    }
    if (tokens > kMaxStackTokens || intermediate_size != kIntermediateDim || out_features != kHiddenDim || blocksize != kBlockSize) {
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
    if (!slot.gate_up_packed || !slot.down_packed || !x || !out) return;
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (accumulate && weight == 0.0f) return;

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
    if (!slot.gate_up_packed || !x || !h) return;
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || blocksize <= 0) return;
    if (tokens > kMaxStackTokens || in_features != kHiddenDim || intermediate_size != kIntermediateDim || blocksize != kBlockSize) {
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
    if (!slot.down_packed || !h || !out) return;
    if (tokens <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (accumulate && weight == 0.0f) return;
    if (!accumulate && weight == 0.0f) {
        std::memset(out, 0, static_cast<size_t>(tokens) * out_features * sizeof(float));
        return;
    }
    if (tokens > kMaxStackTokens || intermediate_size != kIntermediateDim || out_features != kHiddenDim || blocksize != kBlockSize) {
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
    if (!slot.gate_up_packed || !slot.down_packed || !x || !out) return;
    if (tokens <= 0 || in_features <= 0 || intermediate_size <= 0 || out_features <= 0 || blocksize <= 0) return;
    if (accumulate && weight == 0.0f) return;

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

void cpu_expert_batch_forward(const CpuExpertTask* tasks, int count,
    const bf16* x, int tokens, bf16* h, float* out, const float* quant_map, CpuKernelBackend backend,
    CpuExpertPhaseTimings* timings, bool nf4_scale_shift, std::optional<bool> a64_specialize_override,
    std::optional<bool> a64_row2_override, std::optional<bool> per_expert_dag_override,
    const CpuExpertDagTestHooks* dag_test_hooks) {
    if (timings) *timings = {};
    if (count < 0 || count > kMaxExpertBatchTasks || tokens < 1 || tokens > 4)
        throw std::invalid_argument("CPU expert batch exceeds m=4/tasks=40 capacity");
    if (count == 0) return;
    if (!tasks || !x || !h || !out) throw std::invalid_argument("Null CPU expert batch buffer");
    for (int c = 0; c < count; ++c)
        if (tasks[c].token < 0 || tasks[c].token >= tokens || !tasks[c].slot.gate_up_packed || !tasks[c].slot.down_packed)
            throw std::invalid_argument("Invalid CPU expert batch task");
#if defined(__AVX2__)
    if (backend == CpuKernelBackend::Nf4Q8Activation64Avx2 || backend == CpuKernelBackend::Nf4Q8Avx2 || backend == CpuKernelBackend::Nf4Q8Row4Avx2 ||
        backend == CpuKernelBackend::Nf4Q8Block64Avx2) {
        static const bool a64_specialize = aila::env::read_flag("AILA_Q4E_CPU_A64_SPECIALIZE", true);
        static const bool a64_row2 = aila::env::read_flag("AILA_Q4E_CPU_A64_ROW2", true);
        static const bool per_expert_dag = aila::env::read_flag("AILA_Q4E_CPU_PER_EXPERT_DAG", false);
        nf4_q8_batch_forward(tasks, count, x, tokens, h, out, resolve_quant_map(quant_map),
            backend == CpuKernelBackend::Nf4Q8Row4Avx2, backend == CpuKernelBackend::Nf4Q8Block64Avx2,
            backend == CpuKernelBackend::Nf4Q8Activation64Avx2, timings, nf4_scale_shift,
            a64_specialize_override.value_or(a64_specialize), a64_row2_override.value_or(a64_row2),
            per_expert_dag_override.value_or(per_expert_dag), dag_test_hooks);
        return;
    }
    alignas(64) float input[4 * kHiddenDim];
    alignas(64) float inter[kMaxExpertBatchTasks * kIntermediateDim];
    const bool prepared = backend == CpuKernelBackend::GpuOrderPreparedAvx2;
    if (prepared) {
        for (int t = 0; t < tokens; ++t)
            for (int col = 0; col < kHiddenDim; ++col)
                input[t * kHiddenDim + lane_input_index(col)] = static_cast<float>(x[t * kHiddenDim + col]);
    } else convert_bf16_to_f32(x, input, tokens * kHiddenDim);
    const float* qmap = resolve_quant_map(quant_map);
    const bool ordered = prepared || backend == CpuKernelBackend::GpuOrderAvx2;
    if (!ordered && backend != CpuKernelBackend::Avx2 && backend != CpuKernelBackend::TbbAvx2 && backend != CpuKernelBackend::Auto)
        throw std::invalid_argument("CPU batch supports AVX2 or GPU-order arithmetic");
    constexpr int gate_tile = 16, down_tile = 32;
    tbb::parallel_for(tbb::blocked_range<int>(0, count * (kIntermediateDim / gate_tile), 1),
        [&](const tbb::blocked_range<int>& range) {
            for (int tile = range.begin(); tile < range.end(); ++tile) {
                const int c = tile / (kIntermediateDim / gate_tile);
                const int begin = tile % (kIntermediateDim / gate_tile) * gate_tile;
                const auto& slot = tasks[c].slot;
                const auto* values = input + tasks[c].token * kHiddenDim;
                auto* result = h + c * kIntermediateDim;
                if (!ordered) {
                    cpu_expert_gate_up_swiglu_avx2_impl(slot, values, result, begin, begin + gate_tile, 1, qmap, kHiddenDim, kIntermediateDim, 64);
                } else for (int row = begin; row < begin + gate_tile; ++row) {
                    const float gate = prepared
                        ? gpu_order_dot_prepared(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested, row, values, kHiddenDim, qmap)
                        : gpu_order_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested, row, values, kHiddenDim, qmap);
                    const float up = prepared
                        ? gpu_order_dot_prepared(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested, row + kIntermediateDim, values, kHiddenDim, qmap)
                        : gpu_order_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested, row + kIntermediateDim, values, kHiddenDim, qmap);
                    const float denominator = 1.0f + std::exp(-gate);
                    const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
                    result[row] = bf16(silu * up);
                }
                if (prepared) {
                    for (int row = begin; row < begin + gate_tile; ++row)
                        inter[c * kIntermediateDim + lane_input_index(row)] = static_cast<float>(result[row]);
                } else convert_bf16_to_f32(result + begin, inter + c * kIntermediateDim + begin, gate_tile);
            }
        });
    tbb::parallel_for(tbb::blocked_range<int>(0, count * (kHiddenDim / down_tile), 1),
        [&](const tbb::blocked_range<int>& range) {
            for (int tile = range.begin(); tile < range.end(); ++tile) {
                const int c = tile / (kHiddenDim / down_tile);
                const int begin = tile % (kHiddenDim / down_tile) * down_tile;
                const auto& slot = tasks[c].slot;
                const auto* values = inter + c * kIntermediateDim;
                auto* result = out + c * kHiddenDim;
                if (!ordered) {
                    cpu_expert_down_avx2_impl(slot, values, result, begin, begin + down_tile, 1, qmap, 1.0f, false, kIntermediateDim, kHiddenDim, 64);
                } else for (int row = begin; row < begin + down_tile; ++row)
                    result[row] = prepared
                        ? gpu_order_dot_prepared(slot.down_packed, slot.down_absmax, slot.down_nested, row, values, kIntermediateDim, qmap)
                        : gpu_order_dot(slot.down_packed, slot.down_absmax, slot.down_nested, row, values, kIntermediateDim, qmap);
            }
        });
#else
    throw std::runtime_error("CPU expert batch requires AVX2/FMA");
#endif
}

void cpu_expert_batch_forward_groups(const CpuExpertGroup* groups, int group_count,
    const int* members, const CpuExpertTask* tasks, int pair_count,
    const bf16* x, int tokens, bf16* h, float* out, const float* quant_map,
    CpuKernelBackend backend, CpuExpertPhaseTimings* timings, bool nf4_scale_shift) {
    if (timings) *timings = {};
    if (!groups || !members || !tasks || !x || !h || !out)
        throw std::invalid_argument("Null grouped CPU expert batch buffer");
    if (group_count < 1 || group_count > kMaxExpertBatchTasks)
        throw std::invalid_argument("Grouped CPU expert batch count out of range");
    if (pair_count < group_count || pair_count > kMaxExpertBatchTasks)
        throw std::invalid_argument("Grouped CPU expert pair count out of range");
    if (tokens < 1 || tokens > 4)
        throw std::invalid_argument("CPU expert batch exceeds m=4 capacity");
    alignas(64) std::array<int, kMaxExpertBatchTasks> seen{};
    int covered = 0;
    for (int g = 0; g < group_count; ++g) {
        const int member_begin = groups[g].member_begin, member_count = groups[g].member_count;
        if (!groups[g].slot.gate_up_packed || !groups[g].slot.down_packed)
            throw std::invalid_argument("Invalid grouped CPU expert slot");
        if (member_begin < 0 || member_count < 1 || member_count > 4 || member_begin + member_count > pair_count)
            throw std::invalid_argument("Invalid grouped CPU expert member span");
        for (int mi = 0; mi < member_count; ++mi) {
            const int p = members[member_begin + mi];
            if (p < 0 || p >= pair_count || seen[p])
                throw std::invalid_argument("Grouped CPU expert members must cover each pair exactly once");
            seen[p] = 1; ++covered;
            if (tasks[p].token < 0 || tasks[p].token >= tokens)
                throw std::invalid_argument("Invalid grouped CPU expert task token");
        }
    }
    if (covered != pair_count)
        throw std::invalid_argument("Grouped CPU expert members must cover each pair exactly once");
#if defined(__AVX2__)
    const bool nf4_q8_family = backend == CpuKernelBackend::Nf4Q8Avx2 ||
        backend == CpuKernelBackend::Nf4Q8Row4Avx2 ||
        backend == CpuKernelBackend::Nf4Q8Block64Avx2 ||
        backend == CpuKernelBackend::Nf4Q8Activation64Avx2;
    if (!nf4_q8_family) {
        // Grouping is implemented for the NF4-Q8 family (the production cold
        // service). Every other backend keeps its exact ungrouped bytes.
        cpu_expert_batch_forward(tasks, pair_count, x, tokens, h, out, quant_map, backend, timings, nf4_scale_shift);
        return;
    }
    static const bool a64_specialize = aila::env::read_flag("AILA_Q4E_CPU_A64_SPECIALIZE", true);
    static const bool a64_row2 = aila::env::read_flag("AILA_Q4E_CPU_A64_ROW2", true);
    static const bool per_expert_dag = aila::env::read_flag("AILA_Q4E_CPU_PER_EXPERT_DAG", false);
    nf4_q8_batch_forward(tasks, pair_count, x, tokens, h, out, resolve_quant_map(quant_map),
        backend == CpuKernelBackend::Nf4Q8Row4Avx2, backend == CpuKernelBackend::Nf4Q8Block64Avx2,
        backend == CpuKernelBackend::Nf4Q8Activation64Avx2, timings, nf4_scale_shift,
        a64_specialize, a64_row2, per_expert_dag, nullptr, groups, members, group_count);
#else
    throw std::runtime_error("CPU expert batch requires AVX2/FMA");
#endif
}

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
        case CpuKernelBackend::Nf4Q8Activation64Avx2:
        case CpuKernelBackend::Nf4Q8Block64Avx2:
        case CpuKernelBackend::Nf4Q8Row4Avx2:
        case CpuKernelBackend::Nf4Q8Avx2:
        case CpuKernelBackend::GpuOrderPreparedAvx2:
            throw std::invalid_argument("GpuOrderPreparedAvx2 requires cpu_expert_batch_forward");
        case CpuKernelBackend::GpuOrderAvx2:
#if defined(__AVX2__)
            gpu_order_gate_up(slot, x, h, tokens, quant_map, in_features, intermediate_size, blocksize);
            break;
#else
            throw std::runtime_error("GpuOrderAvx2 requires AVX2/FMA");
#endif
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
        case CpuKernelBackend::Nf4Q8Activation64Avx2:
        case CpuKernelBackend::Nf4Q8Block64Avx2:
        case CpuKernelBackend::Nf4Q8Row4Avx2:
        case CpuKernelBackend::Nf4Q8Avx2:
        case CpuKernelBackend::GpuOrderPreparedAvx2:
            throw std::invalid_argument("GpuOrderPreparedAvx2 requires cpu_expert_batch_forward");
        case CpuKernelBackend::GpuOrderAvx2:
#if defined(__AVX2__)
            gpu_order_down(slot, h, out, tokens, quant_map, weight, accumulate, intermediate_size, out_features, blocksize);
            break;
#else
            throw std::runtime_error("GpuOrderAvx2 requires AVX2/FMA");
#endif
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
        case CpuKernelBackend::Nf4Q8Activation64Avx2:
        case CpuKernelBackend::Nf4Q8Block64Avx2:
        case CpuKernelBackend::Nf4Q8Row4Avx2:
        case CpuKernelBackend::Nf4Q8Avx2:
        case CpuKernelBackend::GpuOrderPreparedAvx2:
            throw std::invalid_argument("GpuOrderPreparedAvx2 requires cpu_expert_batch_forward");
        case CpuKernelBackend::GpuOrderAvx2: {
            if (!slot.gate_up_packed || !slot.down_packed || !x || !out || tokens <= 0) return;
            alignas(64) bf16 local_h[4 * kIntermediateDim];
            bf16* effective_h = scratch_h ? scratch_h : local_h;
            cpu_expert_gate_up_swiglu(slot, x, effective_h, tokens, quant_map, in_features, intermediate_size, blocksize, backend);
            cpu_expert_down(slot, effective_h, out, tokens, quant_map, weight, accumulate, intermediate_size, out_features, blocksize, backend);
            break;
        }
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
