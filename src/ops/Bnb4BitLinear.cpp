#include "Bnb4BitLinear.hpp"
#include "Ops.hpp"
#include "../utils/EnvUtils.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

namespace {

int round_up_seq(int value, int granularity) {
    return ((value + granularity - 1) / granularity) * granularity;
}

// Fused split + SiLU(gate)*up for a [rows, 2*ff_dim] gate_up output.  Reads
// gate/up halves directly from the fused matmul output and writes the
// activated projection, producing exactly the same values as the two-kernel
// split + gelu path while saving one launch and two intermediate buffers.
void moe_gelu_tanh_gate_up_mul(Context& ctx, Tensor& gate_up,
                               Tensor& output, int rows, int ff_dim) {
    bf16* src_ptr = static_cast<bf16*>(gate_up.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());
    constexpr float kAlpha = 0.7978845608028654f;
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(rows) * ff_dim),
        [=](sycl::id<1> idx) {
            const int row = static_cast<int>(idx[0] / ff_dim);
            const int col = static_cast<int>(idx[0] % ff_dim);
            const float g = static_cast<float>(
                src_ptr[static_cast<size_t>(row) * 2 * ff_dim + col]);
            const float u = static_cast<float>(
                src_ptr[static_cast<size_t>(row) * 2 * ff_dim + ff_dim + col]);
            const float v = kAlpha * (g + 0.044715f * g * g * g);
            out_ptr[idx[0]] = bf16(0.5f * g * (1.0f + sycl::tanh(v)) * u);
        });
}

void moe_gather_rows(Context& ctx, const bf16* input_ptr, bf16* src_ptr,
                     const int* token_ids, int rows, int hidden) {
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(rows) * hidden),
        [=](sycl::id<1> idx) {
            const int r = static_cast<int>(idx[0] / hidden);
            const int h = static_cast<int>(idx[0] % hidden);
            src_ptr[idx[0]] =
                input_ptr[static_cast<size_t>(token_ids[r]) * hidden + h];
        });
}

void moe_gather_rows_flat(Context& ctx, const bf16* input_ptr,
                          bf16* dst_ptr, const int* token_ids,
                          int total_rows, int hidden) {
    if (hidden % 8 == 0) {
        const int hidden_vec = hidden / 8;
        ctx.queue().parallel_for(
            sycl::range<1>(static_cast<size_t>(total_rows) * hidden_vec),
            [=](sycl::id<1> idx) {
                const int r = static_cast<int>(idx[0] / hidden_vec);
                const int h_vec = static_cast<int>(idx[0] % hidden_vec);
                const int t = token_ids[r];
                const size_t in_offset = static_cast<size_t>(t) * hidden + h_vec * 8;
                const size_t dst_offset = static_cast<size_t>(r) * hidden + h_vec * 8;
                *reinterpret_cast<sycl::vec<bf16, 8>*>(dst_ptr + dst_offset) =
                    *reinterpret_cast<const sycl::vec<bf16, 8>*>(input_ptr + in_offset);
            });
        return;
    }
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(total_rows) * hidden),
        [=](sycl::id<1> idx) {
            const int r = static_cast<int>(idx[0] / hidden);
            const int h = static_cast<int>(idx[0] % hidden);
            dst_ptr[idx[0]] =
                input_ptr[static_cast<size_t>(token_ids[r]) * hidden + h];
        });
}

// Round-to-nearest-even float -> bf16 as a float with the low 16 mantissa
// bits cleared.  Keeps the intermediate bf16 rounding of the product explicit
// so the device compiler cannot fold the round-trip under -ffast-math.
[[gnu::noinline]] float bf16_round_f32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    const uint32_t rounded =
        (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    float result = 0.0f;
    std::memcpy(&result, &rounded, 4);
    return result;
}

// Exact bf16 -> float widening through explicit bit construction.  A direct
// static_cast<float>(bf16 load) * float on this device compiles to a fused
// bf16-multiply instruction whose product can round one float ulp below the
// IEEE host multiply; widening the bits first keeps the standard f32 path.
inline float bf16_to_f32_raw(bf16 value) {
    uint16_t raw = 0;
    std::memcpy(&raw, &value, 2);
    const uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, 4);
    return result;
}

// Mirrors the host accumulation in Bnb4GroupedMoE::forward() exactly:
//   contribution = bf16(float(down_out[r][h]) * weight[r])
//   acc[token][h] = bf16(float(acc[token][h]) + float(contribution))
// Groups are processed in ascending expert order by the in-order queue, and
// a token appears at most once per group, so no element races across kernels.
void moe_accumulate_rows(Context& ctx, const bf16* src_ptr, bf16* acc_ptr,
                         const int* token_ids, const float* weights,
                         int rows, int hidden) {
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(rows) * hidden),
        [=](sycl::id<1> idx) {
            const int r = static_cast<int>(idx[0] / hidden);
            const int h = static_cast<int>(idx[0] % hidden);
            bf16* dst =
                acc_ptr + static_cast<size_t>(token_ids[r]) * hidden + h;
            const float product =
                bf16_to_f32_raw(src_ptr[idx[0]]) * weights[r];
            const float contribution = bf16_round_f32(product);
            const float sum = static_cast<float>(*dst) + contribution;
            *dst = bf16(bf16_round_f32(sum));
        });
}

// Fused NF4 dequant + matmul for prefill (M > 1).
// Computes C[M,N] = A[M,K] @ dequantize_nf4(B_packed[N,K]), B absmax per block.
template <int BM, int BN, int BK, int TM, int TN, int TILE_M, int TILE_N>
void packed_nf4_gemm_bf16_tiled(Context& ctx,
                                const uint8_t* packed_ptr,
                                const float* quant_map_ptr,
                                const float* absmax_ptr,
                                const bf16* input_ptr,
                                bf16* output_ptr,
                                int M, int N, int K,
                                int blocksize) {
    const int packed_bytes_per_row = K / 2;
    const int blocks_per_row = K / blocksize;
    const int blocks_per_tile = BK / blocksize;

    constexpr int WG_SIZE = TILE_M * TILE_N;

    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        sycl::local_accessor<float, 1> B_absmax_slm(
            sycl::range<1>(BN * blocks_per_tile), cgh);
        sycl::local_accessor<uint8_t, 1> B_nf4_slm(sycl::range<1>(BK * BN / 2), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(grid_n) * TILE_N,
                                             static_cast<size_t>(grid_m) * TILE_M),
                              sycl::range<2>(TILE_N, TILE_M)),
            [=](sycl::nd_item<2> item) {
                const int tx = static_cast<int>(item.get_local_id(0));  // N dimension
                const int ty = static_cast<int>(item.get_local_id(1));  // M dimension
                const int lid = ty * TILE_N + tx;

                const int block_n = static_cast<int>(item.get_group(0));
                const int block_m = static_cast<int>(item.get_group(1));
                const int n0 = block_n * BN;
                const int m0 = block_m * BM;

                if (lid < 16) qmap[lid] = quant_map_ptr[lid];

                // Thread-local accumulators for output tile elements
                float C[TM][TN] = {};

                for (int kb = 0; kb < K; kb += BK) {
                    // Cooperative load A[BM, BK] into SLM
                    const int a_rows = sycl::min(BM, M - m0);
                    const int a_cols = sycl::min(BK, K - kb);
                    const int a_elems = a_rows * a_cols;
                    for (int i = lid; i < a_elems; i += WG_SIZE) {
                        const int r = i / a_cols;
                        const int c = i % a_cols;
                        As[i] = input_ptr[(m0 + r) * K + (kb + c)];
                    }

                    // Cooperative load B NF4 bytes for [BK, BN] tile
                    const int b_bytes = a_cols * BN / 2;
                    for (int i = lid; i < b_bytes; i += WG_SIZE) {
                        const int k_half = i / BN;
                        const int bn_idx = i % BN;
                        const int src_n = n0 + bn_idx;
                        if (src_n < N) {
                            B_nf4_slm[i] = packed_ptr[src_n * packed_bytes_per_row + kb / 2 + k_half];
                        } else {
                            B_nf4_slm[i] = 0x77;
                        }
                    }

                    // Load absmax for each (N column, K block) in this tile.
                    const int absmax_elems = BN * blocks_per_tile;
                    for (int i = lid; i < absmax_elems; i += WG_SIZE) {
                        const int ni = i / blocks_per_tile;
                        const int b = i % blocks_per_tile;
                        const int src_n = n0 + ni;
                        B_absmax_slm[i] = (src_n < N)
                            ? absmax_ptr[src_n * blocks_per_row + kb / blocksize + b]
                            : 0.0f;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Compute partial products for this K tile
                    const int m_local = ty * TM;
                    const int n_local = tx * TN;
                    const int m_local_end = sycl::min(m_local + TM, a_rows);
                    const int n_local_end = sycl::min(n_local + TN, BN);

                    for (int k = 0; k < a_cols; ++k) {
                        const int b_byte_base = (k / 2) * BN;
                        const bool hi = (k % 2 == 0);

                        // Pre-dequantize B values for this k for our N columns
                        float b_dequant[TN];
                        for (int ni = n_local; ni < n_local_end; ++ni) {
                            const uint8_t b = B_nf4_slm[b_byte_base + ni];
                            const uint8_t nf4 = hi ? (b >> 4) : (b & 0x0F);
                            // bitsandbytes dequantize_4bit materializes the
                            // requested compute dtype before matmul.  The
                            // checkpoint requests bf16, so round each NF4
                            // weight before accumulation instead of retaining
                            // extra fp32 precision in the fused path.
                            b_dequant[ni - n_local] = static_cast<float>(bf16(
                                qmap[nf4] * B_absmax_slm[
                                    ni * blocks_per_tile + k / blocksize]));
                        }

                        for (int mi = m_local; mi < m_local_end; ++mi) {
                            const float av = static_cast<float>(As[mi * a_cols + k]);
                            for (int ni = n_local; ni < n_local_end; ++ni) {
                                C[mi - m_local][ni - n_local] += av * b_dequant[ni - n_local];
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Write output
                for (int mi = 0; mi < TM; ++mi) {
                    const int m = m0 + ty * TM + mi;
                    if (m >= M || m - m0 >= BM) continue;
                    for (int ni = 0; ni < TN; ++ni) {
                        const int n = n0 + tx * TN + ni;
                        if (n >= N || n - n0 >= BN) continue;
                        output_ptr[m * N + n] = bf16(C[mi][ni]);
                    }
                }
            });
    });
}

// Fused NF4 dequant + matmul for prefill (M > 1), vectorized variant.
// Computes C[M,N] = A[M,K] @ dequantize_nf4(B_packed[N,K]) with the same
// per-element numerics as packed_nf4_gemm_bf16_tiled:
//   - each NF4 weight is rounded to bf16 once as bf16(qmap[code] * absmax);
//   - C accumulates in float over k in ascending order;
//   - C is rounded to bf16 on store.
// The B tile is dequantized into a bf16 SLM tile once per K block, then the
// inner loop reads vec8 bf16 A/B slices and performs the same float FMAs in
// the same k order.
template <int BM, int BN, int BK, int TM, int TN, int TILE_M, int TILE_N,
          bool SWIZZLE_B = false>
void packed_nf4_gemm_bf16_tiled_v2(Context& ctx,
                                   const uint8_t* packed_ptr,
                                   const float* quant_map_ptr,
                                   const float* absmax_ptr,
                                   const bf16* input_ptr,
                                   bf16* output_ptr,
                                   int M, int N, int K,
                                   int blocksize) {
    const int packed_bytes_per_row = K / 2;
    const int blocks_per_row = K / blocksize;

    constexpr int WG_SIZE = TILE_M * TILE_N;

    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(BN * BK), cgh);
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(grid_n) * TILE_N,
                                             static_cast<size_t>(grid_m) * TILE_M),
                              sycl::range<2>(TILE_N, TILE_M)),
            [=](sycl::nd_item<2> item) {
                using bvec8 = sycl::vec<bf16, 8>;
                const int tx = static_cast<int>(item.get_local_id(0));  // N dimension
                const int ty = static_cast<int>(item.get_local_id(1));  // M dimension
                const int lid = ty * TILE_N + tx;

                const int block_n = static_cast<int>(item.get_group(0));
                const int block_m = static_cast<int>(item.get_group(1));
                const int n0 = block_n * BN;
                const int m0 = block_m * BM;

                if (lid < 16) qmap[lid] = quant_map_ptr[lid];

                bf16* As_ptr = As.get_multi_ptr<sycl::access::decorated::no>().get();
                bf16* Bs_ptr = Bs.get_multi_ptr<sycl::access::decorated::no>().get();

                // Thread-local accumulators for output tile elements
                float C[TM][TN] = {};

                for (int kb = 0; kb < K; kb += BK) {
                    // Cooperative load A[BM, BK] into SLM.
                    const int a_rows = sycl::min(BM, M - m0);
                    const int a_cols = sycl::min(BK, K - kb);
                    if (a_cols % 8 == 0) {
                        const int a_vecs = a_rows * (a_cols / 8);
                        for (int i = lid; i < a_vecs; i += WG_SIZE) {
                            const int r = i / (a_cols / 8);
                            const int c8 = i % (a_cols / 8);
                            const bvec8 v = *reinterpret_cast<const bvec8*>(
                                input_ptr + static_cast<size_t>(m0 + r) * K + kb + c8 * 8);
                            *reinterpret_cast<bvec8*>(As_ptr + r * a_cols + c8 * 8) = v;
                        }
                    } else {
                        const int a_elems = a_rows * a_cols;
                        for (int i = lid; i < a_elems; i += WG_SIZE) {
                            const int r = i / a_cols;
                            const int c = i % a_cols;
                            As_ptr[i] = input_ptr[static_cast<size_t>(m0 + r) * K + kb + c];
                        }
                    }

                    // Cooperative load + dequant B[BK, BN] tile into SLM as
                    // bf16 [BN, a_cols]. Each uint32 word holds 8 NF4 values
                    // that all share one absmax block (blocksize >= 32 and
                    // 4-byte words never straddle a block boundary).
                    if (a_cols % 8 == 0) {
                        const int b_words = BN * (a_cols / 8);
                        for (int i = lid; i < b_words; i += WG_SIZE) {
                            const int ni = i / (a_cols / 8);
                            const int kv = i % (a_cols / 8);
                            const int src_n = n0 + ni;
                            if (src_n < N) {
                                const uint32_t w = *reinterpret_cast<const uint32_t*>(
                                    packed_ptr + static_cast<size_t>(src_n) * packed_bytes_per_row +
                                    kb / 2 + kv * 4);
                                const float am = absmax_ptr[
                                    static_cast<size_t>(src_n) * blocks_per_row +
                                    (kb + kv * 8) / blocksize];
                                const uint8_t b0 = static_cast<uint8_t>(w);
                                const uint8_t b1 = static_cast<uint8_t>(w >> 8);
                                const uint8_t b2 = static_cast<uint8_t>(w >> 16);
                                const uint8_t b3 = static_cast<uint8_t>(w >> 24);
                                bvec8 vals;
                                vals[0] = bf16(qmap[b0 >> 4] * am);
                                vals[1] = bf16(qmap[b0 & 0x0F] * am);
                                vals[2] = bf16(qmap[b1 >> 4] * am);
                                vals[3] = bf16(qmap[b1 & 0x0F] * am);
                                vals[4] = bf16(qmap[b2 >> 4] * am);
                                vals[5] = bf16(qmap[b2 & 0x0F] * am);
                                vals[6] = bf16(qmap[b3 >> 4] * am);
                                vals[7] = bf16(qmap[b3 & 0x0F] * am);
                                *reinterpret_cast<bvec8*>(Bs_ptr + ni * a_cols + kv * 8) = vals;
                            } else {
                                bvec8 zero{};
                                *reinterpret_cast<bvec8*>(Bs_ptr + ni * a_cols + kv * 8) = zero;
                            }
                        }
                    } else {
                        const int b_bytes = BN * (a_cols / 2);
                        for (int i = lid; i < b_bytes; i += WG_SIZE) {
                            const int ni = i / (a_cols / 2);
                            const int kh = i % (a_cols / 2);
                            const int src_n = n0 + ni;
                            const int kk = kh * 2;
                            const uint8_t packed_byte = (src_n < N)
                                ? packed_ptr[static_cast<size_t>(src_n) * packed_bytes_per_row + kb / 2 + kh]
                                : 0;
                            const float am = (src_n < N)
                                ? absmax_ptr[static_cast<size_t>(src_n) * blocks_per_row +
                                             (kb + kk) / blocksize]
                                : 0.0f;
                            Bs_ptr[ni * a_cols + kk] =
                                bf16(qmap[packed_byte >> 4] * am);
                            if (kk + 1 < a_cols) {
                                Bs_ptr[ni * a_cols + kk + 1] =
                                    bf16(qmap[packed_byte & 0x0F] * am);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Compute partial products for this K tile.
                    const int m_local = ty * TM;
                    const int n_local = tx * TN;
                    const int m_local_end = sycl::min(m_local + TM, a_rows);
                    const int n_local_end = sycl::min(n_local + TN, BN);

                    int k = 0;
                    for (; k + 8 <= a_cols; k += 8) {
                        bvec8 bv[TN];
                        for (int ni = n_local; ni < n_local_end; ++ni) {
                            bv[ni - n_local] = *reinterpret_cast<const bvec8*>(
                                Bs_ptr + ni * a_cols + k);
                        }
                        for (int mi = m_local; mi < m_local_end; ++mi) {
                            const bvec8 av = *reinterpret_cast<const bvec8*>(
                                As_ptr + mi * a_cols + k);
                            for (int ni = n_local; ni < n_local_end; ++ni) {
                                float* acc = &C[mi - m_local][ni - n_local];
                                const bvec8& b = bv[ni - n_local];
                                #pragma unroll
                                for (int v = 0; v < 8; ++v) {
                                    *acc += static_cast<float>(av[v]) *
                                            static_cast<float>(b[v]);
                                }
                            }
                        }
                    }
                    for (; k < a_cols; ++k) {
                        for (int ni = n_local; ni < n_local_end; ++ni) {
                            const float bd = static_cast<float>(
                                Bs_ptr[ni * a_cols + k]);
                            for (int mi = m_local; mi < m_local_end; ++mi) {
                                C[mi - m_local][ni - n_local] +=
                                    static_cast<float>(As_ptr[mi * a_cols + k]) * bd;
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Write output
                for (int mi = 0; mi < TM; ++mi) {
                    const int m = m0 + ty * TM + mi;
                    if (m >= M || m - m0 >= BM) continue;
                    for (int ni = 0; ni < TN; ++ni) {
                        const int n = n0 + tx * TN + ni;
                        if (n >= N || n - n0 >= BN) continue;
                        output_ptr[m * N + n] = bf16(C[mi][ni]);
                    }
                }
            });
    });
}

// Double-buffered variant of the v2 NF4 GEMM.  A and B tiles are staged in
// two SLM buffers each; while a tile is computed, the next K tile's global
// loads/dequant are already in flight.  Per-element accumulation order and
// bf16 rounding are identical to v1/v2, so outputs stay bitwise equal; the
// barrier count drops from 2 per K tile to 1.
template <int BM, int BN, int BK, int TM, int TN, int TILE_M, int TILE_N>
void packed_nf4_gemm_bf16_tiled_v3(Context& ctx,
                                   const uint8_t* packed_ptr,
                                   const float* quant_map_ptr,
                                   const float* absmax_ptr,
                                   const bf16* input_ptr,
                                   bf16* output_ptr,
                                   int M, int N, int K,
                                   int blocksize) {
    const int packed_bytes_per_row = K / 2;
    const int blocks_per_row = K / blocksize;

    constexpr int WG_SIZE = TILE_M * TILE_N;

    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(2 * BM * BK), cgh);
        sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(2 * BN * BK), cgh);
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(grid_n) * TILE_N,
                                             static_cast<size_t>(grid_m) * TILE_M),
                              sycl::range<2>(TILE_N, TILE_M)),
            [=](sycl::nd_item<2> item) {
                using bvec8 = sycl::vec<bf16, 8>;
                const int tx = static_cast<int>(item.get_local_id(0));  // N dimension
                const int ty = static_cast<int>(item.get_local_id(1));  // M dimension
                const int lid = ty * TILE_N + tx;

                const int block_n = static_cast<int>(item.get_group(0));
                const int block_m = static_cast<int>(item.get_group(1));
                const int n0 = block_n * BN;
                const int m0 = block_m * BM;

                if (lid < 16) qmap[lid] = quant_map_ptr[lid];

                bf16* As_base = As.get_multi_ptr<sycl::access::decorated::no>().get();
                bf16* Bs_base = Bs.get_multi_ptr<sycl::access::decorated::no>().get();

                const int a_rows = sycl::min(BM, M - m0);

                auto load_tile = [&](int kb, int buf) {
                    bf16* As_ptr = As_base + buf * (BM * BK);
                    bf16* Bs_ptr = Bs_base + buf * (BN * BK);
                    const int a_cols = sycl::min(BK, K - kb);
                    if (a_cols % 8 == 0) {
                        const int a_vecs = a_rows * (a_cols / 8);
                        for (int i = lid; i < a_vecs; i += WG_SIZE) {
                            const int r = i / (a_cols / 8);
                            const int c8 = i % (a_cols / 8);
                            const bvec8 v = *reinterpret_cast<const bvec8*>(
                                input_ptr + static_cast<size_t>(m0 + r) * K + kb + c8 * 8);
                            *reinterpret_cast<bvec8*>(As_ptr + r * a_cols + c8 * 8) = v;
                        }
                    } else {
                        const int a_elems = a_rows * a_cols;
                        for (int i = lid; i < a_elems; i += WG_SIZE) {
                            const int r = i / a_cols;
                            const int c = i % a_cols;
                            As_ptr[i] = input_ptr[static_cast<size_t>(m0 + r) * K + kb + c];
                        }
                    }
                    if (a_cols % 8 == 0) {
                        const int b_words = BN * (a_cols / 8);
                        for (int i = lid; i < b_words; i += WG_SIZE) {
                            const int ni = i / (a_cols / 8);
                            const int kv = i % (a_cols / 8);
                            const int src_n = n0 + ni;
                            if (src_n < N) {
                                const uint32_t w = *reinterpret_cast<const uint32_t*>(
                                    packed_ptr + static_cast<size_t>(src_n) * packed_bytes_per_row +
                                    kb / 2 + kv * 4);
                                const float am = absmax_ptr[
                                    static_cast<size_t>(src_n) * blocks_per_row +
                                    (kb + kv * 8) / blocksize];
                                const uint8_t b0 = static_cast<uint8_t>(w);
                                const uint8_t b1 = static_cast<uint8_t>(w >> 8);
                                const uint8_t b2 = static_cast<uint8_t>(w >> 16);
                                const uint8_t b3 = static_cast<uint8_t>(w >> 24);
                                bvec8 vals;
                                vals[0] = bf16(qmap[b0 >> 4] * am);
                                vals[1] = bf16(qmap[b0 & 0x0F] * am);
                                vals[2] = bf16(qmap[b1 >> 4] * am);
                                vals[3] = bf16(qmap[b1 & 0x0F] * am);
                                vals[4] = bf16(qmap[b2 >> 4] * am);
                                vals[5] = bf16(qmap[b2 & 0x0F] * am);
                                vals[6] = bf16(qmap[b3 >> 4] * am);
                                vals[7] = bf16(qmap[b3 & 0x0F] * am);
                                *reinterpret_cast<bvec8*>(Bs_ptr + ni * a_cols + kv * 8) = vals;
                            } else {
                                bvec8 zero{};
                                *reinterpret_cast<bvec8*>(Bs_ptr + ni * a_cols + kv * 8) = zero;
                            }
                        }
                    } else {
                        const int b_bytes = BN * (a_cols / 2);
                        for (int i = lid; i < b_bytes; i += WG_SIZE) {
                            const int ni = i / (a_cols / 2);
                            const int kh = i % (a_cols / 2);
                            const int src_n = n0 + ni;
                            const int kk = kh * 2;
                            const uint8_t packed_byte = (src_n < N)
                                ? packed_ptr[static_cast<size_t>(src_n) * packed_bytes_per_row + kb / 2 + kh]
                                : 0;
                            const float am = (src_n < N)
                                ? absmax_ptr[static_cast<size_t>(src_n) * blocks_per_row +
                                             (kb + kk) / blocksize]
                                : 0.0f;
                            Bs_ptr[ni * a_cols + kk] =
                                bf16(qmap[packed_byte >> 4] * am);
                            if (kk + 1 < a_cols) {
                                Bs_ptr[ni * a_cols + kk + 1] =
                                    bf16(qmap[packed_byte & 0x0F] * am);
                            }
                        }
                    }
                };

                // Thread-local accumulators for output tile elements
                float C[TM][TN] = {};

                const int ntiles = (K + BK - 1) / BK;
                load_tile(0, 0);
                item.barrier(sycl::access::fence_space::local_space);
                int cur = 0;
                for (int t = 0; t < ntiles; ++t) {
                    const int kb = t * BK;
                    if (t + 1 < ntiles) {
                        load_tile(kb + BK, cur ^ 1);
                    }

                    // Compute partial products for this K tile from buffer cur.
                    const int a_cols = sycl::min(BK, K - kb);
                    bf16* As_ptr = As_base + cur * (BM * BK);
                    bf16* Bs_ptr = Bs_base + cur * (BN * BK);
                    const int m_local = ty * TM;
                    const int n_local = tx * TN;
                    const int m_local_end = sycl::min(m_local + TM, a_rows);
                    const int n_local_end = sycl::min(n_local + TN, BN);

                    int k = 0;
                    for (; k + 8 <= a_cols; k += 8) {
                        bvec8 bv[TN];
                        for (int ni = n_local; ni < n_local_end; ++ni) {
                            bv[ni - n_local] = *reinterpret_cast<const bvec8*>(
                                Bs_ptr + ni * a_cols + k);
                        }
                        for (int mi = m_local; mi < m_local_end; ++mi) {
                            const bvec8 av = *reinterpret_cast<const bvec8*>(
                                As_ptr + mi * a_cols + k);
                            for (int ni = n_local; ni < n_local_end; ++ni) {
                                float* acc = &C[mi - m_local][ni - n_local];
                                const bvec8& b = bv[ni - n_local];
                                #pragma unroll
                                for (int v = 0; v < 8; ++v) {
                                    *acc += static_cast<float>(av[v]) *
                                            static_cast<float>(b[v]);
                                }
                            }
                        }
                    }
                    for (; k < a_cols; ++k) {
                        for (int ni = n_local; ni < n_local_end; ++ni) {
                            const float bd = static_cast<float>(
                                Bs_ptr[ni * a_cols + k]);
                            for (int mi = m_local; mi < m_local_end; ++mi) {
                                C[mi - m_local][ni - n_local] +=
                                    static_cast<float>(As_ptr[mi * a_cols + k]) * bd;
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    cur ^= 1;
                }

                // Write output
                for (int mi = 0; mi < TM; ++mi) {
                    const int m = m0 + ty * TM + mi;
                    if (m >= M || m - m0 >= BM) continue;
                    for (int ni = 0; ni < TN; ++ni) {
                        const int n = n0 + tx * TN + ni;
                        if (n >= N || n - n0 >= BN) continue;
                        output_ptr[m * N + n] = bf16(C[mi][ni]);
                    }
                }
            });
    });
}

// [XMX Matrix Engine] Joint Matrix NF4 GEMM on Intel Arc A770 (DG2)
// SG=8 matches DPAS systolic depth 8.
// BM=32, BN=32, BK=16, TM=8, TN=8, TK=16, SG=8 -> 16 sub-groups, WG=128.
// Uses hardware-natural contiguous SLM store with stride TN=8.
template <int BM = 32, int BN = 32, int BK = 16, int TM = 8, int TN = 8, int TK = 16, int SG = 8>
void packed_nf4_gemm_bf16_joint_matrix(Context& ctx,
                                       const uint8_t* packed_ptr,
                                       const float* quant_map_ptr,
                                       const float* absmax_ptr,
                                       const bf16* input_ptr,
                                       bf16* output_ptr,
                                       int M, int N, int K,
                                       int blocksize) {
    const int packed_bytes_per_row = K / 2;
    const int blocks_per_row = K / blocksize;
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    const int total_wgs = grid_m * grid_n;

    constexpr int SG_M = BM / TM;
    constexpr int SG_N = BN / TN;
    constexpr int NUM_SGS = SG_M * SG_N;
    constexpr int WG_SIZE = NUM_SGS * SG;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(TK * BN), cgh);
        sycl::local_accessor<float, 1> c_slm(sycl::range<1>(NUM_SGS * TM * TN), cgh);
        sycl::local_accessor<float, 1> qmap_slm(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(total_wgs) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int wg_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / SG;
                const int sg_row = sg_id / SG_N;
                const int sg_col = sg_id % SG_N;

                const int bm = wg_idx / grid_n;
                const int bn = wg_idx % grid_n;
                const int m0 = bm * BM;
                const int n0 = bn * BN;

                if (lid < 16) {
                    qmap_slm[lid] = quant_map_ptr[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
                joint_matrix_fill(sg, sub_c, 0.0f);

                for (int kb = 0; kb < K; kb += BK) {
                    for (int pair_idx = lid; pair_idx < (BM * BK) / 2; pair_idx += WG_SIZE) {
                        const int r = pair_idx / (BK / 2);
                        const int c_pair = pair_idx % (BK / 2);
                        const int gm = m0 + r;
                        const int c = c_pair * 2;
                        if (gm < M && kb + c < K) {
                            auto in_u32 = reinterpret_cast<const uint32_t*>(input_ptr + static_cast<size_t>(gm) * K + kb)[c_pair];
                            reinterpret_cast<uint32_t*>(As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = in_u32;
                        } else {
                            reinterpret_cast<uint32_t*>(As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = 0;
                        }
                    }

                    for (int i = lid; i < BN * 2; i += WG_SIZE) {
                        const int col = i / 2;
                        const int half = i % 2;
                        const int src_n = n0 + col;
                        if (src_n < N && kb + half * 8 + 7 < K) {
                            const uint32_t packed_u32 = reinterpret_cast<const uint32_t*>(
                                packed_ptr + static_cast<size_t>(src_n) * packed_bytes_per_row + (kb / 2) + half * 4)[0];
                            const float am = absmax_ptr[static_cast<size_t>(src_n) * blocks_per_row + (kb + half * 8) / blocksize];

                            const uint8_t b0 = static_cast<uint8_t>(packed_u32 & 0xFF);
                            const uint8_t b1 = static_cast<uint8_t>((packed_u32 >> 8) & 0xFF);
                            const uint8_t b2 = static_cast<uint8_t>((packed_u32 >> 16) & 0xFF);
                            const uint8_t b3 = static_cast<uint8_t>((packed_u32 >> 24) & 0xFF);

                            const int base = col * TK + half * 8;
                            Bs[base + 0] = bf16(qmap_slm[(b0 >> 4) & 0x0F] * am);
                            Bs[base + 1] = bf16(qmap_slm[b0 & 0x0F] * am);
                            Bs[base + 2] = bf16(qmap_slm[(b1 >> 4) & 0x0F] * am);
                            Bs[base + 3] = bf16(qmap_slm[b1 & 0x0F] * am);
                            Bs[base + 4] = bf16(qmap_slm[(b2 >> 4) & 0x0F] * am);
                            Bs[base + 5] = bf16(qmap_slm[b2 & 0x0F] * am);
                            Bs[base + 6] = bf16(qmap_slm[(b3 >> 4) & 0x0F] * am);
                            Bs[base + 7] = bf16(qmap_slm[b3 & 0x0F] * am);
                        } else {
                            const int base = col * TK + half * 8;
                            for (int v = 0; v < 8; ++v) {
                                Bs[base + v] = bf16(0.0f);
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;
                    joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major> sub_b;

                    auto pA = As.get_multi_ptr<sycl::access::decorated::no>() + (sg_row * TM) * BK;
                    auto pB = Bs.get_multi_ptr<sycl::access::decorated::no>() + (sg_col * TN) * TK;

                    joint_matrix_load(sg, sub_a, pA, BK);
                    joint_matrix_load(sg, sub_b, pB, TK);
                    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);

                    item.barrier(sycl::access::fence_space::local_space);
                }

                auto pC_slm = c_slm.get_multi_ptr<sycl::access::decorated::no>() + sg_id * (TM * TN);
                joint_matrix_store(sg, sub_c, pC_slm, TN, layout::row_major);
                item.barrier(sycl::access::fence_space::local_space);

                for (int i = lid; i < NUM_SGS * TM * TN; i += WG_SIZE) {
                    const int s = i / (TM * TN);
                    const int rem = i % (TM * TN);
                    const int r = rem / TN;
                    const int c = rem % TN;
                    const int sr = s / SG_N;
                    const int sc = s % SG_N;
                    const int gm = m0 + sr * TM + r;
                    const int gn = n0 + sc * TN + c;
                    if (gm < M && gn < N) {
                        output_ptr[static_cast<size_t>(gm) * N + gn] = bf16(c_slm[i]);
                    }
                }
            });
    });
}

// Generalized-k-block XMX NF4 GEMM. Identical per-element arithmetic and
// k-accumulation order to packed_nf4_gemm_bf16_joint_matrix (TK=16 chunks
// are consumed sequentially), but BK can be 32/64 so the two SLM barriers
// per K tile are amortized over 2x/4x the DPAS work. The packed B tile is
// dequantized to bf16 SLM as [BN][BK] with the same bf16(qmap*am) rounding.
template <int BM = 64, int BN = 64, int BK = 32, int TM = 8, int TN = 8,
          int TK = 16, int SG = 8>
void packed_nf4_gemm_bf16_joint_matrix_wide(Context& ctx,
                                            const uint8_t* packed_ptr,
                                            const float* quant_map_ptr,
                                            const float* absmax_ptr,
                                            const bf16* input_ptr,
                                            bf16* output_ptr,
                                            int M, int N, int K,
                                            int blocksize) {
    static_assert(BK % TK == 0, "BK must be a multiple of the JM K tile");

    const int packed_bytes_per_row = K / 2;
    const int blocks_per_row = K / blocksize;
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    const int total_wgs = grid_m * grid_n;

    constexpr int SG_M = BM / TM;
    constexpr int SG_N = BN / TN;
    constexpr int NUM_SGS = SG_M * SG_N;
    constexpr int WG_SIZE = NUM_SGS * SG;
    constexpr int TK_CHUNKS = BK / TK;
    constexpr int B8_CHUNKS = BK / 8;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(BN * BK), cgh);
        sycl::local_accessor<float, 1> c_slm(sycl::range<1>(NUM_SGS * TM * TN), cgh);
        sycl::local_accessor<float, 1> qmap_slm(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(total_wgs) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int wg_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / SG;
                const int sg_row = sg_id / SG_N;
                const int sg_col = sg_id % SG_N;

                const int bm = wg_idx / grid_n;
                const int bn = wg_idx % grid_n;
                const int m0 = bm * BM;
                const int n0 = bn * BN;

                if (lid < 16) {
                    qmap_slm[lid] = quant_map_ptr[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
                joint_matrix_fill(sg, sub_c, 0.0f);

                for (int kb = 0; kb < K; kb += BK) {
                    // Stage A[BM, BK] as uint32 bf16 pairs.
                    for (int pair_idx = lid; pair_idx < (BM * BK) / 2; pair_idx += WG_SIZE) {
                        const int r = pair_idx / (BK / 2);
                        const int c_pair = pair_idx % (BK / 2);
                        const int gm = m0 + r;
                        const int c = c_pair * 2;
                        if (gm < M && kb + c < K) {
                            auto in_u32 = reinterpret_cast<const uint32_t*>(
                                input_ptr + static_cast<size_t>(gm) * K + kb)[c_pair];
                            reinterpret_cast<uint32_t*>(
                                As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = in_u32;
                        } else {
                            reinterpret_cast<uint32_t*>(
                                As.get_multi_ptr<sycl::access::decorated::no>().get())[pair_idx] = 0;
                        }
                    }

                    // Dequant B[BN, BK] into SLM, 8 NF4 values per item.
                    for (int i = lid; i < BN * B8_CHUNKS; i += WG_SIZE) {
                        const int col = i / B8_CHUNKS;
                        const int chunk = i % B8_CHUNKS;
                        const int src_n = n0 + col;
                        const int k0 = kb + chunk * 8;
                        const int base = col * BK + chunk * 8;
                        if (src_n < N && k0 + 7 < K) {
                            const uint32_t packed_u32 = reinterpret_cast<const uint32_t*>(
                                packed_ptr + static_cast<size_t>(src_n) * packed_bytes_per_row +
                                (kb / 2) + chunk * 4)[0];
                            const float am = absmax_ptr[
                                static_cast<size_t>(src_n) * blocks_per_row +
                                k0 / blocksize];

                            const uint8_t b0 = static_cast<uint8_t>(packed_u32 & 0xFF);
                            const uint8_t b1 = static_cast<uint8_t>((packed_u32 >> 8) & 0xFF);
                            const uint8_t b2 = static_cast<uint8_t>((packed_u32 >> 16) & 0xFF);
                            const uint8_t b3 = static_cast<uint8_t>((packed_u32 >> 24) & 0xFF);

                            auto bs = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
                            bs[base + 0] = bf16(qmap_slm[(b0 >> 4) & 0x0F] * am);
                            bs[base + 1] = bf16(qmap_slm[b0 & 0x0F] * am);
                            bs[base + 2] = bf16(qmap_slm[(b1 >> 4) & 0x0F] * am);
                            bs[base + 3] = bf16(qmap_slm[b1 & 0x0F] * am);
                            bs[base + 4] = bf16(qmap_slm[(b2 >> 4) & 0x0F] * am);
                            bs[base + 5] = bf16(qmap_slm[b2 & 0x0F] * am);
                            bs[base + 6] = bf16(qmap_slm[(b3 >> 4) & 0x0F] * am);
                            bs[base + 7] = bf16(qmap_slm[b3 & 0x0F] * am);
                        } else {
                            auto bs = Bs.get_multi_ptr<sycl::access::decorated::no>().get();
                            for (int v = 0; v < 8; ++v) {
                                bs[base + v] = bf16(0.0f);
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;
                    joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major> sub_b;

                    auto pA = As.get_multi_ptr<sycl::access::decorated::no>() + (sg_row * TM) * BK;
                    auto pB = Bs.get_multi_ptr<sycl::access::decorated::no>() + (sg_col * TN) * BK;

                    for (int chunk = 0; chunk < TK_CHUNKS; ++chunk) {
                        joint_matrix_load(sg, sub_a, pA + chunk * TK, BK);
                        joint_matrix_load(sg, sub_b, pB + chunk * TK, BK);
                        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                auto pC_slm = c_slm.get_multi_ptr<sycl::access::decorated::no>() + sg_id * (TM * TN);
                joint_matrix_store(sg, sub_c, pC_slm, TN, layout::row_major);
                item.barrier(sycl::access::fence_space::local_space);

                for (int i = lid; i < NUM_SGS * TM * TN; i += WG_SIZE) {
                    const int s = i / (TM * TN);
                    const int rem = i % (TM * TN);
                    const int r = rem / TN;
                    const int c = rem % TN;
                    const int sr = s / SG_N;
                    const int sc = s % SG_N;
                    const int gm = m0 + sr * TM + r;
                    const int gn = n0 + sc * TN + c;
                    if (gm < M && gn < N) {
                        output_ptr[static_cast<size_t>(gm) * N + gn] = bf16(c_slm[i]);
                    }
                }
            });
    });
}

template <int BM = 32, int BN = 32, int BK = 16, int TM = 8, int TN = 8, int TK = 16, int SG = 8>
void joint_matrix_bf16_gemm(sycl::queue& q,
                            const bf16* A_ptr,
                            const bf16* B_ptr,
                            bf16* C_ptr,
                            int M, int N, int K) {
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    const int total_wgs = grid_m * grid_n;

    constexpr int SG_M = BM / TM;
    constexpr int SG_N = BN / TN;
    constexpr int NUM_SGS = SG_M * SG_N;
    constexpr int WG_SIZE = NUM_SGS * SG;

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(TK * BN), cgh);
        sycl::local_accessor<float, 1> c_slm(sycl::range<1>(NUM_SGS * TM * TN), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(total_wgs) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int wg_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / SG;
                const int sg_row = sg_id / SG_N;
                const int sg_col = sg_id % SG_N;

                const int bm = wg_idx / grid_n;
                const int bn = wg_idx % grid_n;
                const int m0 = bm * BM;
                const int n0 = bn * BN;

                joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
                joint_matrix_fill(sg, sub_c, 0.0f);

                for (int kb = 0; kb < K; kb += BK) {
                    #pragma unroll
                    for (int i = lid; i < BM * BK; i += WG_SIZE) {
                        const int r = i / BK;
                        const int c = i % BK;
                        const int gm = m0 + r;
                        if (gm < M && kb + c < K) {
                            As[i] = A_ptr[static_cast<size_t>(gm) * K + kb + c];
                        } else {
                            As[i] = bf16(0.0f);
                        }
                    }

                    #pragma unroll
                    for (int i = lid; i < BN * TK; i += WG_SIZE) {
                        const int col = i / TK;
                        const int kh = i % TK;
                        const int src_n = n0 + col;
                        if (src_n < N && kb + kh < K) {
                            Bs[col * TK + kh] = B_ptr[static_cast<size_t>(kb + kh) * N + src_n];
                        } else {
                            Bs[col * TK + kh] = bf16(0.0f);
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;
                    joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major> sub_b;

                    auto pA = As.get_multi_ptr<sycl::access::decorated::no>() + (sg_row * TM) * BK;
                    auto pB = Bs.get_multi_ptr<sycl::access::decorated::no>() + (sg_col * TN) * TK;

                    joint_matrix_load(sg, sub_a, pA, BK);
                    joint_matrix_load(sg, sub_b, pB, TK);
                    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);

                    item.barrier(sycl::access::fence_space::local_space);
                }

                auto pC_slm = c_slm.get_multi_ptr<sycl::access::decorated::no>() + sg_id * (TM * TN);
                joint_matrix_store(sg, sub_c, pC_slm, TN, layout::row_major);
                item.barrier(sycl::access::fence_space::local_space);

                for (int i = lid; i < NUM_SGS * TM * TN; i += WG_SIZE) {
                    const int s = i / (TM * TN);
                    const int rem = i % (TM * TN);
                    const int r = rem / TN;
                    const int c = rem % TN;
                    const int sr = s / SG_N;
                    const int sc = s % SG_N;
                    const int gm = m0 + sr * TM + r;
                    const int gn = n0 + sc * TN + c;
                    if (gm < M && gn < N) {
                        C_ptr[static_cast<size_t>(gm) * N + gn] = bf16(c_slm[i]);
                    }
                }
            });
    });
}

void packed_nf4_gemm_bf16(Context& ctx,
                          const uint8_t* packed_ptr,
                          const float* quant_map_ptr,
                          const float* absmax_ptr,
                          const bf16* input_ptr,
                          bf16* output_ptr,
                          int M, int N, int K,
                          int blocksize,
                          bool allow_dense_xmx = false) {
    static const bool use_a770_bm64 = [&ctx]() {
        const std::string device_name =
            ctx.queue().get_device().get_info<sycl::info::device::name>();
        return device_name.find("Arc(TM) A770") != std::string::npos ||
               device_name.find("Arc A770") != std::string::npos;
    }();
    if (use_a770_bm64) {
        // [DISPATCH] A770 XMX Joint Matrix: hardware DPAS Matrix Engine acceleration (23x faster than ALU)
        // AILA_BNB4_GEMM_XMX is a process-wide override: 1 forces the XMX
        // route on, 0 forces it off. When unset, the caller's
        // allow_dense_xmx default decides. Existing backends keep the
        // historical off-by-default behavior; Qwen3.5 prefill opts in.
        static bool xmx_override = false;
        static const bool xmx_explicit = aila::env::read_flag_if_present(
            "AILA_BNB4_GEMM_XMX", &xmx_override);
        const bool use_xmx = xmx_explicit ? xmx_override : allow_dense_xmx;
        if (use_xmx && (K % 16 == 0) && (N <= 32768) && (K <= 16384) && (blocksize <= 64 || blocksize == 128)) {
            if (M >= 64 && N >= 64) {
                static const int xmx_bk =
                    aila::env::read_int_raw("AILA_BNB4_GEMM_XMX_BK", 0);
                if (xmx_bk >= 64 && (K % 64) == 0) {
                    packed_nf4_gemm_bf16_joint_matrix_wide<64, 64, 64, 8, 8, 16, 8>(
                        ctx, packed_ptr, quant_map_ptr, absmax_ptr, input_ptr,
                        output_ptr, M, N, K, blocksize);
                } else if (xmx_bk >= 32 && (K % 32) == 0) {
                    packed_nf4_gemm_bf16_joint_matrix_wide<64, 64, 32, 8, 8, 16, 8>(
                        ctx, packed_ptr, quant_map_ptr, absmax_ptr, input_ptr,
                        output_ptr, M, N, K, blocksize);
                } else {
                    static const int xmx_tile =
                        aila::env::read_int_raw("AILA_BNB4_GEMM_XMX_TILE", 0);
                    switch (xmx_tile) {
                        case 1:
                            packed_nf4_gemm_bf16_joint_matrix<64, 32, 16, 8, 8, 16, 8>(
                                ctx, packed_ptr, quant_map_ptr, absmax_ptr,
                                input_ptr, output_ptr, M, N, K, blocksize);
                            break;
                        case 2:
                            packed_nf4_gemm_bf16_joint_matrix<32, 64, 16, 8, 8, 16, 8>(
                                ctx, packed_ptr, quant_map_ptr, absmax_ptr,
                                input_ptr, output_ptr, M, N, K, blocksize);
                            break;
                        case 3:
                            packed_nf4_gemm_bf16_joint_matrix<32, 128, 16, 8, 8, 16, 8>(
                                ctx, packed_ptr, quant_map_ptr, absmax_ptr,
                                input_ptr, output_ptr, M, N, K, blocksize);
                            break;
                        case 4:
                            packed_nf4_gemm_bf16_joint_matrix<128, 32, 16, 8, 8, 16, 8>(
                                ctx, packed_ptr, quant_map_ptr, absmax_ptr,
                                input_ptr, output_ptr, M, N, K, blocksize);
                            break;
                        case 5:
                            packed_nf4_gemm_bf16_joint_matrix<64, 128, 16, 8, 8, 16, 8>(
                                ctx, packed_ptr, quant_map_ptr, absmax_ptr,
                                input_ptr, output_ptr, M, N, K, blocksize);
                            break;
                        case 6:
                            packed_nf4_gemm_bf16_joint_matrix<128, 64, 16, 8, 8, 16, 8>(
                                ctx, packed_ptr, quant_map_ptr, absmax_ptr,
                                input_ptr, output_ptr, M, N, K, blocksize);
                            break;
                        default:
                            packed_nf4_gemm_bf16_joint_matrix<64, 64, 16, 8, 8, 16, 8>(
                                ctx, packed_ptr, quant_map_ptr, absmax_ptr,
                                input_ptr, output_ptr, M, N, K, blocksize);
                            break;
                    }
                }
            } else {
                packed_nf4_gemm_bf16_joint_matrix<32, 32, 16, 8, 8, 16, 8>(
                    ctx, packed_ptr, quant_map_ptr, absmax_ptr, input_ptr,
                    output_ptr, M, N, K, blocksize);
            }
            return;
        }
        // [DISPATCH] A770/DG2, oneAPI 2026.1, driver 32.0.101.8860:
        // BK=64 lowers per-WG SLM from ~20.5 KB to ~10.5 KB. Verified for
        // M={3,24,37,96,256}, block={32,64,128}, and all aligned
        // DiffusionGemma N/K shapes. For blocksize 64, 20-sample decoder-shape
        // medians improved 24.6%-36.3% over BK=128; the 25-step pipeline
        // median improved 21.6% and preserved all 256 token IDs exactly.
        if (blocksize <= 64) {
            // [DISPATCH] A770/DG2, oneAPI 2026.1, driver 32.0.101.8860:
            // v2 stages the dequantized bf16 B tile in SLM and vectorizes the
            // inner loop with vec8 loads while preserving the exact per-k
            // accumulation order and bf16 rounding of v1. BM=32/BN=128 with
            // TM=2/TN=8 measured 2.4-2.6x faster than the BN=64/TN=4 geometry
            // on the real 256-row decode shapes and is bitwise identical on
            // real DiffusionGemma weights. BN=16/TN=1 geometries were
            // rejected: the device compiler reassociates their k accumulation
            // under -ffast-math and produces last-ulp differences. AILA_BNB4_GEMM_V2=0
            // selects the original scalar kernel for A/B comparison.
            static const bool use_v2 =
                aila::env::read_flag("AILA_BNB4_GEMM_V2", true);
            if (use_v2) {
                packed_nf4_gemm_bf16_tiled_v2<32, 128, 64, 2, 8, 16, 16>(
                    ctx, packed_ptr, quant_map_ptr, absmax_ptr, input_ptr,
                    output_ptr, M, N, K, blocksize);
            } else {
                packed_nf4_gemm_bf16_tiled<64, 64, 64, 4, 4, 16, 16>(
                    ctx, packed_ptr, quant_map_ptr, absmax_ptr, input_ptr,
                    output_ptr, M, N, K, blocksize);
            }
            return;
        }
        packed_nf4_gemm_bf16_tiled<64, 64, 128, 4, 4, 16, 16>(
            ctx, packed_ptr, quant_map_ptr, absmax_ptr, input_ptr, output_ptr,
            M, N, K, blocksize);
        return;
    }
    // The small-M template is incorrect for M=24 with real DiffusionGemma
    // weights on A770. Use the validated large tile for every M > 1.
    packed_nf4_gemm_bf16_tiled<128, 64, 128, 8, 4, 16, 16>(
        ctx, packed_ptr, quant_map_ptr, absmax_ptr, input_ptr, output_ptr,
        M, N, K, blocksize);
}

// Fused gate+up gemv with SiLU activation for decode (seq_len == 1).
// Computes out[i] = silu(dot(input, gate_weight[i,:])) * dot(input, up_weight[i,:])
// using NF4 packed weights with per-block float absmax.
void packed_nf4_gemv_gate_up_swiglu(Context& ctx,
                                    const uint8_t* gate_packed,
                                    const float* gate_absmax,
                                    const uint8_t* up_packed,
                                    const float* up_absmax,
                                    const float* quant_map_ptr,
                                    const bf16* input_ptr,
                                    bf16* output_ptr,
                                    int in_features,
                                    int intermediate_size,
                                    int blocksize) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    const size_t wg_size = 256;
    const size_t sub_group_size = 16;
    const size_t num_sub_groups = wg_size / sub_group_size;
    const size_t rows_per_group = num_sub_groups;
    const size_t num_groups = (static_cast<size_t>(intermediate_size) + rows_per_group - 1) / rows_per_group;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            using vec8 = sycl::vec<bf16, 8>;
            const int lid = static_cast<int>(item.get_local_id(0));
            const int sg_id = lid / sub_group_size;
            const int sg_lane = lid % sub_group_size;
            const int row = static_cast<int>(item.get_group(0)) * static_cast<int>(rows_per_group) + sg_id;
            const int stride = static_cast<int>(sub_group_size) * 4;

            if (lid < 16) qmap[lid] = quant_map_ptr[lid];
            item.barrier(sycl::access::fence_space::local_space);

            float gate_acc = 0.0f;
            float up_acc = 0.0f;
            if (row < intermediate_size) {
                const int row_byte_base = row * packed_bytes_per_row;
                const int row_block_base = row * blocks_per_row;
                int byte_offset = sg_lane * 4;
                for (; byte_offset + 3 < packed_bytes_per_row;
                     byte_offset += stride) {
                    const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                        gate_packed + row_byte_base + byte_offset);
                    const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                        up_packed + row_byte_base + byte_offset);
                    const float ga = gate_absmax[row_block_base +
                        (byte_offset / (blocksize / 2))];
                    const float ua = up_absmax[row_block_base +
                        (byte_offset / (blocksize / 2))];
                    const int ib = byte_offset * 2;
                    const vec8 in_v = *reinterpret_cast<const vec8*>(input_ptr + ib);
                    const uint8_t gb0 = static_cast<uint8_t>(g4);
                    const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                    const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                    const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                    const uint8_t ub0 = static_cast<uint8_t>(u4);
                    const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                    const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                    const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);
                    gate_acc = sycl::fma(static_cast<float>(in_v[0]), qmap[gb0 >> 4] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(in_v[1]), qmap[gb0 & 0xF] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(in_v[2]), qmap[gb1 >> 4] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(in_v[3]), qmap[gb1 & 0xF] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(in_v[4]), qmap[gb2 >> 4] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(in_v[5]), qmap[gb2 & 0xF] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(in_v[6]), qmap[gb3 >> 4] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(in_v[7]), qmap[gb3 & 0xF] * ga, gate_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[0]), qmap[ub0 >> 4] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[1]), qmap[ub0 & 0xF] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[2]), qmap[ub1 >> 4] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[3]), qmap[ub1 & 0xF] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[4]), qmap[ub2 >> 4] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[5]), qmap[ub2 & 0xF] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[6]), qmap[ub3 >> 4] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(in_v[7]), qmap[ub3 & 0xF] * ua, up_acc);
                }
                for (; byte_offset < packed_bytes_per_row;
                     byte_offset += static_cast<int>(sub_group_size)) {
                    const uint8_t gb = gate_packed[row_byte_base + byte_offset];
                    const uint8_t ub = up_packed[row_byte_base + byte_offset];
                    const float ga = gate_absmax[row_block_base +
                        (byte_offset / (blocksize / 2))];
                    const float ua = up_absmax[row_block_base +
                        (byte_offset / (blocksize / 2))];
                    const int ii = byte_offset * 2;
                    gate_acc = sycl::fma(static_cast<float>(input_ptr[ii]),
                        qmap[gb >> 4] * ga, gate_acc);
                    gate_acc = sycl::fma(static_cast<float>(input_ptr[ii + 1]),
                        qmap[gb & 0xF] * ga, gate_acc);
                    up_acc = sycl::fma(static_cast<float>(input_ptr[ii]),
                        qmap[ub >> 4] * ua, up_acc);
                    up_acc = sycl::fma(static_cast<float>(input_ptr[ii + 1]),
                        qmap[ub & 0xF] * ua, up_acc);
                }
            }

            float gate_sum = sycl::reduce_over_group(item.get_sub_group(), gate_acc, sycl::plus<>());
            float up_sum = sycl::reduce_over_group(item.get_sub_group(), up_acc, sycl::plus<>());
            if (sg_lane == 0 && row < intermediate_size) {
                const float silu_gate = gate_sum / (1.0f + sycl::native::exp(-gate_sum));
                output_ptr[row] = bf16(silu_gate * up_sum);
            }
        });
    });
}

void set_error(std::string* error_message, const std::string& message) {
    if (error_message) {
        *error_message = message;
    }
}

bool can_use_packed_decode_fastpath(const Bnb4BitWeightRef& weight,
                                    bool /*cache_dequantized_weight*/,
                                    int seq_len) {
    if (seq_len != 1) {
        return false;
    }
    if (!weight.packed_weight || !weight.packed_weight->valid()) {
        return false;
    }
    const int blocksize = weight.quant_state.blocksize;
    const int64_t in_features = weight.logical_in_features();
    return blocksize > 0 &&
           (blocksize % 2) == 0 &&
           in_features > 0 &&
           (in_features % blocksize) == 0;
}

void packed_nf4_gemv_bf16(Context& ctx,
                          const Bnb4BitWeightRef& weight,
                          Tensor& absmax_f32,
                          Tensor& input,
                          Tensor& output,
                          int in_features,
                          int out_features,
                          const bf16* add_residual = nullptr) {
    const uint8_t* packed_ptr = static_cast<const uint8_t*>(weight.packed_weight->data());
    const float* quant_map_ptr = static_cast<const float*>(weight.quant_map->data());
    const float* absmax_ptr = static_cast<const float*>(absmax_f32.data());
    const bf16* input_ptr = static_cast<const bf16*>(input.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    const int blocksize = weight.quant_state.blocksize;
    const int packed_bytes_per_row = in_features / 2;
    const int packed_bytes_per_block = blocksize / 2;
    const int blocks_per_row = in_features / blocksize;
    const bool use_sg16 = (packed_bytes_per_row >= 256);

    if (!use_sg16) {
        const size_t wg_size = packed_bytes_per_row >= 1024 ? 256 : 128;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> quant_map_cache(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(out_features) * wg_size, wg_size),
                             [=](sycl::nd_item<1> item) {
                const int output_index = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));

                if (lid < 16) {
                    quant_map_cache[lid] = quant_map_ptr[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                const int row_byte_base = output_index * packed_bytes_per_row;
                const int row_block_base = output_index * blocks_per_row;
                float partial = 0.0f;
                for (int byte_offset = lid; byte_offset < packed_bytes_per_row; byte_offset += static_cast<int>(wg_size)) {
                    const uint8_t packed = packed_ptr[row_byte_base + byte_offset];
                    const float absmax = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const int input_index = byte_offset * 2;
                    const float weight_hi = static_cast<float>(bf16(
                        quant_map_cache[(packed >> 4) & 0x0F] * absmax));
                    const float weight_lo = static_cast<float>(bf16(
                        quant_map_cache[packed & 0x0F] * absmax));
                    partial += static_cast<float>(input_ptr[input_index]) *
                               weight_hi;
                    partial += static_cast<float>(input_ptr[input_index + 1]) *
                               weight_lo;
                }

                const float sum = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
                if (lid == 0) {
                    float val = sum;
                    if (add_residual != nullptr) val += static_cast<float>(add_residual[output_index]);
                    out_ptr[output_index] = bf16(val);
                }
            });
        });
        return;
    }

    static const int gemv_wg_override = aila::env::read_int_raw("AILA_BNB4_GEMV_WG", 0);
    const size_t wg_size = (gemv_wg_override >= 32) ? static_cast<size_t>(gemv_wg_override) : 256;
    const size_t sub_group_size = 16;
    const size_t num_sub_groups = wg_size / sub_group_size;
    const size_t rows_per_group = num_sub_groups;
    const size_t num_groups = (static_cast<size_t>(out_features) + rows_per_group - 1) / rows_per_group;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> quant_map_cache(sycl::range<1>(16), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            const int lid = static_cast<int>(item.get_local_id(0));
            const int sg_id = lid / sub_group_size;
            const int sg_lane = lid % sub_group_size;
            const int group_row_base = static_cast<int>(item.get_group(0)) * static_cast<int>(rows_per_group);
            const int row = group_row_base + sg_id;

            if (lid < 16) {
                quant_map_cache[lid] = quant_map_ptr[lid];
            }
            item.barrier(sycl::access::fence_space::local_space);

            float partial = 0.0f;
            if (row < out_features) {
                using vec8 = sycl::vec<bf16, 8>;
                const int row_byte_base = row * packed_bytes_per_row;
                const int row_block_base = row * blocks_per_row;
                int byte_offset = sg_lane * 4;
                const int stride = static_cast<int>(sub_group_size) * 4;
                for (; byte_offset + 3 < packed_bytes_per_row;
                     byte_offset += stride) {
                    const uint32_t p4 = *reinterpret_cast<const uint32_t*>(
                        packed_ptr + row_byte_base + byte_offset);
                    const float am = absmax_ptr[row_block_base +
                        (byte_offset / packed_bytes_per_block)];
                    const int ib = byte_offset * 2;
                    const uint8_t b0 = static_cast<uint8_t>(p4);
                    const uint8_t b1 = static_cast<uint8_t>(p4 >> 8);
                    const uint8_t b2 = static_cast<uint8_t>(p4 >> 16);
                    const uint8_t b3 = static_cast<uint8_t>(p4 >> 24);
                    const vec8 in_v = *reinterpret_cast<const vec8*>(input_ptr + ib);
                    // Inline dequant into FMA: eliminates 8 float temporaries,
                    // giving the compiler freedom to schedule and reuse registers.
                    partial = sycl::fma(static_cast<float>(in_v[0]),
                        static_cast<float>(bf16(quant_map_cache[b0 >> 4] * am)), partial);
                    partial = sycl::fma(static_cast<float>(in_v[1]),
                        static_cast<float>(bf16(quant_map_cache[b0 & 0xF] * am)), partial);
                    partial = sycl::fma(static_cast<float>(in_v[2]),
                        static_cast<float>(bf16(quant_map_cache[b1 >> 4] * am)), partial);
                    partial = sycl::fma(static_cast<float>(in_v[3]),
                        static_cast<float>(bf16(quant_map_cache[b1 & 0xF] * am)), partial);
                    partial = sycl::fma(static_cast<float>(in_v[4]),
                        static_cast<float>(bf16(quant_map_cache[b2 >> 4] * am)), partial);
                    partial = sycl::fma(static_cast<float>(in_v[5]),
                        static_cast<float>(bf16(quant_map_cache[b2 & 0xF] * am)), partial);
                    partial = sycl::fma(static_cast<float>(in_v[6]),
                        static_cast<float>(bf16(quant_map_cache[b3 >> 4] * am)), partial);
                    partial = sycl::fma(static_cast<float>(in_v[7]),
                        static_cast<float>(bf16(quant_map_cache[b3 & 0xF] * am)), partial);
                }
                for (; byte_offset < packed_bytes_per_row;
                     byte_offset += static_cast<int>(sub_group_size)) {
                    const uint8_t pb = packed_ptr[row_byte_base + byte_offset];
                    const float am = absmax_ptr[row_block_base +
                        (byte_offset / packed_bytes_per_block)];
                    const int ii = byte_offset * 2;
                    partial = sycl::fma(static_cast<float>(input_ptr[ii]),
                        static_cast<float>(bf16(
                            quant_map_cache[(pb >> 4) & 0xF] * am)), partial);
                    partial = sycl::fma(static_cast<float>(input_ptr[ii + 1]),
                        static_cast<float>(bf16(
                            quant_map_cache[pb & 0xF] * am)), partial);
                }
            }

            float sum = sycl::reduce_over_group(item.get_sub_group(), partial, sycl::plus<>());
            if (sg_lane == 0 && row < out_features) {
                out_ptr[row] = bf16(sum);
            }
        });
    });
}

} // namespace

void Bnb4BitLinearScratch::ensure(Context& ctx, int seq_len, int in_features, int out_features, bool need_weight) {
    const bool grow_seq = seq_len > seq_capacity;
    const bool grow_input = in_features > input_dim_capacity;
    const bool grow_output = out_features > output_dim_capacity;
    const bool grow_weight = in_features > weight_in_capacity || out_features > weight_out_capacity;

    const int new_seq_cap = grow_seq ? round_up_seq(seq_len, 64) : seq_capacity;
    const int new_input_dim_cap = std::max(input_dim_capacity, in_features);
    const int new_output_dim_cap = std::max(output_dim_capacity, out_features);
    const int new_weight_in_cap = std::max(weight_in_capacity, in_features);
    const int new_weight_out_cap = std::max(weight_out_capacity, out_features);

    // Tensor move-assignment frees the old USM allocation immediately.  Make
    // sure an earlier in-order kernel is no longer using a buffer before a
    // growth operation replaces it.
    if ((grow_seq || grow_input) && input_bf16.valid() ||
        (grow_seq || grow_output) && output_bf16.valid() ||
        (need_weight && grow_weight && weight_bf16.valid())) {
        ctx.synchronize();
    }

    if (grow_seq || grow_input) {
        input_bf16 = Tensor::allocate(ctx,
                                     {(int64_t)new_seq_cap, (int64_t)new_input_dim_cap},
                                     dnnl::memory::data_type::bf16);
    }

    if (grow_seq || grow_output) {
        output_bf16 = Tensor::allocate(ctx,
                                      {(int64_t)new_seq_cap, (int64_t)new_output_dim_cap},
                                      dnnl::memory::data_type::bf16);
    }

    if (need_weight && grow_weight) {
        weight_bf16 = Tensor::allocate(ctx,
                                      {(int64_t)new_weight_in_cap, (int64_t)new_weight_out_cap},
                                      dnnl::memory::data_type::bf16);
    }

    seq_capacity = new_seq_cap;
    input_dim_capacity = new_input_dim_cap;
    output_dim_capacity = new_output_dim_cap;
    if (need_weight) {
        weight_in_capacity = new_weight_in_cap;
        weight_out_capacity = new_weight_out_cap;
    }
}

bool Bnb4BitLinear::init(Context& ctx, const Bnb4BitWeightRef& weight, std::string* error_message) {
    if (!weight.valid()) {
        set_error(error_message, "Invalid bitsandbytes 4-bit weight reference");
        return false;
    }

    owned_packed_weight_ = Tensor();
    force_dequant_cache_ = false;
    release_quant_after_cache_ = false;
    weight_ = weight;
    in_features_ = static_cast<int>(weight.logical_in_features());
    out_features_ = static_cast<int>(weight.logical_out_features());
    if (in_features_ <= 0 || out_features_ <= 0) {
        set_error(error_message, "bitsandbytes weight has invalid logical shape: " + weight.name);
        return false;
    }

    const int64_t block_count = weight.absmax->numel();
    std::vector<float> host_absmax(static_cast<size_t>(block_count), 0.0f);
    if (weight.quant_state.nested) {
        std::vector<uint8_t> qabsmax(static_cast<size_t>(block_count), 0);
        std::vector<float> nested_absmax(static_cast<size_t>(weight.nested_absmax->numel()), 0.0f);
        std::vector<float> nested_quant_map(static_cast<size_t>(weight.nested_quant_map->numel()), 0.0f);

        ctx.memcpy_d2h(qabsmax.data(), weight.absmax->data(), qabsmax.size() * sizeof(uint8_t));
        ctx.memcpy_d2h(nested_absmax.data(), weight.nested_absmax->data(), nested_absmax.size() * sizeof(float));
        ctx.memcpy_d2h(nested_quant_map.data(), weight.nested_quant_map->data(), nested_quant_map.size() * sizeof(float));

        const int nested_blocksize = weight.quant_state.nested_blocksize;
        for (int64_t i = 0; i < block_count; ++i) {
            const int64_t nested_block = i / nested_blocksize;
            const uint8_t qv = qabsmax[static_cast<size_t>(i)];
            host_absmax[static_cast<size_t>(i)] =
                nested_quant_map[static_cast<size_t>(qv)] * nested_absmax[static_cast<size_t>(nested_block)] +
                weight.quant_state.nested_offset;
        }
    } else {
        ctx.memcpy_d2h(host_absmax.data(), weight.absmax->data(), host_absmax.size() * sizeof(float));
    }
    absmax_f32_ = Tensor::allocate(ctx, {(int64_t)block_count}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(absmax_f32_.data(), host_absmax.data(), host_absmax.size() * sizeof(float));
    finish_init(ctx);
    return true;
}

bool Bnb4BitLinear::init_fused_rows(Context& ctx,
                                    const Bnb4BitLinear& first,
                                    const Bnb4BitLinear& second,
                                    std::string* error_message) {
    return init_fused_rows_impl(ctx, {&first, &second}, error_message);
}

bool Bnb4BitLinear::init_fused_rows(Context& ctx,
                                    const Bnb4BitLinear& first,
                                    const Bnb4BitLinear& second,
                                    const Bnb4BitLinear& third,
                                    std::string* error_message) {
    return init_fused_rows_impl(ctx, {&first, &second, &third}, error_message);
}

bool Bnb4BitLinear::init_fused_rows(Context& ctx,
                                    const Bnb4BitLinear& first,
                                    const Bnb4BitLinear& second,
                                    const Bnb4BitLinear& third,
                                    const Bnb4BitLinear& fourth,
                                    std::string* error_message) {
    return init_fused_rows_impl(ctx, {&first, &second, &third, &fourth}, error_message);
}

bool Bnb4BitLinear::init_fused_rows_impl(Context& ctx,
                                         const std::vector<const Bnb4BitLinear*>& sources,
                                         std::string* error_message) {
    if (sources.size() < 2 || sources.size() > 4) {
        set_error(error_message, "Bnb4BitLinear::init_fused_rows requires 2 to 4 source linears");
        return false;
    }

    const Bnb4BitLinear* first = sources.front();
    if (first == nullptr ||
        !first->weight_.packed_weight ||
        !first->weight_.quant_map ||
        !first->absmax_f32_.valid()) {
        set_error(error_message, "Bnb4BitLinear::init_fused_rows: source linear is not initialized");
        return false;
    }

    const int in_features = first->in_features_;
    const auto& ref_quant_state = first->weight_.quant_state;
    const int blocksize = ref_quant_state.blocksize;
    if (in_features <= 0 || blocksize <= 0 || (blocksize % 2) != 0 || (in_features % blocksize) != 0) {
        set_error(error_message, "Bnb4BitLinear::init_fused_rows: incompatible source weight geometry");
        return false;
    }

    std::vector<float> ref_quant_map(static_cast<size_t>(first->weight_.quant_map->numel()), 0.0f);
    ctx.memcpy_d2h(ref_quant_map.data(),
                   first->weight_.quant_map->data(),
                   ref_quant_map.size() * sizeof(float));

    int64_t total_out_features = 0;
    int64_t total_packed_bytes = 0;
    int64_t total_absmax_values = 0;
    std::string fused_name = "fused(";
    for (size_t source_index = 0; source_index < sources.size(); ++source_index) {
        const Bnb4BitLinear* src = sources[source_index];
        if (src == nullptr ||
            !src->weight_.packed_weight ||
            !src->weight_.quant_map ||
            !src->absmax_f32_.valid()) {
            set_error(error_message, "Bnb4BitLinear::init_fused_rows: source linear is not initialized");
            return false;
        }
        if (src->in_features_ != in_features ||
            src->weight_.quant_state.blocksize != blocksize ||
            src->weight_.quant_state.quant_type != ref_quant_state.quant_type ||
            src->weight_.quant_map->numel() != first->weight_.quant_map->numel()) {
            set_error(error_message, "Bnb4BitLinear::init_fused_rows: incompatible quantized linears");
            return false;
        }

        std::vector<float> src_quant_map(static_cast<size_t>(src->weight_.quant_map->numel()), 0.0f);
        ctx.memcpy_d2h(src_quant_map.data(),
                       src->weight_.quant_map->data(),
                       src_quant_map.size() * sizeof(float));
        for (size_t i = 0; i < ref_quant_map.size(); ++i) {
            if (std::abs(ref_quant_map[i] - src_quant_map[i]) > 1e-6f) {
                set_error(error_message, "Bnb4BitLinear::init_fused_rows: quant maps do not match");
                return false;
            }
        }

        total_out_features += src->out_features_;
        total_packed_bytes += src->weight_.packed_weight->size_bytes();
        total_absmax_values += src->absmax_f32_.numel();
        if (source_index > 0) {
            fused_name += ",";
        }
        fused_name += src->weight_.name;
    }
    fused_name += ")";

    owned_packed_weight_ = Tensor::allocate(ctx, {total_packed_bytes}, dnnl::memory::data_type::u8);
    absmax_f32_ = Tensor::allocate(ctx, {total_absmax_values}, dnnl::memory::data_type::f32);
    force_dequant_cache_ = false;
    release_quant_after_cache_ = false;

    size_t packed_offset = 0;
    size_t absmax_offset = 0;
    auto* fused_packed_ptr = static_cast<uint8_t*>(owned_packed_weight_.data());
    auto* fused_absmax_ptr = static_cast<float*>(absmax_f32_.data());
    for (const Bnb4BitLinear* src : sources) {
        const size_t packed_bytes = src->weight_.packed_weight->size_bytes();
        const size_t absmax_values = static_cast<size_t>(src->absmax_f32_.numel());
        ctx.queue().memcpy(fused_packed_ptr + packed_offset,
                           src->weight_.packed_weight->data(),
                           packed_bytes);
        ctx.queue().memcpy(fused_absmax_ptr + absmax_offset,
                           src->absmax_f32_.data(),
                           absmax_values * sizeof(float));
        packed_offset += packed_bytes;
        absmax_offset += absmax_values;
    }

    weight_ = {};
    weight_.name = fused_name;
    weight_.packed_weight = &owned_packed_weight_;
    weight_.quant_map = first->weight_.quant_map;
    weight_.quant_state = ref_quant_state;
    weight_.quant_state.shape = {total_out_features, (int64_t)in_features};
    in_features_ = in_features;
    out_features_ = static_cast<int>(total_out_features);

    finish_init(ctx);
    ctx.synchronize();
    return true;
}

void Bnb4BitLinear::finish_init(Context& ctx) {
    prim_cache_.clear();
    decode_args_.clear();
    decode_mem_inited_ = false;
    decode_src_ptr_ = nullptr;
    decode_weight_ptr_ = nullptr;
    decode_dst_ptr_ = nullptr;
    decode_scratchpad_ = Tensor();
    cached_weight_bf16_ = Tensor();
    cached_weight_ready_ = false;

    cache_dequantized_weight_ = force_dequant_cache_ ||
                                aila::env::read_flag("AILA_BNB4_CACHE_DEQUANT", false);
    if (cache_dequantized_weight_) {
        cached_weight_bf16_ = Tensor::allocate(ctx,
                                              {(int64_t)in_features_, (int64_t)out_features_},
                                              dnnl::memory::data_type::bf16);
        dequantize_weight(ctx, cached_weight_bf16_);
        ctx.synchronize();
        cached_weight_ready_ = true;
        if (release_quant_after_cache_ && owned_packed_weight_.valid()) {
            owned_packed_weight_ = Tensor();
            weight_.packed_weight = nullptr;
            absmax_f32_ = Tensor();
        }
    }

    // Only create the oneDNN decode primitive if the NF4 fastpath is NOT available.
    // When fastpath is available, decode (seq_len=1) uses the custom GEMV kernel
    // exclusively — the oneDNN primitive would be dead weight.  If the fastpath
    // later becomes unavailable (e.g. AILA_BNB4_CACHE_DEQUANT enabled), the
    // primitive is created lazily on first use via ensure_primitive().
    decode_inited_ = false;
    bool need_decode_dnnl = !can_use_packed_decode_fastpath(weight_, cache_dequantized_weight_, 1);
    if (need_decode_dnnl) {
        decode_src_md_ = dnnl::memory::desc({1, in_features_}, dnnl::memory::data_type::bf16,
                                            dnnl::memory::format_tag::ab);
        decode_weight_md_ = dnnl::memory::desc({in_features_, out_features_}, dnnl::memory::data_type::bf16,
                                               dnnl::memory::format_tag::ab);
        decode_dst_md_ = dnnl::memory::desc({1, out_features_}, dnnl::memory::data_type::bf16,
                                            dnnl::memory::format_tag::ab);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        auto pd = dnnl::matmul::primitive_desc(ctx.engine(), decode_src_md_, decode_weight_md_, decode_dst_md_, attr);
        decode_prim_ = dnnl::matmul(pd);
        size_t scratchpad_size = pd.scratchpad_desc().get_size();
        if (scratchpad_size > 0) {
            decode_scratchpad_ = Tensor::allocate(ctx,
                                                  {static_cast<int64_t>(scratchpad_size)},
                                                  dnnl::memory::data_type::u8);
            decode_scratchpad_mem_ = decode_scratchpad_.make_dnnl_memory(pd.scratchpad_desc());
        }
        decode_inited_ = true;
    }

}

void Bnb4BitLinear::ensure_primitive(Context& ctx, int seq_len) {
    if (seq_len == 1 && decode_inited_) {
        return;
    }
    if (prim_cache_.count(seq_len)) {
        return;
    }

    auto src_md = dnnl::memory::desc({seq_len, in_features_}, dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);
    auto weight_md = dnnl::memory::desc({in_features_, out_features_}, dnnl::memory::data_type::bf16,
                                        dnnl::memory::format_tag::ab);
    auto dst_md = dnnl::memory::desc({seq_len, out_features_}, dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    // Prefill creates one primitive per weight and sequence length.  Keeping a
    // user scratchpad in every cached primitive grows by hundreds of MB across
    // DiffusionGemma's 30 layers and exhausts a 16 GB A770.  Library mode keeps
    // the scratch lifetime at execution scope.
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), src_md, weight_md, dst_md, attr);

    CachedPrimitive cp;
    cp.prim = dnnl::matmul(pd);
    cp.src_md = src_md;
    cp.weight_md = weight_md;
    cp.dst_md = dst_md;
    prim_cache_.emplace(seq_len, std::move(cp));
}

void Bnb4BitLinear::dequantize_weight(Context& ctx, Tensor& weight_bf16_view) {
    const uint8_t* packed_ptr = static_cast<const uint8_t*>(weight_.packed_weight->data());
    const float* quant_map_ptr = static_cast<const float*>(weight_.quant_map->data());
    const float* absmax_ptr = static_cast<const float*>(absmax_f32_.data());
    bf16* out_ptr = static_cast<bf16*>(weight_bf16_view.data());

    const int64_t packed_bytes = weight_.packed_num_bytes();
    const int64_t total_values = weight_.logical_numel();
    const int64_t in_features = weight_.logical_in_features();
    const int64_t out_features = weight_.logical_out_features();
    const int blocksize = weight_.quant_state.blocksize;

    ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(packed_bytes)), [=](sycl::id<1> idx) {
        const int64_t byte_index = static_cast<int64_t>(idx[0]);
        const uint8_t packed = packed_ptr[byte_index];
        const uint8_t hi = static_cast<uint8_t>((packed >> 4) & 0x0F);
        const uint8_t lo = static_cast<uint8_t>(packed & 0x0F);

        const int64_t flat0 = byte_index * 2;
        if (flat0 < total_values) {
            const int64_t row = flat0 / in_features;
            const int64_t col = flat0 % in_features;
            const int64_t out_index = col * out_features + row;
            out_ptr[out_index] = bf16(quant_map_ptr[hi] * absmax_ptr[flat0 / blocksize]);
        }

        const int64_t flat1 = flat0 + 1;
        if (flat1 < total_values) {
            const int64_t row = flat1 / in_features;
            const int64_t col = flat1 % in_features;
            const int64_t out_index = col * out_features + row;
            out_ptr[out_index] = bf16(quant_map_ptr[lo] * absmax_ptr[flat1 / blocksize]);
        }
    });
}

bool Bnb4BitLinear::try_forward_decode_gate_up_swiglu(Context& ctx,
                                                      Bnb4BitLinear& fused_gate_up,
                                                      Tensor& input,
                                                      Tensor& output,
                                                      int ff_dim) {
    if (input.dtype() != dnnl::memory::data_type::bf16 ||
        output.dtype() != dnnl::memory::data_type::bf16) return false;
    if (input.ndim() != 2) return false;

    const auto& w = fused_gate_up.weight_;
    if (!w.packed_weight || !w.packed_weight->valid()) return false;
    if (!w.quant_map || !w.quant_map->valid()) return false;
    if (!fused_gate_up.absmax_f32_.valid()) return false;
    if (fused_gate_up.out_features_ != 2 * ff_dim) return false;

    const int in_features = fused_gate_up.in_features_;
    const int blocksize = w.quant_state.blocksize;
    if (blocksize <= 0 || (blocksize % 2) != 0 || (in_features % blocksize) != 0) return false;

    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;

    const uint8_t* base_packed = static_cast<const uint8_t*>(w.packed_weight->data());
    const float* base_absmax = static_cast<const float*>(fused_gate_up.absmax_f32_.data());
    const float* quant_map_ptr = static_cast<const float*>(w.quant_map->data());
    const bf16* input_ptr = static_cast<const bf16*>(input.data());
    bf16* output_ptr = static_cast<bf16*>(output.data());

    const uint8_t* gate_packed = base_packed;
    const float* gate_absmax = base_absmax;
    const uint8_t* up_packed = base_packed + static_cast<size_t>(ff_dim) * packed_bytes_per_row;
    const float* up_absmax = base_absmax + static_cast<size_t>(ff_dim) * blocks_per_row;

    packed_nf4_gemv_gate_up_swiglu(ctx, gate_packed, gate_absmax,
                                   up_packed, up_absmax,
                                   quant_map_ptr, input_ptr, output_ptr,
                                   in_features, ff_dim, blocksize);
    return true;
}

void Bnb4BitLinear::apply_lora_decode(Context& ctx, Tensor& input, Tensor& output) {
    const bf16* in_ptr = static_cast<const bf16*>(input.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());
    int in_f = in_features_;

    for (const auto& att : lora_attachments_) {
        const bf16* a_ptr = static_cast<const bf16*>(att.lora_a.data());
        const bf16* b_ptr = static_cast<const bf16*>(att.lora_b.data());
        int r = static_cast<int>(att.lora_a.shape(0));
        int rows = att.output_rows;
        int offset = att.output_offset;
        float scale = att.scaling;

        // Small USM temp buffer for intermediate: (r,) float
        float* inter_f32 = sycl::malloc_device<float>(static_cast<size_t>(r), ctx.queue());

        // Step 1: intermediate[k] = sum_j A[k,j] * x[j]
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(r)),
            [=](sycl::id<1> idx) {
                int k = idx[0];
                float sum = 0.0f;
                for (int j = 0; j < in_f; ++j) {
                    sum += static_cast<float>(a_ptr[static_cast<size_t>(k) * in_f + j]) *
                           static_cast<float>(in_ptr[j]);
                }
                inter_f32[k] = sum;
            });

        // Step 2: output[offset + i] += sum_k B[i,k] * intermediate[k] * scale
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(rows)),
            [=](sycl::id<1> idx) {
                int i = idx[0];
                float acc = 0.0f;
                for (int k = 0; k < r; ++k) {
                    acc += static_cast<float>(b_ptr[static_cast<size_t>(i) * r + k]) * inter_f32[k];
                }
                int out_idx = offset + i;
                out_ptr[out_idx] = bf16(static_cast<float>(out_ptr[out_idx]) + acc * scale);
            });

        sycl::free(inter_f32, ctx.queue());
    }
}

void Bnb4BitLinear::apply_lora_prefill(Context& ctx, Tensor& input, Tensor& output, int seq_len) {
    const bf16* in_ptr = static_cast<const bf16*>(input.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());
    int in_f = in_features_;

    for (const auto& att : lora_attachments_) {
        const bf16* a_ptr = static_cast<const bf16*>(att.lora_a.data());
        const bf16* b_ptr = static_cast<const bf16*>(att.lora_b.data());
        int r = static_cast<int>(att.lora_a.shape(0));
        int rows = att.output_rows;
        int offset = att.output_offset;
        float scale = att.scaling;

        // Allocate temp buffer for intermediate: (seq_len, r)
        Tensor inter = Tensor::allocate(ctx, {static_cast<int64_t>(seq_len), static_cast<int64_t>(r)},
                                        dnnl::memory::data_type::f32);
        float* inter_ptr = static_cast<float*>(inter.data());

        // Step 1: intermediate[b,k] = sum_j A[k,j] * input[b,j]
        ctx.queue().parallel_for(sycl::range<2>(static_cast<size_t>(seq_len), static_cast<size_t>(r)),
            [=](sycl::id<2> idx) {
                int b = idx[0];
                int k = idx[1];
                float sum = 0.0f;
                for (int j = 0; j < in_f; ++j) {
                    sum += static_cast<float>(a_ptr[static_cast<size_t>(k) * in_f + j]) *
                           static_cast<float>(in_ptr[static_cast<size_t>(b) * in_f + j]);
                }
                inter_ptr[static_cast<size_t>(b) * r + k] = sum;
            });

        // Step 2: output[b, offset+i] += sum_k B[i,k] * intermediate[b,k] * scale
        int out_f = static_cast<int>(output.shape(1));
        ctx.queue().parallel_for(sycl::range<2>(static_cast<size_t>(seq_len), static_cast<size_t>(rows)),
            [=](sycl::id<2> idx) {
                int b = idx[0];
                int i = idx[1];
                float acc = 0.0f;
                for (int k = 0; k < r; ++k) {
                    acc += static_cast<float>(b_ptr[static_cast<size_t>(i) * r + k]) *
                           inter_ptr[static_cast<size_t>(b) * r + k];
                }
                int out_idx = static_cast<size_t>(b) * out_f + offset + i;
                out_ptr[out_idx] = bf16(static_cast<float>(out_ptr[out_idx]) + acc * scale);
            });
    }
}

void Bnb4BitLinear::forward(Context& ctx,
                            Bnb4BitLinearScratch& scratch,
                            Tensor& input,
                            Tensor& output,
                            int seq_len,
                            const bf16* add_residual) {
    if (input.dtype() != dnnl::memory::data_type::bf16 ||
        output.dtype() != dnnl::memory::data_type::bf16) {
        throw std::runtime_error("Bnb4BitLinear expects bf16 input/output tensors");
    }
    if (seq_len <= 0) {
        throw std::runtime_error("Bnb4BitLinear::forward: seq_len must be positive");
    }

    const bool use_packed_decode_fastpath =
        can_use_packed_decode_fastpath(weight_, cache_dequantized_weight_, seq_len);
    if (use_packed_decode_fastpath) {
        packed_nf4_gemv_bf16(ctx, weight_, absmax_f32_, input, output, in_features_, out_features_, add_residual);
        if (!lora_attachments_.empty()) {
            apply_lora_decode(ctx, input, output);
        }
        return;
    }

    const int blocksize = weight_.quant_state.blocksize;
    static const int fused_prefill_mode =
        aila::env::read_int_raw("AILA_BNB4_FUSED_PREFILL", -1);
    const bool use_fused_prefill = (fused_prefill_mode == 1) ||
                                   (fused_prefill_mode == -1 && allow_dense_xmx_);
    if (use_fused_prefill && seq_len > 1 &&
        (blocksize % 2) == 0 &&
        (128 % blocksize) == 0 &&
        in_features_ > 0 && (in_features_ % blocksize) == 0 &&
        weight_.packed_weight && weight_.packed_weight->valid() &&
        weight_.quant_map && weight_.quant_map->valid()) {
        scratch.ensure(ctx, seq_len, in_features_, out_features_, false);
        Tensor fused_input = Tensor::view(
            ctx, scratch.input_bf16.data(), {seq_len, (int64_t)in_features_},
            dnnl::memory::data_type::bf16);
        Tensor fused_output = Tensor::view(
            ctx, scratch.output_bf16.data(), {seq_len, (int64_t)out_features_},
            dnnl::memory::data_type::bf16);
        const size_t input_bytes = static_cast<size_t>(seq_len) *
                                   static_cast<size_t>(in_features_) * sizeof(bf16);
        const size_t output_bytes = static_cast<size_t>(seq_len) *
                                    static_cast<size_t>(out_features_) * sizeof(bf16);
        ctx.queue().memcpy(fused_input.data(), input.data(), input_bytes);
        packed_nf4_gemm_bf16(ctx,
                             static_cast<const uint8_t*>(weight_.packed_weight->data()),
                             static_cast<const float*>(weight_.quant_map->data()),
                             static_cast<const float*>(absmax_f32_.data()),
                             static_cast<const bf16*>(fused_input.data()),
                             static_cast<bf16*>(fused_output.data()),
                             seq_len, out_features_, in_features_,
                             blocksize, allow_dense_xmx_);
        ctx.queue().memcpy(output.data(), fused_output.data(), output_bytes);
        if (!lora_attachments_.empty()) {
            apply_lora_prefill(ctx, input, output, seq_len);
        }
        return;
    }

    const bool need_scratch_weight = !cache_dequantized_weight_;

    scratch.ensure(ctx, seq_len, in_features_, out_features_, need_scratch_weight);
    Tensor input_bf16 = Tensor::view(ctx, scratch.input_bf16.data(),
                                    {seq_len, (int64_t)in_features_},
                                    dnnl::memory::data_type::bf16);
    Tensor output_bf16 = Tensor::view(ctx, scratch.output_bf16.data(),
                                     {seq_len, (int64_t)out_features_},
                                     dnnl::memory::data_type::bf16);
    Tensor scratch_weight_bf16;
    Tensor* weight_bf16 = nullptr;
    if (cache_dequantized_weight_) {
        weight_bf16 = &cached_weight_bf16_;
    } else {
        scratch_weight_bf16 = Tensor::view(ctx, scratch.weight_bf16.data(),
                                          {(int64_t)in_features_, (int64_t)out_features_},
                                          dnnl::memory::data_type::bf16);
        weight_bf16 = &scratch_weight_bf16;
    }

    ctx.queue().memcpy(input_bf16.data(), input.data(),
                       static_cast<size_t>(seq_len) * static_cast<size_t>(in_features_) * sizeof(bf16));

    if (!cache_dequantized_weight_ || !cached_weight_ready_) {
        dequantize_weight(ctx, *weight_bf16);
        ctx.synchronize();
    }
    ensure_primitive(ctx, seq_len);

    if (seq_len == 1 && decode_inited_) {
        if (!decode_mem_inited_) {
            decode_src_mem_ = dnnl::sycl_interop::make_memory(decode_src_md_, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, input_bf16.data());
            decode_weight_mem_ = dnnl::sycl_interop::make_memory(decode_weight_md_, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, weight_bf16->data());
            decode_dst_mem_ = dnnl::sycl_interop::make_memory(decode_dst_md_, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, output_bf16.data());
            decode_src_ptr_ = input_bf16.data();
            decode_weight_ptr_ = weight_bf16->data();
            decode_dst_ptr_ = output_bf16.data();
            decode_args_ = {
                {DNNL_ARG_SRC, decode_src_mem_},
                {DNNL_ARG_WEIGHTS, decode_weight_mem_},
                {DNNL_ARG_DST, decode_dst_mem_}
            };
            if (decode_scratchpad_.valid()) {
                decode_args_.emplace(DNNL_ARG_SCRATCHPAD, decode_scratchpad_mem_);
            }
            decode_mem_inited_ = true;
        } else {
            if (decode_src_ptr_ != input_bf16.data()) {
                decode_src_mem_.set_data_handle(input_bf16.data());
                decode_src_ptr_ = input_bf16.data();
                decode_args_[DNNL_ARG_SRC] = decode_src_mem_;
            }
            if (decode_weight_ptr_ != weight_bf16->data()) {
                decode_weight_mem_.set_data_handle(weight_bf16->data());
                decode_weight_ptr_ = weight_bf16->data();
                decode_args_[DNNL_ARG_WEIGHTS] = decode_weight_mem_;
            }
            if (decode_dst_ptr_ != output_bf16.data()) {
                decode_dst_mem_.set_data_handle(output_bf16.data());
                decode_dst_ptr_ = output_bf16.data();
                decode_args_[DNNL_ARG_DST] = decode_dst_mem_;
            }
        }
        decode_prim_.execute(ctx.stream(), decode_args_);
    } else {
        auto& cp = prim_cache_[seq_len];
        if (!cp.mem_inited) {
            cp.src_mem = dnnl::sycl_interop::make_memory(cp.src_md, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, input_bf16.data());
            cp.weight_mem = dnnl::sycl_interop::make_memory(cp.weight_md, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, weight_bf16->data());
            cp.dst_mem = dnnl::sycl_interop::make_memory(cp.dst_md, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, output_bf16.data());
            cp.src_ptr = input_bf16.data();
            cp.weight_ptr = weight_bf16->data();
            cp.dst_ptr = output_bf16.data();
            cp.args = {
                {DNNL_ARG_SRC, cp.src_mem},
                {DNNL_ARG_WEIGHTS, cp.weight_mem},
                {DNNL_ARG_DST, cp.dst_mem}
            };
            if (cp.scratchpad.valid()) {
                cp.args.emplace(DNNL_ARG_SCRATCHPAD, cp.scratchpad_mem);
            }
            cp.mem_inited = true;
        } else {
            if (cp.src_ptr != input_bf16.data()) {
                cp.src_mem.set_data_handle(input_bf16.data());
                cp.src_ptr = input_bf16.data();
                cp.args[DNNL_ARG_SRC] = cp.src_mem;
            }
            if (cp.weight_ptr != weight_bf16->data()) {
                cp.weight_mem.set_data_handle(weight_bf16->data());
                cp.weight_ptr = weight_bf16->data();
                cp.args[DNNL_ARG_WEIGHTS] = cp.weight_mem;
            }
            if (cp.dst_ptr != output_bf16.data()) {
                cp.dst_mem.set_data_handle(output_bf16.data());
                cp.dst_ptr = output_bf16.data();
                cp.args[DNNL_ARG_DST] = cp.dst_mem;
            }
        }
        cp.prim.execute(ctx.stream(), cp.args);
    }

    if (!lora_attachments_.empty()) {
        apply_lora_prefill(ctx, input_bf16, output_bf16, seq_len);
    }

    ctx.queue().memcpy(output.data(), output_bf16.data(),
                       static_cast<size_t>(seq_len) * static_cast<size_t>(out_features_) * sizeof(bf16));
}

// ============================================================
// Native bf16 GEMV kernel (no quantization)
// Cooperative SG=16, sub-group FMA reduce.
// Weight is [M, K] row-major (M=out_features, K=in_features).
// Uses contiguous vec8 load from both weight and input.
// ============================================================
namespace ops {

void bf16_gemv_bf16(Context& ctx,
                    Tensor& weight,
                    const bf16* input,
                    bf16* output,
                    int M, int K) {
    const bf16* weight_ptr = weight.data_as<bf16>();
    using vec8 = sycl::vec<bf16, 8>;

    constexpr int SG = 16;
    constexpr int WG = 256;

    int K8 = K / 8;
    int rows_per_group = WG / SG;
    int num_groups = (M + rows_per_group - 1) / rows_per_group;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(num_groups) * WG, WG),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                int sg_id = sg.get_group_linear_id();
                int lane = sg.get_local_linear_id();
                int row = static_cast<int>(item.get_group(0)) * rows_per_group + sg_id;

                float acc = 0.0f;

                if (row < M) {
                    const bf16* w_row = weight_ptr + row * K;

                    // Vec8 main loop
                    for (int k8 = lane; k8 < K8; k8 += SG) {
                        int k = k8 * 8;
                        const vec8 w_vec = *reinterpret_cast<const vec8*>(w_row + k);
                        const vec8 x_vec = *reinterpret_cast<const vec8*>(input + k);

                        #pragma unroll
                        for (int v = 0; v < 8; v++) {
                            acc += static_cast<float>(w_vec[v]) *
                                   static_cast<float>(x_vec[v]);
                        }
                    }

                    // Scalar tail
                    int tail_start = K8 * 8;
                    for (int k = tail_start + lane; k < K; k += SG) {
                        acc += static_cast<float>(w_row[k]) *
                               static_cast<float>(input[k]);
                    }
                }

                float total = sycl::reduce_over_group(sg, acc, sycl::plus<>());
                if (lane == 0 && row < M) {
                    output[row] = bf16(total);
                }
            });
    });
}

} // namespace ops

bool Bnb4GroupedMoE::ExpertWeightSet::init(Context& ctx,
                                           const Bnb4BitWeightRef& weight_ref,
                                           std::string* error_message) {
    (void)ctx;
    ref = weight_ref;
    if (!ref.valid() || !ref.absmax) {
        if (error_message) *error_message = "invalid MoE weight reference";
        return false;
    }
    ready = true;
    return true;
}

bool Bnb4GroupedMoE::init(Context& ctx, const Bnb4BitWeightRef& gate_up,
                          const Bnb4BitWeightRef& down, int hidden_size,
                          int moe_intermediate_size, int num_experts,
                          std::string* error_message) {
    hidden_ = hidden_size;
    moe_intermediate_ = moe_intermediate_size;
    num_experts_ = num_experts;
    if (hidden_ <= 0 || moe_intermediate_ <= 0 || num_experts_ <= 0) {
        if (error_message) *error_message = "invalid MoE dimensions";
        return false;
    }
    if (!gate_up_.init(ctx, gate_up, error_message) ||
        !down_.init(ctx, down, error_message)) {
        return false;
    }
    gate_up_in_features_ = static_cast<int>(gate_up.logical_in_features());
    gate_up_rows_per_expert_ =
        static_cast<int>(gate_up.logical_out_features() / num_experts_);
    down_in_features_ = static_cast<int>(down.logical_in_features());
    down_rows_per_expert_ =
        static_cast<int>(down.logical_out_features() / num_experts_);
    if (gate_up_in_features_ != hidden_ ||
        gate_up_rows_per_expert_ != 2 * moe_intermediate_ ||
        down_in_features_ != moe_intermediate_ ||
        down_rows_per_expert_ != hidden_) {
        if (error_message) {
            *error_message = "MoE weight shapes do not match config";
        }
        return false;
    }
    prims_.clear();
    scratch_rows_ = 0;
    return true;
}

void Bnb4GroupedMoE::ensure_scratch(Context& ctx, int rows) {
    if (rows <= scratch_rows_) return;
    scratch_rows_ = rows;
    src_ = Tensor::allocate(ctx, {scratch_rows_, hidden_},
                            dnnl::memory::data_type::bf16);
    gate_up_out_ = Tensor::allocate(
        ctx, {scratch_rows_, 2 * moe_intermediate_},
        dnnl::memory::data_type::bf16);
    proj_ = Tensor::allocate(ctx, {scratch_rows_, moe_intermediate_},
                             dnnl::memory::data_type::bf16);
    down_out_ = Tensor::allocate(ctx, {scratch_rows_, hidden_},
                                 dnnl::memory::data_type::bf16);
    gate_up_w_ = Tensor::allocate(
        ctx, {gate_up_in_features_, 2 * moe_intermediate_},
                                  dnnl::memory::data_type::bf16);
    down_w_ = Tensor::allocate(ctx, {down_in_features_, hidden_},
                               dnnl::memory::data_type::bf16);
    tokens_dev_ = Tensor::allocate(ctx, {scratch_rows_},
                                   dnnl::memory::data_type::s32);
    weights_dev_ = Tensor::allocate(ctx, {scratch_rows_},
                                    dnnl::memory::data_type::f32);
}

void Bnb4GroupedMoE::run_matmul(Context& ctx, int rows, int in_n, int out_n,
                                uint64_t prim_key, Tensor& src, Tensor& w,
                                Tensor& dst) {
    auto& mp = prims_[prim_key];
    if (!mp.inited) {
        mp.inited = true;
        mp.src_md = dnnl::memory::desc({rows, in_n},
                                       dnnl::memory::data_type::bf16,
                                       dnnl::memory::format_tag::ab);
        mp.w_md = dnnl::memory::desc({in_n, out_n},
                                     dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);
        mp.dst_md = dnnl::memory::desc({rows, out_n},
                                       dnnl::memory::data_type::bf16,
                                       dnnl::memory::format_tag::ab);
        auto pd = dnnl::matmul::primitive_desc(ctx.engine(), mp.src_md,
                                               mp.w_md, mp.dst_md);
        mp.prim = dnnl::matmul(pd);
    }
    dnnl::memory src_mem = src.make_dnnl_memory(mp.src_md);
    dnnl::memory w_mem = w.make_dnnl_memory(mp.w_md);
    dnnl::memory dst_mem = dst.make_dnnl_memory(mp.dst_md);
    pending_memories_.push_back(src_mem);
    pending_memories_.push_back(w_mem);
    pending_memories_.push_back(dst_mem);
    mp.prim.execute(ctx.stream(),
                    {{DNNL_ARG_SRC, src_mem},
                     {DNNL_ARG_WEIGHTS, w_mem},
                     {DNNL_ARG_DST, dst_mem}});
}

void Bnb4GroupedMoE::dequant_subset(Context& ctx, const ExpertWeightSet& set,
                                    int expert, int rows_per_expert,
                                    Tensor& out) {
    const Bnb4BitWeightRef& ref = set.ref;
    const int64_t in_features = ref.logical_in_features();
    const int64_t packed_per_row = in_features / 2;
    const int blocksize = ref.quant_state.blocksize;
    const int64_t blocks_per_row = in_features / blocksize;
    const int64_t row_start = static_cast<int64_t>(expert) * rows_per_expert;
    const uint8_t* packed =
        static_cast<const uint8_t*>(ref.packed_weight->data()) +
        row_start * packed_per_row;
    const float* qmap = static_cast<const float*>(ref.quant_map->data());
    const uint8_t* absmax_codes = nullptr;
    const float* absmax_f32 = nullptr;
    const float* nested_absmax = nullptr;
    const float* nested_qmap = nullptr;
    int nested_blocksize = 1;
    float nested_offset = 0.0f;
    if (ref.quant_state.nested) {
        absmax_codes = static_cast<const uint8_t*>(ref.absmax->data());
        nested_absmax =
            static_cast<const float*>(ref.nested_absmax->data());
        nested_qmap =
            static_cast<const float*>(ref.nested_quant_map->data());
        nested_blocksize = ref.quant_state.nested_blocksize;
        nested_offset = ref.quant_state.nested_offset;
    } else {
        absmax_f32 = static_cast<const float*>(ref.absmax->data());
    }
    bf16* out_ptr = static_cast<bf16*>(out.data());
    const int64_t packed_bytes =
        static_cast<int64_t>(rows_per_expert) * packed_per_row;

    ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(packed_bytes)),
        [=](sycl::id<1> idx) {
            const int64_t byte_index = static_cast<int64_t>(idx[0]);
            // Map consecutive work-items to consecutive expert rows so the
            // two bf16 stores per byte (out[k*rows + n]) are fully coalesced.
            // The packed reads become strided (row stride K/2 bytes), which is
            // cheaper than striding 4x the bytes of bf16 stores.
            const int64_t row = byte_index % rows_per_expert;
            const int64_t byte_in_row = byte_index / rows_per_expert;
            const uint8_t packed_byte =
                packed[row * packed_per_row + byte_in_row];
            const int64_t block = (row_start + row) * blocks_per_row +
                                  byte_in_row / (blocksize / 2);
            float am = 0.0f;
            if (absmax_f32 != nullptr) {
                am = absmax_f32[block];
            } else {
                const uint8_t code = absmax_codes[block];
                am = nested_qmap[code] *
                         nested_absmax[block / nested_blocksize] +
                     nested_offset;
            }
            const int64_t col = byte_in_row * 2;
            out_ptr[col * rows_per_expert + row] =
                bf16(qmap[(packed_byte >> 4) & 0x0F] * am);
            if (col + 1 < in_features) {
                out_ptr[(col + 1) * rows_per_expert + row] =
                    bf16(qmap[packed_byte & 0x0F] * am);
            }
        });
}

void Bnb4GroupedMoE::build_route_layout(
    const std::vector<int>& expert_ids,
    const std::vector<float>& expert_weights,
    int tokens, int top_k) {
    const int total_rows = tokens * top_k;
    route_counts_.assign(static_cast<size_t>(num_experts_), 0);
    route_offsets_.assign(static_cast<size_t>(num_experts_) + 1, 0);
    route_cursors_.resize(static_cast<size_t>(num_experts_));
    route_token_ids_.resize(static_cast<size_t>(total_rows));
    route_weights_.resize(static_cast<size_t>(total_rows));
    active_experts_.clear();
    route_max_rows_ = 0;

    for (int i = 0; i < total_rows; ++i) {
        const int expert = expert_ids[static_cast<size_t>(i)];
        route_counts_[static_cast<size_t>(expert)]++;
    }

    int running_offset = 0;
    for (int e = 0; e < num_experts_; ++e) {
        const int count = route_counts_[static_cast<size_t>(e)];
        route_offsets_[static_cast<size_t>(e)] = running_offset;
        route_cursors_[static_cast<size_t>(e)] = running_offset;
        running_offset += count;
        route_max_rows_ = std::max(route_max_rows_, count);
        if (count > 0) {
            active_experts_.push_back(e);
        }
    }
    route_offsets_[static_cast<size_t>(num_experts_)] = running_offset;

    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            const int idx = t * top_k + k;
            const int expert = expert_ids[static_cast<size_t>(idx)];
            const int pos = route_cursors_[static_cast<size_t>(expert)]++;
            route_token_ids_[static_cast<size_t>(pos)] = t;
            route_weights_[static_cast<size_t>(pos)] = expert_weights[static_cast<size_t>(idx)];
        }
    }
}

void Bnb4GroupedMoE::forward(Context& ctx,
                             const std::vector<bf16>& input,
                             const std::vector<int>& expert_ids,
                             const std::vector<float>& expert_weights,
                             std::vector<bf16>& output,
                             int tokens, int top_k) {
    // Previous invocation's queue work was drained by its blocking d2h;
    // release the oneDNN memory handles from that run.
    pending_memories_.clear();
    static const bool diag = aila::env::read_flag("AILA_DG_MOE_DIAG", false);
    double diag_dequant_ms = 0.0;
    double diag_matmul_ms = 0.0;
    double diag_memcpy_ms = 0.0;
    double diag_host_ms = 0.0;
    double diag_other_ms = 0.0;
    auto diag_t0 = std::chrono::steady_clock::now();
    auto diag_mark = [&]() {
        if (!diag) return;
        const auto now = std::chrono::steady_clock::now();
        diag_other_ms += std::chrono::duration<double, std::milli>(now - diag_t0).count();
        diag_t0 = now;
    };
    auto diag_section = [&](double& bucket) {
        if (!diag) return;
        const auto now = std::chrono::steady_clock::now();
        bucket += std::chrono::duration<double, std::milli>(now - diag_t0).count();
        diag_t0 = now;
    };
    if (tokens <= 0 || top_k <= 0 ||
        static_cast<size_t>(tokens) * hidden_ != input.size() ||
        static_cast<size_t>(tokens) * top_k != expert_ids.size()) {
        throw std::runtime_error("Bnb4GroupedMoE::forward: shape mismatch");
    }
    diag_mark();

    build_route_layout(expert_ids, expert_weights, tokens, top_k);
    diag_mark();
    ensure_scratch(ctx, route_max_rows_);
    diag_mark();

    std::vector<bf16> acc(static_cast<size_t>(tokens) * hidden_, bf16(0.0f));
    std::vector<bf16> gather(static_cast<size_t>(route_max_rows_) * hidden_);
    std::vector<bf16> row_out(static_cast<size_t>(route_max_rows_) * hidden_);

    for (int expert : active_experts_) {
        const int count = route_counts_[static_cast<size_t>(expert)];
        const int offset = route_offsets_[static_cast<size_t>(expert)];
        const int* group_tokens = route_token_ids_.data() + offset;
        const float* group_weights = route_weights_.data() + offset;
        for (int r = 0; r < count; ++r) {
            const int token = group_tokens[r];
            std::memcpy(gather.data() + static_cast<size_t>(r) * hidden_,
                        input.data() + static_cast<size_t>(token) * hidden_,
                        static_cast<size_t>(hidden_) * sizeof(bf16));
        }
        ctx.memcpy_h2d(src_.data(), gather.data(),
                       static_cast<size_t>(count) * hidden_ * sizeof(bf16));
        diag_section(diag_memcpy_ms);

        dequant_subset(ctx, gate_up_, expert, gate_up_rows_per_expert_,
                       gate_up_w_);
        diag_section(diag_dequant_ms);
        run_matmul(ctx, count, gate_up_in_features_, 2 * moe_intermediate_,
                   static_cast<uint64_t>(count) * 2 + 0, src_, gate_up_w_,
                   gate_up_out_);
        diag_section(diag_matmul_ms);
        moe_gelu_tanh_gate_up_mul(ctx, gate_up_out_, proj_, count,
                                  moe_intermediate_);
        diag_section(diag_matmul_ms);

        dequant_subset(ctx, down_, expert, down_rows_per_expert_, down_w_);
        diag_section(diag_dequant_ms);
        run_matmul(ctx, count, down_in_features_, hidden_,
                   static_cast<uint64_t>(count) * 2 + 1, proj_, down_w_,
                   down_out_);
        diag_section(diag_matmul_ms);
        ctx.memcpy_d2h(row_out.data(), down_out_.data(),
                       static_cast<size_t>(count) * hidden_ * sizeof(bf16));
        diag_section(diag_memcpy_ms);

        for (int r = 0; r < count; ++r) {
            const int token = group_tokens[r];
            const float weight = group_weights[r];
            bf16* dst = acc.data() + static_cast<size_t>(token) * hidden_;
            const bf16* src_row =
                row_out.data() + static_cast<size_t>(r) * hidden_;
            for (int h = 0; h < hidden_; ++h) {
                const bf16 contribution =
                    bf16(static_cast<float>(src_row[h]) * weight);
                dst[h] = bf16(static_cast<float>(dst[h]) +
                              static_cast<float>(contribution));
            }
        }
        diag_section(diag_host_ms);
    }

    output = std::move(acc);
    diag_mark();
    if (diag) {
        std::fprintf(stderr,
            "[MoE diag] tokens=%d top_k=%d groups=%zu dequant=%.2f ms "
            "matmul=%.2f ms memcpy=%.2f ms host=%.2f ms other=%.2f ms\n",
            tokens, top_k, active_experts_.size(), diag_dequant_ms, diag_matmul_ms,
            diag_memcpy_ms, diag_host_ms, diag_other_ms);
    }
}

void Bnb4GroupedMoE::forward_gpu(Context& ctx, Tensor& input, Tensor& output,
                                 const std::vector<int>& expert_ids,
                                 const std::vector<float>& expert_weights,
                                 int tokens, int top_k) {
    pending_memories_.clear();
    if (tokens <= 0 || top_k <= 0 ||
        input.shape(0) < tokens || input.shape(1) != hidden_ ||
        output.shape(0) < tokens || output.shape(1) != hidden_ ||
        static_cast<size_t>(tokens) * top_k != expert_ids.size()) {
        throw std::runtime_error("Bnb4GroupedMoE::forward_gpu: shape mismatch");
    }

    build_route_layout(expert_ids, expert_weights, tokens, top_k);
    ensure_scratch(ctx, route_max_rows_);

    bf16* input_ptr = input.data_as<bf16>();
    bf16* output_ptr = output.data_as<bf16>();
    bf16* src_ptr = src_.data_as<bf16>();
    bf16* down_out_ptr = down_out_.data_as<bf16>();
    int* tokens_ptr = static_cast<int*>(tokens_dev_.data());
    float* weights_ptr = static_cast<float*>(weights_dev_.data());

    // acc[token][h] starts at bf16 zero (all-zero bytes).
    ctx.queue().memset(output_ptr, 0,
                       static_cast<size_t>(tokens) * hidden_ * sizeof(bf16));
    sycl::event last_token_copy;
    sycl::event last_weight_copy;
    bool have_pending_copy = false;
    for (int expert : active_experts_) {
        const int count = route_counts_[static_cast<size_t>(expert)];
        const int offset = route_offsets_[static_cast<size_t>(expert)];
        const int* group_tokens = route_token_ids_.data() + offset;
        const float* group_weights = route_weights_.data() + offset;

        // Wait only for the previous group's copies so the host can reuse the
        // source vectors; the in-order queue keeps the current copies ahead
        // of the kernels that consume them.
        if (have_pending_copy) {
            last_token_copy.wait();
            last_weight_copy.wait();
        }
        last_token_copy = ctx.memcpy_h2d_async(
            tokens_ptr, group_tokens,
            static_cast<size_t>(count) * sizeof(int));
        last_weight_copy = ctx.memcpy_h2d_async(
            weights_ptr, group_weights,
            static_cast<size_t>(count) * sizeof(float));
        have_pending_copy = true;

        moe_gather_rows(ctx, input_ptr, src_ptr, tokens_ptr, count, hidden_);
        dequant_subset(ctx, gate_up_, expert, gate_up_rows_per_expert_,
                       gate_up_w_);
        run_matmul(ctx, count, gate_up_in_features_, 2 * moe_intermediate_,
                   static_cast<uint64_t>(count) * 2 + 0, src_, gate_up_w_,
                   gate_up_out_);
        moe_gelu_tanh_gate_up_mul(ctx, gate_up_out_, proj_, count,
                                  moe_intermediate_);

        dequant_subset(ctx, down_, expert, down_rows_per_expert_, down_w_);
        run_matmul(ctx, count, down_in_features_, hidden_,
                   static_cast<uint64_t>(count) * 2 + 1, proj_, down_w_,
                   down_out_);
        moe_accumulate_rows(ctx, down_out_ptr, output_ptr, tokens_ptr,
                            weights_ptr, count, hidden_);
    }
}

void Bnb4GroupedMoE::forward_batched(Context& ctx, Tensor& input,
                                     Tensor& transfer,
                                     Tensor& token_ids_dev,
                                     const std::vector<int>& expert_ids,
                                     const std::vector<float>& expert_weights,
                                     std::vector<bf16>& output,
                                     int tokens, int top_k) {
    pending_memories_.clear();
    try {
        forward_batched_impl(ctx, input, transfer, token_ids_dev, expert_ids,
                             expert_weights, output, tokens, top_k);
    } catch (const std::exception& error) {
        std::fprintf(stderr,
            "[MoE batched] EXCEPTION tokens=%d: %s\n", tokens, error.what());
        throw;
    }
}

void run_grouped_moe_gemm_single_launch(
    Context& ctx,
    const uint8_t* packed_weights,
    const float* quant_map,
    const uint8_t* absmax_codes,
    const float* absmax_f32,
    const float* nested_absmax,
    const float* nested_qmap,
    int nested_blocksize,
    float nested_offset,
    const bf16* input_transfer,
    bf16* output_transfer,
    const Bnb4GroupedMoE::GroupInfo* group_infos_dev,
    int num_groups,
    int in_features,
    int out_features,
    int blocksize,
    int rows_per_expert) {
    constexpr int BM = 32, BN = 64, BK = 16, TM = 8, TN = 8, TK = 16, SG = 8;
    constexpr int SG_M = BM / TM;
    constexpr int SG_N = BN / TN;
    constexpr int NUM_SGS = SG_M * SG_N;
    constexpr int WG_SIZE = NUM_SGS * SG;

    const int grid_n = (out_features + BN - 1) / BN;
    const int total_wgs = num_groups * grid_n;

    const int K = in_features;
    const int N = out_features;
    const int packed_bytes_per_row = K / 2;
    const int blocks_per_row = K / blocksize;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<bf16, 1> Bs(sycl::range<1>(TK * BN), cgh);
        sycl::local_accessor<float, 1> c_slm(sycl::range<1>(NUM_SGS * TM * TN), cgh);
        sycl::local_accessor<float, 1> qmap_slm(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(total_wgs) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int wg_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / SG;
                const int sg_row = sg_id / SG_N;
                const int sg_col = sg_id % SG_N;

                const int group_idx = wg_idx / grid_n;
                const int bn = wg_idx % grid_n;
                const int n0 = bn * BN;

                const auto info = group_infos_dev[group_idx];
                const int expert = info.expert_id;
                const int token_offset = info.token_offset;
                const int M = info.token_count;

                const size_t expert_row_offset = static_cast<size_t>(expert) * rows_per_expert;
                const uint8_t* expert_packed = packed_weights + expert_row_offset * packed_bytes_per_row;
                const bf16* expert_input = input_transfer + static_cast<size_t>(token_offset) * K;
                bf16* expert_output = output_transfer + static_cast<size_t>(token_offset) * N;

                if (lid < 16) {
                    qmap_slm[lid] = quant_map[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
                joint_matrix_fill(sg, sub_c, 0.0f);

                for (int kb = 0; kb < K; kb += BK) {
                    #pragma unroll
                    for (int i = lid; i < BM * BK; i += WG_SIZE) {
                        const int r = i / BK;
                        const int c = i % BK;
                        if (r < M && kb + c < K) {
                            As[i] = expert_input[static_cast<size_t>(r) * K + kb + c];
                        } else {
                            As[i] = bf16(0.0f);
                        }
                    }

                    #pragma unroll
                    for (int i = lid; i < BN * (TK / 2); i += WG_SIZE) {
                        const int col = i / (TK / 2);
                        const int kh = i % (TK / 2);
                        const int src_n = n0 + col;
                        if (src_n < N && kb + kh * 2 + 1 < K) {
                            const uint8_t byte = expert_packed[static_cast<size_t>(src_n) * packed_bytes_per_row + (kb / 2) + kh];
                            const size_t block =
                                (expert_row_offset + src_n) * blocks_per_row +
                                (kb + kh * 2) / blocksize;
                            const float am = (absmax_f32 != nullptr)
                                ? absmax_f32[block]
                                : (nested_qmap[absmax_codes[block]] *
                                       nested_absmax[block / nested_blocksize] +
                                   nested_offset);
                            const uint8_t hi = (byte >> 4) & 0x0F;
                            const uint8_t lo = byte & 0x0F;
                            Bs[col * TK + kh * 2 + 0] = bf16(qmap_slm[hi] * am);
                            Bs[col * TK + kh * 2 + 1] = bf16(qmap_slm[lo] * am);
                        } else if (src_n < N && kb + kh * 2 < K) {
                            const uint8_t byte = expert_packed[static_cast<size_t>(src_n) * packed_bytes_per_row + (kb / 2) + kh];
                            const size_t block =
                                (expert_row_offset + src_n) * blocks_per_row +
                                (kb + kh * 2) / blocksize;
                            const float am = (absmax_f32 != nullptr)
                                ? absmax_f32[block]
                                : (nested_qmap[absmax_codes[block]] *
                                       nested_absmax[block / nested_blocksize] +
                                   nested_offset);
                            const uint8_t hi = (byte >> 4) & 0x0F;
                            Bs[col * TK + kh * 2 + 0] = bf16(qmap_slm[hi] * am);
                            Bs[col * TK + kh * 2 + 1] = bf16(0.0f);
                        } else {
                            Bs[col * TK + kh * 2 + 0] = bf16(0.0f);
                            Bs[col * TK + kh * 2 + 1] = bf16(0.0f);
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;
                    joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major> sub_b;

                    auto pA = As.get_multi_ptr<sycl::access::decorated::no>() + (sg_row * TM) * BK;
                    auto pB = Bs.get_multi_ptr<sycl::access::decorated::no>() + (sg_col * TN) * TK;

                    joint_matrix_load(sg, sub_a, pA, BK);
                    joint_matrix_load(sg, sub_b, pB, TK);
                    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);

                    item.barrier(sycl::access::fence_space::local_space);
                }

                auto pC_slm = c_slm.get_multi_ptr<sycl::access::decorated::no>() + sg_id * (TM * TN);
                joint_matrix_store(sg, sub_c, pC_slm, TN, layout::row_major);
                item.barrier(sycl::access::fence_space::local_space);

                for (int i = lid; i < NUM_SGS * TM * TN; i += WG_SIZE) {
                    const int s = i / (TM * TN);
                    const int rem = i % (TM * TN);
                    const int r = rem / TN;
                    const int c = rem % TN;
                    const int sr = s / SG_N;
                    const int sc = s % SG_N;
                    const int gm = sr * TM + r;
                    const int gn = n0 + sc * TN + c;
                    if (gm < M && gn < N) {
                        expert_output[static_cast<size_t>(gm) * N + gn] = bf16(c_slm[i]);
                    }
                }
            });
    });
}

void moe_gelu_tanh_gate_up_mul_flat(Context& ctx, const bf16* gate_up_out,
                                    bf16* proj, int rows, int inter_dim) {
    constexpr float kAlpha = 0.7978845608028654f;
    const int total = rows * inter_dim;
    ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(total)),
        [=](sycl::id<1> idx) {
            const int i = static_cast<int>(idx[0]);
            const int r = i / inter_dim;
            const int c = i % inter_dim;
            const size_t gu_offset = static_cast<size_t>(r) * (2 * inter_dim);
            const float g = static_cast<float>(gate_up_out[gu_offset + c]);
            const float u = static_cast<float>(gate_up_out[gu_offset + inter_dim + c]);
            const float v = kAlpha * (g + 0.044715f * g * g * g);
            proj[i] = bf16(0.5f * g * (1.0f + sycl::tanh(v)) * u);
        });
}

void run_moe_accumulate_gpu(
    Context& ctx,
    const bf16* transfer,
    const int* dev_indices,
    const float* dev_weights,
    bf16* output,
    int tokens,
    int hidden,
    int top_k) {
    if (hidden % 8 == 0) {
        constexpr int WG_SIZE = 256;
        constexpr int VEC = 8;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<int, 1> s_indices(sycl::range<1>(8), cgh);
            sycl::local_accessor<float, 1> s_weights(sycl::range<1>(8), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(static_cast<size_t>(tokens) * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    const int t = static_cast<int>(item.get_group(0));
                    const int lid = static_cast<int>(item.get_local_id(0));

                    if (lid < 8) {
                        s_indices[lid] = (lid < top_k) ? dev_indices[t * top_k + lid] : 0;
                        s_weights[lid] = (lid < top_k) ? dev_weights[t * top_k + lid] : 0.0f;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    int rows[8];
                    float weights[8];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        rows[k] = s_indices[k];
                        weights[k] = s_weights[k];
                    }

                    for (int h_vec = lid; h_vec < hidden / VEC; h_vec += WG_SIZE) {
                        const int h = h_vec * VEC;
                        sycl::vec<bf16, VEC> out_v;
                        #pragma unroll
                        for (int v = 0; v < VEC; ++v) {
                            out_v[v] = bf16(0.0f);
                        }

                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            if (k < top_k) {
                                const size_t row_offset = static_cast<size_t>(rows[k]) * hidden + h;
                                auto in_v = *reinterpret_cast<const sycl::vec<bf16, VEC>*>(transfer + row_offset);
                                const float w = weights[k];
                                #pragma unroll
                                for (int v = 0; v < VEC; ++v) {
                                    const bf16 contribution = bf16(static_cast<float>(in_v[v]) * w);
                                    out_v[v] = bf16(static_cast<float>(out_v[v]) + static_cast<float>(contribution));
                                }
                            }
                        }

                        *reinterpret_cast<sycl::vec<bf16, VEC>*>(output + static_cast<size_t>(t) * hidden + h) = out_v;
                    }
                });
        });
        return;
    }
    ctx.queue().parallel_for(
        sycl::range<2>(static_cast<size_t>(tokens), static_cast<size_t>(hidden)),
        [=](sycl::item<2> item) {
            const int t = static_cast<int>(item.get_id(0));
            const int h = static_cast<int>(item.get_id(1));
            bf16 acc = bf16(0.0f);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k < top_k) {
                    const int row = dev_indices[t * top_k + k];
                    const float w = dev_weights[t * top_k + k];
                    const bf16 contribution = bf16(static_cast<float>(transfer[static_cast<size_t>(row) * hidden + h]) * w);
                    acc = bf16(static_cast<float>(acc) + static_cast<float>(contribution));
                }
            }
            output[static_cast<size_t>(t) * hidden + h] = acc;
        });
}

void Bnb4GroupedMoE::forward_batched_gpu(
    Context& ctx, Tensor& input, Tensor& transfer, Tensor& token_ids_dev,
    Tensor& output, const std::vector<int>& expert_ids,
    const std::vector<float>& expert_weights, int tokens, int top_k) {
    if (tokens <= 0 || top_k <= 0 ||
        static_cast<size_t>(tokens) * top_k != expert_ids.size()) {
        throw std::runtime_error("Bnb4GroupedMoE::forward_batched_gpu: shape mismatch");
    }
    const int total_rows = tokens * top_k;
    if (transfer.shape(0) < total_rows || transfer.shape(1) != hidden_ ||
        token_ids_dev.shape(0) < total_rows ||
        output.shape(0) < tokens || output.shape(1) != hidden_) {
        throw std::runtime_error(
            "Bnb4GroupedMoE::forward_batched_gpu: buffer too small");
    }

    static const bool grouped_xmx =
        aila::env::read_flag("AILA_DG_MOE_GROUPED_XMX", false);
    if (!grouped_xmx) {
        pending_memories_.clear();
        build_route_layout(expert_ids, expert_weights, tokens, top_k);
        ensure_scratch(ctx, route_max_rows_);

        ctx.memcpy_h2d(token_ids_dev.data(), route_token_ids_.data(),
                       static_cast<size_t>(total_rows) * sizeof(int));

        const bf16* input_ptr = input.data_as<bf16>();
        bf16* transfer_ptr = transfer.data_as<bf16>();
        moe_gather_rows_flat(ctx, input_ptr, transfer_ptr,
                             static_cast<int*>(token_ids_dev.data()),
                             total_rows, hidden_);

        for (int expert : active_experts_) {
            const int count = route_counts_[static_cast<size_t>(expert)];
            const int offset = route_offsets_[static_cast<size_t>(expert)];
            Tensor src_view = Tensor::view(
                ctx, transfer_ptr + static_cast<size_t>(offset) * hidden_,
                {count, hidden_}, dnnl::memory::data_type::bf16);
            Tensor dst_view = Tensor::view(
                ctx, transfer_ptr + static_cast<size_t>(offset) * hidden_,
                {count, hidden_}, dnnl::memory::data_type::bf16);

            dequant_subset(ctx, gate_up_, expert, gate_up_rows_per_expert_,
                           gate_up_w_);
            run_matmul(ctx, count, gate_up_in_features_, 2 * moe_intermediate_,
                       static_cast<uint64_t>(count) * 2 + 0, src_view,
                       gate_up_w_, gate_up_out_);
            moe_gelu_tanh_gate_up_mul(ctx, gate_up_out_, proj_, count,
                                      moe_intermediate_);

            dequant_subset(ctx, down_, expert, down_rows_per_expert_, down_w_);
            run_matmul(ctx, count, down_in_features_, hidden_,
                       static_cast<uint64_t>(count) * 2 + 1, proj_, down_w_,
                       dst_view);
        }

        std::vector<int> token_transfer_indices(static_cast<size_t>(total_rows));
        std::vector<float> token_transfer_weights(static_cast<size_t>(total_rows));
        std::vector<int> token_counts(static_cast<size_t>(tokens), 0);
        for (int expert : active_experts_) {
            const int count = route_counts_[static_cast<size_t>(expert)];
            const int offset = route_offsets_[static_cast<size_t>(expert)];
            const int* group_tokens = route_token_ids_.data() + offset;
            const float* group_weights = route_weights_.data() + offset;
            for (int r = 0; r < count; ++r) {
                const int token = group_tokens[r];
                const int k = token_counts[static_cast<size_t>(token)]++;
                token_transfer_indices[static_cast<size_t>(token * top_k + k)] = offset + r;
                token_transfer_weights[static_cast<size_t>(token * top_k + k)] = group_weights[r];
            }
        }
        if (total_rows > accumulate_scratch_rows_) {
            accumulate_scratch_rows_ = total_rows;
            transfer_weights_dev_ = Tensor::allocate(
                ctx, {accumulate_scratch_rows_},
                dnnl::memory::data_type::f32);
        }
        ctx.memcpy_h2d(token_ids_dev.data(), token_transfer_indices.data(),
                       static_cast<size_t>(total_rows) * sizeof(int));
        ctx.memcpy_h2d(transfer_weights_dev_.data(),
                       token_transfer_weights.data(),
                       static_cast<size_t>(total_rows) * sizeof(float));

        run_moe_accumulate_gpu(
            ctx, transfer_ptr,
            static_cast<const int*>(token_ids_dev.data()),
            static_cast<const float*>(transfer_weights_dev_.data()),
            output.data_as<bf16>(), tokens, hidden_, top_k);
        return;
    }

    std::map<int, std::vector<std::pair<int, float>>> groups;
    int max_rows = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            const int idx = t * top_k + k;
            const int expert = expert_ids[static_cast<size_t>(idx)];
            auto& list = groups[expert];
            list.emplace_back(t, expert_weights[static_cast<size_t>(idx)]);
            max_rows = std::max(max_rows, static_cast<int>(list.size()));
        }
    }
    ensure_scratch(ctx, max_rows);

    std::vector<int> flat_token_ids;
    flat_token_ids.reserve(static_cast<size_t>(total_rows));
    std::vector<int> token_transfer_indices(static_cast<size_t>(total_rows));
    std::vector<float> token_transfer_weights(static_cast<size_t>(total_rows));
    std::vector<int> token_counts(static_cast<size_t>(tokens), 0);

    std::vector<GroupInfo> group_infos;
    group_infos.reserve(groups.size());

    int current_offset = 0;
    for (const auto& [expert, rows] : groups) {
        int count = static_cast<int>(rows.size());
        int remaining = count;
        int chunk_offset = current_offset;
        while (remaining > 0) {
            int take = std::min(remaining, 32);
            group_infos.push_back({expert, chunk_offset, take});
            chunk_offset += take;
            remaining -= take;
        }

        for (const auto& [token, weight] : rows) {
            int row_idx = static_cast<int>(flat_token_ids.size());
            flat_token_ids.push_back(token);
            int k = token_counts[token]++;
            token_transfer_indices[token * top_k + k] = row_idx;
            token_transfer_weights[token * top_k + k] = weight;
        }
        current_offset += count;
    }

    if (total_rows > batched_scratch_rows_ ||
        static_cast<int>(group_infos.size()) > batched_scratch_groups_) {
        batched_scratch_rows_ = total_rows;
        batched_scratch_groups_ = static_cast<int>(group_infos.size());
        batched_gate_up_out_ = Tensor::allocate(
            ctx, {batched_scratch_rows_, 2 * moe_intermediate_},
            dnnl::memory::data_type::bf16);
        batched_proj_ = Tensor::allocate(
            ctx, {batched_scratch_rows_, moe_intermediate_},
            dnnl::memory::data_type::bf16);
        group_infos_dev_ = Tensor::allocate(
            ctx,
            {static_cast<int64_t>(batched_scratch_groups_) *
             static_cast<int64_t>(sizeof(GroupInfo))},
            dnnl::memory::data_type::u8);
        transfer_indices_dev_ = Tensor::allocate(
            ctx, {batched_scratch_rows_}, dnnl::memory::data_type::s32);
        transfer_weights_dev_ = Tensor::allocate(
            ctx, {batched_scratch_rows_}, dnnl::memory::data_type::f32);
    }

    ctx.memcpy_h2d(token_ids_dev.data(), flat_token_ids.data(),
                   static_cast<size_t>(total_rows) * sizeof(int));
    ctx.memcpy_h2d(group_infos_dev_.data(), group_infos.data(),
                   group_infos.size() * sizeof(GroupInfo));
    ctx.memcpy_h2d(transfer_indices_dev_.data(), token_transfer_indices.data(),
                   static_cast<size_t>(total_rows) * sizeof(int));
    ctx.memcpy_h2d(transfer_weights_dev_.data(), token_transfer_weights.data(),
                   static_cast<size_t>(total_rows) * sizeof(float));

    const bf16* input_ptr = input.data_as<bf16>();
    bf16* transfer_ptr = transfer.data_as<bf16>();
    moe_gather_rows_flat(ctx, input_ptr, transfer_ptr,
                         static_cast<int*>(token_ids_dev.data()),
                         total_rows, hidden_);

    // 1. Single kernel launch: Grouped Gate+Up GEMM for ALL active experts
    const int gu_blocksize = gate_up_.ref.quant_state.blocksize;
    const uint8_t* gu_absmax_codes = gate_up_.ref.quant_state.nested ? gate_up_.ref.absmax->data_as<uint8_t>() : nullptr;
    const float* gu_absmax_f32 = gate_up_.ref.quant_state.nested ? nullptr : gate_up_.ref.absmax->data_as<float>();
    const float* gu_nested_absmax = gate_up_.ref.quant_state.nested ? gate_up_.ref.nested_absmax->data_as<float>() : nullptr;
    const float* gu_nested_qmap = gate_up_.ref.quant_state.nested ? gate_up_.ref.nested_quant_map->data_as<float>() : nullptr;

    run_grouped_moe_gemm_single_launch(
        ctx,
        gate_up_.ref.packed_weight->data_as<uint8_t>(),
        gate_up_.ref.quant_map->data_as<float>(),
        gu_absmax_codes,
        gu_absmax_f32,
        gu_nested_absmax,
        gu_nested_qmap,
        gate_up_.ref.quant_state.nested_blocksize,
        gate_up_.ref.quant_state.nested_offset,
        transfer_ptr,
        batched_gate_up_out_.data_as<bf16>(),
        reinterpret_cast<const GroupInfo*>(group_infos_dev_.data()),
        static_cast<int>(group_infos.size()),
        gate_up_in_features_,
        2 * moe_intermediate_,
        gu_blocksize,
        gate_up_rows_per_expert_);

    // 2. Single kernel launch: Flat GELU activation
    moe_gelu_tanh_gate_up_mul_flat(
        ctx,
        batched_gate_up_out_.data_as<bf16>(),
        batched_proj_.data_as<bf16>(),
        total_rows,
        moe_intermediate_);

    // 3. Single kernel launch: Grouped Down GEMM for ALL active experts
    const int down_blocksize = down_.ref.quant_state.blocksize;
    const uint8_t* down_absmax_codes = down_.ref.quant_state.nested ? down_.ref.absmax->data_as<uint8_t>() : nullptr;
    const float* down_absmax_f32 = down_.ref.quant_state.nested ? nullptr : down_.ref.absmax->data_as<float>();
    const float* down_nested_absmax = down_.ref.quant_state.nested ? down_.ref.nested_absmax->data_as<float>() : nullptr;
    const float* down_nested_qmap = down_.ref.quant_state.nested ? down_.ref.nested_quant_map->data_as<float>() : nullptr;

    run_grouped_moe_gemm_single_launch(
        ctx,
        down_.ref.packed_weight->data_as<uint8_t>(),
        down_.ref.quant_map->data_as<float>(),
        down_absmax_codes,
        down_absmax_f32,
        down_nested_absmax,
        down_nested_qmap,
        down_.ref.quant_state.nested_blocksize,
        down_.ref.quant_state.nested_offset,
        batched_proj_.data_as<bf16>(),
        transfer_ptr,
        reinterpret_cast<const GroupInfo*>(group_infos_dev_.data()),
        static_cast<int>(group_infos.size()),
        down_in_features_,
        hidden_,
        down_blocksize,
        down_rows_per_expert_);

    // 4. Single kernel launch: GPU row accumulation directly into output
    run_moe_accumulate_gpu(
        ctx,
        transfer_ptr,
        static_cast<const int*>(transfer_indices_dev_.data()),
        static_cast<const float*>(transfer_weights_dev_.data()),
        output.data_as<bf16>(),
        tokens,
        hidden_,
        top_k);
}

void Bnb4GroupedMoE::forward_batched_impl(
    Context& ctx, Tensor& input, Tensor& transfer, Tensor& token_ids_dev,
    const std::vector<int>& expert_ids,
    const std::vector<float>& expert_weights, std::vector<bf16>& output,
    int tokens, int top_k) {
    static const bool moe_gpu =
        aila::env::read_flag("AILA_DG_MOE_GPU", false);
    if (moe_gpu) {
        Tensor dev_out = Tensor::allocate(ctx, {tokens, hidden_},
                                          dnnl::memory::data_type::bf16);
        forward_batched_gpu(ctx, input, transfer, token_ids_dev, dev_out,
                            expert_ids, expert_weights, tokens, top_k);
        output.resize(static_cast<size_t>(tokens) * hidden_);
        ctx.memcpy_d2h(output.data(), dev_out.data(),
                       output.size() * sizeof(bf16));
        return;
    }

    if (tokens <= 0 || top_k <= 0 ||
        static_cast<size_t>(tokens) * top_k != expert_ids.size()) {
        throw std::runtime_error("Bnb4GroupedMoE::forward_batched: shape mismatch");
    }
    const int total_rows = tokens * top_k;
    if (transfer.shape(0) < total_rows || transfer.shape(1) != hidden_ ||
        token_ids_dev.shape(0) < total_rows) {
        throw std::runtime_error(
            "Bnb4GroupedMoE::forward_batched: transfer buffer too small");
    }

    build_route_layout(expert_ids, expert_weights, tokens, top_k);
    ensure_scratch(ctx, route_max_rows_);

    ctx.memcpy_h2d(token_ids_dev.data(), route_token_ids_.data(),
                   static_cast<size_t>(total_rows) * sizeof(int));

    const bf16* input_ptr = input.data_as<bf16>();
    bf16* transfer_ptr = transfer.data_as<bf16>();
    moe_gather_rows_flat(ctx, input_ptr, transfer_ptr,
                         static_cast<int*>(token_ids_dev.data()),
                         total_rows, hidden_);

    std::vector<bf16> acc(static_cast<size_t>(tokens) * hidden_, bf16(0.0f));
    std::vector<bf16> host_out(static_cast<size_t>(total_rows) * hidden_);
    for (int expert : active_experts_) {
        const int count = route_counts_[static_cast<size_t>(expert)];
        const int offset = route_offsets_[static_cast<size_t>(expert)];
        Tensor src_view = Tensor::view(
            ctx, transfer_ptr + static_cast<size_t>(offset) * hidden_,
            {count, hidden_}, dnnl::memory::data_type::bf16);
        Tensor dst_view = Tensor::view(
            ctx, transfer_ptr + static_cast<size_t>(offset) * hidden_,
            {count, hidden_}, dnnl::memory::data_type::bf16);

        dequant_subset(ctx, gate_up_, expert, gate_up_rows_per_expert_,
                       gate_up_w_);
        run_matmul(ctx, count, gate_up_in_features_, 2 * moe_intermediate_,
                   static_cast<uint64_t>(count) * 2 + 0, src_view,
                   gate_up_w_, gate_up_out_);
        moe_gelu_tanh_gate_up_mul(ctx, gate_up_out_, proj_, count,
                                  moe_intermediate_);

        dequant_subset(ctx, down_, expert, down_rows_per_expert_, down_w_);
        run_matmul(ctx, count, down_in_features_, hidden_,
                   static_cast<uint64_t>(count) * 2 + 1, proj_, down_w_,
                   dst_view);
    }

    ctx.memcpy_d2h(host_out.data(), transfer.data(),
                   host_out.size() * sizeof(bf16));

    for (int expert : active_experts_) {
        const int count = route_counts_[static_cast<size_t>(expert)];
        const int offset = route_offsets_[static_cast<size_t>(expert)];
        const bf16* src_row =
            host_out.data() + static_cast<size_t>(offset) * hidden_;
        const int* group_tokens = route_token_ids_.data() + offset;
        const float* group_weights = route_weights_.data() + offset;
        for (int r = 0; r < count; ++r) {
            const int token = group_tokens[r];
            const float weight = group_weights[r];
            bf16* dst = acc.data() + static_cast<size_t>(token) * hidden_;
            for (int h = 0; h < hidden_; ++h) {
                const bf16 contribution =
                    bf16(static_cast<float>(src_row[static_cast<size_t>(r) * hidden_ + h]) *
                         weight);
                dst[h] = bf16(static_cast<float>(dst[h]) +
                              static_cast<float>(contribution));
            }
        }
    }

    output = std::move(acc);
}
