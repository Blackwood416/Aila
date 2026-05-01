#include "Bnb4BitLinear.hpp"
#include "../utils/EnvUtils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

int round_up_seq(int value, int granularity) {
    return ((value + granularity - 1) / granularity) * granularity;
}

// Fused NF4 dequant + matmul for prefill (M > 1).
// Computes C[M,N] = A[M,K] @ dequantize_nf4(B_packed[N,K]), B absmax per block.
void packed_nf4_gemm_bf16(Context& ctx,
                          const uint8_t* packed_ptr,
                          const float* quant_map_ptr,
                          const float* absmax_ptr,
                          const bf16* input_ptr,
                          bf16* output_ptr,
                          int M, int N, int K,
                          int blocksize) {
    const int packed_bytes_per_row = K / 2;
    const int blocks_per_row = K / blocksize;

    constexpr int BM = 128, BN = 128, BK = 128;
    constexpr int TM = 8, TN = 8;  // per-thread output tile
    constexpr int TILE_M = 16, TILE_N = 16;  // thread grid in work-group
    constexpr int WG_SIZE = TILE_M * TILE_N;

    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bf16, 1> As(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        sycl::local_accessor<float, 1> B_absmax_slm(sycl::range<1>(BN), cgh);
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

                const int rows_per_thr = BM / TILE_M;
                const int cols_per_thr = BN / TILE_N;

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
                        const int k_half = i % (a_cols / 2);
                        const int bn_idx = i / (a_cols / 2);
                        const int src_n = n0 + bn_idx;
                        if (src_n < N) {
                            B_nf4_slm[i] = packed_ptr[src_n * packed_bytes_per_row + kb / 2 + k_half];
                        } else {
                            B_nf4_slm[i] = 0x77;
                        }
                    }

                    // Load absmax for each N column in this tile
                    if (lid < BN) {
                        const int src_n = n0 + lid;
                        B_absmax_slm[lid] = (src_n < N)
                            ? absmax_ptr[src_n * blocks_per_row + kb / blocksize] : 0.0f;
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
                            b_dequant[ni - n_local] = qmap[nf4] * B_absmax_slm[ni];
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
                    if (m >= M) continue;
                    for (int ni = 0; ni < TN; ++ni) {
                        const int n = n0 + tx * TN + ni;
                        if (n >= N) continue;
                        output_ptr[m * N + n] = bf16(C[mi][ni]);
                    }
                }
            });
    });
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
        sycl::local_accessor<bf16, 1> input_slm(sycl::range<1>(in_features), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) {
            const int lid = static_cast<int>(item.get_local_id(0));
            const int sg_id = lid / sub_group_size;
            const int sg_lane = lid % sub_group_size;
            const int row = static_cast<int>(item.get_group(0)) * static_cast<int>(rows_per_group) + sg_id;

            if (lid < 16) qmap[lid] = quant_map_ptr[lid];
            for (int i = lid; i < in_features; i += static_cast<int>(wg_size)) {
                input_slm[i] = input_ptr[i];
            }
            item.barrier(sycl::access::fence_space::local_space);

            float gate_acc = 0.0f;
            float up_acc = 0.0f;
            if (row < intermediate_size) {
                const int row_byte_base = row * packed_bytes_per_row;
                const int row_block_base = row * blocks_per_row;
                for (int byte_offset = sg_lane * 4; byte_offset + 3 < packed_bytes_per_row;
                     byte_offset += static_cast<int>(sub_group_size) * 4) {
                    const uint32_t g4 = *reinterpret_cast<const uint32_t*>(gate_packed + row_byte_base + byte_offset);
                    const uint32_t u4 = *reinterpret_cast<const uint32_t*>(up_packed + row_byte_base + byte_offset);
                    const float g_absmax = gate_absmax[row_block_base + (byte_offset / (blocksize / 2))];
                    const float u_absmax = up_absmax[row_block_base + (byte_offset / (blocksize / 2))];
                    const int input_base = byte_offset * 2;
                    const float i0 = static_cast<float>(input_slm[input_base + 0]);
                    const float i1 = static_cast<float>(input_slm[input_base + 1]);
                    const float i2 = static_cast<float>(input_slm[input_base + 2]);
                    const float i3 = static_cast<float>(input_slm[input_base + 3]);
                    const float i4 = static_cast<float>(input_slm[input_base + 4]);
                    const float i5 = static_cast<float>(input_slm[input_base + 5]);
                    const float i6 = static_cast<float>(input_slm[input_base + 6]);
                    const float i7 = static_cast<float>(input_slm[input_base + 7]);
                    gate_acc += i0 * qmap[static_cast<uint8_t>(g4) >> 4] * g_absmax;
                    gate_acc += i1 * qmap[static_cast<uint8_t>(g4) & 0x0F] * g_absmax;
                    gate_acc += i2 * qmap[static_cast<uint8_t>(g4 >> 8) >> 4] * g_absmax;
                    gate_acc += i3 * qmap[static_cast<uint8_t>(g4 >> 8) & 0x0F] * g_absmax;
                    gate_acc += i4 * qmap[static_cast<uint8_t>(g4 >> 16) >> 4] * g_absmax;
                    gate_acc += i5 * qmap[static_cast<uint8_t>(g4 >> 16) & 0x0F] * g_absmax;
                    gate_acc += i6 * qmap[static_cast<uint8_t>(g4 >> 24) >> 4] * g_absmax;
                    gate_acc += i7 * qmap[static_cast<uint8_t>(g4 >> 24) & 0x0F] * g_absmax;
                    up_acc += i0 * qmap[static_cast<uint8_t>(u4) >> 4] * u_absmax;
                    up_acc += i1 * qmap[static_cast<uint8_t>(u4) & 0x0F] * u_absmax;
                    up_acc += i2 * qmap[static_cast<uint8_t>(u4 >> 8) >> 4] * u_absmax;
                    up_acc += i3 * qmap[static_cast<uint8_t>(u4 >> 8) & 0x0F] * u_absmax;
                    up_acc += i4 * qmap[static_cast<uint8_t>(u4 >> 16) >> 4] * u_absmax;
                    up_acc += i5 * qmap[static_cast<uint8_t>(u4 >> 16) & 0x0F] * u_absmax;
                    up_acc += i6 * qmap[static_cast<uint8_t>(u4 >> 24) >> 4] * u_absmax;
                    up_acc += i7 * qmap[static_cast<uint8_t>(u4 >> 24) & 0x0F] * u_absmax;
                }
                for (int byte_offset = sg_lane; byte_offset < packed_bytes_per_row;
                     byte_offset += static_cast<int>(sub_group_size)) {
                    const uint8_t gb = gate_packed[row_byte_base + byte_offset];
                    const uint8_t ub = up_packed[row_byte_base + byte_offset];
                    const float g_absmax = gate_absmax[row_block_base + (byte_offset / (blocksize / 2))];
                    const float u_absmax = up_absmax[row_block_base + (byte_offset / (blocksize / 2))];
                    const int input_index = byte_offset * 2;
                    const float i0 = static_cast<float>(input_slm[input_index]);
                    const float i1 = static_cast<float>(input_slm[input_index + 1]);
                    gate_acc += i0 * qmap[gb >> 4] * g_absmax;
                    gate_acc += i1 * qmap[gb & 0x0F] * g_absmax;
                    up_acc += i0 * qmap[ub >> 4] * u_absmax;
                    up_acc += i1 * qmap[ub & 0x0F] * u_absmax;
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
                          int out_features) {
    const uint8_t* packed_ptr = static_cast<const uint8_t*>(weight.packed_weight->data());
    const float* quant_map_ptr = static_cast<const float*>(weight.quant_map->data());
    const float* absmax_ptr = static_cast<const float*>(absmax_f32.data());
    const bf16* input_ptr = static_cast<const bf16*>(input.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    const int blocksize = weight.quant_state.blocksize;
    const int packed_bytes_per_row = in_features / 2;
    const int packed_bytes_per_block = blocksize / 2;
    const int blocks_per_row = in_features / blocksize;
    const bool use_sg16 = (packed_bytes_per_row >= 768);

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
                    partial += static_cast<float>(input_ptr[input_index]) *
                               quant_map_cache[(packed >> 4) & 0x0F] *
                               absmax;
                    partial += static_cast<float>(input_ptr[input_index + 1]) *
                               quant_map_cache[packed & 0x0F] *
                               absmax;
                }

                const float sum = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
                if (lid == 0) {
                    out_ptr[output_index] = bf16(sum);
                }
            });
        });
        return;
    }

    const size_t wg_size = 256;
    const size_t sub_group_size = 16;
    const size_t num_sub_groups = wg_size / sub_group_size;
    const size_t rows_per_group = num_sub_groups;
    const size_t num_groups = (static_cast<size_t>(out_features) + rows_per_group - 1) / rows_per_group;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> quant_map_cache(sycl::range<1>(16), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) {
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
                const int row_byte_base = row * packed_bytes_per_row;
                const int row_block_base = row * blocks_per_row;
                int byte_offset = sg_lane * 4;
                for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += static_cast<int>(sub_group_size) * 4) {
                    const uint32_t packed4 = *reinterpret_cast<const uint32_t*>(packed_ptr + row_byte_base + byte_offset);
                    const float absmax = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const int input_base = byte_offset * 2;
                    const uint8_t b0 = static_cast<uint8_t>(packed4);
                    const uint8_t b1 = static_cast<uint8_t>(packed4 >> 8);
                    const uint8_t b2 = static_cast<uint8_t>(packed4 >> 16);
                    const uint8_t b3 = static_cast<uint8_t>(packed4 >> 24);
                    const float q0 = quant_map_cache[b0 >> 4] * absmax;
                    const float q1 = quant_map_cache[b0 & 0x0F] * absmax;
                    const float q2 = quant_map_cache[b1 >> 4] * absmax;
                    const float q3 = quant_map_cache[b1 & 0x0F] * absmax;
                    const float q4 = quant_map_cache[b2 >> 4] * absmax;
                    const float q5 = quant_map_cache[b2 & 0x0F] * absmax;
                    const float q6 = quant_map_cache[b3 >> 4] * absmax;
                    const float q7 = quant_map_cache[b3 & 0x0F] * absmax;
                    partial += static_cast<float>(input_ptr[input_base + 0]) * q0;
                    partial += static_cast<float>(input_ptr[input_base + 1]) * q1;
                    partial += static_cast<float>(input_ptr[input_base + 2]) * q2;
                    partial += static_cast<float>(input_ptr[input_base + 3]) * q3;
                    partial += static_cast<float>(input_ptr[input_base + 4]) * q4;
                    partial += static_cast<float>(input_ptr[input_base + 5]) * q5;
                    partial += static_cast<float>(input_ptr[input_base + 6]) * q6;
                    partial += static_cast<float>(input_ptr[input_base + 7]) * q7;
                }
                for (; byte_offset < packed_bytes_per_row; byte_offset += static_cast<int>(sub_group_size)) {
                    const uint8_t packed = packed_ptr[row_byte_base + byte_offset];
                    const float absmax = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const int input_index = byte_offset * 2;
                    partial += static_cast<float>(input_ptr[input_index]) *
                               quant_map_cache[(packed >> 4) & 0x0F] * absmax;
                    partial += static_cast<float>(input_ptr[input_index + 1]) *
                               quant_map_cache[packed & 0x0F] * absmax;
                }
            }

            float sum = sycl::reduce_over_group(item.get_sub_group(), partial, sycl::plus<>());
            if (sg_lane == 0 && row < out_features) {
                out_ptr[row] = bf16(sum);
            }
        });
    });
}

void packed_nf4_gemv_bf16_2way(Context& ctx,
                               const uint8_t* packed0,
                               const float* quant_map0,
                               const float* absmax0,
                               bf16* out0,
                               int out_features0,
                               const uint8_t* packed1,
                               const float* quant_map1,
                               const float* absmax1,
                               bf16* out1,
                               int out_features1,
                               const bf16* input_ptr,
                               int in_features,
                               int blocksize) {
    const int packed_bytes_per_row = in_features / 2;
    const int packed_bytes_per_block = blocksize / 2;
    const int blocks_per_row = in_features / blocksize;
    const bool use_sg16 = (packed_bytes_per_row >= 768);

    if (!use_sg16) {
        const int total_out_features = out_features0 + out_features1;
        const size_t wg_size = packed_bytes_per_row >= 1024 ? 256 : 128;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> quant_map_cache(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(total_out_features) * wg_size, wg_size),
                             [=](sycl::nd_item<1> item) {
                const int fused_output_index = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const bool use_first = fused_output_index < out_features0;
                const int output_index = use_first ? fused_output_index : (fused_output_index - out_features0);
                const uint8_t* packed_ptr = use_first ? packed0 : packed1;
                const float* absmax_ptr = use_first ? absmax0 : absmax1;
                bf16* out_ptr = use_first ? out0 : out1;

                if (lid < 16) {
                    quant_map_cache[lid] = (use_first ? quant_map0 : quant_map1)[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                const int row_byte_base = output_index * packed_bytes_per_row;
                const int row_block_base = output_index * blocks_per_row;
                float partial = 0.0f;
                for (int byte_offset = lid; byte_offset < packed_bytes_per_row; byte_offset += static_cast<int>(wg_size)) {
                    const uint8_t packed = packed_ptr[row_byte_base + byte_offset];
                    const float absmax = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const int input_index = byte_offset * 2;
                    partial += static_cast<float>(input_ptr[input_index]) *
                               quant_map_cache[(packed >> 4) & 0x0F] * absmax;
                    partial += static_cast<float>(input_ptr[input_index + 1]) *
                               quant_map_cache[packed & 0x0F] * absmax;
                }

                const float sum = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
                if (lid == 0) {
                    out_ptr[output_index] = bf16(sum);
                }
            });
        });
        return;
    }

    const size_t wg_size = 256;
    const size_t sub_group_size = 16;
    const size_t num_sub_groups = wg_size / sub_group_size;
    const size_t rows_per_group = num_sub_groups;
    const int64_t total_out = static_cast<int64_t>(out_features0) + static_cast<int64_t>(out_features1);
    const size_t num_groups = (static_cast<size_t>(total_out) + rows_per_group - 1) / rows_per_group;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> quant_map_cache(sycl::range<1>(16), cgh);
        sycl::local_accessor<bf16, 1> input_cache(sycl::range<1>(in_features), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) {
            const int lid = static_cast<int>(item.get_local_id(0));
            const int sg_id = lid / sub_group_size;
            const int sg_lane = lid % sub_group_size;
            const int64_t global_row = static_cast<int64_t>(item.get_group(0)) * rows_per_group + sg_id;
            const bool use_first = global_row < static_cast<int64_t>(out_features0);
            const int row = static_cast<int>(use_first ? global_row : (global_row - static_cast<int64_t>(out_features0)));
            const uint8_t* packed_ptr = use_first ? packed0 : packed1;
            const float* absmax_ptr = use_first ? absmax0 : absmax1;
            bf16* out_ptr = use_first ? out0 : out1;
            const int out_features = use_first ? out_features0 : out_features1;

            if (lid < 16) {
                quant_map_cache[lid] = quant_map0[lid];
            }
            for (int i = lid; i < in_features; i += static_cast<int>(wg_size)) {
                input_cache[i] = input_ptr[i];
            }
            item.barrier(sycl::access::fence_space::local_space);

            float partial = 0.0f;
            if (row < out_features) {
                const int row_byte_base = row * packed_bytes_per_row;
                const int row_block_base = row * blocks_per_row;
                int byte_offset = sg_lane * 4;
                for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += static_cast<int>(sub_group_size) * 4) {
                    const uint32_t packed4 = *reinterpret_cast<const uint32_t*>(packed_ptr + row_byte_base + byte_offset);
                    const float absmax0 = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const float absmax1 = absmax_ptr[row_block_base + ((byte_offset + 2) / packed_bytes_per_block)];
                    const int input_base = byte_offset * 2;
                    const uint8_t b0 = static_cast<uint8_t>(packed4);
                    const uint8_t b1 = static_cast<uint8_t>(packed4 >> 8);
                    const uint8_t b2 = static_cast<uint8_t>(packed4 >> 16);
                    const uint8_t b3 = static_cast<uint8_t>(packed4 >> 24);
                    partial += static_cast<float>(input_cache[input_base + 0]) * quant_map_cache[b0 >> 4] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 1]) * quant_map_cache[b0 & 0x0F] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 2]) * quant_map_cache[b1 >> 4] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 3]) * quant_map_cache[b1 & 0x0F] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 4]) * quant_map_cache[b2 >> 4] * absmax1;
                    partial += static_cast<float>(input_cache[input_base + 5]) * quant_map_cache[b2 & 0x0F] * absmax1;
                    partial += static_cast<float>(input_cache[input_base + 6]) * quant_map_cache[b3 >> 4] * absmax1;
                    partial += static_cast<float>(input_cache[input_base + 7]) * quant_map_cache[b3 & 0x0F] * absmax1;
                }
                for (; byte_offset < packed_bytes_per_row; byte_offset += static_cast<int>(sub_group_size)) {
                    const uint8_t packed = packed_ptr[row_byte_base + byte_offset];
                    const float absmax = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const int input_index = byte_offset * 2;
                    partial += static_cast<float>(input_cache[input_index]) *
                               quant_map_cache[(packed >> 4) & 0x0F] * absmax;
                    partial += static_cast<float>(input_cache[input_index + 1]) *
                               quant_map_cache[packed & 0x0F] * absmax;
                }
            }

            float sum = sycl::reduce_over_group(item.get_sub_group(), partial, sycl::plus<>());
            if (sg_lane == 0 && row < out_features) {
                out_ptr[row] = bf16(sum);
            }
        });
    });
}

void packed_nf4_gemv_bf16_3way(Context& ctx,
                               const uint8_t* packed0,
                               const float* quant_map0,
                               const float* absmax0,
                               bf16* out0,
                               int out_features0,
                               const uint8_t* packed1,
                               const float* quant_map1,
                               const float* absmax1,
                               bf16* out1,
                               int out_features1,
                               const uint8_t* packed2,
                               const float* quant_map2,
                               const float* absmax2,
                               bf16* out2,
                               int out_features2,
                               const bf16* input_ptr,
                               int in_features,
                               int blocksize) {
    const int packed_bytes_per_row = in_features / 2;
    const int packed_bytes_per_block = blocksize / 2;
    const int blocks_per_row = in_features / blocksize;
    const bool use_sg16 = (packed_bytes_per_row >= 768);

    if (!use_sg16) {
        const int first_cut = out_features0;
        const int second_cut = out_features0 + out_features1;
        const int total_out_features = out_features0 + out_features1 + out_features2;
        const size_t wg_size = packed_bytes_per_row >= 1024 ? 256 : 128;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> quant_map_cache(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(total_out_features) * wg_size, wg_size),
                             [=](sycl::nd_item<1> item) {
                const int fused_output_index = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const bool use_first = fused_output_index < first_cut;
                const bool use_second = !use_first && fused_output_index < second_cut;
                const int output_index = use_first ? fused_output_index
                    : (use_second ? (fused_output_index - first_cut) : (fused_output_index - second_cut));
                const uint8_t* packed_ptr = use_first ? packed0 : (use_second ? packed1 : packed2);
                const float* absmax_ptr = use_first ? absmax0 : (use_second ? absmax1 : absmax2);
                bf16* out_ptr = use_first ? out0 : (use_second ? out1 : out2);

                if (lid < 16) {
                    quant_map_cache[lid] = (use_first ? quant_map0 : (use_second ? quant_map1 : quant_map2))[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                const int row_byte_base = output_index * packed_bytes_per_row;
                const int row_block_base = output_index * blocks_per_row;
                float partial = 0.0f;
                for (int byte_offset = lid; byte_offset < packed_bytes_per_row; byte_offset += static_cast<int>(wg_size)) {
                    const uint8_t packed = packed_ptr[row_byte_base + byte_offset];
                    const float absmax = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const int input_index = byte_offset * 2;
                    partial += static_cast<float>(input_ptr[input_index]) *
                               quant_map_cache[(packed >> 4) & 0x0F] * absmax;
                    partial += static_cast<float>(input_ptr[input_index + 1]) *
                               quant_map_cache[packed & 0x0F] * absmax;
                }

                const float sum = sycl::reduce_over_group(item.get_group(), partial, sycl::plus<float>());
                if (lid == 0) {
                    out_ptr[output_index] = bf16(sum);
                }
            });
        });
        return;
    }

    const size_t wg_size = 256;
    const size_t sub_group_size = 16;
    const size_t num_sub_groups = wg_size / sub_group_size;
    const size_t rows_per_group = num_sub_groups;
    const int64_t total_out = static_cast<int64_t>(out_features0) + static_cast<int64_t>(out_features1) + static_cast<int64_t>(out_features2);
    const size_t num_groups = (static_cast<size_t>(total_out) + rows_per_group - 1) / rows_per_group;
    const int64_t first_cut = static_cast<int64_t>(out_features0);
    const int64_t second_cut = first_cut + static_cast<int64_t>(out_features1);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> quant_map_cache(sycl::range<1>(16), cgh);
        sycl::local_accessor<bf16, 1> input_cache(sycl::range<1>(in_features), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                         [=](sycl::nd_item<1> item) {
            const int lid = static_cast<int>(item.get_local_id(0));
            const int sg_id = lid / sub_group_size;
            const int sg_lane = lid % sub_group_size;
            const int64_t global_row = static_cast<int64_t>(item.get_group(0)) * rows_per_group + sg_id;
            const bool use_first = global_row < first_cut;
            const bool use_second = !use_first && global_row < second_cut;
            const int row = static_cast<int>(use_first ? global_row
                : (use_second ? (global_row - first_cut) : (global_row - second_cut)));
            const uint8_t* packed_ptr = use_first ? packed0 : (use_second ? packed1 : packed2);
            const float* absmax_ptr = use_first ? absmax0 : (use_second ? absmax1 : absmax2);
            bf16* out_ptr = use_first ? out0 : (use_second ? out1 : out2);
            const int out_features = use_first ? out_features0 : (use_second ? out_features1 : out_features2);

            if (lid < 16) {
                quant_map_cache[lid] = quant_map0[lid];
            }
            for (int i = lid; i < in_features; i += static_cast<int>(wg_size)) {
                input_cache[i] = input_ptr[i];
            }
            item.barrier(sycl::access::fence_space::local_space);

            float partial = 0.0f;
            if (row < out_features) {
                const int row_byte_base = row * packed_bytes_per_row;
                const int row_block_base = row * blocks_per_row;
                int byte_offset = sg_lane * 4;
                for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += static_cast<int>(sub_group_size) * 4) {
                    const uint32_t packed4 = *reinterpret_cast<const uint32_t*>(packed_ptr + row_byte_base + byte_offset);
                    const float absmax0 = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const float absmax1 = absmax_ptr[row_block_base + ((byte_offset + 2) / packed_bytes_per_block)];
                    const int input_base = byte_offset * 2;
                    const uint8_t b0 = static_cast<uint8_t>(packed4);
                    const uint8_t b1 = static_cast<uint8_t>(packed4 >> 8);
                    const uint8_t b2 = static_cast<uint8_t>(packed4 >> 16);
                    const uint8_t b3 = static_cast<uint8_t>(packed4 >> 24);
                    partial += static_cast<float>(input_cache[input_base + 0]) * quant_map_cache[b0 >> 4] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 1]) * quant_map_cache[b0 & 0x0F] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 2]) * quant_map_cache[b1 >> 4] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 3]) * quant_map_cache[b1 & 0x0F] * absmax0;
                    partial += static_cast<float>(input_cache[input_base + 4]) * quant_map_cache[b2 >> 4] * absmax1;
                    partial += static_cast<float>(input_cache[input_base + 5]) * quant_map_cache[b2 & 0x0F] * absmax1;
                    partial += static_cast<float>(input_cache[input_base + 6]) * quant_map_cache[b3 >> 4] * absmax1;
                    partial += static_cast<float>(input_cache[input_base + 7]) * quant_map_cache[b3 & 0x0F] * absmax1;
                }
                for (; byte_offset < packed_bytes_per_row; byte_offset += static_cast<int>(sub_group_size)) {
                    const uint8_t packed = packed_ptr[row_byte_base + byte_offset];
                    const float absmax = absmax_ptr[row_block_base + (byte_offset / packed_bytes_per_block)];
                    const int input_index = byte_offset * 2;
                    partial += static_cast<float>(input_cache[input_index]) *
                               quant_map_cache[(packed >> 4) & 0x0F] * absmax;
                    partial += static_cast<float>(input_cache[input_index + 1]) *
                               quant_map_cache[packed & 0x0F] * absmax;
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
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), src_md, weight_md, dst_md, attr);

    CachedPrimitive cp;
    cp.prim = dnnl::matmul(pd);
    cp.src_md = src_md;
    cp.weight_md = weight_md;
    cp.dst_md = dst_md;
    size_t scratchpad_size = pd.scratchpad_desc().get_size();
    if (scratchpad_size > 0) {
        cp.scratchpad = Tensor::allocate(ctx,
                                         {static_cast<int64_t>(scratchpad_size)},
                                         dnnl::memory::data_type::u8);
        cp.scratchpad_mem = cp.scratchpad.make_dnnl_memory(pd.scratchpad_desc());
    }
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

bool Bnb4BitLinear::try_forward_decode_qkv(Context& ctx,
                                           Bnb4BitLinear& q_proj,
                                           Bnb4BitLinear& k_proj,
                                           Bnb4BitLinear& v_proj,
                                           Tensor& input,
                                           Tensor& q_output,
                                           Tensor& k_output,
                                           Tensor& v_output) {
    const auto bf16_dtype = dnnl::memory::data_type::bf16;
    if (input.dtype() != bf16_dtype || input.ndim() != 2 || input.shape(0) != 1) {
        return false;
    }

    const auto is_eligible = [&](Bnb4BitLinear& proj, Tensor& output) -> bool {
        return can_use_packed_decode_fastpath(proj.weight_, proj.cache_dequantized_weight_, 1) &&
               output.dtype() == bf16_dtype &&
               output.ndim() == 2 &&
               output.shape(0) == 1 &&
               input.shape(1) == proj.in_features_ &&
               output.shape(1) == proj.out_features_;
    };

    if (!is_eligible(q_proj, q_output) ||
        !is_eligible(k_proj, k_output) ||
        !is_eligible(v_proj, v_output)) {
        return false;
    }

    const int blocksize = q_proj.weight_.quant_state.blocksize;
    if (k_proj.weight_.quant_state.blocksize != blocksize ||
        v_proj.weight_.quant_state.blocksize != blocksize) {
        return false;
    }

    packed_nf4_gemv_bf16_3way(ctx,
                              static_cast<const uint8_t*>(q_proj.weight_.packed_weight->data()),
                              static_cast<const float*>(q_proj.weight_.quant_map->data()),
                              static_cast<const float*>(q_proj.absmax_f32_.data()),
                              static_cast<bf16*>(q_output.data()),
                              q_proj.out_features_,
                              static_cast<const uint8_t*>(k_proj.weight_.packed_weight->data()),
                              static_cast<const float*>(k_proj.weight_.quant_map->data()),
                              static_cast<const float*>(k_proj.absmax_f32_.data()),
                              static_cast<bf16*>(k_output.data()),
                              k_proj.out_features_,
                              static_cast<const uint8_t*>(v_proj.weight_.packed_weight->data()),
                              static_cast<const float*>(v_proj.weight_.quant_map->data()),
                              static_cast<const float*>(v_proj.absmax_f32_.data()),
                              static_cast<bf16*>(v_output.data()),
                              v_proj.out_features_,
                              static_cast<const bf16*>(input.data()),
                              q_proj.in_features_,
                              blocksize);
    return true;
}

bool Bnb4BitLinear::try_forward_decode_gate_up(Context& ctx,
                                               Bnb4BitLinear& gate_proj,
                                               Bnb4BitLinear& up_proj,
                                               Tensor& input,
                                               Tensor& gate_output,
                                               Tensor& up_output) {
    const auto bf16_dtype = dnnl::memory::data_type::bf16;
    if (input.dtype() != bf16_dtype || input.ndim() != 2 || input.shape(0) != 1) {
        return false;
    }

    const auto is_eligible = [&](Bnb4BitLinear& proj, Tensor& output) -> bool {
        return can_use_packed_decode_fastpath(proj.weight_, proj.cache_dequantized_weight_, 1) &&
               output.dtype() == bf16_dtype &&
               output.ndim() == 2 &&
               output.shape(0) == 1 &&
               input.shape(1) == proj.in_features_ &&
               output.shape(1) == proj.out_features_;
    };

    if (!is_eligible(gate_proj, gate_output) ||
        !is_eligible(up_proj, up_output)) {
        return false;
    }

    const int blocksize = gate_proj.weight_.quant_state.blocksize;
    if (up_proj.weight_.quant_state.blocksize != blocksize) {
        return false;
    }

    packed_nf4_gemv_bf16_2way(ctx,
                              static_cast<const uint8_t*>(gate_proj.weight_.packed_weight->data()),
                              static_cast<const float*>(gate_proj.weight_.quant_map->data()),
                              static_cast<const float*>(gate_proj.absmax_f32_.data()),
                              static_cast<bf16*>(gate_output.data()),
                              gate_proj.out_features_,
                              static_cast<const uint8_t*>(up_proj.weight_.packed_weight->data()),
                              static_cast<const float*>(up_proj.weight_.quant_map->data()),
                              static_cast<const float*>(up_proj.absmax_f32_.data()),
                              static_cast<bf16*>(up_output.data()),
                              up_proj.out_features_,
                              static_cast<const bf16*>(input.data()),
                              gate_proj.in_features_,
                              blocksize);
    return true;
}

bool Bnb4BitLinear::try_forward_decode_gate_up_swiglu(Context& ctx,
                                                      Bnb4BitLinear& fused_gate_up,
                                                      Tensor& input,
                                                      Tensor& output,
                                                      int ff_dim) {
    if (input.dtype() != dnnl::memory::data_type::bf16 ||
        output.dtype() != dnnl::memory::data_type::bf16) return false;
    if (input.ndim() != 2 || input.shape(0) != 1) return false;

    const auto& w = fused_gate_up.weight_;
    if (!w.packed_weight || !w.packed_weight->valid()) return false;
    if (!w.quant_map || !w.quant_map->valid()) return false;
    if (!fused_gate_up.absmax_f32_.valid()) return false;
    if (fused_gate_up.out_features_ != 2 * ff_dim) return false;

    const int in_features = fused_gate_up.in_features_;
    const int blocksize = w.quant_state.blocksize;
    if (blocksize != 128 || (blocksize % 2) != 0 || (in_features % blocksize) != 0) return false;

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

void Bnb4BitLinear::forward(Context& ctx,
                            Bnb4BitLinearScratch& scratch,
                            Tensor& input,
                            Tensor& output,
                            int seq_len) {
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
        packed_nf4_gemv_bf16(ctx, weight_, absmax_f32_, input, output, in_features_, out_features_);
        return;
    }

    const int blocksize = weight_.quant_state.blocksize;
    static bool use_fused_prefill =
        aila::env::read_int_raw("AILA_BNB4_FUSED_PREFILL", 1) != 0;
    if (use_fused_prefill && seq_len > 1 &&
        blocksize == 128 && (blocksize % 2) == 0 &&
        in_features_ > 0 && (in_features_ % blocksize) == 0 &&
        weight_.packed_weight && weight_.packed_weight->valid() &&
        weight_.quant_map && weight_.quant_map->valid()) {
        packed_nf4_gemm_bf16(ctx,
                             static_cast<const uint8_t*>(weight_.packed_weight->data()),
                             static_cast<const float*>(weight_.quant_map->data()),
                             static_cast<const float*>(absmax_f32_.data()),
                             static_cast<const bf16*>(input.data()),
                             static_cast<bf16*>(output.data()),
                             seq_len, out_features_, in_features_,
                             blocksize);
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

    ctx.queue().memcpy(output.data(), output_bf16.data(),
                       static_cast<size_t>(seq_len) * static_cast<size_t>(out_features_) * sizeof(bf16));
}
