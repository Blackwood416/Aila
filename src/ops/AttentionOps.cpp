#include "Ops.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sycl/sycl.hpp>


using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

namespace ops {

namespace {

inline int round_up(int x, int align) {
  return ((x + align - 1) / align) * align;
}

int read_env_int_local(const char *name, int default_value) {
  return aila::env::read_int_raw(name, default_value);
}

bool supports_jm_bf16_f32(const sycl::device &dev, int m, int n, int k) {
  if (!dev.has(sycl::aspect::ext_intel_matrix)) {
    return false;
  }

  using matrix_type = sycl::ext::oneapi::experimental::matrix::matrix_type;
  using sycl::ext::oneapi::experimental::info::device::matrix_combinations;
  auto combos = dev.get_info<matrix_combinations>();
  for (const auto &c : combos) {
    bool m_ok =
        (static_cast<int>(c.msize) == 0) || (static_cast<int>(c.msize) == m);
    if (m_ok && static_cast<int>(c.nsize) == n &&
        static_cast<int>(c.ksize) == k && c.atype == matrix_type::bf16 &&
        c.btype == matrix_type::bf16 && c.ctype == matrix_type::fp32 &&
        c.dtype == matrix_type::fp32) {
      return true;
    }
  }
  return false;
}

constexpr int kDecodeExact256Tile = 128;
constexpr int kDecodeExact256TileWg = 128;
constexpr int kDecodeExact256MergeWg = 128;
constexpr int kDecodeExact256MinLen = 1024;
constexpr int kDecodeExact256PartialStride = 272;
constexpr int kDecodeExact256PartialAccOffset = 16;
constexpr int kPrefillExact256Tile = 128;
constexpr int kPrefillExact256Wg = 128;
constexpr int kPrefillExact256MinSeq = 1024;

void attention_decode_baseline(Context &ctx, Tensor &q, Tensor &k_cache,
                               Tensor &v_cache, Tensor &output,
                               Tensor &scores_buf, int num_heads,
                               int num_kv_heads, int head_dim, int cached_len,
                               int tail_start, int sink_len, int effective_len,
                               int wg_size) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k_cache.data());
  bf16 *v_ptr = static_cast<bf16 *>(v_cache.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());
  (void)scores_buf;

  int heads_per_kv = num_heads / num_kv_heads;
  float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  int max_seq_len = static_cast<int>(k_cache.shape(1));
  int padded_cached_len = round_up(effective_len, 32);

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> shared(sycl::range<1>(padded_cached_len),
                                          cgh);
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
          int head = item.get_group(0);
          int lid = item.get_local_id(0);
          int kv_head = head / heads_per_kv;

          for (int d = lid; d < head_dim; d += wg_size) {
            q_cache[d] = static_cast<float>(q_ptr[head * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int t = lid; t < effective_len; t += wg_size) {
            int cache_t = (t < sink_len) ? t : (tail_start + (t - sink_len));
            float sum = 0.0f;
            const bf16 *k_row =
                k_ptr + kv_head * max_seq_len * head_dim + cache_t * head_dim;
            if ((head_dim & 7) == 0) {
              for (int d = 0; d < head_dim; d += 8) {
                sum += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                sum += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                sum += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                sum += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                sum += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                sum += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                sum += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                sum += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
            } else {
              for (int d = 0; d < head_dim; d++) {
                sum += q_cache[d] * static_cast<float>(k_row[d]);
              }
            }
            shared[t] = sum * scale;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float max_val = -1e30f;
          for (int t = lid; t < effective_len; t += wg_size) {
            max_val = sycl::fmax(max_val, shared[t]);
          }
          max_val = sycl::reduce_over_group(item.get_group(), max_val,
                                            sycl::maximum<float>());

          float sum_val = 0.0f;
          for (int t = lid; t < effective_len; t += wg_size) {
            float e = sycl::native::exp(shared[t] - max_val);
            shared[t] = e;
            sum_val += e;
          }
          sum_val = sycl::reduce_over_group(item.get_group(), sum_val,
                                            sycl::plus<float>());

          for (int t = lid; t < effective_len; t += wg_size) {
            float prob = shared[t] / sum_val;
            shared[t] = prob;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int d = lid; d < head_dim; d += wg_size) {
            float acc = 0.0f;
            for (int t = 0; t < effective_len; t++) {
              int cache_t = (t < sink_len) ? t : (tail_start + (t - sink_len));
              acc += shared[t] *
                     static_cast<float>(v_ptr[kv_head * max_seq_len * head_dim +
                                              cache_t * head_dim + d]);
            }
            o_ptr[head * head_dim + d] = bf16(acc);
          }
        });
  });
}

void attention_decode_exact_head256_partial_merge(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k_cache.data());
  bf16 *v_ptr = static_cast<bf16 *>(v_cache.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());
  float *partials_ptr = static_cast<float *>(partials_buf.data());

  constexpr int head_dim = 256;
  constexpr int tile_t = kDecodeExact256Tile;
  constexpr int tile_wg = kDecodeExact256TileWg;
  constexpr int merge_wg = kDecodeExact256MergeWg;

  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int max_seq_len = static_cast<int>(k_cache.shape(1));
  const int num_tiles = (effective_len + tile_t - 1) / tile_t;
  const int max_tiles = static_cast<int>(partials_buf.shape(1));
  (void)max_tiles;

  // Phase 1: compute per-tile exact softmax stats and exp-weighted V partials.
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);
    sycl::local_accessor<float, 1> logits(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> stats(sycl::range<1>(2), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * num_tiles * tile_wg, tile_wg),
        [=](sycl::nd_item<1> item) {
          const int group = item.get_group(0);
          const int head = group / num_tiles;
          const int tile_idx = group % num_tiles;
          const int lid = item.get_local_id(0);
          const int kv_head = head / heads_per_kv;
          const int tile_start = tile_idx * tile_t;
          const int tile_len = sycl::min(tile_t, effective_len - tile_start);
          float *partial =
              partials_ptr +
              (head * max_tiles + tile_idx) * kDecodeExact256PartialStride;

          for (int d = lid; d < head_dim; d += tile_wg) {
            q_cache[d] = static_cast<float>(q_ptr[head * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float score = -1e30f;
          if (lid < tile_len) {
            const int t = tile_start + lid;
            const int cache_t =
                (t < sink_len) ? t : (tail_start + (t - sink_len));
            const bf16 *k_row =
                k_ptr + kv_head * max_seq_len * head_dim + cache_t * head_dim;
            score = 0.0f;
            for (int d = 0; d < head_dim; d += 8) {
              score += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
              score += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
              score += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
              score += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
              score += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
              score += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
              score += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
              score += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
            }
            score *= scale;
          }
          logits[lid] = score;
          const float tile_max = sycl::reduce_over_group(
              item.get_group(), score, sycl::maximum<float>());
          if (lid == 0) {
            stats[0] = tile_max;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float exp_score = 0.0f;
          if (lid < tile_len) {
            exp_score = sycl::native::exp(logits[lid] - stats[0]);
            logits[lid] = exp_score;
          } else {
            logits[lid] = 0.0f;
          }
          const float tile_sum = sycl::reduce_over_group(
              item.get_group(), exp_score, sycl::plus<float>());
          if (lid == 0) {
            stats[1] = tile_sum;
            partial[0] = stats[0];
            partial[1] = stats[1];
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int d = lid; d < head_dim; d += tile_wg) {
            float acc = 0.0f;
            for (int i = 0; i < tile_len; ++i) {
              const int t = tile_start + i;
              const int cache_t =
                  (t < sink_len) ? t : (tail_start + (t - sink_len));
              acc += logits[i] *
                     static_cast<float>(
                         v_ptr[kv_head * max_seq_len * head_dim +
                               cache_t * head_dim + d]);
            }
            partial[kDecodeExact256PartialAccOffset + d] = acc;
          }
        });
  });

  // Phase 2: merge per-tile statistics and partial outputs exactly.
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> tile_factors(sycl::range<1>(num_tiles), cgh);
    sycl::local_accessor<float, 1> norm_sum(sycl::range<1>(1), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * merge_wg, merge_wg),
        [=](sycl::nd_item<1> item) {
          const int head = item.get_group(0);
          const int lid = item.get_local_id(0);
          if (lid == 0) {
            float global_max = -1e30f;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
              const float *partial =
                  partials_ptr +
                  (head * max_tiles + tile_idx) * kDecodeExact256PartialStride;
              global_max = sycl::fmax(global_max, partial[0]);
            }
            float sum = 0.0f;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
              const float *partial =
                  partials_ptr +
                  (head * max_tiles + tile_idx) * kDecodeExact256PartialStride;
              const float factor = sycl::native::exp(partial[0] - global_max);
              tile_factors[tile_idx] = factor;
              sum += factor * partial[1];
            }
            norm_sum[0] = sum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          const float inv_sum = 1.0f / norm_sum[0];
          for (int d = lid; d < head_dim; d += merge_wg) {
            float acc = 0.0f;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
              const float *partial =
                  partials_ptr +
                  (head * max_tiles + tile_idx) * kDecodeExact256PartialStride;
              acc += tile_factors[tile_idx] *
                     partial[kDecodeExact256PartialAccOffset + d];
            }
            o_ptr[head * head_dim + d] = bf16(acc * inv_sum);
          }
        });
  });
}

template <int TM, int TN, int TK, int SG>
void attention_decode_joint_matrix_tiled(Context &ctx, Tensor &q,
                                         Tensor &k_cache, Tensor &v_cache,
                                         Tensor &output, Tensor &scores_buf,
                                         int num_heads, int num_kv_heads,
                                         int head_dim, int cached_len,
                                         int attn_start, int wg_size) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k_cache.data());
  bf16 *v_ptr = static_cast<bf16 *>(v_cache.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());
  float *scores_ptr = static_cast<float *>(scores_buf.data());

  int heads_per_kv = num_heads / num_kv_heads;
  float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  int max_seq_len = static_cast<int>(k_cache.shape(1));
  int effective_len = cached_len - attn_start;
  int num_tiles = (effective_len + TN - 1) / TN;
  int padded_cached_len = round_up(effective_len, 32);

  // Phase 1: QK^T via joint_matrix (writes unnormalized logits)
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<bf16, 1> a_local(sycl::range<1>(TM * TK), cgh);
    sycl::local_accessor<float, 1> c_local(sycl::range<1>(TM * TN), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * num_tiles * SG, SG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
          auto sg = item.get_sub_group();
          int group = item.get_group(0);
          int head = group / num_tiles;
          int tile = group % num_tiles;
          int t0 = tile * TN;
          int lid = item.get_local_id(0);
          int kv_head = head / heads_per_kv;

          joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major>
              sub_a;
          joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major>
              sub_b;
          joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
          joint_matrix_fill(sg, sub_c, 0.0f);

          const bf16 *q_head = q_ptr + head * head_dim;
          const bf16 *k_head_tile =
              k_ptr + kv_head * max_seq_len * head_dim +
              (attn_start + t0) * head_dim;

          for (int kb = 0; kb < head_dim; kb += TK) {
            for (int i = lid; i < TM * TK; i += SG) {
              int c = i % TK;
              a_local[i] = q_head[kb + c];
            }
            item.barrier(sycl::access::fence_space::local_space);

            auto a_ptr = a_local.get_multi_ptr<sycl::access::decorated::no>();
            auto b_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(k_head_tile + kb);
            joint_matrix_load(sg, sub_a, a_ptr, TK);
            joint_matrix_load(sg, sub_b, b_ptr, head_dim);
            joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
          }

          auto c_ptr = c_local.get_multi_ptr<sycl::access::decorated::no>();
          joint_matrix_store(sg, sub_c, c_ptr, TN, layout::row_major);
          item.barrier(sycl::access::fence_space::local_space);

          if (lid < TN) {
            int t = t0 + lid;
            if (t < effective_len) {
              int cache_t = attn_start + t;
              scores_ptr[head * max_seq_len + cache_t] = c_local[lid] * scale;
            }
          }
        });
  });

  // Phase 2: softmax + V accumulation
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> shared(sycl::range<1>(padded_cached_len),
                                          cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
          int head = item.get_group(0);
          int lid = item.get_local_id(0);
          int kv_head = head / heads_per_kv;
          float *row = scores_ptr + head * max_seq_len;

          for (int t = lid; t < effective_len; t += wg_size) {
            int cache_t = attn_start + t;
            shared[t] = row[cache_t];
          }
          item.barrier(sycl::access::fence_space::local_space);

          float max_val = -1e30f;
          for (int t = lid; t < effective_len; t += wg_size) {
            max_val = sycl::fmax(max_val, shared[t]);
          }
          max_val = sycl::reduce_over_group(item.get_group(), max_val,
                                            sycl::maximum<float>());

          float sum_val = 0.0f;
          for (int t = lid; t < effective_len; t += wg_size) {
            float e = sycl::native::exp(shared[t] - max_val);
            shared[t] = e;
            sum_val += e;
          }
          sum_val = sycl::reduce_over_group(item.get_group(), sum_val,
                                            sycl::plus<float>());

          for (int t = lid; t < effective_len; t += wg_size) {
            float p = shared[t] / sum_val;
            shared[t] = p;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int d = lid; d < head_dim; d += wg_size) {
            float acc = 0.0f;
            for (int t = 0; t < effective_len; t++) {
              int cache_t = attn_start + t;
              acc += shared[t] *
                     static_cast<float>(v_ptr[kv_head * max_seq_len * head_dim +
                                              cache_t * head_dim + d]);
            }
            o_ptr[head * head_dim + d] = bf16(acc);
          }
        });
  });
}

void attention_decode_joint_matrix(Context &ctx, Tensor &q, Tensor &k_cache,
                                   Tensor &v_cache, Tensor &output,
                                   Tensor &scores_buf, int num_heads,
                                   int num_kv_heads, int head_dim,
                                   int cached_len, int attn_start, int tile_id,
                                   int wg_size) {
  if (tile_id == 2) {
    attention_decode_joint_matrix_tiled<8, 8, 16, 8>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, attn_start, wg_size);
  } else if (tile_id == 1) {
    attention_decode_joint_matrix_tiled<32, 32, 16, 16>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, attn_start, wg_size);
  } else {
    attention_decode_joint_matrix_tiled<1, 8, 16, 8>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, attn_start, wg_size);
  }
}

} // namespace

// ============================================================
// SYCL Kernel: Attention Decode (seq_len=1, GQA)
// ============================================================

void attention_decode(Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache,
                      Tensor &output, Tensor &scores_buf, int num_heads,
                      int num_kv_heads, int head_dim, int cached_len,
                      Tensor *exact_partials_buf) {
  constexpr int JM0_M = 1;
  constexpr int JM0_N = 8;
  constexpr int JM0_K = 16;
  constexpr int JM1_M = 32;
  constexpr int JM1_N = 32;
  constexpr int JM1_K = 16;
  constexpr int JM2_M = 8;
  constexpr int JM2_N = 8;
  constexpr int JM2_K = 16;

  // AILA_ATTN_JM:
  //   0 (default): disabled
  //   1: auto (run only when runtime-reported combination is supported)
  //   2: force (skip capability gate, for debugging only)
  static int jm_mode = -1;
  static int jm_supported = -1;
  static int jm_tile_id = -1;
  static int decode_wg = -1;
  static int decode_window = -1;
  static int decode_window_start = -1;
  static int decode_sink = -1;
  static bool jm_log_once = false;
  if (jm_mode < 0) {
    jm_mode = read_env_int_local("AILA_ATTN_JM", 1);
  }
  if (decode_wg < 0) {
    decode_wg = read_env_int_local("AILA_ATTN_DECODE_WG", 512);
    if (decode_wg <= 0)
      decode_wg = 512;
  }
  if (decode_window < 0) {
    decode_window = read_env_int_local("AILA_ATTN_DECODE_WINDOW", 0);
    if (decode_window < 0)
      decode_window = 0;
  }
  if (decode_window_start < 0) {
    decode_window_start =
        read_env_int_local("AILA_ATTN_DECODE_WINDOW_START", -1);
    if (decode_window_start < 0) {
      decode_window_start = std::max(512, decode_window);
    }
  }
  if (decode_sink < 0) {
    decode_sink = read_env_int_local("AILA_ATTN_DECODE_SINK", -1);
    if (decode_sink < 0) {
      decode_sink = 0;
    }
    if (decode_sink < 0) {
      decode_sink = 0;
    }
  }
  if (jm_supported < 0) {
    bool s0 =
        supports_jm_bf16_f32(ctx.queue().get_device(), JM0_M, JM0_N, JM0_K);
    bool s1 =
        supports_jm_bf16_f32(ctx.queue().get_device(), JM1_M, JM1_N, JM1_K);
    bool s2 =
        supports_jm_bf16_f32(ctx.queue().get_device(), JM2_M, JM2_N, JM2_K);
    jm_supported = (s0 || s1 || s2) ? 1 : 0;

    int force_tile = read_env_int_local("AILA_ATTN_JM_TILE", -1);
    if (force_tile == 2 && s2) {
      jm_tile_id = 2;
    } else if (force_tile == 1 && s1) {
      jm_tile_id = 1;
    } else if (force_tile == 0 && s0) {
      jm_tile_id = 0;
    } else if (s0) {
      // Default to the conservative tile first; larger tile can be forced via
      // env for bring-up.
      jm_tile_id = 0;
    } else if (s2) {
      jm_tile_id = 2;
    } else if (s1) {
      jm_tile_id = 1;
    } else {
      jm_tile_id = -1;
    }
  }

  bool allow_jm =
      (jm_mode == 2) || (jm_mode == 1 && jm_supported == 1 && jm_tile_id >= 0);
  bool apply_window = (decode_window > 0 && cached_len > decode_window_start);
  int effective_window = cached_len;
  if (apply_window) {
    effective_window = std::min(cached_len, decode_window);
  }
  int tail_start = cached_len - effective_window;
  int sink_len = 0;
  if (apply_window && decode_sink > 0) {
    sink_len = std::min(decode_sink, std::max(0, cached_len - effective_window));
  }
  int effective_len = effective_window + sink_len;

  // If sink and tail overlap, fall back to full context contiguous view.
  if (effective_len >= cached_len) {
    tail_start = 0;
    sink_len = 0;
    effective_len = cached_len;
  }

  bool contiguous_mode = (sink_len == 0);
  int attn_start = tail_start;

  if (!jm_log_once && jm_mode > 0) {
    AILA_LOG_INFO("[JM] mode=%d, supported=%d, tile_id=%d, decode_wg=%d, decode_window=%d, decode_window_start=%d, decode_sink=%d %s",
                  jm_mode, jm_supported, jm_tile_id, decode_wg, decode_window,
                  decode_window_start, decode_sink,
                  (allow_jm ? "-> enabled" : "-> fallback baseline"));
    jm_log_once = true;
  }
  if (allow_jm && contiguous_mode && head_dim == 128 && cached_len > 0) {
    attention_decode_joint_matrix(ctx, q, k_cache, v_cache, output, scores_buf,
                                  num_heads, num_kv_heads, head_dim, cached_len,
                                  attn_start, jm_tile_id, decode_wg);
    return;
  }

  if (head_dim == 256 && cached_len > 0 && effective_len >= kDecodeExact256MinLen &&
      exact_partials_buf && exact_partials_buf->valid()) {
    attention_decode_exact_head256_partial_merge(
        ctx, q, k_cache, v_cache, output, *exact_partials_buf, num_heads,
        num_kv_heads, tail_start, sink_len, effective_len);
    return;
  }

  attention_decode_baseline(ctx, q, k_cache, v_cache, output, scores_buf,
                            num_heads, num_kv_heads, head_dim, cached_len,
                            tail_start, sink_len, effective_len,
                            decode_wg);
}

// ============================================================
// SYCL Kernel: Attention Prefill (seq_len > 1)
// ============================================================

void attention_prefill(Context &ctx, Tensor &q, Tensor &k, Tensor &v,
                       Tensor &output, Tensor &scores_buf, int seq_len,
                       int num_heads, int num_kv_heads, int head_dim) {
  if (head_dim == 256 && seq_len >= kPrefillExact256MinSeq) {
    bf16 *q_ptr = static_cast<bf16 *>(q.data());
    bf16 *k_ptr = static_cast<bf16 *>(k.data());
    bf16 *v_ptr = static_cast<bf16 *>(v.data());
    bf16 *o_ptr = static_cast<bf16 *>(output.data());

    constexpr int tile_t = kPrefillExact256Tile;
    constexpr int wg_size = kPrefillExact256Wg;
    constexpr int exact_head_dim = 256;
    const int heads_per_kv = num_heads / num_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(exact_head_dim));
    const int q_total = num_heads * exact_head_dim;
    const int k_total = num_kv_heads * exact_head_dim;

    ctx.queue().submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> q_cache(sycl::range<1>(exact_head_dim), cgh);
      sycl::local_accessor<float, 1> scores(sycl::range<1>(tile_t), cgh);
      sycl::local_accessor<float, 1> acc_local(sycl::range<1>(exact_head_dim), cgh);
      sycl::local_accessor<float, 1> merge_state(sycl::range<1>(4), cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(num_heads * seq_len * wg_size, wg_size),
          [=](sycl::nd_item<1> item) {
            const int group = item.get_group(0);
            const int h = group / seq_len;
            const int qi = group % seq_len;
            const int kv_h = h / heads_per_kv;
            const int lid = item.get_local_id(0);

            const bf16 *q_row = q_ptr + qi * q_total + h * exact_head_dim;
            for (int d = lid; d < exact_head_dim; d += wg_size) {
              q_cache[d] = static_cast<float>(q_row[d]);
              acc_local[d] = 0.0f;
            }
            if (lid == 0) {
              merge_state[0] = -1e30f;
              merge_state[1] = 0.0f;
              merge_state[2] = 0.0f;
              merge_state[3] = 0.0f;
            }
            item.barrier(sycl::access::fence_space::local_space);

            for (int tile_start = 0; tile_start <= qi; tile_start += tile_t) {
              const int tile_len = sycl::min(tile_t, qi + 1 - tile_start);

              float score = -1e30f;
              if (lid < tile_len) {
                const bf16 *k_row =
                    k_ptr + (tile_start + lid) * k_total + kv_h * exact_head_dim;
                score = 0.0f;
                for (int d = 0; d < exact_head_dim; d += 8) {
                  score += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                  score += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                  score += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                  score += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                  score += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                  score += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                  score += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                  score += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
                }
                score *= scale;
                scores[lid] = score;
              } else if (lid < tile_t) {
                scores[lid] = -1e30f;
              }

              const float tile_max = sycl::reduce_over_group(
                  item.get_group(), score, sycl::maximum<float>());

              float exp_score = 0.0f;
              if (lid < tile_len) {
                exp_score = sycl::native::exp(scores[lid] - tile_max);
                scores[lid] = exp_score;
              } else if (lid < tile_t) {
                scores[lid] = 0.0f;
              }

              const float tile_sum = sycl::reduce_over_group(
                  item.get_group(), exp_score, sycl::plus<float>());

              if (lid == 0) {
                const float prev_m = merge_state[0];
                const float prev_l = merge_state[1];
                const float new_m = sycl::fmax(prev_m, tile_max);
                const float alpha = sycl::native::exp(prev_m - new_m);
                const float beta = sycl::native::exp(tile_max - new_m);
                merge_state[0] = new_m;
                merge_state[1] = alpha * prev_l + beta * tile_sum;
                merge_state[2] = alpha;
                merge_state[3] = beta;
              }
              item.barrier(sycl::access::fence_space::local_space);

              const float alpha = merge_state[2];
              const float beta = merge_state[3];
              for (int d = lid; d < exact_head_dim; d += wg_size) {
                float tile_acc = 0.0f;
                for (int j = 0; j < tile_len; ++j) {
                  const bf16 *v_row =
                      v_ptr + (tile_start + j) * k_total + kv_h * exact_head_dim;
                  tile_acc += scores[j] * static_cast<float>(v_row[d]);
                }
                acc_local[d] = alpha * acc_local[d] + beta * tile_acc;
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            const float inv_l = 1.0f / merge_state[1];
            bf16 *o_row = o_ptr + qi * q_total + h * exact_head_dim;
            for (int d = lid; d < exact_head_dim; d += wg_size) {
              o_row[d] = bf16(acc_local[d] * inv_l);
            }
          });
    });
    return;
  }

  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  int heads_per_kv = num_heads / num_kv_heads;
  float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  int q_total = num_heads * head_dim;
  int k_total = num_kv_heads * head_dim;

  float *scores_device = static_cast<float *>(scores_buf.data());

  ctx.queue().parallel_for(
      sycl::range<3>(num_heads, seq_len, seq_len), [=](sycl::id<3> idx) {
        int h = idx[0];
        int qi = idx[1];
        int ki = idx[2];
        int kv_h = h / heads_per_kv;

        if (ki > qi) {
          scores_device[h * seq_len * seq_len + qi * seq_len + ki] = -1e30f;
          return;
        }

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          float qv = static_cast<float>(q_ptr[qi * q_total + h * head_dim + d]);
          float kv =
              static_cast<float>(k_ptr[ki * k_total + kv_h * head_dim + d]);
          dot += qv * kv;
        }
        scores_device[h * seq_len * seq_len + qi * seq_len + ki] = dot * scale;
      });

  ctx.queue().parallel_for(
      sycl::range<2>(num_heads, seq_len), [=](sycl::id<2> idx) {
        int h = idx[0];
        int qi = idx[1];
        float *row = scores_device + h * seq_len * seq_len + qi * seq_len;

        float max_val = -1e30f;
        for (int ki = 0; ki <= qi; ki++) {
          max_val = sycl::fmax(max_val, row[ki]);
        }
        float sum = 0.0f;
        for (int ki = 0; ki <= qi; ki++) {
          row[ki] = sycl::exp(row[ki] - max_val);
          sum += row[ki];
        }
        for (int ki = 0; ki <= qi; ki++) {
          row[ki] /= sum;
        }
        for (int ki = qi + 1; ki < seq_len; ki++) {
          row[ki] = 0.0f;
        }
      });

  ctx.queue().parallel_for(
      sycl::range<3>(num_heads, seq_len, head_dim), [=](sycl::id<3> idx) {
        int h = idx[0];
        int qi = idx[1];
        int d = idx[2];
        int kv_h = h / heads_per_kv;

        float acc = 0.0f;
        float *row = scores_device + h * seq_len * seq_len + qi * seq_len;
        for (int t = 0; t <= qi; t++) {
          float vv =
              static_cast<float>(v_ptr[t * k_total + kv_h * head_dim + d]);
          acc += row[t] * vv;
        }
        o_ptr[qi * q_total + h * head_dim + d] = bf16(acc);
      });
}

// ============================================================
// SYCL Kernel: Incremental Prefill Attention
// ============================================================

void attention_prefill_cached(Context& ctx,
                              Tensor& q, Tensor& k_cache, Tensor& v_cache,
                              Tensor& output, Tensor& scores_buf,
                              int seq_len, int start_pos,
                              int num_heads, int num_kv_heads,
                              int head_dim, int max_seq_len) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k_cache.data());
  bf16 *v_ptr = static_cast<bf16 *>(v_cache.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  int heads_per_kv = num_heads / num_kv_heads;
  float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  int q_total = num_heads * head_dim;

  float *scores_device = static_cast<float *>(scores_buf.data());
  int total_len = start_pos + seq_len;

  ctx.queue().parallel_for(
      sycl::range<3>(num_heads, seq_len, total_len), [=](sycl::id<3> idx) {
        int h = idx[0];
        int qi = idx[1];
        int ki = idx[2];
        int kv_h = h / heads_per_kv;

        if (ki > start_pos + qi) {
          scores_device[h * seq_len * total_len + qi * total_len + ki] = -1e30f;
          return;
        }

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          float qv = static_cast<float>(q_ptr[qi * q_total + h * head_dim + d]);
          float kv = static_cast<float>(
              k_ptr[kv_h * max_seq_len * head_dim + ki * head_dim + d]);
          dot += qv * kv;
        }
        scores_device[h * seq_len * total_len + qi * total_len + ki] = dot * scale;
      });

  ctx.queue().parallel_for(
      sycl::range<2>(num_heads, seq_len), [=](sycl::id<2> idx) {
        int h = idx[0];
        int qi = idx[1];
        float *row = scores_device + h * seq_len * total_len + qi * total_len;

        float max_val = -1e30f;
        for (int ki = 0; ki <= start_pos + qi; ki++) {
          max_val = sycl::fmax(max_val, row[ki]);
        }
        float sum = 0.0f;
        for (int ki = 0; ki <= start_pos + qi; ki++) {
          row[ki] = sycl::exp(row[ki] - max_val);
          sum += row[ki];
        }
        for (int ki = 0; ki <= start_pos + qi; ki++) {
          row[ki] /= sum;
        }
        for (int ki = start_pos + qi + 1; ki < total_len; ki++) {
          row[ki] = 0.0f;
        }
      });

  ctx.queue().parallel_for(
      sycl::range<3>(num_heads, seq_len, head_dim), [=](sycl::id<3> idx) {
        int h = idx[0];
        int qi = idx[1];
        int d = idx[2];
        int kv_h = h / heads_per_kv;

        float acc = 0.0f;
        float *row = scores_device + h * seq_len * total_len + qi * total_len;
        for (int t = 0; t <= start_pos + qi; t++) {
          float vv = static_cast<float>(
              v_ptr[kv_h * max_seq_len * head_dim + t * head_dim + d]);
          acc += row[t] * vv;
        }
        o_ptr[qi * q_total + h * head_dim + d] = bf16(acc);
      });
}

// ============================================================
// SYCL Kernel: Copy K/V to Cache
// ============================================================

void copy_to_cache(Context &ctx, Tensor &new_kv, Tensor &cache, int seq_len,
                   int start_pos, int num_heads, int head_dim,
                   int max_seq_len) {
  bf16 *src = static_cast<bf16 *>(new_kv.data());
  bf16 *dst = static_cast<bf16 *>(cache.data());
  int total_dim = num_heads * head_dim;

  ctx.queue().parallel_for(sycl::range<2>(seq_len, total_dim),
                           [=](sycl::id<2> idx) {
                             int s = idx[0];
                             int flat = idx[1];
                             int h = flat / head_dim;
                             int d = flat % head_dim;

                             int src_idx = s * total_dim + flat;
                             int dst_idx = h * max_seq_len * head_dim +
                                           (start_pos + s) * head_dim + d;
                             dst[dst_idx] = src[src_idx];
                           });
}

} // namespace ops
