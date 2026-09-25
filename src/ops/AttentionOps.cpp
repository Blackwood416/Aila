#include "Ops.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <sycl/sycl.hpp>


using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

namespace ops {

namespace {

template <bool Quantized>
struct KVCacheAccessor {
  const void *ptr;
  int max_seq_len;
  int head_dim;

  KVCacheAccessor(const void *p, int msl, int hd)
      : ptr(p), max_seq_len(msl), head_dim(hd) {}

  inline float load(int kv_head, int t, int d) const {
    int idx = kv_head * max_seq_len * head_dim + t * head_dim + d;
    if constexpr (Quantized) {
      return fp8_e4m3fn_to_float(static_cast<const uint8_t *>(ptr)[idx]);
    } else {
      return static_cast<float>(static_cast<const bf16 *>(ptr)[idx]);
    }
  }
};

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
constexpr int kDecodeExact128MinLen = 128;
constexpr int kDecodeExact128PartialStride = 144;
constexpr int kDecodeExact128PartialAccOffset = 16;
constexpr int kPrefillExact256Tile = 128;
constexpr int kPrefillExact256Wg = 128;
constexpr int kPrefillExact256MinSeq = 1024;
constexpr int kPrefillCachedExact256MinTotal = 1024;
constexpr int kPrefillExact128MinSeq = 256;
constexpr int kPrefillCachedExact128MinTotal = 256;
constexpr int kVisionBidiExact64HeadDim = 64;
constexpr int kVisionBidiExact64Tile = 128;
constexpr int kVisionBidiExact64TileAlt = 256;
constexpr int kVisionBidiExact64TileAltMid = 192;
constexpr int kVisionBidiExact64Wg = 64;
constexpr int kVisionBidiExact64MinSeq = 256;
constexpr int kVisionBidiExact64TileDefault = kVisionBidiExact64TileAltMid;
constexpr int kVisionBidiExact64SgAlt = 16;
constexpr int kVisionBidiExact64SgDefault = kVisionBidiExact64SgAlt;
constexpr int kVisionBidiExact64VecDisabled = 0;
constexpr int kVisionBidiExact64VecEnabled = 1;
constexpr int kVisionBidiExact64VecDefault = kVisionBidiExact64VecEnabled;
constexpr int kVisionBidiExact64VUnrollDisabled = 0;
constexpr int kVisionBidiExact64VUnrollEnabled = 1;
constexpr int kVisionBidiExact64VUnrollDefault =
    kVisionBidiExact64VUnrollEnabled;

template <bool Quantized>
void attention_decode_baseline_impl(Context &ctx, Tensor &q, Tensor &k_cache,
                                    Tensor &v_cache, Tensor &output,
                                    Tensor &scores_buf, int num_heads,
                                    int num_kv_heads, int head_dim, int cached_len,
                                    int tail_start, int sink_len, int effective_len,
                                    int wg_size,
                                    const bf16 *gate_ptr = nullptr) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  const void *k_ptr = k_cache.data();
  const void *v_ptr = v_cache.data();
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

          KVCacheAccessor<Quantized> k_acc(k_ptr, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptr, max_seq_len, head_dim);

          for (int d = lid; d < head_dim; d += wg_size) {
            q_cache[d] = static_cast<float>(q_ptr[head * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int t = lid; t < effective_len; t += wg_size) {
            int cache_t = (t < sink_len) ? t : (tail_start + (t - sink_len));
            float sum = 0.0f;
            if ((head_dim & 7) == 0) {
              for (int d = 0; d < head_dim; d += 8) {
                sum += q_cache[d + 0] * k_acc.load(kv_head, cache_t, d + 0);
                sum += q_cache[d + 1] * k_acc.load(kv_head, cache_t, d + 1);
                sum += q_cache[d + 2] * k_acc.load(kv_head, cache_t, d + 2);
                sum += q_cache[d + 3] * k_acc.load(kv_head, cache_t, d + 3);
                sum += q_cache[d + 4] * k_acc.load(kv_head, cache_t, d + 4);
                sum += q_cache[d + 5] * k_acc.load(kv_head, cache_t, d + 5);
                sum += q_cache[d + 6] * k_acc.load(kv_head, cache_t, d + 6);
                sum += q_cache[d + 7] * k_acc.load(kv_head, cache_t, d + 7);
              }
            } else {
              for (int d = 0; d < head_dim; d++) {
                sum += q_cache[d] * k_acc.load(kv_head, cache_t, d);
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
              acc += shared[t] * v_acc.load(kv_head, cache_t, d);
            }
            float val = static_cast<float>(bf16(acc));
            if (gate_ptr != nullptr) {
              float g_v = static_cast<float>(gate_ptr[head * head_dim + d]);
              float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
              val = static_cast<float>(bf16(val * s));
            }
            o_ptr[head * head_dim + d] = bf16(val);
          }
        });
  });
}

void attention_decode_baseline(Context &ctx, Tensor &q, Tensor &k_cache,
                               Tensor &v_cache, Tensor &output,
                               Tensor &scores_buf, int num_heads,
                               int num_kv_heads, int head_dim, int cached_len,
                               int tail_start, int sink_len, int effective_len,
                               int wg_size,
                               const Tensor *gate_buf = nullptr) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
    attention_decode_baseline_impl<true>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, tail_start, sink_len, effective_len, wg_size, gate_ptr);
  } else {
    attention_decode_baseline_impl<false>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, tail_start, sink_len, effective_len, wg_size, gate_ptr);
  }
}

template <bool Quantized>
void attention_decode_exact_head256_partial_merge_impl(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len,
    const bf16 *gate_ptr = nullptr) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  const void *k_ptr = k_cache.data();
  const void *v_ptr = v_cache.data();
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
  // The partials region may be a per-sequence 4-D block (batched decode); a
  // single-row call always uses the first sequence's block and its tile count
  // is the second-to-last extent.
  const int max_tiles = static_cast<int>(
      partials_buf.shape(partials_buf.ndim() == 4 ? 2 : 1));
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

          KVCacheAccessor<Quantized> k_acc(k_ptr, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptr, max_seq_len, head_dim);

          for (int d = lid; d < head_dim; d += tile_wg) {
            q_cache[d] = static_cast<float>(q_ptr[head * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float score = -1e30f;
          if (lid < tile_len) {
            const int t = tile_start + lid;
            const int cache_t =
                (t < sink_len) ? t : (tail_start + (t - sink_len));
            score = 0.0f;
            for (int d = 0; d < head_dim; d += 8) {
              score += q_cache[d + 0] * k_acc.load(kv_head, cache_t, d + 0);
              score += q_cache[d + 1] * k_acc.load(kv_head, cache_t, d + 1);
              score += q_cache[d + 2] * k_acc.load(kv_head, cache_t, d + 2);
              score += q_cache[d + 3] * k_acc.load(kv_head, cache_t, d + 3);
              score += q_cache[d + 4] * k_acc.load(kv_head, cache_t, d + 4);
              score += q_cache[d + 5] * k_acc.load(kv_head, cache_t, d + 5);
              score += q_cache[d + 6] * k_acc.load(kv_head, cache_t, d + 6);
              score += q_cache[d + 7] * k_acc.load(kv_head, cache_t, d + 7);
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
              acc += logits[i] * v_acc.load(kv_head, cache_t, d);
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
            float val = static_cast<float>(bf16(acc * inv_sum));
            if (gate_ptr != nullptr) {
              float g_v = static_cast<float>(gate_ptr[head * head_dim + d]);
              float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
              val = static_cast<float>(bf16(val * s));
            }
            o_ptr[head * head_dim + d] = bf16(val);
          }
        });
  });
}

void attention_decode_exact_head256_partial_merge(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len,
    const Tensor *gate_buf = nullptr) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
    attention_decode_exact_head256_partial_merge_impl<true>(
        ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
        tail_start, sink_len, effective_len, gate_ptr);
  } else {
    attention_decode_exact_head256_partial_merge_impl<false>(
        ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
        tail_start, sink_len, effective_len, gate_ptr);
  }
}

template <bool Quantized>
void attention_decode_exact_head128_partial_merge_impl(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len,
    const bf16 *gate_ptr = nullptr) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  const void *k_ptr = k_cache.data();
  const void *v_ptr = v_cache.data();
  bf16 *o_ptr = static_cast<bf16 *>(output.data());
  float *partials_ptr = static_cast<float *>(partials_buf.data());

  constexpr int head_dim = 128;
  constexpr int tile_t = 128;
  constexpr int tile_wg = 128;
  constexpr int merge_wg = 128;

  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int max_seq_len = static_cast<int>(k_cache.shape(1));
  const int num_tiles = (effective_len + tile_t - 1) / tile_t;
  // See the head_dim 256 merge: a 4-D per-sequence block uses its tile extent.
  const int max_tiles = static_cast<int>(
      partials_buf.shape(partials_buf.ndim() == 4 ? 2 : 1));

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
              (head * max_tiles + tile_idx) * kDecodeExact128PartialStride;

          KVCacheAccessor<Quantized> k_acc(k_ptr, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptr, max_seq_len, head_dim);

          q_cache[lid] = static_cast<float>(q_ptr[head * head_dim + lid]);
          item.barrier(sycl::access::fence_space::local_space);

          float score = -1e30f;
          if (lid < tile_len) {
            const int t = tile_start + lid;
            const int cache_t =
                (t < sink_len) ? t : (tail_start + (t - sink_len));
            score = 0.0f;
            for (int d = 0; d < head_dim; d += 8) {
              score += q_cache[d + 0] * k_acc.load(kv_head, cache_t, d + 0);
              score += q_cache[d + 1] * k_acc.load(kv_head, cache_t, d + 1);
              score += q_cache[d + 2] * k_acc.load(kv_head, cache_t, d + 2);
              score += q_cache[d + 3] * k_acc.load(kv_head, cache_t, d + 3);
              score += q_cache[d + 4] * k_acc.load(kv_head, cache_t, d + 4);
              score += q_cache[d + 5] * k_acc.load(kv_head, cache_t, d + 5);
              score += q_cache[d + 6] * k_acc.load(kv_head, cache_t, d + 6);
              score += q_cache[d + 7] * k_acc.load(kv_head, cache_t, d + 7);
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

          float acc = 0.0f;
          for (int i = 0; i < tile_len; ++i) {
            const int t = tile_start + i;
            const int cache_t =
                (t < sink_len) ? t : (tail_start + (t - sink_len));
            acc += logits[i] * v_acc.load(kv_head, cache_t, lid);
          }
          partial[kDecodeExact128PartialAccOffset + lid] = acc;
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
                  (head * max_tiles + tile_idx) * kDecodeExact128PartialStride;
              global_max = sycl::fmax(global_max, partial[0]);
            }
            float sum = 0.0f;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
              const float *partial =
                  partials_ptr +
                  (head * max_tiles + tile_idx) * kDecodeExact128PartialStride;
              const float factor = sycl::native::exp(partial[0] - global_max);
              tile_factors[tile_idx] = factor;
              sum += factor * partial[1];
            }
            norm_sum[0] = sum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          const float inv_sum = 1.0f / norm_sum[0];
          float acc = 0.0f;
          for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const float *partial =
                partials_ptr +
                (head * max_tiles + tile_idx) * kDecodeExact128PartialStride;
            acc += tile_factors[tile_idx] *
                   partial[kDecodeExact128PartialAccOffset + lid];
          }
          float val = static_cast<float>(bf16(acc * inv_sum));
          if (gate_ptr != nullptr) {
            float g_v = static_cast<float>(gate_ptr[head * head_dim + lid]);
            float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
            val = static_cast<float>(bf16(val * s));
          }
          o_ptr[head * head_dim + lid] = bf16(val);
        });
  });
}

void attention_decode_exact_head128_partial_merge(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len,
    const Tensor *gate_buf = nullptr) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
    attention_decode_exact_head128_partial_merge_impl<true>(
        ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
        tail_start, sink_len, effective_len, gate_ptr);
  } else {
    attention_decode_exact_head128_partial_merge_impl<false>(
        ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
        tail_start, sink_len, effective_len, gate_ptr);
  }
}

template <int TM, int TN, int TK, int SG>
void attention_decode_joint_matrix_tiled(Context &ctx, Tensor &q,
                                         Tensor &k_cache, Tensor &v_cache,
                                         Tensor &output, Tensor &scores_buf,
                                         int num_heads, int num_kv_heads,
                                         int head_dim, int cached_len,
                                         int attn_start, int wg_size,
                                         const bf16 *gate_ptr = nullptr) {
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
  // Q (head_dim=256 bf16 = 512 bytes) fits entirely in SLM.  Pre-load
  // it once so the inner JM loop runs without barriers — the XMX engine
  // can pipeline K-cache global loads with the previous MAD.
  const int q_slm_size = head_dim;
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<bf16, 1> q_slm(sycl::range<1>(q_slm_size), cgh);
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

          // Load entire Q vector into SLM once (256 bf16 = 512 bytes).
          for (int i = lid; i < head_dim; i += SG)
            q_slm[i] = q_head[i];
          item.barrier(sycl::access::fence_space::local_space);

          for (int kb = 0; kb < head_dim; kb += TK) {
            auto a_ptr = q_slm.get_multi_ptr<sycl::access::decorated::no>();
            auto b_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(k_head_tile + kb);
            joint_matrix_load(sg, sub_a, a_ptr + kb, TK);
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

          // V accumulation: work-items 0..(head_dim/8-1) each handle 8
          // output dims with vec8 loads — coalesces V reads and shares
          // the probability across 8 dot products, cutting SLM reads 8×.
          using vec8 = sycl::vec<bf16, 8>;
          const int kv_cache_offset = kv_head * max_seq_len * head_dim;
          if (lid * 8 < head_dim) {
            float acc[8] = {};
            for (int t = 0; t < effective_len; t++) {
              int cache_t = attn_start + t;
              float p = shared[t];
              const vec8 v_row = *reinterpret_cast<const vec8*>(
                  v_ptr + kv_cache_offset + cache_t * head_dim + lid * 8);
              acc[0] += p * static_cast<float>(v_row[0]);
              acc[1] += p * static_cast<float>(v_row[1]);
              acc[2] += p * static_cast<float>(v_row[2]);
              acc[3] += p * static_cast<float>(v_row[3]);
              acc[4] += p * static_cast<float>(v_row[4]);
              acc[5] += p * static_cast<float>(v_row[5]);
              acc[6] += p * static_cast<float>(v_row[6]);
              acc[7] += p * static_cast<float>(v_row[7]);
            }
            for (int k = 0; k < 8; k++) {
              float val = static_cast<float>(bf16(acc[k]));
              if (gate_ptr != nullptr) {
                float g_v = static_cast<float>(gate_ptr[head * head_dim + lid * 8 + k]);
                float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
                val = static_cast<float>(bf16(val * s));
              }
              o_ptr[head * head_dim + lid * 8 + k] = bf16(val);
            }
          }
        });
  });
}

void attention_decode_joint_matrix(Context &ctx, Tensor &q, Tensor &k_cache,
                                   Tensor &v_cache, Tensor &output,
                                   Tensor &scores_buf, int num_heads,
                                   int num_kv_heads, int head_dim,
                                   int cached_len, int attn_start, int tile_id,
                                   int wg_size,
                                   const Tensor *gate_buf = nullptr) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  if (tile_id == 2) {
    attention_decode_joint_matrix_tiled<8, 8, 16, 8>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, attn_start, wg_size, gate_ptr);
  } else if (tile_id == 1) {
    attention_decode_joint_matrix_tiled<32, 32, 16, 16>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, attn_start, wg_size, gate_ptr);
  } else {
    attention_decode_joint_matrix_tiled<1, 8, 16, 8>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, attn_start, wg_size, gate_ptr);
  }
}

// Multi-row baseline decode attention. One work-group per (sequence, head).
// Each row keeps the single-row body verbatim, including the vec8 K dot, the
// reduce_over_group order and the V accumulation loop, so a row output equals
// the single-row kernel for the same cache and position.
template <bool Quantized>
void attention_decode_rows_baseline_impl(
    Context &ctx, Tensor &q, int q_row_stride, Tensor &output,
    Tensor &scores_buf, const DecodeRowGeometry *rows, int row_count,
    int num_heads, int num_kv_heads, int head_dim, int max_cached_len,
    int wg_size) {
  const int o_row_stride = num_heads * head_dim;
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());
  (void)scores_buf;

  const int heads_per_kv = num_heads / num_kv_heads;
  const int max_seq_len = static_cast<int>(rows[0].k_cache->shape(1));
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int padded = round_up(max_cached_len, 32);

  const void *k_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
  const void *v_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
  int cached_lens[4] = {0, 0, 0, 0};
  for (int r = 0; r < row_count; ++r) {
    k_ptrs[r] = rows[r].k_cache->data();
    v_ptrs[r] = rows[r].v_cache->data();
    cached_lens[r] = rows[r].cached_len;
  }

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> shared(sycl::range<1>(padded), cgh);
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            static_cast<size_t>(row_count) * static_cast<size_t>(num_heads) * wg_size,
            wg_size),
        [=](sycl::nd_item<1> item) {
          const int group = static_cast<int>(item.get_group(0));
          const int sequence = group / num_heads;
          const int head = group % num_heads;
          const int lid = static_cast<int>(item.get_local_id(0));
          const int kv_head = head / heads_per_kv;
          const int effective_len = cached_lens[sequence];
          const bf16 *q_row = q_ptr +
              static_cast<size_t>(sequence) * q_row_stride + head * head_dim;
          // The attention output is always a dense num_heads*head_dim row
          // stack, which is not necessarily the input q stride.
          bf16 *o_row = o_ptr +
              static_cast<size_t>(sequence) * o_row_stride + head * head_dim;

          KVCacheAccessor<Quantized> k_acc(k_ptrs[sequence], max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptrs[sequence], max_seq_len, head_dim);

          for (int d = lid; d < head_dim; d += wg_size) {
            q_cache[d] = static_cast<float>(q_row[d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int t = lid; t < effective_len; t += wg_size) {
            float sum = 0.0f;
            if ((head_dim & 7) == 0) {
              for (int d = 0; d < head_dim; d += 8) {
                sum += q_cache[d + 0] * k_acc.load(kv_head, t, d + 0);
                sum += q_cache[d + 1] * k_acc.load(kv_head, t, d + 1);
                sum += q_cache[d + 2] * k_acc.load(kv_head, t, d + 2);
                sum += q_cache[d + 3] * k_acc.load(kv_head, t, d + 3);
                sum += q_cache[d + 4] * k_acc.load(kv_head, t, d + 4);
                sum += q_cache[d + 5] * k_acc.load(kv_head, t, d + 5);
                sum += q_cache[d + 6] * k_acc.load(kv_head, t, d + 6);
                sum += q_cache[d + 7] * k_acc.load(kv_head, t, d + 7);
              }
            } else {
              for (int d = 0; d < head_dim; d++) {
                sum += q_cache[d] * k_acc.load(kv_head, t, d);
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
            const float e = sycl::native::exp(shared[t] - max_val);
            shared[t] = e;
            sum_val += e;
          }
          sum_val = sycl::reduce_over_group(item.get_group(), sum_val,
                                           sycl::plus<float>());

          for (int t = lid; t < effective_len; t += wg_size) {
            shared[t] = shared[t] / sum_val;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int d = lid; d < head_dim; d += wg_size) {
            float acc = 0.0f;
            for (int t = 0; t < effective_len; t++) {
              acc += shared[t] * v_acc.load(kv_head, t, d);
            }
              o_row[d] = bf16(acc);
            }
          });
  });
}

// Multi-row fused Joint-Matrix decode attention for head_dim 128 below 128 tokens.
// One work-group per (sequence, head). Since context length is < 128 tokens (max 128 floats = 512 bytes),
// all intermediate logits fit in local accessor (SLM) without touching global scores_buf.
template <int TM = 1, int TN = 8, int TK = 16, int SG = 8, int WG = 128>
void attention_decode_rows_joint_matrix_impl(
    Context &ctx, Tensor &q, int q_row_stride, Tensor &output,
    const DecodeRowGeometry *rows, int row_count,
    int num_heads, int num_kv_heads, int head_dim) {
  const int o_row_stride = num_heads * head_dim;
  const bf16 *q_ptr = static_cast<const bf16 *>(q.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  const int heads_per_kv = num_heads / num_kv_heads;
  const int max_seq_len = static_cast<int>(rows[0].k_cache->shape(1));
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));

  const void *k_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
  const void *v_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
  int cached_lens[4] = {0, 0, 0, 0};
  for (int r = 0; r < row_count; ++r) {
    k_ptrs[r] = rows[r].k_cache->data();
    v_ptrs[r] = rows[r].v_cache->data();
    cached_lens[r] = rows[r].cached_len;
  }

  constexpr int num_sgs = WG / SG;

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<bf16, 1> q_slm(sycl::range<1>(head_dim), cgh);
    sycl::local_accessor<float, 1> scores_slm(sycl::range<1>(128), cgh);
    sycl::local_accessor<float, 1> c_local(sycl::range<1>(num_sgs * TN), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            static_cast<size_t>(row_count) * static_cast<size_t>(num_heads) * WG, WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
          auto sg = item.get_sub_group();
          const int group = static_cast<int>(item.get_group(0));
          const int sequence = group / num_heads;
          const int head = group % num_heads;
          const int lid = static_cast<int>(item.get_local_id(0));
          const int sg_id = lid / SG;
          const int sg_lid = static_cast<int>(sg.get_local_linear_id());
          const int kv_head = head / heads_per_kv;
          const int effective_len = cached_lens[sequence];

          bf16 *o_head = o_ptr +
              static_cast<size_t>(sequence) * o_row_stride + head * head_dim;

          if (effective_len <= 0) {
            if (lid * 8 < head_dim) {
              for (int k = 0; k < 8; ++k) o_head[lid * 8 + k] = bf16(0.0f);
            }
            return;
          }

          const bf16 *q_head = q_ptr +
              static_cast<size_t>(sequence) * q_row_stride + head * head_dim;
          const bf16 *k_head = static_cast<const bf16 *>(k_ptrs[sequence]) +
              static_cast<size_t>(kv_head) * max_seq_len * head_dim;
          const bf16 *v_head = static_cast<const bf16 *>(v_ptrs[sequence]) +
              static_cast<size_t>(kv_head) * max_seq_len * head_dim;

          // Cooperative Q load into SLM
          for (int d = lid; d < head_dim; d += WG) {
            q_slm[d] = q_head[d];
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Phase 1: Joint-Matrix QK^T
          const int num_tiles = (effective_len + TN - 1) / TN;
          for (int tile = sg_id; tile < num_tiles; tile += num_sgs) {
            const int t0 = tile * TN;
            const bf16 *k_tile = k_head + t0 * head_dim;

            joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> sub_a;
            joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::col_major> sub_b;
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
            joint_matrix_fill(sg, sub_c, 0.0f);

            for (int kb = 0; kb < head_dim; kb += TK) {
              auto a_ptr = q_slm.get_multi_ptr<sycl::access::decorated::no>();
              auto b_ptr = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(k_tile + kb);
              joint_matrix_load(sg, sub_a, a_ptr + kb, TK);
              joint_matrix_load(sg, sub_b, b_ptr, head_dim);
              joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
            }

            auto c_ptr = c_local.get_multi_ptr<sycl::access::decorated::no>();
            joint_matrix_store(sg, sub_c, c_ptr + sg_id * TN, TN, layout::row_major);
            sycl::group_barrier(sg);

            if (sg_lid < TN) {
              const int t = t0 + sg_lid;
              if (t < effective_len && t < 128) {
                scores_slm[t] = c_local[sg_id * TN + sg_lid] * scale;
              }
            }
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Phase 2: Softmax
          float max_val = -1.0e38f;
          for (int t = lid; t < effective_len; t += WG) {
            max_val = sycl::fmax(max_val, scores_slm[t]);
          }
          max_val = sycl::reduce_over_group(item.get_group(), max_val, sycl::maximum<float>());

          float sum_val = 0.0f;
          for (int t = lid; t < effective_len; t += WG) {
            const float e = sycl::native::exp(scores_slm[t] - max_val);
            scores_slm[t] = e;
            sum_val += e;
          }
          sum_val = sycl::reduce_over_group(item.get_group(), sum_val, sycl::plus<float>());

          for (int t = lid; t < effective_len; t += WG) {
            scores_slm[t] = scores_slm[t] / sum_val;
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Phase 2: V Accumulation
          using vec8 = sycl::vec<bf16, 8>;
          if (lid * 8 < head_dim) {
            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (int t = 0; t < effective_len; ++t) {
              const float p = scores_slm[t];
              const vec8 v_row = *reinterpret_cast<const vec8 *>(
                  v_head + t * head_dim + lid * 8);
              acc[0] += p * static_cast<float>(v_row[0]);
              acc[1] += p * static_cast<float>(v_row[1]);
              acc[2] += p * static_cast<float>(v_row[2]);
              acc[3] += p * static_cast<float>(v_row[3]);
              acc[4] += p * static_cast<float>(v_row[4]);
              acc[5] += p * static_cast<float>(v_row[5]);
              acc[6] += p * static_cast<float>(v_row[6]);
              acc[7] += p * static_cast<float>(v_row[7]);
            }
            for (int k = 0; k < 8; ++k) {
              o_head[lid * 8 + k] = bf16(acc[k]);
            }
          }
        });
  });
}

// Multi-row exact partial merge for head_dim 128/256 decode. Every
// (sequence, head, tile) group keeps the single-row body, rounding and
// reduction order verbatim, and each sequence owns a private partials region so
// rows cannot overwrite one another. A row's output therefore stays bitwise
// identical to calling the single-row merge for that row alone.
template <bool Quantized, int HeadDim>
void attention_decode_rows_exact_merge_impl(
    Context &ctx, Tensor &q, int q_row_stride, Tensor &output,
    Tensor &partials_buf, const DecodeRowGeometry *rows, int row_count,
    int num_heads, int num_kv_heads) {
  constexpr int head_dim = HeadDim;
  constexpr int tile_t = 128;
  constexpr int tile_wg = 128;
  constexpr int merge_wg = 128;
  constexpr int partial_stride =
      (HeadDim == 256) ? kDecodeExact256PartialStride : kDecodeExact128PartialStride;
  constexpr int acc_offset =
      (HeadDim == 256) ? kDecodeExact256PartialAccOffset : kDecodeExact128PartialAccOffset;

  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());
  float *partials_ptr = static_cast<float *>(partials_buf.data());

  const int o_row_stride = num_heads * head_dim;
  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int max_seq_len = static_cast<int>(rows[0].k_cache->shape(1));
  const int max_tiles = static_cast<int>(partials_buf.shape(2));
  // Each sequence's block is separated by the buffer's own head extent, which
  // may exceed this layer's attention head count (a hybrid model sizes it for
  // its widest attention layer).
  const int buffer_heads = static_cast<int>(partials_buf.shape(1));

  const void *k_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
  const void *v_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
  int cached_lens[4] = {0, 0, 0, 0};
  int row_tiles[4] = {0, 0, 0, 0};
  for (int r = 0; r < row_count; ++r) {
    k_ptrs[r] = rows[r].k_cache->data();
    v_ptrs[r] = rows[r].v_cache->data();
    cached_lens[r] = rows[r].cached_len;
    row_tiles[r] = (cached_lens[r] + tile_t - 1) / tile_t;
  }

  // Phase 1 grid is flattened over (sequence, head, tile). row_count is at most
  // four, so locating the owning sequence is a short prefix scan. The per-row
  // tile counts differ, which a rectangular grid would either over-run or
  // waste, so the flattening keeps the launched work equal to the per-row sum.
  int total_tile_groups = 0;
  for (int r = 0; r < row_count; ++r) total_tile_groups += num_heads * row_tiles[r];

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);
    sycl::local_accessor<float, 1> logits(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> stats(sycl::range<1>(2), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            static_cast<size_t>(total_tile_groups) * static_cast<size_t>(tile_wg),
            tile_wg),
        [=](sycl::nd_item<1> item) {
          const int group = static_cast<int>(item.get_group(0));
          int sequence = 0;
          int local = group;
          for (int r = 0; r < row_count; ++r) {
            const int groups = num_heads * row_tiles[r];
            if (local < groups) {
              sequence = r;
              break;
            }
            local -= groups;
          }
          const int head = local / row_tiles[sequence];
          const int tile_idx = local % row_tiles[sequence];
          const int lid = static_cast<int>(item.get_local_id(0));
          const int kv_head = head / heads_per_kv;
          const int effective_len = cached_lens[sequence];
          const int tile_start = tile_idx * tile_t;
          const int tile_len = sycl::min(tile_t, effective_len - tile_start);
          float *partial =
              partials_ptr +
              (static_cast<size_t>(sequence) * buffer_heads + head) * max_tiles *
                      partial_stride +
              static_cast<size_t>(tile_idx) * partial_stride;

          KVCacheAccessor<Quantized> k_acc(k_ptrs[sequence], max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptrs[sequence], max_seq_len, head_dim);

          const bf16 *q_row =
              q_ptr + static_cast<size_t>(sequence) * q_row_stride + head * head_dim;
          for (int d = lid; d < head_dim; d += tile_wg) {
            q_cache[d] = static_cast<float>(q_row[d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float score = -1e30f;
          if (lid < tile_len) {
            const int t = tile_start + lid;
            score = 0.0f;
            for (int d = 0; d < head_dim; d += 8) {
              score += q_cache[d + 0] * k_acc.load(kv_head, t, d + 0);
              score += q_cache[d + 1] * k_acc.load(kv_head, t, d + 1);
              score += q_cache[d + 2] * k_acc.load(kv_head, t, d + 2);
              score += q_cache[d + 3] * k_acc.load(kv_head, t, d + 3);
              score += q_cache[d + 4] * k_acc.load(kv_head, t, d + 4);
              score += q_cache[d + 5] * k_acc.load(kv_head, t, d + 5);
              score += q_cache[d + 6] * k_acc.load(kv_head, t, d + 6);
              score += q_cache[d + 7] * k_acc.load(kv_head, t, d + 7);
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
              acc += logits[i] * v_acc.load(kv_head, t, d);
            }
            partial[acc_offset + d] = acc;
          }
        });
  });

  // Phase 2 merges each sequence's per-tile statistics and partial outputs.
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> tile_factors(sycl::range<1>(max_tiles), cgh);
    sycl::local_accessor<float, 1> norm_sum(sycl::range<1>(1), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            static_cast<size_t>(row_count) * static_cast<size_t>(num_heads) *
                static_cast<size_t>(merge_wg),
            merge_wg),
        [=](sycl::nd_item<1> item) {
          const int group = static_cast<int>(item.get_group(0));
          const int sequence = group / num_heads;
          const int head = group % num_heads;
          const int lid = static_cast<int>(item.get_local_id(0));
          const int num_tiles = row_tiles[sequence];
          const float *row_partials =
              partials_ptr +
              (static_cast<size_t>(sequence) * buffer_heads + head) * max_tiles *
                  partial_stride;

          if (lid == 0) {
            float global_max = -1e30f;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
              const float *partial =
                  row_partials + static_cast<size_t>(tile_idx) * partial_stride;
              global_max = sycl::fmax(global_max, partial[0]);
            }
            float sum = 0.0f;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
              const float *partial =
                  row_partials + static_cast<size_t>(tile_idx) * partial_stride;
              const float factor = sycl::native::exp(partial[0] - global_max);
              tile_factors[tile_idx] = factor;
              sum += factor * partial[1];
            }
            norm_sum[0] = sum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          const float inv_sum = 1.0f / norm_sum[0];
          bf16 *o_row =
              o_ptr + static_cast<size_t>(sequence) * o_row_stride + head * head_dim;
          for (int d = lid; d < head_dim; d += merge_wg) {
            float acc = 0.0f;
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
              const float *partial =
                  row_partials + static_cast<size_t>(tile_idx) * partial_stride;
              acc += tile_factors[tile_idx] * partial[acc_offset + d];
            }
            o_row[d] = bf16(acc * inv_sum);
          }
        });
  });
}
} // namespace

// ============================================================
// SYCL Kernel: Attention Decode (seq_len=1, GQA)
// ============================================================

constexpr int JM0_M = 1;
constexpr int JM0_N = 8;
constexpr int JM0_K = 16;
constexpr int JM1_M = 32;
constexpr int JM1_N = 32;
constexpr int JM1_K = 16;
constexpr int JM2_M = 8;
constexpr int JM2_N = 8;
constexpr int JM2_K = 16;

struct JmDecodeConfig {
  bool allow_jm = false;
  int jm_tile_id = -1;
  int jm_mode = -1;
  int jm_supported = -1;
  int decode_wg = 512;
  int decode_window = 0;
  int decode_window_start = 512;
  int decode_sink = 0;
};

const JmDecodeConfig& get_jm_decode_config(const sycl::device &dev) {
  static JmDecodeConfig cfg;
  static bool initialized = false;
  if (!initialized) {
    cfg.jm_mode = read_env_int_local("AILA_ATTN_JM", 1);
    cfg.decode_wg = read_env_int_local("AILA_ATTN_DECODE_WG", 512);
    if (cfg.decode_wg <= 0) cfg.decode_wg = 512;
    cfg.decode_window = read_env_int_local("AILA_ATTN_DECODE_WINDOW", 0);
    if (cfg.decode_window < 0) cfg.decode_window = 0;
    cfg.decode_window_start = read_env_int_local("AILA_ATTN_DECODE_WINDOW_START", -1);
    if (cfg.decode_window_start < 0) cfg.decode_window_start = std::max(512, cfg.decode_window);
    cfg.decode_sink = read_env_int_local("AILA_ATTN_DECODE_SINK", -1);
    if (cfg.decode_sink < 0) cfg.decode_sink = 0;

    bool s0 = supports_jm_bf16_f32(dev, JM0_M, JM0_N, JM0_K);
    bool s1 = supports_jm_bf16_f32(dev, JM1_M, JM1_N, JM1_K);
    bool s2 = supports_jm_bf16_f32(dev, JM2_M, JM2_N, JM2_K);
    cfg.jm_supported = (s0 || s1 || s2) ? 1 : 0;

    int force_tile = read_env_int_local("AILA_ATTN_JM_TILE", -1);
    if (force_tile == 2 && s2) cfg.jm_tile_id = 2;
    else if (force_tile == 1 && s1) cfg.jm_tile_id = 1;
    else if (force_tile == 0 && s0) cfg.jm_tile_id = 0;
    else if (s0) cfg.jm_tile_id = 0;
    else if (s2) cfg.jm_tile_id = 2;
    else if (s1) cfg.jm_tile_id = 1;
    else cfg.jm_tile_id = -1;

    cfg.allow_jm = (cfg.jm_tile_id >= 0) &&
        (cfg.jm_mode == 2 || (cfg.jm_mode == 1 && cfg.jm_supported == 1));
    initialized = true;
  }
  return cfg;
}

void attention_decode(Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache,
                      Tensor &output, Tensor &scores_buf, int num_heads,
                      int num_kv_heads, int head_dim, int cached_len,
                      Tensor *exact_partials_buf,
                      const Tensor *gate_buf) {
  static bool jm_log_once = false;
  const auto &jm_cfg = get_jm_decode_config(ctx.queue().get_device());
  bool allow_jm = jm_cfg.allow_jm;
  int jm_mode = jm_cfg.jm_mode;
  int jm_supported = jm_cfg.jm_supported;
  int jm_tile_id = jm_cfg.jm_tile_id;
  int decode_wg = jm_cfg.decode_wg;
  int decode_window = jm_cfg.decode_window;
  int decode_window_start = jm_cfg.decode_window_start;
  int decode_sink = jm_cfg.decode_sink;
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
    const char* status;
    if (allow_jm) {
      status = "-> enabled";
    } else if (jm_tile_id < 0 && jm_supported == 0) {
      status = "-> disabled (no compatible joint_matrix tile on this device)";
    } else {
      status = "-> fallback baseline";
    }
    AILA_LOG_INFO("[JM] mode=%d, supported=%d, tile_id=%d, decode_wg=%d, decode_window=%d, decode_window_start=%d, decode_sink=%d %s",
                  jm_mode, jm_supported, jm_tile_id, decode_wg, decode_window,
                  decode_window_start, decode_sink, status);
    jm_log_once = true;
  }
  if (head_dim == 128 && cached_len > 0 && effective_len >= kDecodeExact128MinLen &&
      exact_partials_buf && exact_partials_buf->valid()) {
    attention_decode_exact_head128_partial_merge(
        ctx, q, k_cache, v_cache, output, *exact_partials_buf, num_heads,
        num_kv_heads, tail_start, sink_len, effective_len, gate_buf);
    return;
  }

  if (allow_jm && contiguous_mode && head_dim == 128 && cached_len > 0) {
    attention_decode_joint_matrix(ctx, q, k_cache, v_cache, output, scores_buf,
                                  num_heads, num_kv_heads, head_dim, cached_len,
                                  attn_start, jm_tile_id, decode_wg, gate_buf);
    return;
  }

  if (head_dim == 256 && cached_len > 0 && effective_len >= kDecodeExact256MinLen &&
      exact_partials_buf && exact_partials_buf->valid()) {
    attention_decode_exact_head256_partial_merge(
        ctx, q, k_cache, v_cache, output, *exact_partials_buf, num_heads,
        num_kv_heads, tail_start, sink_len, effective_len, gate_buf);
    return;
  }

  attention_decode_baseline(ctx, q, k_cache, v_cache, output, scores_buf,
                            num_heads, num_kv_heads, head_dim, cached_len,
                            tail_start, sink_len, effective_len,
                            decode_wg, gate_buf);
}


void attention_decode_rows(Context &ctx, Tensor &q, int q_row_stride,
                           Tensor &output, Tensor &scores_buf,
                           const DecodeRowGeometry *rows,
                           int row_count, int num_heads, int num_kv_heads,
                           int head_dim, int max_cached_len,
                           Tensor *exact_partials_buf) {
  if (row_count < 1 || row_count > 4 || rows == nullptr) {
    throw std::invalid_argument("attention_decode_rows: row_count must be 1..4");
  }
  for (int r = 0; r < row_count; ++r) {
    if (rows[r].k_cache == nullptr || rows[r].v_cache == nullptr ||
        rows[r].cached_len < 0) {
      throw std::invalid_argument("attention_decode_rows: invalid row geometry");
    }
  }
  // One multi-row kernel covers a row when it matches the single-row choice
  // exactly, so its body, rounding and reduction order are reused verbatim:
  // baseline below the exact-merge threshold, exact partial merge at or above
  // it. A decode window or sink makes effective_len differ from cached_len,
  // head_dim 128 below 128 tokens prefers the joint matrix, an unusable
  // partials region has no merge kernel, and rows straddling a threshold need
  // two different bodies; all of those keep the original per-row calls.
  const int window = read_env_int_local("AILA_ATTN_DECODE_WINDOW", 0);
  const int sink = read_env_int_local("AILA_ATTN_DECODE_SINK", 0);
  // AILA_ATTN_DECODE_ROWS=0 restores the per-row decode attention calls, which
  // lets an A/B comparison run both variants from one binary.
  static const bool multi_row_enabled =
      read_env_int_local("AILA_ATTN_DECODE_ROWS", 1) != 0;
  const bool rows_eligible =
      multi_row_enabled && row_count > 1 && window <= 0 && sink <= 0;
  // The multi-row merge kernels address a private region per sequence, so the
  // partials buffer must be laid out [row, head, tile, stride]. Anything else
  // (a 3-D single-row region, or too few rows) keeps the per-row path.
  const bool row_partials_ok = exact_partials_buf != nullptr &&
      exact_partials_buf->valid() && exact_partials_buf->ndim() == 4 &&
      exact_partials_buf->shape(0) >= row_count &&
      exact_partials_buf->shape(1) >= num_heads;
  // Whether the single-row dispatcher would reach the exact merge depends only
  // on the buffer being present and valid, not on this row-mapped rank.
  const bool partialsUsable =
      exact_partials_buf != nullptr && exact_partials_buf->valid();

  auto per_row = [&]() {
    for (int r = 0; r < row_count; ++r) {
      // q rows live at q_row_stride in the caller buffer (the fused QKV row
      // when there is no output gate), while the attention output is always a
      // dense num_heads*head_dim row stack.
      auto qRow = Tensor::view(ctx, q.data_as<bf16>() +
          static_cast<size_t>(r) * static_cast<size_t>(q_row_stride),
          {1, static_cast<int64_t>(num_heads) * head_dim}, q.dtype());
      auto oRow = Tensor::view(ctx, output.data_as<bf16>() +
          static_cast<size_t>(r) * static_cast<size_t>(num_heads) * head_dim,
          {1, static_cast<int64_t>(num_heads) * head_dim}, output.dtype());
      // The single-row merge indexes [head, tile, stride]; slice this row's
      // block out of the [row, head, tile, stride] region so a per-row call
      // cannot read another sequence's partials.
      Tensor rowPartials;
      if (row_partials_ok) {
        const int64_t bufferHeads = exact_partials_buf->shape(1);
        const int64_t maxTiles = exact_partials_buf->shape(2);
        const int64_t stride = exact_partials_buf->shape(3);
        auto* base = exact_partials_buf->data_as<float>() +
            static_cast<size_t>(r) * static_cast<size_t>(bufferHeads) *
                static_cast<size_t>(maxTiles) * static_cast<size_t>(stride);
        rowPartials = Tensor::view(ctx, base,
            {bufferHeads, maxTiles, stride},
            exact_partials_buf->dtype());
      }
      Tensor* partialsArg = row_partials_ok ? &rowPartials : exact_partials_buf;
      attention_decode(ctx, qRow, *rows[r].k_cache, *rows[r].v_cache, oRow,
                       scores_buf, num_heads, num_kv_heads, head_dim,
                       rows[r].cached_len, partialsArg);
    }
  };

  // Without a window or sink, the single-row dispatcher picks the head_dim 256
  // exact merge exactly when the partials region is usable and the row has
  // reached the threshold; every other head_dim 256 row is baseline.
  bool all_baseline = rows_eligible && head_dim == 256;
  for (int r = 0; r < row_count && all_baseline; ++r) {
    if (partialsUsable && rows[r].cached_len >= kDecodeExact256MinLen) {
      all_baseline = false;
    }
  }
  const int merge_min_len =
      (head_dim == 128) ? kDecodeExact128MinLen : kDecodeExact256MinLen;
  bool all_merge = rows_eligible && row_partials_ok &&
                   (head_dim == 128 || head_dim == 256);
  for (int r = 0; r < row_count && all_merge; ++r) {
    if (rows[r].cached_len < merge_min_len) {
      all_merge = false;
    }
  }

  const bool quantized =
      rows[0].k_cache->dtype() == dnnl::memory::data_type::f8_e4m3;
  for (int r = 1; r < row_count; ++r) {
    if ((rows[r].k_cache->dtype() == dnnl::memory::data_type::f8_e4m3) !=
        quantized) {
      throw std::invalid_argument("attention_decode_rows: mixed cache dtypes");
    }
  }

  const auto &jm_cfg = get_jm_decode_config(ctx.queue().get_device());
  bool all_jm = rows_eligible && jm_cfg.allow_jm && head_dim == 128 && !quantized;
  for (int r = 0; r < row_count && all_jm; ++r) {
    if (rows[r].cached_len >= kDecodeExact128MinLen || rows[r].cached_len <= 0) {
      all_jm = false;
    }
  }
  // Report each (head_dim, path) once so a run's evidence shows which kernel was
  // selected, rather than inferring it from timings. The long-context merge,
  // the short-context baseline, and the short-context joint matrix are distinct entries.
  {
    const int pathIndex = all_merge ? 0 : (all_baseline ? 1 : (all_jm ? 2 : 3));
    static bool reported[2][4] = {{false, false, false, false}, {false, false, false, false}};
    const int dimIndex = (head_dim == 128) ? 0 : 1;
    if (!reported[dimIndex][pathIndex]) {
      reported[dimIndex][pathIndex] = true;
      AILA_LOG_INFO("[ATTN] decode rows=%d head_dim=%d path=%s", row_count,
                    head_dim,
                    all_merge ? "multi-row-exact-merge"
                              : (all_baseline ? "multi-row-baseline"
                                              : (all_jm ? "multi-row-joint-matrix" : "per-row")));
    }
  }
  if (!all_baseline && !all_merge && !all_jm) {
    per_row();
    return;
  }
  if (all_jm) {
    attention_decode_rows_joint_matrix_impl(
        ctx, q, q_row_stride, output, rows, row_count, num_heads, num_kv_heads, head_dim);
    return;
  }
  if (all_merge) {
    if (head_dim == 128) {
      if (quantized) {
        attention_decode_rows_exact_merge_impl<true, 128>(
            ctx, q, q_row_stride, output, *exact_partials_buf, rows, row_count,
            num_heads, num_kv_heads);
      } else {
        attention_decode_rows_exact_merge_impl<false, 128>(
            ctx, q, q_row_stride, output, *exact_partials_buf, rows, row_count,
            num_heads, num_kv_heads);
      }
    } else if (quantized) {
      attention_decode_rows_exact_merge_impl<true, 256>(
          ctx, q, q_row_stride, output, *exact_partials_buf, rows, row_count,
          num_heads, num_kv_heads);
    } else {
      attention_decode_rows_exact_merge_impl<false, 256>(
          ctx, q, q_row_stride, output, *exact_partials_buf, rows, row_count,
          num_heads, num_kv_heads);
    }
    return;
  }
  constexpr int wg_size = 512;
  if (quantized) {
    attention_decode_rows_baseline_impl<true>(
        ctx, q, q_row_stride, output, scores_buf, rows, row_count, num_heads,
        num_kv_heads,
        head_dim, max_cached_len, wg_size);
  } else {
    attention_decode_rows_baseline_impl<false>(
        ctx, q, q_row_stride, output, scores_buf, rows, row_count, num_heads,
        num_kv_heads,
        head_dim, max_cached_len, wg_size);
  }
}

// ============================================================
// SYCL Kernel: Attention Prefill (seq_len > 1)
// ============================================================
// ============================================================
// SYCL Kernel: Attention Prefill (seq_len > 1)
// ============================================================

void attention_prefill(Context &ctx, Tensor &q, Tensor &k, Tensor &v,
                       Tensor &output, Tensor &scores_buf, int seq_len,
                       int num_heads, int num_kv_heads, int head_dim,
                       const Tensor *gate_buf) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  if (head_dim == 128 && seq_len >= kPrefillExact128MinSeq) {
    bf16 *q_ptr = static_cast<bf16 *>(q.data());
    bf16 *k_ptr = static_cast<bf16 *>(k.data());
    bf16 *v_ptr = static_cast<bf16 *>(v.data());
    bf16 *o_ptr = static_cast<bf16 *>(output.data());

    constexpr int tile_t = 128;
    constexpr int wg_size = 128;
    constexpr int exact_head_dim = 128;
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
            q_cache[lid] = static_cast<float>(q_row[lid]);
            acc_local[lid] = 0.0f;
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
              float tile_acc = 0.0f;
              for (int j = 0; j < tile_len; ++j) {
                const bf16 *v_row =
                    v_ptr + (tile_start + j) * k_total + kv_h * exact_head_dim;
                tile_acc += scores[j] * static_cast<float>(v_row[lid]);
              }
              acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
              item.barrier(sycl::access::fence_space::local_space);
            }

            const float inv_l = 1.0f / merge_state[1];
            bf16 *o_row = o_ptr + qi * q_total + h * exact_head_dim;
            const bf16 *g_row = gate_ptr ? gate_ptr + qi * q_total + h * exact_head_dim : nullptr;
            float val = static_cast<float>(bf16(acc_local[lid] * inv_l));
            if (g_row) {
              float g_v = static_cast<float>(g_row[lid]);
              float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
              val = static_cast<float>(bf16(val * s));
            }
            o_row[lid] = bf16(val);
          });
    });
    return;
  }

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
            const bf16 *g_row = gate_ptr ? gate_ptr + qi * q_total + h * exact_head_dim : nullptr;
            for (int d = lid; d < exact_head_dim; d += wg_size) {
              float val = static_cast<float>(bf16(acc_local[d] * inv_l));
              if (g_row) {
                float g_v = static_cast<float>(g_row[d]);
                float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
                val = static_cast<float>(bf16(val * s));
              }
              o_row[d] = bf16(val);
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
        float val = static_cast<float>(bf16(acc));
        if (gate_ptr) {
          float g_v = static_cast<float>(gate_ptr[qi * q_total + h * head_dim + d]);
          float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
          val = static_cast<float>(bf16(val * s));
        }
        o_ptr[qi * q_total + h * head_dim + d] = bf16(val);
      });
}

void attention_bidi_exact_head64(Context &ctx, Tensor &q, Tensor &k, Tensor &v,
                                 Tensor &output, int seq_len,
                                 int num_heads) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  constexpr int exact_head_dim = kVisionBidiExact64HeadDim;
  constexpr int tile_t = kVisionBidiExact64Tile;
  constexpr int wg_size = kVisionBidiExact64Wg;
  constexpr float scale = 0.125f;
  static_assert(exact_head_dim == 64);
  static_assert(tile_t == 2 * wg_size);
  const int total_dim = num_heads * exact_head_dim;

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
          const int lid = item.get_local_id(0);
          const int lid_hi = lid + wg_size;

          const bf16 *q_row = q_ptr + qi * total_dim + h * exact_head_dim;
          q_cache[lid] = static_cast<float>(q_row[lid]);
          acc_local[lid] = 0.0f;
          if (lid == 0) {
            merge_state[0] = -1e30f;
            merge_state[1] = 0.0f;
            merge_state[2] = 0.0f;
            merge_state[3] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score_lo = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * exact_head_dim;
              score_lo = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score_lo += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score_lo += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score_lo += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score_lo += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score_lo += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score_lo += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score_lo += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score_lo += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score_lo *= scale;
              scores[lid] = score_lo;
            } else {
              scores[lid] = -1e30f;
            }

            float score_hi = -1e30f;
            if (lid_hi < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid_hi) * total_dim + h * exact_head_dim;
              score_hi = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score_hi += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score_hi += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score_hi += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score_hi += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score_hi += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score_hi += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score_hi += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score_hi += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score_hi *= scale;
              scores[lid_hi] = score_hi;
            } else {
              scores[lid_hi] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(), sycl::fmax(score_lo, score_hi), sycl::maximum<float>());

            float exp_score_lo = 0.0f;
            if (lid < tile_len) {
              exp_score_lo = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score_lo;
            } else {
              scores[lid] = 0.0f;
            }

            float exp_score_hi = 0.0f;
            if (lid_hi < tile_len) {
              exp_score_hi = sycl::native::exp(scores[lid_hi] - tile_max);
              scores[lid_hi] = exp_score_hi;
            } else {
              scores[lid_hi] = 0.0f;
            }

            const float tile_sum = sycl::reduce_over_group(
                item.get_group(), exp_score_lo + exp_score_hi, sycl::plus<float>());

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
            float tile_acc = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
              const bf16 *v_row =
                  v_ptr + (tile_start + j) * total_dim + h * exact_head_dim;
              tile_acc += scores[j] * static_cast<float>(v_row[lid]);
            }
            acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * exact_head_dim;
          o_row[lid] = bf16(acc_local[lid] * inv_l);
        });
  });
}

void attention_bidi_exact_head64_vec8(Context &ctx, Tensor &q, Tensor &k,
                                      Tensor &v, Tensor &output, int seq_len,
                                      int num_heads) {
  using bvec8 = sycl::vec<bf16, 8>;
  using fvec8 = sycl::vec<float, 8>;

  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  constexpr int exact_head_dim = kVisionBidiExact64HeadDim;
  constexpr int exact_head_chunks = exact_head_dim / 8;
  constexpr int tile_t = kVisionBidiExact64Tile;
  constexpr int wg_size = kVisionBidiExact64Wg;
  constexpr float scale = 0.125f;
  static_assert(exact_head_dim == 64);
  static_assert(tile_t == 2 * wg_size);
  const int total_dim = num_heads * exact_head_dim;

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<fvec8, 1> q_cache(sycl::range<1>(exact_head_chunks), cgh);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> acc_local(sycl::range<1>(exact_head_dim), cgh);
    sycl::local_accessor<float, 1> merge_state(sycl::range<1>(4), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * seq_len * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
          const int group = item.get_group(0);
          const int h = group / seq_len;
          const int qi = group % seq_len;
          const int lid = item.get_local_id(0);
          const int lid_hi = lid + wg_size;

          const bf16 *q_row = q_ptr + qi * total_dim + h * exact_head_dim;
          if (lid < exact_head_chunks) {
            const bvec8 *q_vec = reinterpret_cast<const bvec8 *>(q_row);
            bvec8 q_block_bf16 = q_vec[lid];
            fvec8 q_block;
            for (int lane = 0; lane < 8; ++lane) {
              q_block[lane] = static_cast<float>(q_block_bf16[lane]);
            }
            q_cache[lid] = q_block;
          }
          acc_local[lid] = 0.0f;
          if (lid == 0) {
            merge_state[0] = -1e30f;
            merge_state[1] = 0.0f;
            merge_state[2] = 0.0f;
            merge_state[3] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score_lo = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score_lo = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score_lo += prod[0] + prod[1] + prod[2] + prod[3] +
                            prod[4] + prod[5] + prod[6] + prod[7];
              }
              score_lo *= scale;
              scores[lid] = score_lo;
            } else {
              scores[lid] = -1e30f;
            }

            float score_hi = -1e30f;
            if (lid_hi < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid_hi) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score_hi = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score_hi += prod[0] + prod[1] + prod[2] + prod[3] +
                            prod[4] + prod[5] + prod[6] + prod[7];
              }
              score_hi *= scale;
              scores[lid_hi] = score_hi;
            } else {
              scores[lid_hi] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(), sycl::fmax(score_lo, score_hi),
                sycl::maximum<float>());

            float exp_score_lo = 0.0f;
            if (lid < tile_len) {
              exp_score_lo = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score_lo;
            } else {
              scores[lid] = 0.0f;
            }

            float exp_score_hi = 0.0f;
            if (lid_hi < tile_len) {
              exp_score_hi = sycl::native::exp(scores[lid_hi] - tile_max);
              scores[lid_hi] = exp_score_hi;
            } else {
              scores[lid_hi] = 0.0f;
            }

            const float tile_sum = sycl::reduce_over_group(
                item.get_group(), exp_score_lo + exp_score_hi,
                sycl::plus<float>());

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
            float tile_acc = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
              const bf16 *v_row =
                  v_ptr + (tile_start + j) * total_dim + h * exact_head_dim;
              tile_acc += scores[j] * static_cast<float>(v_row[lid]);
            }
            acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * exact_head_dim;
          o_row[lid] = bf16(acc_local[lid] * inv_l);
        });
  });
}

void attention_bidi_exact_head64_sg16(Context &ctx, Tensor &q, Tensor &k,
                                      Tensor &v, Tensor &output, int seq_len,
                                      int num_heads) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  constexpr int exact_head_dim = kVisionBidiExact64HeadDim;
  constexpr int tile_t = kVisionBidiExact64Tile;
  constexpr int wg_size = kVisionBidiExact64Wg;
  constexpr float scale = 0.125f;
  static_assert(exact_head_dim == 64);
  static_assert(tile_t == 2 * wg_size);
  const int total_dim = num_heads * exact_head_dim;

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(exact_head_dim), cgh);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> acc_local(sycl::range<1>(exact_head_dim), cgh);
    sycl::local_accessor<float, 1> merge_state(sycl::range<1>(4), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * seq_len * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kVisionBidiExact64SgAlt)]] {
          const int group = item.get_group(0);
          const int h = group / seq_len;
          const int qi = group % seq_len;
          const int lid = item.get_local_id(0);
          const int lid_hi = lid + wg_size;

          const bf16 *q_row = q_ptr + qi * total_dim + h * exact_head_dim;
          q_cache[lid] = static_cast<float>(q_row[lid]);
          acc_local[lid] = 0.0f;
          if (lid == 0) {
            merge_state[0] = -1e30f;
            merge_state[1] = 0.0f;
            merge_state[2] = 0.0f;
            merge_state[3] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score_lo = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * exact_head_dim;
              score_lo = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score_lo += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score_lo += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score_lo += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score_lo += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score_lo += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score_lo += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score_lo += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score_lo += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score_lo *= scale;
              scores[lid] = score_lo;
            } else {
              scores[lid] = -1e30f;
            }

            float score_hi = -1e30f;
            if (lid_hi < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid_hi) * total_dim + h * exact_head_dim;
              score_hi = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score_hi += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score_hi += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score_hi += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score_hi += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score_hi += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score_hi += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score_hi += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score_hi += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score_hi *= scale;
              scores[lid_hi] = score_hi;
            } else {
              scores[lid_hi] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(), sycl::fmax(score_lo, score_hi),
                sycl::maximum<float>());

            float exp_score_lo = 0.0f;
            if (lid < tile_len) {
              exp_score_lo = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score_lo;
            } else {
              scores[lid] = 0.0f;
            }

            float exp_score_hi = 0.0f;
            if (lid_hi < tile_len) {
              exp_score_hi = sycl::native::exp(scores[lid_hi] - tile_max);
              scores[lid_hi] = exp_score_hi;
            } else {
              scores[lid_hi] = 0.0f;
            }

            const float tile_sum = sycl::reduce_over_group(
                item.get_group(), exp_score_lo + exp_score_hi,
                sycl::plus<float>());

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
            float tile_acc = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
              const bf16 *v_row =
                  v_ptr + (tile_start + j) * total_dim + h * exact_head_dim;
              tile_acc += scores[j] * static_cast<float>(v_row[lid]);
            }
            acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * exact_head_dim;
          o_row[lid] = bf16(acc_local[lid] * inv_l);
        });
  });
}

void attention_bidi_exact_head64_tile192_vec8(Context &ctx, Tensor &q, Tensor &k,
                                              Tensor &v, Tensor &output,
                                              int seq_len, int num_heads,
                                              bool unroll_v) {
  using bvec8 = sycl::vec<bf16, 8>;
  using fvec8 = sycl::vec<float, 8>;

  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  constexpr int exact_head_dim = kVisionBidiExact64HeadDim;
  constexpr int exact_head_chunks = exact_head_dim / 8;
  constexpr int tile_t = kVisionBidiExact64TileAltMid;
  constexpr int wg_size = kVisionBidiExact64Wg;
  constexpr float scale = 0.125f;
  static_assert(exact_head_dim == 64);
  static_assert(tile_t == 3 * wg_size);
  const int total_dim = num_heads * exact_head_dim;

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<fvec8, 1> q_cache(sycl::range<1>(exact_head_chunks), cgh);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> acc_local(sycl::range<1>(exact_head_dim), cgh);
    sycl::local_accessor<float, 1> merge_state(sycl::range<1>(4), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * seq_len * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
          const int group = item.get_group(0);
          const int h = group / seq_len;
          const int qi = group % seq_len;
          const int lid = item.get_local_id(0);
          const int lid1 = lid + wg_size;
          const int lid2 = lid1 + wg_size;

          const bf16 *q_row = q_ptr + qi * total_dim + h * exact_head_dim;
          if (lid < exact_head_chunks) {
            const bvec8 *q_vec = reinterpret_cast<const bvec8 *>(q_row);
            bvec8 q_block_bf16 = q_vec[lid];
            fvec8 q_block;
            for (int lane = 0; lane < 8; ++lane) {
              q_block[lane] = static_cast<float>(q_block_bf16[lane]);
            }
            q_cache[lid] = q_block;
          }
          acc_local[lid] = 0.0f;
          if (lid == 0) {
            merge_state[0] = -1e30f;
            merge_state[1] = 0.0f;
            merge_state[2] = 0.0f;
            merge_state[3] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score0 = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score0 = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score0 += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] +
                          prod[5] + prod[6] + prod[7];
              }
              score0 *= scale;
              scores[lid] = score0;
            } else {
              scores[lid] = -1e30f;
            }

            float score1 = -1e30f;
            if (lid1 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid1) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score1 = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score1 += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] +
                          prod[5] + prod[6] + prod[7];
              }
              score1 *= scale;
              scores[lid1] = score1;
            } else {
              scores[lid1] = -1e30f;
            }

            float score2 = -1e30f;
            if (lid2 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid2) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score2 = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score2 += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] +
                          prod[5] + prod[6] + prod[7];
              }
              score2 *= scale;
              scores[lid2] = score2;
            } else {
              scores[lid2] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(), sycl::fmax(score0, sycl::fmax(score1, score2)),
                sycl::maximum<float>());

            float exp_score0 = 0.0f;
            if (lid < tile_len) {
              exp_score0 = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score0;
            } else {
              scores[lid] = 0.0f;
            }

            float exp_score1 = 0.0f;
            if (lid1 < tile_len) {
              exp_score1 = sycl::native::exp(scores[lid1] - tile_max);
              scores[lid1] = exp_score1;
            } else {
              scores[lid1] = 0.0f;
            }

            float exp_score2 = 0.0f;
            if (lid2 < tile_len) {
              exp_score2 = sycl::native::exp(scores[lid2] - tile_max);
              scores[lid2] = exp_score2;
            } else {
              scores[lid2] = 0.0f;
            }

            const float tile_sum = sycl::reduce_over_group(
                item.get_group(), exp_score0 + exp_score1 + exp_score2,
                sycl::plus<float>());

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
            float tile_acc = 0.0f;
            if (unroll_v) {
              int j = 0;
              for (; j + 3 < tile_len; j += 4) {
                const bf16 *v_row0 =
                    v_ptr + (tile_start + j + 0) * total_dim + h * exact_head_dim;
                const bf16 *v_row1 =
                    v_ptr + (tile_start + j + 1) * total_dim + h * exact_head_dim;
                const bf16 *v_row2 =
                    v_ptr + (tile_start + j + 2) * total_dim + h * exact_head_dim;
                const bf16 *v_row3 =
                    v_ptr + (tile_start + j + 3) * total_dim + h * exact_head_dim;
                tile_acc += scores[j + 0] * static_cast<float>(v_row0[lid]) +
                            scores[j + 1] * static_cast<float>(v_row1[lid]) +
                            scores[j + 2] * static_cast<float>(v_row2[lid]) +
                            scores[j + 3] * static_cast<float>(v_row3[lid]);
              }
              for (; j < tile_len; ++j) {
                const bf16 *v_row =
                    v_ptr + (tile_start + j) * total_dim + h * exact_head_dim;
                tile_acc += scores[j] * static_cast<float>(v_row[lid]);
              }
            } else {
              for (int j = 0; j < tile_len; ++j) {
                const bf16 *v_row =
                    v_ptr + (tile_start + j) * total_dim + h * exact_head_dim;
                tile_acc += scores[j] * static_cast<float>(v_row[lid]);
              }
            }
            acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * exact_head_dim;
          o_row[lid] = bf16(acc_local[lid] * inv_l);
        });
  });
}

void attention_bidi_exact_head64_tile192_vec8_sg16(Context &ctx, Tensor &q,
                                                   Tensor &k, Tensor &v,
                                                   Tensor &output, int seq_len,
                                                   int num_heads,
                                                   bool unroll_v) {
  using bvec8 = sycl::vec<bf16, 8>;
  using fvec8 = sycl::vec<float, 8>;

  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  constexpr int exact_head_dim = kVisionBidiExact64HeadDim;
  constexpr int exact_head_chunks = exact_head_dim / 8;
  constexpr int tile_t = kVisionBidiExact64TileAltMid;
  constexpr int wg_size = kVisionBidiExact64Wg;
  constexpr float scale = 0.125f;
  static_assert(exact_head_dim == 64);
  static_assert(tile_t == 3 * wg_size);
  const int total_dim = num_heads * exact_head_dim;

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<fvec8, 1> q_cache(sycl::range<1>(exact_head_chunks), cgh);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> acc_local(sycl::range<1>(exact_head_dim), cgh);
    sycl::local_accessor<float, 1> merge_state(sycl::range<1>(4), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * seq_len * wg_size, wg_size),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kVisionBidiExact64SgAlt)]] {
          const int group = item.get_group(0);
          const int h = group / seq_len;
          const int qi = group % seq_len;
          const int lid = item.get_local_id(0);
          const int lid1 = lid + wg_size;
          const int lid2 = lid1 + wg_size;

          const bf16 *q_row = q_ptr + qi * total_dim + h * exact_head_dim;
          if (lid < exact_head_chunks) {
            const bvec8 *q_vec = reinterpret_cast<const bvec8 *>(q_row);
            bvec8 q_block_bf16 = q_vec[lid];
            fvec8 q_block;
            for (int lane = 0; lane < 8; ++lane) {
              q_block[lane] = static_cast<float>(q_block_bf16[lane]);
            }
            q_cache[lid] = q_block;
          }
          acc_local[lid] = 0.0f;
          if (lid == 0) {
            merge_state[0] = -1e30f;
            merge_state[1] = 0.0f;
            merge_state[2] = 0.0f;
            merge_state[3] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score0 = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score0 = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score0 += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] +
                          prod[5] + prod[6] + prod[7];
              }
              score0 *= scale;
              scores[lid] = score0;
            } else {
              scores[lid] = -1e30f;
            }

            float score1 = -1e30f;
            if (lid1 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid1) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score1 = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score1 += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] +
                          prod[5] + prod[6] + prod[7];
              }
              score1 *= scale;
              scores[lid1] = score1;
            } else {
              scores[lid1] = -1e30f;
            }

            float score2 = -1e30f;
            if (lid2 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid2) * total_dim + h * exact_head_dim;
              const bvec8 *k_vec = reinterpret_cast<const bvec8 *>(k_row);
              score2 = 0.0f;
              for (int d = 0; d < exact_head_chunks; ++d) {
                bvec8 k_block_bf16 = k_vec[d];
                fvec8 k_block;
                for (int lane = 0; lane < 8; ++lane) {
                  k_block[lane] = static_cast<float>(k_block_bf16[lane]);
                }
                const fvec8 prod = q_cache[d] * k_block;
                score2 += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] +
                          prod[5] + prod[6] + prod[7];
              }
              score2 *= scale;
              scores[lid2] = score2;
            } else {
              scores[lid2] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(), sycl::fmax(score0, sycl::fmax(score1, score2)),
                sycl::maximum<float>());

            float exp_score0 = 0.0f;
            if (lid < tile_len) {
              exp_score0 = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score0;
            } else {
              scores[lid] = 0.0f;
            }

            float exp_score1 = 0.0f;
            if (lid1 < tile_len) {
              exp_score1 = sycl::native::exp(scores[lid1] - tile_max);
              scores[lid1] = exp_score1;
            } else {
              scores[lid1] = 0.0f;
            }

            float exp_score2 = 0.0f;
            if (lid2 < tile_len) {
              exp_score2 = sycl::native::exp(scores[lid2] - tile_max);
              scores[lid2] = exp_score2;
            } else {
              scores[lid2] = 0.0f;
            }

            const float tile_sum = sycl::reduce_over_group(
                item.get_group(), exp_score0 + exp_score1 + exp_score2,
                sycl::plus<float>());

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
            float tile_acc = 0.0f;
            const bf16 *v_tile =
                v_ptr + tile_start * total_dim + h * exact_head_dim;
            if (unroll_v) {
              float tile_acc0 = 0.0f;
              float tile_acc1 = 0.0f;
              float tile_acc2 = 0.0f;
              float tile_acc3 = 0.0f;
              int j = 0;
              for (; j + 3 < tile_len; j += 4) {
                const float s0 = scores[j + 0];
                const float s1 = scores[j + 1];
                const float s2 = scores[j + 2];
                const float s3 = scores[j + 3];
                tile_acc0 +=
                    s0 * static_cast<float>(v_tile[(j + 0) * total_dim + lid]);
                tile_acc1 +=
                    s1 * static_cast<float>(v_tile[(j + 1) * total_dim + lid]);
                tile_acc2 +=
                    s2 * static_cast<float>(v_tile[(j + 2) * total_dim + lid]);
                tile_acc3 +=
                    s3 * static_cast<float>(v_tile[(j + 3) * total_dim + lid]);
              }
              tile_acc = (tile_acc0 + tile_acc1) + (tile_acc2 + tile_acc3);
              for (; j < tile_len; ++j) {
                tile_acc +=
                    scores[j] * static_cast<float>(v_tile[j * total_dim + lid]);
              }
            } else {
              for (int j = 0; j < tile_len; ++j) {
                tile_acc +=
                    scores[j] * static_cast<float>(v_tile[j * total_dim + lid]);
              }
            }
            acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * exact_head_dim;
          o_row[lid] = bf16(acc_local[lid] * inv_l);
        });
  });
}

void attention_bidi_exact_head64_tile192(Context &ctx, Tensor &q, Tensor &k,
                                         Tensor &v, Tensor &output,
                                         int seq_len, int num_heads) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  constexpr int exact_head_dim = kVisionBidiExact64HeadDim;
  constexpr int tile_t = kVisionBidiExact64TileAltMid;
  constexpr int wg_size = kVisionBidiExact64Wg;
  constexpr float scale = 0.125f;
  static_assert(exact_head_dim == 64);
  static_assert(tile_t == 3 * wg_size);
  const int total_dim = num_heads * exact_head_dim;

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
          const int lid = item.get_local_id(0);
          const int lid1 = lid + wg_size;
          const int lid2 = lid1 + wg_size;

          const bf16 *q_row = q_ptr + qi * total_dim + h * exact_head_dim;
          q_cache[lid] = static_cast<float>(q_row[lid]);
          acc_local[lid] = 0.0f;
          if (lid == 0) {
            merge_state[0] = -1e30f;
            merge_state[1] = 0.0f;
            merge_state[2] = 0.0f;
            merge_state[3] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score0 = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * exact_head_dim;
              score0 = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score0 += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score0 += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score0 += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score0 += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score0 += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score0 += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score0 += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score0 += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score0 *= scale;
              scores[lid] = score0;
            } else {
              scores[lid] = -1e30f;
            }

            float score1 = -1e30f;
            if (lid1 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid1) * total_dim + h * exact_head_dim;
              score1 = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score1 += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score1 += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score1 += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score1 += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score1 += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score1 += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score1 += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score1 += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score1 *= scale;
              scores[lid1] = score1;
            } else {
              scores[lid1] = -1e30f;
            }

            float score2 = -1e30f;
            if (lid2 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid2) * total_dim + h * exact_head_dim;
              score2 = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score2 += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score2 += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score2 += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score2 += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score2 += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score2 += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score2 += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score2 += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score2 *= scale;
              scores[lid2] = score2;
            } else {
              scores[lid2] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(), sycl::fmax(score0, sycl::fmax(score1, score2)),
                sycl::maximum<float>());

            float exp_score0 = 0.0f;
            if (lid < tile_len) {
              exp_score0 = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score0;
            } else {
              scores[lid] = 0.0f;
            }

            float exp_score1 = 0.0f;
            if (lid1 < tile_len) {
              exp_score1 = sycl::native::exp(scores[lid1] - tile_max);
              scores[lid1] = exp_score1;
            } else {
              scores[lid1] = 0.0f;
            }

            float exp_score2 = 0.0f;
            if (lid2 < tile_len) {
              exp_score2 = sycl::native::exp(scores[lid2] - tile_max);
              scores[lid2] = exp_score2;
            } else {
              scores[lid2] = 0.0f;
            }

            const float tile_sum = sycl::reduce_over_group(
                item.get_group(), exp_score0 + exp_score1 + exp_score2,
                sycl::plus<float>());

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
            float tile_acc = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
              const bf16 *v_row =
                  v_ptr + (tile_start + j) * total_dim + h * exact_head_dim;
              tile_acc += scores[j] * static_cast<float>(v_row[lid]);
            }
            acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * exact_head_dim;
          o_row[lid] = bf16(acc_local[lid] * inv_l);
        });
  });
}

void attention_bidi_exact_head64_tile256(Context &ctx, Tensor &q, Tensor &k,
                                         Tensor &v, Tensor &output,
                                         int seq_len, int num_heads) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  constexpr int exact_head_dim = kVisionBidiExact64HeadDim;
  constexpr int tile_t = kVisionBidiExact64TileAlt;
  constexpr int wg_size = kVisionBidiExact64Wg;
  constexpr float scale = 0.125f;
  static_assert(exact_head_dim == 64);
  static_assert(tile_t == 4 * wg_size);
  const int total_dim = num_heads * exact_head_dim;

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
          const int lid = item.get_local_id(0);
          const int lid1 = lid + wg_size;
          const int lid2 = lid1 + wg_size;
          const int lid3 = lid2 + wg_size;

          const bf16 *q_row = q_ptr + qi * total_dim + h * exact_head_dim;
          q_cache[lid] = static_cast<float>(q_row[lid]);
          acc_local[lid] = 0.0f;
          if (lid == 0) {
            merge_state[0] = -1e30f;
            merge_state[1] = 0.0f;
            merge_state[2] = 0.0f;
            merge_state[3] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score0 = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * exact_head_dim;
              score0 = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score0 += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score0 += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score0 += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score0 += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score0 += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score0 += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score0 += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score0 += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score0 *= scale;
              scores[lid] = score0;
            } else {
              scores[lid] = -1e30f;
            }

            float score1 = -1e30f;
            if (lid1 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid1) * total_dim + h * exact_head_dim;
              score1 = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score1 += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score1 += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score1 += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score1 += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score1 += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score1 += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score1 += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score1 += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score1 *= scale;
              scores[lid1] = score1;
            } else {
              scores[lid1] = -1e30f;
            }

            float score2 = -1e30f;
            if (lid2 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid2) * total_dim + h * exact_head_dim;
              score2 = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score2 += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score2 += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score2 += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score2 += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score2 += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score2 += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score2 += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score2 += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score2 *= scale;
              scores[lid2] = score2;
            } else {
              scores[lid2] = -1e30f;
            }

            float score3 = -1e30f;
            if (lid3 < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid3) * total_dim + h * exact_head_dim;
              score3 = 0.0f;
              for (int d = 0; d < exact_head_dim; d += 8) {
                score3 += q_cache[d + 0] * static_cast<float>(k_row[d + 0]);
                score3 += q_cache[d + 1] * static_cast<float>(k_row[d + 1]);
                score3 += q_cache[d + 2] * static_cast<float>(k_row[d + 2]);
                score3 += q_cache[d + 3] * static_cast<float>(k_row[d + 3]);
                score3 += q_cache[d + 4] * static_cast<float>(k_row[d + 4]);
                score3 += q_cache[d + 5] * static_cast<float>(k_row[d + 5]);
                score3 += q_cache[d + 6] * static_cast<float>(k_row[d + 6]);
                score3 += q_cache[d + 7] * static_cast<float>(k_row[d + 7]);
              }
              score3 *= scale;
              scores[lid3] = score3;
            } else {
              scores[lid3] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(),
                sycl::fmax(sycl::fmax(score0, score1), sycl::fmax(score2, score3)),
                sycl::maximum<float>());

            float exp_score0 = 0.0f;
            if (lid < tile_len) {
              exp_score0 = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score0;
            } else {
              scores[lid] = 0.0f;
            }

            float exp_score1 = 0.0f;
            if (lid1 < tile_len) {
              exp_score1 = sycl::native::exp(scores[lid1] - tile_max);
              scores[lid1] = exp_score1;
            } else {
              scores[lid1] = 0.0f;
            }

            float exp_score2 = 0.0f;
            if (lid2 < tile_len) {
              exp_score2 = sycl::native::exp(scores[lid2] - tile_max);
              scores[lid2] = exp_score2;
            } else {
              scores[lid2] = 0.0f;
            }

            float exp_score3 = 0.0f;
            if (lid3 < tile_len) {
              exp_score3 = sycl::native::exp(scores[lid3] - tile_max);
              scores[lid3] = exp_score3;
            } else {
              scores[lid3] = 0.0f;
            }

            const float tile_sum = sycl::reduce_over_group(
                item.get_group(),
                exp_score0 + exp_score1 + exp_score2 + exp_score3,
                sycl::plus<float>());

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
            float tile_acc = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
              const bf16 *v_row =
                  v_ptr + (tile_start + j) * total_dim + h * exact_head_dim;
              tile_acc += scores[j] * static_cast<float>(v_row[lid]);
            }
            acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * exact_head_dim;
          o_row[lid] = bf16(acc_local[lid] * inv_l);
        });
  });
}

void attention_bidi(Context& ctx, Tensor& q, Tensor& k, Tensor& v,
                    Tensor& output, Tensor& scores_buf,
                    int seq_len, int num_heads, int head_dim,
                    float scale) {
  (void)scores_buf;
  if (seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
    return;
  }

  static int vision_exact64_tile = -1;
  static int vision_exact64_sg = -1;
  static int vision_exact64_vec = -1;
  static int vision_exact64_vunroll = -1;
  static bool vision_exact64_log_once = false;
  if (vision_exact64_tile < 0) {
    vision_exact64_tile =
        read_env_int_local("AILA_ATTN_VISION_BIDI_EXACT64_TILE",
                           kVisionBidiExact64TileDefault);
    if (vision_exact64_tile <= 0) {
      vision_exact64_tile = kVisionBidiExact64TileDefault;
    }
  }
  if (vision_exact64_sg < 0) {
    vision_exact64_sg =
        read_env_int_local("AILA_ATTN_VISION_BIDI_EXACT64_SG",
                           kVisionBidiExact64SgDefault);
    if (vision_exact64_sg < 0) {
      vision_exact64_sg = kVisionBidiExact64SgDefault;
    }
  }
  if (vision_exact64_vec < 0) {
    vision_exact64_vec =
        read_env_int_local("AILA_ATTN_VISION_BIDI_EXACT64_VEC",
                           kVisionBidiExact64VecDefault);
    if (vision_exact64_vec < 0) {
      vision_exact64_vec = kVisionBidiExact64VecDefault;
    }
  }
  if (vision_exact64_vunroll < 0) {
    vision_exact64_vunroll = read_env_int_local(
        "AILA_ATTN_VISION_BIDI_EXACT64_VUNROLL",
        kVisionBidiExact64VUnrollDefault);
    if (vision_exact64_vunroll < 0) {
      vision_exact64_vunroll = kVisionBidiExact64VUnrollDefault;
    }
  }

  if (head_dim == kVisionBidiExact64HeadDim && seq_len >= kVisionBidiExact64MinSeq) {
    if (!vision_exact64_log_once &&
        (vision_exact64_tile != kVisionBidiExact64TileDefault ||
         vision_exact64_sg != kVisionBidiExact64SgDefault ||
         vision_exact64_vec != kVisionBidiExact64VecDefault ||
         vision_exact64_vunroll != kVisionBidiExact64VUnrollDefault)) {
      const bool alt_tile = vision_exact64_tile == kVisionBidiExact64TileAlt ||
                            vision_exact64_tile == kVisionBidiExact64TileAltMid;
      const bool alt_sg = vision_exact64_sg == kVisionBidiExact64SgAlt;
      const bool alt_vec = vision_exact64_vec == kVisionBidiExact64VecEnabled;
      AILA_LOG_INFO("[VisionBidiExact64] tile=%d sg=%d vec=%d vunroll=%d -> %s",
                    vision_exact64_tile, vision_exact64_sg, vision_exact64_vec,
                    vision_exact64_vunroll,
                    (alt_tile || alt_sg || alt_vec ||
                     vision_exact64_vunroll != kVisionBidiExact64VUnrollDefault)
                        ? "enabled alt path"
                        : "fallback baseline");
      vision_exact64_log_once = true;
    }

    if (vision_exact64_tile == kVisionBidiExact64TileAltMid) {
      if (vision_exact64_vec == kVisionBidiExact64VecEnabled) {
        if (vision_exact64_sg == kVisionBidiExact64SgAlt) {
          attention_bidi_exact_head64_tile192_vec8_sg16(
              ctx, q, k, v, output, seq_len, num_heads,
              vision_exact64_vunroll != 0);
        } else {
          attention_bidi_exact_head64_tile192_vec8(
              ctx, q, k, v, output, seq_len, num_heads,
              vision_exact64_vunroll != 0);
        }
      } else {
        attention_bidi_exact_head64_tile192(ctx, q, k, v, output, seq_len,
                                            num_heads);
      }
      return;
    }

    if (vision_exact64_tile == kVisionBidiExact64TileAlt) {
      attention_bidi_exact_head64_tile256(ctx, q, k, v, output, seq_len,
                                          num_heads);
      return;
    }

    if (vision_exact64_sg == kVisionBidiExact64SgAlt) {
      attention_bidi_exact_head64_sg16(ctx, q, k, v, output, seq_len,
                                       num_heads);
      return;
    }

    if (vision_exact64_vec == kVisionBidiExact64VecEnabled) {
      attention_bidi_exact_head64_vec8(ctx, q, k, v, output, seq_len,
                                       num_heads);
      return;
    }

    attention_bidi_exact_head64(ctx, q, k, v, output, seq_len, num_heads);
    return;
  }

  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  bf16 *k_ptr = static_cast<bf16 *>(k.data());
  bf16 *v_ptr = static_cast<bf16 *>(v.data());
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  const float effective_scale =
      (scale > 0.0f) ? scale : (1.0f / sycl::sqrt(static_cast<float>(head_dim)));
  const int total_dim = num_heads * head_dim;
  const int wg_size = (head_dim <= 64 && seq_len <= 64) ? 64 : 128;
  const int tile_t = wg_size;

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> acc_local(sycl::range<1>(head_dim), cgh);
    sycl::local_accessor<float, 1> merge_state(sycl::range<1>(4), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_heads * seq_len * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
          const int group = item.get_group(0);
          const int h = group / seq_len;
          const int qi = group % seq_len;
          const int lid = item.get_local_id(0);

          const bf16 *q_row = q_ptr + qi * total_dim + h * head_dim;
          for (int d = lid; d < head_dim; d += wg_size) {
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

          for (int tile_start = 0; tile_start < seq_len; tile_start += tile_t) {
            const int tile_len = sycl::min(tile_t, seq_len - tile_start);

            float score = -1e30f;
            if (lid < tile_len) {
              const bf16 *k_row =
                  k_ptr + (tile_start + lid) * total_dim + h * head_dim;
              score = 0.0f;
              for (int d = 0; d < head_dim; ++d) {
                score += q_cache[d] * static_cast<float>(k_row[d]);
              }
              score *= effective_scale;
              scores[lid] = score;
            } else {
              scores[lid] = -1e30f;
            }

            const float tile_max = sycl::reduce_over_group(
                item.get_group(), score, sycl::maximum<float>());

            float exp_score = 0.0f;
            if (lid < tile_len) {
              exp_score = sycl::native::exp(scores[lid] - tile_max);
              scores[lid] = exp_score;
            } else {
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
            for (int d = lid; d < head_dim; d += wg_size) {
              float tile_acc = 0.0f;
              for (int j = 0; j < tile_len; ++j) {
                const bf16 *v_row =
                    v_ptr + (tile_start + j) * total_dim + h * head_dim;
                tile_acc += scores[j] * static_cast<float>(v_row[d]);
              }
              acc_local[d] = alpha * acc_local[d] + beta * tile_acc;
            }
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float inv_l = merge_state[1] > 0.0f ? 1.0f / merge_state[1] : 0.0f;
          bf16 *o_row = o_ptr + qi * total_dim + h * head_dim;
          for (int d = lid; d < head_dim; d += wg_size) {
            o_row[d] = bf16(acc_local[d] * inv_l);
          }
        });
  });
}

// ============================================================
// SYCL Kernel: Incremental Prefill Attention
// ============================================================

template <bool Quantized>
void attention_prefill_cached_impl(Context& ctx,
                                   Tensor& q, Tensor& k_cache, Tensor& v_cache,
                                   Tensor& output, Tensor& scores_buf,
                                   int seq_len, int start_pos,
                                   int num_heads, int num_kv_heads,
                                   int head_dim, int max_seq_len,
                                   const bf16* gate_ptr = nullptr) {
  const int total_len = start_pos + seq_len;
  if (head_dim == 128 && total_len >= kPrefillCachedExact128MinTotal) {
    bf16 *q_ptr = static_cast<bf16 *>(q.data());
    const void *k_ptr = k_cache.data();
    const void *v_ptr = v_cache.data();
    bf16 *o_ptr = static_cast<bf16 *>(output.data());

    constexpr int tile_t = 128;
    constexpr int wg_size = 128;
    constexpr int exact_head_dim = 128;
    const int heads_per_kv = num_heads / num_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(exact_head_dim));
    const int q_total = num_heads * exact_head_dim;

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
            const int q_end = start_pos + qi;

            const bf16 *q_row = q_ptr + qi * q_total + h * exact_head_dim;
            q_cache[lid] = static_cast<float>(q_row[lid]);
            acc_local[lid] = 0.0f;
            if (lid == 0) {
              merge_state[0] = -1e30f;
              merge_state[1] = 0.0f;
              merge_state[2] = 0.0f;
              merge_state[3] = 0.0f;
            }

            KVCacheAccessor<Quantized> k_acc(k_ptr, max_seq_len, exact_head_dim);
            KVCacheAccessor<Quantized> v_acc(v_ptr, max_seq_len, exact_head_dim);

            item.barrier(sycl::access::fence_space::local_space);

            for (int tile_start = 0; tile_start <= q_end; tile_start += tile_t) {
              const int tile_len = sycl::min(tile_t, q_end + 1 - tile_start);

              float score = -1e30f;
              if (lid < tile_len) {
                const int key_idx = tile_start + lid;
                score = 0.0f;
                for (int d = 0; d < exact_head_dim; d += 8) {
                  score += q_cache[d + 0] * k_acc.load(kv_h, key_idx, d + 0);
                  score += q_cache[d + 1] * k_acc.load(kv_h, key_idx, d + 1);
                  score += q_cache[d + 2] * k_acc.load(kv_h, key_idx, d + 2);
                  score += q_cache[d + 3] * k_acc.load(kv_h, key_idx, d + 3);
                  score += q_cache[d + 4] * k_acc.load(kv_h, key_idx, d + 4);
                  score += q_cache[d + 5] * k_acc.load(kv_h, key_idx, d + 5);
                  score += q_cache[d + 6] * k_acc.load(kv_h, key_idx, d + 6);
                  score += q_cache[d + 7] * k_acc.load(kv_h, key_idx, d + 7);
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
              float tile_acc = 0.0f;
              for (int j = 0; j < tile_len; ++j) {
                const int key_idx = tile_start + j;
                tile_acc += scores[j] * v_acc.load(kv_h, key_idx, lid);
              }
              acc_local[lid] = alpha * acc_local[lid] + beta * tile_acc;
              item.barrier(sycl::access::fence_space::local_space);
            }

            const float inv_l = 1.0f / merge_state[1];
            bf16 *o_row = o_ptr + qi * q_total + h * exact_head_dim;
            const bf16 *g_row = gate_ptr ? gate_ptr + qi * q_total + h * exact_head_dim : nullptr;
            float val = static_cast<float>(bf16(acc_local[lid] * inv_l));
            if (g_row) {
              float g_v = static_cast<float>(g_row[lid]);
              float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
              val = static_cast<float>(bf16(val * s));
            }
            o_row[lid] = bf16(val);
          });
    });
    return;
  }

  if (head_dim == 256 && total_len >= kPrefillCachedExact256MinTotal) {
    bf16 *q_ptr = static_cast<bf16 *>(q.data());
    const void *k_ptr = k_cache.data();
    const void *v_ptr = v_cache.data();
    bf16 *o_ptr = static_cast<bf16 *>(output.data());

    constexpr int tile_t = kPrefillExact256Tile;
    constexpr int wg_size = kPrefillExact256Wg;
    constexpr int exact_head_dim = 256;
    const int heads_per_kv = num_heads / num_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(exact_head_dim));
    const int q_total = num_heads * exact_head_dim;

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
            const int q_end = start_pos + qi;

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

            KVCacheAccessor<Quantized> k_acc(k_ptr, max_seq_len, exact_head_dim);
            KVCacheAccessor<Quantized> v_acc(v_ptr, max_seq_len, exact_head_dim);

            item.barrier(sycl::access::fence_space::local_space);

            for (int tile_start = 0; tile_start <= q_end; tile_start += tile_t) {
              const int tile_len = sycl::min(tile_t, q_end + 1 - tile_start);

              float score = -1e30f;
              if (lid < tile_len) {
                const int key_idx = tile_start + lid;
                score = 0.0f;
                for (int d = 0; d < exact_head_dim; d += 8) {
                  score += q_cache[d + 0] * k_acc.load(kv_h, key_idx, d + 0);
                  score += q_cache[d + 1] * k_acc.load(kv_h, key_idx, d + 1);
                  score += q_cache[d + 2] * k_acc.load(kv_h, key_idx, d + 2);
                  score += q_cache[d + 3] * k_acc.load(kv_h, key_idx, d + 3);
                  score += q_cache[d + 4] * k_acc.load(kv_h, key_idx, d + 4);
                  score += q_cache[d + 5] * k_acc.load(kv_h, key_idx, d + 5);
                  score += q_cache[d + 6] * k_acc.load(kv_h, key_idx, d + 6);
                  score += q_cache[d + 7] * k_acc.load(kv_h, key_idx, d + 7);
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
                  const int key_idx = tile_start + j;
                  tile_acc += scores[j] * v_acc.load(kv_h, key_idx, d);
                }
                acc_local[d] = alpha * acc_local[d] + beta * tile_acc;
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            const float inv_l = 1.0f / merge_state[1];
            bf16 *o_row = o_ptr + qi * q_total + h * exact_head_dim;
            const bf16 *g_row = gate_ptr ? gate_ptr + qi * q_total + h * exact_head_dim : nullptr;
            for (int d = lid; d < exact_head_dim; d += wg_size) {
              float val = static_cast<float>(bf16(acc_local[d] * inv_l));
              if (g_row) {
                float g_v = static_cast<float>(g_row[d]);
                float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
                val = static_cast<float>(bf16(val * s));
              }
              o_row[d] = bf16(val);
            }
          });
    });
    return;
  }

  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  const void *k_ptr = k_cache.data();
  const void *v_ptr = v_cache.data();
  bf16 *o_ptr = static_cast<bf16 *>(output.data());

  int heads_per_kv = num_heads / num_kv_heads;
  float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  int q_total = num_heads * head_dim;

  float *scores_device = static_cast<float *>(scores_buf.data());
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

        KVCacheAccessor<Quantized> k_acc(k_ptr, max_seq_len, head_dim);

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
          float qv = static_cast<float>(q_ptr[qi * q_total + h * head_dim + d]);
          float kv = k_acc.load(kv_h, ki, d);
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

        KVCacheAccessor<Quantized> v_acc(v_ptr, max_seq_len, head_dim);

        float acc = 0.0f;
        float *row = scores_device + h * seq_len * total_len + qi * total_len;
        for (int t = 0; t <= start_pos + qi; t++) {
          float vv = v_acc.load(kv_h, t, d);
          acc += row[t] * vv;
        }
        float val = static_cast<float>(bf16(acc));
        if (gate_ptr) {
          float g_v = static_cast<float>(gate_ptr[qi * q_total + h * head_dim + d]);
          float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
          val = static_cast<float>(bf16(val * s));
        }
        o_ptr[qi * q_total + h * head_dim + d] = bf16(val);
      });
}

void attention_prefill_cached(Context& ctx,
                              Tensor& q, Tensor& k_cache, Tensor& v_cache,
                              Tensor& output, Tensor& scores_buf,
                              int seq_len, int start_pos,
                              int num_heads, int num_kv_heads,
                              int head_dim, int max_seq_len,
                              const Tensor* gate_buf) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  if (k_cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
    attention_prefill_cached_impl<true>(
        ctx, q, k_cache, v_cache, output, scores_buf, seq_len, start_pos,
        num_heads, num_kv_heads, head_dim, max_seq_len, gate_ptr);
  } else {
    attention_prefill_cached_impl<false>(
        ctx, q, k_cache, v_cache, output, scores_buf, seq_len, start_pos,
        num_heads, num_kv_heads, head_dim, max_seq_len, gate_ptr);
  }
}

// ============================================================
// SYCL Kernel: Copy K/V to Cache
// ============================================================

template <bool Quantized>
void copy_to_cache_impl(Context &ctx, Tensor &new_kv, Tensor &cache, int seq_len,
                        int start_pos, int num_heads, int head_dim,
                        int max_seq_len) {
  bf16 *src = static_cast<bf16 *>(new_kv.data());
  void *dst_void = cache.data();
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
                             if constexpr (Quantized) {
                               uint8_t *dst = static_cast<uint8_t *>(dst_void);
                               dst[dst_idx] = float_to_fp8_e4m3fn(static_cast<float>(src[src_idx]));
                             } else {
                               bf16 *dst = static_cast<bf16 *>(dst_void);
                               dst[dst_idx] = src[src_idx];
                             }
                           });
}

void copy_to_cache(Context &ctx, Tensor &new_kv, Tensor &cache, int seq_len,
                   int start_pos, int num_heads, int head_dim,
                   int max_seq_len) {
  if (cache.dtype() == dnnl::memory::data_type::f8_e4m3) {
    copy_to_cache_impl<true>(ctx, new_kv, cache, seq_len, start_pos, num_heads, head_dim, max_seq_len);
  } else {
    copy_to_cache_impl<false>(ctx, new_kv, cache, seq_len, start_pos, num_heads, head_dim, max_seq_len);
  }
}

namespace {

struct MaskedAttnKey {
    int q_len;
    int kv_len;
    int num_heads;
    int num_kv_heads;
    int head_dim;

    bool operator==(const MaskedAttnKey& other) const {
        return q_len == other.q_len && kv_len == other.kv_len &&
               num_heads == other.num_heads &&
               num_kv_heads == other.num_kv_heads &&
               head_dim == other.head_dim;
    }
};

struct MaskedAttnKeyHash {
    size_t operator()(const MaskedAttnKey& key) const {
        size_t h = std::hash<int>()(key.q_len);
        h = h * 31 + std::hash<int>()(key.kv_len);
        h = h * 31 + std::hash<int>()(key.num_heads);
        h = h * 31 + std::hash<int>()(key.num_kv_heads);
        h = h * 31 + std::hash<int>()(key.head_dim);
        return h;
    }
};

struct MaskedAttnPrims {
    dnnl::matmul qk_prim{}, pv_prim{};
    dnnl::memory::desc qk_src_md{}, qk_w_md{}, qk_dst_md{};
    dnnl::memory::desc pv_src_md{}, pv_w_md{}, pv_dst_md{};
    dnnl::memory::desc qk_scratch_md{}, pv_scratch_md{};
    Tensor expanded_q, expanded_k, expanded_v, probs_bf16, attn_tmp;
};

struct MaskedAttentionWorkspace {
    explicit MaskedAttentionWorkspace(Context&) {}
    std::unordered_map<MaskedAttnKey, MaskedAttnPrims, MaskedAttnKeyHash> cache;
};

template <typename T>
auto local_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space,
                                    sycl::access::decorated::no>(p);
}

void run_fused_masked_attention_xmx(
    sycl::queue& q,
    const bf16* Q_ptr,
    const bf16* K_ptr,
    const bf16* V_ptr,
    const float* Mask_ptr,
    bf16* Out_ptr,
    int q_len, int kv_len, int H, int KV, int D) {
    constexpr int BM = 32;
    constexpr int BK = 32;
    constexpr int BD = 32;
    constexpr int TM = 8, TN = 8, TK = 16, SG = 8;
    constexpr int NUM_SGS = (BM / TM) * (BK / TN); // 16 SGs
    constexpr int WG_SIZE = NUM_SGS * SG; // 128 threads
    constexpr int stride_s = 384;

    const int grid_m = (q_len + BM - 1) / BM;
    const int total_wgs = grid_m * H;

    q.submit([&](sycl::handler& cgh) {
        // SLM buffers (Total: 16 KB + 24 KB + 1 KB + 1 KB + 4 KB = 46 KB < 64 KB limit)
        sycl::local_accessor<bf16, 1> Qs(sycl::range<1>(BM * D), cgh);         // 16 KB
        sycl::local_accessor<bf16, 1> S_scores(sycl::range<1>(BM * stride_s), cgh); // 24 KB
        sycl::local_accessor<bf16, 1> Ks(sycl::range<1>(TK * BK), cgh);        // 1 KB
        sycl::local_accessor<bf16, 1> Vs(sycl::range<1>(TK * BD), cgh);        // 1 KB
        sycl::local_accessor<float, 1> Slm_out(sycl::range<1>(BM * BD), cgh);  // 4 KB

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(total_wgs) * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int wg_idx = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / SG;
                const int sg_row = sg_id / 4; // 0..3
                const int sg_col = sg_id % 4; // 0..3

                const int q_tile_idx = wg_idx % grid_m;
                const int h = wg_idx / grid_m;
                const int kv_h = h / (H / KV);
                const int q_start = q_tile_idx * BM;

                for (int i = lid; i < BM * D; i += WG_SIZE) {
                    const int r = i / D;
                    const int c = i % D;
                    const int q_pos = q_start + r;
                    Qs[i] = (q_pos < q_len) ? Q_ptr[static_cast<size_t>(q_pos) * (H * D) + h * D + c] : bf16(0.0f);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k_tile = 0; k_tile < kv_len; k_tile += BK) {
                    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> s_mat;
                    joint_matrix_fill(sg, s_mat, 0.0f);

                    for (int d_step = 0; d_step < D; d_step += TK) {
                        for (int i = lid; i < TK * BK; i += WG_SIZE) {
                            const int r = i / BK;
                            const int c = i % BK;
                            const int k_pos = k_tile + c;
                            const int d_pos = d_step + r;
                            Ks[i] = (k_pos < kv_len && d_pos < D) ? K_ptr[static_cast<size_t>(k_pos) * (KV * D) + kv_h * D + d_pos] : bf16(0.0f);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> a_mat;
                        joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::row_major> b_mat;
                        joint_matrix_load(sg, a_mat, local_ptr(Qs.get_multi_ptr<sycl::access::decorated::no>().get() + (sg_row * TM) * D + d_step), D);
                        joint_matrix_load(sg, b_mat, local_ptr(Ks.get_multi_ptr<sycl::access::decorated::no>().get() + (sg_col * TN)), BK);
                        joint_matrix_mad(sg, s_mat, a_mat, b_mat, s_mat);
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    float* s_out = Slm_out.get_multi_ptr<sycl::access::decorated::no>().get() + (sg_row * TM) * BK + (sg_col * TN);
                    joint_matrix_store(sg, s_mat, local_ptr(s_out), BK, layout::row_major);
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int i = lid; i < BM * BK; i += WG_SIZE) {
                        const int r = i / BK;
                        const int c = i % BK;
                        if (k_tile + c < kv_len) {
                            S_scores[r * stride_s + (k_tile + c)] = bf16(Slm_out[i]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Phase 2: Mask + Softmax per row in S_scores -> in-place P_probs
                for (int r = lid; r < BM; r += WG_SIZE) {
                    const int q_pos = q_start + r;
                    if (q_pos < q_len) {
                        const float* mrow = Mask_ptr + static_cast<size_t>(q_pos) * kv_len;
                        float max_val = -1.0e30f;
                        for (int k = 0; k < kv_len; ++k) {
                            float s = static_cast<float>(S_scores[r * stride_s + k]) + mrow[k];
                            max_val = sycl::max(max_val, s);
                        }
                        float sum = 0.0f;
                        for (int k = 0; k < kv_len; ++k) {
                            float s = static_cast<float>(S_scores[r * stride_s + k]) + mrow[k];
                            const float diff = s - max_val;
                            const float e = (diff > -80.0f) ? sycl::exp(diff) : 0.0f;
                            sum += e;
                        }
                        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                        for (int k = 0; k < kv_len; ++k) {
                            float s = static_cast<float>(S_scores[r * stride_s + k]) + mrow[k];
                            const float diff = s - max_val;
                            const float e = (diff > -80.0f) ? sycl::exp(diff) : 0.0f;
                            S_scores[r * stride_s + k] = bf16(e * inv_sum);
                        }
                        for (int k = kv_len; k < stride_s; ++k) {
                            S_scores[r * stride_s + k] = bf16(0.0f);
                        }
                    } else {
                        for (int k = 0; k < stride_s; ++k) {
                            S_scores[r * stride_s + k] = bf16(0.0f);
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Phase 3: Out[BM, D] = S_scores[BM, kv_len] @ V[kv_len, D]
                for (int d_tile = 0; d_tile < D; d_tile += BD) {
                    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> out_mat;
                    joint_matrix_fill(sg, out_mat, 0.0f);

                    for (int m_step = 0; m_step < kv_len; m_step += TK) {
                        for (int i = lid; i < TK * BD; i += WG_SIZE) {
                            const int r = i / BD;
                            const int c = i % BD;
                            const int v_pos = m_step + r;
                            const int d_pos = d_tile + c;
                            Vs[i] = (v_pos < kv_len && d_pos < D) ? V_ptr[static_cast<size_t>(v_pos) * (KV * D) + kv_h * D + d_pos] : bf16(0.0f);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        joint_matrix<sycl::sub_group, bf16, use::a, TM, TK, layout::row_major> a_mat;
                        joint_matrix<sycl::sub_group, bf16, use::b, TK, TN, layout::row_major> b_mat;
                        joint_matrix_load(sg, a_mat, local_ptr(S_scores.get_multi_ptr<sycl::access::decorated::no>().get() + (sg_row * TM) * stride_s + m_step), stride_s);
                        joint_matrix_load(sg, b_mat, local_ptr(Vs.get_multi_ptr<sycl::access::decorated::no>().get() + (sg_col * TN)), BD);
                        joint_matrix_mad(sg, out_mat, a_mat, b_mat, out_mat);
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    float* o_out = Slm_out.get_multi_ptr<sycl::access::decorated::no>().get() + (sg_row * TM) * BD + (sg_col * TN);
                    joint_matrix_store(sg, out_mat, local_ptr(o_out), BD, layout::row_major);
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int i = lid; i < BM * BD; i += WG_SIZE) {
                        const int r = i / BD;
                        const int c = i % BD;
                        const int q_pos = q_start + r;
                        const int d_pos = d_tile + c;
                        if (q_pos < q_len && d_pos < D) {
                            Out_ptr[static_cast<size_t>(q_pos) * (H * D) + h * D + d_pos] = bf16(Slm_out[i]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
            });
    });
}

} // namespace

void attention_prefill_masked(Context& ctx,
                              Tensor& q, Tensor& k, Tensor& v,
                              Tensor& mask, Tensor& output,
                              Tensor& scores_buf,
                              int q_len, int kv_len,
                              int num_heads, int num_kv_heads,
                              int head_dim) {
    if (num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
        throw std::runtime_error(
            "attention_prefill_masked: invalid head configuration");
    }

    static const bool fused_xmx =
        aila::env::read_flag("AILA_DG_FUSED_ATTN", false);
    if (fused_xmx && kv_len <= 384 && head_dim == 256 &&
        num_heads == 16 && num_kv_heads == 8) {
        run_fused_masked_attention_xmx(
            ctx.queue(),
            q.data_as<bf16>(),
            k.data_as<bf16>(),
            v.data_as<bf16>(),
            mask.data_as<float>(),
            output.data_as<bf16>(),
            q_len, kv_len, num_heads, num_kv_heads, head_dim);
        return;
    }

    const int groups = num_heads / num_kv_heads;

    auto& cache = ctx.workspace<MaskedAttentionWorkspace>().cache;
    MaskedAttnKey key{q_len, kv_len, num_heads, num_kv_heads, head_dim};
    auto entry = cache.find(key);
    if (entry == cache.end()) {
        ctx.make_primitive_cache_room(cache);
        // Publish only a complete entry. Failed allocations release their
        // tensors instead of retaining an initialized entry with null buffers.
        MaskedAttnPrims prims;
        prims.expanded_q = Tensor::allocate(ctx, {num_heads, q_len, head_dim});
        prims.expanded_k = Tensor::allocate(ctx, {num_heads, head_dim, kv_len});
        prims.expanded_v = Tensor::allocate(ctx, {num_heads, kv_len, head_dim});
        prims.probs_bf16 = Tensor::allocate(ctx, {num_heads, q_len, kv_len});
        prims.attn_tmp = Tensor::allocate(ctx, {num_heads, q_len, head_dim});

        prims.qk_src_md = dnnl::memory::desc(
            {num_heads, q_len, head_dim}, dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::abc);
        prims.qk_w_md = dnnl::memory::desc(
            {num_heads, head_dim, kv_len}, dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::abc);
        prims.qk_dst_md = dnnl::memory::desc(
            {num_heads, q_len, kv_len}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::abc);
        prims.pv_src_md = dnnl::memory::desc(
            {num_heads, q_len, kv_len}, dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::abc);
        prims.pv_w_md = dnnl::memory::desc(
            {num_heads, kv_len, head_dim}, dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::abc);
        prims.pv_dst_md = dnnl::memory::desc(
            {num_heads, q_len, head_dim}, dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::abc);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        auto qk_pd = dnnl::matmul::primitive_desc(
            ctx.engine(), prims.qk_src_md, prims.qk_w_md, prims.qk_dst_md, attr);
        prims.qk_prim = dnnl::matmul(qk_pd);
        prims.qk_scratch_md = qk_pd.scratchpad_desc();
        auto pv_pd = dnnl::matmul::primitive_desc(
            ctx.engine(), prims.pv_src_md, prims.pv_w_md, prims.pv_dst_md, attr);
        prims.pv_prim = dnnl::matmul(pv_pd);
        prims.pv_scratch_md = pv_pd.scratchpad_desc();
        entry = cache.emplace(key, std::move(prims)).first;
    }
    auto& prims = entry->second;

    Tensor expanded_q = Tensor::view(
        ctx, prims.expanded_q.data(), {num_heads, q_len, head_dim},
        dnnl::memory::data_type::bf16);
    Tensor expanded_k = Tensor::view(
        ctx, prims.expanded_k.data(), {num_heads, head_dim, kv_len},
        dnnl::memory::data_type::bf16);
    Tensor expanded_v = Tensor::view(
        ctx, prims.expanded_v.data(), {num_heads, kv_len, head_dim},
        dnnl::memory::data_type::bf16);
    Tensor probs_bf16 = Tensor::view(
        ctx, prims.probs_bf16.data(), {num_heads, q_len, kv_len},
        dnnl::memory::data_type::bf16);
    Tensor attn_tmp = Tensor::view(
        ctx, prims.attn_tmp.data(), {num_heads, q_len, head_dim},
        dnnl::memory::data_type::bf16);
    AILA_LOG_DEBUG("[Attn] buffers q=%d t=%d h=%d kv=%d d=%d", q_len, kv_len,
                   num_heads, num_kv_heads, head_dim);

    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* eq_ptr = static_cast<bf16*>(expanded_q.data());
    bf16* ek_ptr = static_cast<bf16*>(expanded_k.data());
    bf16* ev_ptr = static_cast<bf16*>(expanded_v.data());
    const int kv_dim = num_kv_heads * head_dim;

    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(num_heads) * q_len * head_dim),
        [=](sycl::id<1> idx) {
            const int total = num_heads * q_len * head_dim;
            const int d = static_cast<int>(idx[0]) % head_dim;
            const int qq = (static_cast<int>(idx[0]) / head_dim) % q_len;
            const int h = static_cast<int>(idx[0]) / (head_dim * q_len);
            (void)total;
            const int src_q = qq * (num_heads * head_dim) + h * head_dim + d;
            eq_ptr[(static_cast<size_t>(h) * q_len + qq) * head_dim + d] =
                q_ptr[src_q];
        });
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(num_heads) * kv_len * head_dim),
        [=](sycl::id<1> idx) {
            const int d = static_cast<int>(idx[0]) % head_dim;
            const int t = (static_cast<int>(idx[0]) / head_dim) % kv_len;
            const int h = static_cast<int>(idx[0]) / (head_dim * kv_len);
            const int kv_h = h / groups;
            const int src_idx = t * kv_dim + kv_h * head_dim + d;
            ek_ptr[(static_cast<size_t>(h) * head_dim + d) * kv_len + t] =
                k_ptr[src_idx];
            ev_ptr[idx[0]] = v_ptr[src_idx];
        });
    AILA_LOG_DEBUG("[Attn] expand submitted");

    AILA_LOG_DEBUG("[Attn] prims ready");

    dnnl::memory q_mem = expanded_q.make_dnnl_memory(prims.qk_src_md);
    dnnl::memory k_mem = expanded_k.make_dnnl_memory(prims.qk_w_md);
    dnnl::memory scores_mem = scores_buf.make_dnnl_memory(prims.qk_dst_md);
    std::unordered_map<int, dnnl::memory> qk_args{
        {DNNL_ARG_SRC, q_mem}, {DNNL_ARG_WEIGHTS, k_mem}, {DNNL_ARG_DST, scores_mem}};
    if (prims.qk_scratch_md.get_size()) {
        qk_args[DNNL_ARG_SCRATCHPAD] = ctx.scratchpad_memory(prims.qk_scratch_md);
    }
    prims.qk_prim.execute(ctx.stream(), qk_args);
    AILA_LOG_DEBUG("[Attn] qk executed");

    float* scores_ptr = static_cast<float*>(scores_buf.data());
    const float* mask_ptr = static_cast<const float*>(mask.data());
    bf16* probs_ptr = static_cast<bf16*>(probs_bf16.data());

    // Apply mask, softmax over kv_len in fp32, write bf16 probabilities.
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(num_heads) * q_len),
        [=](sycl::id<1> idx) {
            const int h = static_cast<int>(idx[0]) / q_len;
            const int q = static_cast<int>(idx[0]) % q_len;
            float* row = scores_ptr +
                         (static_cast<size_t>(h) * q_len + q) * kv_len;
            const float* mrow = mask_ptr + static_cast<size_t>(q) * kv_len;

            float max_score = -1.0e30f;
            for (int t = 0; t < kv_len; ++t) {
                // torch.matmul(bf16, bf16) returns bf16 in the trusted
                // DiffusionGemma implementation. oneDNN accumulates into our
                // f32 destination, so explicitly apply the observable bf16
                // output rounding before the fp32 softmax.
                const float s = static_cast<float>(bf16(row[t])) + mrow[t];
                row[t] = s;
                max_score = std::max(max_score, s);
            }
            float sum = 0.0f;
            for (int t = 0; t < kv_len; ++t) {
                sum += std::exp(row[t] - max_score);
            }
            const float inv_sum = 1.0f / sum;
            bf16* prow = probs_ptr +
                         (static_cast<size_t>(h) * q_len + q) * kv_len;
            for (int t = 0; t < kv_len; ++t) {
                prow[t] = bf16(std::exp(row[t] - max_score) * inv_sum);
            }
        });
    AILA_LOG_DEBUG("[Attn] softmax submitted");

    dnnl::memory probs_mem = probs_bf16.make_dnnl_memory(prims.pv_src_md);
    dnnl::memory v_mem = expanded_v.make_dnnl_memory(prims.pv_w_md);
    dnnl::memory attn_mem = attn_tmp.make_dnnl_memory(prims.pv_dst_md);
    std::unordered_map<int, dnnl::memory> pv_args{
        {DNNL_ARG_SRC, probs_mem}, {DNNL_ARG_WEIGHTS, v_mem}, {DNNL_ARG_DST, attn_mem}};
    if (prims.pv_scratch_md.get_size()) {
        pv_args[DNNL_ARG_SCRATCHPAD] = ctx.scratchpad_memory(prims.pv_scratch_md);
    }
    prims.pv_prim.execute(ctx.stream(), pv_args);
    AILA_LOG_DEBUG("[Attn] pv executed");

    bf16* attn_ptr = static_cast<bf16*>(attn_tmp.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());
    const int q_dim = num_heads * head_dim;
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(num_heads) * q_len * head_dim),
        [=](sycl::id<1> idx) {
            const int total = num_heads * q_len * head_dim;
            const int d = static_cast<int>(idx[0]) % head_dim;
            const int q = (static_cast<int>(idx[0]) / head_dim) % q_len;
            const int h = static_cast<int>(idx[0]) / (head_dim * q_len);
            out_ptr[static_cast<size_t>(q) * q_dim +
                    static_cast<size_t>(h) * head_dim + d] =
                attn_ptr[(static_cast<size_t>(h) * q_len + q) * head_dim + d];
            (void)total;
        });
}

} // namespace ops
