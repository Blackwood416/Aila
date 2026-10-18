#include "Ops.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <sycl/sycl.hpp>


using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

namespace ops {

namespace {

// Fmt is a KvCacheFormat code: 0 bf16, 1 fp8 e4m3, 2 int8 (per-token scale).
// Keeping it a compile-time int preserves the exact codegen of the bf16 and fp8
// default paths; only the opt-in int8 instantiation pays for a scale lookup.
template <int Fmt>
struct KVCacheAccessor {
  const void *ptr;
  const float *scale; // [num_kv_heads, max_seq_len] f32, int8 only
  int max_seq_len;
  int head_dim;

  KVCacheAccessor(const void *p, const float *s, int msl, int hd)
      : ptr(p), scale(s), max_seq_len(msl), head_dim(hd) {}

  inline float load(int kv_head, int t, int d) const {
    int idx = kv_head * max_seq_len * head_dim + t * head_dim + d;
    if constexpr (Fmt == kKvFmtFp8) {
      return fp8_e4m3fn_to_float(static_cast<const uint8_t *>(ptr)[idx]);
    } else if constexpr (Fmt == kKvFmtInt8) {
      return int8_kv_to_float(static_cast<const int8_t *>(ptr)[idx],
                              scale[kv_head * max_seq_len + t]);
    } else {
      return static_cast<float>(static_cast<const bf16 *>(ptr)[idx]);
    }
  }
};

// Storage format of a KV cache tensor; also used by callers to pick the
// matching template instantiation.
inline int kv_cache_format_of(const Tensor &cache) {
  switch (cache.dtype()) {
    case dnnl::memory::data_type::f8_e4m3: return kKvFmtFp8;
    case dnnl::memory::data_type::s8: return kKvFmtInt8;
    default: return kKvFmtBf16;
  }
}

inline const float *kv_scale_ptr(const Tensor *scale) {
  return (scale && scale->valid()) ? scale->data_as<float>() : nullptr;
}

inline float *kv_scale_write_ptr(Tensor *scale) {
  return (scale && scale->valid()) ? scale->data_as<float>() : nullptr;
}

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
constexpr int kDecodeExact256QsaBlockTopk = 512;
constexpr int kDecodeExact256QsaMaxTiles =
    (kDecodeExact256QsaBlockTopk * 4 + 3 + kDecodeExact256Tile - 1) /
    kDecodeExact256Tile;
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

struct QsaDecodeGraphGeometry {
  int visible_len = 0;
  int selected_blocks = 0;
  int effective_len = 0;
  int actual_tiles = 0;
  bool valid = false;
};

// Every work-group derives the same geometry from the device control table.
// No host scalar selecting the grid or token count enters the graph.
inline QsaDecodeGraphGeometry qsa_decode_graph_geometry(
    const int* control, int max_tiles) {
  QsaDecodeGraphGeometry geometry;
  if (control == nullptr || control[graph_control::abort] != 0) {
    return geometry;
  }

  const int start_pos = control[graph_control::start_pos];
  const int compressed_blocks = control[graph_control::compressed_blocks];
  const long long visible_len_ll = static_cast<long long>(start_pos) + 1;
  if (start_pos < 0 || compressed_blocks < 0 || visible_len_ll <= 0 ||
      visible_len_ll > 0x7fffffffLL) {
    return geometry;
  }

  geometry.visible_len = static_cast<int>(visible_len_ll);
  geometry.selected_blocks =
      compressed_blocks < kDecodeExact256QsaBlockTopk
          ? compressed_blocks
          : kDecodeExact256QsaBlockTopk;
  const int tail = geometry.visible_len & 3;
  geometry.effective_len = geometry.selected_blocks * 4 + tail;
  geometry.actual_tiles =
      (geometry.effective_len + kDecodeExact256Tile - 1) /
      kDecodeExact256Tile;
  geometry.valid =
      geometry.actual_tiles > 0 && geometry.actual_tiles <= max_tiles;
  return geometry;
}

inline int qsa_decode_active_cache_token(
    const int* active_blocks, int selected_blocks, int t) {
  if (t < selected_blocks * 4) {
    return active_blocks[t / 4] * 4 + (t & 3);
  }
  return active_blocks[selected_blocks + (t - selected_blocks * 4)];
}

template <int Quantized>
void attention_decode_baseline_impl(Context &ctx, Tensor &q, Tensor &k_cache,
                                    Tensor &v_cache, Tensor &output,
                                    Tensor &scores_buf, int num_heads,
                                    int num_kv_heads, int head_dim, int cached_len,
                                    int tail_start, int sink_len, int effective_len,
                                     int wg_size,
                                     const bf16 *gate_ptr = nullptr,
                                     const int *active_indices = nullptr,
                                     const int *active_blocks = nullptr,
                                     int num_active_blocks = 0,
                                     const float *k_scale = nullptr,
                                     const float *v_scale = nullptr) {
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

          KVCacheAccessor<Quantized> k_acc(k_ptr, k_scale, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptr, v_scale, max_seq_len, head_dim);

          for (int d = lid; d < head_dim; d += wg_size) {
            q_cache[d] = static_cast<float>(q_ptr[head * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int t = lid; t < effective_len; t += wg_size) {
            int cache_t = 0;
            if (active_blocks != nullptr && num_active_blocks > 0) {
              if (t < num_active_blocks * 4) {
                cache_t = active_blocks[t / 4] * 4 + (t % 4);
              } else {
                cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
              }
            } else if (active_indices) {
              cache_t = active_indices[t];
            } else {
              cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
            }
            float sum = 0.0f;
            if constexpr (!Quantized) {
              using vec8 = sycl::vec<bf16, 8>;
              const bf16 *k_tok = static_cast<const bf16 *>(k_ptr) + kv_head * max_seq_len * head_dim + cache_t * head_dim;
              for (int d = 0; d < head_dim; d += 8) {
                const vec8 k_vec = *reinterpret_cast<const vec8 *>(k_tok + d);
                sum += q_cache[d + 0] * static_cast<float>(k_vec[0]);
                sum += q_cache[d + 1] * static_cast<float>(k_vec[1]);
                sum += q_cache[d + 2] * static_cast<float>(k_vec[2]);
                sum += q_cache[d + 3] * static_cast<float>(k_vec[3]);
                sum += q_cache[d + 4] * static_cast<float>(k_vec[4]);
                sum += q_cache[d + 5] * static_cast<float>(k_vec[5]);
                sum += q_cache[d + 6] * static_cast<float>(k_vec[6]);
                sum += q_cache[d + 7] * static_cast<float>(k_vec[7]);
              }
            } else {
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
              int cache_t = 0;
              if (active_blocks != nullptr && num_active_blocks > 0) {
                if (t < num_active_blocks * 4) {
                  cache_t = active_blocks[t / 4] * 4 + (t % 4);
                } else {
                  cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
                }
              } else if (active_indices) {
                cache_t = active_indices[t];
              } else {
                cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
              }
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
                               const Tensor *gate_buf = nullptr,
                               const int *active_indices = nullptr,
                               const int *active_blocks = nullptr,
                               int num_active_blocks = 0,
                               const Tensor *k_scale = nullptr,
                               const Tensor *v_scale = nullptr) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  const int fmt = kv_cache_format_of(k_cache);
  const float *k_scale_ptr = kv_scale_ptr(k_scale);
  const float *v_scale_ptr = kv_scale_ptr(v_scale);
  if (fmt == kKvFmtInt8 && (!k_scale_ptr || !v_scale_ptr)) {
    throw std::invalid_argument("attention_decode: int8 KV cache requires k_scale/v_scale");
  }
  auto run = [&](auto tag) {
    attention_decode_baseline_impl<decltype(tag)::value>(
        ctx, q, k_cache, v_cache, output, scores_buf, num_heads, num_kv_heads,
        head_dim, cached_len, tail_start, sink_len, effective_len, wg_size, gate_ptr, active_indices,
        active_blocks, num_active_blocks, k_scale_ptr, v_scale_ptr);
  };
  if (fmt == kKvFmtInt8) run(std::integral_constant<int, kKvFmtInt8>{});
  else if (fmt == kKvFmtFp8) run(std::integral_constant<int, kKvFmtFp8>{});
  else run(std::integral_constant<int, kKvFmtBf16>{});
}


// Cached read-once flag: the raw env read allocates, and SYCL kernels cannot
// capture function-local statics.
inline bool kv_int8_vector_reads() {
  // Default on inside the (already opt-in) int8 KV path; set to 0 to restore
  // the per-element accessor reads for an A/B.
  static const bool enabled = aila::env::read_flag("AILA_KV_QUANT_VEC", true);
  return enabled;
}

bool kv_int8_dpas_enabled(const sycl::device &dev) {
  static const bool requested = aila::env::read_flag("AILA_KV_QUANT_DPAS", false);
  if (!requested) return false;
  auto supports = [](const sycl::device &device) {
    if (!device.has(sycl::aspect::ext_intel_matrix)) return false;
    using sycl::ext::oneapi::experimental::info::device::matrix_combinations;
    for (const auto &c : device.get_info<matrix_combinations>()) {
      if ((c.msize == 0 || c.msize == 1) && c.nsize == 8 && c.ksize == 32 &&
          c.atype == matrix_type::sint8 && c.btype == matrix_type::sint8 &&
          c.ctype == matrix_type::sint32 && c.dtype == matrix_type::sint32)
        return true;
    }
    return false;
  };
  static thread_local sycl::device cached_device = dev;
  static thread_local bool supported = supports(dev);
  if (cached_device != dev) {
    cached_device = dev;
    supported = supports(dev);
  }
  return supported;
}

void attention_decode_int8_dpas_partials(
    Context &ctx, const bf16 *q, const int8_t *k, const int8_t *v,
    const float *ks, const float *vs, float *partials, int num_heads,
    int heads_per_kv, int max_seq_len, int max_tiles, int tail_start,
    int effective_len) {
  constexpr int dim = 256, tile = kDecodeExact256Tile, wg = kDecodeExact256TileWg;
  constexpr int sg_size = 8, bk = 32;
  const int num_tiles = (effective_len + tile - 1) / tile;
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> qc(dim, cgh);
    sycl::local_accessor<int8_t, 1> q8(dim, cgh), qr8(dim, cgh);
    sycl::local_accessor<int32_t, 1> dots(2 * tile, cgh);
    sycl::local_accessor<float, 1> logits(tile, cgh), stats(2, cgh);
    cgh.parallel_for(sycl::nd_range<1>(num_heads * num_tiles * wg, wg),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(8)]] {
          const int head = item.get_group(0) / num_tiles;
          const int tile_start = (item.get_group(0) % num_tiles) * tile;
          const int len = sycl::min(tile, effective_len - tile_start);
          const int lid = item.get_local_id(0), kv_head = head / heads_per_kv;
          const size_t cache_base = size_t(kv_head) * max_seq_len;
          float *partial = partials + (head * max_tiles + item.get_group(0) % num_tiles) *
              kDecodeExact256PartialStride;
          float amax = 0.0f;
          for (int d = lid; d < dim; d += wg) {
            qc[d] = float(q[head * dim + d]);
            amax = sycl::fmax(amax, sycl::fabs(qc[d]));
          }
          amax = sycl::reduce_over_group(item.get_group(), amax, sycl::maximum<float>());
          const float qs = amax / 127.0f, qi = qs > 0.0f ? 1.0f / qs : 0.0f;
          float rmax = 0.0f;
          for (int d = lid; d < dim; d += wg) {
            q8[d] = int8_t(sycl::rint(sycl::clamp(qc[d] * qi, -127.0f, 127.0f)));
            rmax = sycl::fmax(rmax, sycl::fabs(qc[d] - float(q8[d]) * qs));
          }
          rmax = sycl::reduce_over_group(item.get_group(), rmax, sycl::maximum<float>());
          const float rs = rmax / 127.0f, ri = rs > 0.0f ? 1.0f / rs : 0.0f;
          for (int d = lid; d < dim; d += wg) {
            qr8[d] = int8_t(sycl::rint(sycl::clamp((qc[d] - float(q8[d]) * qs) * ri,
                                                -127.0f, 127.0f)));
          }
          item.barrier(sycl::access::fence_space::local_space);
          auto sg = item.get_sub_group();
          const int t0 = int(sg.get_group_linear_id()) * sg_size;
          float score = -1e30f;
          // One SG8 owns one N8 chunk and a distinct result slice. A short
          // chunk uses scalar QK so matrix loads never cross the live KV end.
          if (t0 + sg_size <= len) {
            joint_matrix<sycl::sub_group, int8_t, use::a, 1, bk, layout::row_major> a;
            joint_matrix<sycl::sub_group, int8_t, use::b, bk, sg_size, layout::col_major> b;
            joint_matrix<sycl::sub_group, int32_t, use::accumulator, 1, sg_size> base, resid;
            joint_matrix_fill(sg, base, int32_t(0));
            joint_matrix_fill(sg, resid, int32_t(0));
            const int8_t *kt = k + (cache_base + tail_start + tile_start + t0) * dim;
            for (int d = 0; d < dim; d += bk) {
              joint_matrix_load(sg, b, sycl::address_space_cast<
                  sycl::access::address_space::global_space, sycl::access::decorated::no>(kt + d), dim);
              joint_matrix_load(sg, a, q8.get_multi_ptr<sycl::access::decorated::no>() + d, dim);
              joint_matrix_mad(sg, base, a, b, base);
              joint_matrix_load(sg, a, qr8.get_multi_ptr<sycl::access::decorated::no>() + d, dim);
              joint_matrix_mad(sg, resid, a, b, resid);
            }
            auto dp = dots.get_multi_ptr<sycl::access::decorated::no>();
            joint_matrix_store(sg, base, dp + t0, sg_size, layout::row_major);
            joint_matrix_store(sg, resid, dp + tile + t0, sg_size, layout::row_major);
            sycl::group_barrier(sg);
            score = (float(dots[lid]) * qs + float(dots[tile + lid]) * rs) *
                ks[cache_base + tail_start + tile_start + lid] * (1.0f / 16.0f);
          } else if (lid < len) {
            const size_t row = cache_base + tail_start + tile_start + lid;
            score = 0.0f;
            for (int d = 0; d < dim; ++d) score += qc[d] * (float(k[row * dim + d]) * ks[row]);
            score *= 1.0f / 16.0f;
          }
          logits[lid] = score;
          const float tile_max = sycl::reduce_over_group(item.get_group(), score, sycl::maximum<float>());
          if (lid == 0) stats[0] = tile_max;
          item.barrier(sycl::access::fence_space::local_space);
          const float p = lid < len ? sycl::native::exp(logits[lid] - stats[0]) : 0.0f;
          logits[lid] = p;
          const float tile_sum = sycl::reduce_over_group(item.get_group(), p, sycl::plus<float>());
          if (lid == 0) {
            partial[0] = tile_max;
            partial[1] = tile_sum;
          }
          item.barrier(sycl::access::fence_space::local_space);
          float lo = 0.0f, hi = 0.0f;
          for (int t = 0; t < len; ++t) {
            const size_t row = cache_base + tail_start + tile_start + t;
            const float s = vs[row], prob = logits[t];
            lo += prob * (float(v[row * dim + lid]) * s);
            hi += prob * (float(v[row * dim + lid + wg]) * s);
          }
          partial[kDecodeExact256PartialAccOffset + lid] = lo;
          partial[kDecodeExact256PartialAccOffset + lid + wg] = hi;
        });
  });
}

template <int Quantized, bool Dpas = false>
void attention_decode_exact_head256_partial_merge_impl(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len,
    const bf16 *gate_ptr = nullptr,
    const int *active_indices = nullptr,
    const int *active_blocks = nullptr,
    int num_active_blocks = 0,
    const float *k_scale = nullptr,
    const float *v_scale = nullptr) {
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

  // Opt-in int8 read optimisation: hoist the per-token scale out of the d-loop
  // and read the codes with 8-byte vector loads. The per-element product and
  // its accumulation order are unchanged, so results stay bit-identical to the
  // accessor path for the same cache contents.
  const bool kv_i8_vec = kv_int8_vector_reads();

  // Phase 1: compute per-tile exact softmax stats and exp-weighted V partials.
  if constexpr (Dpas) {
    attention_decode_int8_dpas_partials(ctx, q_ptr, static_cast<const int8_t *>(k_ptr),
        static_cast<const int8_t *>(v_ptr), k_scale, v_scale, partials_ptr,
        num_heads, heads_per_kv, max_seq_len, max_tiles, tail_start, effective_len);
  } else {
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

          KVCacheAccessor<Quantized> k_acc(k_ptr, k_scale, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptr, v_scale, max_seq_len, head_dim);

          for (int d = lid; d < head_dim; d += tile_wg) {
            q_cache[d] = static_cast<float>(q_ptr[head * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float score = -1e30f;
          if (lid < tile_len) {
            const int t = tile_start + lid;
            int cache_t = 0;
            if (active_blocks != nullptr && num_active_blocks > 0) {
              if (t < num_active_blocks * 4) {
                cache_t = active_blocks[t / 4] * 4 + (t % 4);
              } else {
                cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
              }
            } else if (active_indices) {
              cache_t = active_indices[t];
            } else {
              cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
            }
            score = 0.0f;
            if constexpr (!Quantized) {
              using vec8 = sycl::vec<bf16, 8>;
              const bf16 *k_tok = static_cast<const bf16 *>(k_ptr) + kv_head * max_seq_len * head_dim + cache_t * head_dim;
              for (int d = 0; d < head_dim; d += 8) {
                const vec8 k_vec = *reinterpret_cast<const vec8 *>(k_tok + d);
                score += q_cache[d + 0] * static_cast<float>(k_vec[0]);
                score += q_cache[d + 1] * static_cast<float>(k_vec[1]);
                score += q_cache[d + 2] * static_cast<float>(k_vec[2]);
                score += q_cache[d + 3] * static_cast<float>(k_vec[3]);
                score += q_cache[d + 4] * static_cast<float>(k_vec[4]);
                score += q_cache[d + 5] * static_cast<float>(k_vec[5]);
                score += q_cache[d + 6] * static_cast<float>(k_vec[6]);
                score += q_cache[d + 7] * static_cast<float>(k_vec[7]);
              }
            } else {
              if constexpr (Quantized == kKvFmtInt8) {
                if (kv_i8_vec) {
                  const int8_t *k_tok = static_cast<const int8_t *>(k_ptr) +
                      (static_cast<size_t>(kv_head) * max_seq_len + cache_t) *
                          head_dim;
                  const float k_row_scale =
                      k_scale[static_cast<size_t>(kv_head) * max_seq_len + cache_t];
                  for (int d = 0; d < head_dim; d += 8) {
                    const auto codes = *reinterpret_cast<const sycl::vec<int8_t, 8> *>(
                        k_tok + d);
                    score += q_cache[d + 0] * (static_cast<float>(codes[0]) * k_row_scale);
                    score += q_cache[d + 1] * (static_cast<float>(codes[1]) * k_row_scale);
                    score += q_cache[d + 2] * (static_cast<float>(codes[2]) * k_row_scale);
                    score += q_cache[d + 3] * (static_cast<float>(codes[3]) * k_row_scale);
                    score += q_cache[d + 4] * (static_cast<float>(codes[4]) * k_row_scale);
                    score += q_cache[d + 5] * (static_cast<float>(codes[5]) * k_row_scale);
                    score += q_cache[d + 6] * (static_cast<float>(codes[6]) * k_row_scale);
                    score += q_cache[d + 7] * (static_cast<float>(codes[7]) * k_row_scale);
                  }
                } else {
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
                }
              } else {
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
              }
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

          if constexpr (Quantized == kKvFmtInt8) {
            // Token-major variant: two d-accumulators per thread (head_dim ==
            // 2 * tile_wg), one scale load and one byte load per token. Term
            // and accumulation order match the accessor loop exactly.
            if (kv_i8_vec && head_dim == 2 * tile_wg) {
              const int8_t *v_head = static_cast<const int8_t *>(v_ptr) +
                  static_cast<size_t>(kv_head) * max_seq_len * head_dim;
              float acc_lo = 0.0f;
              float acc_hi = 0.0f;
              for (int i = 0; i < tile_len; ++i) {
                const int t = tile_start + i;
                int cache_t = 0;
                if (active_blocks != nullptr && num_active_blocks > 0) {
                  if (t < num_active_blocks * 4) {
                    cache_t = active_blocks[t / 4] * 4 + (t % 4);
                  } else {
                    cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
                  }
                } else if (active_indices) {
                  cache_t = active_indices[t];
                } else {
                  cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
                }
                const float p = logits[i];
                const float v_row_scale =
                    v_scale[static_cast<size_t>(kv_head) * max_seq_len + cache_t];
                const int8_t *v_row = v_head + static_cast<size_t>(cache_t) * head_dim;
                acc_lo += p * (static_cast<float>(v_row[lid]) * v_row_scale);
                acc_hi += p * (static_cast<float>(v_row[lid + tile_wg]) * v_row_scale);
              }
              partial[kDecodeExact256PartialAccOffset + lid] = acc_lo;
              partial[kDecodeExact256PartialAccOffset + lid + tile_wg] = acc_hi;
            } else {
              for (int d = lid; d < head_dim; d += tile_wg) {
                float acc = 0.0f;
                for (int i = 0; i < tile_len; ++i) {
                  const int t = tile_start + i;
                  int cache_t = 0;
                  if (active_blocks != nullptr && num_active_blocks > 0) {
                    if (t < num_active_blocks * 4) {
                      cache_t = active_blocks[t / 4] * 4 + (t % 4);
                    } else {
                      cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
                    }
                  } else if (active_indices) {
                    cache_t = active_indices[t];
                  } else {
                    cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
                  }
                  acc += logits[i] * v_acc.load(kv_head, cache_t, d);
                }
                partial[kDecodeExact256PartialAccOffset + d] = acc;
              }
            }
          } else {
            for (int d = lid; d < head_dim; d += tile_wg) {
              float acc = 0.0f;
              for (int i = 0; i < tile_len; ++i) {
                const int t = tile_start + i;
                int cache_t = 0;
                if (active_blocks != nullptr && num_active_blocks > 0) {
                  if (t < num_active_blocks * 4) {
                    cache_t = active_blocks[t / 4] * 4 + (t % 4);
                  } else {
                    cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
                  }
                } else if (active_indices) {
                  cache_t = active_indices[t];
                } else {
                  cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
                }
                acc += logits[i] * v_acc.load(kv_head, cache_t, d);
              }
              partial[kDecodeExact256PartialAccOffset + d] = acc;
            }
          }
        });
  });

  }

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

// AILA_ATTN_KVREUSE="pair2" selects the head-pair K/V reuse phase 1 below;
// anything else (unset, "0", unknown spellings) keeps the unchanged default
// path. Read once per process, so the variant is fixed for the process
// lifetime and an A/B comparison runs one process per value. The switch alone
// is not enough to dispatch the variant: the caller also applies the
// attn_kv_reuse_min_len() length gate below.
inline bool attn_kv_reuse_pair2() {
  static const bool pair2 =
      aila::env::read_string("AILA_ATTN_KVREUSE") == "pair2";
  return pair2;
}

// Length gate for the pair2 path, in attended tokens, read once per process
// (AILA_ATTN_KVREUSE_MIN_LEN, default 2048). The pair2 bench measured -16.0%
// at 4096 and -25.1% at 8192 but +8.0% at 1834 (reproduced, gate
// "reject_1834_regression_le2pct" FAIL in receipt-k5-pair2.json), so shorter
// contexts keep the ungrouped kernel. The gate lives at the attention_decode
// call site, which is the frame that knows cached_len; it does not change any
// arithmetic of either path, only which phase-1 kernel is submitted.
inline int attn_kv_reuse_min_len() {
  static const int min_len =
      read_env_int_local("AILA_ATTN_KVREUSE_MIN_LEN", 2048);
  return min_len;
}

// D256 decode attention with a head group of G=2: one work-group owns a token
// tile for two adjacent q-heads that share a kv-head, so a K row is loaded
// once per token and each V element once per (tile, d) for both heads. The
// grid becomes (num_heads/2)*num_tiles groups of 128 threads.
//
// Everything else is kept verbatim from
// attention_decode_exact_head256_partial_merge_impl: the per-lane token
// mapping, the per-head expression and accumulation order, both
// reduce_over_group trees over the same 128 work items, the exp/max round trip
// through SLM, and the [head][tile][stride] partials layout. At the moment a
// head's reduction runs, every lane holds the same value it holds in the G=1
// kernel for that head and lane index, so each per-tile statistic is bitwise
// equal and the merged output is bitwise equal too.
//
// SLM: q_cache 2*256 f32 + logits 2*128 f32 + 2 f32 stats = 3.1 KB (G=1:
// 2.0 KB), well inside the 128 KB per-Xe-core pool; the private register delta
// is one float (score_b / acc_b) plus the live vec8; the int8 V branch trades
// that for four accumulators (acc_{a,b}_{lo,hi}) plus a hoisted row scale.
//
// Formats: bf16 is dispatched everywhere the variant is eligible. int8 is
// dispatched wherever the default classifier is not the DPAS kernel - its K
// read uses the same vec8 int8 loads and per-token row scale as the G=1
// kv_i8_vec branch and its V read the same token-major form, with the
// dequantized element shared by both heads, so both are bit-identical to the
// G=1 int8 kernel (which is itself documented as bit-identical between its vec
// and accessor forms). fp8 stays on the default path: it has no vec form and
// no campaign behind it.
template <int Quantized>
void attention_decode_exact_head256_pair2_impl(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len,
    const bf16 *gate_ptr = nullptr,
    const int *active_indices = nullptr,
    const int *active_blocks = nullptr,
    int num_active_blocks = 0,
    const float *k_scale = nullptr,
    const float *v_scale = nullptr) {
  bf16 *q_ptr = static_cast<bf16 *>(q.data());
  const void *k_ptr = k_cache.data();
  const void *v_ptr = v_cache.data();
  bf16 *o_ptr = static_cast<bf16 *>(output.data());
  float *partials_ptr = static_cast<float *>(partials_buf.data());

  constexpr int head_dim = 256;
  constexpr int tile_t = kDecodeExact256Tile;
  constexpr int tile_wg = kDecodeExact256TileWg;
  constexpr int merge_wg = kDecodeExact256MergeWg;
  constexpr int pair = 2;

  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int max_seq_len = static_cast<int>(k_cache.shape(1));
  const int num_tiles = (effective_len + tile_t - 1) / tile_t;
  const int max_tiles = static_cast<int>(
      partials_buf.shape(partials_buf.ndim() == 4 ? 2 : 1));
  const int num_pairs = num_heads / pair;

  // Phase 1 (pair2): per-tile exact softmax stats and exp-weighted V partials
  // for both heads of the pair, from one K/V read per element.
  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(pair * head_dim), cgh);
    sycl::local_accessor<float, 1> logits(sycl::range<1>(pair * tile_t), cgh);
    sycl::local_accessor<float, 1> stats(sycl::range<1>(2), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(num_pairs) * num_tiles * tile_wg,
                          tile_wg),
        [=](sycl::nd_item<1> item) {
          const int group = item.get_group(0);
          const int head_pair = group / num_tiles;
          const int tile_idx = group % num_tiles;
          const int lid = item.get_local_id(0);
          const int head_a = head_pair * pair;
          const int head_b = head_a + 1;
          // heads_per_kv is even on this path (dispatcher guard), so the two
          // heads of a pair always resolve to the same kv-head.
          const int kv_head = head_a / heads_per_kv;
          const int tile_start = tile_idx * tile_t;
          const int tile_len = sycl::min(tile_t, effective_len - tile_start);
          float *partial_a = partials_ptr +
              (head_a * max_tiles + tile_idx) * kDecodeExact256PartialStride;
          float *partial_b = partials_ptr +
              (head_b * max_tiles + tile_idx) * kDecodeExact256PartialStride;

          KVCacheAccessor<Quantized> k_acc(k_ptr, k_scale, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptr, v_scale, max_seq_len, head_dim);

          for (int d = lid; d < head_dim; d += tile_wg) {
            q_cache[d] = static_cast<float>(q_ptr[head_a * head_dim + d]);
            q_cache[head_dim + d] = static_cast<float>(q_ptr[head_b * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float score_a = -1e30f;
          float score_b = -1e30f;
          if (lid < tile_len) {
            const int t = tile_start + lid;
            int cache_t = 0;
            if (active_blocks != nullptr && num_active_blocks > 0) {
              if (t < num_active_blocks * 4) {
                cache_t = active_blocks[t / 4] * 4 + (t % 4);
              } else {
                cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
              }
            } else if (active_indices) {
              cache_t = active_indices[t];
            } else {
              cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
            }
            score_a = 0.0f;
            score_b = 0.0f;
            if constexpr (!Quantized) {
              using vec8 = sycl::vec<bf16, 8>;
              const bf16 *k_tok = static_cast<const bf16 *>(k_ptr) + kv_head * max_seq_len * head_dim + cache_t * head_dim;
              for (int d = 0; d < head_dim; d += 8) {
                const vec8 k_vec = *reinterpret_cast<const vec8 *>(k_tok + d);
                score_a += q_cache[d + 0] * static_cast<float>(k_vec[0]);
                score_b += q_cache[head_dim + d + 0] * static_cast<float>(k_vec[0]);
                score_a += q_cache[d + 1] * static_cast<float>(k_vec[1]);
                score_b += q_cache[head_dim + d + 1] * static_cast<float>(k_vec[1]);
                score_a += q_cache[d + 2] * static_cast<float>(k_vec[2]);
                score_b += q_cache[head_dim + d + 2] * static_cast<float>(k_vec[2]);
                score_a += q_cache[d + 3] * static_cast<float>(k_vec[3]);
                score_b += q_cache[head_dim + d + 3] * static_cast<float>(k_vec[3]);
                score_a += q_cache[d + 4] * static_cast<float>(k_vec[4]);
                score_b += q_cache[head_dim + d + 4] * static_cast<float>(k_vec[4]);
                score_a += q_cache[d + 5] * static_cast<float>(k_vec[5]);
                score_b += q_cache[head_dim + d + 5] * static_cast<float>(k_vec[5]);
                score_a += q_cache[d + 6] * static_cast<float>(k_vec[6]);
                score_b += q_cache[head_dim + d + 6] * static_cast<float>(k_vec[6]);
                score_a += q_cache[d + 7] * static_cast<float>(k_vec[7]);
                score_b += q_cache[head_dim + d + 7] * static_cast<float>(k_vec[7]);
              }
            } else if constexpr (Quantized == kKvFmtInt8) {
              // int8 vec-read form, mirroring the G=1 kernel's kv_i8_vec
              // branch: hoist the per-token row scale, read 8 codes per load,
              // dequantize each code once and feed both heads. The per-element
              // term q_cache[...] * (float(code) * row_scale) and its
              // d-ascending accumulation order are identical to the G=1
              // kernel's vec form and to its accessor form, so the scores are
              // bit-identical to the G=1 int8 kernel.
              const int8_t *k_tok = static_cast<const int8_t *>(k_ptr) +
                  (static_cast<size_t>(kv_head) * max_seq_len + cache_t) * head_dim;
              const float k_row_scale =
                  k_scale[static_cast<size_t>(kv_head) * max_seq_len + cache_t];
              for (int d = 0; d < head_dim; d += 8) {
                const auto codes =
                    *reinterpret_cast<const sycl::vec<int8_t, 8> *>(k_tok + d);
                const float kv0 = static_cast<float>(codes[0]) * k_row_scale;
                score_a += q_cache[d + 0] * kv0;
                score_b += q_cache[head_dim + d + 0] * kv0;
                const float kv1 = static_cast<float>(codes[1]) * k_row_scale;
                score_a += q_cache[d + 1] * kv1;
                score_b += q_cache[head_dim + d + 1] * kv1;
                const float kv2 = static_cast<float>(codes[2]) * k_row_scale;
                score_a += q_cache[d + 2] * kv2;
                score_b += q_cache[head_dim + d + 2] * kv2;
                const float kv3 = static_cast<float>(codes[3]) * k_row_scale;
                score_a += q_cache[d + 3] * kv3;
                score_b += q_cache[head_dim + d + 3] * kv3;
                const float kv4 = static_cast<float>(codes[4]) * k_row_scale;
                score_a += q_cache[d + 4] * kv4;
                score_b += q_cache[head_dim + d + 4] * kv4;
                const float kv5 = static_cast<float>(codes[5]) * k_row_scale;
                score_a += q_cache[d + 5] * kv5;
                score_b += q_cache[head_dim + d + 5] * kv5;
                const float kv6 = static_cast<float>(codes[6]) * k_row_scale;
                score_a += q_cache[d + 6] * kv6;
                score_b += q_cache[head_dim + d + 6] * kv6;
                const float kv7 = static_cast<float>(codes[7]) * k_row_scale;
                score_a += q_cache[d + 7] * kv7;
                score_b += q_cache[head_dim + d + 7] * kv7;
              }
            } else {
              for (int d = 0; d < head_dim; d += 8) {
                // One dequantized element per position, consumed by both heads;
                // the accessor value is itself a pure function of the bytes.
                const float kv0 = k_acc.load(kv_head, cache_t, d + 0);
                score_a += q_cache[d + 0] * kv0;
                score_b += q_cache[head_dim + d + 0] * kv0;
                const float kv1 = k_acc.load(kv_head, cache_t, d + 1);
                score_a += q_cache[d + 1] * kv1;
                score_b += q_cache[head_dim + d + 1] * kv1;
                const float kv2 = k_acc.load(kv_head, cache_t, d + 2);
                score_a += q_cache[d + 2] * kv2;
                score_b += q_cache[head_dim + d + 2] * kv2;
                const float kv3 = k_acc.load(kv_head, cache_t, d + 3);
                score_a += q_cache[d + 3] * kv3;
                score_b += q_cache[head_dim + d + 3] * kv3;
                const float kv4 = k_acc.load(kv_head, cache_t, d + 4);
                score_a += q_cache[d + 4] * kv4;
                score_b += q_cache[head_dim + d + 4] * kv4;
                const float kv5 = k_acc.load(kv_head, cache_t, d + 5);
                score_a += q_cache[d + 5] * kv5;
                score_b += q_cache[head_dim + d + 5] * kv5;
                const float kv6 = k_acc.load(kv_head, cache_t, d + 6);
                score_a += q_cache[d + 6] * kv6;
                score_b += q_cache[head_dim + d + 6] * kv6;
                const float kv7 = k_acc.load(kv_head, cache_t, d + 7);
                score_a += q_cache[d + 7] * kv7;
                score_b += q_cache[head_dim + d + 7] * kv7;
              }
            }
            score_a *= scale;
            score_b *= scale;
          }

          // Head a, exact G=1 softmax order: stage, group max, exp, group sum.
          logits[lid] = score_a;
          const float tile_max_a = sycl::reduce_over_group(
              item.get_group(), score_a, sycl::maximum<float>());
          if (lid == 0) {
            stats[0] = tile_max_a;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float exp_a = 0.0f;
          if (lid < tile_len) {
            exp_a = sycl::native::exp(logits[lid] - stats[0]);
            logits[lid] = exp_a;
          } else {
            logits[lid] = 0.0f;
          }
          const float tile_sum_a = sycl::reduce_over_group(
              item.get_group(), exp_a, sycl::plus<float>());
          if (lid == 0) {
            stats[1] = tile_sum_a;
            partial_a[0] = stats[0];
            partial_a[1] = stats[1];
          }
          item.barrier(sycl::access::fence_space::local_space);

          // Head b: same sequence, same lane-to-token mapping, only the SLM
          // slot differs.
          logits[tile_t + lid] = score_b;
          const float tile_max_b = sycl::reduce_over_group(
              item.get_group(), score_b, sycl::maximum<float>());
          if (lid == 0) {
            stats[0] = tile_max_b;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float exp_b = 0.0f;
          if (lid < tile_len) {
            exp_b = sycl::native::exp(logits[tile_t + lid] - stats[0]);
            logits[tile_t + lid] = exp_b;
          } else {
            logits[tile_t + lid] = 0.0f;
          }
          const float tile_sum_b = sycl::reduce_over_group(
              item.get_group(), exp_b, sycl::plus<float>());
          if (lid == 0) {
            stats[1] = tile_sum_b;
            partial_b[0] = stats[0];
            partial_b[1] = stats[1];
          }
          item.barrier(sycl::access::fence_space::local_space);

          if constexpr (Quantized == kKvFmtInt8 && head_dim == 2 * tile_wg) {
            // Token-major int8 form, mirroring the G=1 kv_i8_vec V branch: one
            // row-scale load per token, one byte load per (token, d), and the
            // dequantized value feeds both heads. The per-head term
            // logits[i] * (float(code) * row_scale) and its i-ascending
            // accumulation are identical to the G=1 kernel's vec and accessor
            // forms, so the partials stay bit-identical.
            const int8_t *v_head = static_cast<const int8_t *>(v_ptr) +
                static_cast<size_t>(kv_head) * max_seq_len * head_dim;
            float acc_a_lo = 0.0f, acc_a_hi = 0.0f;
            float acc_b_lo = 0.0f, acc_b_hi = 0.0f;
            for (int i = 0; i < tile_len; ++i) {
              const int t = tile_start + i;
              int cache_t = 0;
              if (active_blocks != nullptr && num_active_blocks > 0) {
                if (t < num_active_blocks * 4) {
                  cache_t = active_blocks[t / 4] * 4 + (t % 4);
                } else {
                  cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
                }
              } else if (active_indices) {
                cache_t = active_indices[t];
              } else {
                cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
              }
              const float p_a = logits[i];
              const float p_b = logits[tile_t + i];
              const float v_row_scale =
                  v_scale[static_cast<size_t>(kv_head) * max_seq_len + cache_t];
              const int8_t *v_row = v_head + static_cast<size_t>(cache_t) * head_dim;
              acc_a_lo += p_a * (static_cast<float>(v_row[lid]) * v_row_scale);
              acc_a_hi += p_a * (static_cast<float>(v_row[lid + tile_wg]) * v_row_scale);
              acc_b_lo += p_b * (static_cast<float>(v_row[lid]) * v_row_scale);
              acc_b_hi += p_b * (static_cast<float>(v_row[lid + tile_wg]) * v_row_scale);
            }
            partial_a[kDecodeExact256PartialAccOffset + lid] = acc_a_lo;
            partial_a[kDecodeExact256PartialAccOffset + lid + tile_wg] = acc_a_hi;
            partial_b[kDecodeExact256PartialAccOffset + lid] = acc_b_lo;
            partial_b[kDecodeExact256PartialAccOffset + lid + tile_wg] = acc_b_hi;
          } else {
            for (int d = lid; d < head_dim; d += tile_wg) {
              float acc_a = 0.0f;
              float acc_b = 0.0f;
              for (int i = 0; i < tile_len; ++i) {
                const int t = tile_start + i;
                int cache_t = 0;
                if (active_blocks != nullptr && num_active_blocks > 0) {
                  if (t < num_active_blocks * 4) {
                    cache_t = active_blocks[t / 4] * 4 + (t % 4);
                  } else {
                    cache_t = active_blocks[num_active_blocks + (t - num_active_blocks * 4)];
                  }
                } else if (active_indices) {
                  cache_t = active_indices[t];
                } else {
                  cache_t = ((t < sink_len) ? t : (tail_start + (t - sink_len)));
                }
                const float v_val = v_acc.load(kv_head, cache_t, d);
                acc_a += logits[i] * v_val;
                acc_b += logits[tile_t + i] * v_val;
              }
              partial_a[kDecodeExact256PartialAccOffset + d] = acc_a;
              partial_b[kDecodeExact256PartialAccOffset + d] = acc_b;
            }
          }
        });
  });

  // Phase 2: byte-for-byte the merge phase of the G=1 kernel, over the same
  // [head][tile][stride] partials.
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
    const Tensor *gate_buf = nullptr,
    const int *active_indices = nullptr,
    const int *active_blocks = nullptr,
    int num_active_blocks = 0,
    const Tensor *k_scale = nullptr,
    const Tensor *v_scale = nullptr,
    bool kv_reuse_len_ok = false) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  const int fmt = kv_cache_format_of(k_cache);
  const float *k_scale_ptr = kv_scale_ptr(k_scale);
  const float *v_scale_ptr = kv_scale_ptr(v_scale);
  if (fmt == kKvFmtInt8 && (!k_scale_ptr || !v_scale_ptr)) {
    throw std::invalid_argument("attention_decode(exact256): int8 KV cache requires k_scale/v_scale");
  }
  // The int8 DPAS partials kernel is only the default classifier when the shape
  // allows it (no routing, no window/sink) and the env asks for it; the pair2
  // branch must not shadow it, because DPAS changes the accumulation grouping.
  const bool int8_dpas_path =
      fmt == kKvFmtInt8 && !active_indices && !active_blocks && sink_len == 0 &&
      kv_int8_dpas_enabled(ctx.queue().get_device());
  // AILA_ATTN_KVREUSE=pair2 opt-in head-pair K/V reuse: bf16, and int8 wherever
  // the default classifier is not the DPAS kernel. A head count that cannot be
  // paired inside one kv-head keeps the default path and reports that once.
  // kv_reuse_len_ok carries the length gate, which the caller decides from the
  // attended length (effective_len) it derived from cached_len.
  if (attn_kv_reuse_pair2()) {
    const bool fmt_ok =
        fmt == kKvFmtBf16 || (fmt == kKvFmtInt8 && !int8_dpas_path);
    const bool heads_ok = fmt_ok && num_kv_heads > 0 &&
        num_heads % 2 == 0 && (num_heads / num_kv_heads) % 2 == 0;
    const bool eligible = kv_reuse_len_ok && heads_ok;
    if (eligible) {
      static bool kvreuse_active_logged = false;
      if (!kvreuse_active_logged) {
        kvreuse_active_logged = true;
        AILA_LOG_INFO("[ATTN] kvreuse=pair2 head_dim=256 fmt=%d heads=%d kv_heads=%d effective_len=%d min_len=%d groups=%d",
                      fmt, num_heads, num_kv_heads,
                      effective_len, attn_kv_reuse_min_len(),
                      (num_heads / 2) *
                          ((effective_len + kDecodeExact256Tile - 1) /
                           kDecodeExact256Tile));
      }
      if (fmt == kKvFmtInt8) {
        attention_decode_exact_head256_pair2_impl<kKvFmtInt8>(
            ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
            tail_start, sink_len, effective_len, gate_ptr, active_indices,
            active_blocks, num_active_blocks, k_scale_ptr, v_scale_ptr);
      } else {
        attention_decode_exact_head256_pair2_impl<kKvFmtBf16>(
            ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
            tail_start, sink_len, effective_len, gate_ptr, active_indices,
            active_blocks, num_active_blocks, k_scale_ptr, v_scale_ptr);
      }
      return;
    }
    // First skip of the process, so an arm can tell "gated off" from "not
    // eligible" without reading the profile JSON.
    static bool kvreuse_skip_logged = false;
    if (!kvreuse_skip_logged) {
      kvreuse_skip_logged = true;
      if (!kv_reuse_len_ok) {
        AILA_LOG_INFO("[ATTN] kvreuse=pair2 skipped len_gate head_dim=256 fmt=%d effective_len=%d min_len=%d",
                      fmt, effective_len, attn_kv_reuse_min_len());
      } else {
        AILA_LOG_INFO("[ATTN] kvreuse=pair2 ignored fmt=%d heads=%d kv_heads=%d head_dim=256",
                      fmt, num_heads, num_kv_heads);
      }
    }
  }
  auto run = [&](auto tag) {
    attention_decode_exact_head256_partial_merge_impl<decltype(tag)::value>(
        ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
        tail_start, sink_len, effective_len, gate_ptr, active_indices,
        active_blocks, num_active_blocks, k_scale_ptr, v_scale_ptr);
  };
  if (int8_dpas_path) {
    attention_decode_exact_head256_partial_merge_impl<kKvFmtInt8, true>(
        ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
        tail_start, sink_len, effective_len, gate_ptr, active_indices,
        active_blocks, num_active_blocks, k_scale_ptr, v_scale_ptr);
  } else if (fmt == kKvFmtInt8) run(std::integral_constant<int, kKvFmtInt8>{});
  else if (fmt == kKvFmtFp8) run(std::integral_constant<int, kKvFmtFp8>{});
  else run(std::integral_constant<int, kKvFmtBf16>{});
}

template <int MaxTiles>
void attention_decode_exact_head256_block_native_graph_impl(
    Context &ctx, const Tensor &q, const Tensor &k_cache,
    const Tensor &v_cache, Tensor &output, Tensor &partials_buf,
    const Tensor &active_blocks, GraphControl control, int num_heads,
    int num_kv_heads, const Tensor *gate_buf) {
  static_assert(MaxTiles >= kDecodeExact256QsaMaxTiles,
                "QSA block-native graph MaxTiles must cover the 512-block bound");

  constexpr int head_dim = 256;
  constexpr int tile_t = kDecodeExact256Tile;
  constexpr int tile_wg = kDecodeExact256TileWg;
  constexpr int merge_wg = kDecodeExact256MergeWg;

  if (control == nullptr) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: null control");
  }
  if (num_heads <= 0 || num_kv_heads <= 0 ||
      num_heads % num_kv_heads != 0) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: invalid head geometry");
  }

  const int q_dim = num_heads * head_dim;
  if (!q.valid() || q.dtype() != dnnl::memory::data_type::bf16 ||
      q.numel() < q_dim) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: q must be BF16 with at least num_heads*256 elements");
  }
  if (!output.valid() || output.dtype() != dnnl::memory::data_type::bf16 ||
      output.numel() < q_dim) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: output must be BF16 with at least num_heads*256 elements");
  }
  if (!k_cache.valid() || !v_cache.valid() ||
      k_cache.dtype() != dnnl::memory::data_type::bf16 ||
      v_cache.dtype() != dnnl::memory::data_type::bf16) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: BF16 KV cache required");
  }
  if (k_cache.ndim() != 3 || v_cache.ndim() != 3 ||
      k_cache.shape(0) != num_kv_heads ||
      v_cache.shape(0) != num_kv_heads ||
      k_cache.shape(2) != head_dim ||
      v_cache.shape(2) != head_dim ||
      k_cache.shape(1) != v_cache.shape(1) || k_cache.shape(1) <= 0 ||
      k_cache.shape(1) > 0x7fffffffLL) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: unsupported KV layout");
  }
  if (!partials_buf.valid() ||
      partials_buf.dtype() != dnnl::memory::data_type::f32 ||
      partials_buf.ndim() != 3 ||
      partials_buf.shape(0) != num_heads ||
      partials_buf.shape(1) < MaxTiles ||
      partials_buf.shape(2) < kDecodeExact256PartialStride) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: partials must be f32 [num_heads, >=MaxTiles, 272]");
  }
  if (!active_blocks.valid() ||
      active_blocks.dtype() != dnnl::memory::data_type::s32 ||
      active_blocks.ndim() != 1 ||
      active_blocks.shape(0) <
          kDecodeExact256QsaBlockTopk + 4) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: active_blocks must hold 512 ids plus 4 tail ids");
  }
  if (gate_buf != nullptr &&
      (!gate_buf->valid() ||
       gate_buf->dtype() != dnnl::memory::data_type::bf16 ||
       gate_buf->numel() < q_dim)) {
    throw std::invalid_argument(
        "attention_decode_exact_head256_block_native_graph: gate must be BF16 with at least num_heads*256 elements");
  }

  const bf16 *q_ptr = q.data_as<bf16>();
  const bf16 *k_ptr = k_cache.data_as<bf16>();
  const bf16 *v_ptr = v_cache.data_as<bf16>();
  bf16 *o_ptr = output.data_as<bf16>();
  float *partials_ptr = partials_buf.data_as<float>();
  const int *active_blocks_ptr = active_blocks.data_as<int>();
  int *const control_ptr = control;
  const bf16 *gate_ptr =
      gate_buf != nullptr ? gate_buf->data_as<bf16>() : nullptr;

  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int max_seq_len = static_cast<int>(k_cache.shape(1));
  const int partial_tile_capacity =
      static_cast<int>(partials_buf.shape(1));

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);
    sycl::local_accessor<float, 1> logits(sycl::range<1>(tile_t), cgh);
    sycl::local_accessor<float, 1> stats(sycl::range<1>(2), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            static_cast<size_t>(num_heads) * MaxTiles * tile_wg, tile_wg),
        [=](sycl::nd_item<1> item) {
          const QsaDecodeGraphGeometry geometry =
              qsa_decode_graph_geometry(control_ptr, MaxTiles);
          if (!geometry.valid) {
            return;
          }

          const int group = item.get_group(0);
          const int head = group / MaxTiles;
          const int tile_idx = group % MaxTiles;
          if (tile_idx >= geometry.actual_tiles) {
            return;
          }

          const int lid = item.get_local_id(0);
          const int kv_head = head / heads_per_kv;
          const int tile_start = tile_idx * tile_t;
          const int tile_len =
              sycl::min(tile_t, geometry.effective_len - tile_start);
          float *partial =
              partials_ptr +
              (head * partial_tile_capacity + tile_idx) *
                  kDecodeExact256PartialStride;

          for (int d = lid; d < head_dim; d += tile_wg) {
            q_cache[d] = static_cast<float>(q_ptr[head * head_dim + d]);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float score = -1e30f;
          if (lid < tile_len) {
            const int t = tile_start + lid;
            const int cache_t = qsa_decode_active_cache_token(
                active_blocks_ptr, geometry.selected_blocks, t);
            using vec8 = sycl::vec<bf16, 8>;
            const bf16 *k_tok =
                k_ptr +
                (static_cast<size_t>(kv_head) * max_seq_len + cache_t) *
                    head_dim;
            score = 0.0f;
            for (int d = 0; d < head_dim; d += 8) {
              const vec8 k_vec =
                  *reinterpret_cast<const vec8 *>(k_tok + d);
              score += q_cache[d + 0] * static_cast<float>(k_vec[0]);
              score += q_cache[d + 1] * static_cast<float>(k_vec[1]);
              score += q_cache[d + 2] * static_cast<float>(k_vec[2]);
              score += q_cache[d + 3] * static_cast<float>(k_vec[3]);
              score += q_cache[d + 4] * static_cast<float>(k_vec[4]);
              score += q_cache[d + 5] * static_cast<float>(k_vec[5]);
              score += q_cache[d + 6] * static_cast<float>(k_vec[6]);
              score += q_cache[d + 7] * static_cast<float>(k_vec[7]);
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
              const int cache_t = qsa_decode_active_cache_token(
                  active_blocks_ptr, geometry.selected_blocks, t);
              acc += logits[i] *
                     static_cast<float>(
                         v_ptr[(static_cast<size_t>(kv_head) *
                                    max_seq_len +
                                cache_t) *
                                   head_dim +
                               d]);
            }
            partial[kDecodeExact256PartialAccOffset + d] = acc;
          }
        });
  });

  ctx.queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> tile_factors(
        sycl::range<1>(MaxTiles), cgh);
    sycl::local_accessor<float, 1> norm_sum(sycl::range<1>(1), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(
            static_cast<size_t>(num_heads) * merge_wg, merge_wg),
        [=](sycl::nd_item<1> item) {
          const QsaDecodeGraphGeometry geometry =
              qsa_decode_graph_geometry(control_ptr, MaxTiles);
          if (!geometry.valid) {
            return;
          }

          const int head = item.get_group(0);
          const int lid = item.get_local_id(0);
          if (lid == 0) {
            float global_max = -1e30f;
            for (int tile_idx = 0; tile_idx < geometry.actual_tiles;
                 ++tile_idx) {
              const float *partial =
                  partials_ptr +
                  (head * partial_tile_capacity + tile_idx) *
                      kDecodeExact256PartialStride;
              global_max = sycl::fmax(global_max, partial[0]);
            }
            float sum = 0.0f;
            for (int tile_idx = 0; tile_idx < geometry.actual_tiles;
                 ++tile_idx) {
              const float *partial =
                  partials_ptr +
                  (head * partial_tile_capacity + tile_idx) *
                      kDecodeExact256PartialStride;
              const float factor =
                  sycl::native::exp(partial[0] - global_max);
              tile_factors[tile_idx] = factor;
              sum += factor * partial[1];
            }
            norm_sum[0] = sum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          const float inv_sum = 1.0f / norm_sum[0];
          for (int d = lid; d < head_dim; d += merge_wg) {
            float acc = 0.0f;
            for (int tile_idx = 0; tile_idx < geometry.actual_tiles;
                 ++tile_idx) {
              const float *partial =
                  partials_ptr +
                  (head * partial_tile_capacity + tile_idx) *
                      kDecodeExact256PartialStride;
              acc += tile_factors[tile_idx] *
                     partial[kDecodeExact256PartialAccOffset + d];
            }
            float val = static_cast<float>(bf16(acc * inv_sum));
            if (gate_ptr != nullptr) {
              const float g_v =
                  static_cast<float>(gate_ptr[head * head_dim + d]);
              const float s = 1.0f / (1.0f + sycl::native::exp(-g_v));
              val = static_cast<float>(bf16(val * s));
            }
            o_ptr[head * head_dim + d] = bf16(val);
          }
        });
  });
}

template void attention_decode_exact_head256_block_native_graph_impl<
    kDecodeExact256QsaMaxTiles>(
    Context &ctx, const Tensor &q, const Tensor &k_cache,
    const Tensor &v_cache, Tensor &output, Tensor &partials_buf,
    const Tensor &active_blocks, GraphControl control, int num_heads,
    int num_kv_heads, const Tensor *gate_buf);


template <int Quantized>
void attention_decode_exact_head128_partial_merge_impl(
    Context &ctx, Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &output,
    Tensor &partials_buf, int num_heads, int num_kv_heads, int tail_start,
    int sink_len, int effective_len,
    const bf16 *gate_ptr = nullptr,
    const float *k_scale = nullptr,
    const float *v_scale = nullptr) {
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

          KVCacheAccessor<Quantized> k_acc(k_ptr, k_scale, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptr, v_scale, max_seq_len, head_dim);

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
    const Tensor *gate_buf = nullptr,
    const Tensor *k_scale = nullptr,
    const Tensor *v_scale = nullptr) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  const int fmt = kv_cache_format_of(k_cache);
  const float *k_scale_ptr = kv_scale_ptr(k_scale);
  const float *v_scale_ptr = kv_scale_ptr(v_scale);
  if (fmt == kKvFmtInt8 && (!k_scale_ptr || !v_scale_ptr)) {
    throw std::invalid_argument("attention_decode(exact128): int8 KV cache requires k_scale/v_scale");
  }
  auto run = [&](auto tag) {
    attention_decode_exact_head128_partial_merge_impl<decltype(tag)::value>(
        ctx, q, k_cache, v_cache, output, partials_buf, num_heads, num_kv_heads,
        tail_start, sink_len, effective_len, gate_ptr, k_scale_ptr, v_scale_ptr);
  };
  if (fmt == kKvFmtInt8) run(std::integral_constant<int, kKvFmtInt8>{});
  else if (fmt == kKvFmtFp8) run(std::integral_constant<int, kKvFmtFp8>{});
  else run(std::integral_constant<int, kKvFmtBf16>{});
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
template <int Quantized>
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

          KVCacheAccessor<Quantized> k_acc(k_ptrs[sequence], nullptr, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptrs[sequence], nullptr, max_seq_len, head_dim);

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
template <int Quantized, int HeadDim>
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

          KVCacheAccessor<Quantized> k_acc(k_ptrs[sequence], nullptr, max_seq_len, head_dim);
          KVCacheAccessor<Quantized> v_acc(v_ptrs[sequence], nullptr, max_seq_len, head_dim);

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

void attention_decode_exact_head256_block_native_graph(
    Context &ctx, const Tensor &q, const Tensor &k_cache,
    const Tensor &v_cache, Tensor &output, Tensor &partials_buf,
    const Tensor &active_blocks, GraphControl control, int num_heads,
    int num_kv_heads, const Tensor *gate_buf) {
  attention_decode_exact_head256_block_native_graph_impl<
      kDecodeExact256QsaMaxTiles>(
      ctx, q, k_cache, v_cache, output, partials_buf, active_blocks,
      control, num_heads, num_kv_heads, gate_buf);
}

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
                      const Tensor *gate_buf,
                      const int *active_indices,
                      int num_active_tokens,
                      const int *active_blocks,
                      int num_active_blocks,
                      const Tensor *k_scale,
                      const Tensor *v_scale) {
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
  const int kv_fmt = kv_cache_format_of(k_cache);

  int effective_len = cached_len;
  int tail_start = 0;
  int sink_len = 0;
  bool contiguous_mode = true;

  if (active_blocks != nullptr && num_active_blocks > 0) {
    const int tail_len = cached_len - (cached_len / 4) * 4;
    effective_len = num_active_blocks * 4 + tail_len;
    contiguous_mode = false;
  } else if (active_indices != nullptr && num_active_tokens > 0) {
    effective_len = num_active_tokens;
    contiguous_mode = false;
  } else {
    bool apply_window = (decode_window > 0 && cached_len > decode_window_start);
    int effective_window = cached_len;
    if (apply_window) {
      effective_window = std::min(cached_len, decode_window);
    }
    tail_start = cached_len - effective_window;
    if (apply_window && decode_sink > 0) {
      sink_len = std::min(decode_sink, std::max(0, cached_len - effective_window));
    }
    effective_len = effective_window + sink_len;

    // If sink and tail overlap, fall back to full context contiguous view.
    if (effective_len >= cached_len) {
      tail_start = 0;
      sink_len = 0;
      effective_len = cached_len;
    }
    contiguous_mode = (sink_len == 0);
  }
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
      exact_partials_buf && exact_partials_buf->valid() && !active_indices && !active_blocks) {
    attention_decode_exact_head128_partial_merge(
        ctx, q, k_cache, v_cache, output, *exact_partials_buf, num_heads,
        num_kv_heads, tail_start, sink_len, effective_len, gate_buf, k_scale, v_scale);
    return;
  }

  // S0: the joint-matrix decode kernel reads a bf16 cache only. Quantized
  // caches must take the baseline path, which dequantizes per element. The
  // multi-row entry point already applies the same guard.
  if (allow_jm && kv_fmt == kKvFmtBf16 && contiguous_mode && head_dim == 128 &&
      cached_len > 0 && !active_indices && !active_blocks) {
    attention_decode_joint_matrix(ctx, q, k_cache, v_cache, output, scores_buf,
                                  num_heads, num_kv_heads, head_dim, cached_len,
                                  attn_start, jm_tile_id, decode_wg, gate_buf);
    return;
  }

  if (head_dim == 256 && cached_len > 0 && effective_len >= kDecodeExact256MinLen &&
      exact_partials_buf && exact_partials_buf->valid()) {
    // Length gate for AILA_ATTN_KVREUSE=pair2, decided here because this frame
    // owns both lengths. effective_len <= cached_len on every path, so the
    // dense shape (cached_len == effective_len) and the indexed shape
    // (cached_len is the whole prefix, effective_len the attended subset) are
    // both held to the floor; at 1834 attended tokens the pair2 kernel
    // regressed +8.0% and must not be reached.
    const int kv_reuse_min_len = attn_kv_reuse_min_len();
    const bool kv_reuse_len_ok =
        cached_len >= kv_reuse_min_len && effective_len >= kv_reuse_min_len;
    attention_decode_exact_head256_partial_merge(
        ctx, q, k_cache, v_cache, output, *exact_partials_buf, num_heads,
        num_kv_heads, tail_start, sink_len, effective_len, gate_buf, active_indices,
        active_blocks, num_active_blocks, k_scale, v_scale, kv_reuse_len_ok);
    return;
  }

  attention_decode_baseline(ctx, q, k_cache, v_cache, output, scores_buf,
                            num_heads, num_kv_heads, head_dim, cached_len,
                            tail_start, sink_len, effective_len,
                            decode_wg, gate_buf, active_indices,
                            active_blocks, num_active_blocks, k_scale, v_scale);
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
  // INT8 KV is prototype-scoped to the single-row Qwen4Exp dense decode path.
  // A batched call would need one scale tensor per row (scales are per
  // session), which the shared signature cannot express, so reject instead of
  // silently picking whichever row's scale happened to be passed.
  for (int r = 0; r < row_count; ++r) {
    if (rows[r].k_cache->dtype() == dnnl::memory::data_type::s8 ||
        rows[r].v_cache->dtype() == dnnl::memory::data_type::s8) {
      throw std::invalid_argument(
          "attention_decode_rows: int8 KV cache is not supported on the "
          "multi-row decode path");
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

template <int Quantized>
void attention_prefill_cached_impl(Context& ctx,
                                   Tensor& q, Tensor& k_cache, Tensor& v_cache,
                                   Tensor& output, Tensor& scores_buf,
                                   int seq_len, int start_pos,
                                   int num_heads, int num_kv_heads,
                                   int head_dim, int max_seq_len,
                                   const bf16* gate_ptr = nullptr,
                                   const float* k_scale = nullptr,
                                   const float* v_scale = nullptr) {
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

            KVCacheAccessor<Quantized> k_acc(k_ptr, k_scale, max_seq_len, exact_head_dim);
            KVCacheAccessor<Quantized> v_acc(v_ptr, v_scale, max_seq_len, exact_head_dim);

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

            KVCacheAccessor<Quantized> k_acc(k_ptr, k_scale, max_seq_len, exact_head_dim);
            KVCacheAccessor<Quantized> v_acc(v_ptr, v_scale, max_seq_len, exact_head_dim);

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

        KVCacheAccessor<Quantized> k_acc(k_ptr, k_scale, max_seq_len, head_dim);

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

        KVCacheAccessor<Quantized> v_acc(v_ptr, v_scale, max_seq_len, head_dim);

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
                              const Tensor* gate_buf,
                              const Tensor* k_scale,
                              const Tensor* v_scale) {
  const bf16 *gate_ptr = gate_buf && gate_buf->valid() ? gate_buf->data_as<bf16>() : nullptr;
  const int fmt = kv_cache_format_of(k_cache);
  const float *k_scale_ptr = kv_scale_ptr(k_scale);
  const float *v_scale_ptr = kv_scale_ptr(v_scale);
  if (fmt == kKvFmtInt8 && (!k_scale_ptr || !v_scale_ptr)) {
    throw std::invalid_argument("attention_prefill_cached: int8 KV cache requires k_scale/v_scale");
  }
  auto run = [&](auto tag) {
    attention_prefill_cached_impl<decltype(tag)::value>(
        ctx, q, k_cache, v_cache, output, scores_buf, seq_len, start_pos,
        num_heads, num_kv_heads, head_dim, max_seq_len, gate_ptr,
        k_scale_ptr, v_scale_ptr);
  };
  if (fmt == kKvFmtInt8) run(std::integral_constant<int, kKvFmtInt8>{});
  else if (fmt == kKvFmtFp8) run(std::integral_constant<int, kKvFmtFp8>{});
  else run(std::integral_constant<int, kKvFmtBf16>{});
}

// ============================================================
// SYCL Kernel: Copy K/V to Cache
// ============================================================

template <int Quantized>
void copy_to_cache_impl(Context &ctx, Tensor &new_kv, Tensor &cache, int seq_len,
                        int start_pos, int num_heads, int head_dim,
                        int max_seq_len, float *scale = nullptr) {
  bf16 *src = static_cast<bf16 *>(new_kv.data());
  void *dst_void = cache.data();
  int total_dim = num_heads * head_dim;

  if constexpr (Quantized == kKvFmtInt8) {
    // Per-(head, token) symmetric int8 write: one work-group per cache row
    // computes the row amax, publishes scale = amax/127, and stores codes.
    // A single pass, so no extra device traffic beyond the write itself.
    int8_t *dst = static_cast<int8_t *>(dst_void);
    constexpr int wg_size = 128;
    ctx.queue().submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> row_max(sycl::range<1>(wg_size), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>((size_t)seq_len * num_heads * wg_size, wg_size),
          [=](sycl::nd_item<1> item) {
            const int group = item.get_group(0);
            const int s = group / num_heads;
            const int h = group % num_heads;
            const int lid = item.get_local_id(0);
            const int src_base = s * total_dim + h * head_dim;
            float amax = 0.0f;
            for (int d = lid; d < head_dim; d += wg_size) {
              amax = sycl::fmax(amax, sycl::fabs(static_cast<float>(src[src_base + d])));
            }
            row_max[lid] = amax;
            item.barrier(sycl::access::fence_space::local_space);
            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                row_max[lid] = sycl::fmax(row_max[lid], row_max[lid + stride]);
              }
              item.barrier(sycl::access::fence_space::local_space);
            }
            const float row_scale = row_max[0] / 127.0f;
            if (lid == 0) {
              scale[h * max_seq_len + (start_pos + s)] = row_scale;
            }
            const int dst_base =
                h * max_seq_len * head_dim + (start_pos + s) * head_dim;
            for (int d = lid; d < head_dim; d += wg_size) {
              dst[dst_base + d] =
                  float_to_int8_kv(static_cast<float>(src[src_base + d]), row_scale);
            }
          });
    });
    return;
  }

  ctx.queue().parallel_for(sycl::range<2>(seq_len, total_dim),
                           [=](sycl::id<2> idx) {
                             int s = idx[0];
                             int flat = idx[1];
                             int h = flat / head_dim;
                             int d = flat % head_dim;

                             int src_idx = s * total_dim + flat;
                             int dst_idx = h * max_seq_len * head_dim +
                                           (start_pos + s) * head_dim + d;
                             if constexpr (Quantized == kKvFmtFp8) {
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
                   int max_seq_len, Tensor* scale_arg) {
  const int fmt = kv_cache_format_of(cache);
  // copy_to_cache writes a single tensor, so the caller passes that tensor's
  // own scale (k_scale for K, v_scale for V).
  float *scale = kv_scale_write_ptr(scale_arg);
  if (fmt == kKvFmtInt8 && !scale) {
    throw std::invalid_argument("copy_to_cache: int8 KV cache requires a scale tensor");
  }
  if (fmt == kKvFmtInt8) {
    copy_to_cache_impl<kKvFmtInt8>(ctx, new_kv, cache, seq_len, start_pos, num_heads, head_dim, max_seq_len, scale);
  } else if (fmt == kKvFmtFp8) {
    copy_to_cache_impl<kKvFmtFp8>(ctx, new_kv, cache, seq_len, start_pos, num_heads, head_dim, max_seq_len);
  } else {
    copy_to_cache_impl<kKvFmtBf16>(ctx, new_kv, cache, seq_len, start_pos, num_heads, head_dim, max_seq_len);
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

template <int Fmt>
void attention_prefill_qsa_batched_impl(
    Context& ctx,
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    const Tensor& active_blocks,
    Tensor& output,
    int count,
    int start_pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_topk,
    int compress_ratio,
    const Tensor* gate_buf,
    const Tensor* valid_counts,
    const Tensor* k_scale, const Tensor* v_scale) {
  if (count <= 0) return;
  if (head_dim != 256) {
    throw std::invalid_argument("attention_prefill_qsa_batched requires head_dim == 256");
  }
  if constexpr (Fmt == kKvFmtInt8) {
    if (!k_scale || !v_scale || !k_scale->valid() || !v_scale->valid())
      throw std::invalid_argument(
          "attention_prefill_qsa_batched: int8 KV cache requires k_scale/v_scale");
  }
  if (block_topk <= 0 || compress_ratio <= 0) {
    throw std::invalid_argument("attention_prefill_qsa_batched requires positive block_topk and compress_ratio");
  }
  const bf16* q_ptr = q.data_as<bf16>();
  const bf16* k_ptr = static_cast<const bf16*>(k_cache.data());
  const bf16* v_ptr = static_cast<const bf16*>(v_cache.data());
  const int8_t* k8_ptr = static_cast<const int8_t*>(k_cache.data());
  const int8_t* v8_ptr = static_cast<const int8_t*>(v_cache.data());
  const float* k_scale_ptr = (k_scale && k_scale->valid()) ? k_scale->data_as<float>() : nullptr;
  const float* v_scale_ptr = (v_scale && v_scale->valid()) ? v_scale->data_as<float>() : nullptr;
  const int* active_blocks_ptr = active_blocks.data_as<int>();
  bf16* o_ptr = output.data_as<bf16>();
  const bf16* gate_ptr = (gate_buf && gate_buf->valid()) ? gate_buf->data_as<bf16>() : nullptr;
  const int* valid_counts_ptr = (valid_counts && valid_counts->valid()) ? valid_counts->data_as<int>() : nullptr;

  const int heads_per_kv = std::max(1, (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1);
  const int max_seq_len = static_cast<int>(k_cache.shape(1));
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  constexpr int wg_size = 64;
  const size_t active_blocks_stride = static_cast<size_t>(block_topk + 4);

  const int heads_per_wg_cfg = read_env_int_local("AILA_QSA_HEADS_PER_WG", 2);
  const int s_heads_per_wg_env = (heads_per_wg_cfg <= 0)
      ? ((count >= 16 && (num_heads % 4 == 0) && (heads_per_kv % 4 == 0)) ? 4 : 2)
      : heads_per_wg_cfg;

  if (s_heads_per_wg_env == 4 && (num_heads % 4 == 0) && (heads_per_kv % 4 == 0)) {
    // Phase P2 (R4b): Multi-Q-head parallelism (4 Q heads sharing 1 KV head per workgroup)
    const int num_groups = num_heads / 4;
    ctx.queue().submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> q_cache(sycl::range<2>(4, head_dim), cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(static_cast<size_t>(count) * num_groups * wg_size, wg_size),
          [=](sycl::nd_item<1> item) {
            const int group = item.get_group(0);
            const int row = group / num_groups;
            const int g_idx = group % num_groups;
            const int lid = item.get_local_id(0);
            const int h0 = g_idx * 4;
            const int kv_head = h0 / heads_per_kv;

            auto k_at = [&](int tok, int d, size_t off) -> float {
              if constexpr (Fmt == kKvFmtInt8) {
                return static_cast<float>(k8_ptr[off + d]) *
                       k_scale_ptr[static_cast<size_t>(kv_head) * max_seq_len + tok];
              } else {
                return static_cast<float>(k_ptr[off + d]);
              }
            };
            auto v_at = [&](int tok, int d, size_t off) -> float {
              if constexpr (Fmt == kKvFmtInt8) {
                return static_cast<float>(v8_ptr[off + d]) *
                       v_scale_ptr[static_cast<size_t>(kv_head) * max_seq_len + tok];
              } else {
                return static_cast<float>(v_ptr[off + d]);
              }
            };

            int num_active_blocks = 0;
            int tail_len = 0;
            int tail_start = 0;
            bool is_sparse = false;

            if (valid_counts_ptr != nullptr) {
              num_active_blocks = valid_counts_ptr[row * 2 + 0];
              tail_len = valid_counts_ptr[row * 2 + 1];
              is_sparse = (num_active_blocks >= block_topk);
              const int pos = start_pos + row;
              tail_start = (pos + 1) - tail_len;
            } else {
              const int pos = start_pos + row;
              const int visible_len = pos + 1;
              const int num_complete_blocks = visible_len / compress_ratio;
              is_sparse = (num_complete_blocks > block_topk);
              num_active_blocks = is_sparse ? block_topk : num_complete_blocks;
              tail_start = num_complete_blocks * compress_ratio;
              tail_len = std::max(0, visible_len - tail_start);
            }

            for (int k = 0; k < 4; ++k) {
              const size_t q_off = (static_cast<size_t>(row) * num_heads + (h0 + k)) * head_dim;
              for (int d = lid; d < head_dim; d += wg_size) {
                q_cache[k][d] = static_cast<float>(q_ptr[q_off + d]);
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            const int* my_blocks = active_blocks_ptr + static_cast<size_t>(row) * active_blocks_stride;
            const size_t kv_offset = (static_cast<size_t>(kv_head) * max_seq_len) * head_dim;

            float running_max[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
            float running_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int d0 = lid;
            const int d1 = lid + 64;
            const int d2 = lid + 128;
            const int d3 = lid + 192;
            float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float acc3[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // Loop over active blocks
            for (int b = 0; b < num_active_blocks; ++b) {
              const int blk = is_sparse ? my_blocks[b] : b;
              const int base_t = blk * compress_ratio;

              for (int c = 0; c < compress_ratio; ++c) {
                const int cache_t = base_t + c;
                const size_t tok_offset = kv_offset + static_cast<size_t>(cache_t) * head_dim;
                const float k0 = k_at(cache_t, d0, tok_offset);
                const float k1 = k_at(cache_t, d1, tok_offset);
                const float k2 = k_at(cache_t, d2, tok_offset);
                const float k3 = k_at(cache_t, d3, tok_offset);

                float rescale[4];
                float exp_score[4];
                for (int k = 0; k < 4; ++k) {
                  const float local_dot = q_cache[k][d0] * k0 + q_cache[k][d1] * k1 + q_cache[k][d2] * k2 + q_cache[k][d3] * k3;
                  const float dot = sycl::reduce_over_group(item.get_group(), local_dot, sycl::plus<float>());
                  const float score = dot * scale;
                  const float new_max = sycl::fmax(running_max[k], score);
                  rescale[k] = (running_max[k] <= -1e20f) ? 0.0f : sycl::native::exp(running_max[k] - new_max);
                  exp_score[k] = sycl::native::exp(sycl::fmax(-88.0f, score - new_max));
                  running_sum[k] = running_sum[k] * rescale[k] + exp_score[k];
                  running_max[k] = new_max;
                }

                const float v0 = v_at(cache_t, d0, tok_offset);
                const float v1 = v_at(cache_t, d1, tok_offset);
                const float v2 = v_at(cache_t, d2, tok_offset);
                const float v3 = v_at(cache_t, d3, tok_offset);

                for (int k = 0; k < 4; ++k) {
                  acc0[k] = acc0[k] * rescale[k] + exp_score[k] * v0;
                  acc1[k] = acc1[k] * rescale[k] + exp_score[k] * v1;
                  acc2[k] = acc2[k] * rescale[k] + exp_score[k] * v2;
                  acc3[k] = acc3[k] * rescale[k] + exp_score[k] * v3;
                }
              }
            }

            // Loop over tail tokens
            for (int t = 0; t < tail_len; ++t) {
              const int cache_t = tail_start + t;
              const size_t tok_offset = kv_offset + static_cast<size_t>(cache_t) * head_dim;
              const float k0 = k_at(cache_t, d0, tok_offset);
              const float k1 = k_at(cache_t, d1, tok_offset);
              const float k2 = k_at(cache_t, d2, tok_offset);
              const float k3 = k_at(cache_t, d3, tok_offset);

              float rescale[4];
              float exp_score[4];
              for (int k = 0; k < 4; ++k) {
                const float local_dot = q_cache[k][d0] * k0 + q_cache[k][d1] * k1 + q_cache[k][d2] * k2 + q_cache[k][d3] * k3;
                const float dot = sycl::reduce_over_group(item.get_group(), local_dot, sycl::plus<float>());
                const float score = dot * scale;
                const float new_max = sycl::fmax(running_max[k], score);
                rescale[k] = (running_max[k] <= -1e20f) ? 0.0f : sycl::native::exp(running_max[k] - new_max);
                exp_score[k] = sycl::native::exp(sycl::fmax(-88.0f, score - new_max));
                running_sum[k] = running_sum[k] * rescale[k] + exp_score[k];
                running_max[k] = new_max;
              }

              const float v0 = v_at(cache_t, d0, tok_offset);
              const float v1 = v_at(cache_t, d1, tok_offset);
              const float v2 = v_at(cache_t, d2, tok_offset);
              const float v3 = v_at(cache_t, d3, tok_offset);

              for (int k = 0; k < 4; ++k) {
                acc0[k] = acc0[k] * rescale[k] + exp_score[k] * v0;
                acc1[k] = acc1[k] * rescale[k] + exp_score[k] * v1;
                acc2[k] = acc2[k] * rescale[k] + exp_score[k] * v2;
                acc3[k] = acc3[k] * rescale[k] + exp_score[k] * v3;
              }
            }

            for (int k = 0; k < 4; ++k) {
              const float inv_sum = (running_sum[k] > 0.0f) ? (1.0f / running_sum[k]) : 0.0f;
              float o0 = static_cast<float>(bf16(acc0[k] * inv_sum));
              float o1 = static_cast<float>(bf16(acc1[k] * inv_sum));
              float o2 = static_cast<float>(bf16(acc2[k] * inv_sum));
              float o3 = static_cast<float>(bf16(acc3[k] * inv_sum));

              const size_t out_head_offset = (static_cast<size_t>(row) * num_heads + (h0 + k)) * head_dim;
              if (gate_ptr != nullptr) {
                const float g0 = static_cast<float>(gate_ptr[out_head_offset + d0]);
                const float g1 = static_cast<float>(gate_ptr[out_head_offset + d1]);
                const float g2 = static_cast<float>(gate_ptr[out_head_offset + d2]);
                const float g3 = static_cast<float>(gate_ptr[out_head_offset + d3]);
                o0 = static_cast<float>(bf16(o0 * (1.0f / (1.0f + sycl::native::exp(-g0)))));
                o1 = static_cast<float>(bf16(o1 * (1.0f / (1.0f + sycl::native::exp(-g1)))));
                o2 = static_cast<float>(bf16(o2 * (1.0f / (1.0f + sycl::native::exp(-g2)))));
                o3 = static_cast<float>(bf16(o3 * (1.0f / (1.0f + sycl::native::exp(-g3)))));
              }

              o_ptr[out_head_offset + d0] = bf16(o0);
              o_ptr[out_head_offset + d1] = bf16(o1);
              o_ptr[out_head_offset + d2] = bf16(o2);
              o_ptr[out_head_offset + d3] = bf16(o3);
            }
          });
    });
  } else if (s_heads_per_wg_env >= 2 && (num_heads % 2 == 0) && (heads_per_kv % 2 == 0)) {
    // Phase P2 (R4b): Multi-Q-head parallelism (2 Q heads sharing 1 KV head per workgroup)
    const int num_pairs = num_heads / 2;
    ctx.queue().submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> q_cache0(sycl::range<1>(head_dim), cgh);
      sycl::local_accessor<float, 1> q_cache1(sycl::range<1>(head_dim), cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(static_cast<size_t>(count) * num_pairs * wg_size, wg_size),
          [=](sycl::nd_item<1> item) {
            const int group = item.get_group(0);
            const int row = group / num_pairs;
            const int pair_idx = group % num_pairs;
            const int lid = item.get_local_id(0);
            const int head0 = pair_idx * 2;
            const int head1 = pair_idx * 2 + 1;
            const int kv_head = head0 / heads_per_kv;

            auto k_at = [&](int tok, int d, size_t off) -> float {
              if constexpr (Fmt == kKvFmtInt8) {
                return static_cast<float>(k8_ptr[off + d]) *
                       k_scale_ptr[static_cast<size_t>(kv_head) * max_seq_len + tok];
              } else {
                return static_cast<float>(k_ptr[off + d]);
              }
            };
            auto v_at = [&](int tok, int d, size_t off) -> float {
              if constexpr (Fmt == kKvFmtInt8) {
                return static_cast<float>(v8_ptr[off + d]) *
                       v_scale_ptr[static_cast<size_t>(kv_head) * max_seq_len + tok];
              } else {
                return static_cast<float>(v_ptr[off + d]);
              }
            };

            int num_active_blocks = 0;
            int tail_len = 0;
            int tail_start = 0;
            bool is_sparse = false;

            if (valid_counts_ptr != nullptr) {
              num_active_blocks = valid_counts_ptr[row * 2 + 0];
              tail_len = valid_counts_ptr[row * 2 + 1];
              is_sparse = (num_active_blocks >= block_topk);
              const int pos = start_pos + row;
              tail_start = (pos + 1) - tail_len;
            } else {
              const int pos = start_pos + row;
              const int visible_len = pos + 1;
              const int num_complete_blocks = visible_len / compress_ratio;
              is_sparse = (num_complete_blocks > block_topk);
              num_active_blocks = is_sparse ? block_topk : num_complete_blocks;
              tail_start = num_complete_blocks * compress_ratio;
              tail_len = std::max(0, visible_len - tail_start);
            }

            const size_t q_head0_offset = (static_cast<size_t>(row) * num_heads + head0) * head_dim;
            const size_t q_head1_offset = (static_cast<size_t>(row) * num_heads + head1) * head_dim;
            for (int d = lid; d < head_dim; d += wg_size) {
              q_cache0[d] = static_cast<float>(q_ptr[q_head0_offset + d]);
              q_cache1[d] = static_cast<float>(q_ptr[q_head1_offset + d]);
            }
            item.barrier(sycl::access::fence_space::local_space);

            const int* my_blocks = active_blocks_ptr + static_cast<size_t>(row) * active_blocks_stride;
            const size_t kv_offset = (static_cast<size_t>(kv_head) * max_seq_len) * head_dim;

            float running_max0 = -1e30f, running_max1 = -1e30f;
            float running_sum0 = 0.0f, running_sum1 = 0.0f;
            const int d0 = lid;
            const int d1 = lid + 64;
            const int d2 = lid + 128;
            const int d3 = lid + 192;
            float acc0_0 = 0.0f, acc1_0 = 0.0f, acc2_0 = 0.0f, acc3_0 = 0.0f;
            float acc0_1 = 0.0f, acc1_1 = 0.0f, acc2_1 = 0.0f, acc3_1 = 0.0f;

            // Loop over active blocks
            for (int b = 0; b < num_active_blocks; ++b) {
              const int blk = is_sparse ? my_blocks[b] : b;
              const int base_t = blk * compress_ratio;

              for (int c = 0; c < compress_ratio; ++c) {
                const int cache_t = base_t + c;
                const size_t tok_offset = kv_offset + static_cast<size_t>(cache_t) * head_dim;
                const float k0 = k_at(cache_t, d0, tok_offset);
                const float k1 = k_at(cache_t, d1, tok_offset);
                const float k2 = k_at(cache_t, d2, tok_offset);
                const float k3 = k_at(cache_t, d3, tok_offset);

                const float local_dot0 = q_cache0[d0] * k0 + q_cache0[d1] * k1 + q_cache0[d2] * k2 + q_cache0[d3] * k3;
                const float local_dot1 = q_cache1[d0] * k0 + q_cache1[d1] * k1 + q_cache1[d2] * k2 + q_cache1[d3] * k3;

                const float dot0 = sycl::reduce_over_group(item.get_group(), local_dot0, sycl::plus<float>());
                const float dot1 = sycl::reduce_over_group(item.get_group(), local_dot1, sycl::plus<float>());

                const float score0 = dot0 * scale;
                const float score1 = dot1 * scale;

                const float new_max0 = sycl::fmax(running_max0, score0);
                const float rescale0 = (running_max0 <= -1e20f) ? 0.0f : sycl::native::exp(running_max0 - new_max0);
                const float exp_score0 = sycl::native::exp(sycl::fmax(-88.0f, score0 - new_max0));
                running_sum0 = running_sum0 * rescale0 + exp_score0;
                running_max0 = new_max0;

                const float new_max1 = sycl::fmax(running_max1, score1);
                const float rescale1 = (running_max1 <= -1e20f) ? 0.0f : sycl::native::exp(running_max1 - new_max1);
                const float exp_score1 = sycl::native::exp(sycl::fmax(-88.0f, score1 - new_max1));
                running_sum1 = running_sum1 * rescale1 + exp_score1;
                running_max1 = new_max1;

                const float v0 = v_at(cache_t, d0, tok_offset);
                const float v1 = v_at(cache_t, d1, tok_offset);
                const float v2 = v_at(cache_t, d2, tok_offset);
                const float v3 = v_at(cache_t, d3, tok_offset);

                acc0_0 = acc0_0 * rescale0 + exp_score0 * v0;
                acc1_0 = acc1_0 * rescale0 + exp_score0 * v1;
                acc2_0 = acc2_0 * rescale0 + exp_score0 * v2;
                acc3_0 = acc3_0 * rescale0 + exp_score0 * v3;

                acc0_1 = acc0_1 * rescale1 + exp_score1 * v0;
                acc1_1 = acc1_1 * rescale1 + exp_score1 * v1;
                acc2_1 = acc2_1 * rescale1 + exp_score1 * v2;
                acc3_1 = acc3_1 * rescale1 + exp_score1 * v3;
              }
            }

            // Loop over tail tokens
            for (int t = 0; t < tail_len; ++t) {
              const int cache_t = tail_start + t;
              const size_t tok_offset = kv_offset + static_cast<size_t>(cache_t) * head_dim;
              const float k0 = k_at(cache_t, d0, tok_offset);
              const float k1 = k_at(cache_t, d1, tok_offset);
              const float k2 = k_at(cache_t, d2, tok_offset);
              const float k3 = k_at(cache_t, d3, tok_offset);

              const float local_dot0 = q_cache0[d0] * k0 + q_cache0[d1] * k1 + q_cache0[d2] * k2 + q_cache0[d3] * k3;
              const float local_dot1 = q_cache1[d0] * k0 + q_cache1[d1] * k1 + q_cache1[d2] * k2 + q_cache1[d3] * k3;

              const float dot0 = sycl::reduce_over_group(item.get_group(), local_dot0, sycl::plus<float>());
              const float dot1 = sycl::reduce_over_group(item.get_group(), local_dot1, sycl::plus<float>());

              const float score0 = dot0 * scale;
              const float score1 = dot1 * scale;

              const float new_max0 = sycl::fmax(running_max0, score0);
              const float rescale0 = (running_max0 <= -1e20f) ? 0.0f : sycl::native::exp(running_max0 - new_max0);
              const float exp_score0 = sycl::native::exp(sycl::fmax(-88.0f, score0 - new_max0));
              running_sum0 = running_sum0 * rescale0 + exp_score0;
              running_max0 = new_max0;

              const float new_max1 = sycl::fmax(running_max1, score1);
              const float rescale1 = (running_max1 <= -1e20f) ? 0.0f : sycl::native::exp(running_max1 - new_max1);
              const float exp_score1 = sycl::native::exp(sycl::fmax(-88.0f, score1 - new_max1));
              running_sum1 = running_sum1 * rescale1 + exp_score1;
              running_max1 = new_max1;

              const float v0 = v_at(cache_t, d0, tok_offset);
              const float v1 = v_at(cache_t, d1, tok_offset);
              const float v2 = v_at(cache_t, d2, tok_offset);
              const float v3 = v_at(cache_t, d3, tok_offset);

              acc0_0 = acc0_0 * rescale0 + exp_score0 * v0;
              acc1_0 = acc1_0 * rescale0 + exp_score0 * v1;
              acc2_0 = acc2_0 * rescale0 + exp_score0 * v2;
              acc3_0 = acc3_0 * rescale0 + exp_score0 * v3;

              acc0_1 = acc0_1 * rescale1 + exp_score1 * v0;
              acc1_1 = acc1_1 * rescale1 + exp_score1 * v1;
              acc2_1 = acc2_1 * rescale1 + exp_score1 * v2;
              acc3_1 = acc3_1 * rescale1 + exp_score1 * v3;
            }

            const float inv_sum0 = (running_sum0 > 0.0f) ? (1.0f / running_sum0) : 0.0f;
            float o0_0 = static_cast<float>(bf16(acc0_0 * inv_sum0));
            float o1_0 = static_cast<float>(bf16(acc1_0 * inv_sum0));
            float o2_0 = static_cast<float>(bf16(acc2_0 * inv_sum0));
            float o3_0 = static_cast<float>(bf16(acc3_0 * inv_sum0));

            const float inv_sum1 = (running_sum1 > 0.0f) ? (1.0f / running_sum1) : 0.0f;
            float o0_1 = static_cast<float>(bf16(acc0_1 * inv_sum1));
            float o1_1 = static_cast<float>(bf16(acc1_1 * inv_sum1));
            float o2_1 = static_cast<float>(bf16(acc2_1 * inv_sum1));
            float o3_1 = static_cast<float>(bf16(acc3_1 * inv_sum1));

            const size_t out_head0_offset = (static_cast<size_t>(row) * num_heads + head0) * head_dim;
            const size_t out_head1_offset = (static_cast<size_t>(row) * num_heads + head1) * head_dim;

            if (gate_ptr != nullptr) {
              const float g0_0 = static_cast<float>(gate_ptr[out_head0_offset + d0]);
              const float g1_0 = static_cast<float>(gate_ptr[out_head0_offset + d1]);
              const float g2_0 = static_cast<float>(gate_ptr[out_head0_offset + d2]);
              const float g3_0 = static_cast<float>(gate_ptr[out_head0_offset + d3]);
              o0_0 = static_cast<float>(bf16(o0_0 * (1.0f / (1.0f + sycl::native::exp(-g0_0)))));
              o1_0 = static_cast<float>(bf16(o1_0 * (1.0f / (1.0f + sycl::native::exp(-g1_0)))));
              o2_0 = static_cast<float>(bf16(o2_0 * (1.0f / (1.0f + sycl::native::exp(-g2_0)))));
              o3_0 = static_cast<float>(bf16(o3_0 * (1.0f / (1.0f + sycl::native::exp(-g3_0)))));

              const float g0_1 = static_cast<float>(gate_ptr[out_head1_offset + d0]);
              const float g1_1 = static_cast<float>(gate_ptr[out_head1_offset + d1]);
              const float g2_1 = static_cast<float>(gate_ptr[out_head1_offset + d2]);
              const float g3_1 = static_cast<float>(gate_ptr[out_head1_offset + d3]);
              o0_1 = static_cast<float>(bf16(o0_1 * (1.0f / (1.0f + sycl::native::exp(-g0_1)))));
              o1_1 = static_cast<float>(bf16(o1_1 * (1.0f / (1.0f + sycl::native::exp(-g1_1)))));
              o2_1 = static_cast<float>(bf16(o2_1 * (1.0f / (1.0f + sycl::native::exp(-g2_1)))));
              o3_1 = static_cast<float>(bf16(o3_1 * (1.0f / (1.0f + sycl::native::exp(-g3_1)))));
            }

            o_ptr[out_head0_offset + d0] = bf16(o0_0);
            o_ptr[out_head0_offset + d1] = bf16(o1_0);
            o_ptr[out_head0_offset + d2] = bf16(o2_0);
            o_ptr[out_head0_offset + d3] = bf16(o3_0);

            o_ptr[out_head1_offset + d0] = bf16(o0_1);
            o_ptr[out_head1_offset + d1] = bf16(o1_1);
            o_ptr[out_head1_offset + d2] = bf16(o2_1);
            o_ptr[out_head1_offset + d3] = bf16(o3_1);
          });
    });
  } else {
    // Baseline 1 head per workgroup fallback
    ctx.queue().submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(static_cast<size_t>(count) * num_heads * wg_size, wg_size),
          [=](sycl::nd_item<1> item) {
            const int group = item.get_group(0);
            const int row = group / num_heads;
            const int head = group % num_heads;
            const int lid = item.get_local_id(0);
            const int kv_head = head / heads_per_kv;

            auto k_at = [&](int tok, int d, size_t off) -> float {
              if constexpr (Fmt == kKvFmtInt8) {
                return static_cast<float>(k8_ptr[off + d]) *
                       k_scale_ptr[static_cast<size_t>(kv_head) * max_seq_len + tok];
              } else {
                return static_cast<float>(k_ptr[off + d]);
              }
            };
            auto v_at = [&](int tok, int d, size_t off) -> float {
              if constexpr (Fmt == kKvFmtInt8) {
                return static_cast<float>(v8_ptr[off + d]) *
                       v_scale_ptr[static_cast<size_t>(kv_head) * max_seq_len + tok];
              } else {
                return static_cast<float>(v_ptr[off + d]);
              }
            };

            int num_active_blocks = 0;
            int tail_len = 0;
            int tail_start = 0;
            bool is_sparse = false;

            if (valid_counts_ptr != nullptr) {
              num_active_blocks = valid_counts_ptr[row * 2 + 0];
              tail_len = valid_counts_ptr[row * 2 + 1];
              is_sparse = (num_active_blocks >= block_topk);
              const int pos = start_pos + row;
              tail_start = (pos + 1) - tail_len;
            } else {
              const int pos = start_pos + row;
              const int visible_len = pos + 1;
              const int num_complete_blocks = visible_len / compress_ratio;
              is_sparse = (num_complete_blocks > block_topk);
              num_active_blocks = is_sparse ? block_topk : num_complete_blocks;
              tail_start = num_complete_blocks * compress_ratio;
              tail_len = std::max(0, visible_len - tail_start);
            }

            const size_t q_head_offset = (static_cast<size_t>(row) * num_heads + head) * head_dim;
            for (int d = lid; d < head_dim; d += wg_size) {
              q_cache[d] = static_cast<float>(q_ptr[q_head_offset + d]);
            }
            item.barrier(sycl::access::fence_space::local_space);

            const int* my_blocks = active_blocks_ptr + static_cast<size_t>(row) * active_blocks_stride;
            const size_t kv_offset = (static_cast<size_t>(kv_head) * max_seq_len) * head_dim;

            float running_max = -1e30f;
            float running_sum = 0.0f;
            const int d0 = lid;
            const int d1 = lid + 64;
            const int d2 = lid + 128;
            const int d3 = lid + 192;
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

            // Loop over active blocks
            for (int b = 0; b < num_active_blocks; ++b) {
              const int blk = is_sparse ? my_blocks[b] : b;
              const int base_t = blk * compress_ratio;

              for (int c = 0; c < compress_ratio; ++c) {
                const int cache_t = base_t + c;
                const size_t tok_offset = kv_offset + static_cast<size_t>(cache_t) * head_dim;
                const float k0 = k_at(cache_t, d0, tok_offset);
                const float k1 = k_at(cache_t, d1, tok_offset);
                const float k2 = k_at(cache_t, d2, tok_offset);
                const float k3 = k_at(cache_t, d3, tok_offset);
                const float local_dot = q_cache[d0] * k0 + q_cache[d1] * k1 + q_cache[d2] * k2 + q_cache[d3] * k3;
                const float dot = sycl::reduce_over_group(item.get_group(), local_dot, sycl::plus<float>());
                const float score = dot * scale;

                const float new_max = sycl::fmax(running_max, score);
                const float rescale = (running_max <= -1e20f) ? 0.0f : sycl::native::exp(running_max - new_max);
                const float exp_score = sycl::native::exp(sycl::fmax(-88.0f, score - new_max));
                running_sum = running_sum * rescale + exp_score;
                running_max = new_max;

                const float v0 = v_at(cache_t, d0, tok_offset);
                const float v1 = v_at(cache_t, d1, tok_offset);
                const float v2 = v_at(cache_t, d2, tok_offset);
                const float v3 = v_at(cache_t, d3, tok_offset);

                acc0 = acc0 * rescale + exp_score * v0;
                acc1 = acc1 * rescale + exp_score * v1;
                acc2 = acc2 * rescale + exp_score * v2;
                acc3 = acc3 * rescale + exp_score * v3;
              }
            }

            // Loop over tail tokens
            for (int t = 0; t < tail_len; ++t) {
              const int cache_t = tail_start + t;
              const size_t tok_offset = kv_offset + static_cast<size_t>(cache_t) * head_dim;
              const float k0 = k_at(cache_t, d0, tok_offset);
              const float k1 = k_at(cache_t, d1, tok_offset);
              const float k2 = k_at(cache_t, d2, tok_offset);
              const float k3 = k_at(cache_t, d3, tok_offset);
              const float local_dot = q_cache[d0] * k0 + q_cache[d1] * k1 + q_cache[d2] * k2 + q_cache[d3] * k3;
              const float dot = sycl::reduce_over_group(item.get_group(), local_dot, sycl::plus<float>());
              const float score = dot * scale;

              const float new_max = sycl::fmax(running_max, score);
              const float rescale = (running_max <= -1e20f) ? 0.0f : sycl::native::exp(running_max - new_max);
              const float exp_score = sycl::native::exp(sycl::fmax(-88.0f, score - new_max));
              running_sum = running_sum * rescale + exp_score;
              running_max = new_max;

              const float v0 = v_at(cache_t, d0, tok_offset);
              const float v1 = v_at(cache_t, d1, tok_offset);
              const float v2 = v_at(cache_t, d2, tok_offset);
              const float v3 = v_at(cache_t, d3, tok_offset);

              acc0 = acc0 * rescale + exp_score * v0;
              acc1 = acc1 * rescale + exp_score * v1;
              acc2 = acc2 * rescale + exp_score * v2;
              acc3 = acc3 * rescale + exp_score * v3;
            }

            const float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
            float o0 = static_cast<float>(bf16(acc0 * inv_sum));
            float o1 = static_cast<float>(bf16(acc1 * inv_sum));
            float o2 = static_cast<float>(bf16(acc2 * inv_sum));
            float o3 = static_cast<float>(bf16(acc3 * inv_sum));

            const size_t out_head_offset = (static_cast<size_t>(row) * num_heads + head) * head_dim;
            if (gate_ptr != nullptr) {
              const float g0 = static_cast<float>(gate_ptr[out_head_offset + d0]);
              const float g1 = static_cast<float>(gate_ptr[out_head_offset + d1]);
              const float g2 = static_cast<float>(gate_ptr[out_head_offset + d2]);
              const float g3 = static_cast<float>(gate_ptr[out_head_offset + d3]);
              o0 = static_cast<float>(bf16(o0 * (1.0f / (1.0f + sycl::native::exp(-g0)))));
              o1 = static_cast<float>(bf16(o1 * (1.0f / (1.0f + sycl::native::exp(-g1)))));
              o2 = static_cast<float>(bf16(o2 * (1.0f / (1.0f + sycl::native::exp(-g2)))));
              o3 = static_cast<float>(bf16(o3 * (1.0f / (1.0f + sycl::native::exp(-g3)))));
            }

            o_ptr[out_head_offset + d0] = bf16(o0);
            o_ptr[out_head_offset + d1] = bf16(o1);
            o_ptr[out_head_offset + d2] = bf16(o2);
            o_ptr[out_head_offset + d3] = bf16(o3);
          });
    });
  }
}


void attention_prefill_qsa_batched(
    Context& ctx,
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    const Tensor& active_blocks,
    Tensor& output,
    int count,
    int start_pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_topk,
    int compress_ratio,
    const Tensor* gate_buf,
    const Tensor* valid_counts,
    const Tensor* k_scale,
    const Tensor* v_scale) {
  const int fmt = kv_cache_format_of(k_cache);
  if (fmt == kKvFmtInt8) {
    attention_prefill_qsa_batched_impl<kKvFmtInt8>(
        ctx, q, k_cache, v_cache, active_blocks, output, count, start_pos,
        num_heads, num_kv_heads, head_dim, block_topk, compress_ratio, gate_buf,
        valid_counts, k_scale, v_scale);
    return;
  }
  if (fmt != kKvFmtBf16)
    throw std::invalid_argument(
        "attention_prefill_qsa_batched requires a bf16 or int8 KV cache");
  attention_prefill_qsa_batched_impl<kKvFmtBf16>(
      ctx, q, k_cache, v_cache, active_blocks, output, count, start_pos,
      num_heads, num_kv_heads, head_dim, block_topk, compress_ratio, gate_buf,
      valid_counts, k_scale, v_scale);
}

void attention_prefill_qsa_batched_host(
    const bf16* q,
    const bf16* k_cache,
    const bf16* v_cache,
    const int* active_blocks,
    bf16* output,
    int count,
    int start_pos,
    int capacity,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_topk,
    int compress_ratio,
    const bf16* gate,
    const int* valid_counts) {
  if (count <= 0) return;
  if (head_dim != 256) {
    throw std::invalid_argument("attention_prefill_qsa_batched_host requires head_dim == 256");
  }
  if (block_topk <= 0 || compress_ratio <= 0) {
    throw std::invalid_argument("attention_prefill_qsa_batched_host requires positive block_topk and compress_ratio");
  }
  const int heads_per_kv = std::max(1, (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const size_t active_blocks_stride = static_cast<size_t>(block_topk + 4);

  for (int row = 0; row < count; ++row) {
    int num_active_blocks = 0;
    int tail_len = 0;
    int tail_start = 0;
    bool is_sparse = false;

    if (valid_counts != nullptr) {
      num_active_blocks = valid_counts[row * 2 + 0];
      tail_len = valid_counts[row * 2 + 1];
      is_sparse = (num_active_blocks >= block_topk);
      const int pos = start_pos + row;
      tail_start = (pos + 1) - tail_len;
    } else {
      const int pos = start_pos + row;
      const int visible_len = pos + 1;
      const int num_complete_blocks = visible_len / compress_ratio;
      is_sparse = (num_complete_blocks > block_topk);
      num_active_blocks = is_sparse ? block_topk : num_complete_blocks;
      tail_start = num_complete_blocks * compress_ratio;
      tail_len = std::max(0, visible_len - tail_start);
    }
    const int* my_blocks = active_blocks + static_cast<size_t>(row) * active_blocks_stride;

    std::vector<int> active_tokens;
    active_tokens.reserve(num_active_blocks * compress_ratio + tail_len);
    for (int b = 0; b < num_active_blocks; ++b) {
      const int blk = is_sparse ? my_blocks[b] : b;
      for (int c = 0; c < compress_ratio; ++c) {
        active_tokens.push_back(blk * compress_ratio + c);
      }
    }
    for (int t = 0; t < tail_len; ++t) {
      active_tokens.push_back(tail_start + t);
    }
    const int n_tokens = static_cast<int>(active_tokens.size());

    for (int head = 0; head < num_heads; ++head) {
      const int kv_head = head / heads_per_kv;
      const size_t q_off = (static_cast<size_t>(row) * num_heads + head) * head_dim;
      const size_t kv_head_off = static_cast<size_t>(kv_head) * capacity * head_dim;

      std::vector<float> scores(n_tokens);
      float max_score = -1e30f;
      for (int i = 0; i < n_tokens; ++i) {
        const int tok = active_tokens[i];
        const size_t k_off = kv_head_off + static_cast<size_t>(tok) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += static_cast<float>(q[q_off + d]) * static_cast<float>(k_cache[k_off + d]);
        }
        scores[i] = dot * scale;
        if (scores[i] > max_score) max_score = scores[i];
      }

      float sum_exp = 0.0f;
      for (int i = 0; i < n_tokens; ++i) {
        scores[i] = std::exp(scores[i] - max_score);
        sum_exp += scores[i];
      }
      const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

      for (int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int i = 0; i < n_tokens; ++i) {
          const int tok = active_tokens[i];
          const size_t v_off = kv_head_off + static_cast<size_t>(tok) * head_dim;
          acc += scores[i] * static_cast<float>(v_cache[v_off + d]);
        }
        float val = static_cast<float>(bf16(acc * inv_sum));
        if (gate != nullptr) {
          float g = static_cast<float>(gate[q_off + d]);
          float s = 1.0f / (1.0f + std::exp(-g));
          val = static_cast<float>(bf16(val * s));
        }
        output[q_off + d] = bf16(val);
      }
    }
  }
}
} // namespace ops
