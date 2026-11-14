#include "ops/QsaIndexerOps.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ops {

namespace {

// 0: legacy; 1: native one-bit reduction; 2/3: parallel 4/8-bit radix; 4: vec4 loads.
struct QsaSelectorPolicy {
    int mode = 0;
    explicit QsaSelectorPolicy(Context& ctx) {
        const auto device = ctx.queue().get_device();
        const auto name = device.get_info<sycl::info::device::name>();
        if (ctx.queue().get_backend() == sycl::backend::ext_oneapi_level_zero &&
            device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
            name.find("A770") != std::string::npos) mode = 4;
    }
};

int qsa_selector_mode(Context& ctx, int topk, int ratio, int blocks) {
    static const int mode = aila::env::read_int_raw("AILA_QSA_SELECTOR", -1);
    if (mode >= 0) return mode;
    // A770/Level Zero, Qwen's 512-block budget, through 262K visible tokens.
    // Cache device identification per Context, never in a process-global device flag.
    if (topk != 512 || ratio != 4 || blocks > 65536) return 0;
    return ctx.workspace<QsaSelectorPolicy>().mode;
}

// Map IEEE-754 values to an unsigned key with the same order as numeric comparison.
// QSA scores are non-negative, but keeping the helper well-defined for the public
// operator also gives NaNs a deterministic lowest rank and treats signed zero as a tie.
inline std::uint32_t qsa_score_order_key(float score) {
    std::uint32_t bits = sycl::bit_cast<std::uint32_t>(score);
    const std::uint32_t magnitude = bits & 0x7fffffffU;
    if ((magnitude & 0x7f800000U) == 0x7f800000U && (magnitude & 0x007fffffU) != 0) {
        return 0U; // NaNs rank below every numeric score.
    }
    if (magnitude == 0U) bits = 0U; // +0 and -0 compare equal.
    return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

// Phase P2 (R4c): Total order contract for complement selection.
// Candidate a is strictly worse than candidate b if its score key is strictly lower,
// or if score keys are equal and its block ID is strictly greater (ties prefer smaller ID,
// so complement selection drops the largest ID first).
inline bool qsa_candidate_is_worse(std::uint32_t a_key, int a_id, std::uint32_t b_key, int b_id) {
    if (a_key != b_key) return a_key < b_key;
    return a_id > b_id;
}

// Exact integer-key selection. Each subgroup owns a histogram to bound local
// atomic contention; all lanes participate in digit scans and stable compaction.
// Output remains ascending by block ID, including ties at the cutoff.
template <int Bits, int LoadWidth = 1>
void qsa_radix_select_blocks(Context& ctx, const float* scores, int* output,
    int queries, int score_stride, int start_pos, int topk, int ratio,
    size_t output_stride, int* valid_counts) {
    constexpr int wg = 256, bins = 1 << Bits, subgroups = wg / 16;
    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> hist(sycl::range<1>(subgroups * bins), cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(queries) * wg, wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = item.get_local_id(0), row = item.get_group(0);
                const int length = start_pos + row + 1, blocks = length / ratio;
                const int tail = length % ratio, selected = sycl::min(blocks, topk);
                const float* sc = scores + size_t(row) * score_stride;
                int* out = output + size_t(row) * output_stride;
                const auto group = item.get_group();
                if (valid_counts && lid == 0) {
                    valid_counts[2 * row] = selected;
                    valid_counts[2 * row + 1] = tail;
                }
                if (lid < tail) out[selected + lid] = blocks * ratio + lid;
                if (blocks <= topk) {
                    for (int b = lid; b < blocks; b += wg) out[b] = b;
                    return;
                }
                std::uint32_t prefix = 0, mask = 0;
                int rank = topk;
                for (int shift = 32 - Bits; shift >= 0; shift -= Bits) {
                    for (int i = lid; i < subgroups * bins; i += wg) hist[i] = 0;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int b = lid * LoadWidth; b < blocks; b += wg * LoadWidth) {
                        sycl::vec<float, LoadWidth> values;
                        if (b + LoadWidth <= blocks) {
                            values.load(0, sycl::address_space_cast<sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(sc + b));
                        } else {
                            for (int j = 0; j < LoadWidth; ++j)
                                values[j] = b + j < blocks ? sc[b + j] : 0.0f;
                        }
                        for (int j = 0; j < LoadWidth; ++j) {
                            const auto key = qsa_score_order_key(values[j]);
                            if (b + j < blocks && (key & mask) == prefix) {
                                const int bin = (key >> shift) & (bins - 1);
                                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                    sycl::memory_scope::work_group,
                                    sycl::access::address_space::local_space> count(
                                        hist[(lid / 16) * bins + bin]);
                                count.fetch_add(1);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    int count = 0;
                    if (lid < bins) {
                        for (int sg = 0; sg < subgroups; ++sg)
                            count += hist[sg * bins + (bins - 1 - lid)];
                    }
                    const int inclusive = sycl::inclusive_scan_over_group(group, count, sycl::plus<int>());
                    const int before = inclusive - count;
                    const bool cutoff = lid < bins && before < rank && inclusive >= rank;
                    const int digit = sycl::reduce_over_group(group, cutoff ? bins - 1 - lid : 0, sycl::maximum<int>());
                    const int above = sycl::reduce_over_group(group, cutoff ? before : 0, sycl::plus<int>());
                    prefix |= std::uint32_t(digit) << shift;
                    mask |= std::uint32_t(bins - 1) << shift;
                    rank -= above;
                    item.barrier(sycl::access::fence_space::local_space);
                }
                // Contiguous per-lane intervals make global compaction stable.
                const int per = (blocks + wg - 1) / wg;
                const int begin = lid * per, end = sycl::min(begin + per, blocks);
                int greater = 0, equal = 0;
                for (int b = begin; b < end; ++b) {
                    const auto key = qsa_score_order_key(sc[b]);
                    greater += key > prefix; equal += key == prefix;
                }
                const int eq_before = sycl::exclusive_scan_over_group(group, equal, sycl::plus<int>());
                int ties = sycl::min(equal, sycl::max(0, rank - eq_before));
                int offset = sycl::exclusive_scan_over_group(group, greater + ties, sycl::plus<int>());
                for (int b = begin; b < end; ++b) {
                    const auto key = qsa_score_order_key(sc[b]);
                    if (key > prefix || (key == prefix && ties > 0)) {
                        out[offset++] = b;
                        if (key == prefix) --ties;
                    }
                }
            });
    });
}

inline float calculate_rotary_position(
    int global_pos, int d, float pos_scalar,
    bool interleaved, const int* pos_t, const int* pos_h, const int* pos_w,
    int prompt_pos_len, int mrope_section_h, int mrope_section_w) {
    float position_value = pos_scalar;
    if (interleaved && pos_t && pos_h && pos_w) {
        int stream = 0;
        if ((d % 3) == 1 && (d / 3) < mrope_section_h) {
            stream = 1;
        } else if ((d % 3) == 2 && (d / 3) < mrope_section_w) {
            stream = 2;
        } else {
            stream = 0;
        }

        int t_val = (global_pos < prompt_pos_len) ? pos_t[global_pos] : static_cast<int>(pos_scalar);
        int h_val = (global_pos < prompt_pos_len) ? pos_h[global_pos] : static_cast<int>(pos_scalar);
        int w_val = (global_pos < prompt_pos_len) ? pos_w[global_pos] : static_cast<int>(pos_scalar);
        if (stream == 1) position_value = static_cast<float>(h_val);
        else if (stream == 2) position_value = static_cast<float>(w_val);
        else position_value = static_cast<float>(t_val);
    }
    return position_value;
}

} // anonymous namespace

namespace {

void qsa_prepare_q_and_cache_k_impl(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    Tensor& k_cache,
    Tensor& q_out,
    int pos,
    GraphControl control,
    int layer,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    int k_cache_base_token = 0) {
    (void)mrope_section_t;
    if (!control && pos < 0) return;
    const bf16* qk_ptr = indexer_qk.data_as<bf16>();
    const bf16* qn_ptr = q_norm_w.data_as<bf16>();
    bf16* k_cache_ptr = k_cache.data_as<bf16>();
    bf16* q_out_ptr = q_out.data_as<bf16>();
    const int capacity = static_cast<int>(k_cache.shape(0));
    const int cache_base = control ? 0 : k_cache_base_token;

    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kWgSize = 128;
    constexpr int kNumHeads = 4;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<float, 1> local_q(sycl::range<1>(kHeadDim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(kNumHeads * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int head = item.get_group(0);
                const int lid = item.get_local_id(0);
                int active_pos = pos;
                if (control) {
                    int visible_len = 0;
                    int compressed_blocks = 0;
                    if (!graph_control_pending_blocks(
                            control, capacity / 4, visible_len,
                            compressed_blocks, layer))
                        return;
                    active_pos = visible_len - 1;
                }
                if (active_pos < cache_base || active_pos - cache_base >= capacity) {
                    if (control) graph_control_mark_abort(control);
                    return;
                }
                const float pos_scalar = static_cast<float>(
                    active_pos >= prompt_pos_len
                        ? active_pos + text_pos_delta
                        : active_pos);
                const int q_offset = head * kHeadDim + lid;

                const float qv = static_cast<float>(qk_ptr[q_offset]);
                local_sum[lid] = qv * qv;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float q_rms = sycl::sqrt(local_sum[0] / static_cast<float>(kHeadDim) + 1e-6f);
                const float q_normed = (qv / q_rms) * static_cast<float>(qn_ptr[lid]);
                local_q[lid] = q_normed;
                item.barrier(sycl::access::fence_space::local_space);

                if (lid < kHalfDim) {
                    const float pos_val = calculate_rotary_position(
                        active_pos, lid, pos_scalar, interleaved, pos_t,
                        pos_h, pos_w,
                        prompt_pos_len, mrope_section_h, mrope_section_w);
                    const float freq = 1.0f / sycl::pow(theta, (2.0f * lid) / static_cast<float>(kRotaryDim));
                    const float angle = pos_val * freq;
                    const float c = sycl::native::cos(angle);
                    const float s = sycl::native::sin(angle);
                    const float q0 = local_q[lid];
                    const float q1 = local_q[lid + kHalfDim];
                    q_out_ptr[head * kHeadDim + lid] = bf16(q0 * c - q1 * s);
                    q_out_ptr[head * kHeadDim + lid + kHalfDim] = bf16(q1 * c + q0 * s);
                } else if (lid >= kRotaryDim) {
                    q_out_ptr[head * kHeadDim + lid] = bf16(local_q[lid]);
                }

                // If head 0, copy raw token_k into k_cache
                if (head == 0) {
                    k_cache_ptr[static_cast<size_t>(active_pos - cache_base) * kHeadDim + lid] =
                        qk_ptr[kNumHeads * kHeadDim + lid];
                }
            });
    });
}

} // namespace

void qsa_prepare_q_and_cache_k(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    Tensor& k_cache,
    Tensor& q_out,
    int pos,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    int k_cache_base_token) {
    qsa_prepare_q_and_cache_k_impl(
        ctx, indexer_qk, q_norm_w, k_cache, q_out, pos, nullptr, 0, theta,
        interleaved, pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w, k_cache_base_token);
}

void qsa_prepare_q_and_cache_k_graph(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    Tensor& k_cache,
    Tensor& q_out,
    GraphControl control,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    int layer) {
    if (!control)
        throw std::invalid_argument(
            "qsa_prepare_q_and_cache_k_graph requires graph control");
    if (graph_control_layer_blocks_slot(layer) < 0)
        throw std::invalid_argument(
            "qsa_prepare_q_and_cache_k_graph layer out of range");
    if (!indexer_qk.valid() ||
        indexer_qk.dtype() != dnnl::memory::data_type::bf16 ||
        indexer_qk.numel() < 5 * 128 ||
        !q_norm_w.valid() ||
        q_norm_w.dtype() != dnnl::memory::data_type::bf16 ||
        q_norm_w.numel() < 128 ||
        !k_cache.valid() ||
        k_cache.dtype() != dnnl::memory::data_type::bf16 ||
        k_cache.ndim() != 2 ||
        k_cache.shape(0) <= 0 ||
        k_cache.shape(0) > std::numeric_limits<int>::max() ||
        k_cache.shape(1) != 128 ||
        !q_out.valid() ||
        q_out.dtype() != dnnl::memory::data_type::bf16 ||
        q_out.numel() < 4 * 128) {
        throw std::invalid_argument(
            "qsa_prepare_q_and_cache_k_graph shape mismatch");
    }
    qsa_prepare_q_and_cache_k_impl(
        ctx, indexer_qk, q_norm_w, k_cache, q_out, -1, control, layer, theta,
        interleaved, pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w);
}

void qsa_cache_k_batched(
    Context& ctx,
    const Tensor& indexer_qk,
    Tensor& k_cache,
    int start_pos,
    int count,
    int k_cache_base_token) {
    if (count <= 0) return;
    const bf16* qk_ptr = indexer_qk.data_as<bf16>();
    bf16* k_cache_ptr = k_cache.data_as<bf16>();

    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int kTotalDim = (kNumHeads + 1) * kHeadDim; // 640
    const int rel_start = start_pos - k_cache_base_token;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>(static_cast<size_t>(count) * kHeadDim),
            [=](sycl::id<1> idx) {
                const int i = static_cast<int>(idx[0]);
                const int t = i / kHeadDim;
                const int d = i % kHeadDim;
                k_cache_ptr[static_cast<size_t>(rel_start + t) * kHeadDim + d] =
                    qk_ptr[static_cast<size_t>(t) * kTotalDim + kNumHeads * kHeadDim + d];
            });
    });
}

void qsa_compute_block_scores(
    Context& ctx,
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& k_norm_w,
    Tensor& scores_out,
    int num_complete_blocks,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w) {
    (void)mrope_section_t;
    if (num_complete_blocks <= 0) return;

    const bf16* q_ptr = q.data_as<bf16>();
    const bf16* k_cache_ptr = k_cache.data_as<bf16>();
    const bf16* kn_ptr = k_norm_w.data_as<bf16>();
    float* scores_ptr = scores_out.data_as<float>();

    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kWgSize = 128;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<float, 1> local_k(sycl::range<1>(kHeadDim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_complete_blocks * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int b = item.get_group(0);
                const int lid = item.get_local_id(0);

                // 1. Mean-pool 4 raw keys
                const size_t b0 = static_cast<size_t>(4 * b + 0) * kHeadDim + lid;
                const size_t b1 = static_cast<size_t>(4 * b + 1) * kHeadDim + lid;
                const size_t b2 = static_cast<size_t>(4 * b + 2) * kHeadDim + lid;
                const size_t b3 = static_cast<size_t>(4 * b + 3) * kHeadDim + lid;

                const float k0 = static_cast<float>(k_cache_ptr[b0]);
                const float k1 = static_cast<float>(k_cache_ptr[b1]);
                const float k2 = static_cast<float>(k_cache_ptr[b2]);
                const float k3 = static_cast<float>(k_cache_ptr[b3]);
                const float k_pooled_raw = (k0 + k1 + k2 + k3) * 0.25f;
                // Match PyTorch: cast to bf16 then back to float
                const float k_pooled = static_cast<float>(bf16(k_pooled_raw));

                // 2. RMSNorm across 128 dims
                local_sum[lid] = k_pooled * k_pooled;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float k_rms = sycl::sqrt(local_sum[0] / static_cast<float>(kHeadDim) + 1e-6f);
                const float k_normed = (k_pooled / k_rms) * static_cast<float>(kn_ptr[lid]);
                local_k[lid] = k_normed;
                item.barrier(sycl::access::fence_space::local_space);

                // 3. RoPE at group_starts = 4 * b
                const int blk_pos = 4 * b;
                const float pos_scalar = static_cast<float>(blk_pos >= prompt_pos_len ? blk_pos + text_pos_delta : blk_pos);
                if (lid < kHalfDim) {
                    const float pos_val = calculate_rotary_position(
                        blk_pos, lid, pos_scalar, interleaved, pos_t, pos_h, pos_w,
                        prompt_pos_len, mrope_section_h, mrope_section_w);
                    const float freq = 1.0f / sycl::pow(theta, (2.0f * lid) / static_cast<float>(kRotaryDim));
                    const float angle = pos_val * freq;
                    const float c = sycl::native::cos(angle);
                    const float s = sycl::native::sin(angle);
                    const float kv0 = local_k[lid];
                    const float kv1 = local_k[lid + kHalfDim];
                    local_k[lid] = kv0 * c - kv1 * s;
                    local_k[lid + kHalfDim] = kv1 * c + kv0 * s;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // 4. Dot product with 4 query heads
                const float kl = local_k[lid];
                const float p0 = static_cast<float>(q_ptr[0 * kHeadDim + lid]) * kl;
                const float p1 = static_cast<float>(q_ptr[1 * kHeadDim + lid]) * kl;
                const float p2 = static_cast<float>(q_ptr[2 * kHeadDim + lid]) * kl;
                const float p3 = static_cast<float>(q_ptr[3 * kHeadDim + lid]) * kl;

                const float dot0 = sycl::reduce_over_group(item.get_group(), p0, sycl::plus<float>());
                const float dot1 = sycl::reduce_over_group(item.get_group(), p1, sycl::plus<float>());
                const float dot2 = sycl::reduce_over_group(item.get_group(), p2, sycl::plus<float>());
                const float dot3 = sycl::reduce_over_group(item.get_group(), p3, sycl::plus<float>());

                if (lid == 0) {
                    const float s0 = sycl::fmax(0.0f, dot0);
                    const float s1 = sycl::fmax(0.0f, dot1);
                    const float s2 = sycl::fmax(0.0f, dot2);
                    const float s3 = sycl::fmax(0.0f, dot3);
                    constexpr float inv_sqrt_d = 0.08838834764831845f; // 1.0 / sqrt(128.0f)
                    scores_ptr[b] = (s0 + s1 + s2 + s3) * inv_sqrt_d;
                }
            });
    });
}

int qsa_select_topk_and_expand_tokens(
    Context& ctx,
    const Tensor& scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    Tensor& active_tokens_out) {
    if (cached_len <= 0 || block_topk <= 0) return 0;

    if (num_complete_blocks <= block_topk) {
        // All complete blocks + tail are selected: dense identity [0 .. cached_len - 1]
        int* out_ptr = active_tokens_out.data_as<int>();
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::range<1>(cached_len),
                [=](sycl::id<1> idx) {
                    out_ptr[idx[0]] = static_cast<int>(idx[0]);
                });
        });
        return cached_len;
    }

    if (num_complete_blocks == block_topk + 1) {
        // R4c: Complement selection for B = K + 1 (e.g. 513 -> 512).
        // Find single worst block via minimum-reduction, then emit all remaining blocks.
        const float* scores_ptr = scores.data_as<float>();
        int* out_ptr = active_tokens_out.data_as<int>();
        constexpr int kWgSize = 256;

        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<std::uint32_t, 1> local_worst_key(sycl::range<1>(kWgSize), cgh);
            sycl::local_accessor<int, 1> local_worst_id(sycl::range<1>(kWgSize), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(kWgSize, kWgSize),
                [=](sycl::nd_item<1> item) {
                    const int lid = item.get_local_id(0);

                    std::uint32_t my_worst_key = 0xffffffffU;
                    int my_worst_id = -1;
                    for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if (my_worst_id < 0 || qsa_candidate_is_worse(key, b, my_worst_key, my_worst_id)) {
                            my_worst_key = key;
                            my_worst_id = b;
                        }
                    }
                    local_worst_key[lid] = my_worst_key;
                    local_worst_id[lid] = my_worst_id;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            const std::uint32_t other_key = local_worst_key[lid + stride];
                            const int other_id = local_worst_id[lid + stride];
                            if (other_id >= 0) {
                                if (local_worst_id[lid] < 0 || qsa_candidate_is_worse(other_key, other_id, local_worst_key[lid], local_worst_id[lid])) {
                                    local_worst_key[lid] = other_key;
                                    local_worst_id[lid] = other_id;
                                }
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    const int worst_id = local_worst_id[0];

                    for (int idx = lid; idx < block_topk; idx += kWgSize) {
                        const int blk = (idx < worst_id) ? idx : (idx + 1);
                        for (int c = 0; c < compress_ratio; ++c) {
                            out_ptr[idx * compress_ratio + c] = blk * compress_ratio + c;
                        }
                    }

                    const int tail_start = num_complete_blocks * compress_ratio;
                    const int tail_len = cached_len - tail_start;
                    if (lid < tail_len) {
                        out_ptr[block_topk * compress_ratio + lid] = tail_start + lid;
                    }
                });
        });

        const int tail_len = cached_len - (num_complete_blocks * compress_ratio);
        return block_topk * compress_ratio + tail_len;
    }

    const float* scores_ptr = scores.data_as<float>();
    int* out_ptr = active_tokens_out.data_as<int>();
    constexpr int kWgSize = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> local_cnt_one(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> selected_blocks(sycl::range<1>(std::max(1, block_topk)), cgh);
        sycl::local_accessor<std::uint32_t, 1> cutoff_key(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> remaining_rank(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_id(0);

                // Radix-select the exact kth-largest ordered float key. Every lane
                // scans a disjoint subset; each bit decision is a work-group reduction.
                if (lid == 0) {
                    cutoff_key[0] = 0U;
                    remaining_rank[0] = block_topk;
                }
                item.barrier(sycl::access::fence_space::local_space);
                std::uint32_t prefix_mask = 0U;
                for (int bit = 31; bit >= 0; --bit) {
                    const std::uint32_t bit_mask = std::uint32_t{1} << bit;
                    const std::uint32_t prefix = cutoff_key[0];
                    int count_one = 0;
                    for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if ((key & prefix_mask) == prefix && (key & bit_mask) != 0U) {
                            ++count_one;
                        }
                    }
                    local_cnt_one[lid] = count_one;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) local_cnt_one[lid] += local_cnt_one[lid + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0) {
                        if (local_cnt_one[0] >= remaining_rank[0]) {
                            cutoff_key[0] |= bit_mask;
                        } else {
                            remaining_rank[0] -= local_cnt_one[0];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    prefix_mask |= bit_mask;
                }

                // Thread 0 emits the selected set in ascending block order. Equal
                // scores therefore use the documented block-id tie break.
                if (lid == 0) {
                    const std::uint32_t threshold = cutoff_key[0];
                    int above_count = 0;
                    for (int b = 0; b < num_complete_blocks; ++b) {
                        if (qsa_score_order_key(scores_ptr[b]) > threshold) ++above_count;
                    }
                    int ties_needed = block_topk - above_count;
                    int added = 0;
                    for (int b = 0; b < num_complete_blocks; ++b) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if (key > threshold || (key == threshold && ties_needed > 0)) {
                            selected_blocks[added++] = b;
                            if (key == threshold) --ties_needed;
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Expand selected blocks and append the incomplete trailing block.
                for (int idx = lid; idx < block_topk; idx += kWgSize) {
                    const int blk = selected_blocks[idx];
                    for (int c = 0; c < compress_ratio; ++c) {
                        out_ptr[idx * compress_ratio + c] = blk * compress_ratio + c;
                    }
                }

                // 5. Append trailing tokens
                const int tail_start = num_complete_blocks * compress_ratio;
                const int tail_len = cached_len - tail_start;
                if (lid < tail_len) {
                    out_ptr[block_topk * compress_ratio + lid] = tail_start + lid;
                }
            });
    });

    const int tail_len = cached_len - (num_complete_blocks * compress_ratio);
    return block_topk * compress_ratio + tail_len;
}

int qsa_prepare_causal_query_and_select_tokens(
    Context& ctx,
    const Tensor& indexer_qk_row,
    const Tensor& q_norm_w,
    const Tensor& k_norm_w,
    Tensor& k_cache,
    Tensor& q_out_row,
    Tensor& scores_out,
    Tensor& active_tokens_out,
    int pos,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    Tensor* compressed_k,
    int* num_compressed_blocks) {
    if (pos < 0 || block_topk <= 0 || compress_ratio <= 0) {
        throw std::invalid_argument("qsa_prepare_causal_query_and_select_tokens: invalid position or block geometry");
    }
    const int visible_len = pos + 1;
    qsa_prepare_q_and_cache_k(
        ctx, indexer_qk_row, q_norm_w, k_cache, q_out_row, pos, theta, interleaved,
        pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w);

    const int num_complete_blocks = visible_len / compress_ratio;

    if (compressed_k != nullptr && num_compressed_blocks != nullptr && compressed_k->valid()) {
        if (num_complete_blocks > *num_compressed_blocks) {
            const int blocks_to_compress = num_complete_blocks - *num_compressed_blocks;
            qsa_compress_key_blocks(
                ctx, k_cache, k_norm_w, *compressed_k, *num_compressed_blocks,
                blocks_to_compress, theta, interleaved, pos_t, pos_h, pos_w,
                prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w);
            *num_compressed_blocks = num_complete_blocks;
        }
    }

    if (num_complete_blocks <= block_topk) {
        return 0;
    }

    if (compressed_k != nullptr && num_compressed_blocks != nullptr && compressed_k->valid()) {
        qsa_compute_block_scores_compressed(
            ctx, q_out_row, *compressed_k, scores_out, num_complete_blocks);
    } else {
        qsa_compute_block_scores(
            ctx, q_out_row, k_cache, k_norm_w, scores_out, num_complete_blocks, theta, interleaved,
            pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
            mrope_section_t, mrope_section_h, mrope_section_w);
    }
    return qsa_select_topk_and_expand_tokens(
        ctx, scores_out, num_complete_blocks, block_topk, visible_len, compress_ratio, active_tokens_out);
}

// ============================================================================
// Host reference implementations
// ============================================================================

void qsa_prepare_q_and_cache_k_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    bf16* k_cache,
    bf16* q_out,
    int pos,
    int capacity,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w) {
    (void)mrope_section_t;
    if (pos < 0 || (capacity > 0 && pos >= capacity)) return;
    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kNumHeads = 4;

    const float pos_scalar = static_cast<float>(pos >= prompt_pos_len ? pos + text_pos_delta : pos);

    for (int h = 0; h < kNumHeads; ++h) {
        float sum_sq = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            float v = static_cast<float>(indexer_qk[h * kHeadDim + d]);
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(kHeadDim) + 1e-6f);
        std::vector<float> normed_q(kHeadDim);
        for (int d = 0; d < kHeadDim; ++d) {
            float v = static_cast<float>(indexer_qk[h * kHeadDim + d]);
            normed_q[d] = (v / rms) * static_cast<float>(q_norm_w[d]);
        }
        for (int d = 0; d < kHalfDim; ++d) {
            float pos_val = calculate_rotary_position(
                pos, d, pos_scalar, interleaved, pos_t, pos_h, pos_w,
                prompt_pos_len, mrope_section_h, mrope_section_w);
            float freq = 1.0f / std::pow(theta, (2.0f * d) / static_cast<float>(kRotaryDim));
            float angle = pos_val * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float q0 = normed_q[d];
            float q1 = normed_q[d + kHalfDim];
            q_out[h * kHeadDim + d] = bf16(q0 * c - q1 * s);
            q_out[h * kHeadDim + d + kHalfDim] = bf16(q1 * c + q0 * s);
        }
        for (int d = kRotaryDim; d < kHeadDim; ++d) {
            q_out[h * kHeadDim + d] = bf16(normed_q[d]);
        }
    }

    for (int d = 0; d < kHeadDim; ++d) {
        k_cache[pos * kHeadDim + d] = indexer_qk[kNumHeads * kHeadDim + d];
    }
}

void qsa_compute_block_scores_host(
    const bf16* q,
    const bf16* k_cache,
    const bf16* k_norm_w,
    float* scores_out,
    int num_complete_blocks,
    int capacity,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w) {
    (void)capacity;
    (void)mrope_section_t;
    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kNumHeads = 4;

    for (int b = 0; b < num_complete_blocks; ++b) {
        std::vector<float> pooled_k(kHeadDim);
        float sum_sq = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            float k0 = static_cast<float>(k_cache[(4 * b + 0) * kHeadDim + d]);
            float k1 = static_cast<float>(k_cache[(4 * b + 1) * kHeadDim + d]);
            float k2 = static_cast<float>(k_cache[(4 * b + 2) * kHeadDim + d]);
            float k3 = static_cast<float>(k_cache[(4 * b + 3) * kHeadDim + d]);
            float pk = static_cast<float>(bf16((k0 + k1 + k2 + k3) * 0.25f));
            pooled_k[d] = pk;
            sum_sq += pk * pk;
        }

        float k_rms = std::sqrt(sum_sq / static_cast<float>(kHeadDim) + 1e-6f);
        std::vector<float> normed_k(kHeadDim);
        for (int d = 0; d < kHeadDim; ++d) {
            normed_k[d] = (pooled_k[d] / k_rms) * static_cast<float>(k_norm_w[d]);
        }

        const int blk_pos = 4 * b;
        const float pos_scalar = static_cast<float>(blk_pos >= prompt_pos_len ? blk_pos + text_pos_delta : blk_pos);
        for (int d = 0; d < kHalfDim; ++d) {
            float pos_val = calculate_rotary_position(
                blk_pos, d, pos_scalar, interleaved, pos_t, pos_h, pos_w,
                prompt_pos_len, mrope_section_h, mrope_section_w);
            float freq = 1.0f / std::pow(theta, (2.0f * d) / static_cast<float>(kRotaryDim));
            float angle = pos_val * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float kv0 = normed_k[d];
            float kv1 = normed_k[d + kHalfDim];
            normed_k[d] = kv0 * c - kv1 * s;
            normed_k[d + kHalfDim] = kv1 * c + kv0 * s;
        }

        float sum_relu_dot = 0.0f;
        for (int h = 0; h < kNumHeads; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                dot += static_cast<float>(q[h * kHeadDim + d]) * normed_k[d];
            }
            sum_relu_dot += std::max(0.0f, dot);
        }
        constexpr float inv_sqrt_d = 0.08838834764831845f;
        scores_out[b] = sum_relu_dot * inv_sqrt_d;
    }
}

int qsa_select_topk_and_expand_tokens_host(
    const float* scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    int* active_tokens_out) {
    if (cached_len <= 0 || block_topk <= 0) return 0;
    if (num_complete_blocks <= block_topk) {
        for (int i = 0; i < cached_len; ++i) {
            active_tokens_out[i] = i;
        }
        return cached_len;
    }

    if (num_complete_blocks == block_topk + 1) {
        // R4c: Complement selection for B = K + 1 (513 -> 512).
        std::uint32_t worst_key = 0xffffffffU;
        int worst_id = -1;
        for (int i = 0; i < num_complete_blocks; ++i) {
            const std::uint32_t key = qsa_score_order_key(scores[i]);
            if (worst_id < 0 || qsa_candidate_is_worse(key, i, worst_key, worst_id)) {
                worst_key = key;
                worst_id = i;
            }
        }
        for (int i = 0; i < block_topk; ++i) {
            int b = (i < worst_id) ? i : (i + 1);
            for (int c = 0; c < compress_ratio; ++c) {
                active_tokens_out[i * compress_ratio + c] = b * compress_ratio + c;
            }
        }
        int tail_start = num_complete_blocks * compress_ratio;
        int tail_len = cached_len - tail_start;
        for (int i = 0; i < tail_len; ++i) {
            active_tokens_out[block_topk * compress_ratio + i] = tail_start + i;
        }
        return block_topk * compress_ratio + tail_len;
    }

    std::vector<std::pair<float, int>> items(num_complete_blocks);
    for (int i = 0; i < num_complete_blocks; ++i) {
        items[i] = {scores[i], i};
    }

    std::partial_sort(
        items.begin(), items.begin() + block_topk, items.end(),
        [](const auto& a, const auto& b) {
            const std::uint32_t a_key = qsa_score_order_key(a.first);
            const std::uint32_t b_key = qsa_score_order_key(b.first);
            if (a_key != b_key) return a_key > b_key;
            return a.second < b.second;
        });

    std::vector<int> selected_blocks(block_topk);
    for (int i = 0; i < block_topk; ++i) {
        selected_blocks[i] = items[i].second;
    }
    std::sort(selected_blocks.begin(), selected_blocks.end());

    for (int i = 0; i < block_topk; ++i) {
        int b = selected_blocks[i];
        for (int c = 0; c < compress_ratio; ++c) {
            active_tokens_out[i * compress_ratio + c] = b * compress_ratio + c;
        }
    }

    int tail_start = num_complete_blocks * compress_ratio;
    int tail_len = cached_len - tail_start;
    for (int i = 0; i < tail_len; ++i) {
        active_tokens_out[block_topk * compress_ratio + i] = tail_start + i;
    }

    return block_topk * compress_ratio + tail_len;
}

void qsa_compute_block_scores_upstream_ref(
    const bf16* q,
    const bf16* k_cache,
    const bf16* k_norm_w_raw,
    float* scores_out,
    int num_complete_blocks,
    float theta,
    float eps) {
    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kNumHeads = 4;
    constexpr float inv_sqrt_d = 0.08838834764831845f;

    std::vector<float> inv_freq(kHalfDim);
    for (int d = 0; d < kHalfDim; ++d) {
        inv_freq[d] = 1.0f / std::pow(theta, (2.0f * d) / static_cast<float>(kRotaryDim));
    }

    for (int b = 0; b < num_complete_blocks; ++b) {
        // 1. Mean pool 4 tokens into bf16
        std::vector<bf16> pooled_k(kHeadDim);
        float sum_sq = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            float k0 = static_cast<float>(k_cache[(4 * b + 0) * kHeadDim + d]);
            float k1 = static_cast<float>(k_cache[(4 * b + 1) * kHeadDim + d]);
            float k2 = static_cast<float>(k_cache[(4 * b + 2) * kHeadDim + d]);
            float k3 = static_cast<float>(k_cache[(4 * b + 3) * kHeadDim + d]);
            bf16 pk = bf16((k0 + k1 + k2 + k3) * 0.25f);
            pooled_k[d] = pk;
            float pk_f = static_cast<float>(pk);
            sum_sq += pk_f * pk_f;
        }

        // 2. RMSNorm with (1.0 + w_raw), output rounded to bf16
        float k_rsqrt = 1.0f / std::sqrt(sum_sq / static_cast<float>(kHeadDim) + eps);
        std::vector<bf16> normed_k(kHeadDim);
        for (int d = 0; d < kHeadDim; ++d) {
            float norm_val = static_cast<float>(pooled_k[d]) * k_rsqrt * (1.0f + static_cast<float>(k_norm_w_raw[d]));
            normed_k[d] = bf16(norm_val);
        }

        // 3. RoPE at position 4 * b, rotated in bf16
        const float blk_pos = static_cast<float>(4 * b);
        std::vector<bf16> rotated_k = normed_k;
        for (int d = 0; d < kHalfDim; ++d) {
            float angle = blk_pos * inv_freq[d];
            bf16 c = bf16(std::cos(angle));
            bf16 s = bf16(std::sin(angle));
            bf16 x1 = normed_k[d];
            bf16 x2 = normed_k[d + kHalfDim];
            rotated_k[d] = bf16(static_cast<float>(x1) * static_cast<float>(c) - static_cast<float>(x2) * static_cast<float>(s));
            rotated_k[d + kHalfDim] = bf16(static_cast<float>(x2) * static_cast<float>(c) + static_cast<float>(x1) * static_cast<float>(s));
        }

        // 4. Dot product with Q in float, relu, head sum, divide by sqrt(128)
        float sum_relu_dot = 0.0f;
        for (int h = 0; h < kNumHeads; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                dot += static_cast<float>(q[h * kHeadDim + d]) * static_cast<float>(rotated_k[d]);
            }
            sum_relu_dot += std::max(0.0f, dot);
        }
        scores_out[b] = sum_relu_dot * inv_sqrt_d;
    }
}

namespace {

template <bool GraphMode>
void qsa_compress_key_blocks_impl(
    Context& ctx,
    const Tensor& k_cache,
    const Tensor& k_norm_w,
    Tensor& compressed_k,
    int start_block,
    int num_blocks,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    GraphControl control,
    int layer,
    int k_cache_base_token = 0) {
    (void)mrope_section_t;
    if (!GraphMode && num_blocks <= 0) return;
    if (GraphMode && !control) {
        throw std::invalid_argument(
            "graph QSA compression requires GraphControl");
    }
    const bf16* k_cache_ptr = k_cache.data_as<bf16>();
    const bf16* kn_ptr = k_norm_w.data_as<bf16>();
    bf16* comp_k_ptr = compressed_k.data_as<bf16>();

    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kWgSize = 128;
    const int max_blocks =
        static_cast<int>(compressed_k.numel() / kHeadDim);
    const int cache_capacity = static_cast<int>(k_cache.shape(0));
    const int k_cache_base_block = GraphMode ? 0 : (k_cache_base_token / 4);
    if constexpr (!GraphMode) {
        if (start_block < k_cache_base_block || num_blocks > max_blocks - start_block ||
            static_cast<size_t>(start_block - k_cache_base_block + num_blocks - 1) * 4 + 3 >=
                static_cast<size_t>(cache_capacity))
            throw std::invalid_argument("QSA compression block range is invalid");
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<float, 1> local_k(sycl::range<1>(kHeadDim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(
                static_cast<size_t>(GraphMode ? 1 : num_blocks) * kWgSize,
                kWgSize),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_id(0);
                int b = start_block + item.get_group(0);
                bool active = true;
                if constexpr (GraphMode) {
                    int visible_len = 0;
                    int current_blocks = 0;
                    if (!graph_control_pending_blocks(
                            control, max_blocks, visible_len,
                            current_blocks, layer))
                        return;
                    const int target_blocks = visible_len / 4;
                    if (current_blocks == target_blocks) {
                        active = false;
                    } else {
                        b = current_blocks;
                    }
                }
                if constexpr (GraphMode) {
                    if (active &&
                        (b < 0 || b >= max_blocks ||
                         static_cast<size_t>(b) * 4 + 3 >=
                             static_cast<size_t>(cache_capacity))) {
                        active = false;
                        if (lid == 0) graph_control_mark_abort(control);
                    }
                }
                if (!active) return;

                // 1. Mean-pool 4 tokens
                const int src_b = b - k_cache_base_block;
                const size_t b0 = (static_cast<size_t>(4 * src_b + 0) * kHeadDim) + lid;
                const size_t b1 = (static_cast<size_t>(4 * src_b + 1) * kHeadDim) + lid;
                const size_t b2 = (static_cast<size_t>(4 * src_b + 2) * kHeadDim) + lid;
                const size_t b3 = (static_cast<size_t>(4 * src_b + 3) * kHeadDim) + lid;

                const float k0 = static_cast<float>(k_cache_ptr[b0]);
                const float k1 = static_cast<float>(k_cache_ptr[b1]);
                const float k2 = static_cast<float>(k_cache_ptr[b2]);
                const float k3 = static_cast<float>(k_cache_ptr[b3]);
                const float k_pooled_raw = (k0 + k1 + k2 + k3) * 0.25f;
                const float k_pooled = static_cast<float>(bf16(k_pooled_raw));

                // 2. RMSNorm across 128 dims
                local_sum[lid] = k_pooled * k_pooled;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float k_rms = sycl::sqrt(local_sum[0] / static_cast<float>(kHeadDim) + 1e-6f);
                const float k_normed = (k_pooled / k_rms) * static_cast<float>(kn_ptr[lid]);
                local_k[lid] = k_normed;
                item.barrier(sycl::access::fence_space::local_space);

                // 3. RoPE at group_starts = 4 * b
                const int blk_pos = 4 * b;
                const float pos_scalar = static_cast<float>(blk_pos >= prompt_pos_len ? blk_pos + text_pos_delta : blk_pos);
                if (lid < kHalfDim) {
                    const float pos_val = calculate_rotary_position(
                        blk_pos, lid, pos_scalar, interleaved, pos_t, pos_h, pos_w,
                        prompt_pos_len, mrope_section_h, mrope_section_w);
                    const float freq = 1.0f / sycl::pow(theta, (2.0f * lid) / static_cast<float>(kRotaryDim));
                    const float angle = pos_val * freq;
                    const float c = sycl::native::cos(angle);
                    const float s = sycl::native::sin(angle);
                    const float kv0 = local_k[lid];
                    const float kv1 = local_k[lid + kHalfDim];
                    comp_k_ptr[static_cast<size_t>(b) * kHeadDim + lid] = bf16(kv0 * c - kv1 * s);
                    comp_k_ptr[static_cast<size_t>(b) * kHeadDim + lid + kHalfDim] = bf16(kv1 * c + kv0 * s);
                } else if (lid >= kRotaryDim) {
                    comp_k_ptr[static_cast<size_t>(b) * kHeadDim + lid] = bf16(local_k[lid]);
                }
                if constexpr (GraphMode) {
                    item.barrier(
                        sycl::access::fence_space::global_and_local);
                    if (lid == 0)
                        control[graph_control_layer_blocks_slot(layer)] = b + 1;
                }
            });
    });
}

} // namespace

void qsa_compress_key_blocks(
    Context& ctx,
    const Tensor& k_cache,
    const Tensor& k_norm_w,
    Tensor& compressed_k,
    int start_block,
    int num_blocks,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    int k_cache_base_token) {
    qsa_compress_key_blocks_impl<false>(
        ctx, k_cache, k_norm_w, compressed_k, start_block, num_blocks,
        theta, interleaved, pos_t, pos_h, pos_w, prompt_pos_len,
        text_pos_delta, mrope_section_t, mrope_section_h,
        mrope_section_w, nullptr, 0, k_cache_base_token);
}

void qsa_compress_one_completed_block_graph(
    Context& ctx,
    const Tensor& k_cache,
    const Tensor& k_norm_w,
    Tensor& compressed_k,
    GraphControl control,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    int layer) {
    if (!control) {
        throw std::invalid_argument(
            "qsa_compress_one_completed_block_graph requires GraphControl");
    }
    if (graph_control_layer_blocks_slot(layer) < 0) {
        throw std::invalid_argument(
            "qsa_compress_one_completed_block_graph layer out of range");
    }
    if (!k_cache.valid() ||
        k_cache.dtype() != dnnl::memory::data_type::bf16 ||
        k_cache.ndim() != 2 ||
        k_cache.shape(0) <= 0 ||
        k_cache.shape(0) > std::numeric_limits<int>::max() ||
        k_cache.shape(1) != 128 ||
        !k_norm_w.valid() ||
        k_norm_w.dtype() != dnnl::memory::data_type::bf16 ||
        k_norm_w.numel() < 128 ||
        !compressed_k.valid() ||
        compressed_k.dtype() != dnnl::memory::data_type::bf16 ||
        compressed_k.ndim() != 2 ||
        compressed_k.shape(0) <= 0 ||
        compressed_k.shape(0) > std::numeric_limits<int>::max() ||
        compressed_k.shape(1) != 128) {
        throw std::invalid_argument(
            "qsa_compress_one_completed_block_graph shape mismatch");
    }
    qsa_compress_key_blocks_impl<true>(
        ctx, k_cache, k_norm_w, compressed_k, 0, 1, theta, interleaved,
        pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w, control, layer);
}

void qsa_compress_key_blocks_host(
    const bf16* k_cache,
    const bf16* k_norm_w,
    bf16* compressed_k,
    int start_block,
    int num_blocks,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w) {
    (void)mrope_section_t;
    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;

    for (int idx = 0; idx < num_blocks; ++idx) {
        const int b = start_block + idx;
        std::vector<float> pooled_k(kHeadDim);
        float sum_sq = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            float k0 = static_cast<float>(k_cache[(4 * b + 0) * kHeadDim + d]);
            float k1 = static_cast<float>(k_cache[(4 * b + 1) * kHeadDim + d]);
            float k2 = static_cast<float>(k_cache[(4 * b + 2) * kHeadDim + d]);
            float k3 = static_cast<float>(k_cache[(4 * b + 3) * kHeadDim + d]);
            float pk = static_cast<float>(bf16((k0 + k1 + k2 + k3) * 0.25f));
            pooled_k[d] = pk;
            sum_sq += pk * pk;
        }

        float k_rms = std::sqrt(sum_sq / static_cast<float>(kHeadDim) + 1e-6f);
        std::vector<float> normed_k(kHeadDim);
        for (int d = 0; d < kHeadDim; ++d) {
            normed_k[d] = (pooled_k[d] / k_rms) * static_cast<float>(k_norm_w[d]);
        }

        const int blk_pos = 4 * b;
        const float pos_scalar = static_cast<float>(blk_pos >= prompt_pos_len ? blk_pos + text_pos_delta : blk_pos);
        for (int d = 0; d < kHalfDim; ++d) {
            float pos_val = calculate_rotary_position(
                blk_pos, d, pos_scalar, interleaved, pos_t, pos_h, pos_w,
                prompt_pos_len, mrope_section_h, mrope_section_w);
            float freq = 1.0f / std::pow(theta, (2.0f * d) / static_cast<float>(kRotaryDim));
            float angle = pos_val * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            float kv0 = normed_k[d];
            float kv1 = normed_k[d + kHalfDim];
            compressed_k[static_cast<size_t>(b) * kHeadDim + d] = bf16(kv0 * c - kv1 * s);
            compressed_k[static_cast<size_t>(b) * kHeadDim + d + kHalfDim] = bf16(kv1 * c + kv0 * s);
        }
        for (int d = kRotaryDim; d < kHeadDim; ++d) {
            compressed_k[static_cast<size_t>(b) * kHeadDim + d] = bf16(normed_k[d]);
        }
    }
}

void qsa_compute_block_scores_compressed(
    Context& ctx,
    const Tensor& q,
    const Tensor& compressed_k,
    Tensor& scores_out,
    int num_complete_blocks) {
    if (num_complete_blocks <= 0) return;
    const bf16* q_ptr = q.data_as<bf16>();
    const bf16* comp_k_ptr = compressed_k.data_as<bf16>();
    float* scores_ptr = scores_out.data_as<float>();

    constexpr int kHeadDim = 128;
    constexpr int kWgSize = 128;
    constexpr float inv_sqrt_d = 0.08838834764831845f;

    // Block-native long-context fastpath. Past 514 complete blocks a per-block
    // 128-lane work-group pays more launch and reduction overhead than the
    // block's 512 multiply-adds justify, so one 128-thread work-group owns eight
    // blocks: each 16-lane sub-group reduces one block from sixteen vec8 loads
    // of its compressed key. Q is staged once per work-group in SLM.
    static const bool s_fast_block_scores =
        aila::env::read_flag("AILA_QSA_FAST_BLOCK_SCORES", true);
    constexpr int kFastBlocksPerWg = 8;
    constexpr int kFastSgSize = kWgSize / kFastBlocksPerWg;
    if (num_complete_blocks > 514 && s_fast_block_scores) {
        using kvec8 = sycl::vec<bf16, 8>;
        const int num_groups = (num_complete_blocks + kFastBlocksPerWg - 1) / kFastBlocksPerWg;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_slm(4 * kHeadDim, cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(static_cast<size_t>(num_groups) * kWgSize, kWgSize),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kFastSgSize)]] {
                    const int lid = item.get_local_id(0);
                    for (int i = lid; i < 4 * kHeadDim; i += kWgSize)
                        q_slm[i] = static_cast<float>(q_ptr[i]);
                    item.barrier(sycl::access::fence_space::local_space);

                    const int b = static_cast<int>(item.get_group(0)) * kFastBlocksPerWg +
                        lid / kFastSgSize;
                    if (b >= num_complete_blocks) return;
                    const auto sg = item.get_sub_group();
                    const int lane = static_cast<int>(sg.get_local_linear_id());
                    const int d0 = lane * 8;
                    const kvec8 k_vec = *reinterpret_cast<const kvec8*>(
                        comp_k_ptr + static_cast<size_t>(b) * kHeadDim + d0);
                    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float kl = static_cast<float>(k_vec[j]);
                        const int d = d0 + j;
                        p0 += q_slm[0 * kHeadDim + d] * kl;
                        p1 += q_slm[1 * kHeadDim + d] * kl;
                        p2 += q_slm[2 * kHeadDim + d] * kl;
                        p3 += q_slm[3 * kHeadDim + d] * kl;
                    }
                    const float dot0 = sycl::reduce_over_group(sg, p0, sycl::plus<float>());
                    const float dot1 = sycl::reduce_over_group(sg, p1, sycl::plus<float>());
                    const float dot2 = sycl::reduce_over_group(sg, p2, sycl::plus<float>());
                    const float dot3 = sycl::reduce_over_group(sg, p3, sycl::plus<float>());

                    if (lane == 0 && b < num_complete_blocks) {
                        const float r0 = sycl::max(0.0f, dot0);
                        const float r1 = sycl::max(0.0f, dot1);
                        const float r2 = sycl::max(0.0f, dot2);
                        const float r3 = sycl::max(0.0f, dot3);
                        scores_ptr[b] = (r0 + r1 + r2 + r3) * inv_sqrt_d;
                    }
                });
        });
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_complete_blocks * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int b = item.get_group(0);
                const int lid = item.get_local_id(0);

                const float kl = static_cast<float>(comp_k_ptr[static_cast<size_t>(b) * kHeadDim + lid]);
                const float p0 = static_cast<float>(q_ptr[0 * kHeadDim + lid]) * kl;
                const float p1 = static_cast<float>(q_ptr[1 * kHeadDim + lid]) * kl;
                const float p2 = static_cast<float>(q_ptr[2 * kHeadDim + lid]) * kl;
                const float p3 = static_cast<float>(q_ptr[3 * kHeadDim + lid]) * kl;

                const float dot0 = sycl::reduce_over_group(item.get_group(), p0, sycl::plus<float>());
                const float dot1 = sycl::reduce_over_group(item.get_group(), p1, sycl::plus<float>());
                const float dot2 = sycl::reduce_over_group(item.get_group(), p2, sycl::plus<float>());
                const float dot3 = sycl::reduce_over_group(item.get_group(), p3, sycl::plus<float>());

                if (lid == 0) {
                    const float r0 = sycl::max(0.0f, dot0);
                    const float r1 = sycl::max(0.0f, dot1);
                    const float r2 = sycl::max(0.0f, dot2);
                    const float r3 = sycl::max(0.0f, dot3);
                    scores_ptr[b] = (r0 + r1 + r2 + r3) * inv_sqrt_d;
                }
            });
    });
}

void qsa_compute_block_scores_compressed_graph(
    Context& ctx,
    const Tensor& q,
    const Tensor& compressed_k,
    Tensor& scores_out,
    GraphControl control,
    int max_blocks,
    int layer) {
    if (!control || max_blocks <= 0 ||
        graph_control_layer_blocks_slot(layer) < 0)
        throw std::invalid_argument(
            "qsa_compute_block_scores_compressed_graph requires valid control, max_blocks and layer");
    if (!q.valid() || q.dtype() != dnnl::memory::data_type::bf16 ||
        q.numel() < 4 * 128 ||
        !compressed_k.valid() ||
        compressed_k.dtype() != dnnl::memory::data_type::bf16 ||
        compressed_k.numel() < int64_t(max_blocks) * 128 ||
        !scores_out.valid() ||
        scores_out.dtype() != dnnl::memory::data_type::f32 ||
        scores_out.numel() < max_blocks) {
        throw std::invalid_argument(
            "qsa_compute_block_scores_compressed_graph shape mismatch");
    }
    const bf16* q_ptr = q.data_as<bf16>();
    const bf16* comp_k_ptr = compressed_k.data_as<bf16>();
    float* scores_ptr = scores_out.data_as<float>();
    int* const control_ptr = control;

    constexpr int kHeadDim = 128;
    constexpr int kWgSize = 128;
    constexpr float inv_sqrt_d = 0.08838834764831845f;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> control_state(
            sycl::range<1>(2), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(max_blocks) * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_id(0);
                if (lid == 0) {
                    int visible_len = 0;
                    int active_blocks = 0;
                    const bool valid = graph_control_consistent_blocks(
                        control_ptr, max_blocks, visible_len,
                        active_blocks, layer);
                    control_state[0] = valid ? 1 : 0;
                    control_state[1] = active_blocks;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (control_state[0] == 0) return;
                const int b = item.get_group(0);
                if (b >= control_state[1]) return;

                const float kl = static_cast<float>(comp_k_ptr[static_cast<size_t>(b) * kHeadDim + lid]);
                const float p0 = static_cast<float>(q_ptr[0 * kHeadDim + lid]) * kl;
                const float p1 = static_cast<float>(q_ptr[1 * kHeadDim + lid]) * kl;
                const float p2 = static_cast<float>(q_ptr[2 * kHeadDim + lid]) * kl;
                const float p3 = static_cast<float>(q_ptr[3 * kHeadDim + lid]) * kl;

                const float dot0 = sycl::reduce_over_group(item.get_group(), p0, sycl::plus<float>());
                const float dot1 = sycl::reduce_over_group(item.get_group(), p1, sycl::plus<float>());
                const float dot2 = sycl::reduce_over_group(item.get_group(), p2, sycl::plus<float>());
                const float dot3 = sycl::reduce_over_group(item.get_group(), p3, sycl::plus<float>());

                if (lid == 0) {
                    const float r0 = sycl::max(0.0f, dot0);
                    const float r1 = sycl::max(0.0f, dot1);
                    const float r2 = sycl::max(0.0f, dot2);
                    const float r3 = sycl::max(0.0f, dot3);
                    scores_ptr[b] = (r0 + r1 + r2 + r3) * inv_sqrt_d;
                }
            });
    });
}

void qsa_compute_block_scores_compressed_host(
    const bf16* q,
    const bf16* compressed_k,
    float* scores_out,
    int num_complete_blocks) {
    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr float inv_sqrt_d = 0.08838834764831845f;

    for (int b = 0; b < num_complete_blocks; ++b) {
        float sum_relu_dot = 0.0f;
        for (int h = 0; h < kNumHeads; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                dot += static_cast<float>(q[h * kHeadDim + d]) *
                       static_cast<float>(compressed_k[static_cast<size_t>(b) * kHeadDim + d]);
            }
            sum_relu_dot += std::max(0.0f, dot);
        }
        scores_out[b] = sum_relu_dot * inv_sqrt_d;
    }
}

int qsa_prepare_causal_query_and_select_tokens_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    const bf16* k_norm_w,
    bf16* k_cache,
    bf16* q_out,
    float* scores_out,
    int* active_tokens_out,
    int pos,
    int capacity,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    bf16* compressed_k,
    int* num_compressed_blocks) {
    if (pos < 0 || block_topk <= 0 || compress_ratio <= 0) {
        throw std::invalid_argument("qsa_prepare_causal_query_and_select_tokens_host: invalid position or block geometry");
    }
    const int visible_len = pos + 1;
    qsa_prepare_q_and_cache_k_host(
        indexer_qk, q_norm_w, k_cache, q_out, pos, capacity, theta, interleaved,
        pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w);

    const int num_complete_blocks = visible_len / compress_ratio;

    if (compressed_k != nullptr && num_compressed_blocks != nullptr) {
        if (num_complete_blocks > *num_compressed_blocks) {
            const int blocks_to_compress = num_complete_blocks - *num_compressed_blocks;
            qsa_compress_key_blocks_host(
                k_cache, k_norm_w, compressed_k, *num_compressed_blocks,
                blocks_to_compress, theta, interleaved, pos_t, pos_h, pos_w,
                prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w);
            *num_compressed_blocks = num_complete_blocks;
        }
    }

    if (num_complete_blocks <= block_topk) {
        return 0;
    }

    if (compressed_k != nullptr && num_compressed_blocks != nullptr) {
        qsa_compute_block_scores_compressed_host(
            q_out, compressed_k, scores_out, num_complete_blocks);
    } else {
        qsa_compute_block_scores_host(
            q_out, k_cache, k_norm_w, scores_out, num_complete_blocks, capacity, theta, interleaved,
            pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
            mrope_section_t, mrope_section_h, mrope_section_w);
    }
    return qsa_select_topk_and_expand_tokens_host(
        scores_out, num_complete_blocks, block_topk, visible_len, compress_ratio, active_tokens_out);
}

int qsa_select_topk_blocks(
    Context& ctx,
    const Tensor& scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    Tensor& active_blocks_out) {
    if (cached_len <= 0 || block_topk <= 0) return 0;
    const int tail_start = num_complete_blocks * compress_ratio;
    const int tail_len = std::max(0, cached_len - tail_start);
    const int selector = qsa_selector_mode(ctx, block_topk, compress_ratio, num_complete_blocks);
    if (selector >= 2 && selector <= 4 && num_complete_blocks > block_topk + 1 &&
        compress_ratio > 0 && num_complete_blocks == cached_len / compress_ratio) {
        if (selector == 2)
            qsa_radix_select_blocks<4>(ctx, scores.data_as<float>(), active_blocks_out.data_as<int>(),
                1, num_complete_blocks, cached_len - 1, block_topk, compress_ratio, 0, nullptr);
        else if (selector == 3)
            qsa_radix_select_blocks<8>(ctx, scores.data_as<float>(), active_blocks_out.data_as<int>(),
                1, num_complete_blocks, cached_len - 1, block_topk, compress_ratio, 0, nullptr);
        else
            qsa_radix_select_blocks<8, 4>(ctx, scores.data_as<float>(), active_blocks_out.data_as<int>(),
                1, num_complete_blocks, cached_len - 1, block_topk, compress_ratio, 0, nullptr);
        return block_topk;
    }

    if (num_complete_blocks <= block_topk) {
        int* out_ptr = active_blocks_out.data_as<int>();
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::range<1>(std::max(1, num_complete_blocks + tail_len)),
                [=](sycl::id<1> idx) {
                    const int i = static_cast<int>(idx[0]);
                    if (i < num_complete_blocks) {
                        out_ptr[i] = i;
                    } else if (i < num_complete_blocks + tail_len) {
                        out_ptr[i] = tail_start + (i - num_complete_blocks);
                    }
                });
        });
        return num_complete_blocks;
    }

    if (num_complete_blocks == block_topk + 1) {
        // R4c: Complement selection for B = K + 1 (513 -> 512).
        int* out_ptr = active_blocks_out.data_as<int>();
        const float* scores_ptr = scores.data_as<float>();
        constexpr int kWgSize = 256;

        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<std::uint32_t, 1> local_worst_key(sycl::range<1>(kWgSize), cgh);
            sycl::local_accessor<int, 1> local_worst_id(sycl::range<1>(kWgSize), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(kWgSize, kWgSize),
                [=](sycl::nd_item<1> item) {
                    const int lid = item.get_local_id(0);

                    std::uint32_t my_worst_key = 0xffffffffU;
                    int my_worst_id = -1;
                    for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if (my_worst_id < 0 || qsa_candidate_is_worse(key, b, my_worst_key, my_worst_id)) {
                            my_worst_key = key;
                            my_worst_id = b;
                        }
                    }
                    local_worst_key[lid] = my_worst_key;
                    local_worst_id[lid] = my_worst_id;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            const std::uint32_t other_key = local_worst_key[lid + stride];
                            const int other_id = local_worst_id[lid + stride];
                            if (other_id >= 0) {
                                if (local_worst_id[lid] < 0 || qsa_candidate_is_worse(other_key, other_id, local_worst_key[lid], local_worst_id[lid])) {
                                    local_worst_key[lid] = other_key;
                                    local_worst_id[lid] = other_id;
                                }
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    const int worst_id = local_worst_id[0];

                    for (int idx = lid; idx < block_topk; idx += kWgSize) {
                        out_ptr[idx] = (idx < worst_id) ? idx : (idx + 1);
                    }

                    if (lid < tail_len) {
                        out_ptr[block_topk + lid] = tail_start + lid;
                    }
                });
        });

        return block_topk;
    }

    const float* scores_ptr = scores.data_as<float>();
    int* out_ptr = active_blocks_out.data_as<int>();
    constexpr int kWgSize = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> local_cnt_one(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<std::uint32_t, 1> cutoff_key(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> remaining_rank(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_id(0);

                if (lid == 0) {
                    cutoff_key[0] = 0U;
                    remaining_rank[0] = block_topk;
                }
                item.barrier(sycl::access::fence_space::local_space);
                std::uint32_t prefix_mask = 0U;
                for (int bit = 31; bit >= 0; --bit) {
                    const std::uint32_t bit_mask = std::uint32_t{1} << bit;
                    const std::uint32_t prefix = cutoff_key[0];
                    int count_one = 0;
                    for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if ((key & prefix_mask) == prefix && (key & bit_mask) != 0U) {
                            ++count_one;
                        }
                    }
                    if (selector == 1) {
                        const int count = sycl::reduce_over_group(item.get_group(), count_one, sycl::plus<int>());
                        if (lid == 0) local_cnt_one[0] = count;
                    } else {
                        local_cnt_one[lid] = count_one;
                        item.barrier(sycl::access::fence_space::local_space);
                        for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                            if (lid < stride) local_cnt_one[lid] += local_cnt_one[lid + stride];
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                    }
                    if (lid == 0) {
                        if (local_cnt_one[0] >= remaining_rank[0]) {
                            cutoff_key[0] |= bit_mask;
                        } else {
                            remaining_rank[0] -= local_cnt_one[0];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    prefix_mask |= bit_mask;
                }

                // Contiguous per-lane intervals make the compaction stable: every
                // lane writes its own run at its prefix offset, so out_ptr stays
                // ascending by block ID without a serial single-lane pass.
                const std::uint32_t threshold = cutoff_key[0];
                const int ties_available = remaining_rank[0];
                const int per = (num_complete_blocks + kWgSize - 1) / kWgSize;
                const int begin = lid * per;
                const int end = sycl::min(begin + per, num_complete_blocks);
                int greater = 0, equal = 0;
                for (int b = begin; b < end; ++b) {
                    const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                    greater += key > threshold;
                    equal += key == threshold;
                }
                const int eq_before = sycl::exclusive_scan_over_group(
                    item.get_group(), equal, sycl::plus<int>());
                int ties = sycl::min(equal, sycl::max(0, ties_available - eq_before));
                int offset = sycl::exclusive_scan_over_group(
                    item.get_group(), greater + ties, sycl::plus<int>());
                for (int b = begin; b < end; ++b) {
                    const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                    if (key > threshold || (key == threshold && ties > 0)) {
                        out_ptr[offset++] = b;
                        if (key == threshold) --ties;
                    }
                }

                if (lid < tail_len) {
                    out_ptr[block_topk + lid] = tail_start + lid;
                }
            });
    });

    return block_topk;
}

namespace {

void qsa_select_topk_blocks_graph_legacy_impl(
    Context& ctx,
    const Tensor& scores,
    Tensor& active_blocks_out,
    GraphControl control,
    int block_topk,
    int compress_ratio,
    int max_blocks,
    int layer) {
    if (!control || block_topk <= 0 || compress_ratio <= 0 ||
        max_blocks <= 0 || graph_control_layer_blocks_slot(layer) < 0)
        throw std::invalid_argument(
            "qsa_select_topk_blocks_graph requires valid graph geometry and layer");
    if (!scores.valid() ||
        scores.dtype() != dnnl::memory::data_type::f32 ||
        scores.numel() < max_blocks ||
        !active_blocks_out.valid() ||
        active_blocks_out.dtype() != dnnl::memory::data_type::s32 ||
        active_blocks_out.numel() < block_topk + compress_ratio) {
        throw std::invalid_argument(
            "qsa_select_topk_blocks_graph shape mismatch");
    }
    const float* scores_ptr = scores.data_as<float>();
    int* out_ptr = active_blocks_out.data_as<int>();
    int* const control_ptr = control;
    constexpr int kWgSize = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> local_cnt_one(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<std::uint32_t, 1> local_worst_key(
            sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> local_worst_id(
            sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> selected_blocks(
            sycl::range<1>(std::max(1, block_topk)), cgh);
        sycl::local_accessor<std::uint32_t, 1> cutoff_key(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> remaining_rank(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_id(0);
                int visible_len = 0;
                int active_blocks = 0;
                if (!graph_control_consistent_blocks(
                        control_ptr, max_blocks, visible_len,
                        active_blocks, layer))
                    return;
                const int tail_start = active_blocks * compress_ratio;
                const int tail_len = sycl::min(
                    compress_ratio, sycl::max(0, visible_len - tail_start));

                // Uniform, device-derived dispatch. No host value selects the path.
                if (active_blocks <= block_topk) {
                    for (int b = lid; b < active_blocks; b += kWgSize) {
                        out_ptr[b] = b;
                    }
                    if (lid < tail_len) {
                        out_ptr[active_blocks + lid] = tail_start + lid;
                    }
                    return;
                }

                if (active_blocks == block_topk + 1) {
                    std::uint32_t my_worst_key = 0xffffffffU;
                    int my_worst_id = -1;
                    for (int b = lid; b < active_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if (my_worst_id < 0 ||
                            qsa_candidate_is_worse(key, b, my_worst_key, my_worst_id)) {
                            my_worst_key = key;
                            my_worst_id = b;
                        }
                    }
                    local_worst_key[lid] = my_worst_key;
                    local_worst_id[lid] = my_worst_id;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            const std::uint32_t other_key = local_worst_key[lid + stride];
                            const int other_id = local_worst_id[lid + stride];
                            if (other_id >= 0 &&
                                (local_worst_id[lid] < 0 ||
                                 qsa_candidate_is_worse(
                                     other_key, other_id,
                                     local_worst_key[lid], local_worst_id[lid]))) {
                                local_worst_key[lid] = other_key;
                                local_worst_id[lid] = other_id;
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    const int worst_id = local_worst_id[0];
                    for (int idx = lid; idx < block_topk; idx += kWgSize) {
                        out_ptr[idx] = (idx < worst_id) ? idx : (idx + 1);
                    }
                    if (lid < tail_len) {
                        out_ptr[block_topk + lid] = tail_start + lid;
                    }
                    return;
                }

                if (lid == 0) {
                    cutoff_key[0] = 0U;
                    remaining_rank[0] = block_topk;
                }
                item.barrier(sycl::access::fence_space::local_space);
                std::uint32_t prefix_mask = 0U;
                for (int bit = 31; bit >= 0; --bit) {
                    const std::uint32_t bit_mask = std::uint32_t{1} << bit;
                    const std::uint32_t prefix = cutoff_key[0];
                    int count_one = 0;
                    for (int b = lid; b < active_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if ((key & prefix_mask) == prefix && (key & bit_mask) != 0U) {
                            ++count_one;
                        }
                    }
                    local_cnt_one[lid] = count_one;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) local_cnt_one[lid] += local_cnt_one[lid + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0) {
                        if (local_cnt_one[0] >= remaining_rank[0]) {
                            cutoff_key[0] |= bit_mask;
                        } else {
                            remaining_rank[0] -= local_cnt_one[0];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    prefix_mask |= bit_mask;
                }

                if (lid == 0) {
                    const std::uint32_t threshold = cutoff_key[0];
                    int above_count = 0;
                    for (int b = 0; b < active_blocks; ++b) {
                        if (qsa_score_order_key(scores_ptr[b]) > threshold) ++above_count;
                    }
                    int ties_needed = block_topk - above_count;
                    int added = 0;
                    for (int b = 0; b < active_blocks; ++b) {
                        const std::uint32_t key = qsa_score_order_key(scores_ptr[b]);
                        if (key > threshold || (key == threshold && ties_needed > 0)) {
                            selected_blocks[added++] = b;
                            if (key == threshold) --ties_needed;
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < block_topk; idx += kWgSize) {
                    out_ptr[idx] = selected_blocks[idx];
                }
                if (lid < tail_len) {
                    out_ptr[block_topk + lid] = tail_start + lid;
                }
            });
    });
}

void qsa_select_topk_blocks_graph_radix_impl(
    Context& ctx,
    const Tensor& scores,
    Tensor& active_blocks_out,
    GraphControl control,
    int block_topk,
    int compress_ratio,
    int max_blocks,
    int layer) {
    if (!control || block_topk <= 0 || compress_ratio <= 0 ||
        max_blocks <= 0 || graph_control_layer_blocks_slot(layer) < 0)
        throw std::invalid_argument(
            "qsa_select_topk_blocks_graph requires valid graph geometry and layer");
    if (!scores.valid() ||
        scores.dtype() != dnnl::memory::data_type::f32 ||
        scores.numel() < max_blocks ||
        !active_blocks_out.valid() ||
        active_blocks_out.dtype() != dnnl::memory::data_type::s32 ||
        active_blocks_out.numel() < block_topk + compress_ratio) {
        throw std::invalid_argument(
            "qsa_select_topk_blocks_graph shape mismatch");
    }

    const float* scores_ptr = scores.data_as<float>();
    int* out_ptr = active_blocks_out.data_as<int>();
    int* const control_ptr = control;
    constexpr int kWgSize = 256;
    constexpr int kBits = 8;
    constexpr int kBins = 1 << kBits;
    constexpr int kSubgroups = kWgSize / 16;
    constexpr int kLoadWidth = 4;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> geometry(
            sycl::range<1>(4), cgh);
        sycl::local_accessor<int, 1> hist(
            sycl::range<1>(kSubgroups * kBins), cgh);
        sycl::local_accessor<std::uint32_t, 1> local_worst_key(
            sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> local_worst_id(
            sycl::range<1>(kWgSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(kWgSize, kWgSize),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(16)]] {
                const int lid = item.get_local_id(0);
                if (lid == 0) {
                    int visible_len = 0;
                    int active_blocks = 0;
                    const bool valid = graph_control_consistent_blocks(
                        control_ptr, max_blocks, visible_len,
                        active_blocks, layer);
                    geometry[0] = valid ? 1 : 0;
                    geometry[1] = visible_len;
                    geometry[2] = active_blocks;
                    geometry[3] = valid
                        ? sycl::min(
                              compress_ratio,
                              sycl::max(
                                  0, visible_len -
                                         active_blocks * compress_ratio))
                        : 0;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (geometry[0] == 0) return;

                const int active_blocks = geometry[2];
                const int tail_len = geometry[3];
                const int tail_start = active_blocks * compress_ratio;

                if (active_blocks <= block_topk) {
                    for (int b = lid; b < active_blocks; b += kWgSize) {
                        out_ptr[b] = b;
                    }
                    if (lid < tail_len) {
                        out_ptr[active_blocks + lid] = tail_start + lid;
                    }
                    return;
                }

                if (active_blocks == block_topk + 1) {
                    std::uint32_t my_worst_key = 0xffffffffU;
                    int my_worst_id = -1;
                    for (int b = lid; b < active_blocks; b += kWgSize) {
                        const std::uint32_t key =
                            qsa_score_order_key(scores_ptr[b]);
                        if (my_worst_id < 0 ||
                            qsa_candidate_is_worse(
                                key, b, my_worst_key, my_worst_id)) {
                            my_worst_key = key;
                            my_worst_id = b;
                        }
                    }
                    local_worst_key[lid] = my_worst_key;
                    local_worst_id[lid] = my_worst_id;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            const std::uint32_t other_key =
                                local_worst_key[lid + stride];
                            const int other_id =
                                local_worst_id[lid + stride];
                            if (other_id >= 0 &&
                                (local_worst_id[lid] < 0 ||
                                 qsa_candidate_is_worse(
                                     other_key, other_id,
                                     local_worst_key[lid],
                                     local_worst_id[lid]))) {
                                local_worst_key[lid] = other_key;
                                local_worst_id[lid] = other_id;
                            }
                        }
                        item.barrier(
                            sycl::access::fence_space::local_space);
                    }

                    const int worst_id = local_worst_id[0];
                    for (int idx = lid; idx < block_topk;
                         idx += kWgSize) {
                        out_ptr[idx] =
                            (idx < worst_id) ? idx : (idx + 1);
                    }
                    if (lid < tail_len) {
                        out_ptr[block_topk + lid] = tail_start + lid;
                    }
                    return;
                }

                if (lid < tail_len) {
                    out_ptr[block_topk + lid] = tail_start + lid;
                }

                std::uint32_t prefix = 0;
                std::uint32_t mask = 0;
                int rank = block_topk;
                const auto group = item.get_group();
                for (int shift = 32 - kBits; shift >= 0;
                     shift -= kBits) {
                    for (int i = lid; i < kSubgroups * kBins;
                         i += kWgSize) {
                        hist[i] = 0;
                    }
                    item.barrier(
                        sycl::access::fence_space::local_space);

                    for (int b = lid * kLoadWidth; b < active_blocks;
                         b += kWgSize * kLoadWidth) {
                        sycl::vec<float, kLoadWidth> values;
                        if (b + kLoadWidth <= active_blocks) {
                            values.load(
                                0,
                                sycl::address_space_cast<
                                    sycl::access::address_space::
                                        global_space,
                                    sycl::access::decorated::no>(
                                    scores_ptr + b));
                        } else {
                            for (int j = 0; j < kLoadWidth; ++j) {
                                values[j] = b + j < active_blocks
                                    ? scores_ptr[b + j]
                                    : 0.0f;
                            }
                        }
                        for (int j = 0; j < kLoadWidth; ++j) {
                            if (b + j >= active_blocks) continue;
                            const std::uint32_t key =
                                qsa_score_order_key(values[j]);
                            if ((key & mask) != prefix) continue;
                            const int bin =
                                (key >> shift) & (kBins - 1);
                            sycl::atomic_ref<
                                int, sycl::memory_order::relaxed,
                                sycl::memory_scope::work_group,
                                sycl::access::address_space::
                                    local_space>
                                count(
                                    hist[(lid / 16) * kBins + bin]);
                            count.fetch_add(1);
                        }
                    }
                    item.barrier(
                        sycl::access::fence_space::local_space);

                    int count = 0;
                    if (lid < kBins) {
                        for (int sg = 0; sg < kSubgroups; ++sg) {
                            count += hist[
                                sg * kBins + (kBins - 1 - lid)];
                        }
                    }
                    const int inclusive =
                        sycl::inclusive_scan_over_group(
                            group, count, sycl::plus<int>());
                    const int before = inclusive - count;
                    const bool cutoff =
                        lid < kBins && before < rank &&
                        inclusive >= rank;
                    const int digit = sycl::reduce_over_group(
                        group, cutoff ? kBins - 1 - lid : 0,
                        sycl::maximum<int>());
                    const int above = sycl::reduce_over_group(
                        group, cutoff ? before : 0,
                        sycl::plus<int>());
                    prefix |=
                        std::uint32_t(digit) << shift;
                    mask |= std::uint32_t(kBins - 1) << shift;
                    rank -= above;
                    item.barrier(
                        sycl::access::fence_space::local_space);
                }

                const int per =
                    (active_blocks + kWgSize - 1) / kWgSize;
                const int begin = lid * per;
                const int end =
                    sycl::min(begin + per, active_blocks);
                int greater = 0;
                int equal = 0;
                for (int b = begin; b < end; ++b) {
                    const auto key = qsa_score_order_key(scores_ptr[b]);
                    greater += key > prefix;
                    equal += key == prefix;
                }
                const int eq_before =
                    sycl::exclusive_scan_over_group(
                        group, equal, sycl::plus<int>());
                int ties = sycl::min(
                    equal, sycl::max(0, rank - eq_before));
                int offset = sycl::exclusive_scan_over_group(
                    group, greater + ties, sycl::plus<int>());
                for (int b = begin; b < end; ++b) {
                    const auto key = qsa_score_order_key(scores_ptr[b]);
                    if (key > prefix ||
                        (key == prefix && ties > 0)) {
                        out_ptr[offset++] = b;
                        if (key == prefix) --ties;
                    }
                }
            });
    });
}

} // anonymous namespace

void qsa_select_topk_blocks_graph(
    Context& ctx,
    const Tensor& scores,
    Tensor& active_blocks_out,
    GraphControl control,
    int block_topk,
    int compress_ratio,
    int max_blocks,
    int layer,
    int selector_mode) {
    if (selector_mode == 0) {
        qsa_select_topk_blocks_graph_legacy_impl(
            ctx, scores, active_blocks_out, control, block_topk,
            compress_ratio, max_blocks, layer);
        return;
    }
    if (selector_mode == 1) {
        qsa_select_topk_blocks_graph_radix_impl(
            ctx, scores, active_blocks_out, control, block_topk,
            compress_ratio, max_blocks, layer);
        return;
    }
    throw std::invalid_argument(
        "qsa_select_topk_blocks_graph selector_mode must be 0 or 1");
}

int qsa_select_topk_blocks_host(
    const float* scores,
    int num_complete_blocks,
    int block_topk,
    int cached_len,
    int compress_ratio,
    int* active_blocks_out) {
    if (cached_len <= 0 || block_topk <= 0) return 0;
    const int tail_start = num_complete_blocks * compress_ratio;
    const int tail_len = std::max(0, cached_len - tail_start);

    if (num_complete_blocks <= block_topk) {
        for (int b = 0; b < num_complete_blocks; ++b) {
            active_blocks_out[b] = b;
        }
        for (int t = 0; t < tail_len; ++t) {
            active_blocks_out[num_complete_blocks + t] = tail_start + t;
        }
        return num_complete_blocks;
    }

    if (num_complete_blocks == block_topk + 1) {
        // R4c: Complement selection for B = K + 1 (513 -> 512).
        std::uint32_t worst_key = 0xffffffffU;
        int worst_id = -1;
        for (int b = 0; b < num_complete_blocks; ++b) {
            const std::uint32_t key = qsa_score_order_key(scores[b]);
            if (worst_id < 0 || qsa_candidate_is_worse(key, b, worst_key, worst_id)) {
                worst_key = key;
                worst_id = b;
            }
        }
        for (int i = 0; i < block_topk; ++i) {
            active_blocks_out[i] = (i < worst_id) ? i : (i + 1);
        }
        for (int t = 0; t < tail_len; ++t) {
            active_blocks_out[block_topk + t] = tail_start + t;
        }
        return block_topk;
    }

    struct Candidate {
        std::uint32_t key;
        int id;
    };
    std::vector<Candidate> cands(num_complete_blocks);
    for (int b = 0; b < num_complete_blocks; ++b) {
        cands[b] = {qsa_score_order_key(scores[b]), b};
    }
    std::nth_element(
        cands.begin(), cands.begin() + block_topk, cands.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.key != b.key) return a.key > b.key;
            return a.id < b.id;
        });
    std::vector<int> selected(block_topk);
    for (int i = 0; i < block_topk; ++i) {
        selected[i] = cands[i].id;
    }
    std::sort(selected.begin(), selected.end());

    for (int i = 0; i < block_topk; ++i) {
        active_blocks_out[i] = selected[i];
    }
    for (int t = 0; t < tail_len; ++t) {
        active_blocks_out[block_topk + t] = tail_start + t;
    }
    return block_topk;
}

int qsa_prepare_causal_query_and_select_blocks(
    Context& ctx,
    const Tensor& indexer_qk_row,
    const Tensor& q_norm_w,
    const Tensor& k_norm_w,
    Tensor& k_cache,
    Tensor& q_out_row,
    Tensor& scores_out,
    Tensor& active_blocks_out,
    int pos,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    Tensor* compressed_k,
    int* num_compressed_blocks,
    int k_cache_base_token) {
    if (pos < 0 || block_topk <= 0 || compress_ratio <= 0) {
        throw std::invalid_argument("qsa_prepare_causal_query_and_select_blocks: invalid position or block geometry");
    }
    const int visible_len = pos + 1;
    qsa_prepare_q_and_cache_k(
        ctx, indexer_qk_row, q_norm_w, k_cache, q_out_row, pos, theta, interleaved,
        pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w, k_cache_base_token);

    const int num_complete_blocks = visible_len / compress_ratio;

    if (compressed_k != nullptr && num_compressed_blocks != nullptr && compressed_k->valid()) {
        if (num_complete_blocks > *num_compressed_blocks) {
            const int blocks_to_compress = num_complete_blocks - *num_compressed_blocks;
            qsa_compress_key_blocks(
                ctx, k_cache, k_norm_w, *compressed_k, *num_compressed_blocks,
                blocks_to_compress, theta, interleaved, pos_t, pos_h, pos_w,
                prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w,
                k_cache_base_token);
            *num_compressed_blocks = num_complete_blocks;
        }
    }

    if (num_complete_blocks <= block_topk) {
        return 0;
    }

    if (compressed_k != nullptr && num_compressed_blocks != nullptr && compressed_k->valid()) {
        qsa_compute_block_scores_compressed(
            ctx, q_out_row, *compressed_k, scores_out, num_complete_blocks);
    } else {
        qsa_compute_block_scores(
            ctx, q_out_row, k_cache, k_norm_w, scores_out, num_complete_blocks, theta, interleaved,
            pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
            mrope_section_t, mrope_section_h, mrope_section_w);
    }
    return qsa_select_topk_blocks(
        ctx, scores_out, num_complete_blocks, block_topk, visible_len, compress_ratio, active_blocks_out);
}

int qsa_prepare_causal_query_and_select_blocks_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    const bf16* k_norm_w,
    bf16* k_cache,
    bf16* q_out,
    float* scores_out,
    int* active_blocks_out,
    int pos,
    int capacity,
    int block_topk,
    int compress_ratio,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    bf16* compressed_k,
    int* num_compressed_blocks) {
    if (pos < 0 || block_topk <= 0 || compress_ratio <= 0) {
        throw std::invalid_argument("qsa_prepare_causal_query_and_select_blocks_host: invalid position or block geometry");
    }
    const int visible_len = pos + 1;
    qsa_prepare_q_and_cache_k_host(
        indexer_qk, q_norm_w, k_cache, q_out, pos, capacity, theta, interleaved,
        pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w);

    const int num_complete_blocks = visible_len / compress_ratio;

    if (compressed_k != nullptr && num_compressed_blocks != nullptr) {
        if (num_complete_blocks > *num_compressed_blocks) {
            const int blocks_to_compress = num_complete_blocks - *num_compressed_blocks;
            qsa_compress_key_blocks_host(
                k_cache, k_norm_w, compressed_k, *num_compressed_blocks,
                blocks_to_compress, theta, interleaved, pos_t, pos_h, pos_w,
                prompt_pos_len, text_pos_delta, mrope_section_t, mrope_section_h, mrope_section_w);
            *num_compressed_blocks = num_complete_blocks;
        }
    }

    if (num_complete_blocks <= block_topk) {
        return 0;
    }

    if (compressed_k != nullptr && num_compressed_blocks != nullptr) {
        qsa_compute_block_scores_compressed_host(
            q_out, compressed_k, scores_out, num_complete_blocks);
    } else {
        qsa_compute_block_scores_host(
            q_out, k_cache, k_norm_w, scores_out, num_complete_blocks, capacity, theta, interleaved,
            pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
            mrope_section_t, mrope_section_h, mrope_section_w);
    }
    return qsa_select_topk_blocks_host(
        scores_out, num_complete_blocks, block_topk, visible_len, compress_ratio, active_blocks_out);
}

void qsa_prepare_q_batched(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    Tensor& q_out,
    int start_pos,
    int count,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w) {
    (void)mrope_section_t;
    if (count <= 0) return;
    const bf16* qk_ptr = indexer_qk.data_as<bf16>();
    const bf16* qn_ptr = q_norm_w.data_as<bf16>();
    bf16* q_out_ptr = q_out.data_as<bf16>();

    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kWgSize = 128;
    constexpr int kNumHeads = 4;
    constexpr int kOutDim = 640;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<float, 1> local_q(sycl::range<1>(kHeadDim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(count) * kNumHeads * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int group = item.get_group(0);
                const int row = group / kNumHeads;
                const int head = group % kNumHeads;
                const int lid = item.get_local_id(0);
                const int pos = start_pos + row;

                const float pos_scalar = static_cast<float>(pos >= prompt_pos_len ? pos + text_pos_delta : pos);
                const size_t q_offset = static_cast<size_t>(row) * kOutDim + head * kHeadDim + lid;
                const size_t out_offset = static_cast<size_t>(row) * (kNumHeads * kHeadDim) + head * kHeadDim;

                const float qv = static_cast<float>(qk_ptr[q_offset]);
                local_sum[lid] = qv * qv;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                const float q_rms = sycl::sqrt(local_sum[0] / static_cast<float>(kHeadDim) + 1e-6f);
                const float q_normed = (qv / q_rms) * static_cast<float>(qn_ptr[lid]);
                local_q[lid] = q_normed;
                item.barrier(sycl::access::fence_space::local_space);

                if (lid < kHalfDim) {
                    const float pos_val = calculate_rotary_position(
                        pos, lid, pos_scalar, interleaved, pos_t, pos_h, pos_w,
                        prompt_pos_len, mrope_section_h, mrope_section_w);
                    const float freq = 1.0f / sycl::pow(theta, (2.0f * lid) / static_cast<float>(kRotaryDim));
                    const float angle = pos_val * freq;
                    const float c = sycl::native::cos(angle);
                    const float s = sycl::native::sin(angle);
                    const float q0 = local_q[lid];
                    const float q1 = local_q[lid + kHalfDim];
                    q_out_ptr[out_offset + lid] = bf16(q0 * c - q1 * s);
                    q_out_ptr[out_offset + lid + kHalfDim] = bf16(q1 * c + q0 * s);
                } else if (lid >= kRotaryDim) {
                    q_out_ptr[out_offset + lid] = bf16(local_q[lid]);
                }
            });
    });
}

void qsa_prepare_q_batched_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    bf16* q_out,
    int start_pos,
    int count,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w) {
    (void)mrope_section_t;
    constexpr int kNumHeads = 4;
    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kOutDim = 640;

    for (int r = 0; r < count; ++r) {
        const int pos = start_pos + r;
        const bf16* row_in = indexer_qk + static_cast<size_t>(r) * kOutDim;
        bf16* row_out = q_out + static_cast<size_t>(r) * (kNumHeads * kHeadDim);
        const float pos_scalar = static_cast<float>(pos >= prompt_pos_len ? pos + text_pos_delta : pos);

        for (int h = 0; h < kNumHeads; ++h) {
            float sum_sq = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                float v = static_cast<float>(row_in[h * kHeadDim + d]);
                sum_sq += v * v;
            }
            float rms = std::sqrt(sum_sq / static_cast<float>(kHeadDim) + 1e-6f);
            std::vector<float> normed_q(kHeadDim);
            for (int d = 0; d < kHeadDim; ++d) {
                float v = static_cast<float>(row_in[h * kHeadDim + d]);
                normed_q[d] = (v / rms) * static_cast<float>(q_norm_w[d]);
            }
            for (int d = 0; d < kHalfDim; ++d) {
                float pos_val = calculate_rotary_position(
                    pos, d, pos_scalar, interleaved, pos_t, pos_h, pos_w,
                    prompt_pos_len, mrope_section_h, mrope_section_w);
                float freq = 1.0f / std::pow(theta, (2.0f * d) / static_cast<float>(kRotaryDim));
                float angle = pos_val * freq;
                float c = std::cos(angle);
                float s = std::sin(angle);
                float q0 = normed_q[d];
                float q1 = normed_q[d + kHalfDim];
                row_out[h * kHeadDim + d] = bf16(q0 * c - q1 * s);
                row_out[h * kHeadDim + d + kHalfDim] = bf16(q1 * c + q0 * s);
            }
            for (int d = kRotaryDim; d < kHeadDim; ++d) {
                row_out[h * kHeadDim + d] = bf16(normed_q[d]);
            }
        }
    }
}

void qsa_compute_block_scores_compressed_batched(
    Context& ctx,
    const Tensor& q,
    const Tensor& compressed_k,
    Tensor& scores_out,
    int query_count,
    int max_complete_blocks,
    int compress_ratio,
    int start_pos) {
    if (query_count <= 0 || max_complete_blocks <= 0) return;
    const bf16* q_ptr = q.data_as<bf16>();
    const bf16* comp_k_ptr = compressed_k.data_as<bf16>();
    float* scores_ptr = scores_out.data_as<float>();

    constexpr int kHeadDim = 128;
    constexpr int kWgSize = 128;
    constexpr int kNumHeads = 4;
    constexpr float inv_sqrt_d = 0.08838834764831845f;

    static const bool s_fast_scores = aila::env::read_flag("AILA_QSA_FAST_BLOCK_SCORES", true);
    if (s_fast_scores) {
        // 8 blocks per 128-thread work-group (8 sub-groups of 16 lanes):
        // Q[4, 128] is staged into SLM once per 8 blocks (8x fewer Q global loads),
        // compressed_k is loaded via vec<bf16, 8>, and each sub-group reduces its block
        // via pure register shuffles (zero work-group reduction barriers).
        constexpr int kBlocksPerWg = 8;
        const int block_tiles = (max_complete_blocks + kBlocksPerWg - 1) / kBlocksPerWg;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_slm(sycl::range<1>(kNumHeads * kHeadDim), cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(static_cast<size_t>(query_count) * block_tiles * kWgSize, kWgSize),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg = item.get_sub_group();
                    const int group = item.get_group(0);
                    const int query = group / block_tiles;
                    const int b_tile = group % block_tiles;
                    const int lid = item.get_local_id(0);
                    const int sg_id = lid >> 4;
                    const int lane = lid & 15;
                    const int b = b_tile * kBlocksPerWg + sg_id;

                    const int pos = start_pos + query;
                    const int visible_len = pos + 1;
                    const int num_complete_blocks = visible_len / compress_ratio;
                    const size_t q_base = static_cast<size_t>(query) * (kNumHeads * kHeadDim);

                    #pragma unroll
                    for (int h = 0; h < kNumHeads; ++h) {
                        q_slm[h * kHeadDim + lid] = static_cast<float>(q_ptr[q_base + h * kHeadDim + lid]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (b >= max_complete_blocks) return;
                    if (b >= num_complete_blocks) {
                        if (lane == 0) {
                            scores_ptr[static_cast<size_t>(query) * max_complete_blocks + b] = -1e30f;
                        }
                        return;
                    }

                    const int d_base = lane * 8;
                    const auto kv8 = *reinterpret_cast<const sycl::vec<bf16, 8>*>(
                        comp_k_ptr + static_cast<size_t>(b) * kHeadDim + d_base);
                    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const float kl = static_cast<float>(kv8[u]);
                        const int d = d_base + u;
                        p0 += q_slm[0 * kHeadDim + d] * kl;
                        p1 += q_slm[1 * kHeadDim + d] * kl;
                        p2 += q_slm[2 * kHeadDim + d] * kl;
                        p3 += q_slm[3 * kHeadDim + d] * kl;
                    }

                    const float dot0 = sycl::reduce_over_group(sg, p0, sycl::plus<float>());
                    const float dot1 = sycl::reduce_over_group(sg, p1, sycl::plus<float>());
                    const float dot2 = sycl::reduce_over_group(sg, p2, sycl::plus<float>());
                    const float dot3 = sycl::reduce_over_group(sg, p3, sycl::plus<float>());

                    if (lane == 0) {
                        const float r0 = sycl::fmax(0.0f, dot0);
                        const float r1 = sycl::fmax(0.0f, dot1);
                        const float r2 = sycl::fmax(0.0f, dot2);
                        const float r3 = sycl::fmax(0.0f, dot3);
                        scores_ptr[static_cast<size_t>(query) * max_complete_blocks + b] =
                            (r0 + r1 + r2 + r3) * inv_sqrt_d;
                    }
                });
        });
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(query_count) * max_complete_blocks * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int group = item.get_group(0);
                const int query = group / max_complete_blocks;
                const int b = group % max_complete_blocks;
                const int lid = item.get_local_id(0);

                const int pos = start_pos + query;
                const int visible_len = pos + 1;
                const int num_complete_blocks = visible_len / compress_ratio;

                if (b >= num_complete_blocks) {
                    if (lid == 0) {
                        scores_ptr[static_cast<size_t>(query) * max_complete_blocks + b] = -1e30f;
                    }
                    return;
                }

                const size_t q_base = static_cast<size_t>(query) * (kNumHeads * kHeadDim);
                const float kl = static_cast<float>(comp_k_ptr[static_cast<size_t>(b) * kHeadDim + lid]);
                const float p0 = static_cast<float>(q_ptr[q_base + 0 * kHeadDim + lid]) * kl;
                const float p1 = static_cast<float>(q_ptr[q_base + 1 * kHeadDim + lid]) * kl;
                const float p2 = static_cast<float>(q_ptr[q_base + 2 * kHeadDim + lid]) * kl;
                const float p3 = static_cast<float>(q_ptr[q_base + 3 * kHeadDim + lid]) * kl;

                const float dot0 = sycl::reduce_over_group(item.get_group(), p0, sycl::plus<float>());
                const float dot1 = sycl::reduce_over_group(item.get_group(), p1, sycl::plus<float>());
                const float dot2 = sycl::reduce_over_group(item.get_group(), p2, sycl::plus<float>());
                const float dot3 = sycl::reduce_over_group(item.get_group(), p3, sycl::plus<float>());

                if (lid == 0) {
                    const float r0 = sycl::fmax(0.0f, dot0);
                    const float r1 = sycl::fmax(0.0f, dot1);
                    const float r2 = sycl::fmax(0.0f, dot2);
                    const float r3 = sycl::fmax(0.0f, dot3);
                    scores_ptr[static_cast<size_t>(query) * max_complete_blocks + b] = (r0 + r1 + r2 + r3) * inv_sqrt_d;
                }
            });
    });
}

void qsa_compute_block_scores_compressed_batched_host(
    const bf16* q,
    const bf16* compressed_k,
    float* scores_out,
    int query_count,
    int max_complete_blocks,
    int compress_ratio,
    int start_pos) {
    constexpr int kNumHeads = 4;
    constexpr int kHeadDim = 128;
    for (int query = 0; query < query_count; ++query) {
        const int pos = start_pos + query;
        const int visible_len = pos + 1;
        const int num_complete_blocks = visible_len / compress_ratio;
        const bf16* q_row = q + static_cast<size_t>(query) * (kNumHeads * kHeadDim);
        float* scores_row = scores_out + static_cast<size_t>(query) * max_complete_blocks;

        for (int b = 0; b < max_complete_blocks; ++b) {
            if (b >= num_complete_blocks) {
                scores_row[b] = -1e30f;
                continue;
            }
            float sum_relu = 0.0f;
            for (int h = 0; h < kNumHeads; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<float>(q_row[h * kHeadDim + d]) *
                           static_cast<float>(compressed_k[static_cast<size_t>(b) * kHeadDim + d]);
                }
                sum_relu += std::max(0.0f, dot);
            }
            scores_row[b] = sum_relu / std::sqrt(128.0f);
        }
    }
}

void qsa_select_topk_blocks_batched(
    Context& ctx,
    const Tensor& scores,
    Tensor& active_blocks_out,
    int query_count,
    int max_complete_blocks,
    int start_pos,
    int block_topk,
    int compress_ratio,
    Tensor* valid_counts_out) {
    if (query_count <= 0 || block_topk <= 0) return;
    const float* scores_ptr = scores.data_as<float>();
    int* out_ptr = active_blocks_out.data_as<int>();
    int* valid_counts_ptr = (valid_counts_out && valid_counts_out->valid()) ? valid_counts_out->data_as<int>() : nullptr;

    constexpr int kWgSize = 256;
    const size_t out_stride = static_cast<size_t>(block_topk + 4);
    const int selector = qsa_selector_mode(ctx, block_topk, compress_ratio, (start_pos + query_count) / compress_ratio);
    if (selector >= 2 && selector <= 4) {
        if (selector == 2)
            qsa_radix_select_blocks<4>(ctx, scores_ptr, out_ptr, query_count, max_complete_blocks,
                start_pos, block_topk, compress_ratio, out_stride, valid_counts_ptr);
        else if (selector == 3)
            qsa_radix_select_blocks<8>(ctx, scores_ptr, out_ptr, query_count, max_complete_blocks,
                start_pos, block_topk, compress_ratio, out_stride, valid_counts_ptr);
        else
            qsa_radix_select_blocks<8, 4>(ctx, scores_ptr, out_ptr, query_count, max_complete_blocks,
                start_pos, block_topk, compress_ratio, out_stride, valid_counts_ptr);
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> local_cnt_one(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<std::uint32_t, 1> local_worst_key(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> selected_blocks(sycl::range<1>(std::max(1, block_topk)), cgh);
        sycl::local_accessor<std::uint32_t, 1> cutoff_key(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> remaining_rank(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(query_count) * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int query = item.get_group(0);
                const int lid = item.get_local_id(0);

                const int pos = start_pos + query;
                const int visible_len = pos + 1;
                const int num_complete_blocks = visible_len / compress_ratio;
                const int tail_start = num_complete_blocks * compress_ratio;
                const int tail_len = std::max(0, visible_len - tail_start);
                int* my_out = out_ptr + static_cast<size_t>(query) * out_stride;
                const float* my_scores = scores_ptr + static_cast<size_t>(query) * max_complete_blocks;

                if (num_complete_blocks <= block_topk) {
                    for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                        my_out[b] = b;
                    }
                    if (lid < tail_len) {
                        my_out[num_complete_blocks + lid] = tail_start + lid;
                    }
                    if (valid_counts_ptr != nullptr && lid == 0) {
                        valid_counts_ptr[query * 2 + 0] = num_complete_blocks;
                        valid_counts_ptr[query * 2 + 1] = tail_len;
                    }
                    return;
                }

                if (valid_counts_ptr != nullptr && lid == 0) {
                    valid_counts_ptr[query * 2 + 0] = block_topk;
                    valid_counts_ptr[query * 2 + 1] = tail_len;
                }

                if (num_complete_blocks == block_topk + 1) {
                    // R4c: Complement selection for B = K + 1 (513 -> 512).
                    std::uint32_t my_worst_key = 0xffffffffU;
                    int my_worst_id = -1;
                    for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(my_scores[b]);
                        if (my_worst_id < 0 || qsa_candidate_is_worse(key, b, my_worst_key, my_worst_id)) {
                            my_worst_key = key;
                            my_worst_id = b;
                        }
                    }
                    local_worst_key[lid] = my_worst_key;
                    local_cnt_one[lid] = my_worst_id;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            const std::uint32_t other_key = local_worst_key[lid + stride];
                            const int other_id = local_cnt_one[lid + stride];
                            if (other_id >= 0) {
                                if (local_cnt_one[lid] < 0 || qsa_candidate_is_worse(other_key, other_id, local_worst_key[lid], local_cnt_one[lid])) {
                                    local_worst_key[lid] = other_key;
                                    local_cnt_one[lid] = other_id;
                                }
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    const int worst_id = local_cnt_one[0];

                    for (int idx = lid; idx < block_topk; idx += kWgSize) {
                        my_out[idx] = (idx < worst_id) ? idx : (idx + 1);
                    }

                    if (lid < tail_len) {
                        my_out[block_topk + lid] = tail_start + lid;
                    }
                    return;
                }

                if (lid == 0) {
                    cutoff_key[0] = 0U;
                    remaining_rank[0] = block_topk;
                }
                item.barrier(sycl::access::fence_space::local_space);
                std::uint32_t prefix_mask = 0U;
                for (int bit = 31; bit >= 0; --bit) {
                    const std::uint32_t bit_mask = std::uint32_t{1} << bit;
                    const std::uint32_t prefix = cutoff_key[0];
                    int count_one = 0;
                    for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                        const std::uint32_t key = qsa_score_order_key(my_scores[b]);
                        if ((key & prefix_mask) == prefix && (key & bit_mask) != 0U) {
                            ++count_one;
                        }
                    }
                    local_cnt_one[lid] = count_one;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = kWgSize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) local_cnt_one[lid] += local_cnt_one[lid + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0) {
                        if (local_cnt_one[0] >= remaining_rank[0]) {
                            cutoff_key[0] |= bit_mask;
                        } else {
                            remaining_rank[0] -= local_cnt_one[0];
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    prefix_mask |= bit_mask;
                }

                if (lid == 0) {
                    const std::uint32_t threshold = cutoff_key[0];
                    int above_count = 0;
                    for (int b = 0; b < num_complete_blocks; ++b) {
                        if (qsa_score_order_key(my_scores[b]) > threshold) ++above_count;
                    }
                    int ties_needed = block_topk - above_count;
                    int added = 0;
                    for (int b = 0; b < num_complete_blocks; ++b) {
                        const std::uint32_t key = qsa_score_order_key(my_scores[b]);
                        if (key > threshold || (key == threshold && ties_needed > 0)) {
                            selected_blocks[added++] = b;
                            if (key == threshold) --ties_needed;
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < block_topk; idx += kWgSize) {
                    my_out[idx] = selected_blocks[idx];
                }

                if (lid < tail_len) {
                    my_out[block_topk + lid] = tail_start + lid;
                }
            });
    });
}

void qsa_select_topk_blocks_batched_host(
    const float* scores,
    int* active_blocks_out,
    int query_count,
    int max_complete_blocks,
    int start_pos,
    int block_topk,
    int compress_ratio,
    int* valid_counts_out) {
    const size_t out_stride = static_cast<size_t>(block_topk + 4);
    for (int query = 0; query < query_count; ++query) {
        const int pos = start_pos + query;
        const int visible_len = pos + 1;
        const int num_complete_blocks = visible_len / compress_ratio;
        const int tail_start = num_complete_blocks * compress_ratio;
        const int tail_len = std::max(0, visible_len - tail_start);
        const float* my_scores = scores + static_cast<size_t>(query) * max_complete_blocks;
        int* my_out = active_blocks_out + static_cast<size_t>(query) * out_stride;
        const int num_act = qsa_select_topk_blocks_host(
            my_scores, num_complete_blocks, block_topk, visible_len, compress_ratio, my_out);
        if (valid_counts_out) {
            valid_counts_out[query * 2 + 0] = num_act;
            valid_counts_out[query * 2 + 1] = tail_len;
        }
    }
}

void qsa_pre_indexer_batched(
    Context& ctx,
    const Tensor& indexer_qk,
    const Tensor& q_norm_w,
    const Tensor& k_norm_w,
    Tensor& k_cache,
    Tensor& q_out,
    Tensor* compressed_k,
    int* num_compressed_blocks,
    int start_pos,
    int count,
    int compress_ratio,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    int k_cache_base_token) {
    if (count <= 0) return;

    // 1. Cache raw K tokens for this chunk
    qsa_cache_k_batched(ctx, indexer_qk, k_cache, start_pos, count, k_cache_base_token);

    // 2. Compress newly completed blocks if compressed_k is valid
    if (compressed_k != nullptr && compressed_k->valid() && num_compressed_blocks != nullptr) {
        const int total_complete_blocks = (start_pos + count) / compress_ratio;
        if (total_complete_blocks > *num_compressed_blocks) {
            const int start_block = *num_compressed_blocks;
            const int blocks_to_compress = total_complete_blocks - start_block;
            qsa_compress_key_blocks(
                ctx, k_cache, k_norm_w, *compressed_k,
                start_block, blocks_to_compress, theta, interleaved,
                pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
                mrope_section_t, mrope_section_h, mrope_section_w, k_cache_base_token);
            *num_compressed_blocks = total_complete_blocks;
        }
    }

    // 3. Prepare queries (norm + RoPE)
    qsa_prepare_q_batched(
        ctx, indexer_qk, q_norm_w, q_out,
        start_pos, count, theta, interleaved,
        pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w);
}

void qsa_pre_indexer_batched_host(
    const bf16* indexer_qk,
    const bf16* q_norm_w,
    const bf16* k_norm_w,
    bf16* k_cache,
    bf16* q_out,
    bf16* compressed_k,
    int* num_compressed_blocks,
    int start_pos,
    int count,
    int compress_ratio,
    int capacity,
    float theta,
    bool interleaved,
    const int* pos_t,
    const int* pos_h,
    const int* pos_w,
    int prompt_pos_len,
    int text_pos_delta,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w) {
    if (count <= 0) return;

    // 1. Cache raw K tokens for this chunk
    constexpr int kOutDim = 640;
    constexpr int kHeadDim = 128;
    for (int r = 0; r < count; ++r) {
        const int pos = start_pos + r;
        const bf16* k_src = indexer_qk + static_cast<size_t>(r) * kOutDim + 512;
        bf16* k_dst = k_cache + static_cast<size_t>(pos) * kHeadDim;
        std::copy(k_src, k_src + kHeadDim, k_dst);
    }

    // 2. Compress newly completed blocks if compressed_k is valid
    if (compressed_k != nullptr && num_compressed_blocks != nullptr) {
        const int total_complete_blocks = (start_pos + count) / compress_ratio;
        if (total_complete_blocks > *num_compressed_blocks) {
            const int start_block = *num_compressed_blocks;
            const int blocks_to_compress = total_complete_blocks - start_block;
            qsa_compress_key_blocks_host(
                k_cache, k_norm_w, compressed_k,
                start_block, blocks_to_compress, theta, interleaved,
                pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
                mrope_section_t, mrope_section_h, mrope_section_w);
            *num_compressed_blocks = total_complete_blocks;
        }
    }

    // 3. Prepare queries
    qsa_prepare_q_batched_host(
        indexer_qk, q_norm_w, q_out,
        start_pos, count, theta, interleaved,
        pos_t, pos_h, pos_w, prompt_pos_len, text_pos_delta,
        mrope_section_t, mrope_section_h, mrope_section_w);
}

} // namespace ops

