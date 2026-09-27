#include "ops/QsaIndexerOps.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ops {

namespace {

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
    int mrope_section_w) {
    (void)mrope_section_t;
    if (pos < 0) return;
    const bf16* qk_ptr = indexer_qk.data_as<bf16>();
    const bf16* qn_ptr = q_norm_w.data_as<bf16>();
    bf16* k_cache_ptr = k_cache.data_as<bf16>();
    bf16* q_out_ptr = q_out.data_as<bf16>();

    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kWgSize = 128;
    constexpr int kNumHeads = 4;

    const float pos_scalar = static_cast<float>(pos >= prompt_pos_len ? pos + text_pos_delta : pos);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<float, 1> local_q(sycl::range<1>(kHeadDim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(kNumHeads * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int head = item.get_group(0);
                const int lid = item.get_local_id(0);
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
                        pos, lid, pos_scalar, interleaved, pos_t, pos_h, pos_w,
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
                    k_cache_ptr[static_cast<size_t>(pos) * kHeadDim + lid] = qk_ptr[kNumHeads * kHeadDim + lid];
                }
            });
    });
}

void qsa_cache_k_batched(
    Context& ctx,
    const Tensor& indexer_qk,
    Tensor& k_cache,
    int start_pos,
    int count) {
    if (count <= 0) return;
    const bf16* qk_ptr = indexer_qk.data_as<bf16>();
    bf16* k_cache_ptr = k_cache.data_as<bf16>();

    constexpr int kHeadDim = 128;
    constexpr int kNumHeads = 4;
    constexpr int kTotalDim = (kNumHeads + 1) * kHeadDim; // 640

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>(static_cast<size_t>(count) * kHeadDim),
            [=](sycl::id<1> idx) {
                const int i = static_cast<int>(idx[0]);
                const int t = i / kHeadDim;
                const int d = i % kHeadDim;
                k_cache_ptr[static_cast<size_t>(start_pos + t) * kHeadDim + d] =
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
    int mrope_section_w) {
    (void)mrope_section_t;
    if (num_blocks <= 0) return;
    const bf16* k_cache_ptr = k_cache.data_as<bf16>();
    const bf16* kn_ptr = k_norm_w.data_as<bf16>();
    bf16* comp_k_ptr = compressed_k.data_as<bf16>();

    constexpr int kHeadDim = 128;
    constexpr int kRotaryDim = 64;
    constexpr int kHalfDim = 32;
    constexpr int kWgSize = 128;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<float, 1> local_k(sycl::range<1>(kHeadDim), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_blocks * kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int b = start_block + item.get_group(0);
                const int lid = item.get_local_id(0);

                // 1. Mean-pool 4 tokens
                const size_t b0 = (static_cast<size_t>(4 * b + 0) * kHeadDim) + lid;
                const size_t b1 = (static_cast<size_t>(4 * b + 1) * kHeadDim) + lid;
                const size_t b2 = (static_cast<size_t>(4 * b + 2) * kHeadDim) + lid;
                const size_t b3 = (static_cast<size_t>(4 * b + 3) * kHeadDim) + lid;

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
            });
    });
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

    const float* scores_ptr = scores.data_as<float>();
    int* out_ptr = active_blocks_out.data_as<int>();
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

                for (int idx = lid; idx < block_topk; idx += kWgSize) {
                    out_ptr[idx] = selected_blocks[idx];
                }

                if (lid < tail_len) {
                    out_ptr[block_topk + lid] = tail_start + lid;
                }
            });
    });

    return block_topk;
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
    int* num_compressed_blocks) {
    if (pos < 0 || block_topk <= 0 || compress_ratio <= 0) {
        throw std::invalid_argument("qsa_prepare_causal_query_and_select_blocks: invalid position or block geometry");
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
    int compress_ratio) {
    if (query_count <= 0 || block_topk <= 0) return;
    const float* scores_ptr = scores.data_as<float>();
    int* out_ptr = active_blocks_out.data_as<int>();

    constexpr int kWgSize = 256;
    const size_t out_stride = static_cast<size_t>(block_topk + 4);

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> local_cnt_one(sycl::range<1>(kWgSize), cgh);
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
    int compress_ratio) {
    const size_t out_stride = static_cast<size_t>(block_topk + 4);
    for (int query = 0; query < query_count; ++query) {
        const int pos = start_pos + query;
        const int visible_len = pos + 1;
        const int num_complete_blocks = visible_len / compress_ratio;
        const float* my_scores = scores + static_cast<size_t>(query) * max_complete_blocks;
        int* my_out = active_blocks_out + static_cast<size_t>(query) * out_stride;
        qsa_select_topk_blocks_host(
            my_scores, num_complete_blocks, block_topk, visible_len, compress_ratio, my_out);
    }
}

} // namespace ops

