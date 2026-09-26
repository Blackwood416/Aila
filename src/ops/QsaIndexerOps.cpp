#include "ops/QsaIndexerOps.hpp"
#include <algorithm>
#include <cmath>

namespace ops {

namespace {

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
        sycl::local_accessor<float, 1> local_max(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<float, 1> local_min(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> local_cnt_gt(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> local_cnt_ge(sycl::range<1>(kWgSize), cgh);
        sycl::local_accessor<int, 1> selected_blocks(sycl::range<1>(std::max(1, block_topk)), cgh);
        sycl::local_accessor<int, 1> blk_counter(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(kWgSize, kWgSize),
            [=](sycl::nd_item<1> item) {
                const int lid = item.get_local_id(0);

                // 1. Find min and max score across blocks
                float max_val = -1e30f;
                float min_val = 1e30f;
                for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                    float s = scores_ptr[b];
                    max_val = sycl::fmax(max_val, s);
                    min_val = sycl::fmin(min_val, s);
                }
                local_max[lid] = max_val;
                local_min[lid] = min_val;
                item.barrier(sycl::access::fence_space::local_space);

                for (int s = kWgSize / 2; s > 0; s >>= 1) {
                    if (lid < s) {
                        local_max[lid] = sycl::fmax(local_max[lid], local_max[lid + s]);
                        local_min[lid] = sycl::fmin(local_min[lid], local_min[lid + s]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const float global_max = local_max[0];
                const float global_min = local_min[0];
                item.barrier(sycl::access::fence_space::local_space);

                float low = global_min;
                float high = global_max;

                if (global_max > global_min) {
                    // 2. Binary search for threshold separating top-k
                    for (int iter = 0; iter < 32; ++iter) {
                        const float mid = (low + high) * 0.5f;
                        int cnt_gt = 0;
                        int cnt_ge = 0;
                        for (int b = lid; b < num_complete_blocks; b += kWgSize) {
                            float s = scores_ptr[b];
                            if (s > mid) cnt_gt++;
                            if (s >= mid) cnt_ge++;
                        }
                        local_cnt_gt[lid] = cnt_gt;
                        local_cnt_ge[lid] = cnt_ge;
                        item.barrier(sycl::access::fence_space::local_space);

                        for (int s = kWgSize / 2; s > 0; s >>= 1) {
                            if (lid < s) {
                                local_cnt_gt[lid] += local_cnt_gt[lid + s];
                                local_cnt_ge[lid] += local_cnt_ge[lid + s];
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                        const int total_gt = local_cnt_gt[0];
                        const int total_ge = local_cnt_ge[0];
                        item.barrier(sycl::access::fence_space::local_space);

                        if (total_gt >= block_topk) {
                            low = mid;
                        } else if (total_ge < block_topk) {
                            high = mid;
                        } else {
                            // Exact hit: total_gt < block_topk && total_ge >= block_topk
                            low = mid;
                            high = mid;
                            break;
                        }
                    }
                }

                // 3. Thread 0 gathers top block_topk blocks in strictly ascending block order
                if (lid == 0) {
                    if (global_max <= global_min) {
                        for (int b = 0; b < block_topk; ++b) {
                            selected_blocks[b] = b;
                        }
                        blk_counter[0] = block_topk;
                    } else {
                        int count_high = 0;
                        for (int b = 0; b < num_complete_blocks; ++b) {
                            if (scores_ptr[b] > high) {
                                count_high++;
                            }
                        }
                        const int needed_ties = block_topk - count_high;
                        int added = 0;
                        int ties_added = 0;
                        for (int b = 0; b < num_complete_blocks && added < block_topk; ++b) {
                            float s = scores_ptr[b];
                            if (s > high) {
                                selected_blocks[added++] = b;
                            } else if (s >= low && ties_added < needed_ties) {
                                selected_blocks[added++] = b;
                                ties_added++;
                            }
                        }
                        // Safety fallback if extreme float rounding left added < block_topk
                        if (added < block_topk) {
                            for (int b = 0; b < num_complete_blocks && added < block_topk; ++b) {
                                float s = scores_ptr[b];
                                if (s < low) {
                                    selected_blocks[added++] = b;
                                }
                            }
                        }
                        // Ensure strictly increasing order in case safety fallback was hit
                        for (int i = 1; i < block_topk; ++i) {
                            int key = selected_blocks[i];
                            int j = i - 1;
                            while (j >= 0 && selected_blocks[j] > key) {
                                selected_blocks[j + 1] = selected_blocks[j];
                                j--;
                            }
                            selected_blocks[j + 1] = key;
                        }
                        blk_counter[0] = added;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                // 4. Parallel token expansion for the selected blocks
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
            if (a.first != b.first) return a.first > b.first;
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

} // namespace ops
