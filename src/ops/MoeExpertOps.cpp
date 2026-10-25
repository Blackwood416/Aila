#include "MoeExpertOps.hpp"
#include <sycl/ext/intel/math.hpp>
#include <type_traits>
#include "utils/EnvUtils.hpp"

namespace ops {

static constexpr float kDefaultNf4QuantMap[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

void router_softmax_topk_host(
    const bf16* router_scores_ptr,
    const bf16* per_expert_scale_ptr,
    int* expert_ids,
    float* expert_weights,
    int tokens,
    int num_experts,
    int top_k,
    bool norm_topk_prob) {
    for (int t = 0; t < tokens; ++t) {
        const bf16* row = router_scores_ptr + static_cast<size_t>(t) * num_experts;
        float max_s = -1.0e30f;
        for (int e = 0; e < num_experts; ++e) {
            max_s = std::max(max_s, static_cast<float>(row[e]));
        }
        std::vector<std::pair<float, int>> probs(static_cast<size_t>(num_experts));
        float sum_exp = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            float v = std::exp(static_cast<float>(row[e]) - max_s);
            probs[static_cast<size_t>(e)] = {v, e};
            sum_exp += v;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int e = 0; e < num_experts; ++e) {
            probs[static_cast<size_t>(e)].first *= inv_sum;
        }
        std::partial_sort(
            probs.begin(), probs.begin() + top_k, probs.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
        float topk_sum = 0.0f;
        if (norm_topk_prob) {
            for (int k = 0; k < top_k; ++k) {
                topk_sum += probs[static_cast<size_t>(k)].first;
            }
        }
        float inv_topk_sum = (norm_topk_prob && topk_sum > 0.0f) ? (1.0f / topk_sum) : 1.0f;
        for (int k = 0; k < top_k; ++k) {
            const auto& item_pair = probs[static_cast<size_t>(k)];
            expert_ids[t * top_k + k] = item_pair.second;
            float w = item_pair.first * inv_topk_sum;
            if (per_expert_scale_ptr) {
                w *= static_cast<float>(per_expert_scale_ptr[item_pair.second]);
            }
            expert_weights[t * top_k + k] = w;
        }
    }
}

void router_topk_ids_host(
    const bf16* router_scores_ptr,
    int* expert_ids,
    int tokens,
    int num_experts,
    int top_k) {
    for (int t = 0; t < tokens; ++t) {
        const bf16* row = router_scores_ptr + static_cast<size_t>(t) * num_experts;
        std::vector<std::pair<float, int>> items(num_experts);
        for (int i = 0; i < num_experts; ++i) {
            items[static_cast<size_t>(i)] = {static_cast<float>(row[i]), i};
        }
        std::partial_sort(items.begin(), items.begin() + top_k, items.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
        for (int k = 0; k < top_k; ++k) {
            expert_ids[t * top_k + k] = items[static_cast<size_t>(k)].second;
        }
    }
}

template <bool Resident>
static sycl::event launch_router_softmax_topk(
    Context& ctx,
    const bf16* router_scores_ptr,
    const bf16* per_expert_scale_ptr,
    int* expert_ids_dev,
    float* expert_weights_dev,
    int tokens,
    int num_experts,
    int top_k,
    bool norm_topk_prob,
    const int* resident_table,
    int* resident_selected,
    int* host_ids,
    float* host_weights,
    int* host_selected) {
    if (num_experts != 512) {
        std::vector<bf16> scores_host(static_cast<size_t>(tokens) * num_experts);
        std::vector<bf16> scale_host(per_expert_scale_ptr ? num_experts : 0);
        ctx.memcpy_d2h(scores_host.data(), router_scores_ptr, scores_host.size() * sizeof(bf16));
        if (per_expert_scale_ptr) {
            ctx.memcpy_d2h(scale_host.data(), per_expert_scale_ptr, scale_host.size() * sizeof(bf16));
        }
        std::vector<int> ids_host(static_cast<size_t>(tokens) * top_k);
        std::vector<float> w_host(static_cast<size_t>(tokens) * top_k);
        router_softmax_topk_host(
            scores_host.data(), per_expert_scale_ptr ? scale_host.data() : nullptr,
            ids_host.data(), w_host.data(), tokens, num_experts, top_k, norm_topk_prob);
        ctx.memcpy_h2d(expert_ids_dev, ids_host.data(), ids_host.size() * sizeof(int));
        ctx.memcpy_h2d(expert_weights_dev, w_host.data(), w_host.size() * sizeof(float));
        return {};
    }

    constexpr int wg = 512;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scores_slm(sycl::range<1>(wg), cgh);
        sycl::local_accessor<int, 1> indices_slm(sycl::range<1>(wg), cgh);
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(wg), cgh);
        sycl::local_accessor<int, 1> red_idx_slm(sycl::range<1>(wg), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(tokens) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int token = static_cast<int>(item.get_group(0));
                const int tid = static_cast<int>(item.get_local_id(0));
                const bf16* row = router_scores_ptr + static_cast<size_t>(token) * num_experts;

                float s = static_cast<float>(row[tid]);
                scores_slm[tid] = s;
                indices_slm[tid] = tid;
                red_slm[tid] = s;
                item.barrier(sycl::access::fence_space::local_space);

                const int sg_id = tid / 16;
                const int sg_lane = tid % 16;
                auto sg = item.get_sub_group();

                // Max reduction
                float sg_max = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                if (sg_lane == 0) {
                    red_slm[sg_id] = sg_max;
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (sg_id == 0) {
                    float m = sycl::max(red_slm[sg_lane], red_slm[sg_lane + 16]);
                    float final_max = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                    if (sg_lane == 0) {
                        red_slm[0] = final_max;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                const float max_score = red_slm[0];
                item.barrier(sycl::access::fence_space::local_space);

                // Softmax
                float exp_val = sycl::native::exp(s - max_score);
                scores_slm[tid] = exp_val;
                float sg_sum = sycl::reduce_over_group(sg, exp_val, sycl::plus<>());
                if (sg_lane == 0) {
                    red_slm[sg_id] = sg_sum;
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (sg_id == 0) {
                    float s2 = red_slm[sg_lane] + red_slm[sg_lane + 16];
                    float final_sum = sycl::reduce_over_group(sg, s2, sycl::plus<>());
                    if (sg_lane == 0) {
                        red_slm[0] = final_sum;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                const float inv_sum = 1.0f / red_slm[0];
                scores_slm[tid] = exp_val * inv_sum;
                item.barrier(sycl::access::fence_space::local_space);

                // Top-K Selection
                for (int k = 0; k < top_k; ++k) {
                    float cand_val = (tid >= k) ? scores_slm[tid] : -1.0e30f;
                    int cand_idx = tid;

                    // Reduce within subgroup via XOR shuffles
#pragma unroll
                    for (int offset = 8; offset > 0; offset >>= 1) {
                        float other_val = sycl::permute_group_by_xor(sg, cand_val, offset);
                        int other_idx = sycl::permute_group_by_xor(sg, cand_idx, offset);
                        if (other_val > cand_val || (other_val == cand_val && other_idx < cand_idx)) {
                            cand_val = other_val;
                            cand_idx = other_idx;
                        }
                    }
                    if (sg_lane == 0) {
                        red_slm[sg_id] = cand_val;
                        red_idx_slm[sg_id] = cand_idx;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        float v0 = red_slm[sg_lane];
                        int i0 = red_idx_slm[sg_lane];
                        float v1 = red_slm[sg_lane + 16];
                        int i1 = red_idx_slm[sg_lane + 16];
                        if (v1 > v0 || (v1 == v0 && i1 < i0)) {
                            v0 = v1;
                            i0 = i1;
                        }
#pragma unroll
                        for (int offset = 8; offset > 0; offset >>= 1) {
                            float other_val = sycl::permute_group_by_xor(sg, v0, offset);
                            int other_idx = sycl::permute_group_by_xor(sg, i0, offset);
                            if (other_val > v0 || (other_val == v0 && other_idx < i0)) {
                                v0 = other_val;
                                i0 = other_idx;
                            }
                        }
                        if (sg_lane == 0) {
                            red_idx_slm[0] = i0;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (tid == 0) {
                        int best_pos = red_idx_slm[0];
                        if (best_pos != k) {
                            float tmp_v = scores_slm[k];
                            int tmp_i = indices_slm[k];
                            scores_slm[k] = scores_slm[best_pos];
                            indices_slm[k] = indices_slm[best_pos];
                            scores_slm[best_pos] = tmp_v;
                            indices_slm[best_pos] = tmp_i;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Renormalization and Write
                if (tid == 0) {
                    float topk_sum = 0.0f;
                    if (norm_topk_prob) {
                        for (int k = 0; k < top_k; ++k) {
                            topk_sum += scores_slm[k];
                        }
                    }
                    float inv_topk_sum = (norm_topk_prob && topk_sum > 0.0f) ? (1.0f / topk_sum) : 1.0f;
                    for (int k = 0; k < top_k; ++k) {
                        int chosen_exp = indices_slm[k];
                        expert_ids_dev[token * top_k + k] = chosen_exp;
                        float w = scores_slm[k] * inv_topk_sum;
                        if (per_expert_scale_ptr) {
                            w *= static_cast<float>(per_expert_scale_ptr[chosen_exp]);
                        }
                        expert_weights_dev[token * top_k + k] = w;
                        if constexpr (Resident) {
                            // The complete top-k arithmetic above is shared with
                            // the original route. Classification only consumes
                            // its final ID/weight, without a second kernel.
                            const int slot = resident_table[chosen_exp];
                            resident_selected[k] = slot;
                            host_selected[k] = slot;
                            host_ids[k] = chosen_exp;
                            host_weights[k] = w;
                        }
                    }
                }
            });
    });
}

sycl::event router_softmax_topk_device(
    Context& ctx, const bf16* router_scores_ptr, const bf16* per_expert_scale_ptr,
    int* expert_ids_dev, float* expert_weights_dev, int tokens, int num_experts,
    int top_k, bool norm_topk_prob) {
    return launch_router_softmax_topk<false>(ctx, router_scores_ptr, per_expert_scale_ptr,
        expert_ids_dev, expert_weights_dev, tokens, num_experts, top_k, norm_topk_prob,
        nullptr, nullptr, nullptr, nullptr, nullptr);
}

sycl::event router_softmax_topk_resident(
    Context& ctx, const bf16* router_scores_ptr, const bf16* per_expert_scale_ptr,
    int* expert_ids_dev, float* expert_weights_dev, const int* resident_table,
    int* resident_selected, int* host_ids, float* host_weights, int* host_selected,
    int num_experts, int top_k, bool norm_topk_prob) {
    if (num_experts != 512 || top_k < 1 || top_k > 16)
        throw std::invalid_argument("Resident router requires one token, 512 experts and top-k 1..16");
    if (!router_scores_ptr || !expert_ids_dev || !expert_weights_dev || !resident_table ||
        !resident_selected || !host_ids || !host_weights || !host_selected)
        throw std::invalid_argument("Resident router requires all route/classification outputs");
    return launch_router_softmax_topk<true>(ctx, router_scores_ptr, per_expert_scale_ptr,
        expert_ids_dev, expert_weights_dev, 1, num_experts, top_k, norm_topk_prob,
        resident_table, resident_selected, host_ids, host_weights, host_selected);
}

sycl::event router_topk_ids_device(
    Context& ctx,
    const bf16* router_scores_ptr,
    int* expert_ids_dev,
    int tokens,
    int num_experts,
    int top_k) {
    if (num_experts != 512) {
        std::vector<bf16> scores_host(static_cast<size_t>(tokens) * num_experts);
        ctx.memcpy_d2h(scores_host.data(), router_scores_ptr, scores_host.size() * sizeof(bf16));
        std::vector<int> ids_host(static_cast<size_t>(tokens) * top_k);
        router_topk_ids_host(scores_host.data(), ids_host.data(), tokens, num_experts, top_k);
        ctx.memcpy_h2d(expert_ids_dev, ids_host.data(), ids_host.size() * sizeof(int));
        return {};
    }

    constexpr int wg = 512;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scores_slm(sycl::range<1>(wg), cgh);
        sycl::local_accessor<float, 1> red_slm(sycl::range<1>(32), cgh);
        sycl::local_accessor<int, 1> red_idx_slm(sycl::range<1>(32), cgh);
        sycl::local_accessor<int, 1> selected_ids(sycl::range<1>(64), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(tokens) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int token = static_cast<int>(item.get_group(0));
                const int tid = static_cast<int>(item.get_local_id(0));
                const bf16* row = router_scores_ptr + static_cast<size_t>(token) * num_experts;

                scores_slm[tid] = static_cast<float>(row[tid]);
                item.barrier(sycl::access::fence_space::local_space);

                const int sg_id = tid / 16;
                const int sg_lane = tid % 16;
                auto sg = item.get_sub_group();

                // Top-K selection directly on raw logits without softmax
                for (int k = 0; k < top_k; ++k) {
                    bool already_selected = false;
                    for (int prev = 0; prev < k; ++prev) {
                        if (selected_ids[prev] == tid) {
                            already_selected = true;
                            break;
                        }
                    }
                    float cand_val = already_selected ? -1.0e38f : scores_slm[tid];
                    int cand_idx = already_selected ? 999999 : tid;

                    // Reduce within subgroup via XOR shuffles
#pragma unroll
                    for (int offset = 8; offset > 0; offset >>= 1) {
                        float other_val = sycl::permute_group_by_xor(sg, cand_val, offset);
                        int other_idx = sycl::permute_group_by_xor(sg, cand_idx, offset);
                        if (other_val > cand_val || (other_val == cand_val && other_idx < cand_idx)) {
                            cand_val = other_val;
                            cand_idx = other_idx;
                        }
                    }
                    if (sg_lane == 0) {
                        red_slm[sg_id] = cand_val;
                        red_idx_slm[sg_id] = cand_idx;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        float v0 = red_slm[sg_lane];
                        int i0 = red_idx_slm[sg_lane];
                        float v1 = red_slm[sg_lane + 16];
                        int i1 = red_idx_slm[sg_lane + 16];
                        if (v1 > v0 || (v1 == v0 && i1 < i0)) {
                            v0 = v1;
                            i0 = i1;
                        }
#pragma unroll
                        for (int offset = 8; offset > 0; offset >>= 1) {
                            float other_val = sycl::permute_group_by_xor(sg, v0, offset);
                            int other_idx = sycl::permute_group_by_xor(sg, i0, offset);
                            if (other_val > v0 || (other_val == v0 && other_idx < i0)) {
                                v0 = other_val;
                                i0 = other_idx;
                            }
                        }
                        if (sg_lane == 0) {
                            red_idx_slm[0] = i0;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (tid == 0) {
                        int best_pos = red_idx_slm[0];
                        selected_ids[k] = best_pos;
                        expert_ids_dev[token * top_k + k] = best_pos;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
            });
    });
}

namespace {

template <bool UseNested, bool FixedNested256 = false>
inline float load_expert_absmax(const float* absmax_f32,
                                const MoeNestedScaleDev& nested,
                                size_t index) {
    if constexpr (UseNested) {
        // Q4E validates nested blocksize 256 at load, so the fixed-geometry
        // instantiation replaces the runtime 64-bit division with an exact
        // shift; unsigned x/256 == x>>8 for every index.
        size_t nested_block;
        if constexpr (FixedNested256) nested_block = index >> 8;
        else nested_block = index / static_cast<size_t>(nested.blocksize);
        // Match the host fill's two f32 rounding steps. A single contracted
        // FMA changes some reconstructed absmax values by 1 ULP, which can
        // change greedy routing after the 48-layer recurrence.
        const float product = sycl::fma(
            nested.nested_quant_map[nested.codes[index]],
            nested.nested_absmax[nested_block], 0.0f);
        return sycl::fma(product, 1.0f, nested.offset);
    } else {
        return absmax_f32[index];
    }
}

// These kernels intentionally approximate NF4 arithmetic. They keep the
// resident descriptor layout, nested-scale reconstruction and BF16 SwiGLU
// storage boundary, while replacing eight floating multiplies with two
// signed packed dot products. The default NF4 codebook has max(abs)=1.
inline bool moe_q8_isfinite(float value) {
    // The engine builds with -ffast-math. Inspect exponent bits so its
    // finite-math assumption cannot erase the quantizer's sanitization.
    return (sycl::bit_cast<uint32_t>(value) & 0x7f800000u) != 0x7f800000u;
}

inline int32_t moe_q8_dot_acc(int32_t packed_a, int32_t packed_b, int32_t acc) {
    // Same signed-byte multiply/add idiom as oneAPI dot_acc(int32,int32,int32).
    // The vendor header defines non-inline external overloads, so including
    // it in both this runtime TU and the prototype tests violates ODR.
    const int8_t* a = reinterpret_cast<const int8_t*>(&packed_a);
    const int8_t* b = reinterpret_cast<const int8_t*>(&packed_b);
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + acc;
}

bool moe_q8_fast_rte_enabled() {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_GPU_Q8_FAST_RTE", false);
    return enabled;
}

inline int8_t quantize_moe_q8_value(float value, float maximum, bool fast_rte) {
    if (!moe_q8_isfinite(value) || !(maximum > 0.0f) || !moe_q8_isfinite(maximum)) return 0;
    if (fast_rte && maximum > 0x1p-110f && maximum <= 0x1p120f) {
        // Approximate reciprocal supplies a candidate integer. BF16 values
        // times these small integers are exact in f32; scaled half-thresholds
        // repair the candidate and preserve the exact rational RTE decision.
        // The range keeps reciprocal/threshold arithmetic away from FTZ and
        // overflow; extreme blocks retain the correctly rounded division.
        int magnitude = static_cast<int>(sycl::rint(sycl::clamp(sycl::fabs(value) * (127.0f / maximum), 0.0f, 127.0f)));
        const float numerator = sycl::fma(sycl::fabs(value), 254.0f / 256.0f, 0.0f);
        const float lower = sycl::fma(maximum, static_cast<float>(2 * magnitude - 1) / 256.0f, 0.0f);
        const float upper = sycl::fma(maximum, static_cast<float>(2 * magnitude + 1) / 256.0f, 0.0f);
        if (numerator > upper || (numerator == upper && (magnitude & 1))) ++magnitude;
        else if (numerator < lower || (numerator == lower && (magnitude & 1))) --magnitude;
        return static_cast<int8_t>(value < 0.0f ? -magnitude : magnitude);
    }
    // BF16 * 127 is exact in f32. Correctly rounded division prevents the
    // engine's fast-math reciprocal from moving exact half-way RTE ties.
    // Scale large BF16 inputs by a power of two first to keep v*127 finite.
    const float ratio = maximum > 0x1p120f
        ? sycl::ext::intel::math::fdiv_rn(sycl::fma(value, 127.0f / 256.0f, 0.0f), maximum / 256.0f)
        : sycl::ext::intel::math::fdiv_rn(value * 127.0f, maximum);
    const float rounded = sycl::rint(sycl::clamp(ratio, -127.0f, 127.0f));
    return static_cast<int8_t>(rounded);
}

template <typename QuantMap>
inline int32_t pack_moe_nf4_q8_half(uint32_t packed, int half, const QuantMap& qmap) {
    const uint32_t bytes = packed >> (half * 16);
    // NF4 packs the first logical element into each byte's HIGH nibble.
    const uint32_t p0 = static_cast<uint32_t>(qmap[(bytes >> 4) & 15]) & 255u;
    const uint32_t p1 = static_cast<uint32_t>(qmap[bytes & 15]) & 255u;
    const uint32_t p2 = static_cast<uint32_t>(qmap[(bytes >> 12) & 15]) & 255u;
    const uint32_t p3 = static_cast<uint32_t>(qmap[(bytes >> 8) & 15]) & 255u;
    return sycl::bit_cast<int32_t>(p0 | (p1 << 8) | (p2 << 16) | (p3 << 24));
}

size_t moe_q8_max_work_group_size(Context& ctx) {
    const sycl::device device = ctx.queue().get_device();
    // One cached device per submitting thread. Distinct Context objects may
    // share a device; switching devices refreshes the physical capability.
    thread_local sycl::device cached_device = device;
    thread_local size_t cached_max = cached_device.get_info<sycl::info::device::max_work_group_size>();
    if (device != cached_device) {
        cached_device = device;
        cached_max = device.get_info<sycl::info::device::max_work_group_size>();
    }
    return cached_max;
}

// A complete work-group owns one rank/block; masked ranks exit uniformly
// before any work-group collective. This is also the non-WG1024 fallback.
sycl::event launch_moe_quantize_q8_64(Context& ctx, const bf16* input,
    int8_t* quantized, float* scales, int ranks, int features, const int* selected) {
    constexpr int wg = 64;
    const int blocks = features / 64;
    const bool fast_rte = moe_q8_fast_rte_enabled();
    return ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(size_t(ranks) * blocks * wg, wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t block = item.get_group(0);
                const int rank = static_cast<int>(block / blocks);
                if (selected && selected[rank] < 0) return;
                const size_t index = block * 64 + item.get_local_id(0);
                uint16_t bits = reinterpret_cast<const uint16_t*>(input)[index];
                if ((bits & 0x7f80u) == 0x7f80u) bits = 0;
                const float value = static_cast<float>(sycl::bit_cast<bf16>(bits));
                const float magnitude = sycl::fabs(value);
                const float maximum = sycl::reduce_over_group(item.get_group(), magnitude, sycl::maximum<float>());
                const float scale = maximum / 127.0f;
                if (item.get_local_id(0) == 0) scales[block] = scale;
                quantized[index] = quantize_moe_q8_value(value, maximum, fast_rte);
            });
    });
}

// Shared by the retired resident path and the adaptive decode path. With
// selected == nullptr every rank uses its own descriptor index (adaptive),
// and token_ids selects the Q8 input row per rank (nullptr means token 0).
// Descriptors and tokens are passed as either device-visible pointers or
// by-value 16-entry mirrors captured into the kernel (matching the BF16
// grouped path, so no raw host stack pointer is ever dereferenced on device).
template <typename Tokens>
inline int moe_q8_token_of(const Tokens& tokens, int rank) {
    if constexpr (std::is_pointer_v<Tokens>) return tokens ? tokens[rank] : 0;
    else return tokens[static_cast<size_t>(rank)];
}

template <bool UseNested, int WorkGroup, bool FixedNested256 = false,
          typename Slots, typename Tokens = const int*>
sycl::event launch_moe_resident_gate_up_q8_64(Context& ctx,
    const Slots& descriptors, const int* selected,
    const int8_t* xq, const float* xs, bf16* h, int8_t* hq, float* hs,
    int top_k, int hidden, int intermediate, Tokens token_ids = nullptr,
    const sycl::event* dep_event = nullptr) {
    constexpr int sg_size = 16;
    constexpr int rows_per_group = WorkGroup / sg_size;
    const int groups_per_rank = intermediate / rows_per_group;
    const int packed_per_row = hidden / 2;
    const int blocks_per_row = hidden / 64;
    constexpr float lut_scale = 1.0f / 127.0f;
    const bool fast_rte = moe_q8_fast_rte_enabled();
    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<int32_t, 1> qmap(sycl::range<1>(16), cgh);
        sycl::local_accessor<float, 1> h_block(sycl::range<1>(64), cgh);
        sycl::local_accessor<float, 1> h_scale(sycl::range<1>(1), cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(top_k) * groups_per_rank * WorkGroup, WorkGroup),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int rank = static_cast<int>(item.get_group(0) / groups_per_rank);
                const int tile = static_cast<int>(item.get_group(0) % groups_per_rank);
                const int slot_index = selected ? selected[rank] : rank;
                // I%64 and rows_per_group|64 ensure the ENTIRE WG shares rank.
                if (slot_index < 0) return;
                const size_t token_base = static_cast<size_t>(moe_q8_token_of(token_ids, rank)) * hidden;
                const int8_t* xq_token = xq + token_base;
                const float* xs_token = xs + token_base / 64;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sg_size;
                const int lane = lid % sg_size;
                const int row = tile * rows_per_group + sg_id;
                if (lid < 16) qmap[lid] = static_cast<int32_t>(sycl::rint(kDefaultNf4QuantMap[lid] * 127.0f));
                item.barrier(sycl::access::fence_space::local_space);
                const auto sg = item.get_sub_group();
                const auto& slot = descriptors[slot_index];
                const size_t gate_bytes = size_t(row) * packed_per_row;
                const size_t up_bytes = size_t(row + intermediate) * packed_per_row;
                const size_t gate_blocks = size_t(row) * blocks_per_row;
                const size_t up_blocks = size_t(row + intermediate) * blocks_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int byte_offset = lane * 4; byte_offset < packed_per_row; byte_offset += sg_size * 4) {
                    const uint32_t g4 = *reinterpret_cast<const uint32_t*>(slot.gate_up_packed + gate_bytes + byte_offset);
                    const uint32_t u4 = *reinterpret_cast<const uint32_t*>(slot.gate_up_packed + up_bytes + byte_offset);
                    const int input_block = byte_offset / 32;
                    const float ga_leader = (lane == 0 || lane == 8)
                        ? load_expert_absmax<UseNested, FixedNested256>(slot.gate_up_absmax, slot.gate_up_nested, gate_blocks + input_block) : 0.0f;
                    const float ua_leader = (lane == 0 || lane == 8)
                        ? load_expert_absmax<UseNested, FixedNested256>(slot.gate_up_absmax, slot.gate_up_nested, up_blocks + input_block) : 0.0f;
                    const float ga0 = sycl::group_broadcast(sg, ga_leader, 0);
                    const float ga8 = sycl::group_broadcast(sg, ga_leader, 8);
                    const float ua0 = sycl::group_broadcast(sg, ua_leader, 0);
                    const float ua8 = sycl::group_broadcast(sg, ua_leader, 8);
                    const float ga = lane < 8 ? ga0 : ga8;
                    const float ua = lane < 8 ? ua0 : ua8;
                    const int8_t* input = xq_token + byte_offset * 2;
                    const int32_t x0 = *reinterpret_cast<const int32_t*>(input);
                    const int32_t x1 = *reinterpret_cast<const int32_t*>(input + 4);
                    const int32_t gate_dot = moe_q8_dot_acc(pack_moe_nf4_q8_half(g4, 0, qmap), x0, 0)
                        + moe_q8_dot_acc(pack_moe_nf4_q8_half(g4, 1, qmap), x1, 0);
                    const int32_t up_dot = moe_q8_dot_acc(pack_moe_nf4_q8_half(u4, 0, qmap), x0, 0)
                        + moe_q8_dot_acc(pack_moe_nf4_q8_half(u4, 1, qmap), x1, 0);
                    const float input_scale = xs_token[input_block];
                    gate_acc = sycl::fma(static_cast<float>(gate_dot), (ga * lut_scale) * input_scale, gate_acc);
                    up_acc = sycl::fma(static_cast<float>(up_dot), (ua * lut_scale) * input_scale, up_acc);
                }
                const float gate = sycl::reduce_over_group(sg, gate_acc, sycl::plus<float>());
                const float up = sycl::reduce_over_group(sg, up_acc, sycl::plus<float>());
                if (lane == 0) {
                    const float silu_gate = gate / (1.0f + sycl::native::exp(-gate));
                    const bf16 rounded = bf16(silu_gate * up);
                    h[size_t(rank) * intermediate + row] = rounded;
                    if constexpr (WorkGroup == 1024) {
                        uint16_t bits = sycl::bit_cast<uint16_t>(rounded);
                        if ((bits & 0x7f80u) == 0x7f80u) bits = 0;
                        h_block[sg_id] = static_cast<float>(sycl::bit_cast<bf16>(bits));
                    }
                }
                if constexpr (WorkGroup == 1024) {
                    item.barrier(sycl::access::fence_space::local_space);
                    if (sg_id == 0) {
                        float magnitude = 0.0f;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float value = h_block[lane + j * sg_size];
                            magnitude = sycl::max(magnitude, sycl::fabs(value));
                        }
                        const float maximum = sycl::reduce_over_group(sg, magnitude, sycl::maximum<float>());
                        if (lane == 0) {
                            h_scale[0] = maximum;
                            hs[size_t(rank) * (intermediate / 64) + tile] = maximum / 127.0f;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    if (lane == 0) hq[size_t(rank) * intermediate + row] = quantize_moe_q8_value(h_block[sg_id], h_scale[0], fast_rte);
                }
            });
    });
}

// Shared by the retired resident path and the adaptive decode path. With
// selected == nullptr every rank uses its own descriptor index. When
// decode_top_k is positive the rank result lands in the (token, ranks[rank])
// slot of the decode-wide down accumulator instead of the compact
// [top_k, hidden] resident buffer.
template <bool UseNested, bool FixedNested256 = false, typename Slots,
          typename Tokens = const int*, typename Ranks = const int*>
sycl::event launch_moe_resident_down_q8_64(Context& ctx,
    const Slots& descriptors, const int* selected,
    const int8_t* hq, const float* hs, float* down,
    int top_k, int hidden, int intermediate, Tokens token_ids = nullptr,
    Ranks ranks = nullptr, int decode_top_k = 0,
    const sycl::event* dep_event = nullptr) {
    constexpr int wg = 256, sg_size = 16, rows_per_group = wg / sg_size;
    constexpr float lut_scale = 1.0f / 127.0f;
    const int groups_per_rank = hidden / rows_per_group;
    const int packed_per_row = intermediate / 2;
    const int blocks_per_row = intermediate / 64;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<int32_t, 1> qmap(sycl::range<1>(16), cgh);
        cgh.parallel_for(sycl::nd_range<1>(size_t(top_k) * groups_per_rank * wg, wg),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int rank = static_cast<int>(item.get_group(0) / groups_per_rank);
                const int slot_index = selected ? selected[rank] : rank;
                if (slot_index < 0) return;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int lane = lid % sg_size;
                const int row = static_cast<int>(item.get_group(0) % groups_per_rank) * rows_per_group + lid / sg_size;
                if (lid < 16) qmap[lid] = static_cast<int32_t>(sycl::rint(kDefaultNf4QuantMap[lid] * 127.0f));
                item.barrier(sycl::access::fence_space::local_space);
                const auto sg = item.get_sub_group();
                const auto& slot = descriptors[slot_index];
                const size_t bytes_base = size_t(row) * packed_per_row;
                const size_t blocks_base = size_t(row) * blocks_per_row;
                const int8_t* input = hq + size_t(rank) * intermediate;
                const float* scales = hs + size_t(rank) * blocks_per_row;
                float acc = 0.0f;
                for (int byte_offset = lane * 4; byte_offset < packed_per_row; byte_offset += sg_size * 4) {
                    const uint32_t d4 = *reinterpret_cast<const uint32_t*>(slot.down_packed + bytes_base + byte_offset);
                    const int input_block = byte_offset / 32;
                    const float da_leader = (lane == 0 || lane == 8)
                        ? load_expert_absmax<UseNested, FixedNested256>(slot.down_absmax, slot.down_nested, blocks_base + input_block) : 0.0f;
                    const float da0 = sycl::group_broadcast(sg, da_leader, 0);
                    const float da8 = sycl::group_broadcast(sg, da_leader, 8);
                    const float da = lane < 8 ? da0 : da8;
                    const int32_t x0 = *reinterpret_cast<const int32_t*>(input + byte_offset * 2);
                    const int32_t x1 = *reinterpret_cast<const int32_t*>(input + byte_offset * 2 + 4);
                    const int32_t dot = moe_q8_dot_acc(pack_moe_nf4_q8_half(d4, 0, qmap), x0, 0)
                        + moe_q8_dot_acc(pack_moe_nf4_q8_half(d4, 1, qmap), x1, 0);
                    acc = sycl::fma(static_cast<float>(dot), (da * lut_scale) * scales[input_block], acc);
                }
                const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) {
                    // The adaptive decode accumulator is indexed by the
                    // (token, rank) route slot, exactly like the BF16 path;
                    // the resident path keeps its compact [top_k, hidden] form.
                    size_t out_rank = static_cast<size_t>(rank);
                    if (decode_top_k > 0) {
                        out_rank = static_cast<size_t>(moe_q8_token_of(token_ids, rank)) *
                                       static_cast<size_t>(decode_top_k) +
                                   static_cast<size_t>(moe_q8_token_of(ranks, rank));
                    }
                    down[out_rank * hidden + row] = sum;
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_decode_gate_up_swiglu_grouped(
    Context& ctx,
    const std::array<MoeExpertSlotDev, 16>& slot_ptrs,
    const std::array<int, 16>& token_ptrs,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(num_active_experts) * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                if (global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);
                    const auto& slot = slot_ptrs[static_cast<size_t>(expert_idx)];

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            // Each half-subgroup shares one outer scale block.
                            // Use fixed broadcast sources because subgroup
                            // collectives require a uniform source lane.
                            const int leader = sg_lane < 8 ? 0 : 8;
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const int token = token_ptrs[static_cast<size_t>(expert_idx)];
                        const bf16* token_x = x + static_cast<size_t>(token) * in_features;
                        const vec8 in_v = *reinterpret_cast<const vec8*>(token_x + byte_offset * 2);

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
                }

                float gate_sum = sycl::reduce_over_group(item.get_sub_group(), gate_acc, sycl::plus<>());
                float up_sum = sycl::reduce_over_group(item.get_sub_group(), up_acc, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const float silu_gate = gate_sum / (1.0f + sycl::native::exp(-gate_sum));
                    h[global_row] = bf16(silu_gate * up_sum);
                }
            });
    });
}

template <bool UseNested, bool Resident = false, bool FixedNested256 = false>
sycl::event launch_moe_decode_gate_up_swiglu_grouped_dev(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr, const int* resident_slots = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(num_active_experts) * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                if (global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);
                    const int slot_index = Resident ? resident_slots[expert_idx] : expert_idx;
                    // All lanes of this subgroup share the rank. The only
                    // work-group barrier is above this uniform subgroup exit.
                    if constexpr (Resident) { if (slot_index < 0) return; }
                    const auto& slot = slots[slot_index];

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const int token = token_ids ? token_ids[expert_idx] : 0;
                        const bf16* token_x = x + static_cast<size_t>(token) * in_features;
                        const vec8 in_v = *reinterpret_cast<const vec8*>(token_x + byte_offset * 2);

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
                }

                float gate_sum = sycl::reduce_over_group(item.get_sub_group(), gate_acc, sycl::plus<>());
                float up_sum = sycl::reduce_over_group(item.get_sub_group(), up_acc, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const float silu_gate = gate_sum / (1.0f + sycl::native::exp(-gate_sum));
                    h[global_row] = bf16(silu_gate * up_sum);
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_decode_down_weighted_accumulate(
    Context& ctx,
    const std::array<MoeExpertSlotDev, 16>& slot_ptrs,
    const std::array<float, 16>& weight_ptrs,
    const float* quant_map,
    const bf16* h,
    float* acc_f32,
    bf16* out_bf16,
    int num_active_experts,
    int in_features,
    int out_features,
    int blocksize,
    const bf16* shared_out,
    const bf16* shared_gate) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    const int k_capped = std::min(num_active_experts, 16);
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t num_groups = (static_cast<size_t>(out_features) + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float total_val = 0.0f;
                if (row < static_cast<size_t>(out_features)) {
                    for (int e = 0; e < k_capped; ++e) {
                        const float w_exp = weight_ptrs[static_cast<size_t>(e)];
                        if (w_exp == 0.0f) continue;

                        const auto& slot = slot_ptrs[static_cast<size_t>(e)];
                        const size_t byte_base = row * packed_bytes_per_row;
                        const size_t block_base = row * blocks_per_row;
                        const bf16* in_expert = h + static_cast<size_t>(e) * in_features;

                        float expert_acc = 0.0f;
                        int byte_offset = sg_lane * 4;
                        for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                            const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                                slot.down_packed + byte_base + byte_offset);
                            const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                            float da;
                            if (blocksize == 64) {
                                const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                    ? load_expert_absmax<UseNested, FixedNested256>(
                                          slot.down_absmax, slot.down_nested, scale_index)
                                    : 0.0f;
                                const float da0 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 0);
                                const float da8 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 8);
                                da = sg_lane < 8 ? da0 : da8;
                            } else {
                                da = load_expert_absmax<UseNested, FixedNested256>(
                                    slot.down_absmax, slot.down_nested, scale_index);
                            }
                            const vec8 in_v = *reinterpret_cast<const vec8*>(in_expert + byte_offset * 2);

                            const uint8_t db0 = static_cast<uint8_t>(d4);
                            const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                            const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                            const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                            expert_acc = sycl::fma(static_cast<float>(in_v[0]), qmap[db0 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[1]), qmap[db0 & 0xF] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[2]), qmap[db1 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[3]), qmap[db1 & 0xF] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[4]), qmap[db2 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[5]), qmap[db2 & 0xF] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[6]), qmap[db3 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[7]), qmap[db3 & 0xF] * da, expert_acc);
                        }
                        float expert_sum = sycl::reduce_over_group(item.get_sub_group(), expert_acc, sycl::plus<>());
                        total_val += w_exp * expert_sum;
                    }
                }

                if (sg_lane == 0 && row < static_cast<size_t>(out_features)) {
                    if (acc_f32) {
                        acc_f32[row] += total_val;
                    }
                    if (out_bf16) {
                        const bf16 routed = bf16(acc_f32 ? acc_f32[row] : total_val);
                        if (shared_out && shared_gate) {
                            const float gate = static_cast<float>(shared_gate[0]);
                            const float gate_act = 1.0f / (1.0f + sycl::native::exp(-gate));
                            out_bf16[row] = bf16(static_cast<float>(routed) +
                                                 gate_act * static_cast<float>(shared_out[row]));
                        } else {
                            out_bf16[row] = routed;
                        }
                    }
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_decode_down_to_rank_f32(
    Context& ctx,
    const std::array<MoeExpertSlotDev, 16>& slot_ptrs,
    const std::array<int, 16>& rank_ptrs,
    const std::array<int, 16>& token_ptrs,
    const bf16* h,
    float* down_acc,
    int num_active_experts,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const float* quant_map,
    const sycl::event* dep_event) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(num_active_experts) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc = 0.0f;
                if (global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const auto& slot = slot_ptrs[static_cast<size_t>(expert_idx)];
                    const int token = token_ptrs[static_cast<size_t>(expert_idx)];
                    const int rank = rank_ptrs[static_cast<size_t>(expert_idx)];
                    const bf16* in_expert = h + static_cast<size_t>(expert_idx) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                        float da;
                        if (blocksize == 64) {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested, FixedNested256>(
                                slot.down_absmax, slot.down_nested, scale_index);
                        }
                        const vec8 in_v = *reinterpret_cast<const vec8*>(in_expert + byte_offset * 2);

                        const uint8_t db0 = static_cast<uint8_t>(d4);
                        const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                        const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                        const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                        expert_acc = sycl::fma(static_cast<float>(in_v[0]), qmap[db0 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[1]), qmap[db0 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[2]), qmap[db1 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[3]), qmap[db1 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[4]), qmap[db2 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[5]), qmap[db2 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[6]), qmap[db3 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[7]), qmap[db3 & 0xF] * da, expert_acc);
                    }
                }

                float expert_sum = sycl::reduce_over_group(item.get_sub_group(), expert_acc, sycl::plus<>());
                if (sg_lane == 0 && global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const int token = token_ptrs[static_cast<size_t>(expert_idx)];
                    const int rank = rank_ptrs[static_cast<size_t>(expert_idx)];
                    const size_t out_pos = (static_cast<size_t>(token) * top_k + rank) * out_features + row;
                    down_acc[out_pos] = expert_sum;
                }
            });
    });
}

template <bool UseNested, bool Resident = false, bool FixedNested256 = false>
sycl::event launch_moe_decode_down_to_rank_f32_dev(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* ranks,
    const int* token_ids,
    const bf16* h,
    float* down_acc,
    int num_active_experts,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const float* quant_map,
    const sycl::event* dep_event, const int* resident_slots = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(num_active_experts) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc = 0.0f;
                if (global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const int slot_index = Resident ? resident_slots[expert_idx] : expert_idx;
                    // All lanes of this subgroup share the rank. The only
                    // work-group barrier is above this uniform subgroup exit.
                    if constexpr (Resident) { if (slot_index < 0) return; }
                    const auto& slot = slots[slot_index];
                    const int token = token_ids ? token_ids[expert_idx] : 0;
                    const int rank = Resident ? expert_idx : ranks[expert_idx];
                    const bf16* in_expert = h + static_cast<size_t>(expert_idx) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                        float da;
                        if (blocksize == 64) {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested, FixedNested256>(
                                slot.down_absmax, slot.down_nested, scale_index);
                        }
                        const vec8 in_v = *reinterpret_cast<const vec8*>(in_expert + byte_offset * 2);

                        const uint8_t db0 = static_cast<uint8_t>(d4);
                        const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                        const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                        const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                        expert_acc = sycl::fma(static_cast<float>(in_v[0]), qmap[db0 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[1]), qmap[db0 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[2]), qmap[db1 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[3]), qmap[db1 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[4]), qmap[db2 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[5]), qmap[db2 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[6]), qmap[db3 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[7]), qmap[db3 & 0xF] * da, expert_acc);
                    }
                }

                float expert_sum = sycl::reduce_over_group(item.get_sub_group(), expert_acc, sycl::plus<>());
                if (sg_lane == 0 && global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const int token = token_ids ? token_ids[expert_idx] : 0;
                    const int rank = Resident ? expert_idx : ranks[expert_idx];
                    const size_t out_pos = (static_cast<size_t>(token) * top_k + rank) * out_features + row;
                    down_acc[out_pos] = expert_sum;
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_prefill_gate_up_rows(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int tokens,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(tokens) * num_active_experts * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                if (global_row < total_rows) {
                    const size_t expert_intermediate = static_cast<size_t>(num_active_experts) * intermediate_size;
                    const int token = static_cast<int>(global_row / expert_intermediate);
                    const size_t rem = global_row % expert_intermediate;
                    const int expert_idx = static_cast<int>(rem / intermediate_size);
                    const int row = static_cast<int>(rem % intermediate_size);

                    const auto& slot = slots[static_cast<size_t>(token) * num_active_experts + expert_idx];
                    const bf16* x_token = x + static_cast<size_t>(token) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const int leader = sg_lane < 8 ? 0 : 8;
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v = *reinterpret_cast<const vec8*>(x_token + byte_offset * 2);

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
                }

                float gate_sum = sycl::reduce_over_group(item.get_sub_group(), gate_acc, sycl::plus<>());
                float up_sum = sycl::reduce_over_group(item.get_sub_group(), up_acc, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const float silu_gate = gate_sum / (1.0f + sycl::native::exp(-gate_sum));
                    h[global_row] = bf16(silu_gate * up_sum);
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_prefill_down_rows(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* weights,
    const float* quant_map,
    const bf16* h,
    bf16* y,
    int tokens,
    int num_active_experts,
    int in_features,
    int out_features,
    int blocksize,
    const bf16* shared_out = nullptr,
    const bf16* shared_gate = nullptr,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(tokens) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float total_val = 0.0f;
                if (global_row < total_rows) {
                    const int token = static_cast<int>(global_row / out_features);
                    const size_t row = global_row % out_features;
                    const size_t byte_base = row * packed_bytes_per_row;
                    const size_t block_base = row * blocks_per_row;
                    const size_t token_expert_base = static_cast<size_t>(token) * num_active_experts;
                    const bf16* h_token = h + token_expert_base * in_features;

                    float exp_weights[16];
                    const int k_capped = std::min(num_active_experts, 16);
                    for (int e = 0; e < k_capped; ++e) {
                        exp_weights[e] = weights[token_expert_base + e];
                    }

                    for (int e = 0; e < num_active_experts; ++e) {
                        const float w_exp = (e < 16) ? exp_weights[e] : weights[token_expert_base + e];
                        if (w_exp == 0.0f) continue;

                        const auto& slot = slots[token_expert_base + e];
                        const bf16* in_expert = h_token + static_cast<size_t>(e) * in_features;

                        float expert_acc = 0.0f;
                        int byte_offset = sg_lane * 4;
                        for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                            const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                                slot.down_packed + byte_base + byte_offset);
                            const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                            float da;
                            if (blocksize == 64) {
                                const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                    ? load_expert_absmax<UseNested, FixedNested256>(
                                          slot.down_absmax, slot.down_nested, scale_index)
                                    : 0.0f;
                                const float da0 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 0);
                                const float da8 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 8);
                                da = sg_lane < 8 ? da0 : da8;
                            } else {
                                da = load_expert_absmax<UseNested, FixedNested256>(
                                    slot.down_absmax, slot.down_nested, scale_index);
                            }
                            const vec8 in_v = *reinterpret_cast<const vec8*>(in_expert + byte_offset * 2);

                            const uint8_t db0 = static_cast<uint8_t>(d4);
                            const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                            const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                            const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                            expert_acc = sycl::fma(static_cast<float>(in_v[0]), qmap[db0 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[1]), qmap[db0 & 0xF] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[2]), qmap[db1 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[3]), qmap[db1 & 0xF] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[4]), qmap[db2 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[5]), qmap[db2 & 0xF] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[6]), qmap[db3 >> 4] * da, expert_acc);
                            expert_acc = sycl::fma(static_cast<float>(in_v[7]), qmap[db3 & 0xF] * da, expert_acc);
                        }
                        float expert_sum = sycl::reduce_over_group(item.get_sub_group(), expert_acc, sycl::plus<>());
                        total_val += w_exp * expert_sum;
                    }
                }

                if (sg_lane == 0 && global_row < total_rows) {
                    const int token = static_cast<int>(global_row / out_features);
                    const bf16 routed = bf16(total_val);
                    if (shared_out && shared_gate) {
                        const float gate = static_cast<float>(shared_gate[token]);
                        const float gate_act = 1.0f / (1.0f + sycl::native::exp(-gate));
                        y[global_row] = bf16(static_cast<float>(routed) +
                                             gate_act * static_cast<float>(shared_out[global_row]));
                    } else {
                        y[global_row] = routed;
                    }
                }
            });
    });
}

// ---------------------------------------------------------------------------
// P9 / AILA_Q4E_PREFILL_ACT_SLM: activation-row staging in SLM.
//
// Every expert-major prefill kernel below maps one 16-lane sub-group to one
// output row, and the 16 sub-groups of a work-group walk 16 consecutive rows.
// The per-activation row count (intermediate_size for gate/up, out_features
// for down) is a multiple of 16 in every production shape, so all 16
// sub-groups of a work-group belong to the same activation, yet each of them
// re-reads that whole activation row from global/L1. With this flag the work
// group stages its activation row(s) once in SLM and every sub-group then
// reads the same element offsets from SLM.
//
// The staged bytes, every consumer index and every FMA/reduce order are
// unchanged, so the kernels stay bit-exact; only the memory space of the
// activation loads changes. Default 0 keeps the historical global reads.
// ---------------------------------------------------------------------------
namespace {
constexpr int kQ4eActSlmMaxBytes = 32 * 1024; // >32 KB/WG costs residency [A770]
bool q4e_prefill_act_slm_enabled() {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_PREFILL_ACT_SLM", false);
    return enabled;
}
// rows_per_group == wg_size / sub_group_size == 16 in all of these kernels.
// The staging is only valid when one work-group serves one activation (or one
// quad of activations), which needs features_per_act % rows_per_group == 0;
// the cooperative vec8 copy needs in_features % 8 == 0.
bool q4e_act_slm_shape_ok(int in_features, int features_per_act, int acts_per_wg) {
    if (in_features <= 0 || features_per_act <= 0) return false;
    if ((in_features % 8) != 0) return false;
    if ((features_per_act % 16) != 0) return false;
    const int tile_bytes = acts_per_wg * in_features * static_cast<int>(sizeof(bf16));
    return tile_bytes <= kQ4eActSlmMaxBytes;
}
} // namespace

template <bool UseNested, bool FixedNested256 = false, bool ActSlm = false>
sycl::event launch_moe_expert_major_prefill_gate_up(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_acts,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_acts) * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        // P9: this work-group's activation row, staged once for all 16 sub-groups.
        sycl::local_accessor<bf16, 1> x_slm(
            sycl::range<1>(ActSlm ? in_features : 1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const size_t wg_base_row = item.get_group(0) * rows_per_group; // uniform
                const bool wg_active = wg_base_row < total_rows;               // uniform
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                bf16* const x_slm_ptr = x_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (ActSlm) {
                    // Every lane of this work-group consumes act_idx =
                    // wg_base_row / intermediate_size: the dispatcher only enables
                    // this path when intermediate_size % rows_per_group == 0, so no
                    // work-group straddles two activations.
                    if (wg_active) {
                        const int act_idx = static_cast<int>(wg_base_row / intermediate_size);
                        const bf16* src = x + static_cast<size_t>(token_ids[act_idx]) * in_features;
                        for (int e = lid * 8; e < in_features;
                             e += static_cast<int>(wg_size) * 8) {
                            *reinterpret_cast<vec8*>(x_slm_ptr + e) =
                                *reinterpret_cast<const vec8*>(src + e);
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                if (global_row < total_rows) {
                    const int act_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);

                    const auto& slot = slots[act_idx];
                    const int token = token_ids[act_idx];
                    const bf16* x_token = x + static_cast<size_t>(token) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        vec8 in_v;
                        if constexpr (ActSlm) {
                            in_v = *reinterpret_cast<const vec8*>(x_slm_ptr + byte_offset * 2);
                        } else {
                            in_v = *reinterpret_cast<const vec8*>(x_token + byte_offset * 2);
                        }

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
                }

                float gate_sum = sycl::reduce_over_group(item.get_sub_group(), gate_acc, sycl::plus<>());
                float up_sum = sycl::reduce_over_group(item.get_sub_group(), up_acc, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const int act_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);
                    const int token = token_ids[act_idx];
                    const int rank = ranks[act_idx];
                    const float silu_gate = gate_sum / (1.0f + sycl::native::exp(-gate_sum));
                    const size_t out_pos = (static_cast<size_t>(token) * top_k + rank) * intermediate_size + row;
                    h[out_pos] = bf16(silu_gate * up_sum);
                }
            });
    });
}

template <bool UseNested, bool Compact = false, bool FixedNested256 = false, bool ActSlm = false>
sycl::event launch_moe_expert_major_prefill_down(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_acts,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_acts) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        // P9: this work-group's h row, staged once for all 16 sub-groups.
        sycl::local_accessor<bf16, 1> x_slm(
            sycl::range<1>(ActSlm ? in_features : 1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const size_t wg_base_row = item.get_group(0) * rows_per_group; // uniform
                const bool wg_active = wg_base_row < total_rows;               // uniform
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                bf16* const x_slm_ptr = x_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (ActSlm) {
                    // Every lane of this work-group consumes act_idx =
                    // wg_base_row / out_features: the dispatcher only enables this
                    // path when out_features % rows_per_group == 0, so no work-group
                    // straddles two activations.
                    if (wg_active) {
                        const int act_idx = static_cast<int>(wg_base_row / out_features);
                        const int act_token = token_ids[act_idx];
                        const int act_rank = ranks[act_idx];
                        const bf16* src = h +
                            (static_cast<size_t>(act_token) * top_k + act_rank) * in_features;
                        for (int e = lid * 8; e < in_features;
                             e += static_cast<int>(wg_size) * 8) {
                            *reinterpret_cast<vec8*>(x_slm_ptr + e) =
                                *reinterpret_cast<const vec8*>(src + e);
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc = 0.0f;
                if (global_row < total_rows) {
                    const int act_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);

                    const auto& slot = slots[act_idx];
                    const int token = token_ids[act_idx];
                    const int rank = ranks[act_idx];
                    const bf16* in_expert = h + (static_cast<size_t>(token) * top_k + rank) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                        float da;
                        if (blocksize == 64) {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested, FixedNested256>(
                                slot.down_absmax, slot.down_nested, scale_index);
                        }
                        vec8 in_v;
                        if constexpr (ActSlm) {
                            in_v = *reinterpret_cast<const vec8*>(x_slm_ptr + byte_offset * 2);
                        } else {
                            in_v = *reinterpret_cast<const vec8*>(in_expert + byte_offset * 2);
                        }

                        const uint8_t db0 = static_cast<uint8_t>(d4);
                        const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                        const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                        const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                        expert_acc = sycl::fma(static_cast<float>(in_v[0]), qmap[db0 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[1]), qmap[db0 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[2]), qmap[db1 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[3]), qmap[db1 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[4]), qmap[db2 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[5]), qmap[db2 & 0xF] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[6]), qmap[db3 >> 4] * da, expert_acc);
                        expert_acc = sycl::fma(static_cast<float>(in_v[7]), qmap[db3 & 0xF] * da, expert_acc);
                    }
                }

                float expert_sum = sycl::reduce_over_group(item.get_sub_group(), expert_acc, sycl::plus<>());
                if (sg_lane == 0 && global_row < total_rows) {
                    const int act_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const int token = token_ids[act_idx];
                    const int rank = ranks[act_idx];
                    const size_t out_pos = (Compact ? size_t(act_idx) : size_t(token) * top_k + rank) * out_features + row;
                    down_acc[out_pos] = expert_sum;
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_expert_major_prefill_gate_up_2token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_pairs,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_pairs) * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc0 = 0.0f;
                float up_acc0 = 0.0f;
                float gate_acc1 = 0.0f;
                float up_acc1 = 0.0f;

                if (global_row < total_rows) {
                    const int pair_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);

                    const auto& slot = slots[pair_idx * 2];
                    const int token0 = token_ids[pair_idx * 2];
                    const int token1 = token_ids[pair_idx * 2 + 1];
                    const bf16* x_token0 = x + static_cast<size_t>(token0) * in_features;
                    const bf16* x_token1 = x + static_cast<size_t>(token1) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        const float gw0 = qmap[gb0 >> 4] * ga;
                        const float gw1 = qmap[gb0 & 0xF] * ga;
                        const float gw2 = qmap[gb1 >> 4] * ga;
                        const float gw3 = qmap[gb1 & 0xF] * ga;
                        const float gw4 = qmap[gb2 >> 4] * ga;
                        const float gw5 = qmap[gb2 & 0xF] * ga;
                        const float gw6 = qmap[gb3 >> 4] * ga;
                        const float gw7 = qmap[gb3 & 0xF] * ga;

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);

                        const float uw0 = qmap[ub0 >> 4] * ua;
                        const float uw1 = qmap[ub0 & 0xF] * ua;
                        const float uw2 = qmap[ub1 >> 4] * ua;
                        const float uw3 = qmap[ub1 & 0xF] * ua;
                        const float uw4 = qmap[ub2 >> 4] * ua;
                        const float uw5 = qmap[ub2 & 0xF] * ua;
                        const float uw6 = qmap[ub3 >> 4] * ua;
                        const float uw7 = qmap[ub3 & 0xF] * ua;

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                    }
                }

                float gate_sum0 = sycl::reduce_over_group(item.get_sub_group(), gate_acc0, sycl::plus<>());
                float up_sum0 = sycl::reduce_over_group(item.get_sub_group(), up_acc0, sycl::plus<>());
                float gate_sum1 = sycl::reduce_over_group(item.get_sub_group(), gate_acc1, sycl::plus<>());
                float up_sum1 = sycl::reduce_over_group(item.get_sub_group(), up_acc1, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const int pair_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);

                    const int token0 = token_ids[pair_idx * 2];
                    const int rank0 = ranks[pair_idx * 2];
                    const float silu_gate0 = gate_sum0 / (1.0f + sycl::native::exp(-gate_sum0));
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * intermediate_size + row;
                    h[out_pos0] = bf16(silu_gate0 * up_sum0);

                    const int token1 = token_ids[pair_idx * 2 + 1];
                    const int rank1 = ranks[pair_idx * 2 + 1];
                    const float silu_gate1 = gate_sum1 / (1.0f + sycl::native::exp(-gate_sum1));
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * intermediate_size + row;
                    h[out_pos1] = bf16(silu_gate1 * up_sum1);
                }
            });
    });
}

template <bool UseNested, bool Compact = false, bool FixedNested256 = false>
sycl::event launch_moe_expert_major_prefill_down_2token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_pairs,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_pairs) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc0 = 0.0f;
                float expert_acc1 = 0.0f;

                if (global_row < total_rows) {
                    const int pair_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);

                    const auto& slot = slots[pair_idx * 2];
                    const int token0 = token_ids[pair_idx * 2];
                    const int rank0 = ranks[pair_idx * 2];
                    const int token1 = token_ids[pair_idx * 2 + 1];
                    const int rank1 = ranks[pair_idx * 2 + 1];

                    const bf16* in_expert0 = h + (static_cast<size_t>(token0) * top_k + rank0) * in_features;
                    const bf16* in_expert1 = h + (static_cast<size_t>(token1) * top_k + rank1) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                        float da;
                        if (blocksize == 64) {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested, FixedNested256>(
                                slot.down_absmax, slot.down_nested, scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(in_expert0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(in_expert1 + byte_offset * 2);

                        const uint8_t db0 = static_cast<uint8_t>(d4);
                        const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                        const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                        const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                        const float dw0 = qmap[db0 >> 4] * da;
                        const float dw1 = qmap[db0 & 0xF] * da;
                        const float dw2 = qmap[db1 >> 4] * da;
                        const float dw3 = qmap[db1 & 0xF] * da;
                        const float dw4 = qmap[db2 >> 4] * da;
                        const float dw5 = qmap[db2 & 0xF] * da;
                        const float dw6 = qmap[db3 >> 4] * da;
                        const float dw7 = qmap[db3 & 0xF] * da;

                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[0]), dw0, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[0]), dw0, expert_acc1);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[1]), dw1, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[1]), dw1, expert_acc1);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[2]), dw2, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[2]), dw2, expert_acc1);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[3]), dw3, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[3]), dw3, expert_acc1);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[4]), dw4, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[4]), dw4, expert_acc1);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[5]), dw5, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[5]), dw5, expert_acc1);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[6]), dw6, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[6]), dw6, expert_acc1);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[7]), dw7, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[7]), dw7, expert_acc1);
                    }
                }

                float expert_sum0 = sycl::reduce_over_group(item.get_sub_group(), expert_acc0, sycl::plus<>());
                float expert_sum1 = sycl::reduce_over_group(item.get_sub_group(), expert_acc1, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const int pair_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);

                    const int token0 = token_ids[pair_idx * 2];
                    const int rank0 = ranks[pair_idx * 2];
                    const size_t out_pos0 = (Compact ? size_t(pair_idx * 2 + 0) : size_t(token0) * top_k + rank0) * out_features + row;
                    down_acc[out_pos0] = expert_sum0;

                    const int token1 = token_ids[pair_idx * 2 + 1];
                    const int rank1 = ranks[pair_idx * 2 + 1];
                    const size_t out_pos1 = (Compact ? size_t(pair_idx * 2 + 1) : size_t(token1) * top_k + rank1) * out_features + row;
                    down_acc[out_pos1] = expert_sum1;
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false, bool ActSlm = false>
sycl::event launch_moe_expert_major_prefill_gate_up_4token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_quads) * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        // P9: the quad's four activation rows, staged once for all 16 sub-groups.
        sycl::local_accessor<bf16, 1> x_slm(
            sycl::range<1>(ActSlm ? 4 * in_features : 1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const size_t wg_base_row = item.get_group(0) * rows_per_group; // uniform
                const bool wg_active = wg_base_row < total_rows;               // uniform
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                bf16* const x_slm_ptr = x_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (ActSlm) {
                    // quad_idx is work-group uniform: the dispatcher only enables
                    // this path when intermediate_size % rows_per_group == 0.
                    if (wg_active) {
                        const int quad_idx = static_cast<int>(wg_base_row / intermediate_size);
                        const int base = quad_idx * 4;
                        const bf16* src0 = x + static_cast<size_t>(token_ids[base + 0]) * in_features;
                        const bf16* src1 = x + static_cast<size_t>(token_ids[base + 1]) * in_features;
                        const bf16* src2 = x + static_cast<size_t>(token_ids[base + 2]) * in_features;
                        const bf16* src3 = x + static_cast<size_t>(token_ids[base + 3]) * in_features;
                        for (int e = lid * 8; e < in_features;
                             e += static_cast<int>(wg_size) * 8) {
                            *reinterpret_cast<vec8*>(x_slm_ptr + e) =
                                *reinterpret_cast<const vec8*>(src0 + e);
                            *reinterpret_cast<vec8*>(x_slm_ptr + in_features + e) =
                                *reinterpret_cast<const vec8*>(src1 + e);
                            *reinterpret_cast<vec8*>(x_slm_ptr + 2 * in_features + e) =
                                *reinterpret_cast<const vec8*>(src2 + e);
                            *reinterpret_cast<vec8*>(x_slm_ptr + 3 * in_features + e) =
                                *reinterpret_cast<const vec8*>(src3 + e);
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc0 = 0.0f;
                float up_acc0 = 0.0f;
                float gate_acc1 = 0.0f;
                float up_acc1 = 0.0f;
                float gate_acc2 = 0.0f;
                float up_acc2 = 0.0f;
                float gate_acc3 = 0.0f;
                float up_acc3 = 0.0f;

                if (global_row < total_rows) {
                    const int quad_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);

                    const auto& slot = slots[quad_idx * 4];
                    const int token0 = token_ids[quad_idx * 4];
                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const bf16* x_token0 = x + static_cast<size_t>(token0) * in_features;
                    const bf16* x_token1 = x + static_cast<size_t>(token1) * in_features;
                    const bf16* x_token2 = x + static_cast<size_t>(token2) * in_features;
                    const bf16* x_token3 = x + static_cast<size_t>(token3) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        vec8 in_v0;
                        vec8 in_v1;
                        vec8 in_v2;
                        vec8 in_v3;
                        if constexpr (ActSlm) {
                            const int elem_offset = byte_offset * 2;
                            in_v0 = *reinterpret_cast<const vec8*>(x_slm_ptr + elem_offset);
                            in_v1 = *reinterpret_cast<const vec8*>(x_slm_ptr + in_features + elem_offset);
                            in_v2 = *reinterpret_cast<const vec8*>(x_slm_ptr + 2 * in_features + elem_offset);
                            in_v3 = *reinterpret_cast<const vec8*>(x_slm_ptr + 3 * in_features + elem_offset);
                        } else {
                            in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                            in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);
                            in_v2 = *reinterpret_cast<const vec8*>(x_token2 + byte_offset * 2);
                            in_v3 = *reinterpret_cast<const vec8*>(x_token3 + byte_offset * 2);
                        }

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        const float gw0 = qmap[gb0 >> 4] * ga;
                        const float gw1 = qmap[gb0 & 0xF] * ga;
                        const float gw2 = qmap[gb1 >> 4] * ga;
                        const float gw3 = qmap[gb1 & 0xF] * ga;
                        const float gw4 = qmap[gb2 >> 4] * ga;
                        const float gw5 = qmap[gb2 & 0xF] * ga;
                        const float gw6 = qmap[gb3 >> 4] * ga;
                        const float gw7 = qmap[gb3 & 0xF] * ga;

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[0]), gw0, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[0]), gw0, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[1]), gw1, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[1]), gw1, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[2]), gw2, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[2]), gw2, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[3]), gw3, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[3]), gw3, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[4]), gw4, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[4]), gw4, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[5]), gw5, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[5]), gw5, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[6]), gw6, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[6]), gw6, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[7]), gw7, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[7]), gw7, gate_acc3);

                        const float uw0 = qmap[ub0 >> 4] * ua;
                        const float uw1 = qmap[ub0 & 0xF] * ua;
                        const float uw2 = qmap[ub1 >> 4] * ua;
                        const float uw3 = qmap[ub1 & 0xF] * ua;
                        const float uw4 = qmap[ub2 >> 4] * ua;
                        const float uw5 = qmap[ub2 & 0xF] * ua;
                        const float uw6 = qmap[ub3 >> 4] * ua;
                        const float uw7 = qmap[ub3 & 0xF] * ua;

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[0]), uw0, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[0]), uw0, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[1]), uw1, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[1]), uw1, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[2]), uw2, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[2]), uw2, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[3]), uw3, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[3]), uw3, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[4]), uw4, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[4]), uw4, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[5]), uw5, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[5]), uw5, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[6]), uw6, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[6]), uw6, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[7]), uw7, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[7]), uw7, up_acc3);
                    }
                }

                float gate_sum0 = sycl::reduce_over_group(item.get_sub_group(), gate_acc0, sycl::plus<>());
                float up_sum0 = sycl::reduce_over_group(item.get_sub_group(), up_acc0, sycl::plus<>());
                float gate_sum1 = sycl::reduce_over_group(item.get_sub_group(), gate_acc1, sycl::plus<>());
                float up_sum1 = sycl::reduce_over_group(item.get_sub_group(), up_acc1, sycl::plus<>());
                float gate_sum2 = sycl::reduce_over_group(item.get_sub_group(), gate_acc2, sycl::plus<>());
                float up_sum2 = sycl::reduce_over_group(item.get_sub_group(), up_acc2, sycl::plus<>());
                float gate_sum3 = sycl::reduce_over_group(item.get_sub_group(), gate_acc3, sycl::plus<>());
                float up_sum3 = sycl::reduce_over_group(item.get_sub_group(), up_acc3, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const int quad_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);

                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const float silu_gate0 = gate_sum0 / (1.0f + sycl::native::exp(-gate_sum0));
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * intermediate_size + row;
                    h[out_pos0] = bf16(silu_gate0 * up_sum0);

                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const float silu_gate1 = gate_sum1 / (1.0f + sycl::native::exp(-gate_sum1));
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * intermediate_size + row;
                    h[out_pos1] = bf16(silu_gate1 * up_sum1);

                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const float silu_gate2 = gate_sum2 / (1.0f + sycl::native::exp(-gate_sum2));
                    const size_t out_pos2 = (static_cast<size_t>(token2) * top_k + rank2) * intermediate_size + row;
                    h[out_pos2] = bf16(silu_gate2 * up_sum2);

                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];
                    const float silu_gate3 = gate_sum3 / (1.0f + sycl::native::exp(-gate_sum3));
                    const size_t out_pos3 = (static_cast<size_t>(token3) * top_k + rank3) * intermediate_size + row;
                    h[out_pos3] = bf16(silu_gate3 * up_sum3);
                }
            });
    });
}

// Phase Q4E WG-share: same per-(quad,row) arithmetic and reduction order as the
// 4-token triple, but the workgroup mapping is regrouped so that ShareQuads
// subgroups stream the same weight rows for different token quads. Every
// subgroup still owns exactly one (quad, row) pair, so the per-token FMA
// chains, scale broadcasts and reduce_over_group order are unchanged and the
// results stay bit-exact against the 4-token path.
template <bool UseNested, bool FixedNested256 = false, int ShareQuads = 2>
sycl::event launch_moe_expert_major_prefill_gate_up_4token_wgshare(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_wg = (wg_size / sub_group_size) / static_cast<size_t>(ShareQuads);
    static_assert(ShareQuads == 2 || ShareQuads == 4 || ShareQuads == 8,
                  "ShareQuads must be 2, 4 or 8");
    const int row_blocks = (intermediate_size + static_cast<int>(rows_per_wg) - 1) / static_cast<int>(rows_per_wg);
    const int quad_groups = (total_quads + ShareQuads - 1) / ShareQuads;
    const size_t num_groups = static_cast<size_t>(row_blocks) * static_cast<size_t>(quad_groups);

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const int row_block = static_cast<int>(item.get_group(0)) / quad_groups;
                const int quad_group = static_cast<int>(item.get_group(0)) % quad_groups;
                const int row = row_block * static_cast<int>(rows_per_wg) + sg_id / ShareQuads;
                const int quad_idx = quad_group * ShareQuads + sg_id % ShareQuads;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc0 = 0.0f;
                float up_acc0 = 0.0f;
                float gate_acc1 = 0.0f;
                float up_acc1 = 0.0f;
                float gate_acc2 = 0.0f;
                float up_acc2 = 0.0f;
                float gate_acc3 = 0.0f;
                float up_acc3 = 0.0f;

                if (row < intermediate_size && quad_idx < total_quads) {

                    const auto& slot = slots[quad_idx * 4];
                    const int token0 = token_ids[quad_idx * 4];
                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const bf16* x_token0 = x + static_cast<size_t>(token0) * in_features;
                    const bf16* x_token1 = x + static_cast<size_t>(token1) * in_features;
                    const bf16* x_token2 = x + static_cast<size_t>(token2) * in_features;
                    const bf16* x_token3 = x + static_cast<size_t>(token3) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(x_token2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(x_token3 + byte_offset * 2);

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        const float gw0 = qmap[gb0 >> 4] * ga;
                        const float gw1 = qmap[gb0 & 0xF] * ga;
                        const float gw2 = qmap[gb1 >> 4] * ga;
                        const float gw3 = qmap[gb1 & 0xF] * ga;
                        const float gw4 = qmap[gb2 >> 4] * ga;
                        const float gw5 = qmap[gb2 & 0xF] * ga;
                        const float gw6 = qmap[gb3 >> 4] * ga;
                        const float gw7 = qmap[gb3 & 0xF] * ga;

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[0]), gw0, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[0]), gw0, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[1]), gw1, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[1]), gw1, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[2]), gw2, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[2]), gw2, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[3]), gw3, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[3]), gw3, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[4]), gw4, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[4]), gw4, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[5]), gw5, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[5]), gw5, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[6]), gw6, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[6]), gw6, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[7]), gw7, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[7]), gw7, gate_acc3);

                        const float uw0 = qmap[ub0 >> 4] * ua;
                        const float uw1 = qmap[ub0 & 0xF] * ua;
                        const float uw2 = qmap[ub1 >> 4] * ua;
                        const float uw3 = qmap[ub1 & 0xF] * ua;
                        const float uw4 = qmap[ub2 >> 4] * ua;
                        const float uw5 = qmap[ub2 & 0xF] * ua;
                        const float uw6 = qmap[ub3 >> 4] * ua;
                        const float uw7 = qmap[ub3 & 0xF] * ua;

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[0]), uw0, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[0]), uw0, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[1]), uw1, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[1]), uw1, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[2]), uw2, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[2]), uw2, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[3]), uw3, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[3]), uw3, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[4]), uw4, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[4]), uw4, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[5]), uw5, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[5]), uw5, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[6]), uw6, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[6]), uw6, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[7]), uw7, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[7]), uw7, up_acc3);
                    }
                }

                float gate_sum0 = sycl::reduce_over_group(item.get_sub_group(), gate_acc0, sycl::plus<>());
                float up_sum0 = sycl::reduce_over_group(item.get_sub_group(), up_acc0, sycl::plus<>());
                float gate_sum1 = sycl::reduce_over_group(item.get_sub_group(), gate_acc1, sycl::plus<>());
                float up_sum1 = sycl::reduce_over_group(item.get_sub_group(), up_acc1, sycl::plus<>());
                float gate_sum2 = sycl::reduce_over_group(item.get_sub_group(), gate_acc2, sycl::plus<>());
                float up_sum2 = sycl::reduce_over_group(item.get_sub_group(), up_acc2, sycl::plus<>());
                float gate_sum3 = sycl::reduce_over_group(item.get_sub_group(), gate_acc3, sycl::plus<>());
                float up_sum3 = sycl::reduce_over_group(item.get_sub_group(), up_acc3, sycl::plus<>());

                if (sg_lane == 0 && row < intermediate_size && quad_idx < total_quads) {

                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const float silu_gate0 = gate_sum0 / (1.0f + sycl::native::exp(-gate_sum0));
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * intermediate_size + row;
                    h[out_pos0] = bf16(silu_gate0 * up_sum0);

                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const float silu_gate1 = gate_sum1 / (1.0f + sycl::native::exp(-gate_sum1));
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * intermediate_size + row;
                    h[out_pos1] = bf16(silu_gate1 * up_sum1);

                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const float silu_gate2 = gate_sum2 / (1.0f + sycl::native::exp(-gate_sum2));
                    const size_t out_pos2 = (static_cast<size_t>(token2) * top_k + rank2) * intermediate_size + row;
                    h[out_pos2] = bf16(silu_gate2 * up_sum2);

                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];
                    const float silu_gate3 = gate_sum3 / (1.0f + sycl::native::exp(-gate_sum3));
                    const size_t out_pos3 = (static_cast<size_t>(token3) * top_k + rank3) * intermediate_size + row;
                    h[out_pos3] = bf16(silu_gate3 * up_sum3);
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false, int ShareQuads = 2>
sycl::event launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_wg = (wg_size / sub_group_size) / static_cast<size_t>(ShareQuads);
    static_assert(ShareQuads == 2 || ShareQuads == 4 || ShareQuads == 8,
                  "ShareQuads must be 2, 4 or 8");
    const int row_blocks = (intermediate_size + static_cast<int>(rows_per_wg) - 1) / static_cast<int>(rows_per_wg);
    const int quad_groups = (total_quads + ShareQuads - 1) / ShareQuads;
    const size_t num_groups = static_cast<size_t>(row_blocks) * static_cast<size_t>(quad_groups);

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const int row_block = static_cast<int>(item.get_group(0)) / quad_groups;
                const int quad_group = static_cast<int>(item.get_group(0)) % quad_groups;
                const int row = row_block * static_cast<int>(rows_per_wg) + sg_id / ShareQuads;
                const int quad_idx = quad_group * ShareQuads + sg_id % ShareQuads;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc0 = 0.0f;
                float up_acc0 = 0.0f;
                float gate_acc1 = 0.0f;
                float up_acc1 = 0.0f;
                float gate_acc2 = 0.0f;
                float up_acc2 = 0.0f;
                float gate_acc3 = 0.0f;
                float up_acc3 = 0.0f;

                if (row < intermediate_size && quad_idx < total_quads) {

                    const auto& slot = slots[quad_idx * 4];
                    const int token0 = token_ids[quad_idx * 4];
                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const bf16* x_token0 = x + static_cast<size_t>(token0) * in_features;
                    const bf16* x_token1 = x + static_cast<size_t>(token1) * in_features;
                    const bf16* x_token2 = x + static_cast<size_t>(token2) * in_features;
                    const bf16* x_token3 = x + static_cast<size_t>(token3) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(x_token2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(x_token3 + byte_offset * 2);

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        const float gw0 = static_cast<float>(gb0 >> 4) *  ga;
                        const float gw1 = static_cast<float>(gb0 & 0xF) *  ga;
                        const float gw2 = static_cast<float>(gb1 >> 4) *  ga;
                        const float gw3 = static_cast<float>(gb1 & 0xF) *  ga;
                        const float gw4 = static_cast<float>(gb2 >> 4) *  ga;
                        const float gw5 = static_cast<float>(gb2 & 0xF) *  ga;
                        const float gw6 = static_cast<float>(gb3 >> 4) *  ga;
                        const float gw7 = static_cast<float>(gb3 & 0xF) *  ga;

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[0]), gw0, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[0]), gw0, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[1]), gw1, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[1]), gw1, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[2]), gw2, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[2]), gw2, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[3]), gw3, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[3]), gw3, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[4]), gw4, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[4]), gw4, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[5]), gw5, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[5]), gw5, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[6]), gw6, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[6]), gw6, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[7]), gw7, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[7]), gw7, gate_acc3);

                        const float uw0 = static_cast<float>(ub0 >> 4) *  ua;
                        const float uw1 = static_cast<float>(ub0 & 0xF) *  ua;
                        const float uw2 = static_cast<float>(ub1 >> 4) *  ua;
                        const float uw3 = static_cast<float>(ub1 & 0xF) *  ua;
                        const float uw4 = static_cast<float>(ub2 >> 4) *  ua;
                        const float uw5 = static_cast<float>(ub2 & 0xF) *  ua;
                        const float uw6 = static_cast<float>(ub3 >> 4) *  ua;
                        const float uw7 = static_cast<float>(ub3 & 0xF) *  ua;

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[0]), uw0, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[0]), uw0, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[1]), uw1, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[1]), uw1, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[2]), uw2, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[2]), uw2, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[3]), uw3, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[3]), uw3, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[4]), uw4, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[4]), uw4, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[5]), uw5, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[5]), uw5, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[6]), uw6, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[6]), uw6, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[7]), uw7, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[7]), uw7, up_acc3);
                    }
                }

                float gate_sum0 = sycl::reduce_over_group(item.get_sub_group(), gate_acc0, sycl::plus<>());
                float up_sum0 = sycl::reduce_over_group(item.get_sub_group(), up_acc0, sycl::plus<>());
                float gate_sum1 = sycl::reduce_over_group(item.get_sub_group(), gate_acc1, sycl::plus<>());
                float up_sum1 = sycl::reduce_over_group(item.get_sub_group(), up_acc1, sycl::plus<>());
                float gate_sum2 = sycl::reduce_over_group(item.get_sub_group(), gate_acc2, sycl::plus<>());
                float up_sum2 = sycl::reduce_over_group(item.get_sub_group(), up_acc2, sycl::plus<>());
                float gate_sum3 = sycl::reduce_over_group(item.get_sub_group(), gate_acc3, sycl::plus<>());
                float up_sum3 = sycl::reduce_over_group(item.get_sub_group(), up_acc3, sycl::plus<>());

                if (sg_lane == 0 && row < intermediate_size && quad_idx < total_quads) {

                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const float silu_gate0 = gate_sum0 / (1.0f + sycl::native::exp(-gate_sum0));
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * intermediate_size + row;
                    h[out_pos0] = bf16(silu_gate0 * up_sum0);

                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const float silu_gate1 = gate_sum1 / (1.0f + sycl::native::exp(-gate_sum1));
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * intermediate_size + row;
                    h[out_pos1] = bf16(silu_gate1 * up_sum1);

                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const float silu_gate2 = gate_sum2 / (1.0f + sycl::native::exp(-gate_sum2));
                    const size_t out_pos2 = (static_cast<size_t>(token2) * top_k + rank2) * intermediate_size + row;
                    h[out_pos2] = bf16(silu_gate2 * up_sum2);

                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];
                    const float silu_gate3 = gate_sum3 / (1.0f + sycl::native::exp(-gate_sum3));
                    const size_t out_pos3 = (static_cast<size_t>(token3) * top_k + rank3) * intermediate_size + row;
                    h[out_pos3] = bf16(silu_gate3 * up_sum3);
                }
            });
    });
}


template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_expert_major_prefill_gate_up_4token_wgshare_slm(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    constexpr int ShareQuads = 4;
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_wg = (wg_size / sub_group_size) / static_cast<size_t>(ShareQuads);
    const int row_blocks = (intermediate_size + static_cast<int>(rows_per_wg) - 1) / static_cast<int>(rows_per_wg);
    const int quad_groups = (total_quads + ShareQuads - 1) / ShareQuads;
    const size_t num_groups = static_cast<size_t>(row_blocks) * static_cast<size_t>(quad_groups);
    const int stride = static_cast<int>(sub_group_size) * 4;
    // Full 64-byte chunks are processed with a uniform trip count so the
    // producer/consumer barriers stay workgroup-uniform; only the per-lane
    // remainder uses the original per-lane loop bound.
    const int full_tiles = packed_bytes_per_row / 64;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        // [row group][gate|up][16 lanes * 9-float padded tile] = 1152 floats.
        sycl::local_accessor<float, 1> weight_tile(
            sycl::range<1>(rows_per_wg * 2 * sub_group_size * 9), cgh);
        sycl::local_accessor<int, 1> expert_shared(sycl::range<1>(rows_per_wg), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / static_cast<int>(sub_group_size);
                const int sg_lane = lid % static_cast<int>(sub_group_size);
                const int row_group = sg_id / ShareQuads;
                const int sg_in_group = sg_id % ShareQuads;
                const int row_block = static_cast<int>(item.get_group(0)) / quad_groups;
                const int quad_group = static_cast<int>(item.get_group(0)) % quad_groups;
                const int row = row_block * static_cast<int>(rows_per_wg) + row_group;
                const int quad_idx = quad_group * ShareQuads + sg_in_group;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                if (lid < static_cast<int>(rows_per_wg)) {
                    int same = 0;
                    const int first_quad = quad_group * ShareQuads;
                    if (first_quad < total_quads) {
                        same = 1;
                        const uint8_t* base = slots[first_quad * 4].gate_up_packed;
                        for (int i = 1; i < ShareQuads; ++i) {
                            if (first_quad + i >= total_quads ||
                                slots[(first_quad + i) * 4].gate_up_packed != base) {
                                same = 0;
                                break;
                            }
                        }
                    }
                    expert_shared[lid] = same;
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc0 = 0.0f;
                float up_acc0 = 0.0f;
                float gate_acc1 = 0.0f;
                float up_acc1 = 0.0f;
                float gate_acc2 = 0.0f;
                float up_acc2 = 0.0f;
                float gate_acc3 = 0.0f;
                float up_acc3 = 0.0f;

                const bool row_valid = row < intermediate_size;
                const bool consume = row_valid && quad_idx < total_quads;
                const bool shared_path =
                    row_valid && blocksize == 64 && expert_shared[row_group] != 0;
                const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                const size_t up_byte_base =
                    static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                const size_t up_block_base =
                    static_cast<size_t>(row + intermediate_size) * blocks_per_row;
                float* gate_tile = &weight_tile[row_group * 2 * static_cast<int>(sub_group_size) * 9];
                float* up_tile = &weight_tile[(row_group * 2 + 1) * static_cast<int>(sub_group_size) * 9];
                const bf16* x_token0 = x;
                const bf16* x_token1 = x;
                const bf16* x_token2 = x;
                const bf16* x_token3 = x;
                if (consume) {
                    x_token0 = x + static_cast<size_t>(token_ids[quad_idx * 4]) * in_features;
                    x_token1 = x + static_cast<size_t>(token_ids[quad_idx * 4 + 1]) * in_features;
                    x_token2 = x + static_cast<size_t>(token_ids[quad_idx * 4 + 2]) * in_features;
                    x_token3 = x + static_cast<size_t>(token_ids[quad_idx * 4 + 3]) * in_features;
                }

                for (int t = 0; t < full_tiles; ++t) {
                    const int byte_offset = sg_lane * 4 + 64 * t;
                    if (shared_path) {
                        const auto& slot = slots[(quad_group * ShareQuads) * 4];
                        float ga_lo = 0.0f;
                        float ga_hi = 0.0f;
                        float ua_lo = 0.0f;
                        float ua_hi = 0.0f;
                        const size_t gate_scale = gate_block_base + 2 * t;
                        const size_t up_scale = up_block_base + 2 * t;
                        if (sg_lane == 0) {
                            ga_lo = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale);
                        } else if (sg_lane == 1) {
                            ga_hi = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale + 1);
                        } else if (sg_lane == 2) {
                            ua_lo = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale);
                        } else if (sg_lane == 3) {
                            ua_hi = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale + 1);
                        }
                        const auto prod_sg = item.get_sub_group();
                        ga_lo = sycl::group_broadcast(prod_sg, ga_lo, 0);
                        ga_hi = sycl::group_broadcast(prod_sg, ga_hi, 1);
                        ua_lo = sycl::group_broadcast(prod_sg, ua_lo, 2);
                        ua_hi = sycl::group_broadcast(prod_sg, ua_hi, 3);
                        const int producer_lane =
                            sg_in_group * static_cast<int>(sub_group_size) + sg_lane;
                        const float ga_prod = producer_lane < 32 ? ga_lo : ga_hi;
                        const float ua_prod = producer_lane < 32 ? ua_lo : ua_hi;
                        const uint8_t gate_byte = slot.gate_up_packed[
                            gate_byte_base + static_cast<size_t>(64 * t + producer_lane)];
                        const uint8_t up_byte = slot.gate_up_packed[
                            up_byte_base + static_cast<size_t>(64 * t + producer_lane)];
                        const int tile_slot = (producer_lane / 4) * 9 + 2 * (producer_lane % 4);
                        gate_tile[tile_slot] = qmap[gate_byte >> 4] * ga_prod;
                        gate_tile[tile_slot + 1] = qmap[gate_byte & 0xF] * ga_prod;
                        up_tile[tile_slot] = qmap[up_byte >> 4] * ua_prod;
                        up_tile[tile_slot + 1] = qmap[up_byte & 0xF] * ua_prod;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float gw0 = 0.0f;
                    float gw1 = 0.0f;
                    float gw2 = 0.0f;
                    float gw3 = 0.0f;
                    float gw4 = 0.0f;
                    float gw5 = 0.0f;
                    float gw6 = 0.0f;
                    float gw7 = 0.0f;
                    float uw0 = 0.0f;
                    float uw1 = 0.0f;
                    float uw2 = 0.0f;
                    float uw3 = 0.0f;
                    float uw4 = 0.0f;
                    float uw5 = 0.0f;
                    float uw6 = 0.0f;
                    float uw7 = 0.0f;
                    if (shared_path) {
                        if (consume) {
                            const int tile_base = sg_lane * 9;
                            gw0 = gate_tile[tile_base + 0];
                            gw1 = gate_tile[tile_base + 1];
                            gw2 = gate_tile[tile_base + 2];
                            gw3 = gate_tile[tile_base + 3];
                            gw4 = gate_tile[tile_base + 4];
                            gw5 = gate_tile[tile_base + 5];
                            gw6 = gate_tile[tile_base + 6];
                            gw7 = gate_tile[tile_base + 7];
                            uw0 = up_tile[tile_base + 0];
                            uw1 = up_tile[tile_base + 1];
                            uw2 = up_tile[tile_base + 2];
                            uw3 = up_tile[tile_base + 3];
                            uw4 = up_tile[tile_base + 4];
                            uw5 = up_tile[tile_base + 5];
                            uw6 = up_tile[tile_base + 6];
                            uw7 = up_tile[tile_base + 7];
                        }
                    } else if (consume) {
                        const auto& slot = slots[quad_idx * 4];
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        gw0 = qmap[gb0 >> 4] * ga;
                        gw1 = qmap[gb0 & 0xF] * ga;
                        gw2 = qmap[gb1 >> 4] * ga;
                        gw3 = qmap[gb1 & 0xF] * ga;
                        gw4 = qmap[gb2 >> 4] * ga;
                        gw5 = qmap[gb2 & 0xF] * ga;
                        gw6 = qmap[gb3 >> 4] * ga;
                        gw7 = qmap[gb3 & 0xF] * ga;

                        uw0 = qmap[ub0 >> 4] * ua;
                        uw1 = qmap[ub0 & 0xF] * ua;
                        uw2 = qmap[ub1 >> 4] * ua;
                        uw3 = qmap[ub1 & 0xF] * ua;
                        uw4 = qmap[ub2 >> 4] * ua;
                        uw5 = qmap[ub2 & 0xF] * ua;
                        uw6 = qmap[ub3 >> 4] * ua;
                        uw7 = qmap[ub3 & 0xF] * ua;
                    }
                    if (consume) {
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(x_token2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(x_token3 + byte_offset * 2);

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[0]), gw0, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[0]), gw0, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[1]), gw1, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[1]), gw1, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[2]), gw2, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[2]), gw2, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[3]), gw3, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[3]), gw3, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[4]), gw4, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[4]), gw4, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[5]), gw5, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[5]), gw5, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[6]), gw6, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[6]), gw6, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[7]), gw7, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[7]), gw7, gate_acc3);

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[0]), uw0, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[0]), uw0, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[1]), uw1, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[1]), uw1, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[2]), uw2, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[2]), uw2, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[3]), uw3, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[3]), uw3, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[4]), uw4, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[4]), uw4, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[5]), uw5, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[5]), uw5, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[6]), uw6, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[6]), uw6, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[7]), uw7, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[7]), uw7, up_acc3);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (consume) {
                    for (int byte_offset = sg_lane * 4 + 64 * full_tiles;
                         byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const auto& slot = slots[quad_idx * 4];
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(x_token2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(x_token3 + byte_offset * 2);

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        const float gw0 = qmap[gb0 >> 4] * ga;
                        const float gw1 = qmap[gb0 & 0xF] * ga;
                        const float gw2 = qmap[gb1 >> 4] * ga;
                        const float gw3 = qmap[gb1 & 0xF] * ga;
                        const float gw4 = qmap[gb2 >> 4] * ga;
                        const float gw5 = qmap[gb2 & 0xF] * ga;
                        const float gw6 = qmap[gb3 >> 4] * ga;
                        const float gw7 = qmap[gb3 & 0xF] * ga;

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[0]), gw0, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[0]), gw0, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[1]), gw1, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[1]), gw1, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[2]), gw2, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[2]), gw2, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[3]), gw3, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[3]), gw3, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[4]), gw4, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[4]), gw4, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[5]), gw5, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[5]), gw5, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[6]), gw6, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[6]), gw6, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[7]), gw7, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[7]), gw7, gate_acc3);

                        const float uw0 = qmap[ub0 >> 4] * ua;
                        const float uw1 = qmap[ub0 & 0xF] * ua;
                        const float uw2 = qmap[ub1 >> 4] * ua;
                        const float uw3 = qmap[ub1 & 0xF] * ua;
                        const float uw4 = qmap[ub2 >> 4] * ua;
                        const float uw5 = qmap[ub2 & 0xF] * ua;
                        const float uw6 = qmap[ub3 >> 4] * ua;
                        const float uw7 = qmap[ub3 & 0xF] * ua;

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[0]), uw0, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[0]), uw0, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[1]), uw1, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[1]), uw1, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[2]), uw2, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[2]), uw2, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[3]), uw3, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[3]), uw3, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[4]), uw4, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[4]), uw4, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[5]), uw5, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[5]), uw5, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[6]), uw6, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[6]), uw6, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[7]), uw7, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[7]), uw7, up_acc3);
                    }
                }

                float gate_sum0 = sycl::reduce_over_group(item.get_sub_group(), gate_acc0, sycl::plus<>());
                float up_sum0 = sycl::reduce_over_group(item.get_sub_group(), up_acc0, sycl::plus<>());
                float gate_sum1 = sycl::reduce_over_group(item.get_sub_group(), gate_acc1, sycl::plus<>());
                float up_sum1 = sycl::reduce_over_group(item.get_sub_group(), up_acc1, sycl::plus<>());
                float gate_sum2 = sycl::reduce_over_group(item.get_sub_group(), gate_acc2, sycl::plus<>());
                float up_sum2 = sycl::reduce_over_group(item.get_sub_group(), up_acc2, sycl::plus<>());
                float gate_sum3 = sycl::reduce_over_group(item.get_sub_group(), gate_acc3, sycl::plus<>());
                float up_sum3 = sycl::reduce_over_group(item.get_sub_group(), up_acc3, sycl::plus<>());

                if (sg_lane == 0 && row < intermediate_size && quad_idx < total_quads) {

                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const float silu_gate0 = gate_sum0 / (1.0f + sycl::native::exp(-gate_sum0));
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * intermediate_size + row;
                    h[out_pos0] = bf16(silu_gate0 * up_sum0);

                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const float silu_gate1 = gate_sum1 / (1.0f + sycl::native::exp(-gate_sum1));
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * intermediate_size + row;
                    h[out_pos1] = bf16(silu_gate1 * up_sum1);

                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const float silu_gate2 = gate_sum2 / (1.0f + sycl::native::exp(-gate_sum2));
                    const size_t out_pos2 = (static_cast<size_t>(token2) * top_k + rank2) * intermediate_size + row;
                    h[out_pos2] = bf16(silu_gate2 * up_sum2);

                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];
                    const float silu_gate3 = gate_sum3 / (1.0f + sycl::native::exp(-gate_sum3));
                    const size_t out_pos3 = (static_cast<size_t>(token3) * top_k + rank3) * intermediate_size + row;
                    h[out_pos3] = bf16(silu_gate3 * up_sum3);
                }
            });
    });
}

template <bool UseNested, bool FixedNested256 = false, int ShareQuads = 2>
sycl::event launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_wg = (wg_size / sub_group_size) / static_cast<size_t>(ShareQuads);
    static_assert(ShareQuads == 2 || ShareQuads == 4 || ShareQuads == 8,
                  "ShareQuads must be 2, 4 or 8");
    const int row_blocks = (intermediate_size + static_cast<int>(rows_per_wg) - 1) / static_cast<int>(rows_per_wg);
    const int quad_groups = (total_quads + ShareQuads - 1) / ShareQuads;
    const size_t num_groups = static_cast<size_t>(row_blocks) * static_cast<size_t>(quad_groups);

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const int row_block = static_cast<int>(item.get_group(0)) / quad_groups;
                const int quad_group = static_cast<int>(item.get_group(0)) % quad_groups;
                const int row = row_block * static_cast<int>(rows_per_wg) + sg_id / ShareQuads;
                const int quad_idx = quad_group * ShareQuads + sg_id % ShareQuads;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc0 = 0.0f;
                float up_acc0 = 0.0f;
                float gate_acc1 = 0.0f;
                float up_acc1 = 0.0f;
                float gate_acc2 = 0.0f;
                float up_acc2 = 0.0f;
                float gate_acc3 = 0.0f;
                float up_acc3 = 0.0f;

                if (row < intermediate_size && quad_idx < total_quads) {

                    const auto& slot = slots[quad_idx * 4];
                    const int token0 = token_ids[quad_idx * 4];
                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const bf16* x_token0 = x + static_cast<size_t>(token0) * in_features;
                    const bf16* x_token1 = x + static_cast<size_t>(token1) * in_features;
                    const bf16* x_token2 = x + static_cast<size_t>(token2) * in_features;
                    const bf16* x_token3 = x + static_cast<size_t>(token3) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(x_token2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(x_token3 + byte_offset * 2);

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        const float gw0 = ga;
                        const float gw1 = ga;
                        const float gw2 = ga;
                        const float gw3 = ga;
                        const float gw4 = ga;
                        const float gw5 = ga;
                        const float gw6 = ga;
                        const float gw7 = ga;

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[0]), gw0, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[0]), gw0, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[1]), gw1, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[1]), gw1, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[2]), gw2, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[2]), gw2, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[3]), gw3, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[3]), gw3, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[4]), gw4, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[4]), gw4, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[5]), gw5, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[5]), gw5, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[6]), gw6, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[6]), gw6, gate_acc3);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[7]), gw7, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[7]), gw7, gate_acc3);

                        const float uw0 = ua;
                        const float uw1 = ua;
                        const float uw2 = ua;
                        const float uw3 = ua;
                        const float uw4 = ua;
                        const float uw5 = ua;
                        const float uw6 = ua;
                        const float uw7 = ua;

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[0]), uw0, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[0]), uw0, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[1]), uw1, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[1]), uw1, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[2]), uw2, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[2]), uw2, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[3]), uw3, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[3]), uw3, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[4]), uw4, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[4]), uw4, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[5]), uw5, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[5]), uw5, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[6]), uw6, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[6]), uw6, up_acc3);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[7]), uw7, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[7]), uw7, up_acc3);
                    }
                }

                float gate_sum0 = sycl::reduce_over_group(item.get_sub_group(), gate_acc0, sycl::plus<>());
                float up_sum0 = sycl::reduce_over_group(item.get_sub_group(), up_acc0, sycl::plus<>());
                float gate_sum1 = sycl::reduce_over_group(item.get_sub_group(), gate_acc1, sycl::plus<>());
                float up_sum1 = sycl::reduce_over_group(item.get_sub_group(), up_acc1, sycl::plus<>());
                float gate_sum2 = sycl::reduce_over_group(item.get_sub_group(), gate_acc2, sycl::plus<>());
                float up_sum2 = sycl::reduce_over_group(item.get_sub_group(), up_acc2, sycl::plus<>());
                float gate_sum3 = sycl::reduce_over_group(item.get_sub_group(), gate_acc3, sycl::plus<>());
                float up_sum3 = sycl::reduce_over_group(item.get_sub_group(), up_acc3, sycl::plus<>());

                if (sg_lane == 0 && row < intermediate_size && quad_idx < total_quads) {

                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const float silu_gate0 = gate_sum0 / (1.0f + sycl::native::exp(-gate_sum0));
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * intermediate_size + row;
                    h[out_pos0] = bf16(silu_gate0 * up_sum0);

                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const float silu_gate1 = gate_sum1 / (1.0f + sycl::native::exp(-gate_sum1));
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * intermediate_size + row;
                    h[out_pos1] = bf16(silu_gate1 * up_sum1);

                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const float silu_gate2 = gate_sum2 / (1.0f + sycl::native::exp(-gate_sum2));
                    const size_t out_pos2 = (static_cast<size_t>(token2) * top_k + rank2) * intermediate_size + row;
                    h[out_pos2] = bf16(silu_gate2 * up_sum2);

                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];
                    const float silu_gate3 = gate_sum3 / (1.0f + sycl::native::exp(-gate_sum3));
                    const size_t out_pos3 = (static_cast<size_t>(token3) * top_k + rank3) * intermediate_size + row;
                    h[out_pos3] = bf16(silu_gate3 * up_sum3);
                }
            });
    });
}


template <bool UseNested, bool Compact = false, bool FixedNested256 = false, bool ActSlm = false>
sycl::event launch_moe_expert_major_prefill_down_4token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_quads,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_quads) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        // P9: the quad's four gate/up activation rows (h), staged once per WG.
        sycl::local_accessor<bf16, 1> x_slm(
            sycl::range<1>(ActSlm ? 4 * in_features : 1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const size_t wg_base_row = item.get_group(0) * rows_per_group; // uniform
                const bool wg_active = wg_base_row < total_rows;               // uniform
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                bf16* const x_slm_ptr = x_slm.get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (ActSlm) {
                    // quad_idx is work-group uniform: the dispatcher only enables
                    // this path when out_features % rows_per_group == 0.
                    if (wg_active) {
                        const int quad_idx = static_cast<int>(wg_base_row / out_features);
                        const int base = quad_idx * 4;
                        const bf16* src0 = h + (static_cast<size_t>(token_ids[base + 0]) * top_k +
                                                ranks[base + 0]) * in_features;
                        const bf16* src1 = h + (static_cast<size_t>(token_ids[base + 1]) * top_k +
                                                ranks[base + 1]) * in_features;
                        const bf16* src2 = h + (static_cast<size_t>(token_ids[base + 2]) * top_k +
                                                ranks[base + 2]) * in_features;
                        const bf16* src3 = h + (static_cast<size_t>(token_ids[base + 3]) * top_k +
                                                ranks[base + 3]) * in_features;
                        for (int e = lid * 8; e < in_features;
                             e += static_cast<int>(wg_size) * 8) {
                            *reinterpret_cast<vec8*>(x_slm_ptr + e) =
                                *reinterpret_cast<const vec8*>(src0 + e);
                            *reinterpret_cast<vec8*>(x_slm_ptr + in_features + e) =
                                *reinterpret_cast<const vec8*>(src1 + e);
                            *reinterpret_cast<vec8*>(x_slm_ptr + 2 * in_features + e) =
                                *reinterpret_cast<const vec8*>(src2 + e);
                            *reinterpret_cast<vec8*>(x_slm_ptr + 3 * in_features + e) =
                                *reinterpret_cast<const vec8*>(src3 + e);
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc0 = 0.0f;
                float expert_acc1 = 0.0f;
                float expert_acc2 = 0.0f;
                float expert_acc3 = 0.0f;

                if (global_row < total_rows) {
                    const int quad_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);

                    const auto& slot = slots[quad_idx * 4];
                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];

                    const bf16* in_expert0 = h + (static_cast<size_t>(token0) * top_k + rank0) * in_features;
                    const bf16* in_expert1 = h + (static_cast<size_t>(token1) * top_k + rank1) * in_features;
                    const bf16* in_expert2 = h + (static_cast<size_t>(token2) * top_k + rank2) * in_features;
                    const bf16* in_expert3 = h + (static_cast<size_t>(token3) * top_k + rank3) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                        float da;
                        if (blocksize == 64) {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested, FixedNested256>(
                                slot.down_absmax, slot.down_nested, scale_index);
                        }
                        vec8 in_v0;
                        vec8 in_v1;
                        vec8 in_v2;
                        vec8 in_v3;
                        if constexpr (ActSlm) {
                            const int elem_offset = byte_offset * 2;
                            in_v0 = *reinterpret_cast<const vec8*>(x_slm_ptr + elem_offset);
                            in_v1 = *reinterpret_cast<const vec8*>(x_slm_ptr + in_features + elem_offset);
                            in_v2 = *reinterpret_cast<const vec8*>(x_slm_ptr + 2 * in_features + elem_offset);
                            in_v3 = *reinterpret_cast<const vec8*>(x_slm_ptr + 3 * in_features + elem_offset);
                        } else {
                            in_v0 = *reinterpret_cast<const vec8*>(in_expert0 + byte_offset * 2);
                            in_v1 = *reinterpret_cast<const vec8*>(in_expert1 + byte_offset * 2);
                            in_v2 = *reinterpret_cast<const vec8*>(in_expert2 + byte_offset * 2);
                            in_v3 = *reinterpret_cast<const vec8*>(in_expert3 + byte_offset * 2);
                        }

                        const uint8_t db0 = static_cast<uint8_t>(d4);
                        const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                        const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                        const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                        const float dw0 = qmap[db0 >> 4] * da;
                        const float dw1 = qmap[db0 & 0xF] * da;
                        const float dw2 = qmap[db1 >> 4] * da;
                        const float dw3 = qmap[db1 & 0xF] * da;
                        const float dw4 = qmap[db2 >> 4] * da;
                        const float dw5 = qmap[db2 & 0xF] * da;
                        const float dw6 = qmap[db3 >> 4] * da;
                        const float dw7 = qmap[db3 & 0xF] * da;

                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[0]), dw0, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[0]), dw0, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[0]), dw0, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[0]), dw0, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[1]), dw1, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[1]), dw1, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[1]), dw1, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[1]), dw1, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[2]), dw2, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[2]), dw2, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[2]), dw2, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[2]), dw2, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[3]), dw3, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[3]), dw3, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[3]), dw3, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[3]), dw3, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[4]), dw4, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[4]), dw4, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[4]), dw4, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[4]), dw4, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[5]), dw5, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[5]), dw5, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[5]), dw5, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[5]), dw5, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[6]), dw6, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[6]), dw6, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[6]), dw6, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[6]), dw6, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[7]), dw7, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[7]), dw7, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[7]), dw7, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[7]), dw7, expert_acc3);
                    }
                }

                float expert_sum0 = sycl::reduce_over_group(item.get_sub_group(), expert_acc0, sycl::plus<>());
                float expert_sum1 = sycl::reduce_over_group(item.get_sub_group(), expert_acc1, sycl::plus<>());
                float expert_sum2 = sycl::reduce_over_group(item.get_sub_group(), expert_acc2, sycl::plus<>());
                float expert_sum3 = sycl::reduce_over_group(item.get_sub_group(), expert_acc3, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const int quad_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);

                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const size_t out_pos0 = (Compact ? size_t(quad_idx * 4 + 0) : size_t(token0) * top_k + rank0) * out_features + row;
                    down_acc[out_pos0] = expert_sum0;

                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const size_t out_pos1 = (Compact ? size_t(quad_idx * 4 + 1) : size_t(token1) * top_k + rank1) * out_features + row;
                    down_acc[out_pos1] = expert_sum1;

                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const size_t out_pos2 = (Compact ? size_t(quad_idx * 4 + 2) : size_t(token2) * top_k + rank2) * out_features + row;
                    down_acc[out_pos2] = expert_sum2;

                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];
                    const size_t out_pos3 = (Compact ? size_t(quad_idx * 4 + 3) : size_t(token3) * top_k + rank3) * out_features + row;
                    down_acc[out_pos3] = expert_sum3;
                }
            });
    });
}

// Phase Q4E WG-share (down projection): see the gate/up wgshare kernel above.
template <bool UseNested, bool Compact = false, bool FixedNested256 = false, int ShareQuads = 2>
sycl::event launch_moe_expert_major_prefill_down_4token_wgshare(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_quads,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_wg = (wg_size / sub_group_size) / static_cast<size_t>(ShareQuads);
    static_assert(ShareQuads == 2 || ShareQuads == 4 || ShareQuads == 8,
                  "ShareQuads must be 2, 4 or 8");
    const int row_blocks = (out_features + static_cast<int>(rows_per_wg) - 1) / static_cast<int>(rows_per_wg);
    const int quad_groups = (total_quads + ShareQuads - 1) / ShareQuads;
    const size_t num_groups = static_cast<size_t>(row_blocks) * static_cast<size_t>(quad_groups);

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const int row_block = static_cast<int>(item.get_group(0)) / quad_groups;
                const int quad_group = static_cast<int>(item.get_group(0)) % quad_groups;
                const int row = row_block * static_cast<int>(rows_per_wg) + sg_id / ShareQuads;
                const int quad_idx = quad_group * ShareQuads + sg_id % ShareQuads;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc0 = 0.0f;
                float expert_acc1 = 0.0f;
                float expert_acc2 = 0.0f;
                float expert_acc3 = 0.0f;

                if (row < out_features && quad_idx < total_quads) {

                    const auto& slot = slots[quad_idx * 4];
                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];

                    const bf16* in_expert0 = h + (static_cast<size_t>(token0) * top_k + rank0) * in_features;
                    const bf16* in_expert1 = h + (static_cast<size_t>(token1) * top_k + rank1) * in_features;
                    const bf16* in_expert2 = h + (static_cast<size_t>(token2) * top_k + rank2) * in_features;
                    const bf16* in_expert3 = h + (static_cast<size_t>(token3) * top_k + rank3) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                        float da;
                        if (blocksize == 64) {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested, FixedNested256>(
                                slot.down_absmax, slot.down_nested, scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(in_expert0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(in_expert1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(in_expert2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(in_expert3 + byte_offset * 2);

                        const uint8_t db0 = static_cast<uint8_t>(d4);
                        const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                        const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                        const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                        const float dw0 = qmap[db0 >> 4] * da;
                        const float dw1 = qmap[db0 & 0xF] * da;
                        const float dw2 = qmap[db1 >> 4] * da;
                        const float dw3 = qmap[db1 & 0xF] * da;
                        const float dw4 = qmap[db2 >> 4] * da;
                        const float dw5 = qmap[db2 & 0xF] * da;
                        const float dw6 = qmap[db3 >> 4] * da;
                        const float dw7 = qmap[db3 & 0xF] * da;

                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[0]), dw0, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[0]), dw0, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[0]), dw0, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[0]), dw0, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[1]), dw1, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[1]), dw1, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[1]), dw1, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[1]), dw1, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[2]), dw2, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[2]), dw2, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[2]), dw2, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[2]), dw2, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[3]), dw3, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[3]), dw3, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[3]), dw3, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[3]), dw3, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[4]), dw4, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[4]), dw4, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[4]), dw4, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[4]), dw4, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[5]), dw5, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[5]), dw5, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[5]), dw5, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[5]), dw5, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[6]), dw6, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[6]), dw6, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[6]), dw6, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[6]), dw6, expert_acc3);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[7]), dw7, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[7]), dw7, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[7]), dw7, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[7]), dw7, expert_acc3);
                    }
                }

                float expert_sum0 = sycl::reduce_over_group(item.get_sub_group(), expert_acc0, sycl::plus<>());
                float expert_sum1 = sycl::reduce_over_group(item.get_sub_group(), expert_acc1, sycl::plus<>());
                float expert_sum2 = sycl::reduce_over_group(item.get_sub_group(), expert_acc2, sycl::plus<>());
                float expert_sum3 = sycl::reduce_over_group(item.get_sub_group(), expert_acc3, sycl::plus<>());

                if (sg_lane == 0 && row < out_features && quad_idx < total_quads) {

                    const int token0 = token_ids[quad_idx * 4];
                    const int rank0 = ranks[quad_idx * 4];
                    const size_t out_pos0 = (Compact ? size_t(quad_idx * 4 + 0) : size_t(token0) * top_k + rank0) * out_features + row;
                    down_acc[out_pos0] = expert_sum0;

                    const int token1 = token_ids[quad_idx * 4 + 1];
                    const int rank1 = ranks[quad_idx * 4 + 1];
                    const size_t out_pos1 = (Compact ? size_t(quad_idx * 4 + 1) : size_t(token1) * top_k + rank1) * out_features + row;
                    down_acc[out_pos1] = expert_sum1;

                    const int token2 = token_ids[quad_idx * 4 + 2];
                    const int rank2 = ranks[quad_idx * 4 + 2];
                    const size_t out_pos2 = (Compact ? size_t(quad_idx * 4 + 2) : size_t(token2) * top_k + rank2) * out_features + row;
                    down_acc[out_pos2] = expert_sum2;

                    const int token3 = token_ids[quad_idx * 4 + 3];
                    const int rank3 = ranks[quad_idx * 4 + 3];
                    const size_t out_pos3 = (Compact ? size_t(quad_idx * 4 + 3) : size_t(token3) * top_k + rank3) * out_features + row;
                    down_acc[out_pos3] = expert_sum3;
                }
            });
    });
}

// Phase G4: 8-token weight reuse. Identical arithmetic and reduction order to
// the 4-token triple, but eight token streams share one weight read. Each token
// keeps its own accumulator register, so every per-token FMA chain is unchanged
// (bit-exact against the single-token reference).
template <bool UseNested, bool FixedNested256 = false>
sycl::event launch_moe_expert_major_prefill_gate_up_8token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_eights,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_eights) * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float gate_acc0 = 0.0f;
                float up_acc0 = 0.0f;
                float gate_acc1 = 0.0f;
                float up_acc1 = 0.0f;
                float gate_acc2 = 0.0f;
                float up_acc2 = 0.0f;
                float gate_acc3 = 0.0f;
                float up_acc3 = 0.0f;
                float gate_acc4 = 0.0f;
                float up_acc4 = 0.0f;
                float gate_acc5 = 0.0f;
                float up_acc5 = 0.0f;
                float gate_acc6 = 0.0f;
                float up_acc6 = 0.0f;
                float gate_acc7 = 0.0f;
                float up_acc7 = 0.0f;

                if (global_row < total_rows) {
                    const int oct_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);

                    const auto& slot = slots[oct_idx * 8];
                    const int token0 = token_ids[oct_idx * 8];
                    const int token1 = token_ids[oct_idx * 8 + 1];
                    const int token2 = token_ids[oct_idx * 8 + 2];
                    const int token3 = token_ids[oct_idx * 8 + 3];
                    const int token4 = token_ids[oct_idx * 8 + 4];
                    const int token5 = token_ids[oct_idx * 8 + 5];
                    const int token6 = token_ids[oct_idx * 8 + 6];
                    const int token7 = token_ids[oct_idx * 8 + 7];
                    const bf16* x_token0 = x + static_cast<size_t>(token0) * in_features;
                    const bf16* x_token1 = x + static_cast<size_t>(token1) * in_features;
                    const bf16* x_token2 = x + static_cast<size_t>(token2) * in_features;
                    const bf16* x_token3 = x + static_cast<size_t>(token3) * in_features;
                    const bf16* x_token4 = x + static_cast<size_t>(token4) * in_features;
                    const bf16* x_token5 = x + static_cast<size_t>(token5) * in_features;
                    const bf16* x_token6 = x + static_cast<size_t>(token6) * in_features;
                    const bf16* x_token7 = x + static_cast<size_t>(token7) * in_features;

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base = static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base = static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        const size_t gate_scale_index = gate_block_base + (byte_offset / (blocksize / 2));
                        const size_t up_scale_index = up_block_base + (byte_offset / (blocksize / 2));
                        float ga;
                        float ua;
                        if (blocksize == 64) {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.gate_up_absmax, slot.gate_up_nested, up_scale_index)
                                : 0.0f;
                            const float ga0 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 0);
                            const float ga8 = sycl::group_broadcast(
                                item.get_sub_group(), ga_leader, 8);
                            const float ua0 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 0);
                            const float ua8 = sycl::group_broadcast(
                                item.get_sub_group(), ua_leader, 8);
                            ga = sg_lane < 8 ? ga0 : ga8;
                            ua = sg_lane < 8 ? ua0 : ua8;
                        } else {
                            ga = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested, FixedNested256>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(x_token0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(x_token1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(x_token2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(x_token3 + byte_offset * 2);
                        const vec8 in_v4 = *reinterpret_cast<const vec8*>(x_token4 + byte_offset * 2);
                        const vec8 in_v5 = *reinterpret_cast<const vec8*>(x_token5 + byte_offset * 2);
                        const vec8 in_v6 = *reinterpret_cast<const vec8*>(x_token6 + byte_offset * 2);
                        const vec8 in_v7 = *reinterpret_cast<const vec8*>(x_token7 + byte_offset * 2);

                        const uint8_t gb0 = static_cast<uint8_t>(g4);
                        const uint8_t gb1 = static_cast<uint8_t>(g4 >> 8);
                        const uint8_t gb2 = static_cast<uint8_t>(g4 >> 16);
                        const uint8_t gb3 = static_cast<uint8_t>(g4 >> 24);
                        const uint8_t ub0 = static_cast<uint8_t>(u4);
                        const uint8_t ub1 = static_cast<uint8_t>(u4 >> 8);
                        const uint8_t ub2 = static_cast<uint8_t>(u4 >> 16);
                        const uint8_t ub3 = static_cast<uint8_t>(u4 >> 24);

                        const float gw0 = qmap[gb0 >> 4] * ga;
                        const float gw1 = qmap[gb0 & 0xF] * ga;
                        const float gw2 = qmap[gb1 >> 4] * ga;
                        const float gw3 = qmap[gb1 & 0xF] * ga;
                        const float gw4 = qmap[gb2 >> 4] * ga;
                        const float gw5 = qmap[gb2 & 0xF] * ga;
                        const float gw6 = qmap[gb3 >> 4] * ga;
                        const float gw7 = qmap[gb3 & 0xF] * ga;

                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[0]), gw0, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[0]), gw0, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[0]), gw0, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[0]), gw0, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[0]), gw0, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[0]), gw0, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[0]), gw0, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[0]), gw0, gate_acc7);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[1]), gw1, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[1]), gw1, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[1]), gw1, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[1]), gw1, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[1]), gw1, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[1]), gw1, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[1]), gw1, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[1]), gw1, gate_acc7);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[2]), gw2, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[2]), gw2, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[2]), gw2, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[2]), gw2, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[2]), gw2, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[2]), gw2, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[2]), gw2, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[2]), gw2, gate_acc7);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[3]), gw3, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[3]), gw3, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[3]), gw3, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[3]), gw3, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[3]), gw3, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[3]), gw3, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[3]), gw3, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[3]), gw3, gate_acc7);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[4]), gw4, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[4]), gw4, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[4]), gw4, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[4]), gw4, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[4]), gw4, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[4]), gw4, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[4]), gw4, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[4]), gw4, gate_acc7);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[5]), gw5, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[5]), gw5, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[5]), gw5, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[5]), gw5, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[5]), gw5, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[5]), gw5, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[5]), gw5, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[5]), gw5, gate_acc7);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[6]), gw6, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[6]), gw6, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[6]), gw6, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[6]), gw6, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[6]), gw6, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[6]), gw6, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[6]), gw6, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[6]), gw6, gate_acc7);
                        gate_acc0 = sycl::fma(static_cast<float>(in_v0[7]), gw7, gate_acc0);
                        gate_acc1 = sycl::fma(static_cast<float>(in_v1[7]), gw7, gate_acc1);
                        gate_acc2 = sycl::fma(static_cast<float>(in_v2[7]), gw7, gate_acc2);
                        gate_acc3 = sycl::fma(static_cast<float>(in_v3[7]), gw7, gate_acc3);
                        gate_acc4 = sycl::fma(static_cast<float>(in_v4[7]), gw7, gate_acc4);
                        gate_acc5 = sycl::fma(static_cast<float>(in_v5[7]), gw7, gate_acc5);
                        gate_acc6 = sycl::fma(static_cast<float>(in_v6[7]), gw7, gate_acc6);
                        gate_acc7 = sycl::fma(static_cast<float>(in_v7[7]), gw7, gate_acc7);

                        const float uw0 = qmap[ub0 >> 4] * ua;
                        const float uw1 = qmap[ub0 & 0xF] * ua;
                        const float uw2 = qmap[ub1 >> 4] * ua;
                        const float uw3 = qmap[ub1 & 0xF] * ua;
                        const float uw4 = qmap[ub2 >> 4] * ua;
                        const float uw5 = qmap[ub2 & 0xF] * ua;
                        const float uw6 = qmap[ub3 >> 4] * ua;
                        const float uw7 = qmap[ub3 & 0xF] * ua;

                        up_acc0 = sycl::fma(static_cast<float>(in_v0[0]), uw0, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[0]), uw0, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[0]), uw0, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[0]), uw0, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[0]), uw0, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[0]), uw0, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[0]), uw0, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[0]), uw0, up_acc7);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[1]), uw1, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[1]), uw1, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[1]), uw1, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[1]), uw1, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[1]), uw1, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[1]), uw1, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[1]), uw1, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[1]), uw1, up_acc7);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[2]), uw2, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[2]), uw2, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[2]), uw2, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[2]), uw2, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[2]), uw2, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[2]), uw2, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[2]), uw2, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[2]), uw2, up_acc7);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[3]), uw3, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[3]), uw3, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[3]), uw3, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[3]), uw3, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[3]), uw3, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[3]), uw3, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[3]), uw3, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[3]), uw3, up_acc7);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[4]), uw4, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[4]), uw4, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[4]), uw4, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[4]), uw4, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[4]), uw4, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[4]), uw4, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[4]), uw4, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[4]), uw4, up_acc7);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[5]), uw5, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[5]), uw5, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[5]), uw5, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[5]), uw5, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[5]), uw5, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[5]), uw5, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[5]), uw5, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[5]), uw5, up_acc7);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[6]), uw6, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[6]), uw6, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[6]), uw6, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[6]), uw6, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[6]), uw6, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[6]), uw6, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[6]), uw6, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[6]), uw6, up_acc7);
                        up_acc0 = sycl::fma(static_cast<float>(in_v0[7]), uw7, up_acc0);
                        up_acc1 = sycl::fma(static_cast<float>(in_v1[7]), uw7, up_acc1);
                        up_acc2 = sycl::fma(static_cast<float>(in_v2[7]), uw7, up_acc2);
                        up_acc3 = sycl::fma(static_cast<float>(in_v3[7]), uw7, up_acc3);
                        up_acc4 = sycl::fma(static_cast<float>(in_v4[7]), uw7, up_acc4);
                        up_acc5 = sycl::fma(static_cast<float>(in_v5[7]), uw7, up_acc5);
                        up_acc6 = sycl::fma(static_cast<float>(in_v6[7]), uw7, up_acc6);
                        up_acc7 = sycl::fma(static_cast<float>(in_v7[7]), uw7, up_acc7);
                    }
                }

                float gate_sum0 = sycl::reduce_over_group(item.get_sub_group(), gate_acc0, sycl::plus<>());
                float up_sum0 = sycl::reduce_over_group(item.get_sub_group(), up_acc0, sycl::plus<>());
                float gate_sum1 = sycl::reduce_over_group(item.get_sub_group(), gate_acc1, sycl::plus<>());
                float up_sum1 = sycl::reduce_over_group(item.get_sub_group(), up_acc1, sycl::plus<>());
                float gate_sum2 = sycl::reduce_over_group(item.get_sub_group(), gate_acc2, sycl::plus<>());
                float up_sum2 = sycl::reduce_over_group(item.get_sub_group(), up_acc2, sycl::plus<>());
                float gate_sum3 = sycl::reduce_over_group(item.get_sub_group(), gate_acc3, sycl::plus<>());
                float up_sum3 = sycl::reduce_over_group(item.get_sub_group(), up_acc3, sycl::plus<>());
                float gate_sum4 = sycl::reduce_over_group(item.get_sub_group(), gate_acc4, sycl::plus<>());
                float up_sum4 = sycl::reduce_over_group(item.get_sub_group(), up_acc4, sycl::plus<>());
                float gate_sum5 = sycl::reduce_over_group(item.get_sub_group(), gate_acc5, sycl::plus<>());
                float up_sum5 = sycl::reduce_over_group(item.get_sub_group(), up_acc5, sycl::plus<>());
                float gate_sum6 = sycl::reduce_over_group(item.get_sub_group(), gate_acc6, sycl::plus<>());
                float up_sum6 = sycl::reduce_over_group(item.get_sub_group(), up_acc6, sycl::plus<>());
                float gate_sum7 = sycl::reduce_over_group(item.get_sub_group(), gate_acc7, sycl::plus<>());
                float up_sum7 = sycl::reduce_over_group(item.get_sub_group(), up_acc7, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const int oct_idx = static_cast<int>(global_row / intermediate_size);
                    const int row = static_cast<int>(global_row % intermediate_size);

                    const int token0 = token_ids[oct_idx * 8];
                    const int rank0 = ranks[oct_idx * 8];
                    const float silu_gate0 = gate_sum0 / (1.0f + sycl::native::exp(-gate_sum0));
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * intermediate_size + row;
                    h[out_pos0] = bf16(silu_gate0 * up_sum0);

                    const int token1 = token_ids[oct_idx * 8 + 1];
                    const int rank1 = ranks[oct_idx * 8 + 1];
                    const float silu_gate1 = gate_sum1 / (1.0f + sycl::native::exp(-gate_sum1));
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * intermediate_size + row;
                    h[out_pos1] = bf16(silu_gate1 * up_sum1);

                    const int token2 = token_ids[oct_idx * 8 + 2];
                    const int rank2 = ranks[oct_idx * 8 + 2];
                    const float silu_gate2 = gate_sum2 / (1.0f + sycl::native::exp(-gate_sum2));
                    const size_t out_pos2 = (static_cast<size_t>(token2) * top_k + rank2) * intermediate_size + row;
                    h[out_pos2] = bf16(silu_gate2 * up_sum2);

                    const int token3 = token_ids[oct_idx * 8 + 3];
                    const int rank3 = ranks[oct_idx * 8 + 3];
                    const float silu_gate3 = gate_sum3 / (1.0f + sycl::native::exp(-gate_sum3));
                    const size_t out_pos3 = (static_cast<size_t>(token3) * top_k + rank3) * intermediate_size + row;
                    h[out_pos3] = bf16(silu_gate3 * up_sum3);

                    const int token4 = token_ids[oct_idx * 8 + 4];
                    const int rank4 = ranks[oct_idx * 8 + 4];
                    const float silu_gate4 = gate_sum4 / (1.0f + sycl::native::exp(-gate_sum4));
                    const size_t out_pos4 = (static_cast<size_t>(token4) * top_k + rank4) * intermediate_size + row;
                    h[out_pos4] = bf16(silu_gate4 * up_sum4);

                    const int token5 = token_ids[oct_idx * 8 + 5];
                    const int rank5 = ranks[oct_idx * 8 + 5];
                    const float silu_gate5 = gate_sum5 / (1.0f + sycl::native::exp(-gate_sum5));
                    const size_t out_pos5 = (static_cast<size_t>(token5) * top_k + rank5) * intermediate_size + row;
                    h[out_pos5] = bf16(silu_gate5 * up_sum5);

                    const int token6 = token_ids[oct_idx * 8 + 6];
                    const int rank6 = ranks[oct_idx * 8 + 6];
                    const float silu_gate6 = gate_sum6 / (1.0f + sycl::native::exp(-gate_sum6));
                    const size_t out_pos6 = (static_cast<size_t>(token6) * top_k + rank6) * intermediate_size + row;
                    h[out_pos6] = bf16(silu_gate6 * up_sum6);

                    const int token7 = token_ids[oct_idx * 8 + 7];
                    const int rank7 = ranks[oct_idx * 8 + 7];
                    const float silu_gate7 = gate_sum7 / (1.0f + sycl::native::exp(-gate_sum7));
                    const size_t out_pos7 = (static_cast<size_t>(token7) * top_k + rank7) * intermediate_size + row;
                    h[out_pos7] = bf16(silu_gate7 * up_sum7);
                }
            });
    });
}

template <bool UseNested, bool Compact = false, bool FixedNested256 = false>
sycl::event launch_moe_expert_major_prefill_down_8token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_eights,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event = nullptr) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / blocksize;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(total_eights) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / sub_group_size;
                const int sg_lane = lid % sub_group_size;
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    qmap[lid] = quant_map ? quant_map[lid] : kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc0 = 0.0f;
                float expert_acc1 = 0.0f;
                float expert_acc2 = 0.0f;
                float expert_acc3 = 0.0f;
                float expert_acc4 = 0.0f;
                float expert_acc5 = 0.0f;
                float expert_acc6 = 0.0f;
                float expert_acc7 = 0.0f;

                if (global_row < total_rows) {
                    const int oct_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);

                    const auto& slot = slots[oct_idx * 8];
                    const int token0 = token_ids[oct_idx * 8];
                    const int rank0 = ranks[oct_idx * 8];
                    const int token1 = token_ids[oct_idx * 8 + 1];
                    const int rank1 = ranks[oct_idx * 8 + 1];
                    const int token2 = token_ids[oct_idx * 8 + 2];
                    const int rank2 = ranks[oct_idx * 8 + 2];
                    const int token3 = token_ids[oct_idx * 8 + 3];
                    const int rank3 = ranks[oct_idx * 8 + 3];
                    const int token4 = token_ids[oct_idx * 8 + 4];
                    const int rank4 = ranks[oct_idx * 8 + 4];
                    const int token5 = token_ids[oct_idx * 8 + 5];
                    const int rank5 = ranks[oct_idx * 8 + 5];
                    const int token6 = token_ids[oct_idx * 8 + 6];
                    const int rank6 = ranks[oct_idx * 8 + 6];
                    const int token7 = token_ids[oct_idx * 8 + 7];
                    const int rank7 = ranks[oct_idx * 8 + 7];

                    const bf16* in_expert0 = h + (static_cast<size_t>(token0) * top_k + rank0) * in_features;
                    const bf16* in_expert1 = h + (static_cast<size_t>(token1) * top_k + rank1) * in_features;
                    const bf16* in_expert2 = h + (static_cast<size_t>(token2) * top_k + rank2) * in_features;
                    const bf16* in_expert3 = h + (static_cast<size_t>(token3) * top_k + rank3) * in_features;
                    const bf16* in_expert4 = h + (static_cast<size_t>(token4) * top_k + rank4) * in_features;
                    const bf16* in_expert5 = h + (static_cast<size_t>(token5) * top_k + rank5) * in_features;
                    const bf16* in_expert6 = h + (static_cast<size_t>(token6) * top_k + rank6) * in_features;
                    const bf16* in_expert7 = h + (static_cast<size_t>(token7) * top_k + rank7) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_index = block_base + (byte_offset / (blocksize / 2));
                        float da;
                        if (blocksize == 64) {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested, FixedNested256>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested, FixedNested256>(
                                slot.down_absmax, slot.down_nested, scale_index);
                        }
                        const vec8 in_v0 = *reinterpret_cast<const vec8*>(in_expert0 + byte_offset * 2);
                        const vec8 in_v1 = *reinterpret_cast<const vec8*>(in_expert1 + byte_offset * 2);
                        const vec8 in_v2 = *reinterpret_cast<const vec8*>(in_expert2 + byte_offset * 2);
                        const vec8 in_v3 = *reinterpret_cast<const vec8*>(in_expert3 + byte_offset * 2);
                        const vec8 in_v4 = *reinterpret_cast<const vec8*>(in_expert4 + byte_offset * 2);
                        const vec8 in_v5 = *reinterpret_cast<const vec8*>(in_expert5 + byte_offset * 2);
                        const vec8 in_v6 = *reinterpret_cast<const vec8*>(in_expert6 + byte_offset * 2);
                        const vec8 in_v7 = *reinterpret_cast<const vec8*>(in_expert7 + byte_offset * 2);

                        const uint8_t db0 = static_cast<uint8_t>(d4);
                        const uint8_t db1 = static_cast<uint8_t>(d4 >> 8);
                        const uint8_t db2 = static_cast<uint8_t>(d4 >> 16);
                        const uint8_t db3 = static_cast<uint8_t>(d4 >> 24);

                        const float dw0 = qmap[db0 >> 4] * da;
                        const float dw1 = qmap[db0 & 0xF] * da;
                        const float dw2 = qmap[db1 >> 4] * da;
                        const float dw3 = qmap[db1 & 0xF] * da;
                        const float dw4 = qmap[db2 >> 4] * da;
                        const float dw5 = qmap[db2 & 0xF] * da;
                        const float dw6 = qmap[db3 >> 4] * da;
                        const float dw7 = qmap[db3 & 0xF] * da;

                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[0]), dw0, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[0]), dw0, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[0]), dw0, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[0]), dw0, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[0]), dw0, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[0]), dw0, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[0]), dw0, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[0]), dw0, expert_acc7);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[1]), dw1, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[1]), dw1, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[1]), dw1, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[1]), dw1, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[1]), dw1, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[1]), dw1, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[1]), dw1, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[1]), dw1, expert_acc7);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[2]), dw2, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[2]), dw2, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[2]), dw2, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[2]), dw2, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[2]), dw2, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[2]), dw2, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[2]), dw2, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[2]), dw2, expert_acc7);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[3]), dw3, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[3]), dw3, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[3]), dw3, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[3]), dw3, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[3]), dw3, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[3]), dw3, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[3]), dw3, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[3]), dw3, expert_acc7);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[4]), dw4, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[4]), dw4, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[4]), dw4, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[4]), dw4, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[4]), dw4, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[4]), dw4, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[4]), dw4, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[4]), dw4, expert_acc7);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[5]), dw5, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[5]), dw5, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[5]), dw5, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[5]), dw5, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[5]), dw5, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[5]), dw5, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[5]), dw5, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[5]), dw5, expert_acc7);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[6]), dw6, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[6]), dw6, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[6]), dw6, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[6]), dw6, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[6]), dw6, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[6]), dw6, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[6]), dw6, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[6]), dw6, expert_acc7);
                        expert_acc0 = sycl::fma(static_cast<float>(in_v0[7]), dw7, expert_acc0);
                        expert_acc1 = sycl::fma(static_cast<float>(in_v1[7]), dw7, expert_acc1);
                        expert_acc2 = sycl::fma(static_cast<float>(in_v2[7]), dw7, expert_acc2);
                        expert_acc3 = sycl::fma(static_cast<float>(in_v3[7]), dw7, expert_acc3);
                        expert_acc4 = sycl::fma(static_cast<float>(in_v4[7]), dw7, expert_acc4);
                        expert_acc5 = sycl::fma(static_cast<float>(in_v5[7]), dw7, expert_acc5);
                        expert_acc6 = sycl::fma(static_cast<float>(in_v6[7]), dw7, expert_acc6);
                        expert_acc7 = sycl::fma(static_cast<float>(in_v7[7]), dw7, expert_acc7);
                    }
                }

                float expert_sum0 = sycl::reduce_over_group(item.get_sub_group(), expert_acc0, sycl::plus<>());
                float expert_sum1 = sycl::reduce_over_group(item.get_sub_group(), expert_acc1, sycl::plus<>());
                float expert_sum2 = sycl::reduce_over_group(item.get_sub_group(), expert_acc2, sycl::plus<>());
                float expert_sum3 = sycl::reduce_over_group(item.get_sub_group(), expert_acc3, sycl::plus<>());
                float expert_sum4 = sycl::reduce_over_group(item.get_sub_group(), expert_acc4, sycl::plus<>());
                float expert_sum5 = sycl::reduce_over_group(item.get_sub_group(), expert_acc5, sycl::plus<>());
                float expert_sum6 = sycl::reduce_over_group(item.get_sub_group(), expert_acc6, sycl::plus<>());
                float expert_sum7 = sycl::reduce_over_group(item.get_sub_group(), expert_acc7, sycl::plus<>());

                if (sg_lane == 0 && global_row < total_rows) {
                    const int oct_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);

                    const int token0 = token_ids[oct_idx * 8];
                    const int rank0 = ranks[oct_idx * 8];
                    const size_t out_pos0 = (Compact ? size_t(oct_idx * 8 + 0) : size_t(token0) * top_k + rank0) * out_features + row;
                    down_acc[out_pos0] = expert_sum0;

                    const int token1 = token_ids[oct_idx * 8 + 1];
                    const int rank1 = ranks[oct_idx * 8 + 1];
                    const size_t out_pos1 = (Compact ? size_t(oct_idx * 8 + 1) : size_t(token1) * top_k + rank1) * out_features + row;
                    down_acc[out_pos1] = expert_sum1;

                    const int token2 = token_ids[oct_idx * 8 + 2];
                    const int rank2 = ranks[oct_idx * 8 + 2];
                    const size_t out_pos2 = (Compact ? size_t(oct_idx * 8 + 2) : size_t(token2) * top_k + rank2) * out_features + row;
                    down_acc[out_pos2] = expert_sum2;

                    const int token3 = token_ids[oct_idx * 8 + 3];
                    const int rank3 = ranks[oct_idx * 8 + 3];
                    const size_t out_pos3 = (Compact ? size_t(oct_idx * 8 + 3) : size_t(token3) * top_k + rank3) * out_features + row;
                    down_acc[out_pos3] = expert_sum3;

                    const int token4 = token_ids[oct_idx * 8 + 4];
                    const int rank4 = ranks[oct_idx * 8 + 4];
                    const size_t out_pos4 = (Compact ? size_t(oct_idx * 8 + 4) : size_t(token4) * top_k + rank4) * out_features + row;
                    down_acc[out_pos4] = expert_sum4;

                    const int token5 = token_ids[oct_idx * 8 + 5];
                    const int rank5 = ranks[oct_idx * 8 + 5];
                    const size_t out_pos5 = (Compact ? size_t(oct_idx * 8 + 5) : size_t(token5) * top_k + rank5) * out_features + row;
                    down_acc[out_pos5] = expert_sum5;

                    const int token6 = token_ids[oct_idx * 8 + 6];
                    const int rank6 = ranks[oct_idx * 8 + 6];
                    const size_t out_pos6 = (Compact ? size_t(oct_idx * 8 + 6) : size_t(token6) * top_k + rank6) * out_features + row;
                    down_acc[out_pos6] = expert_sum6;

                    const int token7 = token_ids[oct_idx * 8 + 7];
                    const int rank7 = ranks[oct_idx * 8 + 7];
                    const size_t out_pos7 = (Compact ? size_t(oct_idx * 8 + 7) : size_t(token7) * top_k + rank7) * out_features + row;
                    down_acc[out_pos7] = expert_sum7;
                }
            });
    });
}

} // namespace

sycl::event moe_resident_classify(Context& ctx, const int* ids, const float* weights,
    const int* table, int* selected, int* host_ids, float* host_weights, int* host_selected,
    int top_k, int experts) {
    if (top_k < 1 || top_k > 16) throw std::invalid_argument("Resident top-k bound");
    return ctx.queue().parallel_for(sycl::range<1>(16), [=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]] {
        const int rank = index[0];
        if (rank < top_k) {
            const int expert = ids[rank];
            const int slot = expert >= 0 && expert < experts ? table[expert] : -1;
            selected[rank] = slot; host_selected[rank] = slot;
            host_ids[rank] = expert; host_weights[rank] = weights[rank];
        }
    });
}

sycl::event moe_resident_classify_device(Context& ctx, const int* ids,
    const int* table, int* selected, int top_k, int experts) {
    if (top_k < 1 || top_k > 16)
        throw std::invalid_argument("Resident top-k bound");
    return ctx.queue().parallel_for(
        sycl::range<1>(16),
        [=](sycl::id<1> index) [[sycl::reqd_sub_group_size(16)]] {
            const int rank = static_cast<int>(index[0]);
            if (rank < top_k) {
                const int expert = ids[rank];
                selected[rank] =
                    expert >= 0 && expert < experts
                        ? table[expert]
                        : -1;
            }
        });
}

sycl::event moe_resident_gate_up(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const bf16* x, bf16* h, int top_k, int hidden, int intermediate, bool nested,
    bool nested_fixed256) {
    if (top_k < 1 || top_k > 16 || hidden % 64 || intermediate % 16)
        throw std::invalid_argument("Resident gate/up shape");
    if (nested) return nested_fixed256 ? launch_moe_decode_gate_up_swiglu_grouped_dev<true,true,true>(ctx,descriptors,nullptr,nullptr,x,h,top_k,hidden,intermediate,64,nullptr,selected)
        : launch_moe_decode_gate_up_swiglu_grouped_dev<true,true,false>(ctx,descriptors,nullptr,nullptr,x,h,top_k,hidden,intermediate,64,nullptr,selected);
    return launch_moe_decode_gate_up_swiglu_grouped_dev<false,true>(ctx,descriptors,nullptr,nullptr,x,h,top_k,hidden,intermediate,64,nullptr,selected);
}

sycl::event moe_resident_down(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const bf16* h, float* down, int top_k, int hidden, int intermediate, bool nested,
    bool nested_fixed256) {
    if (top_k < 1 || top_k > 16 || hidden % 16 || intermediate % 64)
        throw std::invalid_argument("Resident down shape");
    if (nested) return nested_fixed256 ? launch_moe_decode_down_to_rank_f32_dev<true,true,true>(ctx,descriptors,nullptr,nullptr,h,down,top_k,top_k,intermediate,hidden,64,nullptr,nullptr,selected)
        : launch_moe_decode_down_to_rank_f32_dev<true,true,false>(ctx,descriptors,nullptr,nullptr,h,down,top_k,top_k,intermediate,hidden,64,nullptr,nullptr,selected);
    return launch_moe_decode_down_to_rank_f32_dev<false,true>(ctx,descriptors,nullptr,nullptr,h,down,top_k,top_k,intermediate,hidden,64,nullptr,nullptr,selected);
}

sycl::event quantize_moe_input_q8_64(Context& ctx, const bf16* x,
    int8_t* xq, float* xs, int hidden) {
    if (hidden <= 0 || hidden % 64 || !x || !xq || !xs)
        throw std::invalid_argument("Resident Q8 input shape or arena");
    return launch_moe_quantize_q8_64(ctx, x, xq, xs, 1, hidden, nullptr);
}

sycl::event moe_resident_gate_up_q8_64(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const int8_t* xq, const float* xs, bf16* h,
    int8_t* hq, float* hs, int top_k, int hidden, int intermediate, bool nested,
    bool nested_fixed256) {
    if (top_k < 1 || top_k > 16 || hidden <= 0 || hidden % 64 ||
        intermediate <= 0 || intermediate % 64 || !descriptors || !selected ||
        !xq || !xs || !h || !hq || !hs)
        throw std::invalid_argument("Resident Q8 gate/up shape or arena");
    // A770 provides WG1024. Keep a portable fallback for devices whose
    // physical work-group limit is smaller; it quantizes rounded h separately.
    const size_t max_group = moe_q8_max_work_group_size(ctx);
    if (max_group >= 1024) {
        if (nested) return nested_fixed256 ? launch_moe_resident_gate_up_q8_64<true,1024,true>(ctx, descriptors, selected, xq, xs, h, hq, hs, top_k, hidden, intermediate)
            : launch_moe_resident_gate_up_q8_64<true,1024,false>(ctx, descriptors, selected, xq, xs, h, hq, hs, top_k, hidden, intermediate);
        return launch_moe_resident_gate_up_q8_64<false,1024>(ctx, descriptors, selected, xq, xs, h, hq, hs, top_k, hidden, intermediate);
    }
    if (max_group < 256) throw std::invalid_argument("Resident Q8 gate/up needs work-group size >=256");
    if (nested) { if (nested_fixed256) launch_moe_resident_gate_up_q8_64<true,256,true>(ctx, descriptors, selected, xq, xs, h, hq, hs, top_k, hidden, intermediate);
        else launch_moe_resident_gate_up_q8_64<true,256,false>(ctx, descriptors, selected, xq, xs, h, hq, hs, top_k, hidden, intermediate); }
    else launch_moe_resident_gate_up_q8_64<false,256>(ctx, descriptors, selected, xq, xs, h, hq, hs, top_k, hidden, intermediate);
    return launch_moe_quantize_q8_64(ctx, h, hq, hs, top_k, intermediate, selected);
}

sycl::event moe_resident_down_q8_64(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const int8_t* hq, const float* hs, float* down,
    int top_k, int hidden, int intermediate, bool nested, bool nested_fixed256) {
    if (top_k < 1 || top_k > 16 || hidden <= 0 || hidden % 64 ||
        intermediate <= 0 || intermediate % 64 || !descriptors || !selected ||
        !hq || !hs || !down)
        throw std::invalid_argument("Resident Q8 down shape or arena");
    if (nested) return nested_fixed256 ? launch_moe_resident_down_q8_64<true,true>(ctx, descriptors, selected, hq, hs, down, top_k, hidden, intermediate)
        : launch_moe_resident_down_q8_64<true,false>(ctx, descriptors, selected, hq, hs, down, top_k, hidden, intermediate);
    return launch_moe_resident_down_q8_64<false>(ctx, descriptors, selected, hq, hs, down, top_k, hidden, intermediate);
}

sycl::event quantize_moe_input_q8_64_rows(Context& ctx, const bf16* x,
    int8_t* xq, float* xs, int rows, int hidden) {
    if (rows <= 0 || hidden <= 0 || hidden % 64 || !x || !xq || !xs)
        throw std::invalid_argument("Adaptive Q8 input shape or arena");
    return launch_moe_quantize_q8_64(ctx, x, xq, xs, rows, hidden, nullptr);
}

// Adaptive decode Q8 gate/up. Reuses the resident signed-dot kernel with the
// per-rank token offset; descriptor arrays that live in host USM are mirrored
// into kernel by-value arrays exactly like the BF16 grouped path.
sycl::event moe_decode_gate_up_swiglu_grouped_q8(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int8_t* xq,
    const float* xs,
    bf16* h,
    int8_t* hq,
    float* hs,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    const int* token_ids,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (num_active_experts <= 0) return {};
    if (in_features <= 0 || in_features % 64 || intermediate_size <= 0 ||
        intermediate_size % 64 || !slots || !xq || !xs || !h || !hq || !hs)
        throw std::invalid_argument("Adaptive Q8 gate/up shape or arena");

    bool is_nested = use_nested;
    const auto alloc_type = sycl::get_pointer_type(slots, ctx.queue().get_context());
    const bool is_device = (alloc_type == sycl::usm::alloc::device);
    if (!is_nested && !is_device) {
        is_nested = (slots[0].gate_up_nested.codes != nullptr);
    }

    const size_t max_group = moe_q8_max_work_group_size(ctx);
    if (max_group < 256)
        throw std::invalid_argument("Adaptive Q8 gate/up needs work-group size >=256");
    // WG1024 fuses SwiGLU quantization into the projection; smaller devices
    // write the BF16 boundary first and quantize it in a second pass.
    // AILA_Q4E_GPU_Q8_ADAPTIVE_WG is an opt-in experiment knob: 0 keeps the
    // automatic choice, 256 forces the two-pass WG256 grid.
    static const int wg_override = aila::env::read_int("AILA_Q4E_GPU_Q8_ADAPTIVE_WG", 0);
    const bool fused = wg_override == 0 ? (max_group >= 1024)
                                        : (wg_override >= 1024 && max_group >= 1024);

    auto launch = [&](const auto& descriptor_source, const auto& tokens) -> sycl::event {
        if (fused) {
            if (is_nested) {
                if (nested_fixed256) return launch_moe_resident_gate_up_q8_64<true, 1024, true>(
                    ctx, descriptor_source, nullptr, xq, xs, h, hq, hs, num_active_experts, in_features,
                    intermediate_size, tokens, dep_event);
                return launch_moe_resident_gate_up_q8_64<true, 1024, false>(
                    ctx, descriptor_source, nullptr, xq, xs, h, hq, hs, num_active_experts, in_features,
                    intermediate_size, tokens, dep_event);
            }
            return launch_moe_resident_gate_up_q8_64<false, 1024>(
                ctx, descriptor_source, nullptr, xq, xs, h, hq, hs, num_active_experts, in_features,
                intermediate_size, tokens, dep_event);
        }
        if (is_nested) {
            if (nested_fixed256) launch_moe_resident_gate_up_q8_64<true, 256, true>(
                ctx, descriptor_source, nullptr, xq, xs, h, hq, hs, num_active_experts, in_features,
                intermediate_size, tokens, dep_event);
            else launch_moe_resident_gate_up_q8_64<true, 256, false>(
                ctx, descriptor_source, nullptr, xq, xs, h, hq, hs, num_active_experts, in_features,
                intermediate_size, tokens, dep_event);
        } else {
            launch_moe_resident_gate_up_q8_64<false, 256>(
                ctx, descriptor_source, nullptr, xq, xs, h, hq, hs, num_active_experts, in_features,
                intermediate_size, tokens, dep_event);
        }
        return launch_moe_quantize_q8_64(ctx, h, hq, hs, num_active_experts, intermediate_size, nullptr);
    };

    if (!is_device) {
        if (num_active_experts > 16)
            throw std::invalid_argument("Adaptive Q8 gate/up host descriptor array exceeds 16 experts");
        std::array<MoeExpertSlotDev, 16> slot_ptrs{};
        std::array<int, 16> token_ptrs{};
        for (int i = 0; i < num_active_experts; ++i) {
            slot_ptrs[static_cast<size_t>(i)] = slots[i];
            token_ptrs[static_cast<size_t>(i)] = token_ids ? token_ids[i] : 0;
        }
        return launch(slot_ptrs, token_ptrs);
    }
    return launch(slots, token_ids);
}

sycl::event moe_decode_down_to_rank_f32_q8(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* ranks,
    const int8_t* hq,
    const float* hs,
    float* down_acc,
    int num_active_experts,
    int top_k,
    int in_features,
    int out_features,
    const int* token_ids,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (num_active_experts <= 0 || out_features <= 0 || in_features <= 0) return {};
    if (in_features % 64 || out_features % 64 || top_k < 1 || top_k > 16 ||
        !slots || !hq || !hs || !down_acc)
        throw std::invalid_argument("Adaptive Q8 down shape or arena");

    bool is_nested = use_nested;
    const auto alloc_type = sycl::get_pointer_type(slots, ctx.queue().get_context());
    const bool is_device = (alloc_type == sycl::usm::alloc::device);
    if (!is_nested && !is_device) {
        is_nested = (slots[0].down_nested.codes != nullptr);
    }

    auto launch = [&](const auto& descriptor_source, const auto& tokens,
                      const auto& route_ranks) -> sycl::event {
        if (is_nested) {
            if (nested_fixed256) return launch_moe_resident_down_q8_64<true, true>(
                ctx, descriptor_source, nullptr, hq, hs, down_acc, num_active_experts, out_features,
                in_features, tokens, route_ranks, top_k, dep_event);
            return launch_moe_resident_down_q8_64<true, false>(
                ctx, descriptor_source, nullptr, hq, hs, down_acc, num_active_experts, out_features,
                in_features, tokens, route_ranks, top_k, dep_event);
        }
        return launch_moe_resident_down_q8_64<false>(
            ctx, descriptor_source, nullptr, hq, hs, down_acc, num_active_experts, out_features,
            in_features, tokens, route_ranks, top_k, dep_event);
    };

    if (!ranks)
        throw std::invalid_argument("Adaptive Q8 down needs the route rank map");
    if (!is_device) {
        if (num_active_experts > 16)
            throw std::invalid_argument("Adaptive Q8 down host descriptor array exceeds 16 experts");
        std::array<MoeExpertSlotDev, 16> slot_ptrs{};
        std::array<int, 16> token_ptrs{};
        std::array<int, 16> rank_ptrs{};
        for (int i = 0; i < num_active_experts; ++i) {
            slot_ptrs[static_cast<size_t>(i)] = slots[i];
            token_ptrs[static_cast<size_t>(i)] = token_ids ? token_ids[i] : 0;
            rank_ptrs[static_cast<size_t>(i)] = ranks[i];
        }
        return launch(slot_ptrs, token_ptrs, rank_ptrs);
    }
    return launch(slots, token_ids, ranks);
}

sycl::event moe_decode_gate_up_swiglu_grouped(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    const int* token_ids,
    bool use_nested,
    bool nested_fixed256) {
    if (num_active_experts <= 0) return {};

    bool is_nested = use_nested;
    const auto alloc_type = sycl::get_pointer_type(slots, ctx.queue().get_context());
    const bool is_device = (alloc_type == sycl::usm::alloc::device);
    if (!is_nested && slots != nullptr && !is_device) {
        is_nested = (slots[0].gate_up_nested.codes != nullptr);
    }

    if (!is_device && num_active_experts <= 16) {
        std::array<MoeExpertSlotDev, 16> slot_ptrs{};
        std::array<int, 16> token_ptrs{};
        for (int i = 0; i < num_active_experts; ++i) {
            slot_ptrs[static_cast<size_t>(i)] = slots[i];
            token_ptrs[static_cast<size_t>(i)] = token_ids ? token_ids[i] : 0;
        }

        if (is_nested) {
            if (nested_fixed256) return launch_moe_decode_gate_up_swiglu_grouped<true, true>(
                ctx, slot_ptrs, token_ptrs, quant_map, x, h, num_active_experts,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_decode_gate_up_swiglu_grouped<true, false>(
                ctx, slot_ptrs, token_ptrs, quant_map, x, h, num_active_experts,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_decode_gate_up_swiglu_grouped<false>(
            ctx, slot_ptrs, token_ptrs, quant_map, x, h, num_active_experts,
            in_features, intermediate_size, blocksize, dep_event);
    }

    if (is_nested) {
        if (nested_fixed256) return launch_moe_decode_gate_up_swiglu_grouped_dev<true, false, true>(
            ctx, slots, token_ids, quant_map, x, h, num_active_experts,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_decode_gate_up_swiglu_grouped_dev<true, false, false>(
            ctx, slots, token_ids, quant_map, x, h, num_active_experts,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_decode_gate_up_swiglu_grouped_dev<false>(
        ctx, slots, token_ids, quant_map, x, h, num_active_experts,
        in_features, intermediate_size, blocksize, dep_event);
}

sycl::event moe_decode_down_weighted_accumulate(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const float* weights,
    const bf16* h,
    float* acc_f32,
    bf16* out_bf16,
    int num_active_experts,
    int in_features,
    int out_features,
    int blocksize,
    const bf16* shared_out,
    const bf16* shared_gate,
    bool nested_fixed256) {
    if (num_active_experts <= 0) return {};

    std::array<MoeExpertSlotDev, 16> slot_ptrs{};
    std::array<float, 16> weight_ptrs{};
    const int k_capped = std::min(num_active_experts, 16);
    for (int i = 0; i < k_capped; ++i) {
        slot_ptrs[static_cast<size_t>(i)] = slots[i];
        weight_ptrs[static_cast<size_t>(i)] = weights[i];
    }

    if (slots[0].down_nested.codes != nullptr) {
        if (nested_fixed256) return launch_moe_decode_down_weighted_accumulate<true, true>(
            ctx, slot_ptrs, weight_ptrs, quant_map, h, acc_f32, out_bf16,
            num_active_experts, in_features, out_features, blocksize,
            shared_out, shared_gate);
        return launch_moe_decode_down_weighted_accumulate<true, false>(
            ctx, slot_ptrs, weight_ptrs, quant_map, h, acc_f32, out_bf16,
            num_active_experts, in_features, out_features, blocksize,
            shared_out, shared_gate);
    }
    return launch_moe_decode_down_weighted_accumulate<false>(
        ctx, slot_ptrs, weight_ptrs, quant_map, h, acc_f32, out_bf16,
        num_active_experts, in_features, out_features, blocksize,
        shared_out, shared_gate);
}

sycl::event moe_decode_down_to_rank_f32(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* ranks,
    const bf16* h,
    float* down_acc,
    int num_active_experts,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const int* token_ids,
    const float* quant_map,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (num_active_experts <= 0 || out_features <= 0 || in_features <= 0) return {};

    bool is_nested = use_nested;
    const auto alloc_type = sycl::get_pointer_type(slots, ctx.queue().get_context());
    const bool is_device = (alloc_type == sycl::usm::alloc::device);
    if (!is_nested && slots != nullptr && !is_device) {
        is_nested = (slots[0].down_nested.codes != nullptr);
    }

    if (!is_device && num_active_experts <= 16) {
        std::array<MoeExpertSlotDev, 16> slot_ptrs{};
        std::array<int, 16> rank_ptrs{};
        std::array<int, 16> token_ptrs{};
        const int k_capped = std::min(num_active_experts, 16);
        for (int i = 0; i < k_capped; ++i) {
            slot_ptrs[static_cast<size_t>(i)] = slots[i];
            rank_ptrs[static_cast<size_t>(i)] = ranks[i];
            token_ptrs[static_cast<size_t>(i)] = token_ids ? token_ids[i] : 0;
        }

        if (is_nested) {
            if (nested_fixed256) return launch_moe_decode_down_to_rank_f32<true, true>(
                ctx, slot_ptrs, rank_ptrs, token_ptrs, h, down_acc,
                num_active_experts, top_k, in_features, out_features, blocksize,
                quant_map, dep_event);
            return launch_moe_decode_down_to_rank_f32<true, false>(
                ctx, slot_ptrs, rank_ptrs, token_ptrs, h, down_acc,
                num_active_experts, top_k, in_features, out_features, blocksize,
                quant_map, dep_event);
        }
        return launch_moe_decode_down_to_rank_f32<false>(
            ctx, slot_ptrs, rank_ptrs, token_ptrs, h, down_acc,
            num_active_experts, top_k, in_features, out_features, blocksize,
            quant_map, dep_event);
    }

    if (is_nested) {
        if (nested_fixed256) return launch_moe_decode_down_to_rank_f32_dev<true, false, true>(
            ctx, slots, ranks, token_ids, h, down_acc,
            num_active_experts, top_k, in_features, out_features, blocksize,
            quant_map, dep_event);
        return launch_moe_decode_down_to_rank_f32_dev<true, false, false>(
            ctx, slots, ranks, token_ids, h, down_acc,
            num_active_experts, top_k, in_features, out_features, blocksize,
            quant_map, dep_event);
    }
    return launch_moe_decode_down_to_rank_f32_dev<false>(
        ctx, slots, ranks, token_ids, h, down_acc,
        num_active_experts, top_k, in_features, out_features, blocksize,
        quant_map, dep_event);
}

sycl::event moe_prefill_gate_up_rows(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int tokens,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (tokens <= 0 || num_active_experts <= 0) return {};

    bool nested = use_nested;
    if (!nested) {
        auto alloc_type = sycl::get_pointer_type(slots, ctx.queue().get_context());
        if (alloc_type != sycl::usm::alloc::device) {
            if (slots[0].gate_up_nested.codes != nullptr) {
                nested = true;
            }
        }
    }

    if (nested) {
        if (nested_fixed256) return launch_moe_prefill_gate_up_rows<true, true>(
            ctx, slots, quant_map, x, h, tokens, num_active_experts,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_prefill_gate_up_rows<true, false>(
            ctx, slots, quant_map, x, h, tokens, num_active_experts,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_prefill_gate_up_rows<false>(
        ctx, slots, quant_map, x, h, tokens, num_active_experts,
        in_features, intermediate_size, blocksize, dep_event);
}

sycl::event moe_prefill_down_rows(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* weights,
    const float* quant_map,
    const bf16* h,
    bf16* y,
    int tokens,
    int num_active_experts,
    int in_features,
    int out_features,
    int blocksize,
    const bf16* shared_out,
    const bf16* shared_gate,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (tokens <= 0 || num_active_experts <= 0) return {};

    bool nested = use_nested;
    if (!nested) {
        auto alloc_type = sycl::get_pointer_type(slots, ctx.queue().get_context());
        if (alloc_type != sycl::usm::alloc::device) {
            if (slots[0].down_nested.codes != nullptr) {
                nested = true;
            }
        }
    }

    if (nested) {
        if (nested_fixed256) return launch_moe_prefill_down_rows<true, true>(
            ctx, slots, weights, quant_map, h, y, tokens, num_active_experts,
            in_features, out_features, blocksize, shared_out, shared_gate, dep_event);
        return launch_moe_prefill_down_rows<true, false>(
            ctx, slots, weights, quant_map, h, y, tokens, num_active_experts,
            in_features, out_features, blocksize, shared_out, shared_gate, dep_event);
    }
    return launch_moe_prefill_down_rows<false>(
        ctx, slots, weights, quant_map, h, y, tokens, num_active_experts,
        in_features, out_features, blocksize, shared_out, shared_gate, dep_event);
}

void moe_silu_gate_up_mul_flat(
    Context& ctx,
    const bf16* gate_up,
    bf16* out,
    int total_tokens,
    int intermediate_size) {
    const size_t total_elements = static_cast<size_t>(total_tokens) * intermediate_size;
    constexpr size_t wg = 256;
    if ((intermediate_size % 8) == 0) {
        const size_t vecs_per_row = static_cast<size_t>(intermediate_size / 8);
        const size_t total_vecs = static_cast<size_t>(total_tokens) * vecs_per_row;
        const size_t num_wgs = (total_vecs + wg - 1) / wg;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t v_idx = item.get_global_id(0);
                    if (v_idx < total_vecs) {
                        const size_t t = v_idx / vecs_per_row;
                        const size_t c8 = (v_idx - t * vecs_per_row) * 8;
                        const size_t row_base = t * static_cast<size_t>(2 * intermediate_size);
                        const auto g8 = *reinterpret_cast<const sycl::vec<bf16, 8>*>(gate_up + row_base + c8);
                        const auto u8 = *reinterpret_cast<const sycl::vec<bf16, 8>*>(gate_up + row_base + intermediate_size + c8);
                        sycl::vec<bf16, 8> res;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const float g_f = static_cast<float>(g8[k]);
                            const float u_f = static_cast<float>(u8[k]);
                            const float silu_g = g_f / (1.0f + sycl::native::exp(-g_f));
                            res[k] = bf16(silu_g * u_f);
                        }
                        *reinterpret_cast<sycl::vec<bf16, 8>*>(out + v_idx * 8) = res;
                    }
                });
        });
        return;
    }
    const size_t num_wgs = (total_elements + wg - 1) / wg;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t idx = item.get_global_id(0);
                if (idx < total_elements) {
                    const size_t t = idx / intermediate_size;
                    const size_t i = idx % intermediate_size;
                    const size_t row_stride = static_cast<size_t>(2) * intermediate_size;
                    const bf16 g_val = gate_up[t * row_stride + i];
                    const bf16 u_val = gate_up[t * row_stride + intermediate_size + i];
                    const float g_f = static_cast<float>(g_val);
                    const float u_f = static_cast<float>(u_val);
                    const float silu_g = g_f / (1.0f + sycl::native::exp(-g_f));
                    out[idx] = bf16(silu_g * u_f);
                }
            });
    });
}

sycl::event moe_expert_major_prefill_gate_up_2token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_pairs,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (total_pairs <= 0) return {};
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_2token<true, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_pairs, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_expert_major_prefill_gate_up_2token<true, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_pairs, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_gate_up_2token<false>(
        ctx, slots, token_ids, ranks, quant_map, x, h, total_pairs, top_k,
        in_features, intermediate_size, blocksize, dep_event);
}

sycl::event moe_expert_major_prefill_down_2token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_pairs,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (total_pairs <= 0) return {};
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_down_2token<true, false, true>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_pairs, top_k,
            in_features, out_features, blocksize, dep_event);
        return launch_moe_expert_major_prefill_down_2token<true, false, false>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_pairs, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_down_2token<false>(
        ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_pairs, top_k,
        in_features, out_features, blocksize, dep_event);
}

sycl::event moe_expert_major_prefill_gate_up_4token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (total_quads <= 0) return {};
    const bool act_slm = q4e_prefill_act_slm_enabled() &&
        q4e_act_slm_shape_ok(in_features, intermediate_size, 4);
    if (act_slm) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token<true, true, true>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token<true, false, true>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token<false, false, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token<true, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_expert_major_prefill_gate_up_4token<true, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_gate_up_4token<false>(
        ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
        in_features, intermediate_size, blocksize, dep_event);
}

sycl::event moe_expert_major_prefill_down_4token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_quads,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (total_quads <= 0) return {};
    const bool act_slm = q4e_prefill_act_slm_enabled() &&
        q4e_act_slm_shape_ok(in_features, out_features, 4);
    if (act_slm) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_down_4token<true, false, true, true>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
            return launch_moe_expert_major_prefill_down_4token<true, false, false, true>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_down_4token<false, false, false, true>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_down_4token<true, false, true>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
        return launch_moe_expert_major_prefill_down_4token<true, false, false>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_down_4token<false>(
        ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
        in_features, out_features, blocksize, dep_event);
}

// Phase Q4E WG-share: opt-in regrouping of the 4-token quad section. share_quads
// selects how many of the 16 subgroups in a workgroup stream the same weight
// rows for different token quads. Results are bit-exact against the 4-token
// kernels above; invalid share_quads values fall back to the plain 4-token path.
sycl::event moe_expert_major_prefill_gate_up_4token_wgshare(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    int share_quads,
    bool nested_fixed256) {
    if (total_quads <= 0) return {};
    if (share_quads == 2) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare<true, true, 2>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare<true, false, 2>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare<false, false, 2>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (share_quads == 4) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare<true, true, 4>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare<true, false, 4>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare<false, false, 4>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (share_quads == 8) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare<true, true, 8>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare<true, false, 8>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare<false, false, 8>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token<true, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_expert_major_prefill_gate_up_4token<true, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_gate_up_4token<false>(
        ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
        in_features, intermediate_size, blocksize, dep_event);
}
sycl::event moe_expert_major_prefill_gate_up_4token_wgshare_nounpack(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    int share_quads,
    bool nested_fixed256) {
    if (total_quads <= 0) return {};
    if (share_quads == 2) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<true, true, 2>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<true, false, 2>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<false, false, 2>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (share_quads == 4) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<true, true, 4>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<true, false, 4>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<false, false, 4>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (share_quads == 8) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<true, true, 8>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<true, false, 8>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nounpack<false, false, 8>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token<true, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_expert_major_prefill_gate_up_4token<true, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_gate_up_4token<false>(
        ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
        in_features, intermediate_size, blocksize, dep_event);
}
sycl::event moe_expert_major_prefill_gate_up_4token_wgshare_slm(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (total_quads <= 0) return {};
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare_slm<true, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare_slm<true, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_gate_up_4token_wgshare_slm<false, false>(
        ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
        in_features, intermediate_size, blocksize, dep_event);
}
sycl::event moe_expert_major_prefill_gate_up_4token_wgshare_nolut(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    int share_quads,
    bool nested_fixed256) {
    if (total_quads <= 0) return {};
    if (share_quads == 2) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<true, true, 2>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<true, false, 2>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<false, false, 2>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (share_quads == 4) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<true, true, 4>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<true, false, 4>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<false, false, 4>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (share_quads == 8) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<true, true, 8>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<true, false, 8>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up_4token_wgshare_nolut<false, false, 8>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_4token<true, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_expert_major_prefill_gate_up_4token<true, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_gate_up_4token<false>(
        ctx, slots, token_ids, ranks, quant_map, x, h, total_quads, top_k,
        in_features, intermediate_size, blocksize, dep_event);
}

sycl::event moe_expert_major_prefill_down_4token_wgshare(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_quads,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    int share_quads,
    bool nested_fixed256) {
    if (total_quads <= 0) return {};
    if (share_quads == 2) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_down_4token_wgshare<true, false, true, 2>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
            return launch_moe_expert_major_prefill_down_4token_wgshare<true, false, false, 2>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_down_4token_wgshare<false, false, false, 2>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    if (share_quads == 4) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_down_4token_wgshare<true, false, true, 4>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
            return launch_moe_expert_major_prefill_down_4token_wgshare<true, false, false, 4>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_down_4token_wgshare<false, false, false, 4>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    if (share_quads == 8) {
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_down_4token_wgshare<true, false, true, 8>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
            return launch_moe_expert_major_prefill_down_4token_wgshare<true, false, false, 8>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_down_4token_wgshare<false, false, false, 8>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_down_4token<true, false, true>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
        return launch_moe_expert_major_prefill_down_4token<true, false, false>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_down_4token<false>(
        ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_quads, top_k,
        in_features, out_features, blocksize, dep_event);
}

sycl::event moe_expert_major_prefill_gate_up_8token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_eights,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (total_eights <= 0) return {};
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up_8token<true, true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_eights, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return launch_moe_expert_major_prefill_gate_up_8token<true, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_eights, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_gate_up_8token<false>(
        ctx, slots, token_ids, ranks, quant_map, x, h, total_eights, top_k,
        in_features, intermediate_size, blocksize, dep_event);
}

sycl::event moe_expert_major_prefill_down_8token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_eights,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    bool nested_fixed256) {
    if (total_eights <= 0) return {};
    if (use_nested) {
        if (nested_fixed256) return launch_moe_expert_major_prefill_down_8token<true, false, true>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_eights, top_k,
            in_features, out_features, blocksize, dep_event);
        return launch_moe_expert_major_prefill_down_8token<true, false, false>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_eights, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_down_8token<false>(
        ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_eights, top_k,
        in_features, out_features, blocksize, dep_event);
}

namespace {
// Opt-in WG-share regrouping of the 4-token expert-major quad section
// (AILA_Q4E_4TOKEN_WGSHARE = 0/2/4/8). 0 keeps the plain 4-token path, so the
// default behaviour is unchanged. Read once per process.
int q4e_4token_wgshare_shares() {
    static const int shares = [] {
        const int raw = aila::env::read_int("AILA_Q4E_4TOKEN_WGSHARE", 0);
        return (raw == 2 || raw == 4 || raw == 8) ? raw : 0;
    }();
    return shares;
}
} // namespace

sycl::event moe_expert_major_prefill_gate_up(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_acts,
    int top_k,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    int num_pairs,
    int num_quads,
    int num_eights,
    bool nested_fixed256) {
    if (total_acts <= 0) return {};
    if (num_eights <= 0 && num_quads <= 0 && num_pairs <= 0) {
        if (q4e_prefill_act_slm_enabled() &&
            q4e_act_slm_shape_ok(in_features, intermediate_size, 1)) {
            if (use_nested) {
                if (nested_fixed256) return launch_moe_expert_major_prefill_gate_up<true, true, true>(
                    ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
                return launch_moe_expert_major_prefill_gate_up<true, false, true>(
                    ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            }
            return launch_moe_expert_major_prefill_gate_up<false, false, true>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        auto ev = use_nested ? (nested_fixed256 ? launch_moe_expert_major_prefill_gate_up<true, true>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
                in_features, intermediate_size, blocksize, dep_event)
            : launch_moe_expert_major_prefill_gate_up<true, false>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
                in_features, intermediate_size, blocksize, dep_event))
            : launch_moe_expert_major_prefill_gate_up<false, false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
            in_features, intermediate_size, blocksize, dep_event);
        return ev;
    }

    const bool act_slm_quad = q4e_prefill_act_slm_enabled() &&
        q4e_act_slm_shape_ok(in_features, intermediate_size, 4);
    const bool act_slm_single = q4e_prefill_act_slm_enabled() &&
        q4e_act_slm_shape_ok(in_features, intermediate_size, 1);
    int remaining_acts = total_acts;
    int cur_offset = 0;
    sycl::event eight_ev{};
    sycl::event quad_ev{};
    sycl::event pair_ev{};
    sycl::event single_ev{};
    // A default-constructed sycl::event compares unequal to another default
    // event on this toolchain, so the returned fence must be selected by an
    // explicit "was submitted" flag, never by an event comparison.
    bool eight_submitted = false;
    bool quad_submitted = false;
    bool pair_submitted = false;
    bool single_submitted = false;

    // Eights lead the layout (see RunGroupedExperts) so the descending-width
    // chain lets every down section depend on its own gate/up section.
    if (num_eights > 0) {
        const int clamped_eights = std::min(num_eights, remaining_acts / 8);
        if (clamped_eights > 0) {
            if (use_nested) {
                if (nested_fixed256) eight_ev = launch_moe_expert_major_prefill_gate_up_8token<true, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_eights, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
                else eight_ev = launch_moe_expert_major_prefill_gate_up_8token<true, false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_eights, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            } else {
                eight_ev = launch_moe_expert_major_prefill_gate_up_8token<false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_eights, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            }
            eight_submitted = true;
            const int eight_acts = 8 * clamped_eights;
            cur_offset += eight_acts;
            remaining_acts -= eight_acts;
        }
    }

    if (num_quads > 0) {
        const int clamped_quads = std::min(num_quads, remaining_acts / 4);
        if (clamped_quads > 0) {
            const int wgshare = q4e_4token_wgshare_shares();
            if (wgshare) {
                quad_ev = moe_expert_major_prefill_gate_up_4token_wgshare(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_quads, top_k,
                    in_features, intermediate_size, blocksize, dep_event,
                    use_nested, wgshare, nested_fixed256);
            } else if (act_slm_quad) {
                if (use_nested) {
                    if (nested_fixed256) quad_ev = launch_moe_expert_major_prefill_gate_up_4token<true, true, true>(
                        ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                        quant_map, x, h, clamped_quads, top_k,
                        in_features, intermediate_size, blocksize, dep_event);
                    else quad_ev = launch_moe_expert_major_prefill_gate_up_4token<true, false, true>(
                        ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                        quant_map, x, h, clamped_quads, top_k,
                        in_features, intermediate_size, blocksize, dep_event);
                } else {
                    quad_ev = launch_moe_expert_major_prefill_gate_up_4token<false, false, true>(
                        ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                        quant_map, x, h, clamped_quads, top_k,
                        in_features, intermediate_size, blocksize, dep_event);
                }
            } else if (use_nested) {
                if (nested_fixed256) quad_ev = launch_moe_expert_major_prefill_gate_up_4token<true, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_quads, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
                else quad_ev = launch_moe_expert_major_prefill_gate_up_4token<true, false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_quads, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            } else {
                quad_ev = launch_moe_expert_major_prefill_gate_up_4token<false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_quads, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            }
            quad_submitted = true;
            const int quad_acts = 4 * clamped_quads;
            cur_offset += quad_acts;
            remaining_acts -= quad_acts;
        }
    }

    if (num_pairs > 0 && remaining_acts >= 2) {
        const int clamped_pairs = std::min(num_pairs, remaining_acts / 2);
        if (clamped_pairs > 0) {
            if (use_nested) {
                if (nested_fixed256) pair_ev = launch_moe_expert_major_prefill_gate_up_2token<true, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_pairs, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
                else pair_ev = launch_moe_expert_major_prefill_gate_up_2token<true, false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_pairs, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            } else {
                pair_ev = launch_moe_expert_major_prefill_gate_up_2token<false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, clamped_pairs, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            }
            pair_submitted = true;
            const int pair_acts = 2 * clamped_pairs;
            cur_offset += pair_acts;
            remaining_acts -= pair_acts;
        }
    }

    if (remaining_acts > 0) {
        if (act_slm_single) {
            if (use_nested) {
                if (nested_fixed256) single_ev = launch_moe_expert_major_prefill_gate_up<true, true, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, remaining_acts, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
                else single_ev = launch_moe_expert_major_prefill_gate_up<true, false, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, remaining_acts, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            } else {
                single_ev = launch_moe_expert_major_prefill_gate_up<false, false, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, x, h, remaining_acts, top_k,
                    in_features, intermediate_size, blocksize, dep_event);
            }
        } else if (use_nested) {
            if (nested_fixed256) single_ev = launch_moe_expert_major_prefill_gate_up<true, true>(
                ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                quant_map, x, h, remaining_acts, top_k,
                in_features, intermediate_size, blocksize, dep_event);
            else single_ev = launch_moe_expert_major_prefill_gate_up<true, false>(
                ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                quant_map, x, h, remaining_acts, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        } else {
            single_ev = launch_moe_expert_major_prefill_gate_up<false>(
                ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                quant_map, x, h, remaining_acts, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        single_submitted = true;
    }

    if (single_submitted) return single_ev;
    if (pair_submitted) return pair_ev;
    if (quad_submitted) return quad_ev;
    if (eight_submitted) return eight_ev;
    return {};
}

template <bool Compact>
static sycl::event dispatch_moe_expert_major_prefill_down(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_acts,
    int top_k,
    int in_features,
    int out_features,
    int blocksize,
    const sycl::event* dep_event,
    bool use_nested,
    int num_pairs,
    int num_quads,
    int num_eights,
    bool nested_fixed256) {
    if (total_acts <= 0) return {};
    if (num_eights <= 0 && num_quads <= 0 && num_pairs <= 0) {
        if (q4e_prefill_act_slm_enabled() &&
            q4e_act_slm_shape_ok(in_features, out_features, 1)) {
            if (use_nested) {
                if (nested_fixed256) return launch_moe_expert_major_prefill_down<true, Compact, true, true>(
                    ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
                    in_features, out_features, blocksize, dep_event);
                return launch_moe_expert_major_prefill_down<true, Compact, false, true>(
                    ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
                    in_features, out_features, blocksize, dep_event);
            }
            return launch_moe_expert_major_prefill_down<false, Compact, false, true>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        if (use_nested) {
            if (nested_fixed256) return launch_moe_expert_major_prefill_down<true, Compact, true>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
                in_features, out_features, blocksize, dep_event);
            return launch_moe_expert_major_prefill_down<true, Compact, false>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_down<false, Compact>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
            in_features, out_features, blocksize, dep_event);
    }

    const bool act_slm_quad = q4e_prefill_act_slm_enabled() &&
        q4e_act_slm_shape_ok(in_features, out_features, 4);
    const bool act_slm_single = q4e_prefill_act_slm_enabled() &&
        q4e_act_slm_shape_ok(in_features, out_features, 1);
    int remaining_acts = total_acts;
    int cur_offset = 0;
    sycl::event eight_ev{};
    sycl::event quad_ev{};
    sycl::event pair_ev{};
    sycl::event single_ev{};
    // A default-constructed sycl::event compares unequal to another default
    // event on this toolchain, so the returned fence must be selected by an
    // explicit "was submitted" flag, never by an event comparison.
    bool eight_submitted = false;
    bool quad_submitted = false;
    bool pair_submitted = false;
    bool single_submitted = false;

    // Eights lead the layout (see RunGroupedExperts); publishing them first
    // keeps every down section behind its matching gate/up section.
    if (num_eights > 0) {
        const int clamped_eights = std::min(num_eights, remaining_acts / 8);
        if (clamped_eights > 0) {
            if (use_nested) {
                if (nested_fixed256) eight_ev = launch_moe_expert_major_prefill_down_8token<true, Compact, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_eights, top_k,
                    in_features, out_features, blocksize, dep_event);
                else eight_ev = launch_moe_expert_major_prefill_down_8token<true, Compact, false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_eights, top_k,
                    in_features, out_features, blocksize, dep_event);
            } else {
                eight_ev = launch_moe_expert_major_prefill_down_8token<false, Compact>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_eights, top_k,
                    in_features, out_features, blocksize, dep_event);
            }
            eight_submitted = true;
            const int eight_acts = 8 * clamped_eights;
            cur_offset += eight_acts;
            remaining_acts -= eight_acts;
        }
    }

    if (num_quads > 0) {
        const int clamped_quads = std::min(num_quads, remaining_acts / 4);
        if (clamped_quads > 0) {
            const int wgshare = q4e_4token_wgshare_shares();
            if (wgshare) {
                quad_ev = moe_expert_major_prefill_down_4token_wgshare(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_quads, top_k,
                    in_features, out_features, blocksize, dep_event,
                    use_nested, wgshare, nested_fixed256);
            } else if (act_slm_quad) {
                if (use_nested) {
                    if (nested_fixed256) quad_ev = launch_moe_expert_major_prefill_down_4token<true, Compact, true, true>(
                        ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                        quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_quads, top_k,
                        in_features, out_features, blocksize, dep_event);
                    else quad_ev = launch_moe_expert_major_prefill_down_4token<true, Compact, false, true>(
                        ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                        quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_quads, top_k,
                        in_features, out_features, blocksize, dep_event);
                } else {
                    quad_ev = launch_moe_expert_major_prefill_down_4token<false, Compact, false, true>(
                        ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                        quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_quads, top_k,
                        in_features, out_features, blocksize, dep_event);
                }
            } else if (use_nested) {
                if (nested_fixed256) quad_ev = launch_moe_expert_major_prefill_down_4token<true, Compact, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_quads, top_k,
                    in_features, out_features, blocksize, dep_event);
                else quad_ev = launch_moe_expert_major_prefill_down_4token<true, Compact, false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_quads, top_k,
                    in_features, out_features, blocksize, dep_event);
            } else {
                quad_ev = launch_moe_expert_major_prefill_down_4token<false, Compact>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_quads, top_k,
                    in_features, out_features, blocksize, dep_event);
            }
            quad_submitted = true;
            const int quad_acts = 4 * clamped_quads;
            cur_offset += quad_acts;
            remaining_acts -= quad_acts;
        }
    }

    if (num_pairs > 0 && remaining_acts >= 2) {
        const int clamped_pairs = std::min(num_pairs, remaining_acts / 2);
        if (clamped_pairs > 0) {
            if (use_nested) {
                if (nested_fixed256) pair_ev = launch_moe_expert_major_prefill_down_2token<true, Compact, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_pairs, top_k,
                    in_features, out_features, blocksize, dep_event);
                else pair_ev = launch_moe_expert_major_prefill_down_2token<true, Compact, false>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_pairs, top_k,
                    in_features, out_features, blocksize, dep_event);
            } else {
                pair_ev = launch_moe_expert_major_prefill_down_2token<false, Compact>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), clamped_pairs, top_k,
                    in_features, out_features, blocksize, dep_event);
            }
            pair_submitted = true;
            const int pair_acts = 2 * clamped_pairs;
            cur_offset += pair_acts;
            remaining_acts -= pair_acts;
        }
    }

    if (remaining_acts > 0) {
        if (act_slm_single) {
            if (use_nested) {
                if (nested_fixed256) single_ev = launch_moe_expert_major_prefill_down<true, Compact, true, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), remaining_acts, top_k,
                    in_features, out_features, blocksize, dep_event);
                else single_ev = launch_moe_expert_major_prefill_down<true, Compact, false, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), remaining_acts, top_k,
                    in_features, out_features, blocksize, dep_event);
            } else {
                single_ev = launch_moe_expert_major_prefill_down<false, Compact, false, true>(
                    ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                    quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), remaining_acts, top_k,
                    in_features, out_features, blocksize, dep_event);
            }
        } else if (use_nested) {
            if (nested_fixed256) single_ev = launch_moe_expert_major_prefill_down<true, Compact, true>(
                ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), remaining_acts, top_k,
                in_features, out_features, blocksize, dep_event);
            else single_ev = launch_moe_expert_major_prefill_down<true, Compact, false>(
                ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), remaining_acts, top_k,
                in_features, out_features, blocksize, dep_event);
        } else {
            single_ev = launch_moe_expert_major_prefill_down<false, Compact>(
                ctx, slots + cur_offset, token_ids + cur_offset, ranks + cur_offset,
                quant_map, h, down_acc + (Compact ? size_t(cur_offset)*out_features : 0), remaining_acts, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        single_submitted = true;
    }

    if (single_submitted) return single_ev;
    if (pair_submitted) return pair_ev;
    if (quad_submitted) return quad_ev;
    if (eight_submitted) return eight_ev;
    return {};
}

sycl::event moe_expert_major_prefill_down(
    Context& ctx, const MoeExpertSlotDev* slots, const int* tokens, const int* ranks,
    const float* qmap, const bf16* h, float* down, int acts, int top_k,
    int in_features, int out_features, int blocksize, const sycl::event* dep,
    bool nested, int pairs, int quads, int eights, bool compact, bool nested_fixed256) {
    if (compact) return dispatch_moe_expert_major_prefill_down<true>(ctx,slots,tokens,ranks,
        qmap,h,down,acts,top_k,in_features,out_features,blocksize,dep,nested,pairs,quads,eights,nested_fixed256);
    return dispatch_moe_expert_major_prefill_down<false>(ctx,slots,tokens,ranks,
        qmap,h,down,acts,top_k,in_features,out_features,blocksize,dep,nested,pairs,quads,eights,nested_fixed256);
}

void moe_build_inverse_map(Context& ctx, const int* tokens, const int* ranks,
                           int* inverse, int acts, int top_k) {
    if(acts<=0)return;
    if(top_k<1 || top_k>16)throw std::invalid_argument("Compact MoE top_k must be 1..16");
    ctx.queue().parallel_for(sycl::range<1>(acts),[=](sycl::id<1> i) [[sycl::reqd_sub_group_size(16)]] {
        inverse[tokens[i]*top_k+ranks[i]]=int(i[0]);
    });
}

sycl::event moe_merge_compact_down(Context& ctx, const float* down, const int* tokens,
    const int* inverse, const float* weights, int64_t* sum, int* invalid, int first, int acts,
    int top_k, int H, const sycl::event* dep) {
    if(acts<=0)return {};
    if(first<0 || H<=0 || H%4 || top_k<1 || top_k>16)
        throw std::invalid_argument("Invalid compact MoE merge shape");
    // Each token is owned by its first activation in this tile. No float atomics
    // or races when several experts in the tile contribute to the same token.
    return ctx.queue().submit([&](sycl::handler& cgh) {
        if(dep && *dep!=sycl::event{})cgh.depends_on(*dep);
        cgh.parallel_for(sycl::nd_range<2>({size_t(acts),128},{1,128}),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                int a=int(item.get_group(0)), t=tokens[first+a], owner=first+acts;
                for(int k=0;k<top_k;++k){int p=inverse[t*top_k+k];if(p>=first && p<first+acts)owner=sycl::min(owner,p);}
                if(first+a!=owner)return;
                using V=sycl::vec<float,4>;
                using S=sycl::vec<int64_t,4>;
                for(int j=int(item.get_local_id(1))*4;j<H;j+=128*4){
                    S acc=*reinterpret_cast<S*>(sum+size_t(t)*H+j);
                    for(int k=0;k<top_k;++k){
                        int p=inverse[t*top_k+k];
                        if(p>=first && p<first+acts) {
                            const V v=weights[t*top_k+k] * *reinterpret_cast<const V*>(down+size_t(p-first)*H+j);
                            for(int lane=0;lane<4;++lane) {
                                // Q36: <=2^-37 absolute rounding per contribution.
                                // |term|<=2^20 and top_k<=16 keep |sum|<=2^60.
                                // Inspect bits: -ffast-math may remove isfinite checks.
                                const uint32_t magnitude=sycl::bit_cast<uint32_t>(float(v[lane]))&0x7fffffffU;
                                if(magnitude>0x49800000U) {
                                    sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>(*invalid).store(1);
                                } else acc[lane]+=int64_t(sycl::rint(v[lane]*68719476736.f));
                            }
                        }
                    }
                    *reinterpret_cast<S*>(sum+size_t(t)*H+j)=acc;
                }
            });
    });
}

sycl::event moe_finish_compact_down(Context& ctx, const int64_t* sum, bf16* y,
    int tokens, int H, const bf16* shared_out, const bf16* shared_gate) {
    return ctx.queue().parallel_for(sycl::range<1>(size_t(tokens)*H),
        [=](sycl::id<1> id) [[sycl::reqd_sub_group_size(16)]] {
            const size_t i=id[0];const bf16 routed=bf16(float(sum[i])*(1.f/68719476736.f));
            if(shared_out && shared_gate){
                const float g=float(shared_gate[i/H]);
                const float gate=1.f/(1.f+sycl::native::exp(-g));
                y[i]=bf16(float(routed)+gate*float(shared_out[i]));
            } else y[i]=routed;
        });
}

sycl::event moe_ordered_accumulate_down_rows(
    Context& ctx,
    const float* down_acc,
    const float* weights,
    bf16* y,
    int tokens,
    int top_k,
    int out_features,
    const bf16* shared_out,
    const bf16* shared_gate,
    const sycl::event* dep_event,
    const void* cpu_result_packet) {
    if (cpu_result_packet && (tokens != 1 || top_k > 16)) {
        throw std::invalid_argument("CPU result packet requires one token and at most 16 ranks");
    }
    if (tokens <= 0 || out_features <= 0) return {};
    constexpr size_t wg_size = 256;
    const size_t total_elements = static_cast<size_t>(tokens) * out_features;
    const size_t num_wgs = (total_elements + wg_size - 1) / wg_size;
    const auto* cpu_ranks = static_cast<const int*>(cpu_result_packet);
    const auto* cpu_down = cpu_ranks ? reinterpret_cast<const float*>(cpu_ranks + 16) : nullptr;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) {
            cgh.depends_on(*dep_event);
        }
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const size_t gid = item.get_global_id(0);
                if (gid < total_elements) {
                    const int token = static_cast<int>(gid / out_features);
                    const int row = static_cast<int>(gid % out_features);

                    float total_val = 0.0f;
                    const size_t token_base = static_cast<size_t>(token) * top_k;
                    for (int k = 0; k < top_k; ++k) {
                        const float w_exp = weights[token_base + k];
                        const int cpu_row = cpu_ranks ? cpu_ranks[k] : -1;
                        const float val = cpu_row >= 0 ? cpu_down[size_t(cpu_row) * out_features + row]
                                                       : down_acc[(token_base + k) * out_features + row];
                        total_val += w_exp * val;
                    }

                    const bf16 routed = bf16(total_val);
                    if (shared_out && shared_gate) {
                        const float gate = static_cast<float>(shared_gate[token]);
                        const float gate_act = 1.0f / (1.0f + sycl::native::exp(-gate));
                        y[gid] = bf16(static_cast<float>(routed) +
                                      gate_act * static_cast<float>(shared_out[gid]));
                    } else {
                        y[gid] = routed;
                    }
                }
            });
    });
}

sycl::event moe_ordered_accumulate_cpu_host_rows(
    Context& ctx, const float* down_acc, const float* weights, bf16* y,
    int tokens, int top_k, int out_features, const MoeCpuHostDownRows& cpu_host_rows,
    const bf16* shared_out, const bf16* shared_gate, const sycl::event* dep_event) {
    if (tokens != 1 || top_k <= 0 || top_k > 16 || out_features <= 0 ||
        !cpu_host_rows.values || cpu_host_rows.rows <= 0 || cpu_host_rows.rows > 16)
        throw std::invalid_argument("CPU host rows require one token and at most 16 ranks");
    for (int k = 0; k < top_k; ++k)
        if (cpu_host_rows.rank_to_row[k] < -1 || cpu_host_rows.rank_to_row[k] >= cpu_host_rows.rows)
            throw std::invalid_argument("CPU host row mapping is out of range");
    constexpr size_t wg_size = 256;
    const size_t num_wgs = (static_cast<size_t>(out_features) + wg_size - 1) / wg_size;
    const MoeCpuHostDownRows host_rows = cpu_host_rows;
    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) cgh.depends_on(*dep_event);
        cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                const size_t row = item.get_global_id(0);
                if (row >= static_cast<size_t>(out_features)) return;
                float total_val = 0.0f;
                for (int k = 0; k < top_k; ++k) {
                    const float w_exp = weights[k];
                    const int cpu_row = host_rows.rank_to_row[k];
                    const float val = cpu_row >= 0 ? host_rows.values[size_t(cpu_row) * out_features + row]
                                                   : down_acc[size_t(k) * out_features + row];
                    total_val += w_exp * val;
                }
                const bf16 routed = bf16(total_val);
                if (shared_out && shared_gate) {
                    const float gate = static_cast<float>(shared_gate[0]);
                    const float gate_act = 1.0f / (1.0f + sycl::native::exp(-gate));
                    y[row] = bf16(static_cast<float>(routed) +
                                  gate_act * static_cast<float>(shared_out[row]));
                } else y[row] = routed;
            });
    });
}

} // namespace ops
