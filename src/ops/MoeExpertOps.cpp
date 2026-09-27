#include "MoeExpertOps.hpp"

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

sycl::event router_softmax_topk_device(
    Context& ctx,
    const bf16* router_scores_ptr,
    const bf16* per_expert_scale_ptr,
    int* expert_ids_dev,
    float* expert_weights_dev,
    int tokens,
    int num_experts,
    int top_k,
    bool norm_topk_prob) {
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
                    }
                }
            });
    });
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

template <bool UseNested>
inline float load_expert_absmax(const float* absmax_f32,
                                const MoeNestedScaleDev& nested,
                                size_t index) {
    if constexpr (UseNested) {
        // Match the host fill's two f32 rounding steps. A single contracted
        // FMA changes some reconstructed absmax values by 1 ULP, which can
        // change greedy routing after the 48-layer recurrence.
        const float product = sycl::fma(
            nested.nested_quant_map[nested.codes[index]],
            nested.nested_absmax[index / static_cast<size_t>(nested.blocksize)], 0.0f);
        return sycl::fma(product, 1.0f, nested.offset);
    } else {
        return absmax_f32[index];
    }
}

template <bool UseNested>
sycl::event launch_moe_decode_gate_up_swiglu_grouped(
    Context& ctx,
    const std::array<MoeExpertSlotDev, 16>& slot_ptrs,
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
                                ? load_expert_absmax<UseNested>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested>(
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
                            ga = load_expert_absmax<UseNested>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested>(
                                slot.gate_up_absmax, slot.gate_up_nested, up_scale_index);
                        }
                        const vec8 in_v = *reinterpret_cast<const vec8*>(x + byte_offset * 2);

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

template <bool UseNested>
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
                                    ? load_expert_absmax<UseNested>(
                                          slot.down_absmax, slot.down_nested, scale_index)
                                    : 0.0f;
                                const float da0 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 0);
                                const float da8 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 8);
                                da = sg_lane < 8 ? da0 : da8;
                            } else {
                                da = load_expert_absmax<UseNested>(
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

template <bool UseNested>
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
                                ? load_expert_absmax<UseNested>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested>(
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
                            ga = load_expert_absmax<UseNested>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested>(
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

template <bool UseNested>
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
                                    ? load_expert_absmax<UseNested>(
                                          slot.down_absmax, slot.down_nested, scale_index)
                                    : 0.0f;
                                const float da0 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 0);
                                const float da8 = sycl::group_broadcast(
                                    item.get_sub_group(), da_leader, 8);
                                da = sg_lane < 8 ? da0 : da8;
                            } else {
                                da = load_expert_absmax<UseNested>(
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

template <bool UseNested>
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
                                ? load_expert_absmax<UseNested>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested>(
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
                            ga = load_expert_absmax<UseNested>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested>(
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

template <bool UseNested>
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
                                ? load_expert_absmax<UseNested>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested>(
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
                    const int act_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const int token = token_ids[act_idx];
                    const int rank = ranks[act_idx];
                    const size_t out_pos = (static_cast<size_t>(token) * top_k + rank) * out_features + row;
                    down_acc[out_pos] = expert_sum;
                }
            });
    });
}

template <bool UseNested>
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
                                ? load_expert_absmax<UseNested>(
                                      slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? load_expert_absmax<UseNested>(
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
                            ga = load_expert_absmax<UseNested>(
                                slot.gate_up_absmax, slot.gate_up_nested, gate_scale_index);
                            ua = load_expert_absmax<UseNested>(
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

template <bool UseNested>
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
                                ? load_expert_absmax<UseNested>(
                                      slot.down_absmax, slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
                        } else {
                            da = load_expert_absmax<UseNested>(
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
                    const size_t out_pos0 = (static_cast<size_t>(token0) * top_k + rank0) * out_features + row;
                    down_acc[out_pos0] = expert_sum0;

                    const int token1 = token_ids[pair_idx * 2 + 1];
                    const int rank1 = ranks[pair_idx * 2 + 1];
                    const size_t out_pos1 = (static_cast<size_t>(token1) * top_k + rank1) * out_features + row;
                    down_acc[out_pos1] = expert_sum1;
                }
            });
    });
}

} // namespace

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
    const sycl::event* dep_event) {
    if (num_active_experts <= 0) return {};

    std::array<MoeExpertSlotDev, 16> slot_ptrs{};
    const int k_capped = std::min(num_active_experts, 16);
    for (int i = 0; i < k_capped; ++i) {
        slot_ptrs[static_cast<size_t>(i)] = slots[i];
    }

    if (slots[0].gate_up_nested.codes != nullptr) {
        return launch_moe_decode_gate_up_swiglu_grouped<true>(
            ctx, slot_ptrs, quant_map, x, h, num_active_experts,
            in_features, intermediate_size, blocksize, dep_event);
    }
    return launch_moe_decode_gate_up_swiglu_grouped<false>(
        ctx, slot_ptrs, quant_map, x, h, num_active_experts,
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
    const bf16* shared_gate) {
    if (num_active_experts <= 0) return {};

    std::array<MoeExpertSlotDev, 16> slot_ptrs{};
    std::array<float, 16> weight_ptrs{};
    const int k_capped = std::min(num_active_experts, 16);
    for (int i = 0; i < k_capped; ++i) {
        slot_ptrs[static_cast<size_t>(i)] = slots[i];
        weight_ptrs[static_cast<size_t>(i)] = weights[i];
    }

    if (slots[0].down_nested.codes != nullptr) {
        return launch_moe_decode_down_weighted_accumulate<true>(
            ctx, slot_ptrs, weight_ptrs, quant_map, h, acc_f32, out_bf16,
            num_active_experts, in_features, out_features, blocksize,
            shared_out, shared_gate);
    }
    return launch_moe_decode_down_weighted_accumulate<false>(
        ctx, slot_ptrs, weight_ptrs, quant_map, h, acc_f32, out_bf16,
        num_active_experts, in_features, out_features, blocksize,
        shared_out, shared_gate);
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
    bool use_nested) {
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
        return launch_moe_prefill_gate_up_rows<true>(
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
    bool use_nested) {
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
        return launch_moe_prefill_down_rows<true>(
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
    bool use_nested) {
    if (total_pairs <= 0) return {};
    if (use_nested) {
        return launch_moe_expert_major_prefill_gate_up_2token<true>(
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
    bool use_nested) {
    if (total_pairs <= 0) return {};
    if (use_nested) {
        return launch_moe_expert_major_prefill_down_2token<true>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_pairs, top_k,
            in_features, out_features, blocksize, dep_event);
    }
    return launch_moe_expert_major_prefill_down_2token<false>(
        ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_pairs, top_k,
        in_features, out_features, blocksize, dep_event);
}

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
    int num_pairs) {
    if (total_acts <= 0) return {};
    if (num_pairs <= 0) {
        if (use_nested) {
            return launch_moe_expert_major_prefill_gate_up<true>(
                ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
                in_features, intermediate_size, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_gate_up<false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, total_acts, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }

    const int clamped_pairs = std::min(num_pairs, total_acts / 2);
    const int single_acts = total_acts - 2 * clamped_pairs;
    sycl::event pair_ev{};
    if (use_nested) {
        pair_ev = launch_moe_expert_major_prefill_gate_up_2token<true>(
            ctx, slots, token_ids, ranks, quant_map, x, h, clamped_pairs, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    } else {
        pair_ev = launch_moe_expert_major_prefill_gate_up_2token<false>(
            ctx, slots, token_ids, ranks, quant_map, x, h, clamped_pairs, top_k,
            in_features, intermediate_size, blocksize, dep_event);
    }

    if (single_acts <= 0) {
        return pair_ev;
    }

    const int single_offset = 2 * clamped_pairs;
    if (use_nested) {
        return launch_moe_expert_major_prefill_gate_up<true>(
            ctx, slots + single_offset, token_ids + single_offset, ranks + single_offset,
            quant_map, x, h, single_acts, top_k,
            in_features, intermediate_size, blocksize, &pair_ev);
    }
    return launch_moe_expert_major_prefill_gate_up<false>(
        ctx, slots + single_offset, token_ids + single_offset, ranks + single_offset,
        quant_map, x, h, single_acts, top_k,
        in_features, intermediate_size, blocksize, &pair_ev);
}

sycl::event moe_expert_major_prefill_down(
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
    int num_pairs) {
    if (total_acts <= 0) return {};
    if (num_pairs <= 0) {
        if (use_nested) {
            return launch_moe_expert_major_prefill_down<true>(
                ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
                in_features, out_features, blocksize, dep_event);
        }
        return launch_moe_expert_major_prefill_down<false>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, total_acts, top_k,
            in_features, out_features, blocksize, dep_event);
    }

    const int clamped_pairs = std::min(num_pairs, total_acts / 2);
    const int single_acts = total_acts - 2 * clamped_pairs;
    sycl::event pair_ev{};
    if (use_nested) {
        pair_ev = launch_moe_expert_major_prefill_down_2token<true>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, clamped_pairs, top_k,
            in_features, out_features, blocksize, dep_event);
    } else {
        pair_ev = launch_moe_expert_major_prefill_down_2token<false>(
            ctx, slots, token_ids, ranks, quant_map, h, down_acc, clamped_pairs, top_k,
            in_features, out_features, blocksize, dep_event);
    }

    if (single_acts <= 0) {
        return pair_ev;
    }

    const int single_offset = 2 * clamped_pairs;
    if (use_nested) {
        return launch_moe_expert_major_prefill_down<true>(
            ctx, slots + single_offset, token_ids + single_offset, ranks + single_offset,
            quant_map, h, down_acc, single_acts, top_k,
            in_features, out_features, blocksize, &pair_ev);
    }
    return launch_moe_expert_major_prefill_down<false>(
        ctx, slots + single_offset, token_ids + single_offset, ranks + single_offset,
        quant_map, h, down_acc, single_acts, top_k,
        in_features, out_features, blocksize, &pair_ev);
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
    const sycl::event* dep_event) {
    if (tokens <= 0 || out_features <= 0) return {};
    constexpr size_t wg_size = 256;
    const size_t total_elements = static_cast<size_t>(tokens) * out_features;
    const size_t num_wgs = (total_elements + wg_size - 1) / wg_size;

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
                        const float val = down_acc[(token_base + k) * out_features + row];
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

} // namespace ops
