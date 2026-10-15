#include "MoeDecodeVariant.hpp"

namespace ops {
namespace {

// Same table as MoeExpertOps.cpp:8-11 (file-static there, so the device
// compilation of this TU needs its own copy).
static constexpr float kDefaultNf4QuantMap[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// Mirror of load_expert_absmax<UseNested=true> (MoeExpertOps.cpp:406-423) with
// the fixed-256 contract folded in. The two fma steps are kept separate so
// the reconstructed float is bit-identical to the production path.
template <bool FixedScale256>
inline float variant_nested_scale(const MoeNestedScaleDev& nested, size_t index) {
    size_t nested_block;
    if constexpr (FixedScale256) nested_block = index >> 8; // nested.blocksize == 256
    else nested_block = index / static_cast<size_t>(nested.blocksize);
    const float product = sycl::fma(
        nested.nested_quant_map[nested.codes[index]],
        nested.nested_absmax[nested_block], 0.0f);
    return sycl::fma(product, 1.0f, nested.offset);
}

// gate_up GEMV + SwiGLU, device-slot form. Mirror of
// launch_moe_decode_gate_up_swiglu_grouped_dev<true, false, FixedNested256>
// (MoeExpertOps.cpp:853-999) specialized to the Q4E geometry contract
// (blocksize 64, nested codes, sub-group 16, 16 rows per work-group).
template <bool FixedScale256, bool UniformScale>
sycl::event launch_gate_up_variant(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    const sycl::event* dep_event) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / 64;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(num_active_experts) * intermediate_size;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) cgh.depends_on(*dep_event);
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / static_cast<int>(sub_group_size);
                const int sg_lane = lid % static_cast<int>(sub_group_size);
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
                    const auto& slot = slots[expert_idx];

                    const size_t gate_byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t gate_block_base = static_cast<size_t>(row) * blocks_per_row;
                    const size_t up_byte_base =
                        static_cast<size_t>(row + intermediate_size) * packed_bytes_per_row;
                    const size_t up_block_base =
                        static_cast<size_t>(row + intermediate_size) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t g4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + gate_byte_base + byte_offset);
                        const uint32_t u4 = *reinterpret_cast<const uint32_t*>(
                            slot.gate_up_packed + up_byte_base + byte_offset);
                        // Production: byte_offset / (blocksize / 2); exact >> 5 at 64.
                        const size_t scale_step = FixedScale256
                            ? static_cast<size_t>(byte_offset >> 5)
                            : static_cast<size_t>(byte_offset / 32);
                        const size_t gate_scale_index = gate_block_base + scale_step;
                        const size_t up_scale_index = up_block_base + scale_step;
                        float ga;
                        float ua;
                        if constexpr (UniformScale) {
                            ga = variant_nested_scale<FixedScale256>(
                                slot.gate_up_nested, gate_scale_index);
                            ua = variant_nested_scale<FixedScale256>(
                                slot.gate_up_nested, up_scale_index);
                        } else {
                            const float ga_leader = (sg_lane == 0 || sg_lane == 8)
                                ? variant_nested_scale<FixedScale256>(
                                      slot.gate_up_nested, gate_scale_index)
                                : 0.0f;
                            const float ua_leader = (sg_lane == 0 || sg_lane == 8)
                                ? variant_nested_scale<FixedScale256>(
                                      slot.gate_up_nested, up_scale_index)
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

                const float gate_sum =
                    sycl::reduce_over_group(item.get_sub_group(), gate_acc, sycl::plus<>());
                const float up_sum =
                    sycl::reduce_over_group(item.get_sub_group(), up_acc, sycl::plus<>());
                if (sg_lane == 0 && global_row < total_rows) {
                    const float silu_gate = gate_sum / (1.0f + sycl::native::exp(-gate_sum));
                    h[global_row] = bf16(silu_gate * up_sum);
                }
            });
    });
}

// down-to-rank GEMV, device-slot form. Mirror of
// launch_moe_decode_down_to_rank_f32_dev<true, false, FixedNested256>
// (MoeExpertOps.cpp:1211-1335), same specialization contract as above.
template <bool FixedScale256, bool UniformScale>
sycl::event launch_down_rank_variant(
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
    const sycl::event* dep_event) {
    const int packed_bytes_per_row = in_features / 2;
    const int blocks_per_row = in_features / 64;
    constexpr size_t wg_size = 256;
    constexpr size_t sub_group_size = 16;
    constexpr size_t rows_per_group = wg_size / sub_group_size;
    const size_t total_rows = static_cast<size_t>(num_active_experts) * out_features;
    const size_t num_groups = (total_rows + rows_per_group - 1) / rows_per_group;

    return ctx.queue().submit([&](sycl::handler& cgh) {
        if (dep_event && *dep_event != sycl::event{}) cgh.depends_on(*dep_event);
        sycl::local_accessor<float, 1> qmap(sycl::range<1>(16), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                using vec8 = sycl::vec<bf16, 8>;
                const int lid = static_cast<int>(item.get_local_id(0));
                const int sg_id = lid / static_cast<int>(sub_group_size);
                const int sg_lane = lid % static_cast<int>(sub_group_size);
                const size_t global_row = item.get_group(0) * rows_per_group + sg_id;
                const int stride = static_cast<int>(sub_group_size) * 4;

                if (lid < 16) {
                    // Production decode calls this op with quant_map == nullptr
                    // (executor launch_down), so the default codebook is the
                    // only table the variant needs.
                    qmap[lid] = kDefaultNf4QuantMap[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                float expert_acc = 0.0f;
                if (global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const auto& slot = slots[expert_idx];
                    const bf16* in_expert = h + static_cast<size_t>(expert_idx) * in_features;

                    const size_t byte_base = static_cast<size_t>(row) * packed_bytes_per_row;
                    const size_t block_base = static_cast<size_t>(row) * blocks_per_row;

                    int byte_offset = sg_lane * 4;
                    for (; byte_offset + 3 < packed_bytes_per_row; byte_offset += stride) {
                        const uint32_t d4 = *reinterpret_cast<const uint32_t*>(
                            slot.down_packed + byte_base + byte_offset);
                        const size_t scale_step = FixedScale256
                            ? static_cast<size_t>(byte_offset >> 5)
                            : static_cast<size_t>(byte_offset / 32);
                        const size_t scale_index = block_base + scale_step;
                        float da;
                        if constexpr (UniformScale) {
                            da = variant_nested_scale<FixedScale256>(
                                slot.down_nested, scale_index);
                        } else {
                            const float da_leader = (sg_lane == 0 || sg_lane == 8)
                                ? variant_nested_scale<FixedScale256>(
                                      slot.down_nested, scale_index)
                                : 0.0f;
                            const float da0 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 0);
                            const float da8 = sycl::group_broadcast(
                                item.get_sub_group(), da_leader, 8);
                            da = sg_lane < 8 ? da0 : da8;
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

                const float expert_sum =
                    sycl::reduce_over_group(item.get_sub_group(), expert_acc, sycl::plus<>());
                if (sg_lane == 0 && global_row < total_rows) {
                    const int expert_idx = static_cast<int>(global_row / out_features);
                    const int row = static_cast<int>(global_row % out_features);
                    const int token = token_ids ? token_ids[expert_idx] : 0;
                    const int rank = ranks[expert_idx];
                    const size_t out_pos =
                        (static_cast<size_t>(token) * top_k + static_cast<size_t>(rank)) *
                            out_features +
                        static_cast<size_t>(row);
                    down_acc[out_pos] = expert_sum;
                }
            });
    });
}

} // namespace

sycl::event moe_decode_gate_up_variant(
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
    const DecodeMoeVariantOptions& options,
    const sycl::event* dep_event) {
    if (num_active_experts <= 0) return {};
    if (slots == nullptr) return {};
    if (blocksize != 64) {
        // Out-of-contract geometry: delegate to the production instantiations.
        return moe_decode_gate_up_swiglu_grouped(
            ctx, slots, quant_map, x, h, num_active_experts, in_features,
            intermediate_size, blocksize, dep_event, token_ids, true);
    }
    if (options.fixed_scale_256 && options.uniform_scale_loads) {
        return launch_gate_up_variant<true, true>(
            ctx, slots, token_ids, quant_map, x, h, num_active_experts,
            in_features, intermediate_size, dep_event);
    }
    if (options.fixed_scale_256) {
        return launch_gate_up_variant<true, false>(
            ctx, slots, token_ids, quant_map, x, h, num_active_experts,
            in_features, intermediate_size, dep_event);
    }
    if (options.uniform_scale_loads) {
        return launch_gate_up_variant<false, true>(
            ctx, slots, token_ids, quant_map, x, h, num_active_experts,
            in_features, intermediate_size, dep_event);
    }
    return launch_gate_up_variant<false, false>(
        ctx, slots, token_ids, quant_map, x, h, num_active_experts,
        in_features, intermediate_size, dep_event);
}

sycl::event moe_decode_down_rank_variant(
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
    const DecodeMoeVariantOptions& options,
    const sycl::event* dep_event) {
    if (num_active_experts <= 0) return {};
    if (slots == nullptr) return {};
    if (blocksize != 64) {
        return moe_decode_down_to_rank_f32(
            ctx, slots, ranks, h, down_acc, num_active_experts, top_k, in_features,
            out_features, blocksize, token_ids, nullptr, dep_event, true);
    }
    if (options.fixed_scale_256 && options.uniform_scale_loads) {
        return launch_down_rank_variant<true, true>(
            ctx, slots, ranks, token_ids, h, down_acc, num_active_experts, top_k,
            in_features, out_features, dep_event);
    }
    if (options.fixed_scale_256) {
        return launch_down_rank_variant<true, false>(
            ctx, slots, ranks, token_ids, h, down_acc, num_active_experts, top_k,
            in_features, out_features, dep_event);
    }
    if (options.uniform_scale_loads) {
        return launch_down_rank_variant<false, true>(
            ctx, slots, ranks, token_ids, h, down_acc, num_active_experts, top_k,
            in_features, out_features, dep_event);
    }
    return launch_down_rank_variant<false, false>(
        ctx, slots, ranks, token_ids, h, down_acc, num_active_experts, top_k,
        in_features, out_features, dep_event);
}

} // namespace ops
