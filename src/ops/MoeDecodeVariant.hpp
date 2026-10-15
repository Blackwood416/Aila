#pragma once

#include "MoeExpertOps.hpp"

namespace ops {

// R1 decode-MoE kernel variants (2026-10-15, research only; no production
// caller). Both knobs are exact-math rewrites of the nested-scale decode
// instantiations in MoeExpertOps.cpp; they only change how the per-block
// nested absmax is reconstructed:
//
//  - fixed_scale_256: every decode descriptor is built by
//    Qwen4ExpAdaptiveCache::make_slot (Qwen4ExpAdaptiveCache.cpp:312-332),
//    which throws unless the nested blocksize is 256. The two divisions of
//    the generic instantiation are then exact powers of two: `>> 8` for the
//    nested block and `>> 5` for `byte_offset / (blocksize / 2)` at the
//    validated NF4 blocksize 64. The shipped adaptive decode call site
//    (Qwen4ExpBnb4Backend.cpp:1187,1315-1327) already passes
//    nested_fixed256 = true; the variant exists as the A/B control for that
//    specialization and as insurance for call sites that omit the flag
//    (Qwen4ExpExpertExecutor.cpp:1700-1707 / :1174-1177, currently uncalled).
//    Caller contract mirrors MoeExpertOps.hpp:138-139.
//
//  - uniform_scale_loads: every lane of the half-subgroup evaluates the
//    nested absmax chain for its own (uniform) block index instead of the
//    leader-lane load plus four `group_broadcast` calls per iteration. The
//    reconstructed float is identical, so the FMA order is unchanged.
//
// The gate/up output is BF16 and the down output FP32, in both cases with the
// exact same accumulation order as launch_moe_decode_gate_up_swiglu_grouped_dev
// and launch_moe_decode_down_to_rank_f32_dev; a bitwise parity check against
// the production entry points is expected to pass (bench asserts it).
struct DecodeMoeVariantOptions {
    bool fixed_scale_256 = true;
    bool uniform_scale_loads = false;
};

// Device-pointer form only (const MoeExpertSlotDev* in USM device memory),
// matching the production >16-expert dispatch and the bench harness. slot
// descriptors must carry nested metadata with blocksize 256 (compact nested
// codes); token_ids may be null (every expert reads row 0).
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
    const sycl::event* dep_event = nullptr);

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
    const sycl::event* dep_event = nullptr);

} // namespace ops
