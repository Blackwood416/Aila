#pragma once

// E8 decode host-amortization prototypes (2026-10-15).
//
// Goal: remove one kernel submission per hyper-connection call without
// changing any arithmetic.  Production submits, per HC call (count == 1):
//
//   1) inject-add + grouped RMSNorm  (try_hc_decode_inject_add_and_grouped_rms_norm
//                                     or hc_inject_add_and_grouped_rms_norm)
//   2) hc_decode_down_inject         (act / inj rows)
//   3) hc_decode_up_mix              (x = mean_g sigmoid(...) * N)
//
// A direct down+up merge is impossible without a device-scope barrier
// (the up stage consumes all lowrank `act` values produced by the down
// stage), so this TU instead fuses (1)+(2) into a single kernel and moves the
// in-place "H += inj * y" update into the (3) variant.  Every one of the
// 97 HC calls per decode token then costs 2 submissions instead of 3.
//
// Both functions are bit-exact by construction: every arithmetic expression,
// reduction order, BF16 rounding point and FMA order is copied verbatim from
// src/ops/HyperConnectionOps.cpp (grouped_rms_norm / try_hc_decode_inject_add_
// and_grouped_rms_norm / hc_decode_down_inject / hc_decode_up_mix).
//
// Wired behind AILA_Q4E_DECODE_FUSED_HC=1 (default off; flag-off and any
// rejected shape fall back to the production ops byte-identically).  The
// call site is Qwen4ExpBnb4Backend::RunHyperConnectionCombined; see
// docs/plans/qwen38-r1-decode-fusion-2026-10-15.md sections 3.1/3.4 for the
// wiring and the validation arms.

#include "core/Context.hpp"

#include <sycl/sycl.hpp>

namespace q4e_fusion {

using bf16 = sycl::ext::oneapi::bfloat16;

// Fused (inject-add + grouped RMSNorm) + hc_decode_down_inject, single submit.
//
//   H'[g * group_dim + d] = bf16(H[g * group_dim + d] + inj[g] * y[d])
//                           (identity when y == nullptr || inj == nullptr)
//   N[g * group_dim + d]  = bf16(H'[g * group_dim + d] * inv_rms[g] * (1 + w))
//   act[r]  = silu(bf16(W_down[r] . N) / groups)            r < lowrank
//   inj_out[g] = 2 * sigmoid(bf16(W_inject[g] . N) / groups)  g < groups
//
// IMPORTANT CONTRACT: H is read only.  The in-place update of the caller's
// residual stack is performed by run_hc_up_mix_with_hidden_update below
// (submitted next), which keeps this kernel race-free: every work-group reads
// the original H snapshot while no work-group ever writes H.
//
// Shapes: groups == 4, group_dim % 8 == 0, (groups * group_dim) % 8 == 0,
// k_dim * sizeof(bf16) + groups * 256 * sizeof(float) <= 48 KiB of SLM,
// lowrank >= 1, H / y / n_out / norm_w 16-byte aligned when y != nullptr.
// Returns false and submits nothing when the shape is unsupported (caller
// must fall back to the production kernel pair).
bool run_hc_norm_down_fused(
    Context& ctx,
    const bf16* H,           // [groups * group_dim] residual stack (read only)
    const bf16* y,           // [group_dim], or null for a plain RMSNorm
    const float* inj,        // [groups], or null
    const float* norm_w,     // [groups * group_dim], plus_one semantics
    float eps,
    const bf16* w_down,      // [lowrank, groups * group_dim]
    const bf16* w_inject,    // [groups, groups * group_dim], or null
    bf16* n_out,             // [groups * group_dim]
    bf16* act_out,           // [lowrank]
    float* inj_out,          // [groups], written only when w_inject != null
    int lowrank,
    int group_dim,
    int groups);

// hc_decode_up_mix (count == 1) plus the relocated in-place residual update:
//
//   x[d] = mean_g( sigmoid(bf16(W_up[g * group_dim + d] . act)) * N[g * group_dim + d] )
//   H[g * group_dim + d] = bf16(H[g * group_dim + d] + inj[g] * y[d])
//                          (only when H != nullptr && y != nullptr && inj != nullptr)
//
// Each element of H is read and written by exactly one sub-group, so the
// relocation is race-free as long as nothing else writes H between the fused
// kernel above and this one.
void run_hc_up_mix_with_hidden_update(
    Context& ctx,
    const bf16* act,         // [lowrank]
    const bf16* w_up,        // [groups * group_dim, lowrank]
    const bf16* N,           // [groups * group_dim]
    bf16* x,                 // [group_dim]
    bf16* H,                 // [groups * group_dim], may be null (no update)
    const bf16* y,           // [group_dim], may be null
    const float* inj,        // [groups], may be null
    int lowrank,
    int group_dim,
    int groups);

} // namespace q4e_fusion
