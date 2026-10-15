#pragma once

#include "GatedDeltaNetOps.hpp"

namespace ops {

// Rows of the fp32 state ([v_heads, 128, 128], row-major) that the SLM-warm
// variant may stage per work-group. 120 rows * 128 cols * 4 B = 61440 B and
// the q/k/out temporaries need another 1544 B, so the total stays inside the
// 64 KiB per-work-group SLM cap with 2552 B of headroom (A770 reference).
constexpr int kGdnSlmRowsMax = 120;

// True when the SLM-warm variant supports the geometry (same GQA fastpath
// predicate as the production kernel, GatedDeltaNetOps.cpp:6-30).
bool supports_gdn_prefill_slm(const DeltaNetGeometry& geom);

// SLM-warm-row variant of run_linear_delta_prefill_gpu_batched.
//
// Rows [0, slm_rows) of every head's state are staged into work-group local
// memory for the whole window; rows [slm_rows, 128) stay in global USM. Rows
// are private to a lane (row j of column dv is read and written by lane dv
// only), so the per-lane arithmetic, the four-row iteration order and every
// barrier/reduction are identical to the production kernel: the output is
// expected to be bit-identical, and the bench asserts it (parity guard) with
// a tolerance fallback report.
//
// slm_rows is clamped to [0, kGdnSlmRowsMax] and rounded down to a multiple
// of 4 (the state walk visits four rows per iteration; a partial group would
// straddle the SLM/global boundary). Returns false when the geometry is not
// eligible or seq_len <= 0; in that case the caller must fall back to
// run_linear_delta_prefill_gpu_batched.
bool run_linear_delta_prefill_gpu_batched_slm(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* fused_ptr,
    bf16* out_ptr,
    int seq_len,
    int slm_rows);

} // namespace ops
