#pragma once

// R1 prefill expert XMX path (docs/plans/qwen38-r1-prefill-expert-xmx-plan-2026-10-16.md,
// workstream A): per chunk NF4 -> bf16 dequant, activation gather, batched bf16
// oneDNN matmul (gate_up), SwiGLU, batched bf16 matmul (down), scatter.  The
// kernel structure mirrors the measured probe
// tmp/q4e-bottleneck-audit-20261016/expert_shape_probe2.cpp.
//
// AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32 (int, default 0): non-zero keeps the
// gate_up destination (c1) in f32 instead of bf16, so SwiGLU reads the
// unrounded GEMM output.  The down GEMM's A input (a2) stays bf16 either way.
//
// Ownership: the caller owns the chunk plan (device USM), the input x, the
// down_acc rank arena and every slot descriptor; this file owns the scratch,
// its oneDNN primitives and the oneDNN memory handles that must outlive queued
// executions (src/ops/Bnb4BitLinear.hpp:351-355).

#include "MoeExpertOps.hpp"

#include <sycl/sycl.hpp>

namespace ops {

// Per-chunk plan.  Host builds it; device kernels read it.  All pointers are
// device USM except this struct itself.
//
// member_row_base[b] is the exclusive prefix sum of member b's real activation
// rows in member_token_ids / member_ranks, so member_row_base[B] is that packed
// row count.  Member b's real rows are activation slots
// [member_row_base[b], member_row_base[b + 1]) and they occupy a1 band rows
// [b * rows_per_band, b * rows_per_band + count); the remaining rows up to
// rows_per_band are padding and are zero filled.
struct MoeExpertXmxChunk {
    int num_experts = 0;      // B in [1,16]
    int top_k = 0;            // routing top-k; down_acc row = token * top_k + rank
    int rows_per_band = 0;    // M, padded to a multiple of 16
    const int* member_row_base = nullptr;    // [B+1] exclusive real-row bases
    const int* member_token_ids = nullptr;   // [member_row_base[B]] activation token ids
    const int* member_ranks = nullptr;       // [member_row_base[B]] activation top-k ranks
    const MoeExpertSlotDev* slots = nullptr; // [B] NF4 slot descriptors
};

struct MoeExpertXmxScratch {
    // Two weight banks so chunk k+1's dequant can overlap chunk k's GEMMs.
    void* wgu_bf16[2] = {};   // each 16 * 2 * I * H bf16
    void* wdn_bf16[2] = {};   // each 16 * H * I bf16
    void* a1 = {};            // [max_chunk_rows * H] bf16 (gather)
    void* c1 = {};            // [max_chunk_rows * 2I] bf16 (gate_up dst); f32 when c1_f32
    void* a2 = {};            // [max_chunk_rows * I] bf16 (SwiGLU)
    void* c2 = {};            // [max_chunk_rows * H] f32 (down dst)
    // Device staging for per-chunk plan arrays, in int units: [0, 17) row bases,
    // then max_chunk_rows token ids, then max_chunk_rows ranks.  The pipeline
    // reads the chunk's own pointers, so this is spare room for callers that
    // build the plan on the host.
    void* d_plan = {};
    int max_chunk_rows = 0;   // total padded rows (B * M) per chunk; 4096 per the plan
    // Set at scratch_create from AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32: c1 holds
    // f32 gate_up results and SwiGLU reads f32.  Latched per scratch, so a
    // scratch's numeric mode never changes while work is queued.
    bool c1_f32 = false;

    // Internal: geometry + primitive cache owner.  Do not copy this struct and
    // do not restore it by hand; only scratch_create/scratch_destroy manage it.
    void* impl = nullptr;
    int hidden = 0;
    int intermediate = 0;
};

// Allocates the full flag-on footprint (375 MiB at 4096 rows / H=2560 / I=640,
// +10 MiB more when AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32 is set) through ctx, so
// the device budget and the backend VRAM reserve see it.  The hidden/
// intermediate parameters default to the gated production shape; a scratch must
// be recreated if the model shape changes.  Returns false when the allocation
// budget rejects the request (nothing is left allocated); throws
// std::invalid_argument for non-positive dimensions.
bool moe_expert_xmx_scratch_create(Context& ctx, MoeExpertXmxScratch& s,
                                   int max_chunk_rows = 4096, int hidden = 2560,
                                   int intermediate = 640);

// Drains the queue (oneDNN handles and USM may still be referenced by queued
// work) and releases everything.  Safe to call on a default-constructed or
// already destroyed scratch, and never throws.
void moe_expert_xmx_scratch_destroy(Context& ctx, MoeExpertXmxScratch& s);

// Runs the full per-chunk pipeline on ctx.queue() in order and returns the
// scatter event (chained after every earlier step).  ready = this chunk's H2D
// completion event (nullable).  down_acc rows are (token*top_k + rank).
//
// The internal steps are chained through explicit SYCL events except the
// bf16-input SwiGLU step (c1_f32 off), which is ordered by the in-order queue
// (Context always submits on an in-order queue).  Callers must not submit this
// chunk's pipeline to an out-of-order queue.
sycl::event moe_expert_xmx_prefill_chunk(
    Context& ctx, MoeExpertXmxScratch& scratch, const MoeExpertXmxChunk& chunk, int bank,
    const bf16* x, float* down_acc, int hidden, int intermediate, int blocksize,
    const sycl::event* ready);

}  // namespace ops
