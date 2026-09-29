#pragma once
#include "GatedDeltaNetOps.hpp"

namespace ops {
// Experimental T16 triangular GDN. Scratch is per execution slot, allocated
// before submission and reused across layers/chunks. State remains FP32.
struct GdnChunkScratch {
    Tensor prepared; // [token, value_head, 3*128+2]: normalized Q/K, V, beta, decay
    Tensor matrix;   // [head, 16+128, 16]: causal QK / decayed transposed K
    Tensor delta;    // [head, 16, 128]: triangular solution, FP32
    Tensor base;     // [head, 16, 128]: prefix-decayed Q*S0, FP32
    Tensor decay;    // [head]: chunk decay product, FP32
    Tensor raw;      // [token, head, 128]: pre-normalization output, FP32
    int capacity = 0, heads = 0;
    void Reserve(Context& ctx, int tokens, int valueHeads);
};

// Returns false before submitting work for unsupported geometry or insufficient
// scratch. xmx=false is an FP32 mathematical control; xmx=true rounds only the
// final A*delta operands to BF16. Gate products, solve, S0 projections and the
// residual state update retain FP32. Caller must drain before releasing scratch.
bool try_linear_delta_prefill_chunked(Context& ctx, const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights, DeltaNetState& state, GdnChunkScratch& scratch,
    const bf16* input, bf16* output, int count, bool xmx);
} // namespace ops
