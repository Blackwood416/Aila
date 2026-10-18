#pragma once

#include "Ops.hpp"

namespace ops {

// Graph-safe sparse-QSA selector validation. This node must run immediately
// before attention_decode_exact_head256_block_native_graph: it validates every
// active block, tail token and cache row selected by GraphControl, and marks
// sticky abort on failure without modifying attention output or cache state.
void attention_validate_qsa_active_blocks_graph(
    Context& ctx, const Tensor& k_cache, const Tensor& v_cache,
    const Tensor& active_blocks, GraphControl control, int num_kv_heads,
    int layer = 0);

} // namespace ops
