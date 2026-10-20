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

// Diagnostic A/B variant of the block-native graph attention. It caches the
// selected cache row for each active token in SLM once per tile work-group,
// then reuses it for both QK and PV. The production entry point above remains
// on the original path until real-model timing clears the default-on gate.
void attention_decode_exact_head256_block_native_graph_token_cache(
    Context& ctx, const Tensor& q, const Tensor& k_cache,
    const Tensor& v_cache, Tensor& output, Tensor& partials_buf,
    const Tensor& active_blocks, GraphControl control, int num_heads,
    int num_kv_heads, const Tensor* gate_buf = nullptr, int layer = 0);

} // namespace ops
