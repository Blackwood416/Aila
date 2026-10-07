#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

enum class GateActivation {
    Silu,
    Sigmoid,
};

struct DeltaNetGeometry {
    int q_heads = 0;
    int v_heads = 0;
    int head_k_dim = 128;
    int head_v_dim = 128;
    int kernel = 4;
    int conv_rows = 3;
    int q_dim = 0;
    int kv_dim = 0;
    int qkv_dim = 0;
    int z_dim = 0;
    int all_dim = 0;
    float eps = 1e-6f;
    float scale = 0.0f;
    GateActivation gate_act = GateActivation::Silu;

    static DeltaNetGeometry create(
        int q_heads, int v_heads, int head_k_dim, int head_v_dim,
        int kernel, float eps, GateActivation gate_act = GateActivation::Silu) {
        DeltaNetGeometry g;
        g.q_heads = std::max(1, q_heads);
        g.v_heads = v_heads;
        g.head_k_dim = head_k_dim;
        g.head_v_dim = head_v_dim;
        g.kernel = kernel;
        g.conv_rows = std::max(0, kernel - 1);
        g.q_dim = g.q_heads * head_k_dim;
        g.kv_dim = v_heads * head_v_dim;
        g.qkv_dim = 2 * g.q_dim + g.kv_dim;
        g.z_dim = g.kv_dim;
        g.all_dim = g.qkv_dim + g.z_dim + 2 * v_heads;
        g.eps = eps;
        g.scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
        g.gate_act = gate_act;
        return g;
    }
};

struct DeltaNetWeights {
    const bf16* conv_weight = nullptr;  // [qkv_dim * kernel]
    const float* norm_weight = nullptr; // [head_v_dim]
    const float* a_log = nullptr;       // [v_heads]
    const bf16* dt_bias = nullptr;      // [v_heads]
};

struct DeltaNetState {
    float* state = nullptr;             // [v_heads, head_k_dim, head_v_dim]
    float* conv_state = nullptr;        // [conv_rows, qkv_dim]
    int conv_head = 0;
    bool device_state_dirty = false;
};

bool supports_linear_delta_gqa_fastpath(int q_heads, int kv_heads,
                                        int head_k_dim, int head_v_dim,
                                        int kernel, int conv_rows,
                                        bool allow_grouped_fastpath = false);

bool supports_default_grouped_linear_delta_fastpath(int q_heads, int kv_heads,
                                                    int head_k_dim, int head_v_dim,
                                                    int kernel, int conv_rows);

void commit_linear_conv_history(Context& ctx, const bf16* input, int input_stride,
                                float* history, int channels, int history_rows,
                                int initial_head, int token_count);

int linear_delta_gqa_group_size(int q_heads, int kv_heads);
int linear_delta_key_head_for_value_head(int value_head, int q_heads, int kv_heads);
bool linear_delta_is_group_leader(int value_head, int q_heads, int kv_heads);

void run_linear_delta_decode_gpu(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* qkv_ptr,
    const bf16* z_ptr,
    const bf16* a_ptr,
    const bf16* b_ptr,
    bf16* out_ptr,
    // Only grouped fastpath may defer; caller must commit the old ring row
    // after this kernel completes, before a subsequent decode/checkpoint.
    bool defer_grouped_history = false);

void run_linear_delta_prefill_gpu_batched(
    Context& ctx,
    const DeltaNetGeometry& geom,
    const DeltaNetWeights& weights,
    DeltaNetState& state,
    const bf16* fused_ptr,
    bf16* out_ptr,
    int seq_len,
    // Small verifier reference windows may reuse the actual decode kernels.
    bool decode_order = false);

} // namespace ops
