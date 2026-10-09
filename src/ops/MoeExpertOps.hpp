#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

struct MoeNestedScaleDev {
    const uint8_t* codes = nullptr;
    const float* nested_absmax = nullptr;
    const float* nested_quant_map = nullptr;
    float offset = 0.0f;
    int blocksize = 0;
};

struct MoeExpertSlotDev {
    const uint8_t* gate_up_packed = nullptr; // [2 * intermediate_size, in_features / 2]
    const float* gate_up_absmax = nullptr;   // [2 * intermediate_size, in_features / blocksize]
    const uint8_t* down_packed = nullptr;    // [out_features, intermediate_size / 2]
    const float* down_absmax = nullptr;      // [out_features, intermediate_size / blocksize]
    MoeNestedScaleDev gate_up_nested{};
    MoeNestedScaleDev down_nested{};
};

// One expert's contiguous slice of the existing token/rank map. Each tile
// consumes at most 8 rows; padding never reads a token or writes a rank slot.
struct MoeExpertTile {
    int first = 0;
    int rows = 0;
};
static_assert(sizeof(MoeExpertTile) == 2 * sizeof(int), "Expert tile mirror uses two s32 values");

// Experimental DG2 grouped NF4 -> BF16 XMX, FP32 accumulation. A is gathered
// from x[token,H] (gate/up) or h[token,rank,I] (down), W is [N,K] NF4.
// No expanded weights or new activation arena; caller owns maps/tiles/slots
// until the returned final consumer event completes. K must be divisible by 64.
sycl::event moe_grouped_xmx_gate_up(
    Context& ctx, const MoeExpertSlotDev* slots, const MoeExpertTile* tiles,
    int num_tiles, const int* token_ids, const int* ranks, const float* quant_map,
    const bf16* x, bf16* h, int top_k, int hidden, int intermediate,
    bool use_nested, const sycl::event* ready = nullptr);
sycl::event moe_grouped_xmx_down(
    Context& ctx, const MoeExpertSlotDev* slots, const MoeExpertTile* tiles,
    int num_tiles, const int* token_ids, const int* ranks, const float* quant_map,
    const bf16* h, float* down, int top_k, int hidden, int intermediate,
    bool use_nested, const sycl::event* ready = nullptr);

struct MoeNestedScaleHost {
    const uint8_t* codes = nullptr;
    const float* nested_absmax = nullptr;
    const float* nested_quant_map = nullptr;
    float offset = 0.0f;
    int blocksize = 0;
};

struct MoeExpertSlotHost {
    const uint8_t* gate_up_packed = nullptr; // [2 * intermediate_size, in_features / 2]
    const float* gate_up_absmax = nullptr;   // [2 * intermediate_size, in_features / blocksize] (unnested)
    const uint8_t* down_packed = nullptr;    // [out_features, intermediate_size / 2]
    const float* down_absmax = nullptr;      // [out_features, intermediate_size / blocksize] (unnested)
    MoeNestedScaleHost gate_up_nested{};
    MoeNestedScaleHost down_nested{};
};

// 512-expert Top-K Router (Device)
// Each workgroup computes one token, WG size = 512
sycl::event router_softmax_topk_device(
    Context& ctx,
    const bf16* router_scores_ptr,
    const bf16* per_expert_scale_ptr,
    int* expert_ids_dev,
    float* expert_weights_dev,
    int tokens,
    int num_experts = 512,
    int top_k = 10,
    bool norm_topk_prob = true);

// Single-token resident router: identical top-k arithmetic, then device-table
// classification and pinned-host mirrors in the same kernel's final writer.
// Wait for the returned event before reading/reusing any host output.
sycl::event router_softmax_topk_resident(
    Context& ctx,
    const bf16* router_scores_ptr,
    const bf16* per_expert_scale_ptr,
    int* expert_ids_dev,
    float* expert_weights_dev,
    const int* resident_table,
    int* resident_selected,
    int* host_ids,
    float* host_weights,
    int* host_selected,
    int num_experts = 512,
    int top_k = 10,
    bool norm_topk_prob = true);

// 512-expert Top-K Router (Host CPU Reference)
void router_softmax_topk_host(
    const bf16* router_scores_ptr,
    const bf16* per_expert_scale_ptr,
    int* expert_ids,
    float* expert_weights,
    int tokens,
    int num_experts = 512,
    int top_k = 10,
    bool norm_topk_prob = true);

// Lightweight 512-expert Top-K Router (Device, IDs only)
// Skips softmax exp, sum reductions, and weights normalization.
sycl::event router_topk_ids_device(
    Context& ctx,
    const bf16* router_scores_ptr,
    int* expert_ids_dev,
    int tokens,
    int num_experts = 512,
    int top_k = 10);

// Lightweight 512-expert Top-K Router (Host CPU Reference, IDs only)
void router_topk_ids_host(
    const bf16* router_scores_ptr,
    int* expert_ids,
    int tokens,
    int num_experts = 512,
    int top_k = 10);

// Slotted expert decode gate_up GEMV with SwiGLU: h[k, intermediate_size] = silu(gate) * up
// H2: GPU classification plus fixed-rank, masked resident NF4 projections.
sycl::event moe_resident_classify(Context& ctx, const int* ids, const float* weights,
    const int* table, int* selected, int* host_ids, float* host_weights, int* host_selected,
    int top_k, int experts);
// The nested_fixed256 flags below are a caller contract: when nested scales
// are enabled, every referenced descriptor must use nested blocksize 256.
sycl::event moe_resident_gate_up(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const bf16* x, bf16* h, int top_k, int hidden, int intermediate, bool nested,
    bool nested_fixed256 = false);
sycl::event moe_resident_down(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const bf16* h, float* down, int top_k, int hidden, int intermediate, bool nested,
    bool nested_fixed256 = false);

// Experimental resident NF4/Q8 decode. Packed NF4 descriptors retain their
// original row-major format and block-64 nested scales. Only activations and
// the default NF4 codebook are approximated with signed Q8/RTE. Input buffers
// are xq[hidden], xs[hidden/64]; expert buffers are h/hq[top_k,intermediate]
// and hs[top_k,intermediate/64]. Quantization observes the BF16 h boundary.
// Nonfinite values quantize to zero and do not contribute to a block maximum;
// zero blocks have scale zero. Negative selected slots leave expert buffers
// untouched. All arenas and descriptors must live until the last event drains.
// Caller provides dependencies through the engine's in-order compute queue.
sycl::event quantize_moe_input_q8_64(Context& ctx, const bf16* x,
    int8_t* xq, float* xs, int hidden);
sycl::event moe_resident_gate_up_q8_64(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const int8_t* xq, const float* xs, bf16* h,
    int8_t* hq, float* hs, int top_k, int hidden, int intermediate, bool nested,
    bool nested_fixed256 = false);
sycl::event moe_resident_down_q8_64(Context& ctx, const MoeExpertSlotDev* descriptors,
    const int* selected, const int8_t* hq, const float* hs, float* down,
    int top_k, int hidden, int intermediate, bool nested,
    bool nested_fixed256 = false);

// Slotted expert decode gate_up GEMV with SwiGLU: h[k, intermediate_size] = silu(gate) * up
// in_features = 2560, intermediate_size = 640
sycl::event moe_decode_gate_up_swiglu_grouped(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int num_active_experts,
    int in_features = 2560,
    int intermediate_size = 640,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    const int* token_ids = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

inline sycl::event moe_decode_gate_up_swiglu_grouped(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int num_active_experts,
    int in_features,
    int intermediate_size,
    int blocksize,
    const sycl::event& dep_event) {
    return moe_decode_gate_up_swiglu_grouped(
        ctx, slots, quant_map, x, h, num_active_experts,
        in_features, intermediate_size, blocksize, &dep_event);
}

inline sycl::event moe_decode_gate_up_swiglu_grouped(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int num_active_experts,
    const sycl::event& dep_event,
    int in_features = 2560,
    int intermediate_size = 640,
    int blocksize = 64) {
    return moe_decode_gate_up_swiglu_grouped(
        ctx, slots, quant_map, x, h, num_active_experts,
        in_features, intermediate_size, blocksize, &dep_event);
}

// Slotted expert decode down-projection with weighted accumulation:
// out = sum_k (weight[k] * down_k(h[k]))
sycl::event moe_decode_down_weighted_accumulate(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const float* weights,
    const bf16* h,
    float* acc_f32,
    bf16* out_bf16,
    int num_active_experts,
    int in_features = 640,
    int out_features = 2560,
    int blocksize = 64,
    const bf16* shared_out = nullptr,
    const bf16* shared_gate = nullptr,
    bool nested_fixed256 = false);

// Slotted expert decode down-projection writing directly to rank slots of down_acc (unweighted FP32):
// down_acc[(token * top_k + rank) * out_features + row] = down_k(h[k])_row
sycl::event moe_decode_down_to_rank_f32(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* ranks,
    const bf16* h,
    float* down_acc,
    int num_active_experts,
    int top_k = 10,
    int in_features = 640,
    int out_features = 2560,
    int blocksize = 64,
    const int* token_ids = nullptr,
    const float* quant_map = nullptr,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

// Batched prefill expert gate_up GEMV with SwiGLU: h[t, k, intermediate_size] = silu(gate) * up
sycl::event moe_prefill_gate_up_rows(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int tokens,
    int num_active_experts,
    int in_features = 2560,
    int intermediate_size = 640,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

inline sycl::event moe_prefill_gate_up_rows(
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
    const sycl::event& dep_event,
    bool use_nested = false) {
    return moe_prefill_gate_up_rows(
        ctx, slots, quant_map, x, h, tokens, num_active_experts,
        in_features, intermediate_size, blocksize, &dep_event, use_nested);
}

// Batched prefill expert down-projection with weighted accumulation:
// y[t] = sum_k (weight[t, k] * down_k(h[t, k]))
sycl::event moe_prefill_down_rows(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const float* weights,
    const float* quant_map,
    const bf16* h,
    bf16* y,
    int tokens,
    int num_active_experts,
    int in_features = 640,
    int out_features = 2560,
    int blocksize = 64,
    const bf16* shared_out = nullptr,
    const bf16* shared_gate = nullptr,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

inline sycl::event moe_prefill_down_rows(
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
    const sycl::event& dep_event,
    bool use_nested = false) {
    return moe_prefill_down_rows(
        ctx, slots, weights, quant_map, h, y, tokens, num_active_experts,
        in_features, out_features, blocksize, shared_out, shared_gate, &dep_event, use_nested);
}

// Flat SwiGLU activation: out[t * intermediate_size + i] = silu(gate) * up
void moe_silu_gate_up_mul_flat(
    Context& ctx,
    const bf16* gate_up,
    bf16* out,
    int total_tokens,
    int intermediate_size);

// Phase S4, W4 & G3: Expert-Major Prefill Gate/Up and Down
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
    int in_features = 2560,
    int intermediate_size = 640,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    int num_pairs = 0,
    int num_quads = 0,
    bool nested_fixed256 = false);

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
    int in_features = 640,
    int out_features = 2560,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    int num_pairs = 0,
    int num_quads = 0,
    bool compact_output = false,
    bool nested_fixed256 = false);

// Compact output is [acts,H]; input h still uses [token,rank,I]. H must be divisible by 4.
void moe_build_inverse_map(Context& ctx, const int* tokens, const int* ranks,
    int* inverse, int acts, int top_k);
sycl::event moe_merge_compact_down(Context& ctx, const float* down, const int* tokens,
    const int* inverse, const float* weights, int64_t* sum, int* invalid, int first, int acts,
    int top_k, int H, const sycl::event* dep = nullptr);
sycl::event moe_finish_compact_down(Context& ctx, const int64_t* sum,
    sycl::ext::oneapi::bfloat16* y, int tokens, int H,
    const sycl::ext::oneapi::bfloat16* shared_out = nullptr,
    const sycl::ext::oneapi::bfloat16* shared_gate = nullptr);

// Phase W4 (C1): Explicit 2-token weight reuse prefill kernels
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
    int in_features = 2560,
    int intermediate_size = 640,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

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
    int in_features = 640,
    int out_features = 2560,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

// Phase G3: Explicit 4-token weight reuse prefill kernels
sycl::event moe_expert_major_prefill_gate_up_4token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* x,
    bf16* h,
    int total_quads,
    int top_k,
    int in_features = 2560,
    int intermediate_size = 640,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

sycl::event moe_expert_major_prefill_down_4token(
    Context& ctx,
    const MoeExpertSlotDev* slots,
    const int* token_ids,
    const int* ranks,
    const float* quant_map,
    const bf16* h,
    float* down_acc,
    int total_quads,
    int top_k,
    int in_features = 640,
    int out_features = 2560,
    int blocksize = 64,
    const sycl::event* dep_event = nullptr,
    bool use_nested = false,
    bool nested_fixed256 = false);

// Single-token CPU output in this queue's USM host context. The caller keeps
// values immutable until the returned merge event completes. The rank map is
// copied into the kernel arguments, so it needs no device allocation or fence.
struct MoeCpuHostDownRows {
    const float* values = nullptr;
    int rows = 0;
    std::array<int, 16> rank_to_row{};
};

sycl::event moe_ordered_accumulate_down_rows(
    Context& ctx,
    const float* down_acc,
    const float* weights,
    bf16* y,
    int tokens,
    int top_k,
    int out_features = 2560,
    const bf16* shared_out = nullptr,
    const bf16* shared_gate = nullptr,
    const sycl::event* dep_event = nullptr,
    const void* cpu_result_packet = nullptr);

// Separate kernel/interface: the default device/packet kernel captures no host
// pointer or rank map. Host rows and result packets cannot be combined here.
sycl::event moe_ordered_accumulate_cpu_host_rows(
    Context& ctx, const float* down_acc, const float* weights, bf16* y,
    int tokens, int top_k, int out_features, const MoeCpuHostDownRows& cpu_host_rows,
    const bf16* shared_out = nullptr, const bf16* shared_gate = nullptr,
    const sycl::event* dep_event = nullptr);

} // namespace ops
