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

// 512-expert Top-K Router (Device)
// Each workgroup computes one token, WG size = 512
void router_softmax_topk_device(
    Context& ctx,
    const bf16* router_scores_ptr,
    const bf16* per_expert_scale_ptr,
    int* expert_ids_dev,
    float* expert_weights_dev,
    int tokens,
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
    int blocksize = 64);

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
    const bf16* shared_gate = nullptr);

// Flat SwiGLU activation: out[t * intermediate_size + i] = silu(gate) * up
void moe_silu_gate_up_mul_flat(
    Context& ctx,
    const bf16* gate_up,
    bf16* out,
    int total_tokens,
    int intermediate_size);

} // namespace ops
