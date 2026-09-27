#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// Grouped RMSNorm: independent RMSNorm per stream g across group_dim dims
// x: [rows, groups * group_dim]
// w: [groups * group_dim] (optional float, or nullptr for unweighted)
// out: [rows, groups * group_dim]
// plus_one: if true, weight scaling is (1.0 + w[i]), otherwise w[i]
void grouped_rms_norm(
    Context& ctx, const bf16* x, const float* w, float eps, bf16* out,
    int rows, int groups, int group_dim, bool plus_one = false);

void grouped_rms_norm(
    Context& ctx, Tensor& x, Tensor& w, float eps, Tensor& out,
    int rows, int groups, int group_dim, bool plus_one = false);

// Hyper-connection mix reduce: out = (1 / groups) * sum_g (mix * N)
// N: [rows, groups * group_dim]
// mix: [rows, groups * group_dim]
// out: [rows, group_dim]
// apply_sigmoid_to_mix: if true, computes sigmoid on mix in-register before multiplying with N
void hc_mix_reduce(
    Context& ctx, const bf16* N, const bf16* mix, bf16* out,
    int rows, int group_dim, int groups, bool apply_sigmoid_to_mix = false);

// Hyper-connection residual injection add: H[r, g * group_dim + d] += inj[r, g] * y[r, d]
// H: [rows, groups * group_dim] (in-out)
// y: [rows, group_dim]
// inj: [rows, groups] (float or bf16)
void hc_inject_add(
    Context& ctx, bf16* H, const bf16* y, const float* inj,
    int rows, int group_dim, int groups);

void hc_inject_add(
    Context& ctx, bf16* H, const bf16* y, const bf16* inj,
    int rows, int group_dim, int groups);

// Fused residual injection add and grouped RMSNorm (P1 intra-layer fusion):
// Materializes H[r, g * group_dim + d] = bf16(H[r, g * group_dim + d] + inj[r, g] * y[r, d])
// and immediately computes grouped RMSNorm into out_N using exact same rounded values.
void hc_inject_add_and_grouped_rms_norm(
    Context& ctx,
    bf16* H,
    const bf16* y,
    const float* inj,
    const float* norm_w,
    float eps,
    bf16* out_N,
    int rows,
    int groups,
    int group_dim,
    bool plus_one = true);

// Single-row (decode) hyper-connection, stage A. One work-group per output row:
//   act[r] = silu(bf16(W_down[r] . N) / groups)          r < lowrank
//   inj[g] = 2 * sigmoid(bf16(W_inject[g] . N) / groups)  g < groups (skipped if W_inject null)
// N: [groups * group_dim] bf16; W_down: [lowrank, groups * group_dim]; W_inject: [groups, groups * group_dim]
void hc_decode_down_inject(
    Context& ctx, const bf16* N, const bf16* w_down, const bf16* w_inject,
    bf16* act, float* inj, int lowrank, int group_dim, int groups);

// Single-row (decode) hyper-connection, stage B. One sub-group per output dim d:
//   x[d] = mean_g( sigmoid(bf16(W_up[g * group_dim + d] . act)) * N[g * group_dim + d] )
// W_up: [groups * group_dim, lowrank]; act: [lowrank]; x: [group_dim]
void hc_decode_up_mix(
    Context& ctx, const bf16* act, const bf16* w_up, const bf16* N,
    bf16* x, int lowrank, int group_dim, int groups);

// PLE Gate-Value computation:
// g[s] = sum_d (K[s, d] * Q[s, d]) / sqrt(hidden_size)
// g = sign(g) * sqrt(max(|g|, 1e-6))
// gate = sigmoid(g)
// GV[s, d] = gate * V[d]
// K: [rows, hc_count * hidden_size]
// Q: [rows, hc_count * hidden_size]
// V: [rows, hidden_size]
// GV: [rows, hc_count * hidden_size]
void ple_gate_value(
    Context& ctx, const bf16* K, const bf16* Q, const bf16* V, bf16* GV,
    int rows, int hc_count = 4, int hidden_size = 2560);

// PLE Dilated Conv1D + SiLU + Residual Add:
// Dilated depthwise conv1d with kernel_size=4, dilation=3 (9 rows conv state)
// conv_out = silu(conv1d_dilated3(gv_normed))
// out = gv + conv_out
// gv_normed: [rows, channels]
// gv: [rows, channels]
// conv_state: [9, channels] float ring buffer
// conv_head: in-out ring buffer head index (0..8)
// conv_w: [channels, 4] bf16 depthwise weights
// out: [rows, channels]
void ple_dilated_conv_silu_add(
    Context& ctx, const bf16* gv_normed, const bf16* gv,
    float* conv_state, int& conv_head, const bf16* conv_w, bf16* out,
    int rows, int channels = 10240, int kernel_size = 4, int dilation = 3,
    bf16* residual = nullptr);

} // namespace ops
