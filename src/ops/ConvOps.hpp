#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"

// Simple SYCL conv2d + GELU for audio encoder frontend.
// kernel=3, stride=2, pad=1 (same padding for odd input).

namespace ops {

// input:  [N, in_ch, in_h, in_w] bf16
// weight: [out_ch, in_ch, 3, 3] bf16 (OIHW)
// bias:   [out_ch] f32 or bf16
// output: [N, out_ch, out_h, out_w] bf16
// out_h = (in_h + 2*1 - 3) / 2 + 1 = ceil(in_h/2)
// out_w = (in_w + 2*1 - 3) / 2 + 1 = ceil(in_w/2)
void conv2d_gelu(Context& ctx,
                 Tensor& input, Tensor& weight, Tensor& bias,
                 Tensor& output,
                 int batch, int in_ch, int out_ch,
                 int in_h, int in_w, int out_h, int out_w);

// 1D Causal Convolution with dilation
// input:   [batch, in_ch, seq_len]
// weight:  [out_ch, in_ch, kernel_size]
// bias:    [out_ch]
// output:  [batch, out_ch, seq_len]
void causal_conv1d(Context& ctx,
                   Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                   int batch, int in_ch, int out_ch, int seq_len,
                   int kernel_size, int dilation);

// 1D Depthwise Causal Convolution with dilation
// input:   [batch, channels, seq_len]
// weight:  [channels, 1, kernel_size]
// bias:    [channels]
// output:  [batch, channels, seq_len]
void causal_conv1d_dw(Context& ctx,
                      Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                      int batch, int channels, int seq_len,
                      int kernel_size, int dilation);

// 1D Causal Transposed Convolution (Upsampling + Causal slicing)
// input:   [batch, in_ch, in_len]
// weight:  [in_ch, out_ch, kernel_size] (groups = 1)
// bias:    [out_ch]
// output:  [batch, out_ch, in_len * stride] (right_pad elements are skipped in computation)
void causal_conv_transpose1d(Context& ctx,
                             Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                             int batch, int in_ch, int out_ch, int in_len,
                             int kernel_size, int stride);

// 1D Convolution with reflect padding (symmetric on both sides, no causality)
// Used by ECAPA-TDNN speaker encoder.
// input:   [batch, seq_len, in_ch]  (row-major)
// weight:  [out_ch, in_ch, kernel_size]  (row-major)
// bias:    [out_ch]
// output:  [batch, seq_len, out_ch]  (row-major)
// pad: number of reflect-padded elements on each side
// dilation: dilation factor
// relu: if true, applies ReLU activation after conv
void reflect_conv1d(Context& ctx,
                    Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                    int batch, int in_ch, int out_ch, int seq_len,
                    int kernel_size, int dilation, int pad,
                    bool relu);

// In-place element-wise ReLU activation
// n: total number of elements
void relu_inplace(Context& ctx, Tensor& x, int n);

// In-place element-wise Sigmoid activation
void sigmoid_inplace(Context& ctx, Tensor& x, int n);

// In-place element-wise Tanh activation
void tanh_inplace(Context& ctx, Tensor& x, int n);

// Global average pooling over time dimension
// input:  [batch, seq_len, channels]
// output: [batch, 1, channels]
void global_avg_pool_1d(Context& ctx, Tensor& input, Tensor& output,
                        int batch, int seq_len, int channels);

// Softmax over time dimension
// data: [batch, channels, seq_len] — softmax applied over last dim (seq_len)
void softmax_1d_time(Context& ctx, Tensor& data, int batch, int channels, int seq_len);

// Channel-wise scale + add (used for SE block gating)
// dst[c, t] += alpha * src[t] (src is channel-wise scalar broadcast)
void channel_mul_add_inplace(Context& ctx, Tensor& src, Tensor& dst,
                             int channels, int seq_len);

} // namespace ops
