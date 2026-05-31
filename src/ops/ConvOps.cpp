#include "ConvOps.hpp"
#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

void conv2d_gelu(Context& ctx,
                 Tensor& input, Tensor& weight, Tensor& bias,
                 Tensor& output,
                 int batch, int in_ch, int out_ch,
                 int in_h, int in_w, int out_h, int out_w) {
    auto* in = input.data_as<bf16>();
    auto* w = weight.data_as<bf16>();
    auto* out = output.data_as<bf16>();
    // bias may be f32 or bf16; read as float for precision
    auto* b = static_cast<const float*>(bias.data());
    bool bias_is_f32 = (bias.dtype() == dnnl::memory::data_type::f32);

    // Total output elements
    int total_out = batch * out_ch * out_h * out_w;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total_out), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            if (i >= total_out) return;

            // Decompose flat index: n, oc, oh, ow
            int ow = i % out_w;
            int tmp = i / out_w;
            int oh = tmp % out_h;
            tmp /= out_h;
            int oc = tmp % out_ch;
            int n = tmp / out_ch;

            float sum = 0.0f;

            // 3x3 kernel
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int ih = oh * 2 + kh - 1;  // stride=2, pad=1
                        int iw = ow * 2 + kw - 1;

                        // Zero-padding: skip out-of-bounds
                        if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w)
                            continue;

                        // input[n, ic, ih, iw]
                        int in_idx = ((n * in_ch + ic) * in_h + ih) * in_w + iw;
                        // weight[oc, ic, kh, kw] OIHW
                        int w_idx = (((oc * in_ch + ic) * 3) + kh) * 3 + kw;

                        sum += static_cast<float>(in[in_idx]) *
                               static_cast<float>(w[w_idx]);
                    }
                }
            }

            // Add bias
            float bias_val = bias_is_f32 ? b[oc] : static_cast<float>(
                reinterpret_cast<const bf16*>(b)[oc]);
            sum += bias_val;

            // GELU tanh approximation: same as dnnl::eltwise_gelu_tanh
            float x = sum;
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/pi)
            float tanh_val = sycl::tanh(inner);
            float gelu_val = 0.5f * x * (1.0f + tanh_val);

            out[i] = bf16(gelu_val);
        });
    });
}

// 1D Causal Convolution with dilation (row-major: [batch, seq_len, channels])
void causal_conv1d(Context& ctx,
                   Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                   int batch, int in_ch, int out_ch, int seq_len,
                   int kernel_size, int dilation) {
    auto* in_ptr = input.data_as<bf16>();
    auto* w_ptr = weight.data_as<bf16>();
    auto* out_ptr = output.data_as<bf16>();
    const void* b_ptr = bias.data();
    bool bias_is_f32 = (bias.dtype() == dnnl::memory::data_type::f32);

    int total_out = batch * seq_len * out_ch;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total_out), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            if (i >= total_out) return;

            // Decompose index in row-major layout [batch, seq_len, out_ch]
            int oc = i % out_ch;
            int tmp = i / out_ch;
            int t = tmp % seq_len;
            int n = tmp / seq_len;

            float sum = 0.0f;

            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    // Causal index formula:
                    int ih = t - (kernel_size - 1 - k) * dilation;
                    if (ih < 0 || ih >= seq_len) continue;

                    // input row-major layout: [batch, seq_len, in_ch]
                    int in_idx = (n * seq_len + ih) * in_ch + ic;
                    // weight row-major layout: [out_ch, in_ch, kernel_size]
                    int w_idx = (oc * in_ch + ic) * kernel_size + k;

                    sum += static_cast<float>(in_ptr[in_idx]) * static_cast<float>(w_ptr[w_idx]);
                }
            }

            float bias_val = bias_is_f32 ? static_cast<const float*>(b_ptr)[oc] : static_cast<float>(
                static_cast<const bf16*>(b_ptr)[oc]);
            sum += bias_val;

            out_ptr[i] = bf16(sum);
        });
    });
}

// 1D Depthwise Causal Convolution with dilation (row-major: [batch, seq_len, channels])
void causal_conv1d_dw(Context& ctx,
                      Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                      int batch, int channels, int seq_len,
                      int kernel_size, int dilation) {
    auto* in_ptr = input.data_as<bf16>();
    auto* w_ptr = weight.data_as<bf16>();
    auto* out_ptr = output.data_as<bf16>();
    const void* b_ptr = bias.data();
    bool bias_is_f32 = (bias.dtype() == dnnl::memory::data_type::f32);

    int total_out = batch * seq_len * channels;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total_out), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            if (i >= total_out) return;

            // Decompose index in row-major layout [batch, seq_len, channels]
            int c = i % channels;
            int tmp = i / channels;
            int t = tmp % seq_len;
            int n = tmp / seq_len;

            float sum = 0.0f;

            for (int k = 0; k < kernel_size; ++k) {
                int ih = t - (kernel_size - 1 - k) * dilation;
                if (ih < 0 || ih >= seq_len) continue;

                // input row-major layout: [batch, seq_len, channels]
                int in_idx = (n * seq_len + ih) * channels + c;
                // weight row-major layout: [channels, 1, kernel_size]
                int w_idx = c * kernel_size + k;

                sum += static_cast<float>(in_ptr[in_idx]) * static_cast<float>(w_ptr[w_idx]);
            }

            float bias_val = bias_is_f32 ? static_cast<const float*>(b_ptr)[c] : static_cast<float>(
                static_cast<const bf16*>(b_ptr)[c]);
            sum += bias_val;

            out_ptr[i] = bf16(sum);
        });
    });
}

// 1D Causal Transposed Convolution (row-major: [batch, seq_len, channels])
void causal_conv_transpose1d(Context& ctx,
                             Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                             int batch, int in_ch, int out_ch, int in_len,
                             int kernel_size, int stride) {
    auto* in_ptr = input.data_as<bf16>();
    auto* w_ptr = weight.data_as<bf16>();
    auto* out_ptr = output.data_as<bf16>();
    const void* b_ptr = bias.data();
    bool bias_is_f32 = (bias.dtype() == dnnl::memory::data_type::f32);

    int out_len = in_len * stride;
    int total_out = batch * out_len * out_ch;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total_out), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            if (i >= total_out) return;

            // Decompose index in row-major layout [batch, out_len, out_ch]
            int oc = i % out_ch;
            int tmp = i / out_ch;
            int ot = tmp % out_len;
            int n = tmp / out_len;

            float sum = 0.0f;

            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    int rem = ot - k;
                    if (rem >= 0 && (rem % stride == 0)) {
                        int it = rem / stride;
                        if (it < in_len) {
                            // input row-major layout: [batch, in_len, in_ch]
                            int in_idx = (n * in_len + it) * in_ch + ic;
                            // weight row-major layout: [in_ch, out_ch, kernel_size]
                            int w_idx = (ic * out_ch + oc) * kernel_size + k;

                            sum += static_cast<float>(in_ptr[in_idx]) * static_cast<float>(w_ptr[w_idx]);
                        }
                    }
                }
            }

            float bias_val = bias_is_f32 ? static_cast<const float*>(b_ptr)[oc] : static_cast<float>(
                static_cast<const bf16*>(b_ptr)[oc]);
            sum += bias_val;

            out_ptr[i] = bf16(sum);
        });
    });
}

// 1D Convolution with reflect padding (symmetric on both sides)
void reflect_conv1d(Context& ctx,
                    Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                    int batch, int in_ch, int out_ch, int seq_len,
                    int kernel_size, int dilation, int pad,
                    bool relu) {
    auto* in_ptr = input.data_as<bf16>();
    auto* w_ptr = weight.data_as<bf16>();
    auto* out_ptr = output.data_as<bf16>();
    const void* b_ptr = bias.data();
    bool bias_is_f32 = (bias.dtype() == dnnl::memory::data_type::f32);

    int padded_len = seq_len + 2 * pad;
    int total_out = batch * seq_len * out_ch;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total_out), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            if (i >= total_out) return;

            int oc = i % out_ch;
            int tmp = i / out_ch;
            int t = tmp % seq_len;
            int n = tmp / seq_len;

            float sum = 0.0f;

            for (int ic = 0; ic < in_ch; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    // PyTorch Conv1D with "same" padding + reflect mode:
                    //   output[t] = sum_k weight[k] * padded[t + k * dilation]
                    // where padded is the reflect-padded input (PyTorch coordinate system:
                    //   original occupies [0, seq_len-1], reflections at negative & >= seq_len).
                    int idx = t + k * dilation;
                    int src_t;
                    if (idx < 0) {
                        src_t = -idx - 1;       // PyTorch: padded[-1]=input[0]
                    } else if (idx >= seq_len) {
                        src_t = 2 * seq_len - idx - 2;  // PyTorch: padded[L]=input[L-2]
                    } else {
                        src_t = idx;
                    }
                    src_t = sycl::clamp(src_t, 0, seq_len - 1);

                    // input row-major: [batch, seq_len, in_ch]
                    int in_idx = (n * seq_len + src_t) * in_ch + ic;
                    // weight row-major: [out_ch, in_ch, kernel_size]
                    int w_idx = (oc * in_ch + ic) * kernel_size + k;

                    sum += static_cast<float>(in_ptr[in_idx]) * static_cast<float>(w_ptr[w_idx]);
                }
            }

            float bias_val = bias_is_f32 ? static_cast<const float*>(b_ptr)[oc]
                                         : static_cast<float>(static_cast<const bf16*>(b_ptr)[oc]);
            sum += bias_val;

            if (relu) sum = sycl::fmax(sum, 0.0f);
            out_ptr[i] = bf16(sum);
        });
    });
}

void relu_inplace(Context& ctx, Tensor& x, int n) {
    auto* ptr = x.data_as<bf16>();
    ctx.queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float v = static_cast<float>(ptr[i]);
        ptr[i] = bf16(sycl::fmax(v, 0.0f));
    });
}

void sigmoid_inplace(Context& ctx, Tensor& x, int n) {
    auto* ptr = x.data_as<bf16>();
    ctx.queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float v = static_cast<float>(ptr[i]);
        ptr[i] = bf16(1.0f / (1.0f + sycl::exp(-v)));
    });
}

void tanh_inplace(Context& ctx, Tensor& x, int n) {
    auto* ptr = x.data_as<bf16>();
    ctx.queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float v = static_cast<float>(ptr[i]);
        ptr[i] = bf16(sycl::tanh(v));
    });
}

void global_avg_pool_1d(Context& ctx, Tensor& input, Tensor& output,
                        int batch, int seq_len, int channels) {
    auto* in_ptr = input.data_as<bf16>();
    auto* out_ptr = output.data_as<bf16>();
    float inv_len = 1.0f / static_cast<float>(seq_len);

    int total_out = batch * channels;
    ctx.queue().parallel_for(sycl::range<1>(total_out), [=](sycl::id<1> idx) {
        int c = idx % channels;
        int n = idx / channels;

        float sum = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            int in_idx = (n * seq_len + t) * channels + c;
            sum += static_cast<float>(in_ptr[in_idx]);
        }
        out_ptr[idx] = bf16(sum * inv_len);
    });
}

void softmax_1d_time(Context& ctx, Tensor& data, int batch, int channels, int seq_len) {
    auto* ptr = data.data_as<bf16>();
    int total = batch * channels;
    ctx.queue().parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        int c = idx % channels;
        int n = idx / channels;

        float max_val = -1e30f;
        for (int t = 0; t < seq_len; ++t) {
            float v = static_cast<float>(ptr[(n * channels + c) * seq_len + t]);
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            float v = static_cast<float>(ptr[(n * channels + c) * seq_len + t]);
            sum_exp += sycl::exp(v - max_val);
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int t = 0; t < seq_len; ++t) {
            float v = static_cast<float>(ptr[(n * channels + c) * seq_len + t]);
            ptr[(n * channels + c) * seq_len + t] = bf16(sycl::exp(v - max_val) * inv_sum);
        }
    });
}

void channel_mul_add_inplace(Context& ctx, Tensor& src, Tensor& dst,
                             int channels, int seq_len) {
    auto* s_ptr = src.data_as<bf16>();  // [1, 1, channels] — channel-wise scalars
    auto* d_ptr = dst.data_as<bf16>();  // [1, seq_len, channels]

    int total = seq_len * channels;
    ctx.queue().parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        int c = idx % channels;
        int t = idx / channels;
        float g = static_cast<float>(s_ptr[c]);
        d_ptr[idx] = bf16(static_cast<float>(d_ptr[idx]) * (1.0f + g));
    });
}

} // namespace ops
