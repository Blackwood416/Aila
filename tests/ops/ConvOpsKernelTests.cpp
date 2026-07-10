#include "src/ops/ConvOps.cpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

std::vector<float> reference_conv_transpose(
    const std::vector<bf16>& input,
    const std::vector<bf16>& logical_weight,
    const std::vector<bf16>& bias,
    int in_ch,
    int out_ch,
    int in_len,
    int kernel_size,
    int stride) {
    const int out_len = in_len * stride;
    std::vector<float> output(static_cast<size_t>(out_len * out_ch));
    for (int ot = 0; ot < out_len; ++ot) {
        for (int oc = 0; oc < out_ch; ++oc) {
            float sum = static_cast<float>(bias[static_cast<size_t>(oc)]);
            for (int k = ot % stride; k < kernel_size; k += stride) {
                const int it = (ot - k) / stride;
                if (it < 0 || it >= in_len) {
                    continue;
                }
                for (int ic = 0; ic < in_ch; ++ic) {
                    const size_t in_idx = static_cast<size_t>(it * in_ch + ic);
                    const size_t w_idx = static_cast<size_t>((ic * out_ch + oc) * kernel_size + k);
                    sum += static_cast<float>(input[in_idx]) *
                           static_cast<float>(logical_weight[w_idx]);
                }
            }
            output[static_cast<size_t>(ot * out_ch + oc)] = sum;
        }
    }
    return output;
}

double run_conv_transpose_layout(bool vec8_layout) {
    constexpr int InCh = 8;
    constexpr int OutCh = 3;
    constexpr int InLen = 4;
    constexpr int Kernel = 4;
    constexpr int Stride = 2;
    constexpr int OutLen = InLen * Stride;

    Context ctx;
    std::vector<bf16> input(static_cast<size_t>(InLen * InCh));
    std::vector<bf16> logical_weight(static_cast<size_t>(InCh * OutCh * Kernel));
    std::vector<bf16> bias(static_cast<size_t>(OutCh));

    for (int i = 0; i < InLen * InCh; ++i) {
        input[static_cast<size_t>(i)] = bf16(static_cast<float>((i % 9) - 4) * 0.0625f);
    }
    for (int ic = 0; ic < InCh; ++ic) {
        for (int oc = 0; oc < OutCh; ++oc) {
            for (int k = 0; k < Kernel; ++k) {
                const size_t idx = static_cast<size_t>((ic * OutCh + oc) * Kernel + k);
                logical_weight[idx] = bf16(static_cast<float>(((ic + 1) * 3 + oc * 5 + k * 7) % 17 - 8) * 0.03125f);
            }
        }
    }
    for (int oc = 0; oc < OutCh; ++oc) {
        bias[static_cast<size_t>(oc)] = bf16(static_cast<float>(oc - 1) * 0.125f);
    }

    std::vector<bf16> packed_weight(logical_weight.size());
    if (vec8_layout) {
        for (int ic = 0; ic < InCh; ++ic) {
            for (int oc = 0; oc < OutCh; ++oc) {
                for (int k = 0; k < Kernel; ++k) {
                    const size_t src = static_cast<size_t>((ic * OutCh + oc) * Kernel + k);
                    const size_t dst = static_cast<size_t>((oc * Kernel + k) * InCh + ic);
                    packed_weight[dst] = logical_weight[src];
                }
            }
        }
    } else {
        packed_weight = logical_weight;
    }

    Tensor input_dev = Tensor::allocate(ctx, {InLen, InCh});
    Tensor weight_dev = vec8_layout
        ? Tensor::allocate(ctx, {OutCh, Kernel, InCh})
        : Tensor::allocate(ctx, {InCh, OutCh, Kernel});
    Tensor bias_dev = Tensor::allocate(ctx, {OutCh});
    Tensor output_dev = Tensor::allocate(ctx, {OutLen, OutCh});
    ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));
    ctx.memcpy_h2d(weight_dev.data(), packed_weight.data(), packed_weight.size() * sizeof(bf16));
    ctx.memcpy_h2d(bias_dev.data(), bias.data(), bias.size() * sizeof(bf16));

    ops::causal_conv_transpose1d(
        ctx, input_dev, weight_dev, bias_dev, output_dev,
        1, InCh, OutCh, InLen, Kernel, Stride);
    ctx.synchronize();

    std::vector<bf16> actual(static_cast<size_t>(OutLen * OutCh));
    ctx.memcpy_d2h(actual.data(), output_dev.data(), actual.size() * sizeof(bf16));
    const std::vector<float> expected = reference_conv_transpose(
        input, logical_weight, bias, InCh, OutCh, InLen, Kernel, Stride);

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs_diff = std::max(
            max_abs_diff,
            std::abs(static_cast<double>(static_cast<float>(actual[i])) - expected[i]));
    }
    return max_abs_diff;
}

double run_fused_k1_residual_test() {
    constexpr int InCh = 8;
    constexpr int OutCh = 4;
    constexpr int SeqLen = 3;

    Context ctx;
    std::vector<bf16> input(static_cast<size_t>(SeqLen * InCh));
    std::vector<bf16> weight(static_cast<size_t>(OutCh * InCh));
    std::vector<bf16> bias(static_cast<size_t>(OutCh));
    std::vector<bf16> residual(static_cast<size_t>(SeqLen * OutCh));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = bf16(static_cast<float>(static_cast<int>(i % 11) - 5) * 0.03125f);
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = bf16(static_cast<float>(static_cast<int>(i % 13) - 6) * 0.015625f);
    }
    for (int oc = 0; oc < OutCh; ++oc) {
        bias[static_cast<size_t>(oc)] = bf16(static_cast<float>(oc - 2) * 0.0625f);
    }
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = bf16(static_cast<float>(static_cast<int>(i % 7) - 3) * 0.046875f);
    }

    std::vector<float> expected(residual.size());
    for (int t = 0; t < SeqLen; ++t) {
        for (int oc = 0; oc < OutCh; ++oc) {
            float sum = static_cast<float>(bias[static_cast<size_t>(oc)]);
            for (int ic = 0; ic < InCh; ++ic) {
                sum += static_cast<float>(input[static_cast<size_t>(t * InCh + ic)]) *
                       static_cast<float>(weight[static_cast<size_t>(oc * InCh + ic)]);
            }
            const bf16 conv = bf16(sum);
            const size_t idx = static_cast<size_t>(t * OutCh + oc);
            expected[idx] = static_cast<float>(residual[idx]) + static_cast<float>(conv);
        }
    }

    Tensor input_dev = Tensor::allocate(ctx, {SeqLen, InCh});
    Tensor weight_dev = Tensor::allocate(ctx, {OutCh, InCh, 1});
    Tensor bias_dev = Tensor::allocate(ctx, {OutCh});
    Tensor residual_dev = Tensor::allocate(ctx, {SeqLen, OutCh});
    ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));
    ctx.memcpy_h2d(weight_dev.data(), weight.data(), weight.size() * sizeof(bf16));
    ctx.memcpy_h2d(bias_dev.data(), bias.data(), bias.size() * sizeof(bf16));
    ctx.memcpy_h2d(residual_dev.data(), residual.data(), residual.size() * sizeof(bf16));

    ops::causal_conv1d_k1_residual_add(
        ctx, input_dev, weight_dev, bias_dev, residual_dev,
        1, InCh, OutCh, SeqLen);
    ctx.synchronize();

    std::vector<bf16> actual(residual.size());
    ctx.memcpy_d2h(actual.data(), residual_dev.data(), actual.size() * sizeof(bf16));
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs_diff = std::max(
            max_abs_diff,
            std::abs(static_cast<double>(static_cast<float>(actual[i])) - expected[i]));
    }
    return max_abs_diff;
}

}  // namespace

int main() {
    const double legacy_diff = run_conv_transpose_layout(false);
    const double vec8_diff = run_conv_transpose_layout(true);
    const double fused_diff = run_fused_k1_residual_test();
    std::cout << "legacy max_abs_diff=" << legacy_diff << "\n";
    std::cout << "vec8 max_abs_diff=" << vec8_diff << "\n";
    std::cout << "fused k1 residual max_abs_diff=" << fused_diff << "\n";
    if (legacy_diff > 0.02 || vec8_diff > 0.02 || fused_diff > 0.02) {
        std::cerr << "AilaConvOpsKernelTests: failed\n";
        return 1;
    }
    std::cout << "AilaConvOpsKernelTests: passed\n";
    return 0;
}
