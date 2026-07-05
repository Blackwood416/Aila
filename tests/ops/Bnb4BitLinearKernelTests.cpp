#include "src/ops/Bnb4BitLinear.cpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

uint8_t nf4_at(const std::vector<uint8_t>& packed, int n, int k, int k_dim) {
    const int byte_index = n * (k_dim / 2) + (k / 2);
    const uint8_t byte = packed[static_cast<size_t>(byte_index)];
    return (k % 2 == 0) ? static_cast<uint8_t>(byte >> 4)
                        : static_cast<uint8_t>(byte & 0x0F);
}

int run_packed_nf4_gemm_layout_test() {
    constexpr int M = 3;
    constexpr int N = 128;
    constexpr int K = 128;
    constexpr int Block = 128;

    Context ctx;

    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }

    std::vector<float> absmax(N);
    for (int n = 0; n < N; ++n) {
        absmax[static_cast<size_t>(n)] = 0.75f + 0.05f * static_cast<float>(n % 11);
    }

    std::vector<uint8_t> packed(static_cast<size_t>(N * K / 2));
    for (int n = 0; n < N; ++n) {
        for (int kh = 0; kh < K / 2; ++kh) {
            const uint8_t hi = static_cast<uint8_t>((n * 3 + kh * 5) & 0x0F);
            const uint8_t lo = static_cast<uint8_t>((n * 7 + kh * 2 + 1) & 0x0F);
            packed[static_cast<size_t>(n * (K / 2) + kh)] =
                static_cast<uint8_t>((hi << 4) | lo);
        }
    }

    std::vector<bf16> input(static_cast<size_t>(M * K));
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float value = (static_cast<float>((m + 1) * ((k % 13) - 6))) * 0.03125f;
            input[static_cast<size_t>(m * K + k)] = bf16(value);
        }
    }

    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                         dnnl::memory::data_type::u8);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    Tensor absmax_dev = Tensor::allocate(ctx, {N}, dnnl::memory::data_type::f32);
    Tensor input_dev = Tensor::allocate(ctx, {M, K}, dnnl::memory::data_type::bf16);
    Tensor output_dev = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(packed_dev.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(qmap_dev.data(), qmap.data(), qmap.size() * sizeof(float));
    ctx.memcpy_h2d(absmax_dev.data(), absmax.data(), absmax.size() * sizeof(float));
    ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));

    packed_nf4_gemm_bf16(ctx,
                         static_cast<const uint8_t*>(packed_dev.data()),
                         static_cast<const float*>(qmap_dev.data()),
                         static_cast<const float*>(absmax_dev.data()),
                         static_cast<const bf16*>(input_dev.data()),
                         static_cast<bf16*>(output_dev.data()),
                         M, N, K, Block);
    ctx.synchronize();

    std::vector<bf16> output(static_cast<size_t>(M * N));
    ctx.memcpy_d2h(output.data(), output_dev.data(), output.size() * sizeof(bf16));

    double max_abs_diff = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                const uint8_t code = nf4_at(packed, n, k, K);
                expected += static_cast<float>(input[static_cast<size_t>(m * K + k)]) *
                            qmap[static_cast<size_t>(code)] *
                            absmax[static_cast<size_t>(n)];
            }
            const double actual = static_cast<float>(output[static_cast<size_t>(m * N + n)]);
            const double diff = std::abs(actual - expected);
            max_abs_diff = std::max(max_abs_diff, diff);
        }
    }

    if (max_abs_diff > 0.06) {
        std::cerr << "packed_nf4_gemm_bf16 max_abs_diff=" << max_abs_diff
                  << " exceeds tolerance\n";
        return 1;
    }

    std::cout << "packed_nf4_gemm_bf16 max_abs_diff=" << max_abs_diff << "\n";
    return 0;
}

}  // namespace

int main() {
    const int failed = run_packed_nf4_gemm_layout_test();
    if (failed != 0) {
        std::cout << "AilaBnb4BitLinearKernelTests: failed\n";
        return failed;
    }
    std::cout << "AilaBnb4BitLinearKernelTests: passed\n";
    return 0;
}
