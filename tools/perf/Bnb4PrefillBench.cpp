#include "src/ops/Bnb4BitLinear.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

struct BenchShape {
    const char* name;
    int m;
    int n;
    int k;
};

uint8_t nf4_at(const std::vector<uint8_t>& packed, int n, int k, int k_dim) {
    const size_t byte_index = static_cast<size_t>(n) * static_cast<size_t>(k_dim / 2) +
                              static_cast<size_t>(k / 2);
    const uint8_t byte = packed[byte_index];
    return (k % 2 == 0) ? static_cast<uint8_t>(byte >> 4)
                        : static_cast<uint8_t>(byte & 0x0F);
}

std::vector<float> make_qmap() {
    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.015625f;
    }
    return qmap;
}

std::vector<float> make_absmax(int n, int blocks_per_row) {
    std::vector<float> absmax(static_cast<size_t>(n) * static_cast<size_t>(blocks_per_row));
    for (int row = 0; row < n; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            const float value = 0.5f + 0.03125f * static_cast<float>((row * 7 + block * 3) % 9);
            absmax[static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) +
                   static_cast<size_t>(block)] = value;
        }
    }
    return absmax;
}

std::vector<uint8_t> make_packed(int n, int k) {
    std::vector<uint8_t> packed(static_cast<size_t>(n) * static_cast<size_t>(k / 2));
    for (int row = 0; row < n; ++row) {
        for (int kh = 0; kh < k / 2; ++kh) {
            const uint8_t hi = static_cast<uint8_t>((row * 5 + kh * 3) & 0x0F);
            const uint8_t lo = static_cast<uint8_t>((row * 11 + kh * 7 + 1) & 0x0F);
            packed[static_cast<size_t>(row) * static_cast<size_t>(k / 2) +
                   static_cast<size_t>(kh)] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }
    return packed;
}

std::vector<bf16> make_input(int m, int k) {
    std::vector<bf16> input(static_cast<size_t>(m) * static_cast<size_t>(k));
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            const int pattern = ((row + 3) * (col % 17) + col) % 15;
            const float value = (static_cast<float>(pattern) - 7.0f) * 0.001953125f;
            input[static_cast<size_t>(row) * static_cast<size_t>(k) +
                  static_cast<size_t>(col)] = bf16(value);
        }
    }
    return input;
}

double elapsed_ms(const std::chrono::high_resolution_clock::time_point& start,
                  const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double sampled_max_abs_diff(const BenchShape& shape,
                            const std::vector<uint8_t>& packed,
                            const std::vector<float>& qmap,
                            const std::vector<float>& absmax,
                            const std::vector<bf16>& input,
                            const std::vector<bf16>& output,
                            int blocksize) {
    const int blocks_per_row = shape.k / blocksize;
    const int sample_m[] = {0, shape.m / 2, shape.m - 1};
    const int sample_n[] = {0, 1, 17, 63, 127, shape.n / 4, shape.n / 2, shape.n - 1};

    double max_abs_diff = 0.0;
    for (int m_idx : sample_m) {
        if (m_idx < 0 || m_idx >= shape.m) continue;
        for (int n_idx : sample_n) {
            if (n_idx < 0 || n_idx >= shape.n) continue;
            double expected = 0.0;
            for (int k_idx = 0; k_idx < shape.k; ++k_idx) {
                const uint8_t code = nf4_at(packed, n_idx, k_idx, shape.k);
                const int block = k_idx / blocksize;
                expected += static_cast<float>(
                                input[static_cast<size_t>(m_idx) * static_cast<size_t>(shape.k) +
                                      static_cast<size_t>(k_idx)]) *
                            qmap[static_cast<size_t>(code)] *
                            absmax[static_cast<size_t>(n_idx) * static_cast<size_t>(blocks_per_row) +
                                   static_cast<size_t>(block)];
            }
            const double actual = static_cast<float>(
                output[static_cast<size_t>(m_idx) * static_cast<size_t>(shape.n) +
                       static_cast<size_t>(n_idx)]);
            max_abs_diff = std::max(max_abs_diff, std::abs(actual - expected));
        }
    }
    return max_abs_diff;
}

int run_shape(Context& ctx, const BenchShape& shape) {
    constexpr int blocksize = 128;
    const int blocks_per_row = shape.k / blocksize;

    const std::vector<float> qmap = make_qmap();
    const std::vector<float> absmax = make_absmax(shape.n, blocks_per_row);
    const std::vector<uint8_t> packed = make_packed(shape.n, shape.k);
    const std::vector<bf16> input = make_input(shape.m, shape.k);

    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                         dnnl::memory::data_type::u8);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    Tensor absmax_dev = Tensor::allocate(ctx, {static_cast<int64_t>(absmax.size())},
                                         dnnl::memory::data_type::f32);
    Tensor input_dev = Tensor::allocate(ctx, {shape.m, shape.k}, dnnl::memory::data_type::bf16);
    Tensor output_dev = Tensor::allocate(ctx, {shape.m, shape.n}, dnnl::memory::data_type::bf16);

    ctx.memcpy_h2d(packed_dev.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(qmap_dev.data(), qmap.data(), qmap.size() * sizeof(float));
    ctx.memcpy_h2d(absmax_dev.data(), absmax.data(), absmax.size() * sizeof(float));
    ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));
    ctx.synchronize();

    auto run_once = [&]() {
        const auto start = std::chrono::high_resolution_clock::now();
        packed_nf4_gemm_bf16(ctx,
                             static_cast<const uint8_t*>(packed_dev.data()),
                             static_cast<const float*>(qmap_dev.data()),
                             static_cast<const float*>(absmax_dev.data()),
                             static_cast<const bf16*>(input_dev.data()),
                             static_cast<bf16*>(output_dev.data()),
                             shape.m, shape.n, shape.k, blocksize);
        ctx.synchronize();
        const auto end = std::chrono::high_resolution_clock::now();
        return elapsed_ms(start, end);
    };

    const double warmup_ms = run_once();

    constexpr int iterations = 5;
    double best_ms = std::numeric_limits<double>::max();
    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const double ms = run_once();
        best_ms = std::min(best_ms, ms);
        total_ms += ms;
    }

    std::vector<bf16> output(static_cast<size_t>(shape.m) * static_cast<size_t>(shape.n));
    ctx.memcpy_d2h(output.data(), output_dev.data(), output.size() * sizeof(bf16));

    const double max_abs_diff = sampled_max_abs_diff(
        shape, packed, qmap, absmax, input, output, blocksize);
    const double avg_ms = total_ms / static_cast<double>(iterations);
    const double tokens_per_s = static_cast<double>(shape.m) * 1000.0 / best_ms;

    std::cout << shape.name << ',' << shape.m << ',' << shape.n << ',' << shape.k << ','
              << std::fixed << std::setprecision(3)
              << warmup_ms << ',' << best_ms << ',' << avg_ms << ','
              << std::setprecision(6) << max_abs_diff << ','
              << std::setprecision(1) << tokens_per_s << '\n';

    return max_abs_diff <= 0.08 ? 0 : 1;
}

}  // namespace

int main() {
    const BenchShape shapes[] = {
        {"gate_up_like", 118, 10240, 2560},
        {"down_like", 118, 2560, 10240},
        {"stream_prefill_gate_up_like", 91, 10240, 2560},
        {"asr_short_qkv_like", 69, 4096, 2048},
        {"asr_short_o_like", 69, 2048, 2048},
        {"asr_short_gate_up_like", 69, 12288, 2048},
        {"asr_real_gate_up_like", 72, 12288, 2048},
        {"asr_short_down_like", 69, 2048, 6144},
    };

    std::cout << "shape,M,N,K,warmup_ms,best_ms,avg_ms,max_abs_diff,tokens_per_s\n";

    Context ctx;
    int failures = 0;
    for (const BenchShape& shape : shapes) {
        failures += run_shape(ctx, shape);
    }

    if (failures != 0) {
        std::cerr << "AilaBnb4PrefillBench: " << failures << " shape(s) exceeded tolerance\n";
        return 1;
    }
    return 0;
}
