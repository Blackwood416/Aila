#include "src/ops/Bnb4BitLinear.cpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <filesystem>
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

int run_packed_nf4_gemm_layout_test(int M, int Block) {
    constexpr int N = 128;
    constexpr int K = 128;

    Context ctx;

    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }

    const int blocks_per_row = K / Block;
    std::vector<float> absmax(static_cast<size_t>(N * blocks_per_row));
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            absmax[static_cast<size_t>(n * blocks_per_row + b)] =
                0.75f + 0.05f * static_cast<float>((n + b) % 11);
        }
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
    Tensor absmax_dev = Tensor::allocate(
        ctx, {static_cast<int64_t>(absmax.size())},
        dnnl::memory::data_type::f32);
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
    double expected_at_max_diff = 0.0;
    double actual_at_max_diff = 0.0;
    double max_relative_diff = 0.0;
    bool exceeds_tolerance = false;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                const uint8_t code = nf4_at(packed, n, k, K);
                expected +=
                    static_cast<float>(input[static_cast<size_t>(m * K + k)]) *
                    static_cast<float>(bf16(
                        qmap[static_cast<size_t>(code)] *
                        absmax[static_cast<size_t>(
                            n * blocks_per_row + k / Block)]));
            }
            const double actual = static_cast<float>(output[static_cast<size_t>(m * N + n)]);
            const double diff = std::abs(actual - expected);
            max_relative_diff = std::max(
                max_relative_diff,
                diff / std::max(1.0, std::abs(expected)));
            const double tolerance = std::max(0.06, 0.005 * std::abs(expected));
            if (diff > tolerance) {
                exceeds_tolerance = true;
            }
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                expected_at_max_diff = expected;
                actual_at_max_diff = actual;
            }
        }
    }

    if (exceeds_tolerance) {
        std::cerr << "packed_nf4_gemm_bf16 M=" << M << " Block=" << Block
                  << " max_abs_diff=" << max_abs_diff
                  << " expected=" << expected_at_max_diff
                  << " actual=" << actual_at_max_diff
                  << " max_relative_diff=" << max_relative_diff
                  << " exceeds tolerance\n";
        return 1;
    }

    std::cout << "packed_nf4_gemm_bf16 M=" << M << " Block=" << Block
              << " max_abs_diff=" << max_abs_diff
              << " max_relative_diff=" << max_relative_diff << "\n";
    return 0;
}

int run_packed_nf4_gemm_large_test() {
    constexpr int M = 256;
    constexpr int N = 2816;
    constexpr int K = 8192;
    constexpr int Block = 64;

    Context ctx;
    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }
    const int blocks_per_row = K / Block;
    std::vector<float> absmax(static_cast<size_t>(N * blocks_per_row));
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            absmax[static_cast<size_t>(n * blocks_per_row + b)] =
                0.5f + 0.001f * static_cast<float>((n + b) % 97);
        }
    }
    std::vector<uint8_t> packed(static_cast<size_t>(N * K / 2));
    for (size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<uint8_t>((i * 13 + 7) & 0xFF);
    }
    std::vector<bf16> input(static_cast<size_t>(M * K));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = bf16(static_cast<float>((static_cast<int>(i) % 31) - 15) * 0.01f);
    }

    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                         dnnl::memory::data_type::u8);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    Tensor absmax_dev = Tensor::allocate(ctx, {N * blocks_per_row},
                                         dnnl::memory::data_type::f32);
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
    std::cout << "packed_nf4_gemm_bf16 large M=" << M << " N=" << N
              << " K=" << K << " Block=" << Block << " passed\n";
    return 0;
}

int run_packed_nf4_gemm_qproj_test() {
    constexpr int M = 256;
    constexpr int N = 4096;
    constexpr int K = 2816;
    constexpr int Block = 64;

    Context ctx;
    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }
    const int blocks_per_row = K / Block;
    std::vector<float> absmax(static_cast<size_t>(N * blocks_per_row));
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            absmax[static_cast<size_t>(n * blocks_per_row + b)] =
                0.5f + 0.001f * static_cast<float>((n + b) % 97);
        }
    }
    std::vector<uint8_t> packed(static_cast<size_t>(N * K / 2));
    for (size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<uint8_t>((i * 13 + 7) & 0xFF);
    }
    std::vector<bf16> input(static_cast<size_t>(M * K));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = bf16(static_cast<float>((static_cast<int>(i) % 31) - 15) * 0.01f);
    }

    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                         dnnl::memory::data_type::u8);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    Tensor absmax_dev = Tensor::allocate(ctx, {N * blocks_per_row},
                                         dnnl::memory::data_type::f32);
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
    std::vector<bf16> output(static_cast<size_t>(M) * N);
    ctx.memcpy_d2h(output.data(), output_dev.data(), output.size() * sizeof(bf16));
    bool bad = false;
    size_t bad_index = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        uint16_t raw = 0;
        std::memcpy(&raw, &output[i], sizeof(raw));
        if ((raw & 0x7F80u) == 0x7F80u) {
            bad = true;
            bad_index = i;
            break;
        }
    }
    if (bad) {
        std::cerr << "packed_nf4_gemm_bf16 qproj-shape has NaN at " << bad_index
                  << "\n";
        return 1;
    }
    std::cout << "packed_nf4_gemm_bf16 qproj-shape M=" << M << " N=" << N
              << " K=" << K << " Block=" << Block << " passed\n";
    return 0;
}

int run_packed_nf4_gemm_shape_test(int M, int N, int K, bool zero_input = false) {
    constexpr int Block = 64;
    Context ctx;
    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }
    const int blocks_per_row = K / Block;
    std::vector<float> absmax(static_cast<size_t>(N * blocks_per_row));
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            absmax[static_cast<size_t>(n * blocks_per_row + b)] =
                0.5f + 0.001f * static_cast<float>((n + b) % 97);
        }
    }
    std::vector<uint8_t> packed(static_cast<size_t>(N * K / 2));
    for (size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<uint8_t>((i * 13 + 7) & 0xFF);
    }
    std::vector<bf16> input(static_cast<size_t>(M * K));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = zero_input
                       ? bf16(0.0f)
                       : bf16(static_cast<float>((static_cast<int>(i) % 31) - 15) *
                              0.01f);
    }

    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                         dnnl::memory::data_type::u8);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    Tensor absmax_dev = Tensor::allocate(ctx, {N * blocks_per_row},
                                         dnnl::memory::data_type::f32);
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
    std::vector<bf16> output(static_cast<size_t>(M) * N);
    ctx.memcpy_d2h(output.data(), output_dev.data(), output.size() * sizeof(bf16));
    bool bad = false;
    size_t bad_index = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        uint16_t raw = 0;
        std::memcpy(&raw, &output[i], sizeof(raw));
        if ((raw & 0x7F80u) == 0x7F80u) {
            bad = true;
            bad_index = i;
            break;
        }
    }
    if (bad) {
        std::cerr << "packed_nf4_gemm_bf16 M=" << M << " N=" << N << " K=" << K
                  << " has NaN at " << bad_index << "\n";
        return 1;
    }
    // Value check for the first few rows (catches small-template bugs).
    for (int r = 0; r < std::min(3, M); ++r) {
        for (int c = 0; c < std::min(4, N); ++c) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                const int block = k / Block;
                const uint8_t byte = packed[static_cast<size_t>(c * (K / 2) + k / 2)];
                const uint8_t code =
                    (k % 2 == 0) ? static_cast<uint8_t>(byte >> 4)
                                 : static_cast<uint8_t>(byte & 0x0F);
                expected +=
                    static_cast<float>(input[static_cast<size_t>(r * K + k)]) *
                    static_cast<float>(bf16(
                        qmap[code] *
                        absmax[static_cast<size_t>(c * blocks_per_row + block)]));
            }
            const double actual =
                static_cast<float>(output[static_cast<size_t>(r * N + c)]);
            const double rel = std::abs(actual - expected) /
                               std::max(1.0, std::abs(expected));
            if (rel > 0.05) {
                std::cerr << "packed_nf4_gemm_bf16 M=" << M << " N=" << N
                          << " K=" << K << " row=" << r << " col=" << c
                          << " actual=" << actual << " expected=" << expected
                          << " rel=" << rel << "\n";
                return 1;
            }
        }
    }
    if (!zero_input && M == 256 &&
        aila::env::read_flag("AILA_BNB4_BENCH", false)) {
        constexpr int kWarmup = 3;
        constexpr int kSamples = 20;
        for (int i = 0; i < kWarmup; ++i) {
            packed_nf4_gemm_bf16(
                ctx, static_cast<const uint8_t*>(packed_dev.data()),
                static_cast<const float*>(qmap_dev.data()),
                static_cast<const float*>(absmax_dev.data()),
                static_cast<const bf16*>(input_dev.data()),
                static_cast<bf16*>(output_dev.data()), M, N, K, Block);
            ctx.synchronize();
        }
        std::vector<double> samples;
        samples.reserve(kSamples);
        for (int i = 0; i < kSamples; ++i) {
            const auto start = std::chrono::steady_clock::now();
            packed_nf4_gemm_bf16(
                ctx, static_cast<const uint8_t*>(packed_dev.data()),
                static_cast<const float*>(qmap_dev.data()),
                static_cast<const float*>(absmax_dev.data()),
                static_cast<const bf16*>(input_dev.data()),
                static_cast<bf16*>(output_dev.data()), M, N, K, Block);
            ctx.synchronize();
            samples.push_back(std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
        }
        std::sort(samples.begin(), samples.end());
        std::cout << "[BNB4 bench] M=" << M << " N=" << N << " K=" << K
                  << " block=" << Block
                  << " wall_median_ms=" << samples[kSamples / 2]
                  << " p10_ms=" << samples[kSamples / 10]
                  << " p90_ms=" << samples[(kSamples * 9) / 10] << "\n";
    }
    std::cout << "packed_nf4_gemm_bf16 M=" << M << " N=" << N << " K=" << K
              << (zero_input ? " zero-input" : "") << " passed\n";
    return 0;
}

int run_real_scgate_kernel_test() {
    namespace fs = std::filesystem;
    const fs::path dir =
        fs::temp_directory_path() / "dg_scgate";
    if (!fs::exists(dir / "packed.bin")) {
        std::cout << "packed_nf4_gemm_bf16 real-scgate skipped (fixture missing)\n";
        return 0;
    }
    constexpr int M = 256;
    constexpr int N = 2112;
    constexpr int K = 2816;
    constexpr int Block = 64;
    const int blocks_per_row = K / Block;

    auto load_vec = [&](const std::string& name) {
        std::ifstream in(dir / name, std::ios::binary);
        return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    };
    std::vector<uint8_t> packed = load_vec("packed.bin");
    std::vector<uint8_t> absmax_codes = load_vec("absmax.bin");
    std::vector<uint8_t> nested_bytes = load_vec("nested.bin");
    std::vector<uint8_t> nqmap_bytes = load_vec("nqmap.bin");
    std::vector<uint8_t> qmap_bytes = load_vec("qmap.bin");

    std::vector<float> qmap(qmap_bytes.size() / 4);
    std::memcpy(qmap.data(), qmap_bytes.data(), qmap_bytes.size());
    std::vector<float> nested(nested_bytes.size() / 4);
    std::memcpy(nested.data(), nested_bytes.data(), nested_bytes.size());
    std::vector<float> nqmap(nqmap_bytes.size() / 4);
    std::memcpy(nqmap.data(), nqmap_bytes.data(), nqmap_bytes.size());
    float nested_offset = 0.0f;
    {
        std::ifstream off(dir / "offset.txt");
        if (off) off >> nested_offset;
    }

    const int64_t block_count = static_cast<int64_t>(absmax_codes.size());
    std::vector<float> absmax_host(static_cast<size_t>(block_count));
    for (int64_t i = 0; i < block_count; ++i) {
        const int64_t nb = i / 256;
        absmax_host[static_cast<size_t>(i)] =
            nqmap[absmax_codes[static_cast<size_t>(i)]] *
                nested[static_cast<size_t>(nb)] +
            nested_offset;
    }

    Context ctx;
    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                         dnnl::memory::data_type::u8);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    Tensor absmax_dev = Tensor::allocate(ctx, {block_count},
                                         dnnl::memory::data_type::f32);
    Tensor input_dev = Tensor::allocate(ctx, {M, K}, dnnl::memory::data_type::bf16);
    Tensor output_dev = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(packed_dev.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(qmap_dev.data(), qmap.data(), qmap.size() * sizeof(float));
    ctx.memcpy_h2d(absmax_dev.data(), absmax_host.data(),
                   absmax_host.size() * sizeof(float));
    std::vector<bf16> zero_input(static_cast<size_t>(M) * K, bf16(0.0f));
    ctx.memcpy_h2d(input_dev.data(), zero_input.data(),
                   zero_input.size() * sizeof(bf16));

    packed_nf4_gemm_bf16(ctx,
                         static_cast<const uint8_t*>(packed_dev.data()),
                         static_cast<const float*>(qmap_dev.data()),
                         static_cast<const float*>(absmax_dev.data()),
                         static_cast<const bf16*>(input_dev.data()),
                         static_cast<bf16*>(output_dev.data()),
                         M, N, K, Block);
    ctx.synchronize();
    std::vector<bf16> output(static_cast<size_t>(M) * N);
    ctx.memcpy_d2h(output.data(), output_dev.data(), output.size() * sizeof(bf16));
    bool bad = false;
    size_t bad_index = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        uint16_t raw = 0;
        std::memcpy(&raw, &output[i], sizeof(raw));
        if ((raw & 0x7F80u) == 0x7F80u) {
            bad = true;
            bad_index = i;
            break;
        }
    }
    if (bad) {
        std::cerr << "real-scgate kernel has NaN at " << bad_index << "\n";
        return 1;
    }
    std::cout << "real-scgate kernel passed\n";
    return 0;
}

int run_real_qproj_kernel_test() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "dg_scgate";
    if (!fs::exists(dir / "q_packed.bin")) {
        std::cout << "real-qproj skipped (fixture missing)\n";
        return 0;
    }
    constexpr int M = 24;
    constexpr int N = 4096;
    constexpr int K = 2816;
    constexpr int Block = 64;
    const int blocks_per_row = K / Block;
    auto load_vec = [&](const std::string& name) {
        std::ifstream in(dir / name, std::ios::binary);
        return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    };
    std::vector<uint8_t> packed = load_vec("q_packed.bin");
    std::vector<uint8_t> absmax_codes = load_vec("q_absmax.bin");
    std::vector<uint8_t> nested_bytes = load_vec("q_nested.bin");
    std::vector<uint8_t> nqmap_bytes = load_vec("q_nqmap.bin");
    std::vector<uint8_t> qmap_bytes = load_vec("q_qmap.bin");
    std::vector<uint8_t> input_bytes = load_vec("q_input.bin");
    float nested_offset = 0.0f;
    {
        std::ifstream off(dir / "q_offset.txt");
        if (off) off >> nested_offset;
    }
    std::vector<float> qmap(qmap_bytes.size() / 4);
    std::memcpy(qmap.data(), qmap_bytes.data(), qmap_bytes.size());
    std::vector<float> nested(nested_bytes.size() / 4);
    std::memcpy(nested.data(), nested_bytes.data(), nested_bytes.size());
    std::vector<float> nqmap(nqmap_bytes.size() / 4);
    std::memcpy(nqmap.data(), nqmap_bytes.data(), nqmap_bytes.size());
    const int64_t block_count = static_cast<int64_t>(absmax_codes.size());
    std::vector<float> absmax_host(static_cast<size_t>(block_count));
    for (int64_t i = 0; i < block_count; ++i) {
        absmax_host[static_cast<size_t>(i)] =
            nqmap[absmax_codes[static_cast<size_t>(i)]] *
                nested[static_cast<size_t>(i / 256)] +
            nested_offset;
    }
    std::vector<bf16> input(static_cast<size_t>(M) * K);
    std::memcpy(input.data(), input_bytes.data(), input_bytes.size());

    Context ctx;
    Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                         dnnl::memory::data_type::u8);
    Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
    Tensor absmax_dev = Tensor::allocate(ctx, {block_count},
                                         dnnl::memory::data_type::f32);
    Tensor input_dev = Tensor::allocate(ctx, {M, K}, dnnl::memory::data_type::bf16);
    Tensor output_dev = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(packed_dev.data(), packed.data(), packed.size());
    ctx.memcpy_h2d(qmap_dev.data(), qmap.data(), qmap.size() * sizeof(float));
    ctx.memcpy_h2d(absmax_dev.data(), absmax_host.data(),
                   absmax_host.size() * sizeof(float));
    ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));
    packed_nf4_gemm_bf16(ctx,
                         static_cast<const uint8_t*>(packed_dev.data()),
                         static_cast<const float*>(qmap_dev.data()),
                         static_cast<const float*>(absmax_dev.data()),
                         static_cast<const bf16*>(input_dev.data()),
                         static_cast<bf16*>(output_dev.data()),
                         M, N, K, Block);
    ctx.synchronize();
    std::vector<bf16> output(static_cast<size_t>(M) * N);
    ctx.memcpy_d2h(output.data(), output_dev.data(), output.size() * sizeof(bf16));
    // Reference dots across every output tile.  The former cols 0..3-only
    // check missed a DG2 miscompile where exactly that first tile fragment was
    // correct and most later q_proj columns were corrupted.
    bool failed = false;
    const std::vector<int> sampled_cols = {
        0, 1, 2, 3, 4, 5, 15, 16, 63, 64, 127, 128, 255, 256,
        511, 512, 1023, 1024, 2047, 2048, 3071, 3072, 4095};
    for (int r = 0; r < 3; ++r) {
        for (const int c : sampled_cols) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                const int block = k / Block;
                const uint8_t byte = packed[static_cast<size_t>(c * (K / 2) + k / 2)];
                const uint8_t code =
                    (k % 2 == 0) ? static_cast<uint8_t>(byte >> 4)
                                 : static_cast<uint8_t>(byte & 0x0F);
                expected += static_cast<float>(input[static_cast<size_t>(r * K + k)]) *
                            static_cast<float>(bf16(
                                qmap[code] * absmax_host[static_cast<size_t>(
                                    c * blocks_per_row + block)]));
            }
            const double actual =
                static_cast<float>(output[static_cast<size_t>(r * N + c)]);
            const double rel = std::abs(actual - expected) /
                               std::max(1.0, std::abs(expected));
            if (rel > 0.05) {
                std::cerr << "real-qproj row=" << r << " col=" << c
                          << " actual=" << actual << " expected=" << expected
                          << " rel=" << rel << "\n";
                failed = true;
            }
        }
    }
    if (failed) return 1;
    std::cout << "real-qproj kernel passed\n";
    return 0;
}

}  // namespace

int main() {
    const int failed = run_packed_nf4_gemm_layout_test(3, 128) |
                       run_packed_nf4_gemm_layout_test(96, 128) |
                       run_packed_nf4_gemm_layout_test(3, 32) |
                       run_packed_nf4_gemm_layout_test(96, 32) |
                       run_packed_nf4_gemm_large_test() |
                       run_packed_nf4_gemm_qproj_test() |
                       run_packed_nf4_gemm_shape_test(24, 4096, 2816) |
                       run_packed_nf4_gemm_shape_test(256, 2816, 2112) |
                       run_packed_nf4_gemm_shape_test(256, 2112, 2816) |
                       run_packed_nf4_gemm_shape_test(256, 1024, 2816) |
                       run_packed_nf4_gemm_shape_test(256, 4224, 2816) |
                       run_packed_nf4_gemm_shape_test(256, 2112, 2816, true) |
                       run_real_scgate_kernel_test() |
                       run_real_qproj_kernel_test();
    if (failed != 0) {
        std::cout << "AilaBnb4BitLinearKernelTests: failed\n";
        return failed;
    }
    std::cout << "AilaBnb4BitLinearKernelTests: passed\n";
    return 0;
}
