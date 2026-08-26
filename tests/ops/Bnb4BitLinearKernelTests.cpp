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

// Bitwise A/B gate for the vectorized GEMM rewrite: v2 must produce exactly
// the same output bytes as the original scalar tiled kernel on the real
// DiffusionGemma decode shapes (same bf16 rounding and k accumulation order).
int run_v1_v2_bitwise_test() {
    constexpr int Block = 64;
    const std::vector<std::tuple<int, int, int>> shapes = {
        {256, 2816, 2112},
        {256, 2112, 2816},
        {256, 1024, 2816},
        {256, 4224, 2816},
        {24, 4096, 2816},
        {256, 2816, 8192},
        {32, 128, 128},
    };
    Context ctx;
    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }
    int failed = 0;
    for (const auto& [M, N, K] : shapes) {
        const int blocks_per_row = K / Block;
        std::vector<float> absmax(static_cast<size_t>(N * blocks_per_row));
        for (int n = 0; n < N; ++n) {
            for (int b = 0; b < blocks_per_row; ++b) {
                absmax[static_cast<size_t>(n * blocks_per_row + b)] =
                    0.75f + 0.05f * static_cast<float>((n * 3 + b) % 11);
            }
        }
        std::vector<uint8_t> packed(static_cast<size_t>(N * K / 2));
        for (int n = 0; n < N; ++n) {
            for (int kh = 0; kh < K / 2; ++kh) {
                packed[static_cast<size_t>(n * (K / 2) + kh)] =
                    static_cast<uint8_t>((n * 5 + kh * 7) & 0xFF);
            }
        }
        std::vector<bf16> input(static_cast<size_t>(M * K));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = bf16(static_cast<float>((static_cast<int>(i) % 29) - 14) *
                            0.0078125f);
        }

        Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                             dnnl::memory::data_type::u8);
        Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
        Tensor absmax_dev = Tensor::allocate(ctx, {N * blocks_per_row},
                                             dnnl::memory::data_type::f32);
        Tensor input_dev = Tensor::allocate(ctx, {M, K}, dnnl::memory::data_type::bf16);
        Tensor out1_dev = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
        Tensor out2_dev = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(packed_dev.data(), packed.data(), packed.size());
        ctx.memcpy_h2d(qmap_dev.data(), qmap.data(), qmap.size() * sizeof(float));
        ctx.memcpy_h2d(absmax_dev.data(), absmax.data(), absmax.size() * sizeof(float));
        ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));

        const uint8_t* packed_ptr = static_cast<const uint8_t*>(packed_dev.data());
        const float* qmap_ptr = static_cast<const float*>(qmap_dev.data());
        const float* absmax_ptr = static_cast<const float*>(absmax_dev.data());
        const bf16* input_ptr = static_cast<const bf16*>(input_dev.data());
        bf16* out1_ptr = static_cast<bf16*>(out1_dev.data());
        bf16* out2_ptr = static_cast<bf16*>(out2_dev.data());

        packed_nf4_gemm_bf16_tiled<64, 64, 64, 4, 4, 16, 16>(
            ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr, out1_ptr,
            M, N, K, Block);
        packed_nf4_gemm_bf16_tiled_v2<32, 128, 64, 2, 8, 16, 16>(
            ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr, out2_ptr,
            M, N, K, Block);
        ctx.synchronize();

        std::vector<uint8_t> out1(static_cast<size_t>(M) * N * sizeof(bf16));
        std::vector<uint8_t> out2(static_cast<size_t>(M) * N * sizeof(bf16));
        ctx.memcpy_d2h(out1.data(), out1_dev.data(), out1.size());
        ctx.memcpy_d2h(out2.data(), out2_dev.data(), out2.size());
        ctx.synchronize();
        if (out1 != out2) {
            size_t first = 0;
            while (first < out1.size() && out1[first] == out2[first]) ++first;
            const size_t element = first / sizeof(bf16);
            const int m = static_cast<int>(element / N);
            const int n = static_cast<int>(element % N);
            uint16_t v1_raw = 0;
            uint16_t v2_raw = 0;
            std::memcpy(&v1_raw, &out1[first], 2);
            std::memcpy(&v2_raw, &out2[first], 2);
            std::cerr << "v1/v2 bitwise mismatch M=" << M << " N=" << N
                      << " K=" << K << " at (" << m << "," << n
                      << ") v1=" << v1_raw << " v2=" << v2_raw << "\n";
            ++failed;
            continue;
        }
        std::cout << "v1/v2 bitwise A/B M=" << M << " N=" << N << " K=" << K
                  << " passed\n";
    }
    return failed != 0 ? 1 : 0;
}

// Opt-in geometry sweep for the vectorized NF4 GEMM (AILA_BNB4_GEOM_BENCH=1).
// Uses real DiffusionGemma decode shapes and reports 20-sample wall medians
// per template geometry.
int bench_gemm_geometries() {
    using Runner = void (*)(Context&, const uint8_t*, const float*,
                            const float*, const bf16*, bf16*, int, int, int,
                            int);
    struct Variant {
        const char* name;
        Runner run;
    };
    // Only geometries where TILE_M*TM == BM and TILE_N*TN == BN cover the
    // full output tile; anything else silently drops output columns/rows.
    const std::vector<Variant> variants = {
        {"v2_64x64x64_tm4tn4_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<64, 64, 64, 4, 4, 16, 16>},
        {"v2_32x128x64_tm2tn8_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<32, 128, 64, 2, 8, 16, 16>},
        {"v2_32x256x64_tm2tn16_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<32, 256, 64, 2, 16, 16, 16>},
        {"v2_16x128x64_tm1tn8_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<16, 128, 64, 1, 8, 16, 16>},
        {"v2_16x256x64_tm1tn16_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<16, 256, 64, 1, 16, 16, 16>},
        {"v2_32x128x128_tm2tn8_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<32, 128, 128, 2, 8, 16, 16>},
        {"v2_32x64x64_tm2tn4_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<32, 64, 64, 2, 4, 16, 16>},
        {"v2_64x32x64_tm4tn2_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<64, 32, 64, 4, 2, 16, 16>},
        {"v2_32x32x64_tm2tn2_wg256",
         &packed_nf4_gemm_bf16_tiled_v2<32, 32, 64, 2, 2, 16, 16>},
    };
    const std::vector<std::tuple<int, int, int>> shapes = {
        {256, 2816, 2112},
        {256, 4224, 2816},
    };
    constexpr int Block = 64;
    constexpr int kWarmup = 3;
    constexpr int kSamples = 20;

    Context ctx;
    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }

    for (const auto& [M, N, K] : shapes) {
        const int blocks_per_row = K / Block;
        std::vector<float> absmax(static_cast<size_t>(N * blocks_per_row));
        for (int n = 0; n < N; ++n) {
            for (int b = 0; b < blocks_per_row; ++b) {
                absmax[static_cast<size_t>(n * blocks_per_row + b)] =
                    0.75f + 0.05f * static_cast<float>((n * 3 + b) % 11);
            }
        }
        std::vector<uint8_t> packed(static_cast<size_t>(N * K / 2));
        for (int n = 0; n < N; ++n) {
            for (int kh = 0; kh < K / 2; ++kh) {
                packed[static_cast<size_t>(n * (K / 2) + kh)] =
                    static_cast<uint8_t>((n * 5 + kh * 7) & 0xFF);
            }
        }
        std::vector<bf16> input(static_cast<size_t>(M * K));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = bf16(static_cast<float>((static_cast<int>(i) % 29) - 14) *
                            0.0078125f);
        }
        Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(packed.size())},
                                             dnnl::memory::data_type::u8);
        Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
        Tensor absmax_dev = Tensor::allocate(ctx, {N * blocks_per_row},
                                             dnnl::memory::data_type::f32);
        Tensor input_dev = Tensor::allocate(ctx, {M, K}, dnnl::memory::data_type::bf16);
        Tensor output_dev = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
        Tensor ref_dev = Tensor::allocate(ctx, {M, N}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(packed_dev.data(), packed.data(), packed.size());
        ctx.memcpy_h2d(qmap_dev.data(), qmap.data(), qmap.size() * sizeof(float));
        ctx.memcpy_h2d(absmax_dev.data(), absmax.data(), absmax.size() * sizeof(float));
        ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));

        const uint8_t* packed_ptr = static_cast<const uint8_t*>(packed_dev.data());
        const float* qmap_ptr = static_cast<const float*>(qmap_dev.data());
        const float* absmax_ptr = static_cast<const float*>(absmax_dev.data());
        const bf16* input_ptr = static_cast<const bf16*>(input_dev.data());
        bf16* output_ptr = static_cast<bf16*>(output_dev.data());
        bf16* ref_ptr = static_cast<bf16*>(ref_dev.data());

        // Reference output from the shipped v2 geometry (bitwise gate).
        packed_nf4_gemm_bf16_tiled_v2<64, 64, 64, 4, 4, 16, 16>(
            ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr, ref_ptr,
            M, N, K, Block);
        ctx.synchronize();
        std::vector<uint8_t> ref_bytes(static_cast<size_t>(M) * N * sizeof(bf16));
        ctx.memcpy_d2h(ref_bytes.data(), ref_dev.data(), ref_bytes.size());
        ctx.synchronize();

        std::cout << "[BNB4 geom] M=" << M << " N=" << N << " K=" << K << "\n";
        for (const Variant& variant : variants) {
            variant.run(ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr,
                        output_ptr, M, N, K, Block);
            ctx.synchronize();
            std::vector<uint8_t> out_bytes(static_cast<size_t>(M) * N * sizeof(bf16));
            ctx.memcpy_d2h(out_bytes.data(), output_dev.data(), out_bytes.size());
            ctx.synchronize();
            if (out_bytes != ref_bytes) {
                size_t first = 0;
                while (first < out_bytes.size() && out_bytes[first] == ref_bytes[first]) {
                    ++first;
                }
                const size_t element = first / sizeof(bf16);
                const int m = static_cast<int>(element / N);
                const int n = static_cast<int>(element % N);
                uint16_t got_raw = 0;
                uint16_t ref_raw = 0;
                std::memcpy(&got_raw, &out_bytes[first], 2);
                std::memcpy(&ref_raw, &ref_bytes[first], 2);
                std::cout << "  " << variant.name << " BITWISE_FAIL at (" << m
                          << "," << n << ") got=" << got_raw
                          << " ref=" << ref_raw << "\n";
                continue;
            }
            for (int i = 0; i < kWarmup; ++i) {
                variant.run(ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr,
                            output_ptr, M, N, K, Block);
                ctx.synchronize();
            }
            std::vector<double> samples;
            samples.reserve(kSamples);
            for (int i = 0; i < kSamples; ++i) {
                const auto start = std::chrono::steady_clock::now();
                variant.run(ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr,
                            output_ptr, M, N, K, Block);
                ctx.synchronize();
                samples.push_back(std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count());
            }
            std::sort(samples.begin(), samples.end());
            std::cout << "  " << variant.name
                      << " median_ms=" << samples[kSamples / 2]
                      << " p10_ms=" << samples[kSamples / 10]
                      << " p90_ms=" << samples[(kSamples * 9) / 10] << "\n";
        }
    }
    return 0;
}

// Device bf16 rounding probe: converts tie-boundary float values with both
// the bf16 constructor and the bit-RNE helper and reports the raw bf16 bits.
int run_device_bf16_round_probe() {
    const std::vector<uint32_t> inputs = {
        0x3F008000u, 0xBF008000u, 0x3F018000u, 0xBF018000u,
        0x3C2D8000u, 0xBC7C8001u, 0x3C2D8001u, 0x3B988001u,
    };
    Context ctx;
    Tensor in_dev = Tensor::allocate(ctx, {static_cast<int64_t>(inputs.size())},
                                     dnnl::memory::data_type::f32);
    Tensor out_dev = Tensor::allocate(ctx, {static_cast<int64_t>(inputs.size())},
                                      dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(in_dev.data(), inputs.data(), inputs.size() * sizeof(float));
    const float* in_ptr = static_cast<const float*>(in_dev.data());
    float* out_ptr = static_cast<float*>(out_dev.data());
    ctx.queue().parallel_for(sycl::range<1>(inputs.size()), [=](sycl::id<1> idx) {
        const float v = in_ptr[idx[0]];
        const bf16 via_ctor(v);
        const float via_bits = bf16_round_f32(v);
        uint16_t ctor_raw = 0;
        std::memcpy(&ctor_raw, &via_ctor, 2);
        out_ptr[idx[0] * 2] = static_cast<float>(ctor_raw);
        uint16_t bits_raw = 0;
        std::memcpy(&bits_raw, &via_bits, 2);
        out_ptr[idx[0] * 2 + 1] = static_cast<float>(bits_raw);
    });
    // GPU multiply probe: identical float operands must produce the same
    // product as the host.
    Tensor prod_in1 = Tensor::allocate(ctx, {2}, dnnl::memory::data_type::f32);
    Tensor prod_in2 = Tensor::allocate(ctx, {2}, dnnl::memory::data_type::f32);
    Tensor prod_out = Tensor::allocate(ctx, {2}, dnnl::memory::data_type::f32);
    const uint32_t a_bits = 0x3DBF0000u;  // 0.09328
    const uint32_t b_bits = 0x3DE88B64u;  // 0.113597
    const uint32_t c_bits = 0x3E180000u;  // 0.1484375
    const uint32_t d_bits = 0x3DB57943u;  // 0.08861
    const uint32_t src_bits[2] = {a_bits, c_bits};
    const uint32_t wgt_bits[2] = {b_bits, d_bits};
    ctx.memcpy_h2d(prod_in1.data(), src_bits, sizeof(src_bits));
    ctx.memcpy_h2d(prod_in2.data(), wgt_bits, sizeof(wgt_bits));
    const float* p1 = static_cast<const float*>(prod_in1.data());
    const float* p2 = static_cast<const float*>(prod_in2.data());
    float* po = static_cast<float*>(prod_out.data());
    ctx.queue().parallel_for(sycl::range<1>(2), [=](sycl::id<1> idx) {
        po[idx[0]] = p1[idx[0]] * p2[idx[0]];
    });
    ctx.synchronize();
    std::vector<float> prod(2);
    ctx.memcpy_d2h(prod.data(), prod_out.data(), prod.size() * sizeof(float));
    ctx.synchronize();
    for (int i = 0; i < 2; ++i) {
        uint32_t device_bits = 0;
        uint32_t host_bits = 0;
        std::memcpy(&device_bits, &prod[static_cast<size_t>(i)], 4);
        float a = 0.0f;
        float b = 0.0f;
        std::memcpy(&a, &src_bits[i], 4);
        std::memcpy(&b, &wgt_bits[i], 4);
        const float host_product = a * b;
        std::memcpy(&host_bits, &host_product, 4);
        std::cout << "mul-probe " << i << " host=" << std::hex << host_bits
                  << " device=" << device_bits << std::dec
                  << (host_bits == device_bits ? " MATCH" : " DIFF") << "\n";
    }
    // Minimal accumulate-kernel repro: acc=0, src=bf16(0x3DBF),
    // weight=0x3DE88B64.  Host RNE gives 0x3C2E contribution -> 0x3C2E.
    {
        Tensor acc_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16);
        Tensor src_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16);
        Tensor tok_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
        Tensor wgt_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::f32);
        const uint16_t src16 = 0x3DBF;
        const int32_t tok = 0;
        const float weight = 0.113547117f;
        const uint16_t zero16 = 0;
        ctx.memcpy_h2d(acc_dev.data(), &zero16, 2);
        ctx.memcpy_h2d(src_dev.data(), &src16, 2);
        ctx.memcpy_h2d(tok_dev.data(), &tok, 4);
        ctx.memcpy_h2d(wgt_dev.data(), &weight, 4);
        moe_accumulate_rows(
            ctx, src_dev.data_as<bf16>(), acc_dev.data_as<bf16>(),
            static_cast<int*>(tok_dev.data()),
            static_cast<float*>(wgt_dev.data()), 1, 1);
        ctx.synchronize();
        uint16_t acc_raw = 0;
        ctx.memcpy_d2h(&acc_raw, acc_dev.data(), 2);
        ctx.synchronize();
        bf16 src_value;
        std::memcpy(&src_value, &src16, 2);
        const float product = static_cast<float>(src_value) * weight;
        const bf16 contrib(product);
        uint16_t contrib_raw = 0;
        std::memcpy(&contrib_raw, &contrib, 2);
        uint16_t host_raw = 0;
        std::memcpy(&host_raw, &contrib, 2);
        std::cout << "accum-repro src=3DBF weight=0.113547117 host=" << std::hex
                  << host_raw << " device=" << acc_raw << std::dec
                  << (host_raw == acc_raw ? " MATCH" : " DIFF") << "\n";
    }
    // Direct probe of bf16_round_f32 on a computed product vs a constant.
    {
        Tensor in_dev = Tensor::allocate(ctx, {3}, dnnl::memory::data_type::f32);
        Tensor out_dev = Tensor::allocate(ctx, {3}, dnnl::memory::data_type::f32);
        const float c0 = 0.0932769775f;  // float(bf16 0x3DBF)
        const float c1 = 0.113547117f;
        const uint32_t tie_bits = 0x3C2D8000u;
        float c2 = 0.0f;
        std::memcpy(&c2, &tie_bits, 4);
        const float cvals[3] = {c0, c1, c2};
        ctx.memcpy_h2d(in_dev.data(), cvals, sizeof(cvals));
        const float* in_ptr = static_cast<const float*>(in_dev.data());
        float* out_ptr = static_cast<float*>(out_dev.data());
        ctx.queue().parallel_for(sycl::range<1>(3), [=](sycl::id<1> idx) {
            const float product = in_ptr[0] * in_ptr[1];
            const float rounded = bf16_round_f32(product);
            const float rounded_const = bf16_round_f32(in_ptr[2]);
            uint16_t r1 = 0;
            uint16_t r2 = 0;
            uint32_t rb1 = 0;
            uint32_t rb2 = 0;
            std::memcpy(&rb1, &rounded, 4);
            std::memcpy(&rb2, &rounded_const, 4);
            r1 = static_cast<uint16_t>(rb1 >> 16);
            r2 = static_cast<uint16_t>(rb2 >> 16);
            out_ptr[0] = product;
            out_ptr[1] = static_cast<float>(r1);
            out_ptr[2] = static_cast<float>(r2);
        });
        ctx.synchronize();
        std::vector<float> out(3);
        ctx.memcpy_d2h(out.data(), out_dev.data(), out.size() * sizeof(float));
        ctx.synchronize();
        uint32_t pb = 0;
        std::memcpy(&pb, &out[0], 4);
        const uint16_t r1 = static_cast<uint16_t>(out[1]);
        const uint16_t r2 = static_cast<uint16_t>(out[2]);
        uint32_t host_r1 = 0;
        {
            const float host_product = c0 * c1;
            const float host_rounded = bf16_round_f32(host_product);
            std::memcpy(&host_r1, &host_rounded, 4);
            host_r1 >>= 16;
        }
        std::cout << "roundfn-probe product=" << std::hex << pb
                  << " dev_rounded_product=" << r1
                  << " host_rounded_product=" << host_r1
                  << " dev_rounded_const=" << r2 << std::dec << "\n";
    }
    // bf16-load multiply probe: float(bf16 load) * float must round like the
    // host multiply (RNE).  A fused bf16 convert+multiply would be 1 ulp low.
    {
        Tensor s16 = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16);
        Tensor w32 = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::f32);
        Tensor o32 = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::f32);
        const uint16_t sval = 0x3DBF;
        const float wval = 0.113547117f;
        ctx.memcpy_h2d(s16.data(), &sval, 2);
        ctx.memcpy_h2d(w32.data(), &wval, 4);
        const bf16* sp = s16.data_as<bf16>();
        const float* wp = static_cast<const float*>(w32.data());
        float* op = static_cast<float*>(o32.data());
        ctx.queue().parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {
            op[idx[0]] = static_cast<float>(sp[idx[0]]) * wp[idx[0]];
        });
        ctx.synchronize();
        float out = 0.0f;
        ctx.memcpy_d2h(&out, o32.data(), 4);
        ctx.synchronize();
        uint32_t dev_bits = 0;
        std::memcpy(&dev_bits, &out, 4);
        float host_out = 0.0f;
        {
            bf16 s;
            std::memcpy(&s, &sval, 2);
            host_out = static_cast<float>(s) * wval;
        }
        uint32_t host_bits = 0;
        std::memcpy(&host_bits, &host_out, 4);
        std::cout << "bf16mul-probe host=" << std::hex << host_bits
                  << " device=" << dev_bits << std::dec
                  << (host_bits == dev_bits ? " MATCH" : " DIFF") << "\n";
    }
    ctx.synchronize();
    std::vector<float> out(static_cast<size_t>(inputs.size()) * 2);
    ctx.memcpy_d2h(out.data(), out_dev.data(), out.size() * sizeof(float));
    ctx.synchronize();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const uint16_t ctor_raw = static_cast<uint16_t>(out[i * 2]);
        const uint16_t bits_raw = static_cast<uint16_t>(out[i * 2 + 1]);
        uint32_t host_rne = 0;
        {
            float v = 0.0f;
            std::memcpy(&v, &inputs[i], 4);
            host_rne = bf16_round_f32(v);
            uint16_t rne16 = static_cast<uint16_t>(host_rne >> 16);
            std::cout << "round-probe bits=" << std::hex << inputs[i]
                      << " host_rne=" << rne16 << " dev_ctor=" << ctor_raw
                      << " dev_bits=" << bits_raw << std::dec << "\n";
        }
    }
    return 0;
}

// Bitwise A/B gate on the real DiffusionGemma weight fixtures (scgate and
// qproj). Synthetic fixtures cannot reproduce the real absmax/qmap values;
// this test is the one that guards the production token stream.
int run_real_v1_v2_bitwise_test() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "dg_scgate";
    if (!fs::exists(dir / "packed.bin") || !fs::exists(dir / "q_packed.bin")) {
        std::cout << "real v1/v2 bitwise A/B skipped (fixture missing)\n";
        return 0;
    }
    auto load_vec = [&](const std::string& name) {
        std::ifstream in(dir / name, std::ios::binary);
        return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    };
    auto load_float_vec = [&](const std::string& name) {
        const std::vector<uint8_t> bytes = load_vec(name);
        std::vector<float> values(bytes.size() / 4);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
    };
    auto decode_absmax = [&](const std::vector<uint8_t>& codes,
                             const std::vector<float>& nested,
                             const std::vector<float>& nqmap,
                             float offset) {
        std::vector<float> values(codes.size());
        for (size_t i = 0; i < codes.size(); ++i) {
            values[i] = nqmap[codes[i]] * nested[i / 256] + offset;
        }
        return values;
    };
    auto read_offset = [&](const std::string& name) {
        float value = 0.0f;
        std::ifstream off(dir / name);
        if (off) off >> value;
        return value;
    };

    Context ctx;
    int failed = 0;
    struct Fixture {
        const char* label;
        int M, N, K, Block;
        std::vector<uint8_t> packed;
        std::vector<float> qmap;
        std::vector<float> absmax;
        std::vector<bf16> input;
    };
    std::vector<Fixture> fixtures;
    {
        fixtures.push_back({
            "real-scgate", 256, 2112, 2816, 64,
            load_vec("packed.bin"),
            load_float_vec("qmap.bin"),
            decode_absmax(load_vec("absmax.bin"), load_float_vec("nested.bin"),
                          load_float_vec("nqmap.bin"), read_offset("offset.txt")),
            std::vector<bf16>(),
        });
        const std::vector<uint8_t> input_bytes = load_vec("q_input.bin");
        Fixture& fx = fixtures.back();
        fx.input.resize(input_bytes.size() / sizeof(bf16));
        std::memcpy(fx.input.data(), input_bytes.data(), input_bytes.size());
    }
    {
        fixtures.push_back({
            "real-qproj", 256, 4096, 2816, 64,
            load_vec("q_packed.bin"),
            load_float_vec("q_qmap.bin"),
            decode_absmax(load_vec("q_absmax.bin"), load_float_vec("q_nested.bin"),
                          load_float_vec("q_nqmap.bin"), read_offset("q_offset.txt")),
            std::vector<bf16>(),
        });
        const std::vector<uint8_t> input_bytes = load_vec("q_input.bin");
        Fixture& fx = fixtures.back();
        fx.input.resize(input_bytes.size() / sizeof(bf16));
        std::memcpy(fx.input.data(), input_bytes.data(), input_bytes.size());
    }

    for (const Fixture& fx : fixtures) {
        Tensor packed_dev = Tensor::allocate(ctx, {static_cast<int64_t>(fx.packed.size())},
                                             dnnl::memory::data_type::u8);
        Tensor qmap_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32);
        Tensor absmax_dev = Tensor::allocate(ctx, {static_cast<int64_t>(fx.absmax.size())},
                                             dnnl::memory::data_type::f32);
        Tensor input_dev = Tensor::allocate(ctx, {fx.M, fx.K}, dnnl::memory::data_type::bf16);
        Tensor out1_dev = Tensor::allocate(ctx, {fx.M, fx.N}, dnnl::memory::data_type::bf16);
        Tensor out2_dev = Tensor::allocate(ctx, {fx.M, fx.N}, dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(packed_dev.data(), fx.packed.data(), fx.packed.size());
        ctx.memcpy_h2d(qmap_dev.data(), fx.qmap.data(), fx.qmap.size() * sizeof(float));
        ctx.memcpy_h2d(absmax_dev.data(), fx.absmax.data(), fx.absmax.size() * sizeof(float));
        ctx.memcpy_h2d(input_dev.data(), fx.input.data(), fx.input.size() * sizeof(bf16));

        const uint8_t* packed_ptr = static_cast<const uint8_t*>(packed_dev.data());
        const float* qmap_ptr = static_cast<const float*>(qmap_dev.data());
        const float* absmax_ptr = static_cast<const float*>(absmax_dev.data());
        const bf16* input_ptr = static_cast<const bf16*>(input_dev.data());
        bf16* out1_ptr = static_cast<bf16*>(out1_dev.data());
        bf16* out2_ptr = static_cast<bf16*>(out2_dev.data());

        packed_nf4_gemm_bf16_tiled<64, 64, 64, 4, 4, 16, 16>(
            ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr, out1_ptr,
            fx.M, fx.N, fx.K, fx.Block);
        ctx.synchronize();

        std::vector<uint8_t> out1(static_cast<size_t>(fx.M) * fx.N * sizeof(bf16));
        ctx.memcpy_d2h(out1.data(), out1_dev.data(), out1.size());
        ctx.synchronize();

        using Runner = void (*)(Context&, const uint8_t*, const float*,
                                const float*, const bf16*, bf16*, int, int,
                                int, int);
        const std::vector<std::tuple<const char*, Runner, Runner>> geometries = {
            {"64x64_bk64_tm4tn4",
             &packed_nf4_gemm_bf16_tiled<64, 64, 64, 4, 4, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<64, 64, 64, 4, 4, 16, 16>},
            {"64x32_bk64_tm4tn2",
             &packed_nf4_gemm_bf16_tiled<64, 32, 64, 4, 2, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<64, 32, 64, 4, 2, 16, 16>},
            {"32x64_bk64_tm2tn4",
             &packed_nf4_gemm_bf16_tiled<32, 64, 64, 2, 4, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<32, 64, 64, 2, 4, 16, 16>},
            {"32x32_bk64_tm2tn2",
             &packed_nf4_gemm_bf16_tiled<32, 32, 64, 2, 2, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<32, 32, 64, 2, 2, 16, 16>},
            {"16x64_bk64_tm1tn4",
             &packed_nf4_gemm_bf16_tiled<16, 64, 64, 1, 4, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<16, 64, 64, 1, 4, 16, 16>},
            {"32x128_bk64_tm2tn8",
             &packed_nf4_gemm_bf16_tiled<32, 128, 64, 2, 8, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<32, 128, 64, 2, 8, 16, 16>},
            {"128x32_bk64_tm8tn2",
             &packed_nf4_gemm_bf16_tiled<128, 32, 64, 8, 2, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<128, 32, 64, 8, 2, 16, 16>},
            {"64x64_bk128_tm4tn4",
             &packed_nf4_gemm_bf16_tiled<64, 64, 128, 4, 4, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<64, 64, 128, 4, 4, 16, 16>},
            {"128x64_bk64_tm8tn4",
             &packed_nf4_gemm_bf16_tiled<128, 64, 64, 8, 4, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<128, 64, 64, 8, 4, 16, 16>},
            {"64x128_bk64_tm4tn8",
             &packed_nf4_gemm_bf16_tiled<64, 128, 64, 4, 8, 16, 16>,
             &packed_nf4_gemm_bf16_tiled_v2<64, 128, 64, 4, 8, 16, 16>},
        };
        std::cout << "  geom mismatches:";
        for (const auto& [name, scalar, vector] : geometries) {
            scalar(ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr, out2_ptr,
                   fx.M, fx.N, fx.K, fx.Block);
            ctx.synchronize();
            std::vector<uint8_t> scalar_out(static_cast<size_t>(fx.M) * fx.N * sizeof(bf16));
            ctx.memcpy_d2h(scalar_out.data(), out2_dev.data(), scalar_out.size());
            vector(ctx, packed_ptr, qmap_ptr, absmax_ptr, input_ptr, out2_ptr,
                   fx.M, fx.N, fx.K, fx.Block);
            ctx.synchronize();
            std::vector<uint8_t> vector_out(static_cast<size_t>(fx.M) * fx.N * sizeof(bf16));
            ctx.memcpy_d2h(vector_out.data(), out2_dev.data(), vector_out.size());
            ctx.synchronize();
            size_t scalar_mismatches = 0;
            size_t vector_mismatches = 0;
            for (size_t i = 0; i < out1.size(); i += 2) {
                if (out1[i] != scalar_out[i] || out1[i + 1] != scalar_out[i + 1]) {
                    ++scalar_mismatches;
                }
                if (out1[i] != vector_out[i] || out1[i + 1] != vector_out[i + 1]) {
                    ++vector_mismatches;
                }
            }
            std::cout << " " << name << "=s" << scalar_mismatches
                      << "/v" << vector_mismatches;
            if (std::string(name) == "32x128_bk64_tm2tn8" &&
                vector_mismatches != 0) {
                std::cerr << " production v2 geometry is not bitwise identical "
                             "to v1 on " << fx.label << "\n";
                ++failed;
            }
        }
        std::cout << "\n";
    }
    return failed != 0 ? 1 : 0;
}

}  // namespace

int main() {
    if (aila::env::read_flag("AILA_BNB4_GEOM_BENCH", false)) {
        return bench_gemm_geometries();
    }
    if (aila::env::read_flag("AILA_BNB4_ROUND_PROBE", false)) {
        return run_device_bf16_round_probe();
    }
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
                       run_real_qproj_kernel_test() |
                       run_v1_v2_bitwise_test() |
                       run_real_v1_v2_bitwise_test();
    if (failed != 0) {
        std::cout << "AilaBnb4BitLinearKernelTests: failed\n";
        return failed;
    }
    std::cout << "AilaBnb4BitLinearKernelTests: passed\n";
    return 0;
}
