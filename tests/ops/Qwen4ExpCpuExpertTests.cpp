#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/Qwen4ExpCpuExpertOps.hpp"
#include "ops/CpuTilePool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <future>
#include <immintrin.h>
#include <tbb/task_arena.h>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

void expect(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error("Test failure: " + msg);
    }
}

struct SyntheticExpertData {
    static constexpr int in_features = ops::cpu::kHiddenDim;             // 2560
    static constexpr int intermediate_size = ops::cpu::kIntermediateDim; // 640
    static constexpr int out_features = ops::cpu::kHiddenDim;            // 2560
    static constexpr int blocksize = ops::cpu::kBlockSize;               // 64
    static constexpr int nested_blocksize = 256;

    const size_t gu_packed_bytes = static_cast<size_t>(2 * intermediate_size) * (in_features / 2); // 1,638,400 B
    const size_t gu_absmax_count = static_cast<size_t>(2 * intermediate_size) * (in_features / blocksize); // 51,200
    const size_t down_packed_bytes = static_cast<size_t>(out_features) * (intermediate_size / 2); // 819,200 B
    const size_t down_absmax_count = static_cast<size_t>(out_features) * (intermediate_size / blocksize); // 25,600

    std::vector<uint8_t> gu_packed;
    std::vector<uint8_t> gu_codes;
    std::vector<float> gu_nested_absmax;
    std::vector<float> gu_nested_qmap;
    std::vector<float> gu_absmax_unnested;

    std::vector<uint8_t> dn_packed;
    std::vector<uint8_t> dn_codes;
    std::vector<float> dn_nested_absmax;
    std::vector<float> dn_nested_qmap;
    std::vector<float> dn_absmax_unnested;

    std::vector<float> qmap;

    ops::MoeExpertSlotHost host_slot_nested{};
    ops::MoeExpertSlotHost host_slot_unnested{};

    explicit SyntheticExpertData(unsigned seed = 1337) {
        gu_packed.resize(gu_packed_bytes);
        gu_codes.resize(gu_absmax_count);
        gu_nested_absmax.assign((gu_absmax_count + nested_blocksize - 1) / nested_blocksize, 0.05f);
        gu_nested_qmap.resize(256);
        gu_absmax_unnested.resize(gu_absmax_count);

        dn_packed.resize(down_packed_bytes);
        dn_codes.resize(down_absmax_count);
        dn_nested_absmax.assign((down_absmax_count + nested_blocksize - 1) / nested_blocksize, 0.04f);
        dn_nested_qmap.resize(256);
        dn_absmax_unnested.resize(down_absmax_count);

        qmap = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };

        for (int i = 0; i < 256; ++i) {
            gu_nested_qmap[i] = static_cast<float>(i - 128) / 128.0f;
            dn_nested_qmap[i] = static_cast<float>(i - 128) / 128.0f;
        }

        std::mt19937 rng(seed);
        for (auto& v : gu_packed) v = static_cast<uint8_t>(rng() & 0xFF);
        for (auto& v : gu_codes) v = static_cast<uint8_t>(rng() & 0xFF);
        for (auto& v : dn_packed) v = static_cast<uint8_t>(rng() & 0xFF);
        for (auto& v : dn_codes) v = static_cast<uint8_t>(rng() & 0xFF);

        for (size_t i = 0; i < gu_absmax_count; ++i) {
            float p = gu_nested_qmap[gu_codes[i]] * gu_nested_absmax[i / nested_blocksize];
            gu_absmax_unnested[i] = p + 0.02f;
        }
        for (size_t i = 0; i < down_absmax_count; ++i) {
            float p = dn_nested_qmap[dn_codes[i]] * dn_nested_absmax[i / nested_blocksize];
            dn_absmax_unnested[i] = p + 0.01f;
        }

        // Setup host nested slot
        host_slot_nested.gate_up_packed = gu_packed.data();
        host_slot_nested.down_packed = dn_packed.data();
        host_slot_nested.gate_up_nested.codes = gu_codes.data();
        host_slot_nested.gate_up_nested.nested_absmax = gu_nested_absmax.data();
        host_slot_nested.gate_up_nested.nested_quant_map = gu_nested_qmap.data();
        host_slot_nested.gate_up_nested.offset = 0.02f;
        host_slot_nested.gate_up_nested.blocksize = nested_blocksize;

        host_slot_nested.down_nested.codes = dn_codes.data();
        host_slot_nested.down_nested.nested_absmax = dn_nested_absmax.data();
        host_slot_nested.down_nested.nested_quant_map = dn_nested_qmap.data();
        host_slot_nested.down_nested.offset = 0.01f;
        host_slot_nested.down_nested.blocksize = nested_blocksize;

        // Setup host unnested slot
        host_slot_unnested.gate_up_packed = gu_packed.data();
        host_slot_unnested.down_packed = dn_packed.data();
        host_slot_unnested.gate_up_absmax = gu_absmax_unnested.data();
        host_slot_unnested.down_absmax = dn_absmax_unnested.data();
    }
};

void test_nf4_q8_batch(bool row4 = false, unsigned case_seed = 20261003, bool block64 = false) {
    constexpr int H = ops::cpu::kHiddenDim, I = ops::cpu::kIntermediateDim;
    constexpr int count = ops::cpu::kMaxExpertBatchTasks;
    const bool candidate = row4 || block64;
    const auto approximate = block64 ? ops::cpu::CpuKernelBackend::Nf4Q8Block64Avx2
                            : row4 ? ops::cpu::CpuKernelBackend::Nf4Q8Row4Avx2
                                   : ops::cpu::CpuKernelBackend::Nf4Q8Avx2;
    std::vector<std::unique_ptr<SyntheticExpertData>> experts;
    const int expert_count = candidate ? 10 : 4;
    for (int c = 0; c < expert_count; ++c)
        experts.push_back(std::make_unique<SyntheticExpertData>(1337 + c + (case_seed - 20261003) * 17));
    // Include values where nested scale contraction differs from two-step FP32.
    std::mt19937 rng(case_seed);
    std::uniform_real_distribution<float> scale_dist(0.013f, 0.087f);
    for (auto& expert : experts) {
        for (auto& value : expert->gu_nested_absmax) value = scale_dist(rng);
        for (auto& value : expert->dn_nested_absmax) value = scale_dist(rng);
        auto expand = [](const ops::MoeNestedScaleHost& nested, std::vector<float>& output) {
            for (size_t i = 0; i < output.size(); ++i) {
                volatile float product = nested.nested_quant_map[nested.codes[i]] *
                    nested.nested_absmax[i / nested.blocksize];
                output[i] = product + nested.offset;
            }
        };
        expand(expert->host_slot_nested.gate_up_nested, expert->gu_absmax_unnested);
        expand(expert->host_slot_nested.down_nested, expert->dn_absmax_unnested);
    }
    std::vector<bf16> x(4 * H), h(count * I), expected_h(count * I), flat_h(count * I);
    std::vector<float> output(count * H), expected(count * H), flat_output(count * H), reference(count * H);
    std::vector<bf16> h8_h(count * I), h8_flat_h(count * I);
    std::vector<float> h8_output(count * H), h8_flat_output(count * H);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);
    for (int t = 0; t < 4; ++t) {
        // Include small SwiGLU outputs near BF16 boundaries as well as wider
        // gates. A vector exp substitution can be invisible at ordinary scales.
        constexpr float amplitudes[4] = {0.03125f, 0.125f, 1.0f, 4.0f};
        const float amplitude = candidate ? amplitudes[t] : 1.0f;
        for (int col = 0; col < H; ++col) x[t * H + col] = bf16(amplitude * input_dist(rng));
    }
    ops::cpu::CpuExpertTask tasks[count], flat_tasks[count];
    for (int c = 0; c < count; ++c) {
        const int expert = c % expert_count, token = (c / expert_count) % 4;
        tasks[c] = {experts[expert]->host_slot_nested, token};
        flat_tasks[c] = {experts[expert]->host_slot_unnested, token};
    }
    for (int threads : {1, 4, 12}) {
        tbb::task_arena arena(threads);
        arena.execute([&] {
            ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4, h.data(), output.data(), nullptr, approximate);
            for (int c = 0; c < count; ++c) {
                auto single = tasks[c];
                single.token = 0;
                ops::cpu::cpu_expert_batch_forward(&single, 1, x.data() + tasks[c].token * H, 1,
                    expected_h.data() + c * I, expected.data() + c * H, nullptr, approximate);
            }
            ops::cpu::cpu_expert_batch_forward(flat_tasks, count, x.data(), 4,
                flat_h.data(), flat_output.data(), nullptr, approximate);
            if (candidate) {
                ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4,
                    h8_h.data(), h8_output.data(), nullptr, ops::cpu::CpuKernelBackend::Nf4Q8Avx2);
                ops::cpu::cpu_expert_batch_forward(flat_tasks, count, x.data(), 4,
                    h8_flat_h.data(), h8_flat_output.data(), nullptr, ops::cpu::CpuKernelBackend::Nf4Q8Avx2);
            }
        });
        expect(std::memcmp(h.data(), expected_h.data(), h.size() * sizeof(bf16)) == 0,
            "NF4Q8 mixed-token batch intermediate differs from single-task execution");
        expect(std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0,
            "NF4Q8 mixed-token batch output differs from single-task execution");
        expect(std::memcmp(h.data(), flat_h.data(), h.size() * sizeof(bf16)) == 0 &&
               std::memcmp(output.data(), flat_output.data(), output.size() * sizeof(float)) == 0,
            "NF4Q8 nested scales differ from expanded two-step scales");
        if (candidate) {
            expect(std::memcmp(h.data(), h8_h.data(), h.size() * sizeof(bf16)) == 0 &&
                   std::memcmp(output.data(), h8_output.data(), output.size() * sizeof(float)) == 0,
                "NF4Q8 candidate nested h/out differs bitwise from H8");
            expect(std::memcmp(flat_h.data(), h8_flat_h.data(), flat_h.size() * sizeof(bf16)) == 0 &&
                   std::memcmp(flat_output.data(), h8_flat_output.data(), flat_output.size() * sizeof(float)) == 0,
                "NF4Q8 candidate flat h/out differs bitwise from H8");
            ops::cpu::CpuExpertPhaseTimings phases;
            arena.execute([&] {
                ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4,
                    expected_h.data(), expected.data(), nullptr, approximate, &phases);
            });
            expect(std::memcmp(h.data(), expected_h.data(), h.size() * sizeof(bf16)) == 0 &&
                   std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0,
                "NF4Q8 candidate phase instrumentation changes h/out bits");
            expect(phases.gate_up_ns > 0 && phases.quantize_ns > 0 && phases.down_ns > 0,
                "NF4Q8 candidate requested phase instrumentation is empty");
            expect(phases.gate_chunks >= 0 && phases.gate_chunks <= count * (I / 16) &&
                   phases.down_chunks >= 0 && phases.down_chunks <= count * (H / 32),
                "NF4Q8 candidate coarse chunks exceed available tiles");
        }
    }
    // Compare complete nonlinear expert outputs with the existing unquantized
    // activation / FP32-dot backend. A 3% NRMSE budget catches nibble ordering,
    // sign/saturation, and block-scale errors without requiring exact arithmetic.
    tbb::task_arena arena(12);
    arena.execute([&] {
        ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4, expected_h.data(), reference.data(),
            nullptr, ops::cpu::CpuKernelBackend::GpuOrderPreparedAvx2);
    });
    double squared_error = 0.0, squared_reference = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        expect(std::isfinite(output[i]) && std::isfinite(reference[i]), "NF4Q8 full-expert output is non-finite");
        const double difference = double(output[i]) - reference[i];
        squared_error += difference * difference;
        squared_reference += double(reference[i]) * reference[i];
    }
    expect(squared_reference > 0.0, "NF4Q8 approximation fixture has no reference signal");
    const double nrmse = std::sqrt(squared_error / squared_reference);
    expect(nrmse < 0.03, "NF4Q8 full-expert NRMSE exceeds 3% synthetic error budget");
    // Exact zero must hold even with non-zero weight scales and prior outputs.
    std::fill(x.begin(), x.end(), bf16(0.0f));
    for (int threads : {1, 4, 12}) {
        std::fill(h.begin(), h.end(), bf16(123.0f));
        std::fill(output.begin(), output.end(), 123.0f);
        tbb::task_arena zero_arena(threads);
        zero_arena.execute([&] {
            ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4, h.data(), output.data(), nullptr, approximate);
            if (candidate) {
                ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4,
                    h8_h.data(), h8_output.data(), nullptr, ops::cpu::CpuKernelBackend::Nf4Q8Avx2);
                ops::cpu::cpu_expert_batch_forward(flat_tasks, count, x.data(), 4,
                    flat_h.data(), flat_output.data(), nullptr, approximate);
            }
        });
        for (auto value : h) expect(static_cast<float>(value) == 0.0f, "NF4Q8 zero activation intermediate is non-zero");
        for (auto value : output) expect(value == 0.0f, "NF4Q8 zero activation output is non-zero");
        if (candidate) {
            expect(std::memcmp(h.data(), h8_h.data(), h.size() * sizeof(bf16)) == 0 &&
                   std::memcmp(output.data(), h8_output.data(), output.size() * sizeof(float)) == 0 &&
                   std::memcmp(h.data(), flat_h.data(), h.size() * sizeof(bf16)) == 0 &&
                   std::memcmp(output.data(), flat_output.data(), output.size() * sizeof(float)) == 0,
                "NF4Q8 candidate zero h/out differs bitwise from H8 or flat scales");
        }
    }
    std::cout << "[PASS] NF4Q8" << (block64 ? " block64 H8-bitwise," : row4 ? " row4 H8-bitwise," : "")
              << " zero, mixed 4-token/max40-task batch, nested/flat scales, 1/4/12 threads, batch/single partition parity, NRMSE="
              << nrmse << " seed=" << case_seed << " (<0.03; full-model teacher KL gate remains separate)\n";
}

// Independent scalar A64 oracle. It decodes each NF4 nibble individually and
// adds eight signed integer products per lane; no packed SIMD dot operations.
int a64_reference_nearest_even(float value) {
    const float magnitude = std::abs(value);
    const int lower = static_cast<int>(std::floor(magnitude));
    const float fraction = magnitude - static_cast<float>(lower);
    const int rounded = lower + (fraction > 0.5f || (fraction == 0.5f && (lower & 1)) ? 1 : 0);
    return value < 0.0f ? -rounded : rounded;
}

bf16 a64_reference_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t rounded = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    bf16 result;
    std::memcpy(&result, &rounded, sizeof(rounded));
    return result;
}

void a64_reference_quantize(const bf16* input, int columns, std::vector<int8_t>& codes, std::vector<float>& scales) {
    codes.resize(columns); scales.resize(columns / 64);
    for (int begin = 0; begin < columns; begin += 64) {
        float maximum = 0.0f;
        for (int i = 0; i < 64; ++i) maximum = std::max(maximum, std::abs(static_cast<float>(input[begin + i])));
        scales[begin / 64] = maximum / 127.0f;
        const float inverse = maximum > 0.0f ? 127.0f / maximum : 0.0f;
        for (int i = 0; i < 64; ++i) {
            volatile float normalized = static_cast<float>(input[begin + i]) * inverse;
            codes[begin + i] = static_cast<int8_t>(std::clamp(a64_reference_nearest_even(normalized), -127, 127));
        }
    }
}

float a64_reference_dot(const uint8_t* packed, const float* absmax, const ops::MoeNestedScaleHost& nested,
                        int row, int columns, const std::vector<int8_t>& activation,
                        const std::vector<float>& activation_scales, const int8_t* lut, float lut_scale) {
    volatile float lanes[8] = {};
    for (int block = 0; block < columns / 64; ++block) {
        const size_t scale_index = size_t(row) * (columns / 64) + block;
        float scale = absmax ? absmax[scale_index] : 1.0f;
        if (nested.codes) {
            const int bs = nested.blocksize > 0 ? nested.blocksize : 256;
            volatile float product = nested.nested_quant_map[nested.codes[scale_index]] *
                nested.nested_absmax[scale_index / bs];
            scale = product + nested.offset;
        }
        volatile float first_factor = scale * activation_scales[block];
        const float factor = first_factor * lut_scale;
        for (int lane = 0; lane < 8; ++lane) {
            int integer_dot = 0;
            for (int half = 0; half < 2; ++half) {
                for (int j = 0; j < 4; ++j) {
                    const int col = block * 64 + half * 32 + lane * 4 + j;
                    const uint8_t byte = packed[size_t(row) * (columns / 2) + col / 2];
                    const int code = col % 2 == 0 ? byte >> 4 : byte & 15;
                    integer_dot += int(lut[code]) * int(activation[col]);
                }
            }
            expect(std::abs(integer_dot) <= 129032, "A64 integer lane dot exceeds its int32 contract");
            lanes[lane] = std::fma(static_cast<float>(integer_dot), factor, lanes[lane]);
        }
    }
    // Explicit scalar tree matches eight-lane hsum, with each addition rounded.
    volatile float s0 = lanes[0] + lanes[4], s1 = lanes[1] + lanes[5];
    volatile float s2 = lanes[2] + lanes[6], s3 = lanes[3] + lanes[7];
    volatile float even = s0 + s2, odd = s1 + s3;
    return even + odd;
}

void a64_reference_forward(const ops::MoeExpertSlotHost& slot, const bf16* input,
                           const float* quant_map, bf16* h, float* out) {
    constexpr int H = ops::cpu::kHiddenDim, I = ops::cpu::kIntermediateDim;
    float maximum = 0.0f;
    for (int i = 0; i < 16; ++i) maximum = std::max(maximum, std::abs(quant_map[i]));
    const float lut_scale = maximum / 127.0f;
    int8_t lut[16];
    for (int i = 0; i < 16; ++i) {
        volatile float normalized = maximum > 0.0f ? quant_map[i] * (127.0f / maximum) : 0.0f;
        lut[i] = static_cast<int8_t>(a64_reference_nearest_even(normalized));
    }
    std::vector<int8_t> activation;
    std::vector<float> activation_scales;
    a64_reference_quantize(input, H, activation, activation_scales);
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#endif
    for (int row = 0; row < I; ++row) {
        const float gate = a64_reference_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested,
            row, H, activation, activation_scales, lut, lut_scale);
        const float up = a64_reference_dot(slot.gate_up_packed, slot.gate_up_absmax, slot.gate_up_nested,
            row + I, H, activation, activation_scales, lut, lut_scale);
        const float denominator = 1.0f + std::exp(-gate);
        // Scalar division intrinsic prevents test fast-math from substituting
        // an approximate reciprocal; the independent dot oracle stays scalar.
        const float silu = _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(gate), _mm_set_ss(denominator)));
        volatile float value = silu * up;
        h[row] = a64_reference_bf16(value);
    }
    a64_reference_quantize(h, I, activation, activation_scales);
    for (int row = 0; row < H; ++row)
        out[row] = a64_reference_dot(slot.down_packed, slot.down_absmax, slot.down_nested,
            row, I, activation, activation_scales, lut, lut_scale);
}

void test_nf4_q8_activation64() {
    std::cout << "[Test] NF4Q8 A64 independent scalar/integer reference...\n";
    constexpr int H = ops::cpu::kHiddenDim, I = ops::cpu::kIntermediateDim;
    constexpr int count = ops::cpu::kMaxExpertBatchTasks;
    constexpr auto backend = ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2;
    expect(a64_reference_nearest_even(0.5f) == 0 && a64_reference_nearest_even(-0.5f) == 0 &&
        a64_reference_nearest_even(1.5f) == 2 && a64_reference_nearest_even(-2.5f) == -2 &&
        a64_reference_nearest_even(126.5f) == 126, "A64 oracle nearest-even tie contract");
    for (const auto boundary : {std::pair<float, uint16_t>{1.00390625f, 0x3f80},
                               {1.01171875f, 0x3f82}, {-1.00390625f, 0xbf80}}) {
        const bf16 value = a64_reference_bf16(boundary.first);
        uint16_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
        expect(bits == boundary.second, "A64 oracle BF16 nearest-even boundary contract");
    }
    SyntheticExpertData expert(7319);
    std::mt19937 rng(20261004);
    std::uniform_real_distribution<float> scale_dist(0.013f, 0.087f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);
    std::vector<bf16> x(4 * H), reference_h(4 * I), actual_h(count * I), flat_h(count * I);
    std::vector<float> reference_out(4 * H), actual_out(count * H), flat_out(count * H);
    constexpr float tie_values[] = {127.0f, -127.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 126.5f, -126.5f};
    for (int col = 0; col < H; ++col) {
        x[col] = bf16(tie_values[col % 10]);
        x[H + col] = bf16(0.0f);
        x[2 * H + col] = bf16(0.03125f * input_dist(rng));
        x[3 * H + col] = bf16(col % 64 < 32 ? input_dist(rng) : 4.0f * input_dist(rng));
    }
    // Each 64-block has max127 and exact quantization ties. All-positive and
    // all-negative weight rows force worst-case signed maddubs pairs; remaining
    // random rows exercise every NF4 code and lane order.
    for (int block = 0; block < H / 64; ++block) {
        for (int half = 0; half < 2; ++half) {
            for (int j = 0; j < 4; ++j) {
                x[block * 64 + half * 32 + j] = bf16(127.0f);
                x[block * 64 + half * 32 + 4 + j] = bf16(-127.0f);
            }
        }
    }
    std::fill(expert.gu_packed.begin(), expert.gu_packed.begin() + H / 2, uint8_t(0xff));
    std::fill(expert.gu_packed.begin() + H / 2, expert.gu_packed.begin() + H, uint8_t(0x00));
    for (const int bs : {256, 255}) {
        auto prepare = [&](ops::MoeNestedScaleHost& nested, std::vector<float>& scales, std::vector<float>& flat) {
            scales.resize((flat.size() + bs - 1) / bs);
            for (float& value : scales) value = scale_dist(rng);
            nested.nested_absmax = scales.data(); nested.blocksize = bs;
            for (size_t index = 0; index < flat.size(); ++index) {
                volatile float product = nested.nested_quant_map[nested.codes[index]] * scales[index / bs];
                flat[index] = product + nested.offset;
            }
        };
        prepare(expert.host_slot_nested.gate_up_nested, expert.gu_nested_absmax, expert.gu_absmax_unnested);
        prepare(expert.host_slot_nested.down_nested, expert.dn_nested_absmax, expert.dn_absmax_unnested);
        for (int token = 0; token < 4; ++token)
            a64_reference_forward(expert.host_slot_nested, x.data() + token * H, expert.qmap.data(),
                reference_h.data() + token * I, reference_out.data() + token * H);
        for (const int tokens : {1, 2, 3, 4}) {
            ops::cpu::CpuExpertTask tasks[count], flat_tasks[count];
            for (int c = 0; c < count; ++c) {
                tasks[c] = {expert.host_slot_nested, c % tokens};
                flat_tasks[c] = {expert.host_slot_unnested, c % tokens};
            }
            for (const int threads : {1, 4, 12}) {
                tbb::task_arena arena(threads);
                for (const bool shift : {false, true}) for (const bool specialize : {false, true})
                    for (const bool row2 : {false, true}) {
                    arena.execute([&] {
                        ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), tokens, actual_h.data(),
                            actual_out.data(), expert.qmap.data(), backend, nullptr, shift, specialize, row2);
                        ops::cpu::cpu_expert_batch_forward(flat_tasks, count, x.data(), tokens, flat_h.data(),
                            flat_out.data(), expert.qmap.data(), backend, nullptr, shift, specialize, row2);
                    });
                    for (int c = 0; c < count; ++c) {
                        const int token = c % tokens;
                        expect(std::memcmp(actual_h.data() + c * I, reference_h.data() + token * I, I * sizeof(bf16)) == 0,
                            "A64 BF16 h differs from scalar oracle: bs=" + std::to_string(bs) + " task=" + std::to_string(c));
                        expect(std::memcmp(actual_out.data() + c * H, reference_out.data() + token * H, H * sizeof(float)) == 0,
                            "A64 F32 out differs from scalar oracle: bs=" + std::to_string(bs) + " task=" + std::to_string(c));
                    }
                    expect(std::memcmp(actual_h.data(), flat_h.data(), actual_h.size() * sizeof(bf16)) == 0 &&
                        std::memcmp(actual_out.data(), flat_out.data(), actual_out.size() * sizeof(float)) == 0,
                        "A64 nested/flat scale arithmetic differs bitwise");
                }
            }
        }
        for (int row = 0; row < I; ++row) expect(float(reference_h[I + row]) == 0.0f, "A64 zero reference h nonzero");
        for (int row = 0; row < H; ++row) expect(reference_out[H + row] == 0.0f, "A64 zero reference out nonzero");
    }
    std::cout << "[PASS] A64 all h/out scalar-bitwise, 1..4 tokens/40 tasks, 1/4/12 threads, signed/ties/saturation/zero, bs256/255, shift/specialization/row2 independently on/off.\n";
}

void test_nf4_q8_per_expert_dag() {
    constexpr int H=ops::cpu::kHiddenDim,I=ops::cpu::kIntermediateDim,Guard=11;
    constexpr auto Backend=ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2;
    SyntheticExpertData expert0(73011),expert1(73129);
    // A non-power-of-two nested block exercises the generic scale path while
    // expert0 exercises Nested256 and flat entries exercise FlatF32.
    auto prepare_generic=[](ops::MoeNestedScaleHost& nested,std::vector<float>& scales,
                            std::vector<float>& flat) {
        nested.blocksize=255;scales.resize((flat.size()+254)/255);
        for(size_t j=0;j<scales.size();++j)scales[j]=0.021f+float(j%19)*0.0007f;
        nested.nested_absmax=scales.data();
        for(size_t j=0;j<flat.size();++j) {
            volatile float product=nested.nested_quant_map[nested.codes[j]]*scales[j/255];
            flat[j]=product+nested.offset;
        }
    };
    prepare_generic(expert1.host_slot_nested.gate_up_nested,expert1.gu_nested_absmax,expert1.gu_absmax_unnested);
    prepare_generic(expert1.host_slot_nested.down_nested,expert1.dn_nested_absmax,expert1.dn_absmax_unnested);
    std::vector<bf16> x(4*H);
    std::mt19937 rng(20261005);
    for(auto& value:x)value=bf16(float(int(rng()%2049)-1024)/256.0f);
    std::fill_n(x.data()+H,H,bf16(0.0f));
    constexpr float ties[]={127.0f,-127.0f,0.5f,-0.5f,1.5f,-2.5f,126.5f};
    for(int row=0;row<H;++row)x[row]=bf16(ties[row%7]);
    for(int block=0;block<H/64;++block)x[block*64]=bf16(127.0f);
    const auto immutableX=x;
    std::vector<ops::cpu::CpuExpertTask> tasks(40);
    for(int c=0;c<40;++c)tasks[c]={c%3==0?expert0.host_slot_nested:
        c%3==1?expert0.host_slot_unnested:expert1.host_slot_nested,(c*3+1)%4};
    struct Buffers {
        std::vector<bf16> h;
        std::vector<float> out;
        explicit Buffers(int count):h(count*I+2*Guard,bf16(-98.5f)),out(count*H+2*Guard,-9876.5f){}
    };
    auto forward=[&](Buffers& output,int count,bool dag,bool specialize,bool row2,bool shift,
                     const ops::cpu::CpuExpertDagTestHooks* hooks=nullptr,
                     ops::cpu::CpuExpertPhaseTimings* timings=nullptr) {
        ops::cpu::cpu_expert_batch_forward(tasks.data(),count,x.data(),4,output.h.data()+Guard,
            output.out.data()+Guard,nullptr,Backend,timings,shift,specialize,row2,dag,hooks);
    };
    auto matched=[](const Buffers& baseline,const Buffers& candidate) {
        expect(std::memcmp(baseline.h.data(),candidate.h.data(),baseline.h.size()*sizeof(bf16))==0,
            "per-expert DAG changed BF16 h or canaries");
        expect(std::memcmp(baseline.out.data(),candidate.out.data(),baseline.out.size()*sizeof(float))==0,
            "per-expert DAG changed FP32 down or canaries");
    };
    for(int count:{1,3,10,40})for(bool specialize:{false,true})for(bool row2:{false,true})for(bool shift:{false,true}) {
        Buffers baseline(count);
        tbb::task_arena referenceArena(1);
        referenceArena.execute([&] {forward(baseline,count,false,specialize,row2,shift);});
        for(int threads:{1,4,18}) {
            Buffers candidate(count);
            std::atomic<int> active{0};
            ops::cpu::CpuExpertDagTestHooks hooks{-1,&active};
            tbb::task_arena arena(threads);
            arena.execute([&] {forward(candidate,count,true,specialize,row2,shift,&hooks);});
            expect(active.load()==0,"per-expert DAG returned with an active descendant");
            matched(baseline,candidate);
            for(int j=0;j<Guard;++j) {
                expect(float(candidate.h[j])==-98.5f && float(candidate.h[Guard+count*I+j])==-98.5f,
                    "per-expert DAG h canary changed");
                expect(candidate.out[j]==-9876.5f && candidate.out[Guard+count*H+j]==-9876.5f,
                    "per-expert DAG output canary changed");
            }
        }
    }
    // Reentrant batches share immutable input/weights, but no counters, scratch
    // or cancellation context. Inject an asynchronous error in just one batch.
    constexpr int Count=10;
    Buffers expected(Count),good(Count),failed(Count);
    tbb::task_arena referenceArena(1);
    referenceArena.execute([&] {forward(expected,Count,false,true,true,true);});
    std::atomic<int> goodActive{0},failedActive{0};
    ops::cpu::CpuExpertDagTestHooks goodHooks{-1,&goodActive},failureHooks{79,&failedActive};
    auto survivor=std::async(std::launch::async,[&] {
        tbb::task_arena arena(4);arena.execute([&] {forward(good,Count,true,true,true,true,&goodHooks);});
    });
    auto faulted=std::async(std::launch::async,[&] {
        tbb::task_arena arena(4);arena.execute([&] {forward(failed,Count,true,true,true,true,&failureHooks);});
    });
    bool faultObserved=false;
    try {faulted.get();}catch(const std::runtime_error& error) {
        faultObserved=std::string(error.what())=="Injected CPU expert DAG gate failure";
    }
    survivor.get();
    expect(faultObserved,"per-expert DAG did not propagate descendant failure");
    expect(goodActive.load()==0 && failedActive.load()==0,"per-expert DAG exception did not drain descendants");
    matched(expected,good);
    // Reuse the failed call's buffers immediately: no old task may write them.
    tbb::task_arena recoveryArena(18);
    recoveryArena.execute([&] {forward(failed,Count,true,true,true,true,&goodHooks);});
    matched(expected,failed);
    // Serial diagnostic fields retain their original semantics and deliberately
    // select the old path, even when the DAG flag/override requests the candidate.
    ops::cpu::CpuExpertPhaseTimings phases;
    Buffers diagnostic(Count);
    recoveryArena.execute([&] {forward(diagnostic,Count,true,true,true,true,&failureHooks,&phases);});
    matched(expected,diagnostic);
    expect(phases.gate_up_ns>0 && phases.down_ns>0,"DAG diagnostic fallback lost serial phase timings");
    // The opt-in must leave every non-A64 Q8 backend on its original path.
    for(auto backend:{ops::cpu::CpuKernelBackend::Nf4Q8Avx2,
                      ops::cpu::CpuKernelBackend::Nf4Q8Row4Avx2,
                      ops::cpu::CpuKernelBackend::Nf4Q8Block64Avx2}) {
        Buffers baseline(3),candidate(3);
        auto nonA64=[&](Buffers& output,bool dag) {
            ops::cpu::cpu_expert_batch_forward(tasks.data(),3,x.data(),4,output.h.data()+Guard,
                output.out.data()+Guard,nullptr,backend,nullptr,true,true,true,dag,&failureHooks);
        };
        referenceArena.execute([&] {nonA64(baseline,false);});
        recoveryArena.execute([&] {nonA64(candidate,true);});
        matched(baseline,candidate);
    }
    expect(std::memcmp(x.data(),immutableX.data(),x.size()*sizeof(bf16))==0,"per-expert DAG changed shared inputs");
    for (int threads : {1, 4, 18}) {
        ops::cpu::CpuTilePool pool(threads);
        for (int pass = 0; pass < 100; ++pass) {
            std::array<std::atomic<int>, 93> visits{};
            const int tiles = 1 + pass % 93;
            pool.ParallelFor(tiles, [&](int first, int last) {
                for (int tile = first; tile < last; ++tile) ++visits[tile];
            });
            for (int tile = 0; tile < 93; ++tile)
                expect(visits[tile] == int(tile < tiles), "tile pool missed or repeated a tile");
        }
        bool injected = false;
        std::atomic<int> active{0};
        try {
            pool.ParallelFor(93, [&](int first, int last) {
                struct Guard { std::atomic<int>& n; Guard(std::atomic<int>& v) : n(v) { ++n; } ~Guard() { --n; } } guard(active);
                if (first <= 44 && last > 44) throw std::runtime_error("pool test fault");
            });
        } catch (const std::runtime_error& error) { injected = std::string(error.what()) == "pool test fault"; }
        expect(injected && active == 0, "tile pool did not propagate and drain failure");
        for (int count : {1, 3, 10, 40}) {
            Buffers baseline(count), candidate(count);
            referenceArena.execute([&] { forward(baseline, count, false, true, true, true); });
            ops::cpu::CpuTilePoolScope scope(&pool);
            forward(candidate, count, false, true, true, true);
            matched(baseline, candidate);
        }
    }
    std::cout << "[PASS] Persistent tile pool exact coverage, 100 generations, exception drain/reuse, 1/4/18 threads and A64 h/out bits.\n";
    std::cout<<"[PASS] Per-expert A64 DAG old-path h/out bits, flat/nested256/generic255, row2/spec/shift independently on/off, 1/3/10/40 tasks, 1/4/18 threads, canaries, reentrancy, exception drain, immediate reuse and non-A64 fallback.\n";
}

void test_nf4_q8_scale_shift() {
    std::cout << "[Test] NF4Q8 scale shift: explicit on/off BF16/F32 bitwise shadows...\n";
    constexpr int H = ops::cpu::kHiddenDim, I = ops::cpu::kIntermediateDim;
    constexpr int count = ops::cpu::kMaxExpertBatchTasks;
    SyntheticExpertData expert0(2307), expert1(7619);
    SyntheticExpertData* experts[] = {&expert0, &expert1};
    std::mt19937 rng(20261004);
    std::uniform_real_distribution<float> scale_dist(0.013f, 0.087f);
    std::uniform_real_distribution<float> map_dist(-1.0f, 1.0f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);
    std::vector<bf16> x(4 * H), baseline_h(count * I), shifted_h(count * I), flat_h(count * I);
    std::vector<float> baseline_out(count * H), shifted_out(count * H), flat_out(count * H);
    for (int token = 0; token < 4; ++token) {
        constexpr float amplitudes[4] = {0.03125f, 0.125f, 1.0f, 4.0f};
        for (int col = 0; col < H; ++col) x[token * H + col] = bf16(amplitudes[token] * input_dist(rng));
    }
    // Non-power-of-two, other powers-of-two, and nonpositive metadata (effective
    // default 256). Different gate/down sizes exercise independent slot fields.
    for (const int blocksize : {256, 255, 128, 512, 0, -7}) {
        for (int e = 0; e < 2; ++e) {
            auto& expert = *experts[e];
            const int down_blocksize = e == 0 ? blocksize : 255;
            auto prepare = [&](ops::MoeNestedScaleHost& nested, std::vector<float>& scales,
                               std::vector<float>& map, std::vector<float>& flat, int bs) {
                const size_t effective_bs = bs > 0 ? static_cast<size_t>(bs) : 256;
                scales.resize((flat.size() + effective_bs - 1) / effective_bs);
                for (float& value : scales) value = scale_dist(rng);
                for (float& value : map) value = map_dist(rng);
                nested.nested_absmax = scales.data();
                nested.nested_quant_map = map.data();
                nested.blocksize = bs;
                for (size_t index = 0; index < flat.size(); ++index) {
                    volatile float product = map[nested.codes[index]] * scales[index / effective_bs];
                    flat[index] = product + nested.offset;
                }
            };
            prepare(expert.host_slot_nested.gate_up_nested, expert.gu_nested_absmax,
                    expert.gu_nested_qmap, expert.gu_absmax_unnested, blocksize);
            prepare(expert.host_slot_nested.down_nested, expert.dn_nested_absmax,
                    expert.dn_nested_qmap, expert.dn_absmax_unnested, down_blocksize);
        }
        ops::cpu::CpuExpertTask tasks[count], flat_tasks[count];
        for (int c = 0; c < count; ++c) {
            const int e = c % 2, token = (c / 2) % 4;
            tasks[c] = {experts[e]->host_slot_nested, token};
            flat_tasks[c] = {experts[e]->host_slot_unnested, token};
        }
        for (const auto backend : {ops::cpu::CpuKernelBackend::Nf4Q8Avx2,
                                   ops::cpu::CpuKernelBackend::Nf4Q8Row4Avx2,
                                   ops::cpu::CpuKernelBackend::Nf4Q8Block64Avx2}) {
            tbb::task_arena baseline_arena(1);
            baseline_arena.execute([&] {
                // Omitted bool retains the division path for Executor shadows.
                ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4,
                    baseline_h.data(), baseline_out.data(), nullptr, backend);
            });
            for (const int threads : {1, 4, 12}) {
                tbb::task_arena arena(threads);
                arena.execute([&] {
                    ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), 4,
                        shifted_h.data(), shifted_out.data(), nullptr, backend, nullptr, true);
                    ops::cpu::cpu_expert_batch_forward(flat_tasks, count, x.data(), 4,
                        flat_h.data(), flat_out.data(), nullptr, backend, nullptr, true);
                });
                const bool matched = std::memcmp(baseline_h.data(), shifted_h.data(), baseline_h.size() * sizeof(bf16)) == 0 &&
                    std::memcmp(baseline_out.data(), shifted_out.data(), baseline_out.size() * sizeof(float)) == 0 &&
                    std::memcmp(baseline_h.data(), flat_h.data(), baseline_h.size() * sizeof(bf16)) == 0 &&
                    std::memcmp(baseline_out.data(), flat_out.data(), baseline_out.size() * sizeof(float)) == 0;
                expect(matched, "NF4Q8 scale shift changes BF16/F32 bits: blocksize=" + std::to_string(blocksize) +
                    " backend=" + std::to_string(static_cast<int>(backend)) + " threads=" + std::to_string(threads));
            }
        }
    }
    std::cout << "[PASS] NF4Q8 scale shift/default division/expanded scales, mixed40 tasks, three Q8 backends, 1/4/12 threads.\n";
}

void test_batch_scale(bool benchmark) {
    constexpr int H = ops::cpu::kHiddenDim, I = ops::cpu::kIntermediateDim;
    std::vector<std::unique_ptr<SyntheticExpertData>> experts;
    for (int c = 0; c < 40; ++c) experts.push_back(std::make_unique<SyntheticExpertData>(1337 + c));
    std::vector<bf16> x(4 * H), expected_h(40 * I), actual_h(40 * I);
    std::vector<float> expected(40 * H), actual(40 * H);
    std::mt19937 rng(20260928);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& value : x) value = bf16(dist(rng));
    ops::cpu::CpuExpertTask tasks[40];
    for (auto backend : {ops::cpu::CpuKernelBackend::TbbAvx2, ops::cpu::CpuKernelBackend::GpuOrderAvx2,
                         ops::cpu::CpuKernelBackend::GpuOrderPreparedAvx2}) {
        const auto reference = backend == ops::cpu::CpuKernelBackend::GpuOrderPreparedAvx2
            ? ops::cpu::CpuKernelBackend::GpuOrderAvx2 : backend;
        for (int count : {1, 4, 10, 40}) {
            const int tokens = count == 40 ? 4 : 1;
            for (int c = 0; c < count; ++c)
                tasks[c] = {c % 2 ? experts[c]->host_slot_nested : experts[c]->host_slot_unnested, c % tokens};
            auto sequential = [&] {
                for (int c = 0; c < count; ++c)
                    ops::cpu::cpu_expert_forward(tasks[c].slot, x.data() + tasks[c].token * H,
                        expected.data() + c * H, 1, expected_h.data() + c * I, nullptr,
                        1.0f, false, H, I, H, 64, reference);
            };
            auto batch = [&] { ops::cpu::cpu_expert_batch_forward(tasks, count, x.data(), tokens,
                actual_h.data(), actual.data(), nullptr, backend); };
            for (int threads : {4, 12}) {
                tbb::task_arena arena(threads);
                arena.execute([&] { sequential(); batch(); });
                expect(std::memcmp(expected_h.data(), actual_h.data(), count * I * sizeof(bf16)) == 0,
                    "batch intermediate differs from per-expert reference");
                expect(std::memcmp(expected.data(), actual.data(), count * H * sizeof(float)) == 0,
                    "batch output differs from per-expert reference");
                if (!benchmark) continue;
                for (int warm = 0; warm < 3; ++warm) arena.execute([&] { sequential(); batch(); });
                for (int rep = 0; rep < 3; ++rep) {
                    std::vector<double> timings[2];
                    for (int sample = 0; sample < 20; ++sample) {
                        for (int step = 0; step < 2; ++step) {
                            const int mode = (sample + step + rep) % 2;
                            const auto begin = std::chrono::steady_clock::now();
                            arena.execute([&] { if (mode) batch(); else sequential(); });
                            timings[mode].push_back(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - begin).count());
                        }
                    }
                    for (int mode = 0; mode < 2; ++mode) {
                        auto& times = timings[mode];
                        std::sort(times.begin(), times.end());
                        const double median = (times[9] + times[10]) * 0.5;
                        std::cout << std::fixed << std::setprecision(4)
                            << "[CPU scale] backend=" << static_cast<int>(backend) << " tasks=" << count
                            << " tokens=" << tokens << " threads=" << threads << " batch=" << mode
                            << " rep=" << rep << " samples=20 median_ms=" << median
                            << " p10_ms=" << times[1] << " p90_ms=" << times[17]
                            << " per_task_ms=" << median / count << '\n';
                    }
                }
            }
        }
    }
    auto reject = [&](const ops::cpu::CpuExpertTask* list, int count, int tokens) {
        bool caught = false;
        try { ops::cpu::cpu_expert_batch_forward(list, count, x.data(), tokens, actual_h.data(), actual.data()); }
        catch (const std::invalid_argument&) { caught = true; }
        expect(caught, "invalid CPU batch must be rejected before reading/writing buffers");
    };
    reject(tasks, 41, 4); reject(tasks, -1, 4); reject(tasks, 1, 5); reject(nullptr, 1, 1);
    tasks[0].token = 4; reject(tasks, 1, 4);
    ops::cpu::cpu_expert_batch_forward(nullptr, 0, nullptr, 1, nullptr, nullptr);
    std::cout << "[PASS] CPU batch 1/4/10/40 distinct experts, nested/expanded, 4/12 threads, bit-exact\n";
}

void test_nested_rounding_contract() {
    SyntheticExpertData synth;
    std::mt19937 rng(20260928);
    auto vary = [&](std::vector<float>& values, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& value : values) value = dist(rng);
    };
    vary(synth.gu_nested_absmax, 0.013f, 0.087f);
    vary(synth.dn_nested_absmax, 0.017f, 0.093f);
    vary(synth.gu_nested_qmap, -1.0f, 1.0f);
    vary(synth.dn_nested_qmap, -1.0f, 1.0f);
    size_t contraction_differences = 0;
    auto expand = [&](const ops::MoeNestedScaleHost& nested, std::vector<float>& out) {
        for (size_t i = 0; i < out.size(); ++i) {
            const float a = nested.nested_quant_map[nested.codes[i]];
            const float b = nested.nested_absmax[i / nested.blocksize];
            volatile float product = a * b;
            out[i] = product + nested.offset;
            if (out[i] != std::fma(a, b, nested.offset)) ++contraction_differences;
        }
    };
    expand(synth.host_slot_nested.gate_up_nested, synth.gu_absmax_unnested);
    expand(synth.host_slot_nested.down_nested, synth.dn_absmax_unnested);
    expect(contraction_differences > 0, "fixture must distinguish FMA from two-step scales");
    for (int tokens : {1, 4}) {
        std::vector<bf16> x(tokens * synth.in_features), h(tokens * synth.intermediate_size);
        std::vector<bf16> nested_h(h.size()), expanded_h(h.size());
        std::vector<float> nested_out(tokens * synth.out_features), expanded_out(nested_out.size());
        std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
        for (auto& value : x) value = bf16(dist(rng));
        for (auto& value : h) value = bf16(dist(rng));
        for (auto backend : {ops::cpu::CpuKernelBackend::Scalar, ops::cpu::CpuKernelBackend::Avx2,
                             ops::cpu::CpuKernelBackend::TbbAvx2, ops::cpu::CpuKernelBackend::GpuOrderAvx2}) {
            ops::cpu::cpu_expert_gate_up_swiglu(synth.host_slot_nested, x.data(), nested_h.data(), tokens,
                synth.qmap.data(), synth.in_features, synth.intermediate_size, synth.blocksize, backend);
            ops::cpu::cpu_expert_gate_up_swiglu(synth.host_slot_unnested, x.data(), expanded_h.data(), tokens,
                synth.qmap.data(), synth.in_features, synth.intermediate_size, synth.blocksize, backend);
            expect(std::memcmp(nested_h.data(), expanded_h.data(), h.size() * sizeof(bf16)) == 0,
                   "nested gate/up must equal explicitly expanded two-step scales bitwise");
            ops::cpu::cpu_expert_down(synth.host_slot_nested, h.data(), nested_out.data(), tokens, synth.qmap.data(),
                1.0f, false, synth.intermediate_size, synth.out_features, synth.blocksize, backend);
            ops::cpu::cpu_expert_down(synth.host_slot_unnested, h.data(), expanded_out.data(), tokens, synth.qmap.data(),
                1.0f, false, synth.intermediate_size, synth.out_features, synth.blocksize, backend);
            expect(std::memcmp(nested_out.data(), expanded_out.data(), nested_out.size() * sizeof(float)) == 0,
                   "nested down must equal explicitly expanded two-step scales bitwise");
        }
    }
    const auto gpu_order = ops::cpu::CpuKernelBackend::GpuOrderAvx2;
    std::vector<bf16> x(synth.in_features, bf16(0.0f)), h(synth.intermediate_size);
    std::vector<float> out(synth.out_features, 3.0f);
    ops::cpu::cpu_expert_down(synth.host_slot_nested, h.data(), out.data(), 1, nullptr, 0.0f, true,
        synth.intermediate_size, synth.out_features, synth.blocksize, gpu_order);
    expect(out.front() == 3.0f && out.back() == 3.0f, "zero-weight accumulate must be a no-op");
    ops::cpu::cpu_expert_down(synth.host_slot_nested, h.data(), out.data(), 1, nullptr, 0.0f, false,
        synth.intermediate_size, synth.out_features, synth.blocksize, gpu_order);
    expect(std::all_of(out.begin(), out.end(), [](float value) { return value == 0.0f; }), "zero weight must clear output");
    ops::cpu::cpu_expert_forward({}, nullptr, nullptr, 1, nullptr, nullptr, 1.0f, false,
        synth.in_features, synth.intermediate_size, synth.out_features, synth.blocksize, gpu_order);
    bool rejected = false;
    try {
        ops::cpu::cpu_expert_forward(synth.host_slot_nested, x.data(), out.data(), 5, h.data(), nullptr, 1.0f, false,
            synth.in_features, synth.intermediate_size, synth.out_features, synth.blocksize, gpu_order);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "GPU-order path must reject unsupported token capacity before reading buffers");
    std::cout << "[PASS] Two-step nested scales, scalar/AVX2/TBB/GPU-order, m=1/4: bit-exact\n";
}

void test_cpu_self_consistency() {
    std::cout << "\n=== [Test 1] CPU Self-Consistency: Scalar vs AVX2 vs TBB_AVX2 ===\n";
    SyntheticExpertData synth;

    const std::vector<int> test_tokens = {1, 2, 4};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> in_dist(-0.05f, 0.05f);

    for (int tokens : test_tokens) {
        std::vector<bf16> x(tokens * synth.in_features);
        for (auto& v : x) v = bf16(in_dist(rng));

        // 1. Nested scale testing: standard forward (weight=1.0, accumulate=false)
        std::vector<bf16> h_scalar(tokens * synth.intermediate_size);
        std::vector<bf16> h_avx2(tokens * synth.intermediate_size);
        std::vector<bf16> h_tbb(tokens * synth.intermediate_size);

        std::vector<float> out_scalar(tokens * synth.out_features);
        std::vector<float> out_avx2(tokens * synth.out_features);
        std::vector<float> out_tbb(tokens * synth.out_features);

        // Run Scalar
        ops::cpu::cpu_expert_gate_up_swiglu_scalar(
            synth.host_slot_nested, x.data(), h_scalar.data(), tokens, synth.qmap.data());
        ops::cpu::cpu_expert_down_scalar(
            synth.host_slot_nested, h_scalar.data(), out_scalar.data(), tokens, synth.qmap.data());

        // Run AVX2
        ops::cpu::cpu_expert_gate_up_swiglu_avx2(
            synth.host_slot_nested, x.data(), h_avx2.data(), tokens, synth.qmap.data());
        ops::cpu::cpu_expert_down_avx2(
            synth.host_slot_nested, h_avx2.data(), out_avx2.data(), tokens, synth.qmap.data());

        // Run TbbAvx2
        ops::cpu::cpu_expert_gate_up_swiglu_tbb_avx2(
            synth.host_slot_nested, x.data(), h_tbb.data(), tokens, synth.qmap.data());
        ops::cpu::cpu_expert_down_tbb_avx2(
            synth.host_slot_nested, h_tbb.data(), out_tbb.data(), tokens, synth.qmap.data());

        // Compare intermediate h
        float max_diff_h_avx2 = 0.0f;
        float max_diff_h_tbb = 0.0f;
        for (size_t i = 0; i < h_scalar.size(); ++i) {
            float s = static_cast<float>(h_scalar[i]);
            float a = static_cast<float>(h_avx2[i]);
            float t = static_cast<float>(h_tbb[i]);
            max_diff_h_avx2 = std::max(max_diff_h_avx2, std::abs(s - a));
            max_diff_h_tbb = std::max(max_diff_h_tbb, std::abs(s - t));
        }

        // Compare out
        float max_diff_out_avx2 = 0.0f;
        float max_diff_out_tbb = 0.0f;
        for (size_t i = 0; i < out_scalar.size(); ++i) {
            max_diff_out_avx2 = std::max(max_diff_out_avx2, std::abs(out_scalar[i] - out_avx2[i]));
            max_diff_out_tbb = std::max(max_diff_out_tbb, std::abs(out_scalar[i] - out_tbb[i]));
        }

        std::cout << "  [Nested m=" << tokens << "] Intermediate h max_diff: AVX2="
                  << max_diff_h_avx2 << ", TBB=" << max_diff_h_tbb << "\n";
        std::cout << "  [Nested m=" << tokens << "] Output out max_diff: AVX2="
                  << max_diff_out_avx2 << ", TBB=" << max_diff_out_tbb << "\n";

        expect(max_diff_h_avx2 < 5e-4f, "AVX2 h diff exceeded threshold");
        expect(max_diff_h_tbb < 5e-4f, "TBB h diff exceeded threshold");
        expect(max_diff_out_avx2 < 5e-4f, "AVX2 out diff exceeded threshold");
        expect(max_diff_out_tbb < 5e-4f, "TBB out diff exceeded threshold");

        // 2. Nested scale testing: weighted accumulate (weight=0.42, accumulate=true)
        std::vector<float> acc_scalar(tokens * synth.out_features, 1.25f);
        std::vector<float> acc_avx2(tokens * synth.out_features, 1.25f);
        std::vector<float> acc_tbb(tokens * synth.out_features, 1.25f);
        ops::cpu::cpu_expert_down_scalar(
            synth.host_slot_nested, h_scalar.data(), acc_scalar.data(), tokens, synth.qmap.data(),
            0.42f, true);
        ops::cpu::cpu_expert_down_avx2(
            synth.host_slot_nested, h_avx2.data(), acc_avx2.data(), tokens, synth.qmap.data(),
            0.42f, true);
        ops::cpu::cpu_expert_down_tbb_avx2(
            synth.host_slot_nested, h_tbb.data(), acc_tbb.data(), tokens, synth.qmap.data(),
            0.42f, true);

        float max_diff_acc_avx2 = 0.0f;
        float max_diff_acc_tbb = 0.0f;
        for (size_t i = 0; i < acc_scalar.size(); ++i) {
            max_diff_acc_avx2 = std::max(max_diff_acc_avx2, std::abs(acc_scalar[i] - acc_avx2[i]));
            max_diff_acc_tbb = std::max(max_diff_acc_tbb, std::abs(acc_scalar[i] - acc_tbb[i]));
        }
        std::cout << "  [Nested m=" << tokens << "] Accumulate(0.42) out max_diff: AVX2="
                  << max_diff_acc_avx2 << ", TBB=" << max_diff_acc_tbb << "\n";
        expect(max_diff_acc_avx2 < 5e-4f, "AVX2 accumulate out diff exceeded threshold");
        expect(max_diff_acc_tbb < 5e-4f, "TBB accumulate out diff exceeded threshold");

        // 3. Zero weight testing
        std::vector<float> zero_scalar(tokens * synth.out_features, 9.9f);
        std::vector<float> zero_tbb(tokens * synth.out_features, 9.9f);
        ops::cpu::cpu_expert_down_scalar(
            synth.host_slot_nested, h_scalar.data(), zero_scalar.data(), tokens, synth.qmap.data(),
            0.0f, false);
        ops::cpu::cpu_expert_down_tbb_avx2(
            synth.host_slot_nested, h_tbb.data(), zero_tbb.data(), tokens, synth.qmap.data(),
            0.0f, false);
        for (size_t i = 0; i < zero_scalar.size(); ++i) {
            expect(zero_scalar[i] == 0.0f, "Scalar weight=0 should zero output");
            expect(zero_tbb[i] == 0.0f, "TBB weight=0 should zero output");
        }

        // 4. Unnested scale testing
        std::vector<bf16> h_un_scalar(tokens * synth.intermediate_size);
        std::vector<bf16> h_un_tbb(tokens * synth.intermediate_size);
        std::vector<float> out_un_scalar(tokens * synth.out_features);
        std::vector<float> out_un_tbb(tokens * synth.out_features);

        ops::cpu::cpu_expert_gate_up_swiglu_scalar(
            synth.host_slot_unnested, x.data(), h_un_scalar.data(), tokens, synth.qmap.data());
        ops::cpu::cpu_expert_down_scalar(
            synth.host_slot_unnested, h_un_scalar.data(), out_un_scalar.data(), tokens, synth.qmap.data());

        ops::cpu::cpu_expert_gate_up_swiglu_tbb_avx2(
            synth.host_slot_unnested, x.data(), h_un_tbb.data(), tokens, synth.qmap.data());
        ops::cpu::cpu_expert_down_tbb_avx2(
            synth.host_slot_unnested, h_un_tbb.data(), out_un_tbb.data(), tokens, synth.qmap.data());

        float max_diff_un_h = 0.0f;
        for (size_t i = 0; i < h_un_scalar.size(); ++i) {
            max_diff_un_h = std::max(max_diff_un_h, std::abs(static_cast<float>(h_un_scalar[i]) - static_cast<float>(h_un_tbb[i])));
        }
        float max_diff_un_out = 0.0f;
        for (size_t i = 0; i < out_un_scalar.size(); ++i) {
            max_diff_un_out = std::max(max_diff_un_out, std::abs(out_un_scalar[i] - out_un_tbb[i]));
        }
        expect(max_diff_un_h < 5e-4f, "Unnested TBB h diff exceeded threshold");
        expect(max_diff_un_out < 5e-4f, "Unnested TBB out diff exceeded threshold");

        // Verify cpu_expert_forward matches direct gate_up + down
        std::vector<float> out_fwd(tokens * synth.out_features);
        ops::cpu::cpu_expert_forward(
            synth.host_slot_nested, x.data(), out_fwd.data(), tokens, nullptr,
            synth.qmap.data(), 1.0f, false, synth.in_features, synth.intermediate_size,
            synth.out_features, synth.blocksize, ops::cpu::CpuKernelBackend::TbbAvx2);

        float max_diff_fwd = 0.0f;
        for (size_t i = 0; i < out_fwd.size(); ++i) {
            max_diff_fwd = std::max(max_diff_fwd, std::abs(out_tbb[i] - out_fwd[i]));
        }
        expect(max_diff_fwd < 1e-6f, "cpu_expert_forward mismatch with separate calls");
    }
    std::cout << "  => PASSED: CPU Self-Consistency tests.\n";
}

void test_cpu_gpu_equivalence(Context& ctx) {
    std::cout << "\n=== [Test 2] CPU vs GPU Equivalence (Assert max_diff < 1e-3) ===\n";
    SyntheticExpertData synth;

    // Allocate GPU buffers for expert weights
    Tensor gu_packed_dev = Tensor::allocate(ctx, {synth.gu_packed_bytes}, dnnl::memory::data_type::u8);
    Tensor gu_codes_dev = Tensor::allocate(ctx, {synth.gu_absmax_count}, dnnl::memory::data_type::u8);
    Tensor gu_nested_absmax_dev = Tensor::allocate(ctx, {synth.gu_nested_absmax.size()}, dnnl::memory::data_type::f32);
    Tensor gu_nested_qmap_dev = Tensor::allocate(ctx, {synth.gu_nested_qmap.size()}, dnnl::memory::data_type::f32);
    Tensor gu_absmax_unnested_dev = Tensor::allocate(ctx, {synth.gu_absmax_unnested.size()}, dnnl::memory::data_type::f32);

    Tensor dn_packed_dev = Tensor::allocate(ctx, {synth.down_packed_bytes}, dnnl::memory::data_type::u8);
    Tensor dn_codes_dev = Tensor::allocate(ctx, {synth.dn_codes.size()}, dnnl::memory::data_type::u8);
    Tensor dn_nested_absmax_dev = Tensor::allocate(ctx, {synth.dn_nested_absmax.size()}, dnnl::memory::data_type::f32);
    Tensor dn_nested_qmap_dev = Tensor::allocate(ctx, {synth.dn_nested_qmap.size()}, dnnl::memory::data_type::f32);
    Tensor dn_absmax_unnested_dev = Tensor::allocate(ctx, {synth.dn_absmax_unnested.size()}, dnnl::memory::data_type::f32);

    Tensor qmap_dev = Tensor::allocate(ctx, {synth.qmap.size()}, dnnl::memory::data_type::f32);

    ctx.memcpy_h2d(gu_packed_dev.data(), synth.gu_packed.data(), synth.gu_packed_bytes);
    ctx.memcpy_h2d(gu_codes_dev.data(), synth.gu_codes.data(), synth.gu_absmax_count);
    ctx.memcpy_h2d(gu_nested_absmax_dev.data(), synth.gu_nested_absmax.data(), synth.gu_nested_absmax.size() * sizeof(float));
    ctx.memcpy_h2d(gu_nested_qmap_dev.data(), synth.gu_nested_qmap.data(), synth.gu_nested_qmap.size() * sizeof(float));
    ctx.memcpy_h2d(gu_absmax_unnested_dev.data(), synth.gu_absmax_unnested.data(), synth.gu_absmax_unnested.size() * sizeof(float));

    ctx.memcpy_h2d(dn_packed_dev.data(), synth.dn_packed.data(), synth.down_packed_bytes);
    ctx.memcpy_h2d(dn_codes_dev.data(), synth.dn_codes.data(), synth.dn_codes.size());
    ctx.memcpy_h2d(dn_nested_absmax_dev.data(), synth.dn_nested_absmax.data(), synth.dn_nested_absmax.size() * sizeof(float));
    ctx.memcpy_h2d(dn_nested_qmap_dev.data(), synth.dn_nested_qmap.data(), synth.dn_nested_qmap.size() * sizeof(float));
    ctx.memcpy_h2d(dn_absmax_unnested_dev.data(), synth.dn_absmax_unnested.data(), synth.dn_absmax_unnested.size() * sizeof(float));

    ctx.memcpy_h2d(qmap_dev.data(), synth.qmap.data(), synth.qmap.size() * sizeof(float));

    ops::MoeExpertSlotDev dev_slot{};
    dev_slot.gate_up_packed = static_cast<const uint8_t*>(gu_packed_dev.data());
    dev_slot.down_packed = static_cast<const uint8_t*>(dn_packed_dev.data());

    dev_slot.gate_up_nested.codes = static_cast<const uint8_t*>(gu_codes_dev.data());
    dev_slot.gate_up_nested.nested_absmax = static_cast<const float*>(gu_nested_absmax_dev.data());
    dev_slot.gate_up_nested.nested_quant_map = static_cast<const float*>(gu_nested_qmap_dev.data());
    dev_slot.gate_up_nested.offset = 0.02f;
    dev_slot.gate_up_nested.blocksize = synth.nested_blocksize;

    dev_slot.down_nested.codes = static_cast<const uint8_t*>(dn_codes_dev.data());
    dev_slot.down_nested.nested_absmax = static_cast<const float*>(dn_nested_absmax_dev.data());
    dev_slot.down_nested.nested_quant_map = static_cast<const float*>(dn_nested_qmap_dev.data());
    dev_slot.down_nested.offset = 0.01f;
    dev_slot.down_nested.blocksize = synth.nested_blocksize;

    ops::MoeExpertSlotDev dev_slot_unnested{};
    dev_slot_unnested.gate_up_packed = static_cast<const uint8_t*>(gu_packed_dev.data());
    dev_slot_unnested.down_packed = static_cast<const uint8_t*>(dn_packed_dev.data());
    dev_slot_unnested.gate_up_absmax = static_cast<const float*>(gu_absmax_unnested_dev.data());
    dev_slot_unnested.down_absmax = static_cast<const float*>(dn_absmax_unnested_dev.data());

    const std::vector<int> test_batches = {1, 2, 3, 4};
    std::mt19937 rng(999);
    // Typical normalized LLM hidden state distribution
    std::uniform_real_distribution<float> in_dist(-0.03f, 0.03f);

    for (int m : test_batches) {
        std::vector<bf16> x_host(m * synth.in_features);
        for (auto& v : x_host) v = bf16(in_dist(rng));

        Tensor x_dev = Tensor::allocate(ctx, {m, synth.in_features}, dnnl::memory::data_type::bf16);
        Tensor h_dev = Tensor::allocate(ctx, {m, synth.intermediate_size}, dnnl::memory::data_type::bf16);
        Tensor out_dev = Tensor::allocate(ctx, {m, synth.out_features}, dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));

        // 1. GPU Computation (nested, unweighted)
        std::vector<float> gpu_out_host(m * synth.out_features);
        std::vector<bf16> gpu_h_host(m * synth.intermediate_size);

        for (int t = 0; t < m; ++t) {
            const auto* x_t = static_cast<const bf16*>(x_dev.data()) + t * synth.in_features;
            auto* h_t = static_cast<bf16*>(h_dev.data()) + t * synth.intermediate_size;
            auto* out_f32_t = static_cast<float*>(out_dev.data()) + t * synth.out_features;

            // Zero out acc buffer
            std::vector<float> zero_buf(synth.out_features, 0.0f);
            ctx.memcpy_h2d(out_f32_t, zero_buf.data(), synth.out_features * sizeof(float));

            ops::moe_decode_gate_up_swiglu_grouped(
                ctx, &dev_slot, static_cast<const float*>(qmap_dev.data()),
                x_t, h_t, 1, synth.in_features, synth.intermediate_size, synth.blocksize);

            const float weight = 1.0f;
            ops::moe_decode_down_weighted_accumulate(
                ctx, &dev_slot, static_cast<const float*>(qmap_dev.data()),
                &weight, h_t, out_f32_t, nullptr, 1, synth.intermediate_size, synth.out_features, synth.blocksize);
        }
        ctx.synchronize();

        ctx.memcpy_d2h(gpu_h_host.data(), h_dev.data(), gpu_h_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(gpu_out_host.data(), out_dev.data(), gpu_out_host.size() * sizeof(float));

        // 2. CPU Computation (testing TbbAvx2)
        std::vector<bf16> cpu_h_host(m * synth.intermediate_size);
        std::vector<float> cpu_out_host(m * synth.out_features);

        ops::cpu::cpu_expert_gate_up_swiglu_tbb_avx2(
            synth.host_slot_nested, x_host.data(), cpu_h_host.data(), m, synth.qmap.data(),
            synth.in_features, synth.intermediate_size, synth.blocksize);
        ops::cpu::cpu_expert_down_tbb_avx2(
            synth.host_slot_nested, cpu_h_host.data(), cpu_out_host.data(), m, synth.qmap.data(),
            1.0f, false, synth.intermediate_size, synth.out_features, synth.blocksize);

        // 3. Precision Verification
        float max_diff_h = 0.0f;
        for (size_t i = 0; i < cpu_h_host.size(); ++i) {
            float g = static_cast<float>(gpu_h_host[i]);
            float c = static_cast<float>(cpu_h_host[i]);
            max_diff_h = std::max(max_diff_h, std::abs(g - c));
        }

        float max_diff_out = 0.0f;
        for (size_t i = 0; i < cpu_out_host.size(); ++i) {
            float g = gpu_out_host[i];
            float c = cpu_out_host[i];
            max_diff_out = std::max(max_diff_out, std::abs(g - c));
        }

        std::cout << "  [Nested m=" << m << "] Intermediate h max_diff: "
                  << max_diff_h << " (target < 1e-3)\n";
        std::cout << "  [Nested m=" << m << "] Output out max_diff: "
                  << max_diff_out << " (target < 1e-3)\n";

        expect(max_diff_h < 1e-3f, "m=" + std::to_string(m) + " intermediate h max_diff >= 1e-3");
        expect(max_diff_out < 1e-3f, "m=" + std::to_string(m) + " output out max_diff >= 1e-3");

        std::vector<float> ordered_out(gpu_out_host.size());
        ops::cpu::cpu_expert_down(synth.host_slot_nested, gpu_h_host.data(), ordered_out.data(), m,
            synth.qmap.data(), 1.0f, false, synth.intermediate_size, synth.out_features, synth.blocksize,
            ops::cpu::CpuKernelBackend::GpuOrderAvx2);
        expect(std::memcmp(gpu_out_host.data(), ordered_out.data(), ordered_out.size() * sizeof(float)) == 0,
               "GPU-order down must match GPU bitwise on the same BF16 intermediate");

        // 4. Test GPU vs CPU with fractional weight & accumulate=true
        const float test_weight = 0.35f;
        std::vector<float> gpu_acc_host(m * synth.out_features);
        std::vector<float> cpu_acc_host(m * synth.out_features);
        for (int i = 0; i < m * synth.out_features; ++i) {
            float init_v = 0.5f * static_cast<float>((i * 13) % 17) / 17.0f;
            gpu_acc_host[i] = init_v;
            cpu_acc_host[i] = init_v;
        }
        ctx.memcpy_h2d(out_dev.data(), gpu_acc_host.data(), gpu_acc_host.size() * sizeof(float));
        for (int t = 0; t < m; ++t) {
            auto* h_t = static_cast<bf16*>(h_dev.data()) + t * synth.intermediate_size;
            auto* out_f32_t = static_cast<float*>(out_dev.data()) + t * synth.out_features;
            ops::moe_decode_down_weighted_accumulate(
                ctx, &dev_slot, static_cast<const float*>(qmap_dev.data()),
                &test_weight, h_t, out_f32_t, nullptr, 1, synth.intermediate_size, synth.out_features, synth.blocksize);
        }
        ctx.synchronize();
        ctx.memcpy_d2h(gpu_acc_host.data(), out_dev.data(), gpu_acc_host.size() * sizeof(float));

        ops::cpu::cpu_expert_down_tbb_avx2(
            synth.host_slot_nested, cpu_h_host.data(), cpu_acc_host.data(), m, synth.qmap.data(),
            test_weight, true, synth.intermediate_size, synth.out_features, synth.blocksize);

        float max_diff_acc = 0.0f;
        for (size_t i = 0; i < cpu_acc_host.size(); ++i) {
            max_diff_acc = std::max(max_diff_acc, std::abs(gpu_acc_host[i] - cpu_acc_host[i]));
        }
        std::cout << "  [Nested m=" << m << "] Accumulate(0.35) out max_diff: "
                  << max_diff_acc << " (target < 1e-3)\n";
        expect(max_diff_acc < 1e-3f, "Accumulate GPU vs CPU max_diff >= 1e-3");
    }

    // 5. Unnested scale GPU vs CPU equivalence (m=1, 2)
    for (int m : {1, 2}) {
        std::vector<bf16> x_host(m * synth.in_features);
        for (auto& v : x_host) v = bf16(in_dist(rng));

        Tensor x_dev = Tensor::allocate(ctx, {m, synth.in_features}, dnnl::memory::data_type::bf16);
        Tensor h_dev = Tensor::allocate(ctx, {m, synth.intermediate_size}, dnnl::memory::data_type::bf16);
        Tensor out_dev = Tensor::allocate(ctx, {m, synth.out_features}, dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(x_dev.data(), x_host.data(), x_host.size() * sizeof(bf16));

        std::vector<float> gpu_out_host(m * synth.out_features);
        std::vector<bf16> gpu_h_host(m * synth.intermediate_size);

        for (int t = 0; t < m; ++t) {
            const auto* x_t = static_cast<const bf16*>(x_dev.data()) + t * synth.in_features;
            auto* h_t = static_cast<bf16*>(h_dev.data()) + t * synth.intermediate_size;
            auto* out_f32_t = static_cast<float*>(out_dev.data()) + t * synth.out_features;

            std::vector<float> zero_buf(synth.out_features, 0.0f);
            ctx.memcpy_h2d(out_f32_t, zero_buf.data(), synth.out_features * sizeof(float));

            ops::moe_decode_gate_up_swiglu_grouped(
                ctx, &dev_slot_unnested, static_cast<const float*>(qmap_dev.data()),
                x_t, h_t, 1, synth.in_features, synth.intermediate_size, synth.blocksize);

            const float weight = 1.0f;
            ops::moe_decode_down_weighted_accumulate(
                ctx, &dev_slot_unnested, static_cast<const float*>(qmap_dev.data()),
                &weight, h_t, out_f32_t, nullptr, 1, synth.intermediate_size, synth.out_features, synth.blocksize);
        }
        ctx.synchronize();

        ctx.memcpy_d2h(gpu_h_host.data(), h_dev.data(), gpu_h_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(gpu_out_host.data(), out_dev.data(), gpu_out_host.size() * sizeof(float));

        std::vector<bf16> cpu_h_host(m * synth.intermediate_size);
        std::vector<float> cpu_out_host(m * synth.out_features);

        ops::cpu::cpu_expert_gate_up_swiglu_tbb_avx2(
            synth.host_slot_unnested, x_host.data(), cpu_h_host.data(), m, synth.qmap.data(),
            synth.in_features, synth.intermediate_size, synth.blocksize);
        ops::cpu::cpu_expert_down_tbb_avx2(
            synth.host_slot_unnested, cpu_h_host.data(), cpu_out_host.data(), m, synth.qmap.data(),
            1.0f, false, synth.intermediate_size, synth.out_features, synth.blocksize);

        float max_diff_h = 0.0f;
        for (size_t i = 0; i < cpu_h_host.size(); ++i) {
            max_diff_h = std::max(max_diff_h, std::abs(static_cast<float>(gpu_h_host[i]) - static_cast<float>(cpu_h_host[i])));
        }
        float max_diff_out = 0.0f;
        for (size_t i = 0; i < cpu_out_host.size(); ++i) {
            max_diff_out = std::max(max_diff_out, std::abs(gpu_out_host[i] - cpu_out_host[i]));
        }

        std::cout << "  [Unnested m=" << m << "] Intermediate h max_diff: "
                  << max_diff_h << " (target < 1e-3)\n";
        std::cout << "  [Unnested m=" << m << "] Output out max_diff: "
                  << max_diff_out << " (target < 1e-3)\n";

        expect(max_diff_h < 1e-3f, "Unnested GPU vs CPU h diff >= 1e-3");
        expect(max_diff_out < 1e-3f, "Unnested GPU vs CPU out diff >= 1e-3");
    }

    std::cout << "  => PASSED: CPU vs GPU Equivalence tests (max_diff < 1e-3 confirmed).\n";
}

void test_edge_cases_and_robustness() {
    std::cout << "\n=== [Test 3] Edge Cases and Robustness Verification ===\n";
    SyntheticExpertData synth;

    // 1. Boundary: tokens <= 0 (must not crash, no-op)
    bf16 dummy_x[16]{};
    bf16 dummy_h[16]{};
    float dummy_out[16]{};
    ops::cpu::cpu_expert_gate_up_swiglu(synth.host_slot_nested, dummy_x, dummy_h, 0);
    ops::cpu::cpu_expert_down(synth.host_slot_nested, dummy_h, dummy_out, 0);
    ops::cpu::cpu_expert_forward(synth.host_slot_nested, dummy_x, dummy_out, 0);
    ops::cpu::cpu_expert_gate_up_swiglu_avx2(synth.host_slot_nested, dummy_x, dummy_h, -1);
    ops::cpu::cpu_expert_down_avx2(synth.host_slot_nested, dummy_h, dummy_out, -1);
    std::cout << "  [Passed] tokens <= 0 no-op verification\n";

    // 2. Boundary: nullptr checks (must safely no-op instead of segfaulting)
    ops::MoeExpertSlotHost null_slot{};
    ops::cpu::cpu_expert_gate_up_swiglu(null_slot, dummy_x, dummy_h, 1);
    ops::cpu::cpu_expert_down(null_slot, dummy_h, dummy_out, 1);
    ops::cpu::cpu_expert_forward(null_slot, dummy_x, dummy_out, 1);
    ops::cpu::cpu_expert_gate_up_swiglu(synth.host_slot_nested, nullptr, dummy_h, 1);
    ops::cpu::cpu_expert_gate_up_swiglu(synth.host_slot_nested, dummy_x, nullptr, 1);
    ops::cpu::cpu_expert_down(synth.host_slot_nested, nullptr, dummy_out, 1);
    ops::cpu::cpu_expert_down(synth.host_slot_nested, dummy_h, nullptr, 1);
    std::cout << "  [Passed] nullptr safely handled without crash\n";

    // 3. Robustness: Non-standard dimensions fallback to scalar
    // e.g., in_features=128, intermediate_size=64, blocksize=64
    const int custom_H = 128;
    const int custom_I = 64;
    const int custom_bs = 64;
    const size_t c_gu_bytes = static_cast<size_t>(2 * custom_I) * (custom_H / 2);
    const size_t c_gu_blocks = static_cast<size_t>(2 * custom_I) * (custom_H / custom_bs);
    const size_t c_dn_bytes = static_cast<size_t>(custom_H) * (custom_I / 2);
    const size_t c_dn_blocks = static_cast<size_t>(custom_H) * (custom_I / custom_bs);

    std::vector<uint8_t> c_gu_packed(c_gu_bytes, 0x12);
    std::vector<float> c_gu_absmax(c_gu_blocks, 0.05f);
    std::vector<uint8_t> c_dn_packed(c_dn_bytes, 0x34);
    std::vector<float> c_dn_absmax(c_dn_blocks, 0.04f);

    ops::MoeExpertSlotHost custom_slot{};
    custom_slot.gate_up_packed = c_gu_packed.data();
    custom_slot.down_packed = c_dn_packed.data();
    custom_slot.gate_up_absmax = c_gu_absmax.data();
    custom_slot.down_absmax = c_dn_absmax.data();

    std::vector<bf16> c_x(custom_H, bf16(0.01f));
    std::vector<bf16> c_h_scalar(custom_I);
    std::vector<bf16> c_h_avx2(custom_I);
    std::vector<bf16> c_h_tbb(custom_I);
    std::vector<float> c_out_scalar(custom_H);
    std::vector<float> c_out_avx2(custom_H);
    std::vector<float> c_out_tbb(custom_H);

    ops::cpu::cpu_expert_forward_scalar(custom_slot, c_x.data(), c_out_scalar.data(), 1, c_h_scalar.data(),
                                        synth.qmap.data(), 1.0f, false, custom_H, custom_I, custom_H, custom_bs);
    ops::cpu::cpu_expert_forward_avx2(custom_slot, c_x.data(), c_out_avx2.data(), 1, c_h_avx2.data(),
                                      synth.qmap.data(), 1.0f, false, custom_H, custom_I, custom_H, custom_bs);
    ops::cpu::cpu_expert_forward_tbb_avx2(custom_slot, c_x.data(), c_out_tbb.data(), 1, c_h_tbb.data(),
                                          synth.qmap.data(), 1.0f, false, custom_H, custom_I, custom_H, custom_bs);

    for (int i = 0; i < custom_H; ++i) {
        expect(c_out_scalar[i] == c_out_avx2[i], "Custom shape AVX2 fallback did not match scalar");
        expect(c_out_scalar[i] == c_out_tbb[i], "Custom shape TBB fallback did not match scalar");
    }
    std::cout << "  [Passed] Non-standard dimension fallback gracefully executes scalar and matches bitwise\n";
    std::cout << "  => PASSED: Edge Cases and Robustness tests.\n";
}

void run_microbenchmark() {
    std::cout << "\n=================================================================\n";
    std::cout << "            H2a CPU Expert Microbenchmark & Feasibility            \n";
    std::cout << "=================================================================\n";
    SyntheticExpertData synth;

    std::mt19937 rng(555);
    std::uniform_real_distribution<float> in_dist(-0.03f, 0.03f);

    auto benchmark_run = [&](int m, ops::cpu::CpuKernelBackend backend, const std::string& name) {
        std::vector<bf16> x(m * synth.in_features);
        for (auto& v : x) v = bf16(in_dist(rng));
        std::vector<bf16> h(m * synth.intermediate_size);
        std::vector<float> out(m * synth.out_features);

        // Warmup
        for (int i = 0; i < 5; ++i) {
            ops::cpu::cpu_expert_forward(
                synth.host_slot_nested, x.data(), out.data(), m, h.data(), synth.qmap.data(),
                1.0f, false, synth.in_features, synth.intermediate_size, synth.out_features,
                synth.blocksize, backend);
        }

        // Benchmark runs
        const int iters = (backend == ops::cpu::CpuKernelBackend::Scalar) ? 10 : 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            ops::cpu::cpu_expert_forward(
                synth.host_slot_nested, x.data(), out.data(), m, h.data(), synth.qmap.data(),
                1.0f, false, synth.in_features, synth.intermediate_size, synth.out_features,
                synth.blocksize, backend);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double per_token_ms = elapsed_ms / m;
        double throughput_tok_s = 1000.0 / per_token_ms;

        std::cout << "  | " << std::setw(3) << m
                  << " | " << std::setw(12) << name
                  << " | " << std::fixed << std::setprecision(4) << std::setw(9) << elapsed_ms << " ms"
                  << " | " << std::setw(9) << per_token_ms << " ms/tok"
                  << " | " << std::setprecision(1) << std::setw(9) << throughput_tok_s << " tok/s |\n";
    };

    std::cout << "  +-----+--------------+--------------+---------------+--------------+\n";
    std::cout << "  |  m  | Backend      | Latency (ms) | Time/tok (ms) | Throughput   |\n";
    std::cout << "  +-----+--------------+--------------+---------------+--------------+\n";

    benchmark_run(1, ops::cpu::CpuKernelBackend::Scalar, "Scalar");
    benchmark_run(1, ops::cpu::CpuKernelBackend::Avx2, "AVX2 (1T)");
    benchmark_run(1, ops::cpu::CpuKernelBackend::TbbAvx2, "TBB+AVX2");
    benchmark_run(1, ops::cpu::CpuKernelBackend::GpuOrderAvx2, "GPU-order");

    benchmark_run(2, ops::cpu::CpuKernelBackend::Avx2, "AVX2 (1T)");
    benchmark_run(2, ops::cpu::CpuKernelBackend::TbbAvx2, "TBB+AVX2");

    benchmark_run(4, ops::cpu::CpuKernelBackend::Avx2, "AVX2 (1T)");
    benchmark_run(4, ops::cpu::CpuKernelBackend::TbbAvx2, "TBB+AVX2");

    std::cout << "  +-----+--------------+--------------+---------------+--------------+\n";
    std::cout << "  Reference PCIe H2D Slot Transfer: ~0.39 ms (~2.53 MB at 6.5 GB/s)\n";
    std::cout << "=================================================================\n\n";
}

} // namespace

int main(int argc, char** argv) {
    std::cout << "Starting Qwen4Exp CPU Expert H2a Verification Suite...\n";
    try {
        if (argc == 2 && std::string(argv[1]) == "--nf4-q8") {
            test_nf4_q8_batch();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--nf4-q8-row4") {
            for (unsigned seed : {20261003u, 20261004u, 20261005u}) test_nf4_q8_batch(true, seed);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--nf4-q8-block64") {
            for (unsigned seed : {20261003u, 20261004u, 20261005u}) test_nf4_q8_batch(false, seed, true);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--batch-scale") {
            test_batch_scale(true);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--nf4-q8-scale-shift") {
            test_nf4_q8_scale_shift();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--nf4-q8-activation64") {
            test_nf4_q8_activation64();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--nf4-q8-dag") {
            test_nf4_q8_per_expert_dag();
            return 0;
        }
        test_batch_scale(false);
        test_nf4_q8_batch();
        for (unsigned seed : {20261003u, 20261004u, 20261005u}) test_nf4_q8_batch(true, seed);
        for (unsigned seed : {20261003u, 20261004u, 20261005u}) test_nf4_q8_batch(false, seed, true);
        test_nf4_q8_scale_shift();
        test_nf4_q8_activation64();
        test_nf4_q8_per_expert_dag();
        test_edge_cases_and_robustness();
        test_nested_rounding_contract();
        test_cpu_self_consistency();

        Context ctx;
        test_cpu_gpu_equivalence(ctx);

        run_microbenchmark();

        std::cout << "All H2a CPU Expert tests and benchmarks completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED with exception: " << e.what() << "\n";
        return 1;
    }
}
