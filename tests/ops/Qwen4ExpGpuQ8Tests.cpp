#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/Ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int H = 2560, I = 640, K = 10, Block = 64, Slots = 3;
constexpr std::array<float, 16> Nf4 = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f};
constexpr std::array<int, K> Selected = {2, 0, -1, 1, 2, -1, 0, 1, 2, 0};

void expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

// Independent ties-to-even implementation; does not depend on the host rounding
// mode, the device quantizer, native dot helpers, or the CPU expert implementation.
int rte(float value) {
    const float lo = std::floor(value), fraction = value - lo;
    const int integer = static_cast<int>(lo);
    return fraction > 0.5f || (fraction == 0.5f && integer % 2 != 0) ? integer + 1 : integer;
}
uint16_t bf16_bits(bf16 value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
bf16 from_bits(uint16_t bits) {
    bf16 value;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}
// Exact integer-rational |value|*127/maximum, rounded ties to even. Raw
// exponent checks remain effective when host/device code enables fast math.
int8_t quantized_reference(bf16 value, bf16 maximum) {
    const uint16_t v = bf16_bits(value), m = bf16_bits(maximum);
    const int ve = (v >> 7) & 255, me = (m >> 7) & 255;
    if (ve == 255 || (m & 0x7fff) == 0) return 0;
    const int shift = std::max(1, me) - std::max(1, ve);
    if (shift >= 16) return 0;
    expect(shift >= 0, "invalid independent quantizer maximum");
    const uint32_t numerator = ((v & 127) + (ve ? 128 : 0)) * 127;
    const uint32_t denominator = ((m & 127) + (me ? 128 : 0)) << shift;
    uint32_t quotient = numerator / denominator, remainder = numerator % denominator;
    if (2 * remainder > denominator || (2 * remainder == denominator && (quotient & 1))) ++quotient;
    return static_cast<int8_t>((v & 0x8000) ? -int(quotient) : int(quotient));
}
void quantize_reference(const bf16* input, int length, int8_t* output, float* scales) {
    for (int begin = 0; begin < length; begin += Block) {
        uint16_t maximumBits = 0;
        for (int j = 0; j < Block; ++j) {
            const uint16_t magnitude = bf16_bits(input[begin + j]) & 0x7fff;
            if ((magnitude & 0x7f80) != 0x7f80) maximumBits = std::max(maximumBits, magnitude);
        }
        const bf16 maximum = from_bits(maximumBits);
        const float scale = static_cast<float>(maximum) / 127.0f;
        scales[begin / Block] = scale;
        for (int j = 0; j < Block; ++j) output[begin + j] = quantized_reference(input[begin + j], maximum);
    }
}
void close(float actual, float expected, float absolute, float relative, const std::string& name) {
    uint32_t actualBits, expectedBits;
    std::memcpy(&actualBits, &actual, sizeof(actualBits));
    std::memcpy(&expectedBits, &expected, sizeof(expectedBits));
    expect((actualBits & 0x7f800000) != 0x7f800000 && (expectedBits & 0x7f800000) != 0x7f800000 &&
        std::abs(actual - expected) <= absolute + relative * std::abs(expected),
        name + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}
template<class T>
Tensor upload(Context& ctx, const std::vector<T>& values, dnnl::memory::data_type dtype) {
    const size_t elements = dtype == dnnl::memory::data_type::u8 ? values.size() * sizeof(T) : values.size();
    Tensor result = Tensor::allocate(ctx, {static_cast<int64_t>(elements)}, dtype);
    ctx.memcpy_h2d(result.data(), values.data(), values.size() * sizeof(T));
    return result;
}
template<class T>
std::vector<T> download(Context& ctx, const Tensor& tensor, size_t count) {
    std::vector<T> result(count);
    ctx.memcpy_d2h(result.data(), tensor.data(), count * sizeof(T));
    return result;
}

struct Projection {
    std::vector<uint8_t> packed, codes;
    std::vector<float> nestedMaximum, nestedMap, expanded;
    Tensor packedDevice, codesDevice, maximumDevice, mapDevice, expandedDevice;
    float offset = 0.0017f;
    static constexpr int NestedBlock = 256;
    void initialize(Context& ctx, int rows, int cols, unsigned seed) {
        std::mt19937 rng(seed);
        packed.resize(size_t(rows) * cols / 2);
        for (auto& byte : packed) byte = static_cast<uint8_t>(rng());
        // Force all nibble codes and asymmetric order into the fixture.
        for (size_t j = 0; j < 16; ++j) packed[j] = static_cast<uint8_t>((j << 4) | (15 - j));
        codes.resize(size_t(rows) * cols / Block);
        for (auto& code : codes) code = static_cast<uint8_t>(rng());
        nestedMaximum.resize((codes.size() + NestedBlock - 1) / NestedBlock);
        for (size_t j = 0; j < nestedMaximum.size(); ++j) nestedMaximum[j] = 0.001f + float((j * 13 + seed) % 19) * 0.00011f;
        nestedMap.resize(256);
        for (int j = 0; j < 256; ++j) nestedMap[j] = float(j + 1) / 257.0f;
        expanded.resize(codes.size());
        for (size_t j = 0; j < codes.size(); ++j) {
            volatile float product = nestedMap[codes[j]] * nestedMaximum[j / NestedBlock];
            expanded[j] = product + offset;
        }
        packedDevice = upload(ctx, packed, dnnl::memory::data_type::u8);
        codesDevice = upload(ctx, codes, dnnl::memory::data_type::u8);
        maximumDevice = upload(ctx, nestedMaximum, dnnl::memory::data_type::f32);
        mapDevice = upload(ctx, nestedMap, dnnl::memory::data_type::f32);
        expandedDevice = upload(ctx, expanded, dnnl::memory::data_type::f32);
    }
    ops::MoeNestedScaleDev nested() const {
        return {codesDevice.data_as<uint8_t>(), maximumDevice.data_as<float>(), mapDevice.data_as<float>(), offset, NestedBlock};
    }
    float scale(size_t index, bool useNested) const {
        if (!useNested) return expanded[index];
        volatile float product = nestedMap[codes[index]] * nestedMaximum[index / NestedBlock];
        return product + offset;
    }
    float dot(int row, int cols, const int8_t* input, const float* inputScale, bool useNested) const {
        // Scalar signed-byte products accumulated as integers for each logical
        // block. Decode HIGH nibble first, independently of packed dot_acc code.
        double sum = 0.0;
        for (int begin = 0; begin < cols; begin += Block) {
            int integerDot = 0;
            for (int j = 0; j < Block; ++j) {
                const size_t element = size_t(row) * cols + begin + j;
                const uint8_t byte = packed[element / 2];
                const int code = element % 2 == 0 ? byte >> 4 : byte & 15;
                integerDot += rte(Nf4[code] * 127.0f) * int(input[begin + j]);
            }
            const float weightScale = scale(size_t(row) * (cols / Block) + begin / Block, useNested);
            const float combinedScale = (weightScale * (1.0f / 127.0f)) * inputScale[begin / Block];
            sum += double(integerDot) * combinedScale;
        }
        return static_cast<float>(sum);
    }
};
struct Expert { Projection gateUp, down; };

struct DenseQ8Projection {
    static constexpr std::array<int8_t, 16> SignedNf4 = {
        -127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127};
    int rows = 64, cols = 0, block = Block;
    std::vector<uint8_t> packed, codes;
    std::vector<float> maximum, nestedMaximum, nestedMap;
    Tensor packedDevice, maximumDevice, mapDevice, nestedMaximumDevice, nestedMapDevice;
    Bnb4BitWeightRef reference;
    Bnb4BitLinear linear;

    void initialize(Context& ctx, int columns, bool nested, int blocksize = Block) {
        cols = columns;
        block = blocksize;
        const size_t elements = size_t(rows) * cols;
        packed.resize(elements / 2);
        std::mt19937 rng(20261007 + columns + blocksize);
        for (int code = 0; code < 16; ++code)
            expect(rte(Nf4[code] * 127.0f) == SignedNf4[code], "dense signed NF4 fixture constant mismatch");
        for (auto& value : packed) value = static_cast<uint8_t>(rng());
        // Every output row includes all sixteen codes in both asymmetric nibble
        // positions. Swapping HIGH/LOW order must change the scalar oracle.
        for (int row = 0; row < rows; ++row)
            for (int j = 0; j < 16; ++j)
                packed[size_t(row) * cols / 2 + j] = static_cast<uint8_t>((j << 4) | (15 - j));
        const size_t blocks = elements / block;
        maximum.resize(blocks);
        if (nested) {
            constexpr int NestedBlock = 256;
            codes.resize(blocks);
            nestedMaximum.resize((blocks + NestedBlock - 1) / NestedBlock);
            nestedMap.resize(256);
            // Exact binary-rational products and offset make the class's host
            // absmax expansion unambiguous under either fused or unfused math.
            for (int j = 0; j < 256; ++j) nestedMap[j] = float(j) / 256.0f;
            for (size_t j = 0; j < nestedMaximum.size(); ++j)
                nestedMaximum[j] = std::ldexp(1.0f, -int(j % 4) - 1);
            for (size_t j = 0; j < blocks; ++j) {
                codes[j] = static_cast<uint8_t>((j * 37 + j / 11) % 256);
                maximum[j] = nestedMap[codes[j]] * nestedMaximum[j / NestedBlock] + 1.0f / 256.0f;
            }
            maximumDevice = upload(ctx, codes, dnnl::memory::data_type::u8);
            nestedMaximumDevice = upload(ctx, nestedMaximum, dnnl::memory::data_type::f32);
            nestedMapDevice = upload(ctx, nestedMap, dnnl::memory::data_type::f32);
            reference.nested_absmax = &nestedMaximumDevice;
            reference.nested_quant_map = &nestedMapDevice;
            reference.quant_state.nested = true;
            reference.quant_state.nested_blocksize = NestedBlock;
            reference.quant_state.nested_offset = 1.0f / 256.0f;
        } else {
            for (size_t j = 0; j < blocks; ++j)
                maximum[j] = j % 31 == 0 ? 0.0f : std::ldexp(float(j % 13 + 1) / 16.0f, int(j % 5) - 4);
            maximumDevice = upload(ctx, maximum, dnnl::memory::data_type::f32);
        }
        packedDevice = upload(ctx, packed, dnnl::memory::data_type::u8);
        mapDevice = upload(ctx, std::vector<float>(Nf4.begin(), Nf4.end()), dnnl::memory::data_type::f32);
        reference.name = "dense_q8_scalar_oracle";
        reference.packed_weight = &packedDevice;
        reference.absmax = &maximumDevice;
        reference.quant_map = &mapDevice;
        reference.quant_state.quant_type = "nf4";
        reference.quant_state.dtype = "bfloat16";
        reference.quant_state.shape = {rows, cols};
        reference.quant_state.blocksize = block;
        std::string error;
        expect(linear.init(ctx, reference, &error), "dense Q8 fixture initialization failed: " + error);
    }

    bf16 scalar_reference(int row, const int8_t* input, const float* scales) const {
        // Independent scalar integer dot over each logical 64-element block,
        // followed by a double sum. No device dot_acc, FP32 lane traversal,
        // subgroup reduction, or original BF16-dequant GEMV is reused here.
        double sum = 0.0;
        for (int begin = 0; begin < cols; begin += Block) {
            int integerDot = 0;
            for (int j = 0; j < Block; ++j) {
                const size_t element = size_t(row) * cols + begin + j;
                const uint8_t byte = packed[element / 2];
                const int code = element % 2 == 0 ? byte >> 4 : byte & 15;
                integerDot += int(SignedNf4[code]) * int(input[begin + j]);
            }
            const float combinedScale = (maximum[size_t(row) * (cols / Block) + begin / Block] *
                (1.0f / 127.0f)) * scales[begin / Block];
            sum += double(integerDot) * double(combinedScale);
        }
        return bf16(static_cast<float>(sum));
    }
};

int ordered_bf16(bf16 value) {
    const uint16_t bits = bf16_bits(value);
    expect((bits & 0x7f80) != 0x7f80, "dense Q8 output/oracle must be finite");
    // Collapse signed zero while preserving the ordering of finite values.
    return bits & 0x8000 ? 0x8000 - int(bits & 0x7fff) : 0x8000 + int(bits);
}

void test_dense_q8_projection(Context& ctx, int columns, bool nested) {
    constexpr int Outputs = 64, Guard = 8;
    DenseQ8Projection projection;
    projection.initialize(ctx, columns, nested);
    Tensor xq = Tensor::allocate(ctx, {columns}, dnnl::memory::data_type::u8);
    Tensor xs = Tensor::allocate(ctx, {columns / Block}, dnnl::memory::data_type::f32);
    Tensor output = Tensor::allocate(ctx, {Outputs + 2 * Guard}, dnnl::memory::data_type::bf16);
    std::vector<bf16> input(columns), poison(Outputs + 2 * Guard, bf16(-33.0f));
    std::vector<int8_t> quantized(columns);
    std::vector<float> scales(columns / Block);
    std::mt19937 rng(20261008 + columns);
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    expect(projection.linear.SupportsDecodeQ8(), "dense NF4 block64 fixture not supported");
    for (int pattern = 0; pattern < 3; ++pattern) {
        constexpr float ties[] = {127.0f, -127.0f, 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
        for (int j = 0; j < columns; ++j) {
            const int within = j % Block, exponent = (j / Block) % 7 - 3;
            const float value = pattern == 0 ? 0.0f : pattern == 1 ?
                (within < 8 ? std::ldexp(ties[within], exponent) : 0.0f) :
                std::ldexp(float(int(rng() % 513) - 256) / 128.0f, exponent);
            input[j] = bf16(value);
        }
        quantize_reference(input.data(), columns, quantized.data(), scales.data());
        if (pattern == 1) {
            constexpr int8_t expectedTies[] = {127, -127, 0, 2, 2, 0, -2, -2};
            for (int begin = 0; begin < columns; begin += Block)
                for (int j = 0; j < 8; ++j)
                    expect(quantized[begin + j] == expectedTies[j], "dense Q8 exact-rational tie fixture failed");
        }
        ctx.memcpy_h2d(xq.data(), quantized.data(), quantized.size());
        ctx.memcpy_h2d(xs.data(), scales.data(), scales.size() * sizeof(float));
        ctx.memcpy_h2d(output.data(), poison.data(), poison.size() * sizeof(bf16));
        expect(projection.linear.TryForwardDecodeQ8(ctx, xq.data_as<int8_t>(), xs.data_as<float>(),
            output.data_as<bf16>() + Guard), "supported dense Q8 projection declined submission");
        ctx.synchronize();
        const auto actual = download<bf16>(ctx, output, poison.size());
        expect(download<int8_t>(ctx, xq, quantized.size()) == quantized, "dense Q8 changed quantized input");
        const auto scalesAfter = download<float>(ctx, xs, scales.size());
        expect(std::memcmp(scalesAfter.data(), scales.data(), scales.size() * sizeof(float)) == 0,
            "dense Q8 changed activation scales");
        int maximumUlp = 0;
        for (int row = 0; row < Outputs; ++row) {
            const bf16 expected = projection.scalar_reference(row, quantized.data(), scales.data());
            const int ulp = std::abs(ordered_bf16(actual[Guard + row]) - ordered_bf16(expected));
            maximumUlp = std::max(maximumUlp, ulp);
            expect(ulp <= 1, "dense Q8 scalar blockdot exceeds one BF16 ULP: K=" + std::to_string(columns) +
                " pattern=" + std::to_string(pattern) + " row=" + std::to_string(row) + " ulp=" + std::to_string(ulp));
        }
        for (int j = 0; j < Guard; ++j) {
            expect(bf16_bits(actual[j]) == bf16_bits(poison[j]), "dense Q8 overwrote output prefix canary");
            expect(bf16_bits(actual[Guard + Outputs + j]) == bf16_bits(poison[Guard + Outputs + j]),
                "dense Q8 overwrote output suffix canary");
        }
        std::cout << "[PASS] Dense NF4Q8 scalar blockdot K=" << columns << " outputs=" << Outputs
            << " nested=" << nested << " pattern=" << pattern << " max_bf16_ulp=" << maximumUlp << '\n';
    }

    // A valid adapter attachment must disable this approximate route. The
    // original BF16 projection remains responsible for executing adapters.
    std::vector<LoraAttachment> adapters(1);
    adapters[0].lora_a = Tensor::allocate(ctx, {1, columns}, dnnl::memory::data_type::bf16);
    adapters[0].lora_b = Tensor::allocate(ctx, {Outputs, 1}, dnnl::memory::data_type::bf16);
    adapters[0].scaling = 1.0f;
    adapters[0].output_offset = 0;
    adapters[0].output_rows = Outputs;
    projection.linear.set_lora(std::move(adapters));
    expect(!projection.linear.SupportsDecodeQ8(), "dense Q8 incorrectly supports LoRA");
    ctx.memcpy_h2d(output.data(), poison.data(), poison.size() * sizeof(bf16));
    expect(!projection.linear.TryForwardDecodeQ8(ctx, xq.data_as<int8_t>(), xs.data_as<float>(),
        output.data_as<bf16>() + Guard), "dense Q8 submitted despite LoRA");
    expect(download<bf16>(ctx, output, poison.size()) == poison, "declined LoRA Q8 route changed output");
    std::cout << "[PASS] Dense NF4Q8 LoRA fallback K=" << columns << " nested=" << nested << '\n';
}

void test_dense_q8_rejections(Context& ctx) {
    Tensor xq = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::u8);
    Tensor xs = Tensor::allocate(ctx, {H / Block}, dnnl::memory::data_type::f32);
    Tensor output = Tensor::allocate(ctx, {64}, dnnl::memory::data_type::bf16);
    std::vector<bf16> poison(64, bf16(-33.0f));
    DenseQ8Projection block32, inputTail, supported;
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    block32.initialize(ctx, H, false, 32);
    // 2552 is aligned for BF16 vector loads and yields an integral total
    // block count across 64 rows, while its individual rows fail K%64.
    inputTail.initialize(ctx, H - 8, false);
    supported.initialize(ctx, H, false);
    for (Bnb4BitLinear* projection : {&block32.linear, &inputTail.linear}) {
        expect(!projection->SupportsDecodeQ8(), "dense Q8 supports invalid block/row shape");
        ctx.memcpy_h2d(output.data(), poison.data(), poison.size() * sizeof(bf16));
        expect(!projection->TryForwardDecodeQ8(ctx, xq.data_as<int8_t>(), xs.data_as<float>(), output.data_as<bf16>()),
            "dense Q8 submitted unsupported block/row shape");
        expect(download<bf16>(ctx, output, poison.size()) == poison, "declined shape Q8 route changed output");
    }
    expect(supported.linear.SupportsDecodeQ8(), "dense Q8 supported rejection-control fixture declined");
    ctx.memcpy_h2d(output.data(), poison.data(), poison.size() * sizeof(bf16));
    expect(!supported.linear.TryForwardDecodeQ8(ctx, nullptr, xs.data_as<float>(), output.data_as<bf16>()),
        "dense Q8 accepted null input");
    expect(!supported.linear.TryForwardDecodeQ8(ctx, xq.data_as<int8_t>(), nullptr, output.data_as<bf16>()),
        "dense Q8 accepted null scale");
    expect(!supported.linear.TryForwardDecodeQ8(ctx, xq.data_as<int8_t>(), xs.data_as<float>(), nullptr),
        "dense Q8 accepted null output");
    expect(download<bf16>(ctx, output, poison.size()) == poison, "declined null-buffer Q8 route changed output");
    std::cout << "[PASS] Dense NF4Q8 block32/input-tail/null-buffer rejection\n";
}

void test_history_projection_fusion(Context& ctx) {
    constexpr int Columns = 6144, Outputs = 64, Channels = 10240, Guard = 8;
    for (bool nested : {false, true}) {
        DenseQ8Projection projection;
        Bnb4BitLinearScratch scratch;
        Tensor input = Tensor::allocate(ctx, {1, Columns}, dnnl::memory::data_type::bf16);
        Tensor reference = Tensor::allocate(ctx, {1, Outputs}, dnnl::memory::data_type::bf16);
        Tensor candidate = Tensor::allocate(ctx, {1, Outputs}, dnnl::memory::data_type::bf16);
        Tensor qkv = Tensor::allocate(ctx, {Channels}, dnnl::memory::data_type::bf16);
        Tensor ring = Tensor::allocate(ctx, {3 * Channels + 2 * Guard}, dnnl::memory::data_type::f32);
        struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
        projection.initialize(ctx, Columns, nested);
        std::mt19937 rng(20261012);
        std::vector<bf16> x(Columns), historyInput(Channels);
        std::vector<float> expected(3 * Channels + 2 * Guard, -9876.5f);
        ctx.memcpy_h2d(ring.data(), expected.data(), expected.size() * sizeof(float));
        for (int step = 0; step < 7; ++step) {
            for (auto& value : x) value = bf16(float(int(rng() % 1025) - 512) / 256.0f);
            for (auto& value : historyInput) value = bf16(float(int(rng() % 2049) - 1024) / 512.0f);
            ctx.memcpy_h2d(input.data(), x.data(), x.size() * sizeof(bf16));
            ctx.memcpy_h2d(qkv.data(), historyInput.data(), historyInput.size() * sizeof(bf16));
            const auto allocations = ctx.device_allocation_count();
            projection.linear.forward(ctx, scratch, input, reference, 1);
            expect(projection.linear.TryForwardDecodeCommitHistory(ctx, input, candidate,
                qkv.data_as<bf16>(), ring.data_as<float>() + Guard, Channels, step % 3),
                "history fusion rejected packed fixture");
            ctx.synchronize();
            expect(ctx.device_allocation_count() == allocations, "history fusion allocated hot USM");
            const auto ref = download<bf16>(ctx, reference, Outputs), actual = download<bf16>(ctx, candidate, Outputs);
            expect(std::memcmp(ref.data(), actual.data(), Outputs * sizeof(bf16)) == 0,
                "history fusion changed original packed projection bits");
            for (int j = 0; j < Channels; ++j)
                expected[Guard + (step % 3) * Channels + j] = float(historyInput[j]);
            const auto actualHistory = download<float>(ctx, ring, expected.size());
            expect(std::memcmp(actualHistory.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "history fusion ring wrap/canary mismatch");
            const auto unchanged = download<bf16>(ctx, qkv, Channels);
            expect(std::memcmp(unchanged.data(), historyInput.data(), Channels * sizeof(bf16)) == 0,
                "history fusion changed input");
        }
        expect(!projection.linear.TryForwardDecodeCommitHistory(ctx, input, candidate,
            nullptr, ring.data_as<float>(), Channels, 0), "history fusion accepted null source");
        expect(!projection.linear.TryForwardDecodeCommitHistory(ctx, input, candidate,
            qkv.data_as<bf16>(), nullptr, Channels, 0), "history fusion accepted null destination");
        std::cout << "[PASS] History/projection fusion original bits, 7 ring steps, guards, no hot USM nested=" << nested << '\n';
    }
}

void test_cpu_host_ordered_merge(Context& ctx) {
    constexpr int Ranks = 16, Guard = 17;
    constexpr float HostCanary = -4321.25f;
    std::vector<float> rankValues(size_t(Ranks) * H), weights(Ranks);
    std::vector<bf16> shared(H), gate(1, bf16(0.375f));
    for(int rank=0;rank<Ranks;++rank) {
        weights[rank]=float(rank+1)/128.0f;
        for(int row=0;row<H;++row)
            rankValues[size_t(rank)*H+row]=float(((row*17+rank*23)%1023)-511)/32.0f;
    }
    for(int row=0;row<H;++row)shared[row]=bf16(float((row%41)-20)/16.0f);
    Tensor down=upload(ctx,rankValues,dnnl::memory::data_type::f32);
    Tensor weight=upload(ctx,weights,dnnl::memory::data_type::f32);
    Tensor sharedDevice=upload(ctx,shared,dnnl::memory::data_type::bf16);
    Tensor gateDevice=upload(ctx,gate,dnnl::memory::data_type::bf16);
    std::vector<bf16> outputPoison(H+2*Guard,from_bits(0xc5e7));
    Tensor reference=upload(ctx,outputPoison,dnnl::memory::data_type::bf16);
    Tensor candidate=upload(ctx,outputPoison,dnnl::memory::data_type::bf16);
    auto* host=static_cast<float*>(ctx.alloc_host((size_t(Ranks)*H+2*Guard)*sizeof(float)));
    // Drain before freeing even if a comparison or kernel submission throws.
    struct HostArena {
        Context& ctx; float* ptr;
        ~HostArena() { try {ctx.synchronize();} catch(...) {} ctx.free_host(ptr); }
    } hostArena{ctx,host};
    sycl::event previousReader{};
    // Reuse one pinned arena across all-cold, mixed and reverse-packed masks.
    for(int fixture=0;fixture<3;++fixture)for(bool withShared:{false,true}) {
        if(previousReader!=sycl::event{})previousReader.wait_and_throw();
        std::fill_n(host,size_t(Ranks)*H+2*Guard,HostCanary);
        ops::MoeCpuHostDownRows cpu;
        cpu.values=host+Guard;cpu.rank_to_row.fill(-1);
        for(int rank=0;rank<Ranks;++rank) {
            const bool cold=fixture==0 || (fixture==1 ? rank%3!=1 : rank%2==1);
            if(cold)cpu.rank_to_row[rank]=cpu.rows++;
        }
        if(fixture==2)for(auto& row:cpu.rank_to_row)if(row>=0)row=cpu.rows-1-row;
        std::vector<float> masked=rankValues;
        for(int rank=0;rank<Ranks;++rank)if(cpu.rank_to_row[rank]>=0) {
            std::copy_n(rankValues.data()+size_t(rank)*H,H,
                host+Guard+size_t(cpu.rank_to_row[rank])*H);
            // Missing the host selection must fail the bitwise comparison.
            std::fill_n(masked.data()+size_t(rank)*H,H,-9012.5f);
        }
        const std::vector<float> immutable(host,host+size_t(Ranks)*H+2*Guard);
        ctx.memcpy_h2d(down.data(),rankValues.data(),rankValues.size()*sizeof(float));
        ops::moe_ordered_accumulate_down_rows(ctx,down.data_as<float>(),weight.data_as<float>(),
            reference.data_as<bf16>()+Guard,1,Ranks,H,
            withShared?sharedDevice.data_as<bf16>():nullptr,withShared?gateDevice.data_as<bf16>():nullptr);
        ctx.memcpy_h2d(down.data(),masked.data(),masked.size()*sizeof(float));
        const auto deviceAllocations=ctx.device_allocation_count(),hostAllocations=ctx.pinned_host_allocation_count();
        previousReader=ops::moe_ordered_accumulate_cpu_host_rows(ctx,down.data_as<float>(),weight.data_as<float>(),
            candidate.data_as<bf16>()+Guard,1,Ranks,H,cpu,
            withShared?sharedDevice.data_as<bf16>():nullptr,withShared?gateDevice.data_as<bf16>():nullptr);
        // The kernel must own a value copy, independent of the next host map.
        cpu.rank_to_row.fill(-1);cpu.rows=0;cpu.values=nullptr;
        previousReader.wait_and_throw();
        expect(ctx.device_allocation_count()==deviceAllocations && ctx.pinned_host_allocation_count()==hostAllocations,
            "CPU host merge allocated hot USM");
        const auto expected=download<bf16>(ctx,reference,outputPoison.size());
        const auto actual=download<bf16>(ctx,candidate,outputPoison.size());
        expect(std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(bf16))==0,
            "CPU host merge changed ordered rank/shared bits");
        for(int j=0;j<Guard;++j) {
            expect(bf16_bits(actual[j])==bf16_bits(outputPoison[j]),"CPU host merge prefix canary");
            expect(bf16_bits(actual[Guard+H+j])==bf16_bits(outputPoison[Guard+H+j]),"CPU host merge suffix canary");
        }
        expect(std::memcmp(host,immutable.data(),immutable.size()*sizeof(float))==0,
            "CPU host merge changed pinned input/unused rows/canaries");
    }
    ops::MoeCpuHostDownRows bad;
    bad.values=host+Guard;bad.rows=1;bad.rank_to_row.fill(-1);
    auto rejects=[&](int tokens,int ranks) {
        bool rejected=false;
        try {ops::moe_ordered_accumulate_cpu_host_rows(ctx,down.data_as<float>(),weight.data_as<float>(),
            candidate.data_as<bf16>()+Guard,tokens,ranks,H,bad);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"CPU host merge accepted invalid shape/map");
    };
    rejects(2,Ranks);rejects(1,17);
    bad.rank_to_row[4]=1;rejects(1,Ranks);
    bad.rank_to_row[4]=-2;rejects(1,Ranks);
    bad.rank_to_row[4]=-1;bad.values=nullptr;rejects(1,Ranks);
    std::cout<<"[PASS] CPU USMhost ordered merge allcold/mixed/permuted, shared bits, immutable map, reader reuse, canaries, no hot USM\n";
}

void test_moe_front_fusion(Context& ctx) {
    constexpr int Experts = 512;
    std::mt19937 rng(20261006);
    std::vector<bf16> routerWeight(size_t(Experts) * H), sharedWeight(H);
    // Mixed-sign, relatively large weights expose accumulation/reduction order
    // changes in both router rows and the final shared-gate subgroup.
    auto weight = [&]() { return bf16(float(int(rng() % 2049) - 1024) / 64.0f); };
    for (auto& value : routerWeight) value = weight();
    for (auto& value : sharedWeight) value = weight();
    Tensor router = upload(ctx, routerWeight, dnnl::memory::data_type::bf16);
    Tensor shared = upload(ctx, sharedWeight, dnnl::memory::data_type::bf16);
    Tensor input = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::bf16);
    Tensor referenceRouter = Tensor::allocate(ctx, {Experts}, dnnl::memory::data_type::bf16);
    Tensor referenceGate = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16);
    Tensor referenceXq = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::u8);
    Tensor referenceXs = Tensor::allocate(ctx, {H / Block}, dnnl::memory::data_type::f32);
    Tensor fusedRouter = Tensor::allocate(ctx, {Experts}, dnnl::memory::data_type::bf16);
    Tensor fusedGate = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::bf16);
    Tensor fusedXq = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::u8);
    Tensor fusedXs = Tensor::allocate(ctx, {H / Block}, dnnl::memory::data_type::f32);
    std::vector<bf16> hostInput(H), poisonRouter(Experts, bf16(-33.0f));
    const bf16 poisonGate(-33.0f);
    std::vector<int8_t> poisonQ(H, int8_t(-101));
    std::vector<float> poisonScale(H / Block, 777.25f);
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (int j = 0; j < H; ++j) {
            const float amplitude = pattern == 2 ? 32.0f : 1.0f;
            hostInput[j] = bf16(pattern == 0 ? 0.0f : amplitude * float(int(rng() % 513) - 256) / 128.0f);
        }
        if (pattern == 1) {
            constexpr float ties[] = {127.0f, -127.0f, 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
            std::fill(hostInput.begin(), hostInput.begin() + Block, bf16(0.0f));
            for (int j = 0; j < 8; ++j) hostInput[j] = bf16(ties[j]);
        }
        ctx.memcpy_h2d(input.data(), hostInput.data(), hostInput.size() * sizeof(bf16));
        ops::bf16_gemv_bf16(ctx, router, input.data_as<bf16>(), referenceRouter.data_as<bf16>(), Experts, H);
        ops::bf16_gemv_bf16(ctx, shared, input.data_as<bf16>(), referenceGate.data_as<bf16>(), 1, H);
        ops::quantize_moe_input_q8_64(ctx, input.data_as<bf16>(), referenceXq.data_as<int8_t>(), referenceXs.data_as<float>(), H);
        ctx.synchronize();
        const auto expectedRouter = download<bf16>(ctx, referenceRouter, Experts);
        const auto expectedGate = download<bf16>(ctx, referenceGate, 1);
        const auto expectedQ = download<int8_t>(ctx, referenceXq, H);
        const auto expectedScale = download<float>(ctx, referenceXs, H / Block);
        for (bool withQ8 : {false, true}) {
            ctx.memcpy_h2d(fusedRouter.data(), poisonRouter.data(), poisonRouter.size() * sizeof(bf16));
            ctx.memcpy_h2d(fusedGate.data(), &poisonGate, sizeof(bf16));
            ctx.memcpy_h2d(fusedXq.data(), poisonQ.data(), poisonQ.size());
            ctx.memcpy_h2d(fusedXs.data(), poisonScale.data(), poisonScale.size() * sizeof(float));
            ops::bf16_moe_front_fused(ctx, router.data_as<bf16>(), shared.data_as<bf16>(), input.data_as<bf16>(),
                fusedRouter.data_as<bf16>(), fusedGate.data_as<bf16>(),
                withQ8 ? fusedXq.data_as<int8_t>() : nullptr,
                withQ8 ? fusedXs.data_as<float>() : nullptr, H, Experts);
            ctx.synchronize();
            const auto actualRouter = download<bf16>(ctx, fusedRouter, Experts);
            const auto actualGate = download<bf16>(ctx, fusedGate, 1);
            const auto actualQ = download<int8_t>(ctx, fusedXq, H);
            const auto actualScale = download<float>(ctx, fusedXs, H / Block);
            expect(std::memcmp(actualRouter.data(), expectedRouter.data(), Experts * sizeof(bf16)) == 0,
                "fused front router differs bitwise from original BF16 GEMV");
            expect(std::memcmp(actualGate.data(), expectedGate.data(), sizeof(bf16)) == 0,
                "fused front shared gate differs bitwise from original BF16 GEMV");
            expect(actualQ == (withQ8 ? expectedQ : poisonQ), "fused front Q8 output/null-mode guard mismatch");
            const auto& expectedScales = withQ8 ? expectedScale : poisonScale;
            expect(std::memcmp(actualScale.data(), expectedScales.data(), expectedScales.size() * sizeof(float)) == 0,
                "fused front Q8 scales differ bitwise/null-mode guard mismatch");
            std::cout << "[PASS] BF16 MoE front fusion bitwise router/shared/Q8 pattern=" << pattern
                << " optional_q8=" << withQ8 << " H=" << H << " experts=" << Experts << '\n';
        }
    }
}

void test_extreme_quantization(Context& ctx) {
    std::vector<bf16> input(H, bf16(0.0f));
    input[0] = from_bits(0x7f7f); // largest finite positive BF16
    input[1] = from_bits(0xff7f);
    input[2] = from_bits(0x7eff); // exactly half the block maximum: 63.5 -> 64
    input[3] = from_bits(0xfeff);
    input[4] = from_bits(0x7f80); // inf ignored, even with fast-math
    input[5] = from_bits(0x7fc1); // NaN ignored
    std::vector<int8_t> expected(H);
    std::vector<float> expectedScale(H / Block);
    quantize_reference(input.data(), H, expected.data(), expectedScale.data());
    Tensor x = upload(ctx, input, dnnl::memory::data_type::bf16);
    Tensor xq = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::u8);
    Tensor xs = Tensor::allocate(ctx, {H / Block}, dnnl::memory::data_type::f32);
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    ops::quantize_moe_input_q8_64(ctx, x.data_as<bf16>(), xq.data_as<int8_t>(), xs.data_as<float>(), H).wait_and_throw();
    expect(download<int8_t>(ctx, xq, H) == expected, "large finite BF16 quantization overflow/tie mismatch");
    const auto actualScale = download<float>(ctx, xs, H / Block);
    for (int j = 0; j < H / Block; ++j) close(actualScale[j], expectedScale[j], 1e-8f, 1e-6f, "large finite scale");
    std::cout << "[PASS] GPU Q8 finite BF16 maximum/half ties/nonfinite/zero blocks\n";
}

void run_case(Context& ctx, std::array<Expert, Slots>& experts, bool nested, bool zero) {
    std::vector<bf16> input(H, bf16(0.0f));
    if (!zero) {
        std::mt19937 rng(20261005);
        for (int j = Block; j < H; ++j) input[j] = bf16(float(int(rng() % 257) - 128) / 128.0f);
        // First block is exactly scale=1, exposing signed ties-to-even.
        constexpr float ties[] = {127.0f, -127.0f, 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
        for (int j = 0; j < 8; ++j) input[j] = bf16(ties[j]);
        // Mixed nonfinite/finite block and a separate all-zero block.
        input[9] = bf16(std::numeric_limits<float>::infinity());
        input[10] = bf16(std::numeric_limits<float>::quiet_NaN());
        std::fill(input.begin() + Block, input.begin() + 2 * Block, bf16(0.0f));
    }
    std::vector<int8_t> expectedX(H);
    std::vector<float> expectedXs(H / Block);
    quantize_reference(input.data(), H, expectedX.data(), expectedXs.data());
    std::vector<ops::MoeExpertSlotDev> slots(Slots);
    for (int j = 0; j < Slots; ++j) {
        slots[j].gate_up_packed = experts[j].gateUp.packedDevice.data_as<uint8_t>();
        slots[j].gate_up_absmax = experts[j].gateUp.expandedDevice.data_as<float>();
        slots[j].down_packed = experts[j].down.packedDevice.data_as<uint8_t>();
        slots[j].down_absmax = experts[j].down.expandedDevice.data_as<float>();
        slots[j].gate_up_nested = experts[j].gateUp.nested();
        slots[j].down_nested = experts[j].down.nested();
    }
    std::vector<int> selected(Selected.begin(), Selected.end());
    std::vector<bf16> poisonH(K * I, bf16(-33.0f));
    std::vector<int8_t> poisonHq(K * I, int8_t(-101));
    std::vector<float> poisonHs(K * I / Block, 777.25f), poisonDown(K * H, -9012.5f);
    Tensor x = upload(ctx, input, dnnl::memory::data_type::bf16);
    Tensor xq = Tensor::allocate(ctx, {H}, dnnl::memory::data_type::u8);
    Tensor xs = Tensor::allocate(ctx, {H / Block}, dnnl::memory::data_type::f32);
    Tensor descriptors = upload(ctx, slots, dnnl::memory::data_type::u8);
    Tensor selection = upload(ctx, selected, dnnl::memory::data_type::s32);
    Tensor h = upload(ctx, poisonH, dnnl::memory::data_type::bf16);
    Tensor hq = upload(ctx, poisonHq, dnnl::memory::data_type::u8);
    Tensor hs = upload(ctx, poisonHs, dnnl::memory::data_type::f32);
    Tensor down = upload(ctx, poisonDown, dnnl::memory::data_type::f32);
    // Last declared: submitted work drains before any input/output arena dies.
    struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
    ops::quantize_moe_input_q8_64(ctx, x.data_as<bf16>(), xq.data_as<int8_t>(), xs.data_as<float>(), H);
    ops::moe_resident_gate_up_q8_64(ctx, descriptors.data_as<ops::MoeExpertSlotDev>(), selection.data_as<int>(),
        xq.data_as<int8_t>(), xs.data_as<float>(), h.data_as<bf16>(), hq.data_as<int8_t>(), hs.data_as<float>(), K, H, I, nested);
    ops::moe_resident_down_q8_64(ctx, descriptors.data_as<ops::MoeExpertSlotDev>(), selection.data_as<int>(),
        hq.data_as<int8_t>(), hs.data_as<float>(), down.data_as<float>(), K, H, I, nested).wait_and_throw();
    const auto actualX = download<int8_t>(ctx, xq, H);
    const auto actualXs = download<float>(ctx, xs, H / Block);
    const auto actualH = download<bf16>(ctx, h, K * I);
    const auto actualHq = download<int8_t>(ctx, hq, K * I);
    const auto actualHs = download<float>(ctx, hs, K * I / Block);
    const auto actualDown = download<float>(ctx, down, K * H);
    expect(actualX == expectedX, "input Q8 zero/signed/RTE/nonfinite quantization mismatch");
    for (int j = 0; j < H / Block; ++j) close(actualXs[j], expectedXs[j], 1e-8f, 1e-6f, "input scale");
    if (!zero) {
        constexpr int8_t expectedTies[] = {127, -127, 0, 2, 2, 0, -2, -2};
        for (int j = 0; j < 8; ++j) expect(actualX[j] == expectedTies[j], "explicit tie fixture failed");
    }
    std::vector<int8_t> quantizedH(I);
    std::vector<float> hScale(I / Block);
    for (int rank = 0; rank < K; ++rank) {
        if (Selected[rank] < 0) {
            for (int j = 0; j < I; ++j) {
                expect(float(actualH[rank * I + j]) == -33.0f, "masked rank wrote BF16 h");
                expect(actualHq[rank * I + j] == -101, "masked rank wrote Q8 h");
            }
            for (int j = 0; j < I / Block; ++j) expect(actualHs[rank * I / Block + j] == 777.25f, "masked rank wrote h scale");
            for (int j = 0; j < H; ++j) expect(actualDown[rank * H + j] == -9012.5f, "masked rank wrote down");
            continue;
        }
        const auto& expert = experts[Selected[rank]];
        for (int row = 0; row < I; ++row) {
            const float gate = expert.gateUp.dot(row, H, expectedX.data(), expectedXs.data(), nested);
            const float up = expert.gateUp.dot(row + I, H, expectedX.data(), expectedXs.data(), nested);
            const float expected = gate / (1.0f + std::exp(-gate)) * up;
            // Includes one BF16 rounding step and native-exp/subgroup reduction
            // differences; at most approximately one BF16 ulp plus a small floor.
            close(float(actualH[rank * I + row]), expected, 2e-6f, 0.008f, "gate/up BF16");
        }
        quantize_reference(actualH.data() + rank * I, I, quantizedH.data(), hScale.data());
        for (int j = 0; j < I; ++j) expect(actualHq[rank * I + j] == quantizedH[j], "h Q8 differs from actual BF16 h quantization");
        for (int j = 0; j < I / Block; ++j) close(actualHs[rank * I / Block + j], hScale[j], 1e-9f, 1e-6f, "h scale");
        for (int row = 0; row < H; ++row) {
            const float expected = expert.down.dot(row, I, actualHq.data() + rank * I, actualHs.data() + rank * I / Block, nested);
            close(actualDown[rank * H + row], expected, 2e-7f, 2e-5f, "down signed-byte dot");
        }
    }
    std::cout << "[PASS] GPU NF4Q8 independent oracle H=" << H << " I=" << I << " top_k=" << K
        << " nested=" << nested << " zero=" << zero << " masked/permuted=1\n";
}
} // namespace

int main() {
    try {
        Context ctx;
        if (!ctx.queue().get_device().is_gpu() || ctx.queue().get_device().get_info<sycl::info::device::max_work_group_size>() < 256) {
            std::cout << "[SKIP] GPU with work-group size >=256 required\n";
            return 77;
        }
        std::array<Expert, Slots> experts;
        struct Drain { Context& ctx; ~Drain() { try { ctx.synchronize(); } catch (...) {} } } drain{ctx};
        test_moe_front_fusion(ctx);
        test_history_projection_fusion(ctx);
        test_cpu_host_ordered_merge(ctx);
        test_extreme_quantization(ctx);
        for (int columns : {H, 6144}) for (bool nested : {false, true})
            test_dense_q8_projection(ctx, columns, nested);
        test_dense_q8_rejections(ctx);
        for (int j = 0; j < Slots; ++j) {
            experts[j].gateUp.initialize(ctx, 2 * I, H, 101 + j * 31);
            experts[j].down.initialize(ctx, H, I, 701 + j * 47);
        }
        for (bool nested : {false, true}) for (bool zero : {true, false}) run_case(ctx, experts, nested, zero);
        std::cout << "All GPU NF4Q8 tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU NF4Q8 test failed: " << error.what() << '\n';
        return 1;
    }
}
