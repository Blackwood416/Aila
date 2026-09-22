#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/Ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
using DataType = dnnl::memory::data_type;
constexpr int kInput = 128, kOutput = 128, kBlock = 64;

void Expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

struct DrainQueue {
    Context& context;
    ~DrainQueue() { try { context.synchronize(); } catch (...) {} }
};

struct Adapter {
    int rank, offset, rows;
    float scale;
    std::vector<bf16> a, b;
};

Adapter MakeAdapter(int rank, int offset, int rows, float scale, int salt) {
    Adapter result{rank, offset, rows, scale};
    result.a.resize(static_cast<size_t>(rank) * kInput);
    result.b.resize(static_cast<size_t>(rows) * rank);
    for (size_t i = 0; i < result.a.size(); ++i)
        result.a[i] = bf16((static_cast<int>((i * 7 + salt) % 13) - 6) / 16.0f);
    for (size_t i = 0; i < result.b.size(); ++i)
        result.b[i] = bf16((static_cast<int>((i * 3 + salt) % 11) - 5) / 16.0f);
    return result;
}

LoraAttachment Upload(Context& context, const Adapter& adapter) {
    LoraAttachment result;
    result.lora_a = Tensor::allocate(context, {adapter.rank, kInput});
    result.lora_b = Tensor::allocate(context, {adapter.rows, adapter.rank});
    context.memcpy_h2d(result.lora_a.data(), adapter.a.data(), adapter.a.size() * sizeof(bf16));
    context.memcpy_h2d(result.lora_b.data(), adapter.b.data(), adapter.b.size() * sizeof(bf16));
    result.scaling = adapter.scale;
    result.output_offset = adapter.offset;
    result.output_rows = adapter.rows;
    return result;
}

std::vector<bf16> Reference(const std::vector<bf16>& input, int rows,
                            const std::vector<uint8_t>& packed,
                            const std::vector<float>& map,
                            const std::vector<Adapter>& adapters) {
    std::vector<bf16> result(static_cast<size_t>(rows) * kOutput);
    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < kOutput; ++out) {
            float sum = 0;
            for (int in = 0; in < kInput; ++in) {
                const auto byte = packed[out * (kInput / 2) + in / 2];
                const int code = in % 2 ? byte & 15 : byte >> 4;
                sum += static_cast<float>(input[row * kInput + in]) * map[code];
            }
            result[row * kOutput + out] = bf16(sum);
        }
        for (const auto& adapter : adapters) {
            std::vector<float> intermediate(adapter.rank, 0);
            for (int r = 0; r < adapter.rank; ++r)
                for (int in = 0; in < kInput; ++in)
                    intermediate[r] += static_cast<float>(adapter.a[r * kInput + in]) *
                                       static_cast<float>(input[row * kInput + in]);
            for (int out = 0; out < adapter.rows; ++out) {
                float update = 0;
                for (int r = 0; r < adapter.rank; ++r)
                    update += static_cast<float>(adapter.b[out * adapter.rank + r]) * intermediate[r];
                auto& value = result[row * kOutput + adapter.offset + out];
                value = bf16(static_cast<float>(value) + update * adapter.scale);
            }
        }
    }
    return result;
}

void CheckOutput(Context& context, Tensor& output, const std::vector<bf16>& expected) {
    std::vector<bf16> actual(expected.size());
    context.memcpy_d2h(actual.data(), output.data(), actual.size() * sizeof(bf16));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float value = static_cast<float>(actual[i]);
        const float reference = static_cast<float>(expected[i]);
        // CPU/device BF16 conversion can differ at a tie. Two BF16 ULPs
        // cover the base projection and overlapping adapter additions.
        const float tolerance = 0.002f + 0.016f * std::abs(reference);
        Expect(std::isfinite(value) && std::abs(value - reference) <= tolerance,
               "LoRA output mismatch at " + std::to_string(i) + ": " +
               std::to_string(value) + " vs " + std::to_string(reference));
    }
}

std::vector<bf16> SwigluReference(const std::vector<bf16>& projected, int rows) {
    constexpr int width = kOutput / 2;
    std::vector<bf16> result(static_cast<size_t>(rows) * width);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
            const float gate = static_cast<float>(projected[row * kOutput + column]);
            const float up = static_cast<float>(projected[row * kOutput + width + column]);
            result[row * width + column] = bf16(gate / (1.0f + std::exp(-gate)) * up);
        }
    }
    return result;
}

void CheckGateUpAdapters(Context& context, const Bnb4BitWeightRef& weight,
                         const std::vector<uint8_t>& packed, const std::vector<float>& map) {
    constexpr int width = kOutput / 2;
    constexpr int rows = 4;
    Bnb4BitLinear projection;
    Bnb4BitLinearScratch scratch;
    Tensor input = Tensor::allocate(context, {rows, kInput});
    Tensor singleProjection = Tensor::allocate(context, {1, kOutput});
    Tensor singleOutput = Tensor::allocate(context, {1, width});
    Tensor batchProjection = Tensor::allocate(context, {rows, kOutput});
    Tensor batchGate = Tensor::allocate(context, {rows, width});
    Tensor batchUp = Tensor::allocate(context, {rows, width});
    std::vector<bf16> hostInput(static_cast<size_t>(rows) * kInput);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < kInput; ++column) {
            hostInput[row * kInput + column] = bf16((row + 1) * (1 + column % 17) / 128.0f);
        }
    }
    DrainQueue drain{context};
    context.memcpy_h2d(input.data(), hostInput.data(), hostInput.size() * sizeof(bf16));
    std::string error;
    Expect(projection.init(context, weight, &error), "Gate/up fixture initialization failed: " + error);
    auto singleInput = Tensor::view(context, input.data(), {1, kInput}, DataType::bf16);
    // The base-only kernel remains eligible. Adapter cases below must execute
    // the real projection and activation fallback, with an independent scalar
    // oracle that includes the BF16 projection boundary.
    Expect(Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
               context, projection, singleInput, singleOutput, width),
           "An unadapted gate/up projection lost its fused decode path");
    context.synchronize();

    auto makeAdapter = [](int rank, int offset, float scale) {
        Adapter adapter{rank, offset, width, scale};
        adapter.a.resize(static_cast<size_t>(rank) * kInput);
        adapter.b.resize(static_cast<size_t>(width) * rank);
        // Positive A/B with distinct scales gives each adapter a visible
        // signal; omitting it must fail the numerical check, not round away.
        for (size_t i = 0; i < adapter.a.size(); ++i) adapter.a[i] = bf16((1 + i % 5) / 32.0f);
        for (size_t i = 0; i < adapter.b.size(); ++i) adapter.b[i] = bf16((1 + i % 3) / 16.0f);
        return adapter;
    };
    const auto gate = makeAdapter(2, 0, 0.75f);
    const auto up = makeAdapter(3, width, -0.5f);
    struct Case {
        const char* name;
        std::vector<Adapter> adapters;
    };
    const std::vector<Case> cases{{"gate-only", {gate}}, {"up-only", {up}}, {"gate-and-up", {gate, up}}};
    const auto base = SwigluReference(Reference(hostInput, rows, packed, map, {}), rows);
    for (const auto& test : cases) {
        std::vector<LoraAttachment> attachments;
        for (const auto& adapter : test.adapters) attachments.push_back(Upload(context, adapter));
        projection.set_lora(std::move(attachments));
        const auto expected = SwigluReference(Reference(hostInput, rows, packed, map, test.adapters), rows);
        bool visibleAdapter = false;
        for (size_t i = 0; i < expected.size(); ++i) {
            const float value = static_cast<float>(expected[i]);
            visibleAdapter |= std::abs(value - static_cast<float>(base[i])) >
                2.0f * (0.002f + 0.016f * std::abs(value));
        }
        Expect(visibleAdapter, std::string(test.name) + ": fixture adapter has no observable signal");
        Expect(!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                   context, projection, singleInput, singleOutput, width),
               std::string(test.name) + ": fused decode bypassed an adapter");
        projection.forward(context, scratch, singleInput, singleProjection, 1);
        ops::fused_gate_up_swiglu(context, singleProjection, singleOutput, width);
        CheckOutput(context, singleOutput, std::vector<bf16>(expected.begin(), expected.begin() + width));

        projection.forward(context, scratch, input, batchProjection, rows);
        ops::split_gate_up(context, batchProjection, batchGate, batchUp, rows, width);
        ops::swiglu(context, batchGate, batchUp, batchGate, rows * width);
        CheckOutput(context, batchGate, expected);
        std::cout << "[PASS] LoRA gate/up " << test.name << " B1=fallback B4=projection rows=" << rows << '\n';
    }
}

void Run() {
    Context context;
    {
        std::vector<float> map(16), absmax(kInput * kOutput / kBlock, 1.0f);
        std::vector<uint8_t> packed(kInput * kOutput / 2);
        for (int i = 0; i < 16; ++i) map[i] = (i - 7) / 8.0f;
        for (size_t i = 0; i < packed.size(); ++i)
            packed[i] = static_cast<uint8_t>(((i * 7 + 3) % 16) * 16 + (i * 5 + 1) % 16);
        Tensor packedDevice = Tensor::allocate(context, {static_cast<int64_t>(packed.size())}, DataType::u8);
        Tensor mapDevice = Tensor::allocate(context, {16}, DataType::f32);
        Tensor absmaxDevice = Tensor::allocate(context, {static_cast<int64_t>(absmax.size())}, DataType::f32);
        context.memcpy_h2d(packedDevice.data(), packed.data(), packed.size());
        context.memcpy_h2d(mapDevice.data(), map.data(), map.size() * sizeof(float));
        context.memcpy_h2d(absmaxDevice.data(), absmax.data(), absmax.size() * sizeof(float));
        Bnb4BitWeightRef weight;
        weight.name = "lora-lifetime-fixture";
        weight.packed_weight = &packedDevice;
        weight.quant_map = &mapDevice;
        weight.absmax = &absmaxDevice;
        weight.quant_state.quant_type = "nf4";
        weight.quant_state.dtype = "bfloat16";
        weight.quant_state.shape = {kOutput, kInput};
        weight.quant_state.blocksize = kBlock;
        Bnb4BitLinear projection;
        Bnb4BitLinearScratch scratch;
        std::vector<Tensor> inputs, outputs;
        std::vector<std::vector<bf16>> hostInputs;
        const std::vector<int> shapes{1, 3, 17, 5, 33};
        DrainQueue drain{context};
        std::string error;
        Expect(projection.init(context, weight, &error), "NF4 fixture initialization failed: " + error);
        for (int rows : shapes) {
            auto& input = inputs.emplace_back(Tensor::allocate(context, {rows, kInput}));
            outputs.emplace_back(Tensor::allocate(context, {rows, kOutput}));
            auto& host = hostInputs.emplace_back(static_cast<size_t>(rows) * kInput);
            for (size_t i = 0; i < host.size(); ++i)
                host[i] = bf16((static_cast<int>((i * 11 + rows) % 23) - 11) / 32.0f);
            context.memcpy_h2d(input.data(), host.data(), host.size() * sizeof(bf16));
        }
        // Warm base projection shapes before counting the additional LoRA
        // allocation. This also verifies the CPU reference without adapters.
        for (size_t i = 0; i < shapes.size(); ++i) {
            projection.forward(context, scratch, inputs[i], outputs[i], shapes[i]);
            context.synchronize();
            CheckOutput(context, outputs[i], Reference(hostInputs[i], shapes[i], packed, map, {}));
        }
        const std::vector<Adapter> adapters{
            MakeAdapter(3, 5, 47, 0.5f, 1),
            MakeAdapter(7, 32, 65, -0.25f, 4)};
        std::vector<LoraAttachment> attachments;
        for (const auto& adapter : adapters) attachments.push_back(Upload(context, adapter));
        projection.set_lora(std::move(attachments));
        const size_t beforeBytes = context.current_allocated_bytes();
        const size_t beforeCount = context.device_allocation_count();
        size_t capacity = 0;
        std::vector<std::vector<bf16>> expected;
        for (size_t i = 0; i < shapes.size(); ++i) {
            expected.push_back(Reference(hostInputs[i], shapes[i], packed, map, adapters));
            projection.forward(context, scratch, inputs[i], outputs[i], shapes[i]);
            capacity = std::max(capacity, static_cast<size_t>(shapes[i]) * 7);
            // This check occurs before drain. The intermediate must remain
            // owned after forward returns, even if its kernels are unfinished.
            Expect(context.current_allocated_bytes() == beforeBytes + capacity * sizeof(float) &&
                   context.device_allocation_count() == beforeCount + 1,
                   "LoRA intermediate was freed before completion or not reused");
        }
        context.synchronize();
        for (size_t i = 0; i < shapes.size(); ++i) CheckOutput(context, outputs[i], expected[i]);
        const size_t stableBytes = context.current_allocated_bytes();
        const size_t stableCount = context.device_allocation_count();
        // Submit repeated mixed decode/prefill shapes without caller drains.
        // Different ranks and overlapping rows share one in-order intermediate.
        for (int round = 0; round < 12; ++round)
            for (size_t i = shapes.size(); i-- > 0;)
                projection.forward(context, scratch, inputs[i], outputs[i], shapes[i]);
        Expect(context.current_allocated_bytes() == stableBytes &&
               context.device_allocation_count() == stableCount,
               "Steady LoRA execution kept allocating temporaries");
        context.synchronize();
        for (size_t i = 0; i < shapes.size(); ++i) CheckOutput(context, outputs[i], expected[i]);
        CheckGateUpAdapters(context, weight, packed, map);
    }
    context.synchronize();
    context.release_execution_resources();
    Expect(context.current_allocated_bytes() == 0 && context.device_allocation_count() == 0,
           "LoRA/model execution resources survived destruction");
}
} // namespace

int main(int argc, char** argv) {
    try {
        Expect(argc == 2 && (std::string(argv[1]) == "fused" || std::string(argv[1]) == "dnnl"),
               "Usage: AilaLoraExecutionTests fused|dnnl");
        const char* fused = std::string(argv[1]) == "fused" ? "1" : "0";
#ifdef _WIN32
        Expect(_putenv_s("AILA_BNB4_FUSED_PREFILL", fused) == 0, "Cannot select NF4 prefill fixture");
#else
        Expect(setenv("AILA_BNB4_FUSED_PREFILL", fused, 1) == 0, "Cannot select NF4 prefill fixture");
#endif
        Run();
        std::cout << "LoRA " << argv[1] << " execution: GPU/scalar-reference numerics, gate/up fallback, retained scratch, growth/reuse and teardown passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LoRA execution failed: " << error.what() << '\n';
        return 1;
    }
}
