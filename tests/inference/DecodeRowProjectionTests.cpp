#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Bnb4BitLinear.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
using DataType = dnnl::memory::data_type;
constexpr int MaxRows = 4;
constexpr int Block = 64;
constexpr size_t Guard = 32;
constexpr size_t DeviceBudget = 128ULL << 20;
constexpr size_t PinnedBudget = 16ULL << 20;
constexpr std::array<float, 16> QuantMap{
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};

void Expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

struct DrainQueue {
    Context& context;
    ~DrainQueue() { try { context.synchronize(); } catch (...) {} }
};

uint16_t Bits(bf16 value) {
    static_assert(sizeof(bf16) == sizeof(uint16_t));
    uint16_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint32_t Mix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

void SelectDefaultPath() {
    // This executable is the gate for the new default SG16/WG256 kernel.
    // Process-local settings precede both paths' read-once environment locals.
#ifdef _WIN32
    Expect(_putenv_s("AILA_BNB4_GEMV_WG", "256") == 0 &&
           _putenv_s("AILA_BNB4_CACHE_DEQUANT", "0") == 0,
           "Cannot select the default packed decode fixture");
#else
    Expect(setenv("AILA_BNB4_GEMV_WG", "256", 1) == 0 &&
           setenv("AILA_BNB4_CACHE_DEQUANT", "0", 1) == 0,
           "Cannot select the default packed decode fixture");
#endif
}

struct Shape {
    const char* name;
    int input;
    int output;
    bool adapters = false;
    bool invalidInputs = false;
};

std::vector<bf16> CaptureOriginalB1(Context& context, Bnb4BitLinear& projection,
                                    Bnb4BitLinearScratch& scratch,
                                    Tensor& input, Tensor& output) {
    const int k = projection.in_features(), n = projection.out_features();
    for (int row = 0; row < MaxRows; ++row) {
        auto rowInput = Tensor::view(context,
            input.data_as<bf16>() + static_cast<size_t>(row) * k, {1, k});
        auto rowOutput = Tensor::view(context,
            output.data_as<bf16>() + static_cast<size_t>(row) * n, {1, n});
        // Do not use ForwardDecodeRows for the oracle: this must exercise the
        // unchanged production B1 projection and its original adapter order.
        projection.forward(context, scratch, rowInput, rowOutput, 1);
    }
    std::vector<bf16> result(static_cast<size_t>(MaxRows) * n);
    context.memcpy_d2h(result.data(), output.data(), result.size() * sizeof(bf16));
    return result;
}

void CheckCpuReference(const Shape& shape, const std::vector<uint8_t>& packed,
                       const std::vector<float>& absmax,
                       const std::vector<bf16>& input,
                       const std::vector<bf16>& actual) {
    const int k = shape.input, n = shape.output;
    std::vector<float> weight(static_cast<size_t>(n) * k);
    for (size_t index = 0; index < weight.size(); ++index) {
        const uint8_t byte = packed[index / 2];
        const int code = index % 2 ? byte & 15 : byte >> 4;
        // Decode rounds each dequantized weight to BF16 before accumulation.
        weight[index] = static_cast<float>(bf16(QuantMap[code] * absmax[index / Block]));
    }

    // An independent double dot product is compared with a derived forward
    // error bound, not a tolerance fitted to observed GPU results. Each SG16
    // lane accumulates ceil(K/16) terms; at most 16 additional FP32 additions
    // cover the subgroup reduction. The small/oneDNN fallback uses the more
    // conservative serial K+128 bound. Final BF16 rounding contributes 2^-8.
    const double operations = k >= 512 && k % Block == 0 ? (k + 15) / 16 + 16 : k + 128;
    const double floatUnit = std::ldexp(1.0, -24);
    const double gamma = operations * floatUnit / (1.0 - operations * floatUnit);
    double maxRatio = 0.0;
    for (int row = 0; row < MaxRows; ++row) {
        for (int out = 0; out < n; ++out) {
            double sum = 0.0, sumAbs = 0.0;
            for (int in = 0; in < k; ++in) {
                const double product = static_cast<double>(weight[static_cast<size_t>(out) * k + in]) *
                    static_cast<double>(static_cast<float>(input[Guard + static_cast<size_t>(row) * k + in]));
                sum += product;
                sumAbs += std::abs(product);
            }
            const double fp32Error = gamma * sumAbs;
            const double bound = fp32Error + (std::abs(sum) + fp32Error) / 256.0 + 1.0e-7;
            const double value = static_cast<float>(actual[static_cast<size_t>(row) * n + out]);
            const double error = std::abs(value - sum);
            if (!std::isfinite(value) || error > bound) {
                throw std::runtime_error(std::string(shape.name) + " CPU reference failed: row=" +
                    std::to_string(row) + " output=" + std::to_string(out) + " actual=" +
                    std::to_string(value) + " double_reference=" + std::to_string(sum) +
                    " error=" + std::to_string(error) + " bound=" + std::to_string(bound));
            }
            maxRatio = std::max(maxRatio, error / bound);
        }
    }
    std::cout << "[CPU] case=" << shape.name << " values=" << actual.size()
              << " within_bound=1 max_error_bound_ratio=" << maxRatio << '\n';
}

void CheckBatches(Context& context, const Shape& shape, const char* mode,
                  Bnb4BitLinear& projection, Bnb4BitLinearScratch& scratch,
                  Tensor& input, Tensor& output, Tensor& outputStorage,
                  const std::vector<bf16>& expected) {
    const bf16 canary(123.0f);
    std::vector<bf16> actual(static_cast<size_t>(outputStorage.numel()), canary);
    // Repeat B1 after the B4 call to catch retained-shape or binding changes.
    for (int rows : {1, 2, 3, 4, 1}) {
        std::fill(actual.begin(), actual.end(), canary);
        context.memcpy_h2d(outputStorage.data(), actual.data(), actual.size() * sizeof(bf16));
        const size_t bytes = context.current_allocated_bytes();
        const size_t count = context.device_allocation_count();
        projection.ForwardDecodeRows(context, scratch, input, output, rows);
        Expect(context.current_allocated_bytes() == bytes && context.device_allocation_count() == count,
               std::string(shape.name) + ": a warmed decode projection allocated more tracked USM");
        context.memcpy_d2h(actual.data(), outputStorage.data(), actual.size() * sizeof(bf16));
        const size_t logical = static_cast<size_t>(rows) * shape.output;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (index >= Guard && index < Guard + logical) {
                const size_t offset = index - Guard;
                if (!std::isfinite(static_cast<float>(actual[index])) ||
                    Bits(actual[index]) != Bits(expected[offset])) {
                    throw std::runtime_error(std::string(shape.name) + " " + mode +
                        " original B1 bitwise mismatch: B=" + std::to_string(rows) +
                        " offset=" + std::to_string(offset) + " actual_bits=" +
                        std::to_string(Bits(actual[index])) + " expected_bits=" +
                        std::to_string(Bits(expected[offset])));
                }
            } else {
                Expect(Bits(actual[index]) == Bits(canary), std::string(shape.name) +
                       ": output prefix, suffix or unused-capacity canary changed at " + std::to_string(index));
            }
        }
        std::cout << "[BITWISE] case=" << shape.name << " mode=" << mode << " B=" << rows
                  << " values=" << logical << " exact=1 canary=1 usm_growth_bytes=0\n";
    }
}

void CheckInvalidInputs(Context& context, Bnb4BitLinear& projection,
                        Bnb4BitLinearScratch& scratch,
                        Tensor& input, Tensor& output, Tensor& outputStorage) {
    const int k = projection.in_features(), n = projection.out_features();
    const bf16 canary(123.0f);
    std::vector<bf16> data(static_cast<size_t>(outputStorage.numel()), canary);
    context.memcpy_h2d(outputStorage.data(), data.data(), data.size() * sizeof(bf16));
    const size_t deviceBytes = context.current_allocated_bytes();
    const size_t deviceCount = context.device_allocation_count();
    const size_t hostBytes = context.current_pinned_host_bytes();
    const size_t hostCount = context.pinned_host_allocation_count();
    size_t rejected = 0;
    auto goodInput = [&] { return Tensor::view(context, input.data(), input.shape()); };
    auto goodOutput = [&] { return Tensor::view(context, output.data(), output.shape()); };
    auto reject = [&](const char* label, Tensor testInput, Tensor testOutput, int rows,
                      Bnb4BitLinear& linear) {
        bool caught = false;
        try {
            linear.ForwardDecodeRows(context, scratch, testInput, testOutput, rows);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        Expect(caught, std::string("Malformed decode input was accepted: ") + label);
        Expect(context.current_allocated_bytes() == deviceBytes && context.device_allocation_count() == deviceCount &&
               context.current_pinned_host_bytes() == hostBytes && context.pinned_host_allocation_count() == hostCount,
               std::string("Invalid decode input changed the USM ledger: ") + label);
        ++rejected;
    };
    for (int rows : {-1, 0, 5, std::numeric_limits<int>::max()})
        reject("row-count", goodInput(), goodOutput(), rows, projection);
    reject("null-input", Tensor(), goodOutput(), 2, projection);
    reject("null-output", goodInput(), Tensor(), 2, projection);
    reject("input-dtype", Tensor::view(context, input.data(), input.shape(), DataType::f32),
           goodOutput(), 2, projection);
    reject("output-dtype", goodInput(), Tensor::view(context, output.data(), output.shape(), DataType::f32),
           2, projection);
    for (const std::vector<int64_t>& badShape : {
            std::vector<int64_t>{}, std::vector<int64_t>{MaxRows * k},
            std::vector<int64_t>{1, 1, MaxRows * k}, std::vector<int64_t>{0, k},
            std::vector<int64_t>{MaxRows, 0}, std::vector<int64_t>{-1, k},
            std::vector<int64_t>{MaxRows, -1},
            std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 2}}) {
        reject("input-shape", Tensor::view(context, input.data(), badShape), goodOutput(), 2, projection);
        reject("output-shape", goodInput(), Tensor::view(context, output.data(), badShape), 2, projection);
    }
    reject("short-input", Tensor::view(context, input.data(), {1, 2 * k - 1}), goodOutput(), 2, projection);
    reject("short-output", goodInput(), Tensor::view(context, output.data(), {1, 2 * n - 1}), 2, projection);
    Bnb4BitLinear uninitialized;
    reject("uninitialized-linear", goodInput(), goodOutput(), 2, uninitialized);
    context.memcpy_d2h(data.data(), outputStorage.data(), data.size() * sizeof(bf16));
    for (bf16 value : data) Expect(Bits(value) == Bits(canary), "Invalid input changed the output buffer");
    std::cout << "[VALIDATION] rejected=" << rejected << " output_unchanged=1 usm_unchanged=1\n";
}

LoraAttachment MakeAdapter(Context& context, int k, int rank, int offset, int rows,
                            float scale, uint32_t salt) {
    LoraAttachment result;
    result.lora_a = Tensor::allocate(context, {rank, k});
    result.lora_b = Tensor::allocate(context, {rows, rank});
    result.output_offset = offset;
    result.output_rows = rows;
    result.scaling = scale;
    std::vector<bf16> a(static_cast<size_t>(rank) * k), b(static_cast<size_t>(rows) * rank);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = bf16((static_cast<int>(Mix(static_cast<uint32_t>(i) + salt) % 31) - 15) / 64.0f);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = bf16((static_cast<int>(Mix(static_cast<uint32_t>(i) + salt + 19) % 23) - 11) / 32.0f);
    context.memcpy_h2d(result.lora_a.data(), a.data(), a.size() * sizeof(bf16));
    context.memcpy_h2d(result.lora_b.data(), b.data(), b.size() * sizeof(bf16));
    return result;
}

void CheckShape(Context& context, const Shape& shape) {
    const int k = shape.input, n = shape.output;
    const size_t values = static_cast<size_t>(k) * n;
    std::vector<uint8_t> packed((values + 1) / 2);
    std::vector<float> absmax((values + Block - 1) / Block);
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<uint8_t>(Mix(static_cast<uint32_t>(i) + 71));
    // Different block scales, all powers of two: exact scaling also keeps the
    // host and device BF16 dequantization reference clear at conversion ties.
    for (size_t i = 0; i < absmax.size(); ++i)
        absmax[i] = std::ldexp(1.0f, static_cast<int>((i * 5 + i / 7) % 4) - 3);

    const size_t inputCapacity = static_cast<size_t>(MaxRows) * (k + 16);
    const size_t outputCapacity = static_cast<size_t>(MaxRows) * (n + 32);
    std::vector<bf16> hostInput(Guard * 2 + inputCapacity, bf16(-17.0f));
    for (int row = 0; row < MaxRows; ++row)
        for (int in = 0; in < k; ++in)
            hostInput[Guard + static_cast<size_t>(row) * k + in] = bf16(
                (static_cast<int>(Mix(static_cast<uint32_t>(in) + 997U * (row + 1)) % 257) - 128) / 256.0f);

    Tensor packedDevice = Tensor::allocate(context, {static_cast<int64_t>(packed.size())}, DataType::u8);
    Tensor mapDevice = Tensor::allocate(context, {16}, DataType::f32);
    Tensor absmaxDevice = Tensor::allocate(context, {static_cast<int64_t>(absmax.size())}, DataType::f32);
    Tensor inputStorage = Tensor::allocate(context, {static_cast<int64_t>(hostInput.size())});
    Tensor outputStorage = Tensor::allocate(context, {static_cast<int64_t>(Guard * 2 + outputCapacity)});
    Tensor originalOutput = Tensor::allocate(context, {MaxRows, n});
    // Deliberately larger shape(1), with logical rows stored tightly. Treating
    // the capacity shape as a row stride would read canaries or miss outputs.
    Tensor input = Tensor::view(context, inputStorage.data_as<bf16>() + Guard, {MaxRows, k + 16});
    Tensor output = Tensor::view(context, outputStorage.data_as<bf16>() + Guard, {MaxRows, n + 32});
    Bnb4BitLinear projection;
    Bnb4BitLinearScratch scratch;
    DrainQueue drain{context};
    context.memcpy_h2d(packedDevice.data(), packed.data(), packed.size());
    context.memcpy_h2d(mapDevice.data(), QuantMap.data(), QuantMap.size() * sizeof(float));
    context.memcpy_h2d(absmaxDevice.data(), absmax.data(), absmax.size() * sizeof(float));
    context.memcpy_h2d(inputStorage.data(), hostInput.data(), hostInput.size() * sizeof(bf16));
    Bnb4BitWeightRef weight;
    weight.name = shape.name;
    weight.packed_weight = &packedDevice;
    weight.quant_map = &mapDevice;
    weight.absmax = &absmaxDevice;
    weight.quant_state.quant_type = "nf4";
    weight.quant_state.dtype = "bfloat16";
    weight.quant_state.shape = {n, k};
    weight.quant_state.blocksize = Block;
    std::string error;
    Expect(projection.init(context, weight, &error), std::string(shape.name) + ": initialization failed: " + error);
    std::cout << "[SHAPE] case=" << shape.name << " K=" << k << " N=" << n << '\n';
    const auto original = CaptureOriginalB1(context, projection, scratch, input, originalOutput);
    CheckCpuReference(shape, packed, absmax, hostInput, original);
    if (shape.invalidInputs) CheckInvalidInputs(context, projection, scratch, input, output, outputStorage);
    CheckBatches(context, shape, "base", projection, scratch, input, output, outputStorage, original);
    if (k % Block == 0) {
        Expect(!scratch.weight_bf16.valid() && !scratch.input_bf16.valid() && !scratch.output_bf16.valid(),
               std::string(shape.name) + ": packed decode unexpectedly allocated prefill scratch");
    }

    if (shape.adapters) {
        std::vector<LoraAttachment> adapters;
        adapters.push_back(MakeAdapter(context, k, 3, 2, 9, 0.5f, 11));
        adapters.push_back(MakeAdapter(context, k, 5, 6, 9, -0.375f, 23));
        projection.set_lora(std::move(adapters));
        const auto adapted = CaptureOriginalB1(context, projection, scratch, input, originalOutput);
        bool changed = false;
        for (size_t index = 0; index < original.size(); ++index) changed |= Bits(original[index]) != Bits(adapted[index]);
        Expect(changed, "Adapter fixture did not produce an observable signal");
        CheckBatches(context, shape, "lora", projection, scratch, input, output, outputStorage, adapted);
    }
    std::vector<bf16> inputAfter(hostInput.size());
    context.memcpy_d2h(inputAfter.data(), inputStorage.data(), inputAfter.size() * sizeof(bf16));
    Expect(std::memcmp(inputAfter.data(), hostInput.data(), hostInput.size() * sizeof(bf16)) == 0,
           std::string(shape.name) + ": projection changed its input or input canaries");
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "SKIP: NF4 decode-row projection GPU fixture requires explicit --run\n";
        return 77;
    }
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: AilaDecodeRowProjectionTests --run\n";
        return 0;
    }
    try {
        Expect(argc == 2 && std::string(argv[1]) == "--run", "Usage: AilaDecodeRowProjectionTests --run");
        SelectDefaultPath();
        std::cout << std::setprecision(9)
                  << "[CONTRACT] B1..4=original_B1_bitwise cpu=double_dot_forward_error_bound"
                     " GEMV_WG=256 CACHE_DEQUANT=0 device_budget_bytes=" << DeviceBudget
                  << " pinned_budget_bytes=" << PinnedBudget << '\n';
        Context context;
        context.set_allocation_limits(DeviceBudget, PinnedBudget);
        const auto device = context.queue().get_device();
        Expect(device.is_gpu(), "This opt-in fixture requires a GPU device");
        std::cout << "[DEVICE] name=" << device.get_info<sycl::info::device::name>()
                  << " driver=" << device.get_info<sycl::info::device::driver_version>() << '\n';
        for (const Shape& shape : {
                Shape{"q3-qkv", 1024, 4096}, Shape{"q3-o", 2048, 1024},
                Shape{"q3-down", 3072, 1024}, Shape{"tail", 1024, 17},
                Shape{"q4e-linear-all", 2560, 16480}, Shape{"q4e-attention-out", 6144, 2560},
                Shape{"q4e-qkv", 2560, 13312}, Shape{"q4e-shared-gate-up", 2560, 1280},
                Shape{"q4e-shared-down", 640, 2560},
                Shape{"q4e-mtp-hidden", 2560, 2560},
                Shape{"small-k", 128, 17, true}, Shape{"non-packed-geometry", 96, 17, true},
                Shape{"lora-and-validation", 512, 17, true, true}}) {
            CheckShape(context, shape);
        }
        context.synchronize();
        context.release_execution_resources();
        std::cout << "[USM] device_bytes=" << context.current_allocated_bytes()
                  << " device_allocations=" << context.device_allocation_count()
                  << " pinned_bytes=" << context.current_pinned_host_bytes()
                  << " pinned_allocations=" << context.pinned_host_allocation_count()
                  << " peak_device_bytes=" << context.peak_allocated_bytes()
                  << " peak_pinned_bytes=" << context.peak_pinned_host_bytes() << '\n';
        Expect(context.current_allocated_bytes() == 0 && context.device_allocation_count() == 0 &&
               context.current_pinned_host_bytes() == 0 && context.pinned_host_allocation_count() == 0,
               "Decode-row fixture leaked tracked USM");
        std::cout << "PASS: NF4 decode-row projection; original-B1 bitwise, CPU base reference, "
                     "LoRA, capacity/shape validation, canaries and USM teardown\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: NF4 decode-row projection: " << error.what() << '\n';
        return 1;
    }
}
