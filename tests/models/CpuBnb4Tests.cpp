#include "models/cpu/CpuBnb4.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_true(TestResults& results, bool value, const char* expression,
                 const char* file, int line) {
    if (value) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression << '\n';
}

void expect_eq_i64(TestResults& results, int64_t actual, int64_t expected,
                   const char* expression, const char* file, int line) {
    if (actual == expected) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression
              << " == " << expected << ", got " << actual << '\n';
}

void expect_eq_string(TestResults& results, const std::string& actual,
                      const std::string& expected, const char* expression,
                      const char* file, int line) {
    if (actual == expected) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression
              << " == \"" << expected << "\", got \"" << actual << "\"\n";
}

void expect_near(TestResults& results, float actual, float expected,
                 float tolerance, const char* expression,
                 const char* file, int line) {
    if (std::fabs(actual - expected) <= tolerance) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression
              << " ~= " << expected << ", got " << actual << '\n';
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_I64(results, expression, expected) \
    expect_eq_i64((results), static_cast<int64_t>(expression), \
                  static_cast<int64_t>(expected), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_STRING(results, expression, expected) \
    expect_eq_string((results), (expression), (expected), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_NEAR(results, expression, expected, tolerance) \
    expect_near((results), static_cast<float>(expression), \
                static_cast<float>(expected), static_cast<float>(tolerance), \
                #expression, __FILE__, __LINE__)

CpuTensorView make_view(const std::string& name,
                        CpuDataType dtype,
                        std::vector<int64_t> shape,
                        const void* data,
                        size_t bytes) {
    CpuTensorView view;
    view.name = name;
    view.dtype = dtype;
    view.shape = std::move(shape);
    view.data = static_cast<const uint8_t*>(data);
    view.bytes = bytes;
    return view;
}

void parse_quant_state_reads_nested_fields(TestResults& results) {
    const std::string json =
        R"({"quant_type":"nf4","dtype":"float32","blocksize":4,)"
        R"("shape":[2,4],"nested_blocksize":2,)"
        R"("nested_dtype":"float32","nested_offset":0.5})";

    CpuBnb4QuantState state;
    std::string error;
    const bool parsed = parse_cpu_bnb4_quant_state_json(json, state, &error);

    AILA_EXPECT_TRUE(results, parsed);
    AILA_EXPECT_TRUE(results, error.empty());
    AILA_EXPECT_EQ_STRING(results, state.quant_type, "nf4");
    AILA_EXPECT_EQ_STRING(results, state.dtype, "float32");
    AILA_EXPECT_EQ_I64(results, state.blocksize, 4);
    AILA_EXPECT_EQ_I64(results, state.shape.size(), 2);
    AILA_EXPECT_EQ_I64(results, state.shape[0], 2);
    AILA_EXPECT_EQ_I64(results, state.shape[1], 4);
    AILA_EXPECT_TRUE(results, state.nested);
    AILA_EXPECT_EQ_I64(results, state.nested_blocksize, 2);
    AILA_EXPECT_EQ_STRING(results, state.nested_dtype, "float32");
    AILA_EXPECT_NEAR(results, state.nested_offset, 0.5f, 0.0001f);
}

void cache_mode_parser_defaults_to_fp16(TestResults& results) {
    AILA_EXPECT_EQ_I64(
        results,
        static_cast<int>(parse_cpu_bnb4_cache_mode("fp16")),
        static_cast<int>(CpuBnb4CacheMode::Fp16));
    AILA_EXPECT_EQ_I64(
        results,
        static_cast<int>(parse_cpu_bnb4_cache_mode("PACKED_NF4")),
        static_cast<int>(CpuBnb4CacheMode::PackedNf4));
    AILA_EXPECT_EQ_I64(
        results,
        static_cast<int>(parse_cpu_bnb4_cache_mode("invalid")),
        static_cast<int>(CpuBnb4CacheMode::Fp16));
}

void nf4_matvec_matches_hand_computed_fixture(TestResults& results) {
    const std::vector<uint8_t> packed = {0x12, 0x34, 0x56, 0x78};
    const std::vector<float> absmax = {0.1f, 0.2f};
    std::vector<float> quant_map(16);
    for (int i = 0; i < 16; ++i) {
        quant_map[static_cast<size_t>(i)] = static_cast<float>(i);
    }

    const CpuTensorView packed_view =
        make_view("linear.weight", CpuDataType::U8, {4}, packed.data(), packed.size());
    const CpuTensorView absmax_view =
        make_view("linear.weight.absmax", CpuDataType::F32, {2},
                  absmax.data(), absmax.size() * sizeof(float));
    const CpuTensorView quant_map_view =
        make_view("linear.weight.quant_map", CpuDataType::F32, {16},
                  quant_map.data(), quant_map.size() * sizeof(float));

    CpuBnb4WeightRef weight;
    weight.name = "linear.weight";
    weight.packed_weight = &packed_view;
    weight.absmax = &absmax_view;
    weight.quant_map = &quant_map_view;
    weight.quant_state.quant_type = "nf4";
    weight.quant_state.blocksize = 4;
    weight.quant_state.shape = {2, 4};

    const float input[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[2] = {0.0f, 0.0f};

    cpu_bnb4_matvec(weight, input, output);

    AILA_EXPECT_NEAR(results, output[0], (1.0f + 2.0f + 3.0f + 4.0f) * 0.1f, 0.0001f);
    AILA_EXPECT_NEAR(results, output[1], (5.0f + 6.0f + 7.0f + 8.0f) * 0.2f, 0.0001f);
}

void packed_nf4_cache_matches_hand_computed_fixture(TestResults& results) {
    const std::vector<uint8_t> packed = {0x21, 0x43, 0x65, 0x87};
    const std::vector<float> absmax = {0.1f, 0.2f};
    std::vector<float> quant_map(16);
    for (int i = 0; i < 16; ++i) {
        quant_map[static_cast<size_t>(i)] = static_cast<float>(i);
    }

    CpuBnb4WeightRef weight;
    weight.cache_mode = CpuBnb4CacheMode::PackedNf4;
    weight.quant_state.blocksize = 4;
    weight.quant_state.shape = {2, 4};
    weight.packed_nf4_codes = packed;
    weight.packed_nf4_absmax = absmax;
    weight.quant_map = nullptr;

    AILA_EXPECT_TRUE(results, weight.dense_weight_f16.empty());
    AILA_EXPECT_EQ_I64(results, weight.packed_nf4_codes.size(), 4);
    AILA_EXPECT_EQ_I64(results, weight.packed_nf4_absmax.size(), 2);
    AILA_EXPECT_EQ_I64(results, weight.cache_bytes(), 12);

    const float input[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[2] = {};
    cpu_nf4_matvec_scalar(weight, quant_map.data(), input, output);
    AILA_EXPECT_NEAR(results, output[0], (1.0f + 2.0f + 3.0f + 4.0f) * 0.1f, 0.0001f);
    AILA_EXPECT_NEAR(results, output[1], (5.0f + 6.0f + 7.0f + 8.0f) * 0.2f, 0.0001f);

    float dispatched[2] = {};
    cpu_nf4_matvec(weight, quant_map.data(), input, dispatched);
    AILA_EXPECT_NEAR(results, dispatched[0], output[0], 0.0002f);
    AILA_EXPECT_NEAR(results, dispatched[1], output[1], 0.0002f);
}

void fp16_conversion_and_dot_match_reference(TestResults& results) {
    const float values[6] = {0.0f, 1.0f, -2.0f, 0.33325f, 65504.0f, 0.00006103515625f};
    uint16_t packed[6] = {};
    float unpacked[6] = {};
    for (int i = 0; i < 6; ++i) {
        packed[i] = cpu_float_to_f16(values[i]);
    }
    cpu_f16_to_f32(packed, unpacked, 6);
    for (int i = 0; i < 6; ++i) {
        AILA_EXPECT_NEAR(results, unpacked[i], values[i], 0.0005f);
    }
    const float smallest_subnormal = 0.000000059604644775390625f;
    AILA_EXPECT_EQ_I64(results, cpu_float_to_f16(smallest_subnormal), 1);
    AILA_EXPECT_NEAR(results, cpu_f16_to_float(1), smallest_subnormal, 1e-12f);

    const float input[6] = {1.0f, 2.0f, 3.0f, 4.0f, 0.001f, -2.0f};
    float expected = 0.0f;
    for (int i = 0; i < 6; ++i) {
        expected += unpacked[i] * input[i];
    }
    AILA_EXPECT_NEAR(results, cpu_f16_dot_f32(packed, input, 6), expected, 0.01f);
}

void repeated_parallel_dense_matvec_matches_reference(TestResults& results) {
    constexpr int kRows = 256;
    constexpr int kCols = 256;
    const std::vector<uint8_t> packed(static_cast<size_t>(kRows * kCols / 2), 0);
    const std::vector<float> absmax(static_cast<size_t>(kRows * kCols / 64), 1.0f);
    const std::vector<float> quant_map(16, 0.0f);
    const CpuTensorView packed_view =
        make_view("parallel.weight", CpuDataType::U8, {kRows * kCols / 2},
                  packed.data(), packed.size());
    const CpuTensorView absmax_view =
        make_view("parallel.weight.absmax", CpuDataType::F32,
                  {static_cast<int64_t>(absmax.size())}, absmax.data(),
                  absmax.size() * sizeof(float));
    const CpuTensorView quant_map_view =
        make_view("parallel.weight.quant_map", CpuDataType::F32, {16},
                  quant_map.data(), quant_map.size() * sizeof(float));

    CpuBnb4WeightRef weight;
    weight.name = "parallel.weight";
    weight.packed_weight = &packed_view;
    weight.absmax = &absmax_view;
    weight.quant_map = &quant_map_view;
    weight.quant_state.quant_type = "nf4";
    weight.quant_state.blocksize = 64;
    weight.quant_state.shape = {kRows, kCols};
    weight.dense_weight_f16.assign(
        static_cast<size_t>(kRows * kCols), cpu_float_to_f16(1.0f));

    const std::vector<float> input(kCols, 1.0f);
    std::vector<float> output(kRows, 0.0f);
    cpu_bnb4_matvec(weight, input.data(), output.data());
    cpu_bnb4_matvec(weight, input.data(), output.data());

    for (float value : output) {
        AILA_EXPECT_NEAR(results, value, static_cast<float>(kCols), 0.0001f);
    }
}

}  // namespace

int main() {
    TestResults results;
    parse_quant_state_reads_nested_fields(results);
    cache_mode_parser_defaults_to_fp16(results);
    nf4_matvec_matches_hand_computed_fixture(results);
    packed_nf4_cache_matches_hand_computed_fixture(results);
    fp16_conversion_and_dot_match_reference(results);
    repeated_parallel_dense_matvec_matches_reference(results);

    std::cout << "AilaCpuBnb4Tests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
