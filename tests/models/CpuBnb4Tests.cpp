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

}  // namespace

int main() {
    TestResults results;
    parse_quant_state_reads_nested_fields(results);
    nf4_matvec_matches_hand_computed_fixture(results);

    std::cout << "AilaCpuBnb4Tests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
