#include "models/cpu/CpuSafetensorsStore.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

void expect_eq_size(TestResults& results, size_t actual, size_t expected,
                    const char* expression, const char* file, int line) {
    if (actual == expected) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression
              << " == " << expected << ", got " << actual << '\n';
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

#define AILA_EXPECT_EQ_SIZE(results, expression, expected) \
    expect_eq_size((results), static_cast<size_t>(expression), \
                   static_cast<size_t>(expected), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_I64(results, expression, expected) \
    expect_eq_i64((results), static_cast<int64_t>(expression), \
                  static_cast<int64_t>(expected), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_NEAR(results, expression, expected, tolerance) \
    expect_near((results), static_cast<float>(expression), \
                static_cast<float>(expected), static_cast<float>(tolerance), \
                #expression, __FILE__, __LINE__)

void write_u64_le(std::ofstream& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        const uint8_t byte = static_cast<uint8_t>((value >> (i * 8)) & 0xff);
        out.write(reinterpret_cast<const char*>(&byte), 1);
    }
}

void write_tiny_safetensors(const std::filesystem::path& path) {
    const std::string header =
        R"({"tensor_a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
        R"("tensor_b":{"dtype":"U8","shape":[3],"data_offsets":[8,11]}})";

    std::ofstream out(path, std::ios::binary);
    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const float a_values[2] = {1.0f, 2.0f};
    const uint8_t b_values[3] = {4, 5, 6};
    out.write(reinterpret_cast<const char*>(a_values), sizeof(a_values));
    out.write(reinterpret_cast<const char*>(b_values), sizeof(b_values));
}

void load_single_file_exposes_host_tensor_views(TestResults& results) {
    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "aila_cpu_safetensors_store_test";
    std::filesystem::create_directories(temp_dir);
    write_tiny_safetensors(temp_dir / "model.safetensors");

    CpuSafetensorsStore store;
    std::string error;
    const bool loaded = store.load_from_dir(temp_dir.string(), &error);

    AILA_EXPECT_TRUE(results, loaded);
    AILA_EXPECT_TRUE(results, error.empty());
    AILA_EXPECT_TRUE(results, store.has("tensor_a"));
    AILA_EXPECT_TRUE(results, store.has("tensor_b"));

    const CpuTensorView& tensor_a = store.get("tensor_a");
    AILA_EXPECT_TRUE(results, tensor_a.dtype == CpuDataType::F32);
    AILA_EXPECT_EQ_SIZE(results, tensor_a.shape.size(), 1);
    AILA_EXPECT_EQ_I64(results, tensor_a.shape[0], 2);
    AILA_EXPECT_EQ_SIZE(results, tensor_a.bytes, 8);
    AILA_EXPECT_NEAR(results, tensor_a.f32_data()[0], 1.0f, 0.0001f);
    AILA_EXPECT_NEAR(results, tensor_a.f32_data()[1], 2.0f, 0.0001f);

    const CpuTensorView& tensor_b = store.get("tensor_b");
    AILA_EXPECT_TRUE(results, tensor_b.dtype == CpuDataType::U8);
    AILA_EXPECT_EQ_SIZE(results, tensor_b.shape.size(), 1);
    AILA_EXPECT_EQ_I64(results, tensor_b.shape[0], 3);
    AILA_EXPECT_EQ_SIZE(results, tensor_b.bytes, 3);
    AILA_EXPECT_EQ_I64(results, tensor_b.u8_data()[2], 6);

    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);
}

}  // namespace

int main() {
    TestResults results;
    load_single_file_exposes_host_tensor_views(results);

    std::cout << "AilaCpuSafetensorsStoreTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
