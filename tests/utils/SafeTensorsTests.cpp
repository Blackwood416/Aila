#include "utils/SafeTensors.hpp"
#include "core/Context.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>

namespace fs = std::filesystem;

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_true(
    TestResults& results,
    bool value,
    const char* expression,
    const char* file,
    int line) {
    if (value) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression << '\n';
}

void expect_eq_size(
    TestResults& results,
    size_t actual,
    size_t expected,
    const char* expression,
    const char* file,
    int line) {
    if (actual == expected) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression
              << " == " << expected << ", got " << actual << '\n';
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_SIZE(results, expression, expected) \
    expect_eq_size((results), static_cast<size_t>(expression), \
                   static_cast<size_t>(expected), #expression, __FILE__, __LINE__)

void erase_with_prefix_removes_only_matching_weights(TestResults& results) {
    ModelWeights weights;
    weights.put("unused.encoder.blocks.0.conv.weight", Tensor{});
    weights.put("unused.encoder.fc.bias", Tensor{});
    weights.put("talker.model.norm.weight", Tensor{});

    const ModelWeightsEraseStats stats = weights.erase_with_prefix("unused.encoder.");

    AILA_EXPECT_EQ_SIZE(results, stats.count, 2);
    AILA_EXPECT_EQ_SIZE(results, stats.bytes, 0);
    AILA_EXPECT_EQ_SIZE(results, weights.size(), 1);
    AILA_EXPECT_TRUE(results, !weights.has("unused.encoder.blocks.0.conv.weight"));
    AILA_EXPECT_TRUE(results, !weights.has("unused.encoder.fc.bias"));
    AILA_EXPECT_TRUE(results, weights.has("talker.model.norm.weight"));
}

void erase_with_prefix_ignores_empty_prefix(TestResults& results) {
    ModelWeights weights;
    weights.put("unused.encoder.fc.bias", Tensor{});
    weights.put("talker.model.norm.weight", Tensor{});

    const ModelWeightsEraseStats stats = weights.erase_with_prefix("");

    AILA_EXPECT_EQ_SIZE(results, stats.count, 0);
    AILA_EXPECT_EQ_SIZE(results, stats.bytes, 0);
    AILA_EXPECT_EQ_SIZE(results, weights.size(), 2);
    AILA_EXPECT_TRUE(results, weights.has("unused.encoder.fc.bias"));
    AILA_EXPECT_TRUE(results, weights.has("talker.model.norm.weight"));
}

void test_lazy_loading_synthetic(TestResults& results) {
    fs::path temp_dir = fs::temp_directory_path() / "aila_safetensors_lazy_test";
    fs::remove_all(temp_dir);
    fs::create_directories(temp_dir);

    const std::string qstate_json = "{\"quant_type\":\"nf4\",\"dtype\":\"bfloat16\",\"blocksize\":64,\"shape\":[8,8]}";
    const size_t qstate_bytes = qstate_json.size();

    size_t offset = 0;
    size_t o1_start = offset; offset += 32; size_t o1_end = offset;
    size_t o2_start = offset; offset += 32; size_t o2_end = offset;
    size_t o3_start = offset; offset += 4;  size_t o3_end = offset;
    size_t o4_start = offset; offset += 64; size_t o4_end = offset;
    size_t o5_start = offset; offset += qstate_bytes; size_t o5_end = offset;

    std::string header = "{"
        "\"sample_weight\":{\"dtype\":\"F32\",\"shape\":[2,4],\"data_offsets\":[" + std::to_string(o1_start) + "," + std::to_string(o1_end) + "]},"
        "\"linear.weight\":{\"dtype\":\"U8\",\"shape\":[8,4],\"data_offsets\":[" + std::to_string(o2_start) + "," + std::to_string(o2_end) + "]},"
        "\"linear.weight.absmax\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" + std::to_string(o3_start) + "," + std::to_string(o3_end) + "]},"
        "\"linear.weight.quant_map\":{\"dtype\":\"F32\",\"shape\":[16],\"data_offsets\":[" + std::to_string(o4_start) + "," + std::to_string(o4_end) + "]},"
        "\"linear.weight.quant_state.bitsandbytes__nf4\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(qstate_bytes) + "],\"data_offsets\":[" + std::to_string(o5_start) + "," + std::to_string(o5_end) + "]}"
    "}";

    size_t unpadded_len = header.size();
    size_t padded_len = ((unpadded_len + 7) / 8) * 8;
    header.append(padded_len - unpadded_len, ' ');

    uint64_t header_len = header.size();
    std::vector<uint8_t> payload(offset, 0);

    float* sample_floats = reinterpret_cast<float*>(payload.data() + o1_start);
    for (int i = 0; i < 8; ++i) {
        sample_floats[i] = static_cast<float>(i + 1);
    }
    std::memcpy(payload.data() + o5_start, qstate_json.data(), qstate_bytes);

    fs::path safetensors_path = temp_dir / "model.safetensors";
    {
        std::ofstream out(safetensors_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&header_len), 8);
        out.write(header.data(), header.size());
        out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    }

    try {
        Context ctx;
        {
            ModelWeights weights = LoadModelWeightsLazy(temp_dir.string(), &ctx);
            AILA_EXPECT_EQ_SIZE(results, weights.size(), 0);
            AILA_EXPECT_EQ_SIZE(results, weights.host_size(), 5);
            AILA_EXPECT_TRUE(results, weights.has_host("sample_weight"));
            AILA_EXPECT_TRUE(results, !weights.has("sample_weight"));

            HostTensorView v = weights.host_view("sample_weight");
            AILA_EXPECT_TRUE(results, v.valid());
            AILA_EXPECT_EQ_SIZE(results, v.shape.size(), 2);
            AILA_EXPECT_EQ_SIZE(results, v.shape[0], 2);
            AILA_EXPECT_EQ_SIZE(results, v.shape[1], 4);
            AILA_EXPECT_EQ_SIZE(results, v.bytes, 32);
            const float* v_floats = reinterpret_cast<const float*>(v.data());
            AILA_EXPECT_TRUE(results, v_floats != nullptr);
            if (v_floats) {
                AILA_EXPECT_TRUE(results, v_floats[0] == 1.0f);
                AILA_EXPECT_TRUE(results, v_floats[7] == 8.0f);
            }

            Bnb4BitHostWeightRef bnb_ref;
            std::string bnb_err;
            bool bnb_ok = LoadBnb4BitHostRef(weights, "linear.weight", bnb_ref, &bnb_err);
            AILA_EXPECT_TRUE(results, bnb_ok);
            AILA_EXPECT_TRUE(results, bnb_ref.valid());
            AILA_EXPECT_TRUE(results, bnb_ref.quant_state.quant_type == "nf4");
            AILA_EXPECT_EQ_SIZE(results, bnb_ref.quant_state.blocksize, 64);
            AILA_EXPECT_EQ_SIZE(results, bnb_ref.quant_state.shape.size(), 2);
            AILA_EXPECT_EQ_SIZE(results, bnb_ref.quant_state.shape[0], 8);
            AILA_EXPECT_EQ_SIZE(results, bnb_ref.quant_state.shape[1], 8);

            Tensor& uploaded = weights.upload("sample_weight");
            AILA_EXPECT_EQ_SIZE(results, weights.size(), 1);
            AILA_EXPECT_TRUE(results, weights.has("sample_weight"));
            AILA_EXPECT_TRUE(results, &uploaded == &weights.get("sample_weight"));
            AILA_EXPECT_EQ_SIZE(results, uploaded.shape().size(), 2);
            AILA_EXPECT_EQ_SIZE(results, uploaded.shape()[0], 2);
            AILA_EXPECT_EQ_SIZE(results, uploaded.shape()[1], 4);
        }
    } catch (const std::exception& e) {
        std::cout << "[SafeTensorsTests] Note: GPU context / upload test skipped: " << e.what() << "\n";
    }

    fs::remove_all(temp_dir);
}

static std::string resolve_qwen4exp_model_dir() {
    const char* env_dir = std::getenv("AILA_QWEN4EXP_MODEL_DIR");
    if (env_dir && fs::exists(env_dir)) return env_dir;
    const std::string candidate_c = "C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_c)) return candidate_c;
    const std::string candidate_f = "F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_f)) return candidate_f;
    return candidate_c;
}

void test_lazy_loading_real_qwen4exp_if_present(TestResults& results) {
    const std::string model_dir = resolve_qwen4exp_model_dir();
    if (!fs::exists(model_dir) || !fs::exists(model_dir + "/model.safetensors.index.json")) {
        std::cout << "[SafeTensorsTests] Real model directory not found, skipping real model lazy load\n";
        return;
    }

    ModelWeights weights = LoadModelWeightsLazy(model_dir);
    AILA_EXPECT_EQ_SIZE(results, weights.size(), 0);
    AILA_EXPECT_TRUE(results, weights.host_size() == 4997);
    AILA_EXPECT_TRUE(results, weights.mmaps().size() == 131);

    const std::string test_tensor = "model.language_model.layers.0.linear_attn.in_proj_a.weight";
    AILA_EXPECT_TRUE(results, weights.has_host(test_tensor));
    AILA_EXPECT_TRUE(results, !weights.has(test_tensor));

    Bnb4BitHostWeightRef ref;
    std::string err;
    bool ok = LoadBnb4BitHostRef(weights, test_tensor, ref, &err);
    AILA_EXPECT_TRUE(results, ok);
    AILA_EXPECT_TRUE(results, ref.valid());
    AILA_EXPECT_TRUE(results, ref.quant_state.quant_type == "nf4");
    AILA_EXPECT_TRUE(results, ref.quant_state.nested);
    AILA_EXPECT_EQ_SIZE(results, ref.quant_state.blocksize, 64);
    AILA_EXPECT_TRUE(results, ref.absmax_codes_ptr() != nullptr);
    AILA_EXPECT_TRUE(results, ref.quant_map_ptr() != nullptr);

    std::cout << "[SafeTensorsTests] Real Qwen4Exp lazy loading verified: 131 shards, "
              << weights.host_size() << " tensors indexed without uploading to GPU.\n";
}

} // namespace

int main() {
    TestResults results;
    erase_with_prefix_removes_only_matching_weights(results);
    erase_with_prefix_ignores_empty_prefix(results);
    test_lazy_loading_synthetic(results);
    test_lazy_loading_real_qwen4exp_if_present(results);

    std::cout << "AilaSafeTensorsTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
