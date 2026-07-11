#include "models/cpu/CpuQ35HybridModel.hpp"

#include <cmath>
#include <atomic>
#include <iostream>
#include <string>

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

void expect_contains(TestResults& results, const std::string& value,
                     const std::string& needle, const char* expression,
                     const char* file, int line) {
    if (value.find(needle) != std::string::npos) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression
              << " to contain \"" << needle << "\", got \"" << value << "\"\n";
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_CONTAINS(results, expression, needle) \
    expect_contains((results), (expression), (needle), #expression, __FILE__, __LINE__)

ModelSpec make_0p8b_q35_spec() {
    ModelSpec spec;
    spec.family = ModelFamily::Qwen35Hybrid;
    spec.model_type = "qwen3_5";
    spec.quantization.quant_method = "bitsandbytes";
    spec.quantization.load_in_4bit = true;
    spec.quantization.bnb_4bit_quant_type = "nf4";
    return spec;
}

void rejects_non_0p8b_spec(TestResults& results) {
    ModelSpec spec = make_0p8b_q35_spec();
    spec.qwen35_text.hidden_size = 2560;
    spec.qwen35_text.num_attention_heads = 16;
    spec.qwen35_text.num_key_value_heads = 4;
    spec.qwen35_text.num_hidden_layers = 32;
    spec.qwen35_text.intermediate_size = 9216;

    CpuSafetensorsStore store;
    CpuQ35HybridModel model;
    std::string error;
    const bool loaded = model.load_from_store(store, spec, 2048, &error);

    AILA_EXPECT_TRUE(results, !loaded);
    AILA_EXPECT_CONTAINS(results, error, "CPU Qwen3.5 backend supports only 0.8B");
    AILA_EXPECT_TRUE(results, !model.loaded());
}

void load_missing_required_weight_reports_name(TestResults& results) {
    const ModelSpec spec = make_0p8b_q35_spec();
    CpuSafetensorsStore store;
    CpuQ35HybridModel model;
    std::string error;

    const bool loaded = model.load_from_store(store, spec, 2048, &error);

    AILA_EXPECT_TRUE(results, !loaded);
    AILA_EXPECT_CONTAINS(results, error, "model.language_model.embed_tokens.weight");
    AILA_EXPECT_TRUE(results, !model.loaded());
}

void rms_norm_q35_adds_one_to_weight(TestResults& results) {
    const float input[2] = {2.0f, 0.0f};
    const float raw_weight[2] = {0.0f, 0.0f};
    float output[2] = {0.0f, 0.0f};

    cpu_q35::q35_rms_norm(input, raw_weight, 2, 0.0f, output);

    AILA_EXPECT_TRUE(results, std::abs(output[0] - std::sqrt(2.0f)) < 0.0001f);
    AILA_EXPECT_TRUE(results, std::abs(output[1]) < 0.0001f);
}

void sigmoid_gate_matches_reference(TestResults& results) {
    const float input[1] = {3.0f};
    const float gate[1] = {0.0f};
    float output[1] = {0.0f};

    cpu_q35::sigmoid_gate(input, gate, 1, output);

    AILA_EXPECT_TRUE(results, std::abs(output[0] - 1.5f) < 0.0001f);
}

void prefill_before_load_reports_error(TestResults& results) {
    CpuQ35HybridModel model;
    const std::vector<int> tokens = {1, 2};
    std::vector<float> logits;
    std::atomic_bool abort_requested{false};
    std::string error;
    const bool ok = model.prefill(
        tokens, 1, &abort_requested, &logits, &error);
    AILA_EXPECT_TRUE(results, !ok);
    AILA_EXPECT_CONTAINS(results, error, "before load");
}

void prefill_batch_parser_accepts_supported_values(TestResults& results) {
    AILA_EXPECT_TRUE(results, parse_cpu_q35_prefill_batch("1") == 1);
    AILA_EXPECT_TRUE(results, parse_cpu_q35_prefill_batch("2") == 2);
    AILA_EXPECT_TRUE(results, parse_cpu_q35_prefill_batch("4") == 4);
    AILA_EXPECT_TRUE(results, parse_cpu_q35_prefill_batch("8") == 8);
    AILA_EXPECT_TRUE(results, parse_cpu_q35_prefill_batch("3") == 1);
    AILA_EXPECT_TRUE(results, parse_cpu_q35_prefill_batch("invalid") == 1);
}

}  // namespace

int main() {
    TestResults results;
    rejects_non_0p8b_spec(results);
    load_missing_required_weight_reports_name(results);
    rms_norm_q35_adds_one_to_weight(results);
    sigmoid_gate_matches_reference(results);
    prefill_before_load_reports_error(results);
    prefill_batch_parser_accepts_supported_values(results);

    std::cout << "AilaCpuQ35HybridModelTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
