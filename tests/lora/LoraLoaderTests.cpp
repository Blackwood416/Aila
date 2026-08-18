#include "lora/LoraLoader.hpp"

#include <iostream>
#include <string>

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_eq(TestResults& results,
               const std::string& actual,
               const std::string& expected,
               const char* expression,
               const char* file,
               int line) {
    if (actual == expected) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << expression
              << " == \"" << expected << "\", got \"" << actual << "\"\n";
}

#define AILA_EXPECT_EQ(results, expression, expected) \
    expect_eq((results), (expression), (expected), #expression, __FILE__, __LINE__)

void normalizes_supported_peft_prefixes(TestResults& results) {
    AILA_EXPECT_EQ(
        results,
        aila::lora::LoraLoader::peft_key_to_base_name(
            "base_model.model.model.language_model.layers.11.self_attn.q_proj.lora_A.weight"),
        "model.layers.11.self_attn.q_proj.weight");
    AILA_EXPECT_EQ(
        results,
        aila::lora::LoraLoader::peft_key_to_base_name(
            "base_model.model.model.layers.3.self_attn.o_proj.lora_B.weight"),
        "model.layers.3.self_attn.o_proj.weight");
    AILA_EXPECT_EQ(
        results,
        aila::lora::LoraLoader::peft_key_to_base_name(
            "base_model.model.layers.7.mlp.down_proj.lora_A.weight"),
        "model.layers.7.mlp.down_proj.weight");
    AILA_EXPECT_EQ(
        results,
        aila::lora::LoraLoader::peft_key_to_base_name(
            "base_model.model.thinker.model.layers.2.self_attn.v_proj.lora_B.weight"),
        "thinker.model.layers.2.self_attn.v_proj.weight");
}

void rejects_non_lora_or_unknown_keys(TestResults& results) {
    AILA_EXPECT_EQ(
        results,
        aila::lora::LoraLoader::peft_key_to_base_name(
            "other.model.layers.0.self_attn.q_proj.lora_A.weight"),
        "");
    AILA_EXPECT_EQ(
        results,
        aila::lora::LoraLoader::peft_key_to_base_name(
            "base_model.model.model.layers.0.self_attn.q_proj.weight"),
        "");
}

}  // namespace

int main() {
    TestResults results;
    normalizes_supported_peft_prefixes(results);
    rejects_non_lora_or_unknown_keys(results);

    std::cout << "AilaLoraLoaderTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
