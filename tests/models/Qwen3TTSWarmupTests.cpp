#include "models/Qwen3TTSWarmup.hpp"

#include <cmath>
#include <iostream>

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

void expect_eq_int(
    TestResults& results,
    int actual,
    int expected,
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

#define AILA_EXPECT_EQ_INT(results, expression, expected) \
    expect_eq_int((results), static_cast<int>(expression), \
                  static_cast<int>(expected), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_SIZE(results, expression, expected) \
    expect_eq_size((results), static_cast<size_t>(expression), \
                   static_cast<size_t>(expected), #expression, __FILE__, __LINE__)

void warmup_text_matches_formatted_minimal_tts_text(TestResults& results) {
    const std::vector<int> tokens = qwen3_tts_minimal_warmup_text_tokens();

    AILA_EXPECT_EQ_SIZE(results, tokens.size(), 9);
    AILA_EXPECT_EQ_INT(results, tokens[0], 151644);
    AILA_EXPECT_EQ_INT(results, tokens[1], 77091);
    AILA_EXPECT_EQ_INT(results, tokens[2], 198);
    AILA_EXPECT_EQ_INT(results, tokens[4], 151645);
    AILA_EXPECT_EQ_INT(results, tokens[5], 198);
    AILA_EXPECT_EQ_INT(results, tokens[6], 151644);
    AILA_EXPECT_EQ_INT(results, tokens[7], 77091);
    AILA_EXPECT_EQ_INT(results, tokens[8], 198);
}

void base_warmup_uses_dummy_speaker_embedding(TestResults& results) {
    const std::vector<float> embedding =
        qwen3_tts_warmup_speaker_embedding(Qwen3TTSModelType::Base, 1024);

    AILA_EXPECT_EQ_SIZE(results, embedding.size(), 1024);
    AILA_EXPECT_TRUE(results, std::abs(embedding.front()) == 0.0f);
    AILA_EXPECT_TRUE(results, std::abs(embedding.back()) == 0.0f);
}

void non_base_warmup_leaves_speaker_embedding_empty(TestResults& results) {
    AILA_EXPECT_TRUE(
        results,
        qwen3_tts_warmup_speaker_embedding(
            Qwen3TTSModelType::CustomVoice, 1024).empty());
    AILA_EXPECT_TRUE(
        results,
        qwen3_tts_warmup_speaker_embedding(
            Qwen3TTSModelType::VoiceDesign, 1024).empty());
    AILA_EXPECT_TRUE(
        results,
        qwen3_tts_warmup_speaker_embedding(
            Qwen3TTSModelType::Base, 0).empty());
}

}  // namespace

int main() {
    TestResults results;
    warmup_text_matches_formatted_minimal_tts_text(results);
    base_warmup_uses_dummy_speaker_embedding(results);
    non_base_warmup_leaves_speaker_embedding_empty(results);

    std::cout << "AilaQwen3TTSWarmupTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
