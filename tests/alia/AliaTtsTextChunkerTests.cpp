#include "alia/AliaTtsTextChunker.hpp"

#include <iostream>
#include <string>
#include <vector>

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

void expect_eq_string(
    TestResults& results,
    const std::string& left,
    const std::string& right,
    const char* left_expression,
    const char* right_expression,
    const char* file,
    int line) {
    if (left == right) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << left_expression
              << " == " << right_expression << ", got \"" << left
              << "\" vs \"" << right << "\"\n";
}

void expect_eq_size(
    TestResults& results,
    size_t left,
    size_t right,
    const char* left_expression,
    const char* right_expression,
    const char* file,
    int line) {
    if (left == right) {
        ++results.passed;
        return;
    }
    ++results.failed;
    std::cerr << file << ':' << line << ": expected " << left_expression
              << " == " << right_expression << ", got " << left
              << " vs " << right << '\n';
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_STRING(results, left, right) \
    expect_eq_string((results), (left), (right), #left, #right, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_SIZE(results, left, right) \
    expect_eq_size((results), (left), (right), #left, #right, __FILE__, __LINE__)

aila::alia::TtsTextChunkPolicy test_policy() {
    aila::alia::TtsTextChunkPolicy policy;
    policy.first_soft_min_chars = 5;
    policy.steady_soft_min_chars = 24;
    policy.first_soft_max_chars = 48;
    policy.steady_soft_max_chars = 120;
    policy.first_hard_min_chars = 8;
    policy.steady_hard_min_chars = 96;
    policy.first_soft_boundary_flush = true;
    policy.coalesce_steady_text_chunks = false;
    return policy;
}

void hard_boundary_below_first_min_waits(TestResults& results) {
    std::string buffer = "hi.";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        false,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_TRUE(results, result.chunks.empty());
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "below_hard_min");
    AILA_EXPECT_EQ_STRING(results, buffer, "hi.");
}

void first_soft_boundary_flushes_after_soft_min(TestResults& results) {
    std::string buffer = "hello, next";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        false,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_EQ_SIZE(results, result.chunks.size(), 1);
    AILA_EXPECT_EQ_STRING(results, result.chunks.front(), "hello,");
    AILA_EXPECT_EQ_STRING(results, buffer, "next");
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "soft_boundary_flush");
}

void early_first_chunk_flushes_without_boundary(TestResults& results) {
    std::string buffer = "abcdefgh";
    const aila::alia::TtsTextChunkRequest request{
        false,
        true,
        true,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_EQ_SIZE(results, result.chunks.size(), 1);
    AILA_EXPECT_EQ_STRING(results, result.chunks.front(), "abcdefgh");
    AILA_EXPECT_TRUE(results, buffer.empty());
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "early_first_chunk");
}

void steady_chunk_ignores_first_chunk_early_flush(TestResults& results) {
    std::string buffer = "abcdefgh";
    const aila::alia::TtsTextChunkRequest request{
        false,
        false,
        true,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_TRUE(results, result.chunks.empty());
    AILA_EXPECT_EQ_STRING(results, buffer, "abcdefgh");
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "below_hard_min");
}

void force_flushes_all_text(TestResults& results) {
    std::string buffer = "abc";
    const aila::alia::TtsTextChunkRequest request{
        true,
        true,
        false,
    };

    const aila::alia::TtsTextChunkResult result =
        aila::alia::take_ready_tts_text_chunks(buffer, test_policy(), request);

    AILA_EXPECT_EQ_SIZE(results, result.chunks.size(), 1);
    AILA_EXPECT_EQ_STRING(results, result.chunks.front(), "abc");
    AILA_EXPECT_TRUE(results, buffer.empty());
    AILA_EXPECT_EQ_STRING(
        results,
        aila::alia::tts_chunk_decision_reason_name(result.reason),
        "force");
}

}  // namespace

int main() {
    TestResults results;
    hard_boundary_below_first_min_waits(results);
    first_soft_boundary_flushes_after_soft_min(results);
    early_first_chunk_flushes_without_boundary(results);
    steady_chunk_ignores_first_chunk_early_flush(results);
    force_flushes_all_text(results);

    std::cout << "AilaAliaTtsTextChunkerTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
