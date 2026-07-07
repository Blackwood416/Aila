#include "alia/AliaTurnScheduler.hpp"

#include <iostream>
#include <string>

namespace aila::env {
int g_q35_prefill_step_override = -1;
bool g_kv_quant_override = false;
}

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

void expect_eq(
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

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ(results, left, right) \
    expect_eq((results), (left), (right), #left, #right, __FILE__, __LINE__)

void final_cached_prefix_rejects_large_suffix_by_default(TestResults& results) {
    aila::alia::AliaTurnSchedulerConfig config;
    config.enabled = true;
    config.max_cached_final_suffix_tokens = 16;

    const aila::alia::AliaFinalPrefixDecision decision =
        aila::alia::decide_final_cached_prefix(config, 90, 28);

    AILA_EXPECT_TRUE(results, !decision.use_cached_prefix);
    AILA_EXPECT_EQ(results, decision.reason, "cached suffix exceeds fast threshold");
}

void final_cached_prefix_keeps_fast_suffix_reason(TestResults& results) {
    aila::alia::AliaTurnSchedulerConfig config;
    config.enabled = true;
    config.max_cached_final_suffix_tokens = 16;

    const aila::alia::AliaFinalPrefixDecision decision =
        aila::alia::decide_final_cached_prefix(config, 90, 12);

    AILA_EXPECT_TRUE(results, decision.use_cached_prefix);
    AILA_EXPECT_EQ(results, decision.reason, "cached suffix within fast threshold");
}

void final_prefix_path_names_fresh_prompt(TestResults& results) {
    aila::alia::AliaFinalPrefixDecision decision;
    decision.use_cached_prefix = true;

    AILA_EXPECT_EQ(results,
                   aila::alia::final_prefix_path_name(decision, 0, 118, 16),
                   "fresh_full");
}

void final_prefix_path_names_decode_suffix(TestResults& results) {
    aila::alia::AliaFinalPrefixDecision decision;
    decision.use_cached_prefix = true;

    AILA_EXPECT_EQ(results,
                   aila::alia::final_prefix_path_name(decision, 90, 12, 16),
                   "cached_decode_suffix");
}

void final_prefix_path_names_batch_suffix(TestResults& results) {
    aila::alia::AliaFinalPrefixDecision decision;
    decision.use_cached_prefix = true;

    AILA_EXPECT_EQ(results,
                   aila::alia::final_prefix_path_name(decision, 90, 20, 16),
                   "cached_batch_suffix");
}

void final_prefix_path_names_rejected_suffix(TestResults& results) {
    aila::alia::AliaFinalPrefixDecision decision;
    decision.use_cached_prefix = false;

    AILA_EXPECT_EQ(results,
                   aila::alia::final_prefix_path_name(decision, 90, 28, 16),
                   "rejected_large_suffix");
}

}  // namespace

int main() {
    TestResults results;
    final_cached_prefix_rejects_large_suffix_by_default(results);
    final_cached_prefix_keeps_fast_suffix_reason(results);
    final_prefix_path_names_fresh_prompt(results);
    final_prefix_path_names_decode_suffix(results);
    final_prefix_path_names_batch_suffix(results);
    final_prefix_path_names_rejected_suffix(results);

    std::cout << "AilaAliaTurnSchedulerTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
