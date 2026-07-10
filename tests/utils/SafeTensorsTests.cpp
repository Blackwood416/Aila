#include "utils/SafeTensors.hpp"

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

}  // namespace

int main() {
    TestResults results;
    erase_with_prefix_removes_only_matching_weights(results);
    erase_with_prefix_ignores_empty_prefix(results);

    std::cout << "AilaSafeTensorsTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
