#include "vision/Qwen35VisionWeights.hpp"

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

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

void temporal_pair_patch_source_is_releasable(TestResults& results) {
    constexpr int hidden_size = 1024;
    constexpr int patch_size = 16;
    constexpr int64_t temporal_pair_numel =
        static_cast<int64_t>(hidden_size) * 3 * 2 * patch_size * patch_size;

    AILA_EXPECT_TRUE(
        results,
        aila::vision::qwen35_visual_temporal_patch_weight_can_release(
            temporal_pair_numel, hidden_size, patch_size));
}

void spatial_patch_source_is_not_releasable(TestResults& results) {
    constexpr int hidden_size = 1024;
    constexpr int patch_size = 16;
    constexpr int64_t spatial_numel =
        static_cast<int64_t>(hidden_size) * 3 * patch_size * patch_size;

    AILA_EXPECT_TRUE(
        results,
        !aila::vision::qwen35_visual_temporal_patch_weight_can_release(
            spatial_numel, hidden_size, patch_size));
}

void invalid_shape_inputs_are_not_releasable(TestResults& results) {
    AILA_EXPECT_TRUE(
        results,
        !aila::vision::qwen35_visual_temporal_patch_weight_can_release(0, 1024, 16));
    AILA_EXPECT_TRUE(
        results,
        !aila::vision::qwen35_visual_temporal_patch_weight_can_release(1024, 0, 16));
    AILA_EXPECT_TRUE(
        results,
        !aila::vision::qwen35_visual_temporal_patch_weight_can_release(1024, 1024, 0));
}

}  // namespace

int main() {
    TestResults results;
    temporal_pair_patch_source_is_releasable(results);
    spatial_patch_source_is_not_releasable(results);
    invalid_shape_inputs_are_not_releasable(results);

    std::cout << "AilaQwen35VisionWeightsTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
