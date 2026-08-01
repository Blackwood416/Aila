#include "ops/SamplingSuppress.hpp"

#include <iostream>
#include <vector>

namespace {

struct TestResults {
    int passed = 0;
    int failed = 0;
};

void expect_true(TestResults& results,
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

void null_suppress_list_allows_all_tokens(TestResults& results) {
    AILA_EXPECT_TRUE(results, !ops::is_token_suppressed(2150, nullptr));
}

void empty_suppress_list_allows_all_tokens(TestResults& results) {
    const std::vector<int> empty;
    AILA_EXPECT_TRUE(results, !ops::is_token_suppressed(2150, &empty));
}

void suppress_list_marks_only_matching_tokens(TestResults& results) {
    const std::vector<int> suppress = {2150, 4204};
    AILA_EXPECT_TRUE(results, ops::is_token_suppressed(2150, &suppress));
    AILA_EXPECT_TRUE(results, ops::is_token_suppressed(4204, &suppress));
    AILA_EXPECT_TRUE(results, !ops::is_token_suppressed(0, &suppress));
}

}  // namespace

int main() {
    TestResults results;
    null_suppress_list_allows_all_tokens(results);
    empty_suppress_list_allows_all_tokens(results);
    suppress_list_marks_only_matching_tokens(results);

    std::cout << "AilaSamplingSuppressTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
