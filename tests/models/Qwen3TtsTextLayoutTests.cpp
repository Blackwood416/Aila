#include "models/Qwen3TtsTextLayout.hpp"

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

void expect_eq_int(TestResults& results,
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

void expect_eq_size(TestResults& results,
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

bool expect_eq_int_vector(const std::vector<int>& actual,
                          const std::vector<int>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

#define AILA_EXPECT_TRUE(results, expression) \
    expect_true((results), (expression), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_INT(results, expression, expected) \
    expect_eq_int((results), static_cast<int>(expression), \
                  static_cast<int>(expected), #expression, __FILE__, __LINE__)

#define AILA_EXPECT_EQ_SIZE(results, expression, expected) \
    expect_eq_size((results), static_cast<size_t>(expression), \
                   static_cast<size_t>(expected), #expression, __FILE__, __LINE__)

void standard_format_parses_body(TestResults& results) {
    const std::vector<int> tokens = {
        151644, 77091, 198,  // <|im_start|>assistant\n
        10, 11, 12,          // body
        151645, 198,         // <|im_end|>\n
        151644, 77091, 198,  // <|im_start|>assistant\n
    };
    const aila::Qwen3TtsTextLayout layout =
        aila::parse_qwen3_tts_text_layout(tokens);

    AILA_EXPECT_EQ_INT(results, layout.text_body_start, 3);
    AILA_EXPECT_EQ_INT(results, layout.text_body_end, 6);
    AILA_EXPECT_TRUE(results, layout.has_im_end);
    AILA_EXPECT_TRUE(results, expect_eq_int_vector(layout.body_tokens, {10, 11, 12}));
}

void missing_im_end_falls_back_to_im_start(TestResults& results) {
    const std::vector<int> tokens = {
        151644, 77091, 198,  // <|im_start|>assistant\n
        10, 11, 12,          // body
        151644, 77091, 198,  // next <|im_start|>assistant\n
    };
    const aila::Qwen3TtsTextLayout layout =
        aila::parse_qwen3_tts_text_layout(tokens);

    AILA_EXPECT_EQ_INT(results, layout.text_body_start, 3);
    AILA_EXPECT_EQ_INT(results, layout.text_body_end, 6);
    AILA_EXPECT_TRUE(results, !layout.has_im_end);
    AILA_EXPECT_TRUE(results, expect_eq_int_vector(layout.body_tokens, {10, 11, 12}));
}

void stream_state_appends_tokens_in_order(TestResults& results) {
    aila::Qwen3TtsStreamTextState state;
    state.Begin({10, 11, 12});

    AILA_EXPECT_TRUE(results, expect_eq_int_vector(state.body_tokens(), {10, 11, 12}));
    AILA_EXPECT_EQ_INT(results, state.first_text_token(), 10);
    AILA_EXPECT_EQ_SIZE(results, state.trailing_len(), 2);

    AILA_EXPECT_TRUE(results, state.Append({13, 14}));
    AILA_EXPECT_TRUE(results, expect_eq_int_vector(
        state.body_tokens(), {10, 11, 12, 13, 14}));
    AILA_EXPECT_EQ_SIZE(results, state.trailing_len(), 4);

    state.Finish();
    AILA_EXPECT_TRUE(results, state.finishing());
    AILA_EXPECT_EQ_SIZE(results, state.trailing_len(), 5);
}

void stream_state_rejects_append_after_finish(TestResults& results) {
    aila::Qwen3TtsStreamTextState state;
    state.Begin({10});
    state.Finish();
    AILA_EXPECT_TRUE(results, !state.Append({11}));
    state.Finish();
    AILA_EXPECT_TRUE(results, state.finishing());
    AILA_EXPECT_TRUE(results, expect_eq_int_vector(state.body_tokens(), {10}));
}

void stream_state_begin_resets_previous_state(TestResults& results) {
    aila::Qwen3TtsStreamTextState state;
    state.Begin({10});
    state.Append({11});
    state.Finish();
    state.Begin({20, 21});

    AILA_EXPECT_TRUE(results, !state.finishing());
    AILA_EXPECT_TRUE(results, expect_eq_int_vector(state.body_tokens(), {20, 21}));
    AILA_EXPECT_EQ_SIZE(results, state.trailing_len(), 1);
}

void empty_body_tokens_keep_state_empty(TestResults& results) {
    aila::Qwen3TtsStreamTextState state;
    state.Begin({});
    AILA_EXPECT_TRUE(results, state.empty());
    AILA_EXPECT_EQ_INT(results, state.first_text_token(), -1);
    AILA_EXPECT_EQ_SIZE(results, state.trailing_len(), 0);
}

}  // namespace

int main() {
    TestResults results;
    standard_format_parses_body(results);
    missing_im_end_falls_back_to_im_start(results);
    stream_state_appends_tokens_in_order(results);
    stream_state_rejects_append_after_finish(results);
    stream_state_begin_resets_previous_state(results);
    empty_body_tokens_keep_state_empty(results);

    std::cout << "AilaQwen3TtsTextLayoutTests: " << results.passed
              << " passed, " << results.failed << " failed\n";
    return results.failed == 0 ? 0 : 1;
}
