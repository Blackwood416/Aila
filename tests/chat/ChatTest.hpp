#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace aila::chat::test {

struct Registry {
    int passed = 0;
    int failed = 0;
};

inline Registry& registry() {
    static Registry r;
    return r;
}

using TestFn = void (*)();

struct TestCase {
    const char* name;
    TestFn fn;
};

inline void record_failure(
    const char* file,
    int line,
    const char* expression,
    const std::string& message) {
    ++registry().failed;
    std::cerr << file << ':' << line << ": expected " << expression;
    if (!message.empty()) {
        std::cerr << " (" << message << ')';
    }
    std::cerr << '\n';
}

inline void expect_true(bool value, const char* expression, const char* file, int line) {
    if (value) {
        ++registry().passed;
        return;
    }

    record_failure(file, line, expression, "value was false");
}

template <typename Left, typename Right>
inline void expect_eq(
    const Left& left,
    const Right& right,
    const char* leftExpression,
    const char* rightExpression,
    const char* file,
    int line) {
    if (left == right) {
        ++registry().passed;
        return;
    }

    std::ostringstream message;
    message << leftExpression << "=" << left << ", " << rightExpression << "=" << right;
    record_failure(file, line, "values to be equal", message.str());
}

} // namespace aila::chat::test

#define AILA_CHAT_EXPECT_TRUE(expression) \
    ::aila::chat::test::expect_true((expression), #expression, __FILE__, __LINE__)

#define AILA_CHAT_EXPECT_EQ(left, right) \
    ::aila::chat::test::expect_eq((left), (right), #left, #right, __FILE__, __LINE__)
