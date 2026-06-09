#include "ChatTest.hpp"
#include "chat/AssistantOutputParser.hpp"

#include <string>

namespace aila::chat::test {

void run_assistant_output_parser_tests() {
    {
        const auto result = parse_assistant_output("hello");
        AILA_CHAT_EXPECT_EQ(result.raw_text, std::string("hello"));
        AILA_CHAT_EXPECT_EQ(result.content, std::string("hello"));
        AILA_CHAT_EXPECT_EQ(result.reasoning_content, std::string(""));
        AILA_CHAT_EXPECT_TRUE(result.tool_calls.empty());
        AILA_CHAT_EXPECT_TRUE(result.warnings.empty());
    }

    {
        const auto result = parse_assistant_output("<think>\nplan\n</think>\n\nanswer");
        AILA_CHAT_EXPECT_EQ(result.raw_text, std::string("<think>\nplan\n</think>\n\nanswer"));
        AILA_CHAT_EXPECT_EQ(result.reasoning_content, std::string("plan"));
        AILA_CHAT_EXPECT_EQ(result.content, std::string("answer"));
        AILA_CHAT_EXPECT_TRUE(result.tool_calls.empty());
        AILA_CHAT_EXPECT_TRUE(result.warnings.empty());
    }

    {
        const auto result = parse_assistant_output("\n</think>\n\nHello!");
        AILA_CHAT_EXPECT_EQ(result.raw_text, std::string("\n</think>\n\nHello!"));
        AILA_CHAT_EXPECT_EQ(result.reasoning_content, std::string(""));
        AILA_CHAT_EXPECT_EQ(result.content, std::string("Hello!"));
        AILA_CHAT_EXPECT_TRUE(result.tool_calls.empty());
        AILA_CHAT_EXPECT_TRUE(result.warnings.empty());
    }

    {
        const auto result = parse_assistant_output(
            "\nThe user wants a short English sentence.\n"
            "I should answer directly.\n"
            "</think>\n\n"
            "Hello there.");
        AILA_CHAT_EXPECT_EQ(result.reasoning_content,
                            std::string("The user wants a short English sentence.\nI should answer directly."));
        AILA_CHAT_EXPECT_EQ(result.content, std::string("Hello there."));
        AILA_CHAT_EXPECT_TRUE(result.tool_calls.empty());
    }

    {
        const auto result = parse_assistant_output("\nHello!\n</think>\n\nHello!");
        AILA_CHAT_EXPECT_EQ(result.reasoning_content, std::string(""));
        AILA_CHAT_EXPECT_EQ(result.content, std::string("Hello!"));
        AILA_CHAT_EXPECT_TRUE(result.tool_calls.empty());
    }

    {
        const auto result = parse_assistant_output(
            "<think>\nplan\n</think>\n"
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>cats</parameter>\n"
            "</function>\n"
            "</tool_call>");
        AILA_CHAT_EXPECT_EQ(result.reasoning_content, std::string("plan"));
        AILA_CHAT_EXPECT_EQ(result.content, std::string(""));
        AILA_CHAT_EXPECT_EQ(static_cast<int>(result.tool_calls.size()), 1);
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].id, std::string("call_0"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].type, std::string("function"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.name, std::string("search"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.arguments_json, std::string(R"({"query":"cats"})"));
        AILA_CHAT_EXPECT_TRUE(result.warnings.empty());
    }

    {
        const auto result = parse_assistant_output(
            "<tool_call>\n"
            "<function=add>\n"
            "<parameter=a>1</parameter>\n"
            "<parameter=b>{\"x\":2}</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=ping>\n"
            "</function>\n"
            "</tool_call>");
        AILA_CHAT_EXPECT_EQ(result.content, std::string(""));
        AILA_CHAT_EXPECT_EQ(static_cast<int>(result.tool_calls.size()), 2);
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].id, std::string("call_0"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.name, std::string("add"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.arguments_json, std::string(R"({"a":"1","b":{"x":2}})"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[1].id, std::string("call_1"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[1].function.name, std::string("ping"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[1].function.arguments_json, std::string("{}"));
        AILA_CHAT_EXPECT_TRUE(result.warnings.empty());
    }

    {
        const std::string raw = "<tool_call>\nnot xml\n</tool_call>";
        const auto result = parse_assistant_output(raw);
        AILA_CHAT_EXPECT_EQ(result.raw_text, raw);
        AILA_CHAT_EXPECT_EQ(result.content, raw);
        AILA_CHAT_EXPECT_TRUE(result.tool_calls.empty());
        AILA_CHAT_EXPECT_TRUE(!result.warnings.empty());
    }

    {
        const auto result = parse_assistant_output(
            "\n<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>\nhello\n</parameter>\n"
            "</function>\n");
        AILA_CHAT_EXPECT_EQ(result.content, std::string(""));
        AILA_CHAT_EXPECT_EQ(result.tool_calls.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.name, std::string("search"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.arguments_json, std::string(R"({"query":"hello"})"));
        AILA_CHAT_EXPECT_TRUE(!result.warnings.empty());
    }

    {
        const auto result = parse_assistant_output(
            "\n<tool_call>\n"
            "<search query=\"hello\">\n"
            "</search>\n");
        AILA_CHAT_EXPECT_EQ(result.content, std::string(""));
        AILA_CHAT_EXPECT_EQ(result.tool_calls.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.name, std::string("search"));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.arguments_json, std::string(R"({"query":"hello"})"));
        AILA_CHAT_EXPECT_TRUE(!result.warnings.empty());
    }

    {
        const auto result = parse_assistant_output(
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>{not json}</parameter>\n"
            "</function>\n"
            "</tool_call>");
        AILA_CHAT_EXPECT_EQ(result.tool_calls.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(result.tool_calls[0].function.arguments_json, std::string(R"({"query":"{not json}"})"));
    }
}

} // namespace aila::chat::test

void run_assistant_output_parser_tests() {
    aila::chat::test::run_assistant_output_parser_tests();
}
