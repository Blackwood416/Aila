#include "ChatTest.hpp"
#include "chat/AssistantOutputParser.hpp"
#include "chat/ToolPolicy.hpp"

#include <string>

namespace aila::chat::test {

void run_tool_policy_tests() {
    {
        ChatRequest request;
        request.tool_choice = ToolChoice::None;

        const auto parsed = parse_assistant_output(
            "<tool_call><function=search></function></tool_call>");
        const auto validation = validate_tool_policy(request, parsed);

        AILA_CHAT_EXPECT_TRUE(!validation.hard_error);
        AILA_CHAT_EXPECT_EQ(validation.warnings.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(validation.warnings[0], std::string("tool call returned despite tool_choice none"));
    }

    {
        ChatRequest request;
        request.tool_choice = ToolChoice::Required;

        const auto parsed = parse_assistant_output("plain answer");
        const auto validation = validate_tool_policy(request, parsed);

        AILA_CHAT_EXPECT_TRUE(!validation.hard_error);
        AILA_CHAT_EXPECT_EQ(validation.warnings.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(validation.warnings[0], std::string("tool_choice required but no tool call was returned"));
    }

    {
        ChatRequest request;
        request.tool_choice = ToolChoice::Function;
        request.tool_choice_function_name = "search";

        const auto parsed = parse_assistant_output("plain answer");
        const auto validation = validate_tool_policy(request, parsed);

        AILA_CHAT_EXPECT_TRUE(!validation.hard_error);
        AILA_CHAT_EXPECT_EQ(validation.warnings.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(
            validation.warnings[0],
            std::string("tool_choice function search was requested but no tool call was returned"));
    }

    {
        ChatRequest request;
        request.tool_choice = ToolChoice::Function;
        request.tool_choice_function_name = "search";

        const auto parsed = parse_assistant_output(
            "<tool_call><function=calculator></function></tool_call>");
        const auto validation = validate_tool_policy(request, parsed);

        AILA_CHAT_EXPECT_TRUE(!validation.hard_error);
        AILA_CHAT_EXPECT_EQ(validation.warnings.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(
            validation.warnings[0],
            std::string("tool_choice function expected search but tool call returned calculator"));
    }

    {
        ChatRequest request;
        request.tool_choice = ToolChoice::Function;
        request.tool_choice_function_name = "search";

        const auto parsed = parse_assistant_output(
            "<tool_call><function=search></function></tool_call>"
            "<tool_call><function=calculator></function></tool_call>");
        const auto validation = validate_tool_policy(request, parsed);

        AILA_CHAT_EXPECT_TRUE(!validation.hard_error);
        AILA_CHAT_EXPECT_EQ(validation.warnings.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(
            validation.warnings[0],
            std::string("tool_choice function expected search but tool call returned calculator"));
    }

    {
        ChatRequest request;
        request.tool_choice = ToolChoice::Auto;

        const auto parsed = parse_assistant_output("plain answer");
        const auto validation = validate_tool_policy(request, parsed);

        AILA_CHAT_EXPECT_TRUE(!validation.hard_error);
        AILA_CHAT_EXPECT_TRUE(validation.warnings.empty());
    }
}

} // namespace aila::chat::test

void run_tool_policy_tests() {
    aila::chat::test::run_tool_policy_tests();
}
