#include "ChatTest.hpp"
#include "chat/ChatFormatter.hpp"
#include "chat/ChatSessionState.hpp"

namespace aila::chat::test {

void run_chat_formatter_tests() {
    ChatRequest req;
    ChatMessage user;
    user.role = ChatRole::User;
    user.content.push_back(ChatContentPart::text_part("hello"));
    req.messages.push_back(user);

    ChatFormatter formatter;
    ChatFormatInput input;
    input.request = &req;
    input.family = ModelFamily::Qwen35Hybrid;
    input.is_qwen35_0p8b = true;
    input.tokenizer_chat_template = "bad official template should not win";
    input.bos_token = "";
    input.eos_token = "<|im_end|>";

    ChatFormatTextResult rendered;
    std::string error;
    AILA_CHAT_EXPECT_TRUE(formatter.render_text(input, true, rendered, &error));
    AILA_CHAT_EXPECT_TRUE(rendered.text.find("qwen3.6-froggeric-v20") == std::string::npos);
    AILA_CHAT_EXPECT_TRUE(rendered.text.find("<|im_start|>user") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(rendered.text.find("<think>\n</think>") != std::string::npos);
    AILA_CHAT_EXPECT_EQ(rendered.template_name, std::string("qwen35-fixed-v20"));

    ChatFormatTextResult stable;
    AILA_CHAT_EXPECT_TRUE(formatter.render_text(input, false, stable, &error));
    AILA_CHAT_EXPECT_TRUE(rendered.text.rfind(stable.text, 0) == 0);

    ChatSessionState session;
    session.set_system_prompt("Be brief.");
    session.add_user_text("What is 1+1?");
    AssistantChatResult assistant;
    assistant.content = "2";
    assistant.reasoning_content = "simple arithmetic";
    session.add_assistant_result(assistant, true);

    ChatRequest preserved = session.to_request();
    AILA_CHAT_EXPECT_EQ(preserved.messages.size(), static_cast<size_t>(3));
    AILA_CHAT_EXPECT_EQ(preserved.messages[0].content[0].text, std::string("Be brief."));
    AILA_CHAT_EXPECT_EQ(preserved.messages[2].reasoning_content, std::string("simple arithmetic"));
    AILA_CHAT_EXPECT_EQ(session.message_count_without_system(), static_cast<size_t>(2));
    AILA_CHAT_EXPECT_EQ(session.turn_count(), static_cast<size_t>(1));

    ChatRequest stripped = session.to_request_without_reasoning();
    AILA_CHAT_EXPECT_EQ(stripped.messages[2].reasoning_content, std::string(""));

    session.add_user_text("next");
    AILA_CHAT_EXPECT_EQ(session.message_count_without_system(), static_cast<size_t>(3));
    AILA_CHAT_EXPECT_TRUE(session.drop_oldest_turn());
    ChatRequest after_drop = session.to_request();
    AILA_CHAT_EXPECT_EQ(after_drop.messages.size(), static_cast<size_t>(2));
    AILA_CHAT_EXPECT_EQ(after_drop.messages[1].content[0].text, std::string("next"));
    AILA_CHAT_EXPECT_TRUE(session.remove_last_user_message());
    AILA_CHAT_EXPECT_EQ(session.message_count_without_system(), static_cast<size_t>(0));
    AILA_CHAT_EXPECT_TRUE(!session.drop_oldest_turn());

    ChatRequest tool_none_req;
    ChatMessage tool_user;
    tool_user.role = ChatRole::User;
    tool_user.content.push_back(ChatContentPart::text_part("hello"));
    tool_none_req.messages.push_back(tool_user);
    ChatTool search_tool;
    search_tool.name = "search";
    search_tool.description = "Search external data";
    search_tool.parameters_json = R"({"type":"object"})";
    tool_none_req.tools.push_back(search_tool);
    tool_none_req.tool_choice = ToolChoice::None;

    ChatFormatInput tool_none_input;
    tool_none_input.request = &tool_none_req;
    tool_none_input.family = ModelFamily::Qwen35Hybrid;
    tool_none_input.is_qwen35_0p8b = false;
    tool_none_input.tokenizer_chat_template = "";
    tool_none_input.bos_token = "";
    tool_none_input.eos_token = "<|im_end|>";

    ChatFormatTextResult tool_none_rendered;
    AILA_CHAT_EXPECT_TRUE(formatter.render_text(tool_none_input, true, tool_none_rendered, &error));
    AILA_CHAT_EXPECT_TRUE(tool_none_rendered.text.find("<tools>") == std::string::npos);
    AILA_CHAT_EXPECT_TRUE(tool_none_rendered.text.find("search") == std::string::npos);

    ChatRequest tool_required_req = tool_none_req;
    tool_required_req.tool_choice = ToolChoice::Required;

    ChatFormatInput tool_required_input = tool_none_input;
    tool_required_input.request = &tool_required_req;

    ChatFormatTextResult tool_required_rendered;
    AILA_CHAT_EXPECT_TRUE(formatter.render_text(tool_required_input, true, tool_required_rendered, &error));
    AILA_CHAT_EXPECT_TRUE(
        tool_required_rendered.text.find("You MUST call at least one available tool.") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(
        tool_required_rendered.text.find("If you have gathered all necessary data and do not need to call a tool") ==
        std::string::npos);
    const std::string forced_tool_suffix = "<|im_start|>assistant\n<think>\n</think>\n";
    AILA_CHAT_EXPECT_TRUE(
        tool_required_rendered.text.size() >= forced_tool_suffix.size() &&
        tool_required_rendered.text.compare(
            tool_required_rendered.text.size() - forced_tool_suffix.size(),
            forced_tool_suffix.size(),
            forced_tool_suffix) == 0);

    ChatRequest tool_function_req = tool_none_req;
    tool_function_req.tool_choice = ToolChoice::Function;
    tool_function_req.tool_choice_function_name = "search";

    ChatFormatInput tool_function_input = tool_none_input;
    tool_function_input.request = &tool_function_req;

    ChatFormatTextResult tool_function_rendered;
    AILA_CHAT_EXPECT_TRUE(formatter.render_text(tool_function_input, true, tool_function_rendered, &error));
    AILA_CHAT_EXPECT_TRUE(
        tool_function_rendered.text.find("You MUST call the search tool.") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(
        tool_function_rendered.text.size() >= forced_tool_suffix.size() &&
        tool_function_rendered.text.compare(
            tool_function_rendered.text.size() - forced_tool_suffix.size(),
            forced_tool_suffix.size(),
            forced_tool_suffix) == 0);

    ChatRequest budget_zero_req;
    ChatMessage budget_user;
    budget_user.role = ChatRole::User;
    budget_user.content.push_back(ChatContentPart::text_part("hello"));
    budget_zero_req.messages.push_back(budget_user);
    budget_zero_req.generation_config.thinking_budget_tokens = 0;

    ChatFormatInput budget_input;
    budget_input.request = &budget_zero_req;
    budget_input.family = ModelFamily::Qwen35Hybrid;
    budget_input.is_qwen35_0p8b = false;
    budget_input.tokenizer_chat_template = "";
    budget_input.bos_token = "";
    budget_input.eos_token = "<|im_end|>";

    ChatFormatTextResult budget_rendered;
    AILA_CHAT_EXPECT_TRUE(formatter.render_text(budget_input, true, budget_rendered, &error));
    AILA_CHAT_EXPECT_TRUE(budget_rendered.text.find("<think>\n</think>") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(budget_rendered.text.rfind("<think>\n") !=
                          budget_rendered.text.size() - std::string("<think>\n").size());
}

} // namespace aila::chat::test

void run_chat_formatter_tests() {
    aila::chat::test::run_chat_formatter_tests();
}
