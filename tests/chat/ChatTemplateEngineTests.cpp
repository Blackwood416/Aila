#include "ChatTest.hpp"
#include "chat/BuiltinTemplates.hpp"
#include "chat/ChatTemplateEngine.hpp"

namespace aila::chat::test {

void run_chat_template_engine_tests() {
    {
        const std::string& tmpl = qwen35_fixed_chat_template();
        AILA_CHAT_EXPECT_TRUE(tmpl.find("qwen3.6-froggeric-v20") != std::string::npos);
        AILA_CHAT_EXPECT_TRUE(tmpl.find("enable_thinking") != std::string::npos);
        AILA_CHAT_EXPECT_TRUE(tmpl.find("<tool_call>") != std::string::npos);
    }

    ChatRequest req;
    ChatMessage sys;
    sys.role = ChatRole::System;
    sys.content.push_back(ChatContentPart::text_part("Be exact."));
    req.messages.push_back(sys);
    ChatMessage user;
    user.role = ChatRole::User;
    user.content.push_back(ChatContentPart::text_part("Hello <|think_off|>"));
    req.messages.push_back(user);

    ChatTool tool;
    tool.name = "search";
    tool.description = "Search";
    tool.parameters_json = R"({"type":"object","properties":{"query":{"type":"string"}}})";
    req.tools.push_back(tool);
    req.template_options.enable_thinking = true;

    ChatTemplateEngine engine;
    ChatTemplateRenderInput input;
    input.template_source = qwen35_fixed_chat_template();
    input.request = &req;
    input.add_generation_prompt = true;
    input.bos_token = "";
    input.eos_token = "<|im_end|>";

    std::string error;
    ChatTemplateRenderOutput output;
    AILA_CHAT_EXPECT_TRUE(engine.render(input, output, &error));
    AILA_CHAT_EXPECT_TRUE(output.text.find("<|im_start|>system") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(output.text.find("Be exact.") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(output.text.find("Hello") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(output.text.find("<|think_off|>") == std::string::npos);
    AILA_CHAT_EXPECT_TRUE(output.text.find("<tools>") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(output.text.find("\"parameters\": {\"type\": \"object\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(output.text.find("\"parameters\":\"{") == std::string::npos);
    AILA_CHAT_EXPECT_TRUE(output.text.find("<think>") != std::string::npos);
}

} // namespace aila::chat::test

void run_chat_template_engine_tests() {
    aila::chat::test::run_chat_template_engine_tests();
}
