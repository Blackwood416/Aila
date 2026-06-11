#include "ChatTest.hpp"
#include "chat/ChatJson.hpp"

#include <vector>

namespace aila::chat::test {

void run_chat_json_tests() {
    AILA_CHAT_EXPECT_EQ(decode_finish_reason(false, false, false), std::string("stop"));
    AILA_CHAT_EXPECT_EQ(decode_finish_reason(true, true, false), std::string("loop_guard"));
    AILA_CHAT_EXPECT_EQ(decode_finish_reason(false, true, false), std::string("length"));

    const std::string request_json = R"({
      "messages": [
        {"role": "developer", "content": "Be exact."},
        {"role": "user", "content": [{"type": "text", "text": "Use search."}]},
        {
          "role": "assistant",
          "content": "calling",
          "reasoning_content": "need data",
          "tool_calls": [{
            "id": "call_a",
            "type": "function",
            "function": {"name": "search", "arguments": "{\"query\":\"cats\"}"}
          }]
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "result"}
      ],
      "tools": [{
        "type": "function",
        "function": {
          "name": "search",
          "description": "Search",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
      }],
      "tool_choice": {"type": "function", "function": {"name": "search"}},
      "chat_template_kwargs": {
        "enable_thinking": false,
        "preserve_thinking": true,
        "max_tool_response_chars": 128
      },
      "temperature": 0.0,
      "max_tokens": 12
    })";

    ChatRequest req;
    std::string error;
    AILA_CHAT_EXPECT_TRUE(parse_chat_request_json(request_json, GenerationConfig{}, req, &error));
    AILA_CHAT_EXPECT_EQ(req.messages.size(), static_cast<size_t>(4));
    AILA_CHAT_EXPECT_EQ(role_to_string(req.messages[0].role), std::string("developer"));
    AILA_CHAT_EXPECT_EQ(req.messages[1].content[0].text, std::string("Use search."));
    AILA_CHAT_EXPECT_EQ(req.messages[2].reasoning_content, std::string("need data"));
    AILA_CHAT_EXPECT_EQ(req.messages[2].tool_calls[0].function.name, std::string("search"));
    AILA_CHAT_EXPECT_EQ(req.messages[2].tool_calls[0].function.arguments_json, std::string(R"({"query":"cats"})"));
    AILA_CHAT_EXPECT_EQ(req.messages[3].tool_call_id, std::string("call_a"));
    AILA_CHAT_EXPECT_EQ(req.tools[0].name, std::string("search"));
    AILA_CHAT_EXPECT_EQ(req.tools[0].parameters_json, std::string(R"({"type":"object","properties":{"query":{"type":"string"}}})"));
    AILA_CHAT_EXPECT_TRUE(req.tool_choice == ToolChoice::Function);
    AILA_CHAT_EXPECT_EQ(static_cast<int>(req.tool_policy), static_cast<int>(ToolPolicyMode::Warn));
    AILA_CHAT_EXPECT_EQ(req.tool_choice_function_name, std::string("search"));
    AILA_CHAT_EXPECT_TRUE(req.template_options.enable_thinking.has_value());
    AILA_CHAT_EXPECT_TRUE(!req.template_options.enable_thinking.value());
    AILA_CHAT_EXPECT_TRUE(req.template_options.preserve_thinking);
    AILA_CHAT_EXPECT_EQ(req.template_options.max_tool_response_chars, 128);
    AILA_CHAT_EXPECT_EQ(req.generation_config.max_new_tokens, 12);
    AILA_CHAT_EXPECT_TRUE(!req.generation_config.do_sample);

    AILA_CHAT_EXPECT_TRUE(parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"reasoning_budget":8})",
        GenerationConfig{},
        req,
        &error));
    AILA_CHAT_EXPECT_EQ(req.generation_config.thinking_budget_tokens, 8);

    AILA_CHAT_EXPECT_TRUE(parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"thinking_budget":0})",
        GenerationConfig{},
        req,
        &error));
    AILA_CHAT_EXPECT_EQ(req.generation_config.thinking_budget_tokens, 0);

    AILA_CHAT_EXPECT_TRUE(parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"thinking_budget_tokens":16})",
        GenerationConfig{},
        req,
        &error));
    AILA_CHAT_EXPECT_EQ(req.generation_config.thinking_budget_tokens, 16);

    AILA_CHAT_EXPECT_TRUE(parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"tool_policy":"warn"})",
        GenerationConfig{},
        req,
        &error));
    AILA_CHAT_EXPECT_EQ(static_cast<int>(req.tool_policy), static_cast<int>(ToolPolicyMode::Warn));

    AILA_CHAT_EXPECT_TRUE(parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"tool_policy":"strict"})",
        GenerationConfig{},
        req,
        &error));
    AILA_CHAT_EXPECT_EQ(static_cast<int>(req.tool_policy), static_cast<int>(ToolPolicyMode::Strict));

    AILA_CHAT_EXPECT_TRUE(!parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"tool_policy":"panic"})",
        GenerationConfig{},
        req,
        &error));
    AILA_CHAT_EXPECT_TRUE(error.find("unsupported tool_policy") != std::string::npos);

    ChatRequest array_req;
    GenerationConfig defaults;
    defaults.max_new_tokens = 7;
    AILA_CHAT_EXPECT_TRUE(parse_chat_request_json(
        R"([{"role":"assistant","tool_calls":[{"function":{"name":"calc","arguments":{"x":2}}}]}])",
        defaults,
        array_req,
        &error));
    AILA_CHAT_EXPECT_EQ(array_req.generation_config.max_new_tokens, 7);
    AILA_CHAT_EXPECT_EQ(array_req.messages.size(), static_cast<size_t>(1));
    AILA_CHAT_EXPECT_EQ(array_req.messages[0].tool_calls[0].type, std::string("function"));
    AILA_CHAT_EXPECT_EQ(array_req.messages[0].tool_calls[0].function.arguments_json, std::string(R"({"x":2})"));

    AssistantChatResult result;
    result.content = "answer";
    result.reasoning_content = "thought";
    result.raw_text = "<think>\nthought\n</think>\nanswer";
    result.finish_reason = "length";
    result.metadata.template_name = "qwen35-fixed-v20";
    result.metadata.model_family = "qwen3_5_hybrid";
    result.metadata.reasoning_budget_tokens = 8;
    result.metadata.reasoning_budget_forced_close = true;
    result.metadata.reasoning_budget_truncated = true;
    result.metadata.tool_policy = "warn";
    result.metadata.tool_choice = "auto";
    ChatToolCall call;
    call.id = "call_0";
    call.function.name = "search";
    call.function.arguments_json = "{\"query\":\"cats\"}";
    result.tool_calls.push_back(call);
    result.warnings.push_back("example warning");

    std::string out = assistant_result_to_json(result);
    AILA_CHAT_EXPECT_TRUE(out.find("\"role\":\"assistant\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"reasoning_content\":\"thought\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"tool_calls\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"finish_reason\":\"length\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"arguments\":\"{\\\"query\\\":\\\"cats\\\"}\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"warnings\":[\"example warning\"]") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"metadata\":{") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"template_name\":\"qwen35-fixed-v20\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"model_family\":\"qwen3_5_hybrid\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"reasoning_budget_tokens\":8") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"reasoning_budget_forced_close\":true") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"reasoning_budget_truncated\":true") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"tool_policy\":\"warn\"") != std::string::npos);
    AILA_CHAT_EXPECT_TRUE(out.find("\"tool_choice\":\"auto\"") != std::string::npos);

    {
        ChatToolCall helper_call;
        helper_call.id = "call_0";
        helper_call.type = "function";
        helper_call.function.name = "search";
        helper_call.function.arguments_json = R"({"query":"cats"})";

        const std::string json = tool_calls_to_json(std::vector<ChatToolCall>{helper_call});

        AILA_CHAT_EXPECT_EQ(
            json,
            std::string(
                "[{\"id\":\"call_0\",\"type\":\"function\","
                "\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"query\\\":\\\"cats\\\"}\"}}]"));
    }
}

} // namespace aila::chat::test

void run_chat_json_tests() {
    aila::chat::test::run_chat_json_tests();
}
