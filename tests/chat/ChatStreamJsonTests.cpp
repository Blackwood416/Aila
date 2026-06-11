#include "ChatTest.hpp"
#include "chat/ChatStreamJson.hpp"

#include <string>

namespace aila::chat::test {

void run_chat_stream_json_tests() {
    {
        StructuredStreamEvent event;
        event.type = StructuredStreamEventType::ReasoningDelta;
        event.text = "plan";

        const std::string json = stream_event_to_json(event);

        AILA_CHAT_EXPECT_EQ(json, std::string("{\"type\":\"reasoning_delta\",\"text\":\"plan\"}"));
    }

    {
        StructuredStreamEvent event;
        event.type = StructuredStreamEventType::ToolCallDelta;
        event.text = "<tool_call>\n<function=search>\n<parameter=query>cats</parameter>\n</function>\n</tool_call>";
        event.tool_call_id = "call_0";
        event.tool_name = "search";
        event.arguments_delta = "{\"query\":\"cats\"}";

        const std::string json = stream_event_to_json(event);

        AILA_CHAT_EXPECT_EQ(
            json,
            std::string(
                "{\"type\":\"tool_call_delta\",\"text\":\"<tool_call>\\n<function=search>\\n"
                "<parameter=query>cats</parameter>\\n</function>\\n</tool_call>\","
                "\"tool_call_id\":\"call_0\","
                "\"tool_name\":\"search\",\"arguments_delta\":\"{\\\"query\\\":\\\"cats\\\"}\"}"));
    }

    {
        StructuredStreamEvent event;
        event.type = StructuredStreamEventType::Final;
        event.finish_reason = "stop";
        event.warnings.push_back("example warning");

        const std::string json = stream_event_to_json(event);

        AILA_CHAT_EXPECT_EQ(
            json,
            std::string(
                "{\"type\":\"final\",\"finish_reason\":\"stop\","
                "\"warnings\":[\"example warning\"]}"));
    }

    {
        AssistantChatResult result;
        result.finish_reason = "tool_calls";
        result.warnings.push_back("example warning");

        ChatToolCall call;
        call.id = "call_0";
        call.type = "function";
        call.function.name = "search";
        call.function.arguments_json = R"({"query":"cats"})";
        result.tool_calls.push_back(call);

        const StructuredStreamEvent event = final_stream_event_from_result(result);
        const std::string json = stream_event_to_json(event);

        AILA_CHAT_EXPECT_EQ(event.finish_reason, std::string("tool_calls"));
        AILA_CHAT_EXPECT_EQ(event.tool_calls.size(), static_cast<size_t>(1));
        AILA_CHAT_EXPECT_EQ(
            json,
            std::string(
                "{\"type\":\"final\",\"finish_reason\":\"tool_calls\","
                "\"warnings\":[\"example warning\"],"
                "\"tool_calls\":[{\"id\":\"call_0\",\"type\":\"function\","
                "\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"query\\\":\\\"cats\\\"}\"}}]}"));
    }
}

} // namespace aila::chat::test

void run_chat_stream_json_tests() {
    aila::chat::test::run_chat_stream_json_tests();
}
