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
        event.tool_call_id = "call_0";
        event.tool_name = "search";
        event.arguments_delta = "{\"query\":\"cats\"}";

        const std::string json = stream_event_to_json(event);

        AILA_CHAT_EXPECT_EQ(
            json,
            std::string(
                "{\"type\":\"tool_call_delta\",\"tool_call_id\":\"call_0\","
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
}

} // namespace aila::chat::test

void run_chat_stream_json_tests() {
    aila::chat::test::run_chat_stream_json_tests();
}
