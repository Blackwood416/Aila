#include "ChatTest.hpp"
#include "chat/StructuredStreamParser.hpp"

#include <string>
#include <vector>

namespace aila::chat::test {

namespace {

std::string collect_text(const std::vector<StructuredStreamEvent>& events,
                         StructuredStreamEventType type) {
    std::string text;
    for (const auto& event : events) {
        if (event.type == type) {
            text += event.text;
        }
    }
    return text;
}

} // namespace

void run_structured_stream_parser_tests() {
    {
        StructuredStreamParser parser;
        std::vector<StructuredStreamEvent> events;
        parser.push("<think>\nplan", events);
        AILA_CHAT_EXPECT_TRUE(events.empty());

        parser.push("\n</think>\nanswer", events);
        AILA_CHAT_EXPECT_EQ(
            collect_text(events, StructuredStreamEventType::ReasoningDelta),
            std::string("plan"));
        AILA_CHAT_EXPECT_EQ(
            collect_text(events, StructuredStreamEventType::ContentDelta),
            std::string("answer"));
    }

    {
        StructuredStreamParser parser;
        std::vector<StructuredStreamEvent> events;
        parser.push("hello <thi", events);
        AILA_CHAT_EXPECT_EQ(
            collect_text(events, StructuredStreamEventType::ContentDelta),
            std::string("hello "));

        parser.push("nk>\nsecret\n</think>\nworld", events);
        AILA_CHAT_EXPECT_EQ(
            collect_text(events, StructuredStreamEventType::ReasoningDelta),
            std::string("secret"));
        AILA_CHAT_EXPECT_EQ(
            collect_text(events, StructuredStreamEventType::ContentDelta),
            std::string("hello world"));
    }

    {
        StructuredStreamParser parser;
        std::vector<StructuredStreamEvent> events;
        parser.push("<tool_call>\n<function=search>\n", events);
        AILA_CHAT_EXPECT_TRUE(events.empty());

        parser.push("<parameter=query>hello</parameter>\n</function>\n</tool_call>", events);
        AILA_CHAT_EXPECT_EQ(
            collect_text(events, StructuredStreamEventType::ToolCallDelta),
            std::string("<tool_call>\n<function=search>\n<parameter=query>hello</parameter>\n</function>\n</tool_call>"));
    }

    {
        StructuredStreamParser parser;
        std::vector<StructuredStreamEvent> events;
        parser.push("answer", events);
        parser.finish(events);
        AILA_CHAT_EXPECT_EQ(
            collect_text(events, StructuredStreamEventType::ContentDelta),
            std::string("answer"));
    }
}

} // namespace aila::chat::test

void run_structured_stream_parser_tests() {
    aila::chat::test::run_structured_stream_parser_tests();
}
