#pragma once

#include "chat/ChatTypes.hpp"

#include <string>
#include <vector>

namespace aila::chat {

enum class StructuredStreamEventType {
    ReasoningDelta,
    ContentDelta,
    ToolCallDelta,
    Warning,
    Final
};

struct StructuredStreamEvent {
    StructuredStreamEventType type = StructuredStreamEventType::ContentDelta;
    std::string text;
    std::string tool_call_id;
    std::string tool_name;
    std::string arguments_delta;
    std::string finish_reason;
    std::vector<std::string> warnings;
    std::vector<ChatToolCall> tool_calls;
};

class StructuredStreamParser {
public:
    void push(const std::string& text, std::vector<StructuredStreamEvent>& out_events);
    void finish(std::vector<StructuredStreamEvent>& out_events);

private:
    enum class State {
        Content,
        Reasoning,
        ToolCall
    };

    void process(std::vector<StructuredStreamEvent>& out_events, bool final);
    void emit(StructuredStreamEventType type,
              const std::string& text,
              std::vector<StructuredStreamEvent>& out_events);

    State state_ = State::Content;
    std::string buffer_;
    size_t next_tool_call_index_ = 0;
};

} // namespace aila::chat
