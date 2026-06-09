#pragma once

#include <string>
#include <vector>

namespace aila::chat {

enum class StructuredStreamEventType {
    ReasoningDelta,
    ContentDelta,
    ToolCallDelta,
    Warning
};

struct StructuredStreamEvent {
    StructuredStreamEventType type = StructuredStreamEventType::ContentDelta;
    std::string text;
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
              std::vector<StructuredStreamEvent>& out_events) const;

    State state_ = State::Content;
    std::string buffer_;
};

} // namespace aila::chat
