#include "chat/ChatStreamJson.hpp"

#include "chat/ChatJson.hpp"

#include <sstream>

namespace aila::chat {

const char* stream_event_type_name(StructuredStreamEventType type) {
    switch (type) {
    case StructuredStreamEventType::ReasoningDelta:
        return "reasoning_delta";
    case StructuredStreamEventType::ContentDelta:
        return "content_delta";
    case StructuredStreamEventType::ToolCallDelta:
        return "tool_call_delta";
    case StructuredStreamEventType::Warning:
        return "warning";
    case StructuredStreamEventType::Final:
        return "final";
    }
    return "content_delta";
}

std::string stream_event_to_json(const StructuredStreamEvent& event) {
    std::ostringstream out;
    out << "{\"type\":\"" << stream_event_type_name(event.type) << "\"";

    if (!event.text.empty()) {
        out << ",\"text\":" << json_escape_string(event.text);
    }
    if (!event.tool_call_id.empty()) {
        out << ",\"tool_call_id\":" << json_escape_string(event.tool_call_id);
    }
    if (!event.tool_name.empty()) {
        out << ",\"tool_name\":" << json_escape_string(event.tool_name);
    }
    if (!event.arguments_delta.empty()) {
        out << ",\"arguments_delta\":" << json_escape_string(event.arguments_delta);
    }
    if (!event.finish_reason.empty()) {
        out << ",\"finish_reason\":" << json_escape_string(event.finish_reason);
    }
    if (!event.warnings.empty()) {
        out << ",\"warnings\":[";
        for (size_t i = 0; i < event.warnings.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << json_escape_string(event.warnings[i]);
        }
        out << "]";
    }

    out << "}";
    return out.str();
}

} // namespace aila::chat
