#pragma once

#include "chat/StructuredStreamParser.hpp"

#include <string>

namespace aila::chat {

const char* stream_event_type_name(StructuredStreamEventType type);
std::string stream_event_to_json(const StructuredStreamEvent& event);
StructuredStreamEvent final_stream_event_from_result(const AssistantChatResult& result);

} // namespace aila::chat
