#pragma once

#include "chat/StructuredStreamParser.hpp"

#include <string>

namespace aila::chat {

const char* stream_event_type_name(StructuredStreamEventType type);
std::string stream_event_to_json(const StructuredStreamEvent& event);

} // namespace aila::chat
