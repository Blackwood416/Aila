#pragma once

#include "chat/ChatTypes.hpp"

#include <string>

namespace aila::chat {

AssistantChatResult parse_assistant_output(const std::string& raw_text);

} // namespace aila::chat
