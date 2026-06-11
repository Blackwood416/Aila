#pragma once

#include "chat/ChatTypes.hpp"

#include <string>

namespace aila::chat {

bool parse_tool_call_content(const std::string& content, ChatToolCall& tool_call);
AssistantChatResult parse_assistant_output(const std::string& raw_text);

} // namespace aila::chat
