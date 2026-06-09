#pragma once

#include "chat/ChatTypes.hpp"

#include <string>

namespace aila::chat {

bool parse_chat_request_json(const std::string& request_json,
                             const GenerationConfig& defaults,
                             ChatRequest& out,
                             std::string* error_message = nullptr);

std::string assistant_result_to_json(const AssistantChatResult& result);

std::string json_escape_string(const std::string& value);

} // namespace aila::chat
