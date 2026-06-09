#pragma once

#include "chat/ChatTypes.hpp"

#include <string>
#include <vector>

namespace aila::chat {

struct ToolPolicyValidation {
    bool hard_error = false;
    std::vector<std::string> warnings;
};

ToolPolicyValidation validate_tool_policy(
    const ChatRequest& request,
    const AssistantChatResult& parsed_output);

} // namespace aila::chat
