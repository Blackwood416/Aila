#pragma once

#include "chat/ChatTypes.hpp"

#include <string>

namespace aila::chat {

struct ChatTemplateRenderInput {
    std::string template_source;
    const ChatRequest* request = nullptr;
    bool add_generation_prompt = true;
    std::string bos_token;
    std::string eos_token;
};

struct ChatTemplateRenderOutput {
    std::string text;
};

class ChatTemplateEngine {
public:
    bool render(const ChatTemplateRenderInput& input,
                ChatTemplateRenderOutput& output,
                std::string* error_message = nullptr) const;
};

} // namespace aila::chat
