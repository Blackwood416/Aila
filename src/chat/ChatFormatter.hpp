#pragma once

#include "chat/ChatTemplateEngine.hpp"

namespace aila::chat {

struct ChatFormatInput {
    const ChatRequest* request = nullptr;
    ModelFamily family = ModelFamily::Unknown;
    bool is_qwen35_0p8b = false;
    std::string tokenizer_chat_template;
    std::string bos_token;
    std::string eos_token;
};

struct ChatFormatTextResult {
    std::string text;
    std::string template_name;
};

class ChatFormatter {
public:
    bool render_text(const ChatFormatInput& input,
                     bool add_generation_prompt,
                     ChatFormatTextResult& output,
                     std::string* error_message = nullptr) const;
};

} // namespace aila::chat
