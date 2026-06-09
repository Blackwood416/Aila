#pragma once

#include "chat/ChatTypes.hpp"

#include <cstddef>

namespace aila::chat {

class ChatSessionState {
public:
    void clear();
    void set_system_prompt(const std::string& prompt);
    void add_user_text(const std::string& text);
    void add_assistant_result(const AssistantChatResult& result, bool preserve_reasoning);
    bool remove_last_user_message();
    bool drop_oldest_turn();
    ChatRequest to_request() const;
    ChatRequest to_request_without_reasoning() const;
    size_t message_count_without_system() const;
    size_t turn_count() const;

private:
    std::string system_prompt_;
    std::vector<ChatMessage> messages_;
};

} // namespace aila::chat
