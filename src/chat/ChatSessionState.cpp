#include "chat/ChatSessionState.hpp"

#include <cstddef>
#include <utility>

namespace aila::chat {

void ChatSessionState::clear() {
    messages_.clear();
}

void ChatSessionState::set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
}

void ChatSessionState::add_user_text(const std::string& text) {
    ChatMessage msg;
    msg.role = ChatRole::User;
    msg.content.push_back(ChatContentPart::text_part(text));
    messages_.push_back(std::move(msg));
}

void ChatSessionState::add_assistant_result(const AssistantChatResult& result, bool preserve_reasoning) {
    ChatMessage msg;
    msg.role = ChatRole::Assistant;
    msg.content.push_back(ChatContentPart::text_part(result.content));
    if (preserve_reasoning) {
        msg.reasoning_content = result.reasoning_content;
    }
    msg.tool_calls = result.tool_calls;
    messages_.push_back(std::move(msg));
}

bool ChatSessionState::remove_last_user_message() {
    if (messages_.empty() || messages_.back().role != ChatRole::User) {
        return false;
    }
    messages_.pop_back();
    return true;
}

bool ChatSessionState::drop_oldest_turn() {
    for (size_t i = 0; i < messages_.size(); ++i) {
        if (messages_[i].role != ChatRole::User) {
            continue;
        }
        if (i + 1 >= messages_.size()) {
            return false;
        }
        size_t end = i + 1;
        while (end < messages_.size() && messages_[end].role != ChatRole::User) {
            ++end;
        }
        messages_.erase(messages_.begin() + static_cast<std::ptrdiff_t>(i),
                        messages_.begin() + static_cast<std::ptrdiff_t>(end));
        return true;
    }
    return false;
}

ChatRequest ChatSessionState::to_request() const {
    ChatRequest req;
    if (!system_prompt_.empty()) {
        ChatMessage sys;
        sys.role = ChatRole::System;
        sys.content.push_back(ChatContentPart::text_part(system_prompt_));
        req.messages.push_back(std::move(sys));
    }
    req.messages.insert(req.messages.end(), messages_.begin(), messages_.end());
    return req;
}

ChatRequest ChatSessionState::to_request_without_reasoning() const {
    ChatRequest req = to_request();
    for (auto& msg : req.messages) {
        msg.reasoning_content.clear();
    }
    return req;
}

size_t ChatSessionState::message_count_without_system() const {
    return messages_.size();
}

size_t ChatSessionState::turn_count() const {
    size_t count = 0;
    for (const auto& msg : messages_) {
        if (msg.role == ChatRole::User) {
            ++count;
        }
    }
    return count;
}

} // namespace aila::chat
