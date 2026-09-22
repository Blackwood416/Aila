#include "engine/Engine.hpp"

void InferenceEngine::reset_context() {
        history_.clear();
        mm_history_.clear();
        chat_session_.clear();
        cached_ids_.clear();
        benchmark_seed_ready_ = false;
        if (backend_) backend_->reset();
        AILA_LOG_INFO("[Context] Conversation reset");
    }

int InferenceEngine::max_context_length() const { return backend_ ? backend_->max_seq_len() : 0; }

size_t InferenceEngine::active_history_message_count() const {
        if (chat_session_.message_count_without_system() > 0) {
            return chat_session_.message_count_without_system();
        }
        if (!mm_history_.empty()) {
            size_t count = 0;
            for (const auto& msg : mm_history_) {
                if (msg.role != "system") ++count;
            }
            return count;
        }
        return history_.size();
    }

size_t InferenceEngine::active_history_turn_count() const {
        size_t count = 0;
        if (chat_session_.turn_count() > 0) {
            return chat_session_.turn_count();
        }
        if (!mm_history_.empty()) {
            for (const auto& msg : mm_history_) {
                if (msg.role == "user") ++count;
            }
            return count;
        }
        for (const auto& msg : history_.messages()) {
            if (msg.role == "user") ++count;
        }
        return count;
    }

int InferenceEngine::conversation_context_length() const {
        auto render_chat_request_len = [&](const aila::chat::ChatRequest& request) -> int {
            aila::chat::ChatFormatTextResult rendered;
            std::string error;
            if (!chat_formatter_.render_text(make_chat_format_input(request), false, rendered, &error)) {
                return -1;
            }
            return static_cast<int>(tokenizer_.encode(rendered.text).size());
        };

        if (chat_session_.message_count_without_system() > 0) {
            aila::chat::ChatRequest request = chat_session_.to_request_without_reasoning();
            int len = render_chat_request_len(request);
            if (len >= 0) {
                return len;
            }
            return context_length();
        }
        if (!mm_history_.empty()) {
            aila::chat::ChatRequest request =
                make_chat_request_from_legacy(mm_history_, GenerationConfig{});
            int len = render_chat_request_len(request);
            if (len >= 0) {
                return len;
            }
            return context_length();
        }
        if (!history_.empty()) {
            return static_cast<int>(
                tokenizer_.apply_chat_template(system_prompt_, history_).size());
        }
        return 0;
    }
