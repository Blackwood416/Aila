#pragma once

namespace aila::chat {

class ThinkingBudgetController {
public:
    void start(bool prompt_opens_think,
               int budget_tokens,
               int think_token_id,
               int end_think_token_id) {
        budget_tokens_ = budget_tokens;
        think_token_id_ = think_token_id;
        end_think_token_id_ = end_think_token_id;
        inside_think_ = prompt_opens_think && budget_tokens > 0;
        used_tokens_ = 0;
        forced_close_ = false;
    }

    bool enabled() const { return budget_tokens_ >= 0; }
    bool positive_budget() const { return budget_tokens_ > 0; }
    bool should_disable_chunked_decode() const { return positive_budget(); }
    bool inside_think() const { return inside_think_; }
    int used_tokens() const { return used_tokens_; }
    bool forced_close() const { return forced_close_; }

    void observe_generated_token(int token_id) {
        if (!positive_budget() || forced_close_) {
            return;
        }
        if (token_id == think_token_id_) {
            inside_think_ = true;
            return;
        }
        if (token_id == end_think_token_id_) {
            inside_think_ = false;
            return;
        }
        if (inside_think_) {
            ++used_tokens_;
        }
    }

    bool needs_forced_close() const {
        return positive_budget() &&
               inside_think_ &&
               !forced_close_ &&
               used_tokens_ >= budget_tokens_;
    }

    void mark_forced_close() {
        forced_close_ = true;
        inside_think_ = false;
    }

private:
    int budget_tokens_ = -1;
    int think_token_id_ = -1;
    int end_think_token_id_ = -1;
    bool inside_think_ = false;
    int used_tokens_ = 0;
    bool forced_close_ = false;
};

} // namespace aila::chat
