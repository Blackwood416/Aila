#pragma once

#include <cstddef>
#include <vector>

namespace aila {

struct Qwen3TtsTextLayout {
    int text_body_start = 3;
    int text_body_end = 0;
    bool has_im_end = false;
    std::vector<int> body_tokens;
};

Qwen3TtsTextLayout parse_qwen3_tts_text_layout(
    const std::vector<int>& formatted_tokens,
    int im_end_id = 151645,
    int im_start_id = 151644);

class Qwen3TtsStreamTextState {
public:
    void Begin(const std::vector<int>& body_tokens);
    bool Append(const std::vector<int>& new_body_tokens);
    void Finish();
    void Reset();

    const std::vector<int>& body_tokens() const { return body_tokens_; }
    bool finishing() const { return finishing_; }
    bool empty() const { return body_tokens_.empty(); }
    int first_text_token() const {
        return body_tokens_.empty() ? -1 : body_tokens_.front();
    }
    size_t trailing_len() const {
        if (body_tokens_.empty()) {
            return 0;
        }
        size_t len = body_tokens_.size() - 1;
        if (finishing_) {
            ++len;
        }
        return len;
    }

private:
    std::vector<int> body_tokens_;
    bool finishing_ = false;
};

}  // namespace aila
