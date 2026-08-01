#include "Qwen3TtsTextLayout.hpp"

#include <algorithm>

namespace aila {

Qwen3TtsTextLayout parse_qwen3_tts_text_layout(
    const std::vector<int>& formatted_tokens,
    int im_end_id,
    int im_start_id) {
    Qwen3TtsTextLayout layout;
    const int length = static_cast<int>(formatted_tokens.size());
    if (length <= 0) {
        return layout;
    }

    layout.text_body_start = std::min(3, length);
    layout.text_body_end = length;

    for (int i = length - 1; i >= layout.text_body_start; --i) {
        if (formatted_tokens[static_cast<size_t>(i)] == im_end_id) {
            layout.text_body_end = i;
            layout.has_im_end = true;
            break;
        }
    }
    if (!layout.has_im_end) {
        for (int i = layout.text_body_start + 1; i < length; ++i) {
            if (formatted_tokens[static_cast<size_t>(i)] == im_start_id) {
                layout.text_body_end = i;
                break;
            }
        }
    }

    layout.body_tokens.assign(
        formatted_tokens.begin() + layout.text_body_start,
        formatted_tokens.begin() + layout.text_body_end);
    return layout;
}

void Qwen3TtsStreamTextState::Begin(const std::vector<int>& body_tokens) {
    body_tokens_ = body_tokens;
    finishing_ = false;
}

bool Qwen3TtsStreamTextState::Append(
    const std::vector<int>& new_body_tokens) {
    if (finishing_ || new_body_tokens.empty()) {
        return false;
    }
    body_tokens_.insert(body_tokens_.end(),
                        new_body_tokens.begin(),
                        new_body_tokens.end());
    return true;
}

void Qwen3TtsStreamTextState::Finish() {
    finishing_ = true;
}

void Qwen3TtsStreamTextState::Reset() {
    body_tokens_.clear();
    finishing_ = false;
}

}  // namespace aila
