#pragma once

#include "engine/Types.hpp"

#include <vector>

inline std::vector<int> qwen3_tts_minimal_warmup_text_tokens() {
    // Matches format_tts_text_for_backend("<dummy-token>"):
    // <|im_start|>assistant\n token <|im_end|>\n<|im_start|>assistant\n
    return {
        151644, 77091, 198,
        0,
        151645, 198, 151644, 77091, 198,
    };
}

inline std::vector<float> qwen3_tts_warmup_speaker_embedding(
    Qwen3TTSModelType model_type,
    int hidden_size) {
    if (model_type != Qwen3TTSModelType::Base || hidden_size <= 0) {
        return {};
    }
    return std::vector<float>(static_cast<size_t>(hidden_size), 0.0f);
}
