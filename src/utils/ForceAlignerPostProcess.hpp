#pragma once

#include <string>
#include <vector>
#include <utility>

namespace aila {

struct AlignedWord {
    std::string text;
    int start_ms;
    int end_ms;
};

class ForceAlignerPostProcess {
public:
    // Encode text for forced alignment.
    // Returns (word_list, prompt_text) where prompt_text has <timestamp> markers.
    // Naive v1: CJK char-by-char, all other languages space-delimited.
    static std::pair<std::vector<std::string>, std::string>
    encode_input(const std::string& text, const std::string& language);

    // Extract timestamps from classify logits.
    // logits: [seq_len, classify_num] in row-major float32.
    // input_ids: [seq_len] token IDs of the original input.
    // Only positions where input_ids[i] == timestamp_token_id produce timestamps.
    static std::vector<int> extract_timestamps(
        const float* logits, int64_t seq_len, int classify_num,
        const int* input_ids, int input_len,
        int timestamp_token_id, int timestamp_segment_time);

    // Fix monotonicity anomalies in timestamps via LIS-based correction.
    // Ported from Python Qwen3ForceAlignProcessor.fix_timestamp().
    static std::vector<int> fix_timestamp(const std::vector<int>& data);

    // Pair consecutive timestamps with words.
    // timestamps: [start_0, end_0, start_1, end_1, ...]
    // words: [word_0, word_1, ...]
    static std::vector<AlignedWord> build_output(
        const std::vector<std::string>& words,
        const std::vector<int>& timestamps);
};

} // namespace aila
