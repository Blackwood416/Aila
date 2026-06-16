#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace aila::alia {

class ModelSlot;

class AliaAsrPipeline {
public:
    explicit AliaAsrPipeline(ModelSlot* slot);

    bool feed_audio(const float* samples, int sample_count);
    void append_stable_text(std::string text);
    bool process_pending();
    void reset();
    void get_text(std::string& out_stable, std::string& out_partial);

    bool ready() const;
    size_t buffered_sample_count() const;
    int partial_full_decode_count() const;
    int partial_tail_decode_count() const;
    ModelSlot* slot() const { return slot_; }
    const std::string& last_error() const { return last_error_; }

private:
    bool transcribe_segment_raw(const std::vector<float>& segment,
                                const std::string& past_text,
                                std::string& language_out,
                                std::string& text_out);
    static int find_split_point(const float* samples,
                                int sample_count,
                                int target_sample,
                                float search_sec);
    static bool should_insert_boundary_space(char prev_ch, char next_ch);

    ModelSlot* slot_ = nullptr;
    mutable std::mutex mutex_;
    std::vector<float> audio_buffer_;
    size_t stable_samples_offset_ = 0;
    size_t partial_processed_audio_size_ = 0;
    size_t partial_processed_stable_offset_ = 0;
    std::string stable_text_;
    std::string partial_text_;
    std::string past_text_;
    std::string last_error_;
    int partial_full_decode_count_ = 0;
    int partial_tail_decode_count_ = 0;
};

void parse_asr_output(const std::string& raw,
                      const std::string& forced_language,
                      std::string& language_out,
                      std::string& text_out);

}  // namespace aila::alia
