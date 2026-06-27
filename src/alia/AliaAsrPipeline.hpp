#pragma once

#include "../audio/AudioPreprocessor.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace aila::alia {

class ModelSlot;

struct AliaAsrMetrics {
    int transcribe_calls = 0;
    int generated_tokens = 0;
    int prefix_reuse_attempts = 0;
    int prefix_reuse_hits = 0;
    int prefix_reused_tokens = 0;
    int prefix_appended_tokens = 0;
    int mel_cache_hits = 0;
    int mel_cache_reused_frames = 0;
    int mel_cache_computed_frames = 0;
    double input_audio_ms = 0.0;
    double mel_ms = 0.0;
    double mel_stft_ms = 0.0;
    double mel_norm_ms = 0.0;
    double mel_cache_max_abs_diff = 0.0;
    double upload_ms = 0.0;
    double encoder_ms = 0.0;
    double encoder_conv_ms = 0.0;
    double encoder_transformer_ms = 0.0;
    double encoder_proj_ms = 0.0;
    double readback_ms = 0.0;
    double prompt_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
};

class AliaAsrPipeline {
public:
    explicit AliaAsrPipeline(ModelSlot* slot);

    bool feed_audio(const float* samples, int sample_count);
    void append_stable_text(std::string text);
    bool process_pending(bool force_partial_decode = true);
    void reset();
    void get_text(std::string& out_stable, std::string& out_partial);
    void get_partial_text(std::string& out_stable, std::string& out_partial);

    bool ready() const;
    size_t buffered_sample_count() const;
    int partial_full_decode_count() const;
    int partial_tail_decode_count() const;
    int partial_throttled_count() const;
    AliaAsrMetrics last_metrics() const;
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
    int partial_throttled_count_ = 0;
    AliaAsrMetrics metrics_;
    MelSpectrogramCache partial_mel_cache_;
    size_t partial_mel_cache_stable_offset_ = 0;

    struct PrefixCache {
        bool valid = false;
        size_t stable_samples_offset = 0;
        int audio_len = 0;
        int prefix_len = 0;
        std::vector<int> prefix_token_ids;

        void reset() {
            valid = false;
            stable_samples_offset = 0;
            audio_len = 0;
            prefix_len = 0;
            prefix_token_ids.clear();
        }
    } prefix_cache_;
};

void parse_asr_output(const std::string& raw,
                      const std::string& forced_language,
                      std::string& language_out,
                      std::string& text_out);

}  // namespace aila::alia
