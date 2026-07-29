#pragma once

#include "alia_api.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace aila::alia {

class AliaAsrPipeline;
class AliaForegroundPipeline;

struct AliaSpeculativeEndpointConfig {
    bool enabled = false;
    int candidate_silence_frames = 5;
    float speech_probability_threshold = 0.5f;
    int text_stable_ms = 160;
    bool allow_cold_candidate = true;
};

struct AliaSpeculativeEndpointOperations {
    std::function<void()> reset_asr;
    std::function<void(std::string&, std::string&)> get_asr_text;
    std::function<void(std::string&, std::string&)> get_candidate_asr_text;
    std::function<AliaErrorCode(const std::string&, const std::string&)> prefill_asr_text;
    std::function<void()> discard_prefill;
    std::function<bool(const std::string&,
                       const AliaGenConfig*,
                       AliaToolCallCallback,
                       AliaAudioCallback,
                       void*)> start_foreground;
    std::function<int()> prefill_token_count;
    std::function<int()> prefill_reused_token_count;
    std::function<int()> prefill_suffix_token_count;
};

AliaSpeculativeEndpointConfig read_speculative_endpoint_config();
bool contains_spoken_utf8_content(const std::string& text);

class AliaSpeculativeEndpointController {
public:
    AliaSpeculativeEndpointController(AliaAsrPipeline* asr_pipeline,
                                      AliaForegroundPipeline* foreground_pipeline);
    AliaSpeculativeEndpointController(AliaSpeculativeEndpointOperations operations,
                                      AliaSpeculativeEndpointConfig config);
    ~AliaSpeculativeEndpointController();

    AliaErrorCode begin();
    AliaErrorCode observe_vad(float speech_probability);
    AliaErrorCode commit(const AliaGenConfig* config,
                         AliaToolCallCallback tool_cb,
                         AliaAudioCallback audio_cb,
                         void* user_data);
    void cancel();
    AliaSpeculativeEndpointMetrics metrics() const;

private:
    void worker_loop();
    void run_candidate(unsigned long long candidate_epoch);
    static std::string combine_asr_text(const std::string& stable_text,
                                        const std::string& partial_text);

    AliaSpeculativeEndpointOperations operations_;
    AliaSpeculativeEndpointConfig config_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stop_ = false;
    bool listening_ = false;
    bool candidate_requested_ = false;
    bool candidate_in_flight_ = false;
    bool candidate_started_for_pause_ = false;
    bool candidate_invalidated_for_resume_ = false;
    int consecutive_silence_frames_ = 0;
    unsigned long long epoch_ = 0;
    unsigned long long requested_epoch_ = 0;
    unsigned long long candidate_text_epoch_ = 0;
    std::string candidate_stable_text_;
    std::string candidate_partial_text_;
    bool candidate_text_ready_ = false;
    AliaSpeculativeEndpointMetrics metrics_{};
};

}  // namespace aila::alia
