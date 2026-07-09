#pragma once

#include "alia_api.h"

#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aila::alia {

class ModelSlot;

struct AliaTtsMetrics {
    int chunks_synthesized = 0;
    int reference_audio_enabled = 0;
    int reference_embedding_dim = 0;
    int first_text_chars = 0;
    int first_text_tokens = 0;
    int first_backend_frames = 0;
    int first_backend_callbacks = 0;
    int first_backend_audio_samples = 0;
    int backend_stream_batch_frames = 0;
    int backend_initial_stream_batch_frames = 0;
    int backend_steady_stream_batch_frames = 0;
    int backend_steady_batch_callback_count = 0;
    int backend_playback_aware_steady_batch = 0;
    int audio_callback_max_frames = 0;
    int pause_silence_segments = 0;
    int pause_silence_ms = 0;
    int silence_lookahead_prefetch_enabled = 0;
    int silence_lookahead_prefetch_min_text_bytes = 0;
    int silence_lookahead_prefetch_attempts = 0;
    int silence_lookahead_prefetch_hits = 0;
    int silence_lookahead_prefetch_tiny_skips = 0;
    int silence_lookahead_prefetch_text_bytes = 0;
    int silence_lookahead_prefetch_audio_callbacks = 0;
    int silence_lookahead_prefetch_audio_samples = 0;
    double reference_embedding_ms = -1.0;
    double first_backend_codes_ms = -1.0;
    double first_backend_mimi_init_ms = -1.0;
    double first_backend_audio_ms = -1.0;
    double first_backend_total_ms = -1.0;
    double backend_total_ms = 0.0;
    double silence_lookahead_prefetch_backend_ms = 0.0;
    std::string reference_audio_path;
    std::string reference_audio_error;
};

class AliaTtsPipeline {
public:
    explicit AliaTtsPipeline(ModelSlot* slot);
    ~AliaTtsPipeline();

    bool enqueue_text(std::string text);
    void begin_turn_metrics();
    bool preload_reference_voice(std::string* error_message = nullptr);
    bool start_async_turn(const AliaGenConfig& config,
                          AliaAudioCallback audio_cb,
                          void* user_data,
                          std::function<bool()> should_cancel = {});
    bool finish_async_turn();
    bool synthesize_pending(const AliaGenConfig& config,
                            AliaAudioCallback audio_cb,
                            void* user_data,
                            std::function<bool()> should_cancel = {});
    void reset();

    bool ready() const;
    size_t pending_text_count() const;
    bool first_audio_callback_emitted() const;
    bool first_audio_synthesis_active() const;
    AliaTtsMetrics last_metrics() const;
    ModelSlot* slot() const { return slot_; }

private:
    struct TtsQueueItem {
        std::string text;
        int silence_ms = 0;
    };

    struct TtsBufferedAudio {
        std::vector<std::vector<float>> callbacks;
        int sample_count = 0;
    };

    bool synthesize_text(const std::string& text,
                         const AliaGenConfig& config,
                         AliaAudioCallback audio_cb,
                         void* user_data,
                         std::function<bool()> should_cancel);
    bool synthesize_text_to_buffered_callbacks(
        const std::string& text,
        const AliaGenConfig& config,
        TtsBufferedAudio& buffered_audio,
        std::function<bool()> should_cancel);
    bool synthesize_silence(int silence_ms,
                            AliaAudioCallback audio_cb,
                            void* user_data,
                            std::function<bool()> should_cancel);
    bool synthesize_silence_with_lookahead(
        int silence_ms,
        const AliaGenConfig& config,
        AliaAudioCallback audio_cb,
        void* user_data,
        std::function<bool()> should_cancel);
    bool ensure_reference_voice_loaded();
    std::vector<std::vector<float>> prepare_audio_callbacks(const std::vector<float>& samples);
    std::vector<float> flush_first_audio_buffer();
    void async_worker_loop();

    ModelSlot* slot_ = nullptr;
    mutable std::mutex mutex_;
    mutable std::mutex reference_voice_mutex_;
    std::condition_variable cv_;
    std::deque<TtsQueueItem> text_queue_;
    AliaTtsMetrics metrics_;
    std::vector<float> first_audio_buffer_;
    std::vector<float> reference_speaker_embedding_;
    std::string reference_audio_path_;
    std::string reference_audio_error_;
    double reference_embedding_ms_ = -1.0;
    bool reference_voice_loaded_ = false;
    bool reference_voice_failed_ = false;
    bool first_audio_callback_emitted_ = false;
    bool first_audio_synthesis_active_ = false;
    std::thread worker_;
    bool async_active_ = false;
    bool async_finishing_ = false;
    bool async_failed_ = false;
    AliaGenConfig async_config_{};
    AliaAudioCallback async_audio_cb_ = nullptr;
    void* async_user_data_ = nullptr;
    std::function<bool()> async_should_cancel_;
};

}  // namespace aila::alia
