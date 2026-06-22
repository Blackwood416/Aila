#pragma once

#include "alia_api.h"

#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace aila::alia {

class ModelSlot;

struct AliaTtsMetrics {
    int chunks_synthesized = 0;
    int first_text_chars = 0;
    int first_text_tokens = 0;
    int first_backend_frames = 0;
    int first_backend_callbacks = 0;
    int first_backend_audio_samples = 0;
    double first_backend_codes_ms = -1.0;
    double first_backend_mimi_init_ms = -1.0;
    double first_backend_audio_ms = -1.0;
    double first_backend_total_ms = -1.0;
    double backend_total_ms = 0.0;
};

class AliaTtsPipeline {
public:
    explicit AliaTtsPipeline(ModelSlot* slot);
    ~AliaTtsPipeline();

    bool enqueue_text(std::string text);
    void begin_turn_metrics();
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
    AliaTtsMetrics last_metrics() const;
    ModelSlot* slot() const { return slot_; }

private:
    bool synthesize_text(const std::string& text,
                         const AliaGenConfig& config,
                         AliaAudioCallback audio_cb,
                         void* user_data,
                         std::function<bool()> should_cancel);
    void async_worker_loop();

    ModelSlot* slot_ = nullptr;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> text_queue_;
    AliaTtsMetrics metrics_;
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
