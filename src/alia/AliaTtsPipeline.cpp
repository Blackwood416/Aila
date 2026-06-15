#include "AliaTtsPipeline.hpp"

#include "ModelSlot.hpp"
#include "../models/IModelBackend.hpp"
#include "../utils/Tokenizer.hpp"

#include <utility>
#include <vector>

namespace aila::alia {
namespace {

constexpr int kTtsStreamBatchFrames = 6;

GenerationConfig translate_tts_generation_config(const AliaGenConfig& config) {
    GenerationConfig translated;
    translated.max_new_tokens = config.max_tokens;
    translated.temperature = config.temperature;
    translated.top_p = config.top_p;
    translated.do_sample = config.temperature > 0.0f;
    return translated;
}

std::string format_tts_text_for_backend(const std::string& text) {
    return "<|im_start|>assistant\n" + text +
           "<|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace

AliaTtsPipeline::AliaTtsPipeline(ModelSlot* slot)
    : slot_(slot) {}

AliaTtsPipeline::~AliaTtsPipeline() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        async_finishing_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool AliaTtsPipeline::enqueue_text(std::string text) {
    if (text.empty()) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        text_queue_.push_back(std::move(text));
    }
    cv_.notify_all();
    return true;
}

void AliaTtsPipeline::begin_turn_metrics() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_ = AliaTtsMetrics{};
}

bool AliaTtsPipeline::start_async_turn(const AliaGenConfig& config,
                                       AliaAudioCallback audio_cb,
                                       void* user_data,
                                       std::function<bool()> should_cancel) {
    if (!audio_cb || !ready()) {
        return false;
    }
    if (worker_.joinable()) {
        worker_.join();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        text_queue_.clear();
        async_config_ = config;
        async_audio_cb_ = audio_cb;
        async_user_data_ = user_data;
        async_should_cancel_ = std::move(should_cancel);
        async_finishing_ = false;
        async_failed_ = false;
        async_active_ = true;
    }

    worker_ = std::thread([this]() { async_worker_loop(); });
    return true;
}

bool AliaTtsPipeline::finish_async_turn() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!async_active_) {
            return !async_failed_;
        }
        async_finishing_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const bool ok = !async_failed_;
    async_active_ = false;
    async_finishing_ = false;
    async_audio_cb_ = nullptr;
    async_user_data_ = nullptr;
    async_should_cancel_ = {};
    return ok;
}

bool AliaTtsPipeline::synthesize_pending(const AliaGenConfig& config,
                                         AliaAudioCallback audio_cb,
                                         void* user_data,
                                         std::function<bool()> should_cancel) {
    if (!audio_cb) {
        return false;
    }

    auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };

    std::deque<std::string> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending.swap(text_queue_);
    }

    if (pending.empty()) {
        return false;
    }
    if (!ready()) {
        return false;
    }

    for (const auto& text : pending) {
        if (cancelled()) {
            break;
        }
        if (!synthesize_text(text, config, audio_cb, user_data, should_cancel)) {
            return false;
        }

        if (cancelled()) {
            break;
        }
    }

    return true;
}

void AliaTtsPipeline::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    text_queue_.clear();
}

bool AliaTtsPipeline::ready() const {
    return slot_ &&
           slot_->state() == ModelSlotState::Loaded &&
           slot_->context() &&
           slot_->tokenizer() &&
           slot_->backend();
}

size_t AliaTtsPipeline::pending_text_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return text_queue_.size();
}

AliaTtsMetrics AliaTtsPipeline::last_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

bool AliaTtsPipeline::synthesize_text(const std::string& text,
                                      const AliaGenConfig& config,
                                      AliaAudioCallback audio_cb,
                                      void* user_data,
                                      std::function<bool()> should_cancel) {
    if (!audio_cb || text.empty() || !ready()) {
        return false;
    }

    auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };

    Context* context = slot_->context();
    Tokenizer* tokenizer = slot_->tokenizer();
    IModelBackend* backend = slot_->backend();
    const std::vector<int> text_tokens =
        tokenizer->encode(format_tts_text_for_backend(text));
    if (text_tokens.empty()) {
        return false;
    }

    std::string backend_error;
    bool emitted_backend_audio = false;
    const bool backend_ok = backend->synthesize_tts_stream(
        *context,
        text_tokens,
        translate_tts_generation_config(config),
        kTtsStreamBatchFrames,
        [&](const std::vector<float>& samples) {
            if (samples.empty()) {
                return;
            }
            if (cancelled()) {
                return;
            }
            emitted_backend_audio = true;
            audio_cb(samples.data(), static_cast<int>(samples.size()), user_data);
        },
        &backend_error,
        should_cancel);
    const IModelBackend::TtsBackendTiming backend_timing =
        backend->last_tts_backend_timing();
    if (!backend_ok || !emitted_backend_audio) {
        return cancelled();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (metrics_.chunks_synthesized == 0) {
            metrics_.first_text_chars = static_cast<int>(text.size());
            metrics_.first_text_tokens = static_cast<int>(text_tokens.size());
            metrics_.first_backend_frames = backend_timing.total_frames;
            metrics_.first_backend_callbacks = backend_timing.callback_count;
            metrics_.first_backend_audio_samples = backend_timing.first_audio_samples;
            metrics_.first_backend_codes_ms = backend_timing.codes_ms;
            metrics_.first_backend_mimi_init_ms = backend_timing.mimi_init_ms;
            metrics_.first_backend_audio_ms = backend_timing.first_audio_ms;
            metrics_.first_backend_total_ms = backend_timing.total_ms;
        }
        ++metrics_.chunks_synthesized;
        if (backend_timing.total_ms > 0.0) {
            metrics_.backend_total_ms += backend_timing.total_ms;
        }
    }
    return true;
}

void AliaTtsPipeline::async_worker_loop() {
    while (true) {
        std::string text;
        AliaGenConfig config{};
        AliaAudioCallback audio_cb = nullptr;
        void* user_data = nullptr;
        std::function<bool()> should_cancel;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return async_finishing_ || !text_queue_.empty();
            });
            if (text_queue_.empty()) {
                if (async_finishing_) {
                    async_active_ = false;
                    return;
                }
                continue;
            }
            text = std::move(text_queue_.front());
            text_queue_.pop_front();
            config = async_config_;
            audio_cb = async_audio_cb_;
            user_data = async_user_data_;
            should_cancel = async_should_cancel_;
        }

        if (should_cancel && should_cancel()) {
            std::lock_guard<std::mutex> lock(mutex_);
            async_failed_ = true;
            continue;
        }
        if (!synthesize_text(text, config, audio_cb, user_data, should_cancel)) {
            std::lock_guard<std::mutex> lock(mutex_);
            async_failed_ = true;
        }
    }
}

}  // namespace aila::alia
