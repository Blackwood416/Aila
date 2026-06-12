#include "AliaTtsPipeline.hpp"

#include "ModelSlot.hpp"
#include "../models/IModelBackend.hpp"
#include "../utils/Tokenizer.hpp"

#include <algorithm>
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

void emit_deterministic_fallback_audio(const std::string& text,
                                       AliaAudioCallback audio_cb,
                                       void* user_data) {
    const int frame_count = std::max(160, static_cast<int>(text.size()) * 16);
    std::vector<float> samples(static_cast<size_t>(frame_count), 0.0f);
    audio_cb(samples.data(), static_cast<int>(samples.size()), user_data);
}

}  // namespace

AliaTtsPipeline::AliaTtsPipeline(ModelSlot* slot)
    : slot_(slot) {}

bool AliaTtsPipeline::enqueue_text(std::string text) {
    if (text.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    text_queue_.push_back(std::move(text));
    return true;
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

    for (const auto& text : pending) {
        if (cancelled()) {
            break;
        }

        bool used_backend_audio = false;
        if (ready()) {
            Context* context = slot_->context();
            Tokenizer* tokenizer = slot_->tokenizer();
            IModelBackend* backend = slot_->backend();
            const std::vector<int> text_tokens =
                tokenizer->encode(format_tts_text_for_backend(text));
            if (!text_tokens.empty()) {
                std::string backend_error;
                bool emitted_backend_audio = false;
                used_backend_audio = backend->synthesize_tts_stream(
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
                used_backend_audio = used_backend_audio || emitted_backend_audio;
            }
        }

        if (cancelled()) {
            break;
        }

        if (!used_backend_audio) {
            emit_deterministic_fallback_audio(text, audio_cb, user_data);
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

}  // namespace aila::alia
