#include "AliaTtsPipeline.hpp"

#include "ModelSlot.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace aila::alia {

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
                                         void* user_data) {
    (void)config;
    if (!audio_cb) {
        return false;
    }

    std::deque<std::string> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending.swap(text_queue_);
    }

    if (pending.empty()) {
        return false;
    }

    for (const auto& text : pending) {
        const int frame_count = std::max(160, static_cast<int>(text.size()) * 16);
        std::vector<float> samples(static_cast<size_t>(frame_count), 0.0f);
        if (!ready()) {
            audio_cb(samples.data(), static_cast<int>(samples.size()), user_data);
            continue;
        }

        audio_cb(samples.data(), static_cast<int>(samples.size()), user_data);
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
