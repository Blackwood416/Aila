#pragma once

#include "alia_api.h"

#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

namespace aila::alia {

class ModelSlot;

class AliaTtsPipeline {
public:
    explicit AliaTtsPipeline(ModelSlot* slot);

    bool enqueue_text(std::string text);
    bool synthesize_pending(const AliaGenConfig& config,
                            AliaAudioCallback audio_cb,
                            void* user_data,
                            std::function<bool()> should_cancel = {});
    void reset();

    bool ready() const;
    size_t pending_text_count() const;
    ModelSlot* slot() const { return slot_; }

private:
    ModelSlot* slot_ = nullptr;
    mutable std::mutex mutex_;
    std::deque<std::string> text_queue_;
};

}  // namespace aila::alia
