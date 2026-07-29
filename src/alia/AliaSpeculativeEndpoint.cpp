#include "AliaSpeculativeEndpoint.hpp"

#include "AliaAsrPipeline.hpp"
#include "AliaForegroundPipeline.hpp"
#include "../utils/EnvUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <utility>

namespace aila::alia {
namespace {

using Clock = std::chrono::steady_clock;

long long elapsed_ms(Clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
}

std::string trim_ascii(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

bool decode_utf8_codepoint(const std::string& text, size_t& offset, unsigned int& codepoint) {
    if (offset >= text.size()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80) {
        codepoint = first;
        ++offset;
        return true;
    }
    int continuation_count = 0;
    unsigned int value = 0;
    if ((first & 0xE0) == 0xC0) {
        continuation_count = 1;
        value = first & 0x1F;
    } else if ((first & 0xF0) == 0xE0) {
        continuation_count = 2;
        value = first & 0x0F;
    } else if ((first & 0xF8) == 0xF0) {
        continuation_count = 3;
        value = first & 0x07;
    } else {
        ++offset;
        return false;
    }
    if (offset + static_cast<size_t>(continuation_count) >= text.size()) {
        offset = text.size();
        return false;
    }
    for (int i = 1; i <= continuation_count; ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[offset + static_cast<size_t>(i)]);
        if ((ch & 0xC0) != 0x80) {
            ++offset;
            return false;
        }
        value = (value << 6) | (ch & 0x3F);
    }
    offset += static_cast<size_t>(continuation_count + 1);
    codepoint = value;
    return true;
}

bool is_unicode_punctuation(unsigned int codepoint) {
    if (codepoint < 0x80) {
        return std::ispunct(static_cast<unsigned char>(codepoint)) != 0 ||
               std::isspace(static_cast<unsigned char>(codepoint)) != 0;
    }
    return (codepoint >= 0x2000 && codepoint <= 0x206F) ||
           (codepoint >= 0x3000 && codepoint <= 0x303F) ||
           (codepoint >= 0xFE10 && codepoint <= 0xFE1F) ||
           (codepoint >= 0xFE30 && codepoint <= 0xFE4F) ||
           (codepoint >= 0xFF00 && codepoint <= 0xFF65);
}

AliaSpeculativeEndpointOperations make_operations(
    AliaAsrPipeline* asr_pipeline,
    AliaForegroundPipeline* foreground_pipeline) {
    AliaSpeculativeEndpointOperations operations;
    operations.reset_asr = [asr_pipeline]() {
        if (asr_pipeline) {
            asr_pipeline->reset();
        }
    };
    operations.get_asr_text = [asr_pipeline](std::string& stable, std::string& partial) {
        if (asr_pipeline) {
            asr_pipeline->get_text(stable, partial);
        }
    };
    operations.get_candidate_asr_text = [asr_pipeline](
        std::string& stable,
        std::string& partial) {
        if (asr_pipeline) {
            asr_pipeline->get_speculative_candidate_text(stable, partial);
        }
    };
    operations.prefill_asr_text = [foreground_pipeline](
        const std::string& stable,
        const std::string& partial) {
        return foreground_pipeline
            ? foreground_pipeline->prefill_asr_text(stable, partial)
            : ALIA_ERR_INVALID_STATE;
    };
    operations.discard_prefill = [foreground_pipeline]() {
        if (foreground_pipeline) {
            foreground_pipeline->discard_asr_prefill();
        }
    };
    operations.start_foreground = [foreground_pipeline](
        const std::string& user_text,
        const AliaGenConfig* config,
        AliaToolCallCallback tool_cb,
        AliaAudioCallback audio_cb,
        void* user_data) {
        return foreground_pipeline && foreground_pipeline->start_turn_with_text(
            user_text, config, tool_cb, audio_cb, user_data);
    };
    operations.prefill_token_count = [foreground_pipeline]() {
        return foreground_pipeline ? foreground_pipeline->last_asr_prefill_token_count() : 0;
    };
    operations.prefill_reused_token_count = [foreground_pipeline]() {
        return foreground_pipeline ? foreground_pipeline->last_asr_prefill_reused_token_count() : 0;
    };
    operations.prefill_suffix_token_count = [foreground_pipeline]() {
        return foreground_pipeline ? foreground_pipeline->last_asr_prefill_suffix_token_count() : 0;
    };
    return operations;
}

}  // namespace

AliaSpeculativeEndpointConfig read_speculative_endpoint_config() {
    AliaSpeculativeEndpointConfig config;
    config.enabled = aila::env::read_flag("AILA_SPECULATIVE_ENDPOINT_PREFILL", false);
    config.candidate_silence_frames = std::max(
        1, aila::env::read_int_raw("AILA_SPECULATIVE_ENDPOINT_SILENCE_FRAMES", 5));
    const int threshold_percent = std::clamp(
        aila::env::read_int_raw("AILA_SPECULATIVE_ENDPOINT_SPEECH_THRESHOLD_PERCENT", 50),
        1,
        99);
    config.speech_probability_threshold = static_cast<float>(threshold_percent) / 100.0f;
    config.text_stable_ms = std::max(
        0, aila::env::read_int_raw("AILA_SPECULATIVE_ENDPOINT_TEXT_STABLE_MS", 160));
    config.allow_cold_candidate =
        aila::env::read_flag("AILA_SPECULATIVE_ENDPOINT_ALLOW_COLD", true);
    return config;
}

bool contains_spoken_utf8_content(const std::string& text) {
    size_t offset = 0;
    while (offset < text.size()) {
        unsigned int codepoint = 0;
        if (!decode_utf8_codepoint(text, offset, codepoint)) {
            continue;
        }
        if (!is_unicode_punctuation(codepoint)) {
            return true;
        }
    }
    return false;
}

AliaSpeculativeEndpointController::AliaSpeculativeEndpointController(
    AliaAsrPipeline* asr_pipeline,
    AliaForegroundPipeline* foreground_pipeline)
    : AliaSpeculativeEndpointController(
          make_operations(asr_pipeline, foreground_pipeline),
          read_speculative_endpoint_config()) {}

AliaSpeculativeEndpointController::AliaSpeculativeEndpointController(
    AliaSpeculativeEndpointOperations operations,
    AliaSpeculativeEndpointConfig config)
    : operations_(std::move(operations)),
      config_(config),
      worker_([this]() { worker_loop(); }) {}

AliaSpeculativeEndpointController::~AliaSpeculativeEndpointController() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        listening_ = false;
        candidate_requested_ = false;
        ++epoch_;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

AliaErrorCode AliaSpeculativeEndpointController::begin() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        ++epoch_;
        listening_ = false;
        candidate_requested_ = false;
        cv_.wait(lock, [this]() { return !candidate_in_flight_; });
        metrics_ = AliaSpeculativeEndpointMetrics{};
        metrics_.state = ALIA_SPECULATIVE_ENDPOINT_LISTENING;
        metrics_.enabled = config_.enabled ? 1 : 0;
        consecutive_silence_frames_ = 0;
        candidate_started_for_pause_ = false;
        candidate_invalidated_for_resume_ = false;
        candidate_text_ready_ = false;
        candidate_stable_text_.clear();
        candidate_partial_text_.clear();
    }
    if (!operations_.reset_asr) {
        return ALIA_ERR_INVALID_STATE;
    }
    operations_.reset_asr();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        listening_ = true;
    }
    return ALIA_OK;
}

AliaErrorCode AliaSpeculativeEndpointController::observe_vad(float speech_probability) {
    if (!std::isfinite(speech_probability) ||
        speech_probability < 0.0f || speech_probability > 1.0f) {
        return ALIA_ERR_INVALID_ARGUMENT;
    }

    bool notify = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!listening_ || !config_.enabled) {
            return ALIA_OK;
        }
        if (speech_probability >= config_.speech_probability_threshold) {
            consecutive_silence_frames_ = 0;
            candidate_started_for_pause_ = false;
            if (!candidate_invalidated_for_resume_ &&
                (candidate_requested_ || candidate_in_flight_ ||
                 metrics_.state == ALIA_SPECULATIVE_ENDPOINT_READY)) {
                ++epoch_;
                candidate_requested_ = false;
                candidate_text_ready_ = false;
                candidate_invalidated_for_resume_ = true;
                ++metrics_.resume_count;
                metrics_.state = ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
            } else if (!candidate_invalidated_for_resume_) {
                metrics_.state = ALIA_SPECULATIVE_ENDPOINT_LISTENING;
            }
            return ALIA_OK;
        }

        ++consecutive_silence_frames_;
        metrics_.last_silence_frames = consecutive_silence_frames_;
        if (!candidate_started_for_pause_ &&
            !candidate_requested_ &&
            !candidate_in_flight_ &&
            consecutive_silence_frames_ >= config_.candidate_silence_frames) {
            candidate_started_for_pause_ = true;
            candidate_invalidated_for_resume_ = false;
            candidate_requested_ = true;
            requested_epoch_ = epoch_;
            ++metrics_.trigger_count;
            ++metrics_.cold_trigger_count;
            metrics_.candidate_silence_ms =
                static_cast<long long>(consecutive_silence_frames_) * 32;
            notify = true;
        }
    }
    if (notify) {
        cv_.notify_all();
    }
    return ALIA_OK;
}

AliaErrorCode AliaSpeculativeEndpointController::commit(
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data) {
    const auto commit_started = Clock::now();
    bool endpoint_enabled = false;
    bool reuse_candidate_text = false;
    unsigned long long commit_epoch = 0;
    std::string stable_text;
    std::string partial_text;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!listening_) {
            return ALIA_ERR_INVALID_STATE;
        }
        metrics_.state = ALIA_SPECULATIVE_ENDPOINT_COMMITTING;
        endpoint_enabled = config_.enabled;
        commit_epoch = epoch_;
        candidate_requested_ = false;
        cv_.wait(lock, [this]() { return !candidate_in_flight_; });
        reuse_candidate_text = endpoint_enabled && candidate_text_ready_ &&
            candidate_text_epoch_ == commit_epoch;
        if (reuse_candidate_text) {
            stable_text = candidate_stable_text_;
            partial_text = candidate_partial_text_;
        }
        listening_ = false;
        metrics_.commit_wait_ms = elapsed_ms(commit_started);
    }

    if (!operations_.get_asr_text || !operations_.start_foreground) {
        return ALIA_ERR_INVALID_STATE;
    }

    const auto final_asr_started = Clock::now();
    if (!reuse_candidate_text) {
        operations_.get_asr_text(stable_text, partial_text);
    }
    const long long final_asr_ms = elapsed_ms(final_asr_started);
    const std::string user_text = combine_asr_text(stable_text, partial_text);
    if (!contains_spoken_utf8_content(user_text)) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.state = ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
        metrics_.final_asr_ms = final_asr_ms;
        metrics_.final_asr_reused_candidate = reuse_candidate_text ? 1 : 0;
        return ALIA_ERR_INVALID_STATE;
    }

    AliaErrorCode prefill_rc = ALIA_OK;
    const auto final_prefill_started = Clock::now();
    if (endpoint_enabled && operations_.prefill_asr_text) {
        prefill_rc = operations_.prefill_asr_text(stable_text, partial_text);
    }
    const long long final_prefill_ms = elapsed_ms(final_prefill_started);
    if (endpoint_enabled && prefill_rc != ALIA_OK && operations_.discard_prefill) {
        operations_.discard_prefill();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (commit_epoch != epoch_ || metrics_.state != ALIA_SPECULATIVE_ENDPOINT_COMMITTING) {
            return ALIA_ERR_ABORTED;
        }
    }

    const bool started = operations_.start_foreground(
        user_text, config, tool_cb, audio_cb, user_data);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.final_asr_ms = final_asr_ms;
        metrics_.final_asr_reused_candidate = reuse_candidate_text ? 1 : 0;
        metrics_.final_prefill_ms = final_prefill_ms;
        metrics_.final_prefill_rc = prefill_rc;
        metrics_.commit_accepted = started ? 1 : 0;
        metrics_.commit_prefill_hit =
            endpoint_enabled && started && prefill_rc == ALIA_OK &&
                    operations_.prefill_token_count &&
                    operations_.prefill_token_count() > 0
                ? 1
                : 0;
        metrics_.final_prefill_tokens = operations_.prefill_token_count
            ? operations_.prefill_token_count()
            : 0;
        metrics_.final_reused_tokens = operations_.prefill_reused_token_count
            ? operations_.prefill_reused_token_count()
            : 0;
        metrics_.final_suffix_tokens = operations_.prefill_suffix_token_count
            ? operations_.prefill_suffix_token_count()
            : 0;
        metrics_.state = started ? ALIA_SPECULATIVE_ENDPOINT_IDLE
                                 : ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
    }
    return started ? ALIA_OK : ALIA_ERR_INVALID_STATE;
}

void AliaSpeculativeEndpointController::cancel() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        listening_ = false;
        candidate_requested_ = false;
        candidate_started_for_pause_ = false;
        candidate_invalidated_for_resume_ = false;
        consecutive_silence_frames_ = 0;
        candidate_text_ready_ = false;
        ++epoch_;
        metrics_.state = ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
    }
    cv_.notify_all();
}

AliaSpeculativeEndpointMetrics AliaSpeculativeEndpointController::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

void AliaSpeculativeEndpointController::worker_loop() {
    while (true) {
        unsigned long long candidate_epoch = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stop_ || candidate_requested_; });
            if (stop_) {
                return;
            }
            candidate_epoch = requested_epoch_;
            candidate_requested_ = false;
            candidate_in_flight_ = true;
            metrics_.state = ALIA_SPECULATIVE_ENDPOINT_PREFILLING;
        }
        run_candidate(candidate_epoch);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            candidate_in_flight_ = false;
        }
        cv_.notify_all();
    }
}

void AliaSpeculativeEndpointController::run_candidate(unsigned long long candidate_epoch) {
    if ((!operations_.get_candidate_asr_text && !operations_.get_asr_text) ||
        !operations_.prefill_asr_text) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (metrics_.state != ALIA_SPECULATIVE_ENDPOINT_COMMITTING) {
            metrics_.state = ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
        }
        return;
    }

    std::string stable_text;
    std::string partial_text;
    const auto asr_started = Clock::now();
    const auto& get_candidate_asr_text = operations_.get_candidate_asr_text
        ? operations_.get_candidate_asr_text
        : operations_.get_asr_text;
    get_candidate_asr_text(stable_text, partial_text);
    const long long asr_ms = elapsed_ms(asr_started);
    const std::string candidate_text = combine_asr_text(stable_text, partial_text);
    if (!contains_spoken_utf8_content(candidate_text)) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.candidate_asr_ms = asr_ms;
        metrics_.candidate_prefill_rc = ALIA_ERR_INVALID_STATE;
        if (metrics_.state != ALIA_SPECULATIVE_ENDPOINT_COMMITTING) {
            metrics_.state = ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
        }
        return;
    }

    const auto prefill_started = Clock::now();
    const AliaErrorCode prefill_rc = operations_.prefill_asr_text(stable_text, partial_text);
    const long long prefill_ms = elapsed_ms(prefill_started);

    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.candidate_asr_ms = asr_ms;
    metrics_.candidate_prefill_ms = prefill_ms;
    metrics_.candidate_prefill_rc = prefill_rc;
    metrics_.candidate_prefill_tokens = operations_.prefill_token_count
        ? operations_.prefill_token_count()
        : 0;
    if (!listening_ || candidate_epoch != epoch_) {
        ++metrics_.stale_completion_count;
        if (metrics_.state != ALIA_SPECULATIVE_ENDPOINT_COMMITTING) {
            metrics_.state = ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
        }
        return;
    }
    candidate_stable_text_ = stable_text;
    candidate_partial_text_ = partial_text;
    candidate_text_epoch_ = candidate_epoch;
    candidate_text_ready_ = true;
    if (metrics_.state != ALIA_SPECULATIVE_ENDPOINT_COMMITTING) {
        metrics_.state = prefill_rc == ALIA_OK && metrics_.candidate_prefill_tokens > 0
            ? ALIA_SPECULATIVE_ENDPOINT_READY
            : ALIA_SPECULATIVE_ENDPOINT_INVALIDATED;
    }
}

std::string AliaSpeculativeEndpointController::combine_asr_text(
    const std::string& stable_text,
    const std::string& partial_text) {
    const std::string stable = trim_ascii(stable_text);
    const std::string partial = trim_ascii(partial_text);
    if (stable.empty()) {
        return partial;
    }
    if (partial.empty()) {
        return stable;
    }
    return stable + " " + partial;
}

}  // namespace aila::alia
