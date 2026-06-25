#include "AliaTtsPipeline.hpp"

#include "ModelSlot.hpp"
#include "../models/IModelBackend.hpp"
#include "../models/Qwen3TTSBackend.hpp"
#include "../audio/SpeakerEncoder.hpp"
#include "../profile/Profiling.hpp"
#include "../utils/EnvUtils.hpp"
#include "../utils/Tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <utility>
#include <vector>

namespace aila::alia {
namespace {

constexpr int kTtsStreamBatchFrames = 12;
constexpr const char* kTtsReferenceAudioEnv = "AILA_TTS_REF_AUDIO";
constexpr const char* kDefaultTtsReferenceAudio = "alia_ref.wav";

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

std::string trim_reference_path(std::string value) {
    auto is_space = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(),
                                            [&](char c) {
                                                return !is_space(static_cast<unsigned char>(c));
                                            }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
                             [&](char c) {
                                 return !is_space(static_cast<unsigned char>(c));
                             }).base(), value.end());
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool reference_audio_disabled(const std::string& value) {
    const std::string lower = lower_ascii(value);
    return lower == "0" || lower == "false" || lower == "off" || lower == "none";
}

std::string resolve_reference_audio_path(std::string* error) {
    std::string raw =
        aila::env::read_string(kTtsReferenceAudioEnv, kDefaultTtsReferenceAudio);
    raw = trim_reference_path(std::move(raw));
    if (raw.empty()) {
        raw = kDefaultTtsReferenceAudio;
    }
    if (reference_audio_disabled(raw)) {
        if (error) {
            *error = std::string(kTtsReferenceAudioEnv) + " disables Alia TTS reference audio";
        }
        return {};
    }

    std::filesystem::path path(raw);
    if (path.is_relative()) {
        path = std::filesystem::current_path() / path;
    }
    path = std::filesystem::absolute(path).lexically_normal();
    return path.make_preferred().string();
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

bool AliaTtsPipeline::preload_reference_voice(std::string* error_message) {
    if (ensure_reference_voice_loaded()) {
        if (error_message) {
            error_message->clear();
        }
        return true;
    }
    if (error_message) {
        std::lock_guard<std::mutex> voice_lock(reference_voice_mutex_);
        *error_message = reference_audio_error_;
    }
    return false;
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

bool AliaTtsPipeline::ensure_reference_voice_loaded() {
    std::lock_guard<std::mutex> voice_lock(reference_voice_mutex_);
    if (reference_voice_loaded_) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.reference_audio_enabled = 1;
        metrics_.reference_embedding_dim =
            static_cast<int>(reference_speaker_embedding_.size());
        metrics_.reference_embedding_ms = reference_embedding_ms_;
        metrics_.reference_audio_path = reference_audio_path_;
        metrics_.reference_audio_error.clear();
        return true;
    }
    if (reference_voice_failed_) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.reference_audio_enabled = 0;
        metrics_.reference_embedding_dim = 0;
        metrics_.reference_embedding_ms = reference_embedding_ms_;
        metrics_.reference_audio_path = reference_audio_path_;
        metrics_.reference_audio_error = reference_audio_error_;
        return false;
    }
    if (!slot_) {
        reference_voice_failed_ = true;
        reference_audio_error_ = "TTS model slot is not configured";
    } else {
        std::string resolve_error;
        reference_audio_path_ = resolve_reference_audio_path(&resolve_error);
        if (reference_audio_path_.empty()) {
            reference_voice_failed_ = true;
            reference_audio_error_ = resolve_error.empty()
                ? "Alia TTS reference audio path is empty"
                : resolve_error;
        } else if (!std::filesystem::exists(std::filesystem::path(reference_audio_path_))) {
            reference_voice_failed_ = true;
            reference_audio_error_ = "Alia TTS reference audio not found: " + reference_audio_path_;
        }
    }

    const auto start = std::chrono::high_resolution_clock::now();
    if (!reference_voice_failed_) {
        std::string error;
        aila::audio::SpeakerEncoder encoder;
        std::filesystem::path weights_path =
            std::filesystem::path(slot_->model_dir()) / "model.safetensors";
        if (!encoder.loadWeights(weights_path.make_preferred().string(), &error)) {
            reference_voice_failed_ = true;
            reference_audio_error_ =
                "Failed to load speaker encoder weights: " + error;
        } else if (!encoder.extractEmbeddingFromFile(reference_audio_path_,
                                                     reference_speaker_embedding_,
                                                     &error)) {
            reference_voice_failed_ = true;
            reference_audio_error_ =
                "Failed to extract Alia TTS reference embedding: " + error;
        } else if (reference_speaker_embedding_.empty()) {
            reference_voice_failed_ = true;
            reference_audio_error_ = "Alia TTS reference embedding is empty";
        } else {
            reference_voice_loaded_ = true;
            reference_audio_error_.clear();
        }
    }
    reference_embedding_ms_ =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.reference_audio_enabled = reference_voice_loaded_ ? 1 : 0;
        metrics_.reference_embedding_dim =
            static_cast<int>(reference_speaker_embedding_.size());
        metrics_.reference_embedding_ms = reference_embedding_ms_;
        metrics_.reference_audio_path = reference_audio_path_;
        metrics_.reference_audio_error = reference_audio_error_;
    }

    if (!reference_voice_loaded_) {
        AILA_LOG_ERROR("[AliaTTS] Reference voice unavailable: %s",
                       reference_audio_error_.c_str());
        return false;
    }

    AILA_LOG_INFO("[AliaTTS] Reference voice loaded: path=%s dim=%zu ms=%.2f",
                  reference_audio_path_.c_str(),
                  reference_speaker_embedding_.size(),
                  reference_embedding_ms_);
    return true;
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
    if (!ensure_reference_voice_loaded()) {
        return cancelled();
    }
    std::vector<float> speaker_embedding;
    {
        std::lock_guard<std::mutex> voice_lock(reference_voice_mutex_);
        speaker_embedding = reference_speaker_embedding_;
    }
    auto* tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend);
    if (!tts_backend) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.reference_audio_enabled = 0;
        metrics_.reference_audio_error = "Alia TTS requires Qwen3TTSBackend for reference voice";
        return false;
    }

    bool emitted_backend_audio = false;
    bool backend_ok = false;
    IModelBackend::TtsBackendTiming backend_timing;
    {
        auto lane_lock = context->lock_execution();
        backend_ok = tts_backend->synthesize_codes_stream(
            *context,
            text_tokens,
            speaker_embedding,
            0,
            {},
            0,
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
                auto callback_unlock = lane_lock.scoped_unlock();
                audio_cb(samples.data(), static_cast<int>(samples.size()), user_data);
            },
            should_cancel);
        backend_timing = tts_backend->last_tts_backend_timing();
    }
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
