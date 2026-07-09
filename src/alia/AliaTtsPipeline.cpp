#include "AliaTtsPipeline.hpp"

#include "AliaTtsTextChunker.hpp"
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
#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>

namespace aila::alia {
namespace {

constexpr int kTtsSamplesPerFrame = 1920;
constexpr const char* kTtsReferenceAudioEnv = "AILA_TTS_REF_AUDIO";
constexpr const char* kDefaultTtsReferenceAudio = "alia_ref.wav";

int tts_stream_batch_frames() {
    static const int frames = std::clamp(
        aila::env::read_int_raw("AILA_TTS_STREAM_BATCH_FRAMES", 4),
        1,
        24);
    return frames;
}

int tts_first_audio_samples() {
    static const int frames = std::clamp(
        aila::env::read_int_raw("AILA_TTS_FIRST_AUDIO_FRAMES", tts_stream_batch_frames()),
        1,
        tts_stream_batch_frames());
    return frames * kTtsSamplesPerFrame;
}

int tts_audio_callback_max_frames() {
    static const int frames = std::clamp(
        aila::env::read_int_raw("AILA_TTS_AUDIO_CALLBACK_MAX_FRAMES",
                                tts_stream_batch_frames()),
        1,
        24);
    return frames;
}

int tts_audio_callback_max_samples() {
    return tts_audio_callback_max_frames() * kTtsSamplesPerFrame;
}

void append_audio_callbacks(std::vector<std::vector<float>>& callbacks,
                            const std::vector<float>& samples,
                            size_t begin,
                            size_t end) {
    if (begin >= end) {
        return;
    }
    const size_t max_samples =
        static_cast<size_t>(std::max(1, tts_audio_callback_max_samples()));
    while (begin < end) {
        const size_t next = std::min(end, begin + max_samples);
        callbacks.emplace_back(samples.begin() + static_cast<std::ptrdiff_t>(begin),
                               samples.begin() + static_cast<std::ptrdiff_t>(next));
        begin = next;
    }
}

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

    std::vector<TtsPreparedSegment> segments =
        split_tts_text_pause_segments(text, read_tts_pause_segment_config());
    if (segments.empty()) {
        return false;
    }

    bool queued = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (TtsPreparedSegment& segment : segments) {
            if (segment.kind == TtsPreparedSegmentKind::Text) {
                if (segment.text.empty()) {
                    continue;
                }
                text_queue_.push_back(TtsQueueItem{std::move(segment.text), 0});
                queued = true;
            } else if (segment.silence_ms > 0) {
                text_queue_.push_back(TtsQueueItem{{}, segment.silence_ms});
                queued = true;
            }
        }
    }
    if (!queued) {
        return false;
    }
    cv_.notify_all();
    return true;
}

void AliaTtsPipeline::begin_turn_metrics() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_ = AliaTtsMetrics{};
    const TtsSilenceLookaheadPrefetchConfig prefetch_config =
        read_tts_silence_lookahead_prefetch_config();
    metrics_.silence_lookahead_prefetch_enabled =
        prefetch_config.enabled ? 1 : 0;
    metrics_.silence_lookahead_prefetch_min_text_bytes =
        prefetch_config.min_text_bytes;
    first_audio_buffer_.clear();
    first_audio_callback_emitted_ = false;
    first_audio_synthesis_active_ = false;
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
        first_audio_buffer_.clear();
        first_audio_callback_emitted_ = false;
        first_audio_synthesis_active_ = false;
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

    std::deque<TtsQueueItem> pending;
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

    for (const auto& item : pending) {
        if (cancelled()) {
            break;
        }
        const bool ok = item.silence_ms > 0
            ? synthesize_silence(item.silence_ms, audio_cb, user_data, should_cancel)
            : synthesize_text(item.text, config, audio_cb, user_data, should_cancel);
        if (!ok) {
            return false;
        }

        if (cancelled()) {
            break;
        }
    }

    std::vector<float> final_first_audio = flush_first_audio_buffer();
    if (!final_first_audio.empty() && !cancelled()) {
        audio_cb(final_first_audio.data(), static_cast<int>(final_first_audio.size()), user_data);
    }

    return true;
}

void AliaTtsPipeline::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    text_queue_.clear();
    first_audio_buffer_.clear();
    first_audio_callback_emitted_ = false;
    first_audio_synthesis_active_ = false;
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

bool AliaTtsPipeline::first_audio_callback_emitted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return first_audio_callback_emitted_;
}

bool AliaTtsPipeline::first_audio_synthesis_active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return first_audio_synthesis_active_;
}

AliaTtsMetrics AliaTtsPipeline::last_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

std::vector<std::vector<float>> AliaTtsPipeline::prepare_audio_callbacks(
    const std::vector<float>& samples) {
    std::vector<std::vector<float>> callbacks;
    if (samples.empty()) {
        return callbacks;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (first_audio_callback_emitted_) {
        metrics_.audio_callback_max_frames = tts_audio_callback_max_frames();
        append_audio_callbacks(callbacks, samples, 0, samples.size());
        return callbacks;
    }

    first_audio_buffer_.insert(first_audio_buffer_.end(), samples.begin(), samples.end());
    metrics_.audio_callback_max_frames = tts_audio_callback_max_frames();
    const int first_audio_samples = tts_first_audio_samples();
    if (static_cast<int>(first_audio_buffer_.size()) < first_audio_samples) {
        return callbacks;
    }

    callbacks.emplace_back(first_audio_buffer_.begin(),
                           first_audio_buffer_.begin() + first_audio_samples);
    metrics_.first_backend_audio_samples = first_audio_samples;
    first_audio_callback_emitted_ = true;
    first_audio_synthesis_active_ = false;
    if (first_audio_buffer_.size() > static_cast<size_t>(first_audio_samples)) {
        append_audio_callbacks(callbacks,
                               first_audio_buffer_,
                               static_cast<size_t>(first_audio_samples),
                               first_audio_buffer_.size());
    }
    first_audio_buffer_.clear();
    return callbacks;
}

std::vector<float> AliaTtsPipeline::flush_first_audio_buffer() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (first_audio_callback_emitted_ || first_audio_buffer_.empty()) {
        return {};
    }

    std::vector<float> padded = std::move(first_audio_buffer_);
    const int first_audio_samples = tts_first_audio_samples();
    padded.resize(first_audio_samples, 0.0f);
    first_audio_buffer_.clear();
    first_audio_callback_emitted_ = true;
    first_audio_synthesis_active_ = false;
    metrics_.first_backend_audio_samples = first_audio_samples;
    metrics_.audio_callback_max_frames = tts_audio_callback_max_frames();
    return padded;
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
    bool first_audio_synthesis = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        first_audio_synthesis =
            metrics_.chunks_synthesized == 0 && !first_audio_callback_emitted_;
        if (first_audio_synthesis) {
            first_audio_synthesis_active_ = true;
        }
    }
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
            tts_stream_batch_frames(),
            [&](const std::vector<float>& samples) {
                if (samples.empty()) {
                    return;
                }
                if (cancelled()) {
                    return;
                }
                emitted_backend_audio = true;
                std::vector<std::vector<float>> callbacks =
                    prepare_audio_callbacks(samples);
                if (callbacks.empty()) {
                    return;
                }
                auto callback_unlock = lane_lock.scoped_unlock();
                for (const auto& callback_samples : callbacks) {
                    if (callback_samples.empty() || cancelled()) {
                        break;
                    }
                    audio_cb(callback_samples.data(),
                             static_cast<int>(callback_samples.size()),
                             user_data);
                }
            },
            should_cancel);
        backend_timing = tts_backend->last_tts_backend_timing();
    }
    if (first_audio_synthesis) {
        std::lock_guard<std::mutex> lock(mutex_);
        first_audio_synthesis_active_ = false;
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
            if (metrics_.first_backend_audio_samples == 0 &&
                backend_timing.first_audio_samples >= tts_first_audio_samples()) {
                metrics_.first_backend_audio_samples = backend_timing.first_audio_samples;
            }
            metrics_.backend_stream_batch_frames = backend_timing.stream_batch_frames;
            metrics_.backend_initial_stream_batch_frames =
                backend_timing.initial_stream_batch_frames;
            metrics_.backend_steady_stream_batch_frames =
                backend_timing.steady_stream_batch_frames;
            metrics_.backend_playback_aware_steady_batch =
                backend_timing.playback_aware_steady_batch;
            metrics_.audio_callback_max_frames = tts_audio_callback_max_frames();
            metrics_.first_backend_codes_ms = backend_timing.codes_ms;
            metrics_.first_backend_mimi_init_ms = backend_timing.mimi_init_ms;
            metrics_.first_backend_audio_ms = backend_timing.first_audio_ms;
            metrics_.first_backend_total_ms = backend_timing.total_ms;
        }
        metrics_.backend_steady_batch_callback_count +=
            backend_timing.steady_batch_callback_count;
        ++metrics_.chunks_synthesized;
        if (backend_timing.total_ms > 0.0) {
            metrics_.backend_total_ms += backend_timing.total_ms;
        }
    }
    return true;
}

bool AliaTtsPipeline::synthesize_text_to_buffered_callbacks(
    const std::string& text,
    const AliaGenConfig& config,
    TtsBufferedAudio& buffered_audio,
    std::function<bool()> should_cancel) {
    buffered_audio.callbacks.clear();
    buffered_audio.sample_count = 0;

    const auto started = std::chrono::high_resolution_clock::now();
    const bool ok = synthesize_text(
        text,
        config,
        [](const float* samples, int sample_count, void* user_data) {
            if (!samples || sample_count <= 0 || !user_data) {
                return;
            }
            auto* audio = static_cast<TtsBufferedAudio*>(user_data);
            audio->callbacks.emplace_back(samples, samples + sample_count);
            audio->sample_count += sample_count;
        },
        &buffered_audio,
        should_cancel);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - started).count();

    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.silence_lookahead_prefetch_backend_ms += elapsed_ms;
    metrics_.silence_lookahead_prefetch_audio_callbacks +=
        static_cast<int>(buffered_audio.callbacks.size());
    metrics_.silence_lookahead_prefetch_audio_samples +=
        buffered_audio.sample_count;
    return ok;
}

bool AliaTtsPipeline::synthesize_silence(int silence_ms,
                                         AliaAudioCallback audio_cb,
                                         void* user_data,
                                         std::function<bool()> should_cancel) {
    if (!audio_cb || silence_ms <= 0) {
        return false;
    }

    auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };
    if (cancelled()) {
        return true;
    }

    constexpr int kSampleRate = 24000;
    const int sample_count = std::max(
        1,
        static_cast<int>((static_cast<long long>(silence_ms) * kSampleRate) / 1000));
    std::vector<float> samples(static_cast<size_t>(sample_count), 0.0f);
    std::vector<std::vector<float>> callbacks = prepare_audio_callbacks(samples);
    for (const auto& callback_samples : callbacks) {
        if (callback_samples.empty() || cancelled()) {
            break;
        }
        audio_cb(callback_samples.data(),
                 static_cast<int>(callback_samples.size()),
                 user_data);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++metrics_.pause_silence_segments;
        metrics_.pause_silence_ms += silence_ms;
        metrics_.audio_callback_max_frames = tts_audio_callback_max_frames();
    }
    return true;
}

bool AliaTtsPipeline::synthesize_silence_with_lookahead(
    int silence_ms,
    const AliaGenConfig& config,
    AliaAudioCallback audio_cb,
    void* user_data,
    std::function<bool()> should_cancel) {
    auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };

    TtsQueueItem lookahead_text;
    bool prefetch = false;
    const TtsSilenceLookaheadPrefetchConfig prefetch_config =
        read_tts_silence_lookahead_prefetch_config();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.silence_lookahead_prefetch_enabled =
            prefetch_config.enabled ? 1 : 0;
        metrics_.silence_lookahead_prefetch_min_text_bytes =
            prefetch_config.min_text_bytes;
        if (prefetch_config.enabled && silence_ms > 0) {
            ++metrics_.silence_lookahead_prefetch_attempts;
        }

        const bool next_is_text =
            !text_queue_.empty() &&
            text_queue_.front().silence_ms <= 0 &&
            !text_queue_.front().text.empty();
        const int next_text_bytes = next_is_text
            ? static_cast<int>(text_queue_.front().text.size())
            : 0;
        const TtsSilenceLookaheadPrefetchRequest request{
            first_audio_callback_emitted_,
            silence_ms,
            next_is_text,
            next_text_bytes,
        };
        prefetch =
            should_prefetch_tts_silence_lookahead(prefetch_config, request);
        if (!prefetch &&
            prefetch_config.enabled &&
            next_text_bytes > 0 &&
            next_text_bytes < prefetch_config.min_text_bytes) {
            ++metrics_.silence_lookahead_prefetch_tiny_skips;
        }
        if (prefetch) {
            lookahead_text = std::move(text_queue_.front());
            text_queue_.pop_front();
            ++metrics_.silence_lookahead_prefetch_hits;
            metrics_.silence_lookahead_prefetch_text_bytes +=
                static_cast<int>(lookahead_text.text.size());
        }
    }

    if (!prefetch) {
        return synthesize_silence(silence_ms, audio_cb, user_data, should_cancel);
    }

    TtsBufferedAudio buffered_audio;
    const bool text_ok =
        synthesize_text_to_buffered_callbacks(
            lookahead_text.text,
            config,
            buffered_audio,
            should_cancel);
    if (!text_ok) {
        return cancelled();
    }
    if (cancelled()) {
        return true;
    }

    if (!synthesize_silence(silence_ms, audio_cb, user_data, should_cancel)) {
        return false;
    }
    if (cancelled()) {
        return true;
    }

    for (const auto& callback_samples : buffered_audio.callbacks) {
        if (callback_samples.empty()) {
            continue;
        }
        if (cancelled()) {
            return true;
        }
        audio_cb(callback_samples.data(),
                 static_cast<int>(callback_samples.size()),
                 user_data);
    }
    return true;
}

void AliaTtsPipeline::async_worker_loop() {
    while (true) {
        TtsQueueItem item;
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
                    AliaAudioCallback finish_audio_cb = async_audio_cb_;
                    void* finish_user_data = async_user_data_;
                    auto finish_should_cancel = async_should_cancel_;
                    lock.unlock();
                    std::vector<float> final_first_audio;
                    if (!(finish_should_cancel && finish_should_cancel())) {
                        final_first_audio = flush_first_audio_buffer();
                    }
                    if (!final_first_audio.empty() && finish_audio_cb) {
                        finish_audio_cb(final_first_audio.data(),
                                        static_cast<int>(final_first_audio.size()),
                                        finish_user_data);
                    }
                    lock.lock();
                    async_active_ = false;
                    return;
                }
                continue;
            }
            item = std::move(text_queue_.front());
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
        const bool ok = item.silence_ms > 0
            ? synthesize_silence_with_lookahead(
                  item.silence_ms,
                  config,
                  audio_cb,
                  user_data,
                  should_cancel)
            : synthesize_text(item.text, config, audio_cb, user_data, should_cancel);
        if (!ok) {
            std::lock_guard<std::mutex> lock(mutex_);
            async_failed_ = true;
        }
    }
}

}  // namespace aila::alia
