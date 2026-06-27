#include "AliaAsrPipeline.hpp"

#include "ModelSlot.hpp"
#include "../audio/AudioPreprocessor.hpp"
#include "../audio/Qwen3ASRAudioEncoder.hpp"
#include "../core/Tensor.hpp"
#include "../models/IModelBackend.hpp"
#include "../models/Qwen3ASRBnb4Backend.hpp"
#include "../ops/Ops.hpp"
#include "../utils/EnvUtils.hpp"
#include "../utils/Tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <exception>
#include <sycl/sycl.hpp>
#include <utility>

namespace aila::alia {
namespace {

using AsrBf16 = sycl::ext::oneapi::bfloat16;

class DeviceAllocation {
public:
    DeviceAllocation(Context& ctx, size_t bytes)
        : ctx_(&ctx),
          ptr_(ctx.alloc_device(bytes)) {}

    ~DeviceAllocation() {
        if (ctx_ && ptr_) {
            ctx_->free_device(ptr_);
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    template <typename T>
    T* as() const {
        return static_cast<T*>(ptr_);
    }

private:
    Context* ctx_ = nullptr;
    void* ptr_ = nullptr;
};

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(0, 1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string to_lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string normalize_language_name(std::string language) {
    language = trim(std::move(language));
    if (language.empty()) {
        return "";
    }

    language[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(language[0])));
    for (size_t i = 1; i < language.size(); ++i) {
        language[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(language[i])));
    }
    return language;
}

bool should_insert_boundary_space_for_text(char prev_ch, char next_ch) {
    if (std::isspace(static_cast<unsigned char>(prev_ch))) {
        return false;
    }
    if (std::isspace(static_cast<unsigned char>(next_ch))) {
        return false;
    }
    if (std::ispunct(static_cast<unsigned char>(next_ch))) {
        return false;
    }
    return true;
}

std::string join_asr_text(const std::string& prefix, const std::string& suffix) {
    std::string trimmed_suffix = trim(suffix);
    if (prefix.empty()) {
        return trimmed_suffix;
    }
    if (trimmed_suffix.empty()) {
        return prefix;
    }

    std::string out = prefix;
    if (should_insert_boundary_space_for_text(out.back(), trimmed_suffix.front())) {
        out += " ";
    }
    out += trimmed_suffix;
    return out;
}

std::string merge_partial_tail(const std::string& prefix, const std::string& tail) {
    std::string trimmed_tail = trim(tail);
    if (prefix.empty() || trimmed_tail.empty()) {
        return join_asr_text(prefix, trimmed_tail);
    }

    const size_t max_overlap = std::min(prefix.size(), trimmed_tail.size());
    size_t best_overlap = 0;
    for (size_t overlap = max_overlap; overlap >= 3; --overlap) {
        bool matches = true;
        const size_t prefix_start = prefix.size() - overlap;
        for (size_t i = 0; i < overlap; ++i) {
            const unsigned char a = static_cast<unsigned char>(prefix[prefix_start + i]);
            const unsigned char b = static_cast<unsigned char>(trimmed_tail[i]);
            if (std::tolower(a) != std::tolower(b)) {
                matches = false;
                break;
            }
        }
        if (matches) {
            best_overlap = overlap;
            break;
        }
        if (overlap == 3) {
            break;
        }
    }

    if (best_overlap > 0) {
        return prefix + trimmed_tail.substr(best_overlap);
    }
    return join_asr_text(prefix, trimmed_tail);
}

}  // namespace

AliaAsrPipeline::AliaAsrPipeline(ModelSlot* slot)
    : slot_(slot) {}

bool AliaAsrPipeline::feed_audio(const float* samples, int sample_count) {
    if (!samples || sample_count <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    audio_buffer_.insert(audio_buffer_.end(), samples, samples + sample_count);
    return true;
}

void AliaAsrPipeline::append_stable_text(std::string text) {
    text = trim(std::move(text));
    if (text.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!stable_text_.empty() &&
        should_insert_boundary_space(stable_text_.back(), text.front())) {
        stable_text_ += " ";
    }
    stable_text_ += text;
    past_text_ = std::move(text);
    partial_text_.clear();
    partial_processed_audio_size_ = 0;
    partial_processed_stable_offset_ = stable_samples_offset_;
    prefix_cache_.reset();
    partial_mel_cache_.reset();
    partial_mel_cache_stable_offset_ = stable_samples_offset_;
}

bool AliaAsrPipeline::process_pending(bool force_partial_decode) {
    if (!ready()) {
        return false;
    }

    try {
        bool processed = false;
        constexpr size_t kMinStableSamples = static_cast<size_t>(6 * 16000);
        constexpr size_t kTargetChunkSamples = static_cast<size_t>(5 * 16000);
        constexpr size_t kMinPartialSamples = static_cast<size_t>(0.5f * 16000);
        const int min_partial_advance_ms =
            std::max(500, aila::env::read_int_raw("AILA_ASR_PARTIAL_MIN_ADVANCE_MS", 1500));
        const size_t min_partial_advance_samples =
            static_cast<size_t>(min_partial_advance_ms * 16);

        std::lock_guard<std::mutex> lock(mutex_);
        while (true) {
            const size_t available = audio_buffer_.size() - stable_samples_offset_;
            if (available < kMinStableSamples) {
                break;
            }

            const int split_relative = find_split_point(
                audio_buffer_.data() + stable_samples_offset_,
                static_cast<int>(available),
                static_cast<int>(kTargetChunkSamples),
                1.0f);
            const size_t split_absolute = stable_samples_offset_ +
                static_cast<size_t>(std::max(split_relative, 0));
            if (split_absolute <= stable_samples_offset_ ||
                split_absolute > audio_buffer_.size()) {
                break;
            }

            std::vector<float> segment(audio_buffer_.begin() +
                                           static_cast<std::ptrdiff_t>(stable_samples_offset_),
                                       audio_buffer_.begin() +
                                           static_cast<std::ptrdiff_t>(split_absolute));
            if (segment.size() < 8000) {
                segment.resize(8000, 0.0f);
            }

            std::string segment_language;
            std::string segment_text;
            if (!transcribe_segment_raw(segment, past_text_, segment_language, segment_text)) {
                return processed;
            }

            if (!segment_text.empty()) {
                if (!stable_text_.empty() &&
                    should_insert_boundary_space(stable_text_.back(), segment_text.front())) {
                    stable_text_ += " ";
                }
                stable_text_ += segment_text;
                past_text_ = segment_text;
            }

            stable_samples_offset_ = split_absolute;
            partial_processed_audio_size_ = 0;
            partial_processed_stable_offset_ = stable_samples_offset_;
            prefix_cache_.reset();
            partial_mel_cache_.reset();
            partial_mel_cache_stable_offset_ = stable_samples_offset_;
            processed = true;
        }

        const size_t remaining = audio_buffer_.size() - stable_samples_offset_;
        if (remaining >= kMinPartialSamples) {
            if (partial_processed_audio_size_ != audio_buffer_.size() ||
                partial_processed_stable_offset_ != stable_samples_offset_) {
                const bool has_cached_partial =
                    partial_processed_audio_size_ > stable_samples_offset_ &&
                    partial_processed_stable_offset_ == stable_samples_offset_;
                const size_t new_samples_since_partial =
                    has_cached_partial && audio_buffer_.size() > partial_processed_audio_size_
                        ? audio_buffer_.size() - partial_processed_audio_size_
                        : 0;
                const bool throttle_initial_partial =
                    !force_partial_decode && !has_cached_partial &&
                    remaining < min_partial_advance_samples;
                const bool throttle_incremental_partial =
                    !force_partial_decode && has_cached_partial &&
                    new_samples_since_partial < min_partial_advance_samples;
                if (throttle_initial_partial || throttle_incremental_partial) {
                    ++partial_throttled_count_;
                    last_error_.clear();
                    return processed;
                }

                std::string partial_language;
                std::string decoded_text;
                bool tail_decode = false;
                auto run_full_partial_decode = [&]() -> bool {
                    std::vector<float> segment(audio_buffer_.begin() +
                                                   static_cast<std::ptrdiff_t>(stable_samples_offset_),
                                               audio_buffer_.end());
                    if (segment.size() < 8000) {
                        segment.resize(8000, 0.0f);
                    }
                    return transcribe_segment_raw(segment, past_text_, partial_language, decoded_text);
                };
                static const bool s_tail_partial_enabled =
                    aila::env::read_flag("AILA_ASR_TAIL_PARTIAL", false);
                constexpr size_t kMinTailSamples = static_cast<size_t>(0.5f * 16000);
                constexpr size_t kTailOverlapSamples = static_cast<size_t>(0.5f * 16000);
                if (s_tail_partial_enabled &&
                    !partial_text_.empty() &&
                    partial_processed_stable_offset_ == stable_samples_offset_ &&
                    partial_processed_audio_size_ > stable_samples_offset_ &&
                    audio_buffer_.size() > partial_processed_audio_size_ &&
                    audio_buffer_.size() - partial_processed_audio_size_ >= kMinTailSamples) {
                    const std::string previous_partial = partial_text_;
                    const size_t tail_start = partial_processed_audio_size_ >
                            stable_samples_offset_ + kTailOverlapSamples
                        ? partial_processed_audio_size_ - kTailOverlapSamples
                        : stable_samples_offset_;
                    std::vector<float> segment(audio_buffer_.begin() +
                                                   static_cast<std::ptrdiff_t>(tail_start),
                                               audio_buffer_.end());
                    if (segment.size() < 8000) {
                        segment.resize(8000, 0.0f);
                    }
                    const std::string context_text = join_asr_text(stable_text_, partial_text_);
                    if (transcribe_segment_raw(segment, context_text, partial_language, decoded_text)) {
                        const std::string merged = merge_partial_tail(partial_text_, decoded_text);
                        if (!decoded_text.empty() && merged != previous_partial) {
                            partial_text_ = merged;
                            tail_decode = true;
                        } else if (run_full_partial_decode()) {
                            partial_text_ = decoded_text;
                        } else {
                            return processed;
                        }
                    } else {
                        return processed;
                    }
                } else {
                    if (run_full_partial_decode()) {
                        partial_text_ = decoded_text;
                    } else {
                        return processed;
                    }
                }

                if (tail_decode) {
                    ++partial_tail_decode_count_;
                } else {
                    ++partial_full_decode_count_;
                }
                partial_processed_audio_size_ = audio_buffer_.size();
                partial_processed_stable_offset_ = stable_samples_offset_;
                processed = true;
            }
        } else {
            partial_text_.clear();
            partial_processed_audio_size_ = 0;
            partial_processed_stable_offset_ = stable_samples_offset_;
        }

        last_error_.clear();
        return processed;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = e.what();
        return false;
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "unknown ASR pipeline failure";
        return false;
    }
}

void AliaAsrPipeline::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    audio_buffer_.clear();
    stable_samples_offset_ = 0;
    partial_processed_audio_size_ = 0;
    partial_processed_stable_offset_ = 0;
    stable_text_.clear();
    partial_text_.clear();
    past_text_.clear();
    last_error_.clear();
    partial_full_decode_count_ = 0;
    partial_tail_decode_count_ = 0;
    partial_throttled_count_ = 0;
    metrics_ = AliaAsrMetrics{};
    prefix_cache_.reset();
    partial_mel_cache_.reset();
    partial_mel_cache_stable_offset_ = 0;
}

void AliaAsrPipeline::get_text(std::string& out_stable, std::string& out_partial) {
    process_pending(true);

    std::lock_guard<std::mutex> lock(mutex_);
    out_stable = stable_text_;
    out_partial = partial_text_;
}

void AliaAsrPipeline::get_partial_text(std::string& out_stable, std::string& out_partial) {
    process_pending(false);

    std::lock_guard<std::mutex> lock(mutex_);
    out_stable = stable_text_;
    out_partial = partial_text_;
}

bool AliaAsrPipeline::ready() const {
    return slot_ &&
           slot_->state() == ModelSlotState::Loaded &&
           slot_->context() &&
           slot_->tokenizer() &&
           slot_->backend() &&
           slot_->audio_encoder();
}

size_t AliaAsrPipeline::buffered_sample_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return audio_buffer_.size();
}

int AliaAsrPipeline::partial_full_decode_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return partial_full_decode_count_;
}

int AliaAsrPipeline::partial_tail_decode_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return partial_tail_decode_count_;
}

int AliaAsrPipeline::partial_throttled_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return partial_throttled_count_;
}

AliaAsrMetrics AliaAsrPipeline::last_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

bool AliaAsrPipeline::transcribe_segment_raw(const std::vector<float>& segment,
                                             const std::string& past_text,
                                             std::string& language_out,
                                             std::string& text_out) {
    language_out.clear();
    text_out.clear();

    if (!ready()) {
        return false;
    }

    static const bool s_profile_asr = aila::env::read_flag("AILA_ASR_PROFILE", false);
    auto total_start = std::chrono::high_resolution_clock::now();
    auto stage_start = total_start;
    AliaAsrMetrics call_metrics;
    call_metrics.transcribe_calls = 1;
    call_metrics.input_audio_ms =
        static_cast<double>(segment.size()) * 1000.0 / 16000.0;
    auto finish_stage = [&](double& target, Context* maybe_context = nullptr) {
        if (!s_profile_asr) {
            return;
        }
        if (maybe_context) {
            maybe_context->synchronize();
        }
        const auto now = std::chrono::high_resolution_clock::now();
        target += std::chrono::duration<double, std::milli>(now - stage_start).count();
        stage_start = now;
    };

    Context* context = slot_->context();
    IModelBackend* backend = slot_->backend();
    Tokenizer* tokenizer = slot_->tokenizer();
    aila::audio::Qwen3ASRAudioEncoder* audio_encoder = slot_->audio_encoder();
    const ModelSpec& spec = slot_->model_spec();
    const Qwen3Config& cfg = spec.qwen3;

    MelSpectrogram mel;
    MelSpectrogramTiming mel_timing;
    std::string prep_error;
    static const bool s_mel_cache = aila::env::read_flag("AILA_ASR_MEL_CACHE", true);
    static const bool s_mel_cache_validate =
        aila::env::read_flag("AILA_ASR_MEL_CACHE_VALIDATE", false);
    bool used_mel_cache = false;
    if (s_mel_cache) {
        if (partial_mel_cache_stable_offset_ != stable_samples_offset_) {
            partial_mel_cache_.reset();
            partial_mel_cache_stable_offset_ = stable_samples_offset_;
        }
        used_mel_cache = partial_mel_cache_.sample_count > 0 &&
                         partial_mel_cache_.sample_count <= segment.size();
    }
    const bool mel_ok = s_mel_cache
        ? compute_mel_spectrogram_cached(segment, mel, partial_mel_cache_,
                                         &prep_error, &mel_timing,
                                         s_mel_cache_validate)
        : compute_mel_spectrogram(segment, mel, &prep_error, &mel_timing);
    if (!mel_ok) {
        last_error_ = "ASR mel spectrogram failed: " + prep_error;
        return false;
    }
    if (used_mel_cache) {
        ++call_metrics.mel_cache_hits;
    }
    call_metrics.mel_cache_reused_frames += mel_timing.reused_frames;
    call_metrics.mel_cache_computed_frames += mel_timing.computed_frames;
    call_metrics.mel_cache_max_abs_diff =
        std::max(call_metrics.mel_cache_max_abs_diff, mel_timing.max_abs_diff);
    call_metrics.mel_stft_ms += mel_timing.stft_ms;
    call_metrics.mel_norm_ms += mel_timing.norm_ms;
    finish_stage(call_metrics.mel_ms);

    std::vector<AsrBf16> mel_bf16(static_cast<size_t>(mel.n_frames) * mel.n_mels);
    for (int frame = 0; frame < mel.n_frames; ++frame) {
        for (int mel_bin = 0; mel_bin < mel.n_mels; ++mel_bin) {
            mel_bf16[static_cast<size_t>(mel_bin) * mel.n_frames + frame] =
                AsrBf16(mel.data[static_cast<size_t>(frame) * mel.n_mels + mel_bin]);
        }
    }

    auto lane_lock = context->lock_execution();
    Tensor mel_device = Tensor::allocate(*context, {1, mel.n_mels, mel.n_frames});
    context->memcpy_h2d(mel_device.data(), mel_bf16.data(), mel_bf16.size() * sizeof(AsrBf16));
    finish_stage(call_metrics.upload_ms, context);

    int audio_len = 0;
    const int output_dim = spec.audio.output_dim;
    const int max_audio_len = ((mel.actual_frames + 99) / 100) * 13 + 32;
    Tensor audio_tmp = Tensor::allocate(*context, {max_audio_len, output_dim});

    std::string audio_error;
    if (!audio_encoder->encode(*context, mel_device, mel.actual_frames,
                               audio_tmp, audio_len, &audio_error)) {
        last_error_ = "ASR audio encoder failed: " + audio_error;
        return false;
    }
    const auto encoder_timing = audio_encoder->last_timing();
    call_metrics.encoder_conv_ms += encoder_timing.conv_ms;
    call_metrics.encoder_transformer_ms += encoder_timing.transformer_ms;
    call_metrics.encoder_proj_ms += encoder_timing.proj_ms;
    finish_stage(call_metrics.encoder_ms, context);

    std::vector<int> prompt_ids;
    auto add_text = [&](const std::string& text) {
        std::vector<int> ids = tokenizer->encode(text);
        prompt_ids.insert(prompt_ids.end(), ids.begin(), ids.end());
    };

    prompt_ids.push_back(cfg.im_start_id);
    add_text("system\nYou are an accurate streaming ASR engine for Alia.");
    prompt_ids.push_back(cfg.im_end_id);
    add_text("\n");

    prompt_ids.push_back(cfg.im_start_id);
    add_text("user\n");
    prompt_ids.push_back(spec.audio_start_token_id);
    for (int i = 0; i < audio_len; ++i) {
        prompt_ids.push_back(spec.audio_token_id);
    }
    prompt_ids.push_back(spec.audio_end_token_id);
    add_text("\n");
    prompt_ids.push_back(cfg.im_end_id);
    add_text("\n");

    prompt_ids.push_back(cfg.im_start_id);
    add_text("assistant\n");
    if (!past_text.empty()) {
        add_text(past_text);
        const int asr_text_id = tokenizer->special_token_id("<asr_text>");
        if (asr_text_id != -1) {
            prompt_ids.push_back(asr_text_id);
        }
    }

    std::vector<int> audio_positions;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (prompt_ids[i] == spec.audio_token_id) {
            audio_positions.push_back(static_cast<int>(i));
        }
    }
    const int audio_prefix_len = audio_positions.empty()
        ? 0
        : audio_positions.back() + 1;

    static const bool s_prefix_reuse =
        aila::env::read_flag("AILA_ASR_PREFIX_REUSE", false);
    int reuse_len = 0;
    int reuse_audio_len = 0;
    if (s_prefix_reuse && prefix_cache_.valid) {
        ++call_metrics.prefix_reuse_attempts;
        const bool prefix_shape_matches =
            prefix_cache_.stable_samples_offset == stable_samples_offset_ &&
            prefix_cache_.audio_len > 0 &&
            audio_len >= prefix_cache_.audio_len &&
            prefix_cache_.prefix_len > 0 &&
            prefix_cache_.prefix_len <= audio_prefix_len &&
            prefix_cache_.prefix_len <= static_cast<int>(prompt_ids.size()) &&
            prefix_cache_.prefix_token_ids.size() ==
                static_cast<size_t>(prefix_cache_.prefix_len);
        const bool prefix_tokens_match = prefix_shape_matches &&
            std::equal(prefix_cache_.prefix_token_ids.begin(),
                       prefix_cache_.prefix_token_ids.end(),
                       prompt_ids.begin());
        if (prefix_tokens_match) {
            reuse_len = prefix_cache_.prefix_len;
            reuse_audio_len = prefix_cache_.audio_len;
        }
    }

    static const bool s_device_embedding_overrides =
        aila::env::read_flag("AILA_ASR_DEVICE_EMBEDDING_OVERRIDES", true);
    auto* asr_bnb4 = s_device_embedding_overrides
        ? dynamic_cast<Qwen3ASRBnb4Backend*>(backend)
        : nullptr;

    bool use_prefix_reuse = false;
    if (reuse_len > 0) {
        use_prefix_reuse = backend->truncate_kv_cache(reuse_len);
        if (!use_prefix_reuse) {
            reuse_len = 0;
            reuse_audio_len = 0;
        }
    }
    if (!use_prefix_reuse) {
        backend->reset();
    }

    std::vector<int> active_audio_positions;
    active_audio_positions.reserve(audio_positions.size());
    for (int pos : audio_positions) {
        if (pos >= reuse_len) {
            active_audio_positions.push_back(pos - reuse_len);
        }
    }
    const int active_audio_len = std::max(0, audio_len - reuse_audio_len);
    Tensor active_audio_view;
    if (asr_bnb4) {
        if (active_audio_len > 0) {
            active_audio_view = Tensor::view(
                *context,
                audio_tmp.data_as<AsrBf16>() +
                    static_cast<size_t>(reuse_audio_len) * output_dim,
                {active_audio_len, output_dim},
                audio_tmp.dtype());
            asr_bnb4->set_embedding_overrides_device(
                active_audio_positions, active_audio_view, active_audio_len, output_dim);
        } else {
            backend->clear_embedding_overrides();
        }
    } else {
        if (active_audio_len > 0) {
            std::vector<AsrBf16> audio_host(static_cast<size_t>(active_audio_len) * output_dim);
            context->memcpy_d2h(
                audio_host.data(),
                audio_tmp.data_as<AsrBf16>() +
                    static_cast<size_t>(reuse_audio_len) * output_dim,
                audio_host.size() * sizeof(AsrBf16));
            finish_stage(call_metrics.readback_ms);
            backend->set_embedding_overrides(active_audio_positions, audio_host, output_dim);
        } else {
            backend->clear_embedding_overrides();
        }
    }

    std::vector<int> pos_t(prompt_ids.size());
    std::vector<int> pos_h(prompt_ids.size());
    std::vector<int> pos_w(prompt_ids.size());
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        pos_t[i] = static_cast<int>(i);
        pos_h[i] = static_cast<int>(i);
        pos_w[i] = static_cast<int>(i);
    }
    backend->set_mrope_positions(*context, pos_t, pos_h, pos_w, 0);
    finish_stage(call_metrics.prompt_ms, context);

    const int prompt_forward_len = static_cast<int>(prompt_ids.size()) - reuse_len;
    DeviceAllocation prompt_device(*context,
                                   static_cast<size_t>(prompt_forward_len) * sizeof(int));
    context->memcpy_h2d(prompt_device.as<int>(), prompt_ids.data() + reuse_len,
                        static_cast<size_t>(prompt_forward_len) * sizeof(int));

    Tensor* logits = &backend->forward(*context, prompt_device.as<int>(),
                                       prompt_forward_len);
    backend->clear_mrope_positions();
    if (use_prefix_reuse) {
        ++call_metrics.prefix_reuse_hits;
        call_metrics.prefix_reused_tokens += reuse_len;
        call_metrics.prefix_appended_tokens += prompt_forward_len;
    }
    finish_stage(call_metrics.prefill_ms, context);

    DeviceAllocation one_token_device(*context, sizeof(int));
    DeviceAllocation argmax_device(*context, sizeof(int));
    std::vector<int> generated_ids;
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 256;
    gen_config.do_sample = false;
    generated_ids.reserve(static_cast<size_t>(gen_config.max_new_tokens));

    for (int step = 0; step < gen_config.max_new_tokens; ++step) {
        int next_token = 0;
        ops::argmax(*context, *logits, cfg.vocab_size, argmax_device.as<int>());
        context->memcpy_d2h(&next_token, argmax_device.as<int>(), sizeof(int));

        if (next_token == cfg.eos_token_id || next_token == cfg.im_end_id) {
            break;
        }

        generated_ids.push_back(next_token);
        context->memcpy_h2d(one_token_device.as<int>(), &next_token, sizeof(int));
        logits = &backend->forward(*context, one_token_device.as<int>(), 1);
    }
    finish_stage(call_metrics.decode_ms, context);

    backend->clear_embedding_overrides();
    if (s_prefix_reuse && audio_prefix_len > 0) {
        if (backend->truncate_kv_cache(audio_prefix_len)) {
            prefix_cache_.valid = true;
            prefix_cache_.stable_samples_offset = stable_samples_offset_;
            prefix_cache_.audio_len = audio_len;
            prefix_cache_.prefix_len = audio_prefix_len;
            prefix_cache_.prefix_token_ids.assign(prompt_ids.begin(),
                                                  prompt_ids.begin() + audio_prefix_len);
        } else {
            prefix_cache_.reset();
        }
    } else {
        prefix_cache_.reset();
    }

    std::string raw = tokenizer->decode(generated_ids);
    parse_asr_output(raw, "", language_out, text_out);
    call_metrics.generated_tokens = static_cast<int>(generated_ids.size());
    call_metrics.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - total_start).count();
    metrics_.transcribe_calls += call_metrics.transcribe_calls;
    metrics_.generated_tokens += call_metrics.generated_tokens;
    metrics_.prefix_reuse_attempts += call_metrics.prefix_reuse_attempts;
    metrics_.prefix_reuse_hits += call_metrics.prefix_reuse_hits;
    metrics_.prefix_reused_tokens += call_metrics.prefix_reused_tokens;
    metrics_.prefix_appended_tokens += call_metrics.prefix_appended_tokens;
    metrics_.mel_cache_hits += call_metrics.mel_cache_hits;
    metrics_.mel_cache_reused_frames += call_metrics.mel_cache_reused_frames;
    metrics_.mel_cache_computed_frames += call_metrics.mel_cache_computed_frames;
    metrics_.input_audio_ms += call_metrics.input_audio_ms;
    metrics_.mel_ms += call_metrics.mel_ms;
    metrics_.mel_stft_ms += call_metrics.mel_stft_ms;
    metrics_.mel_norm_ms += call_metrics.mel_norm_ms;
    metrics_.mel_cache_max_abs_diff =
        std::max(metrics_.mel_cache_max_abs_diff,
                 call_metrics.mel_cache_max_abs_diff);
    metrics_.upload_ms += call_metrics.upload_ms;
    metrics_.encoder_ms += call_metrics.encoder_ms;
    metrics_.encoder_conv_ms += call_metrics.encoder_conv_ms;
    metrics_.encoder_transformer_ms += call_metrics.encoder_transformer_ms;
    metrics_.encoder_proj_ms += call_metrics.encoder_proj_ms;
    metrics_.readback_ms += call_metrics.readback_ms;
    metrics_.prompt_ms += call_metrics.prompt_ms;
    metrics_.prefill_ms += call_metrics.prefill_ms;
    metrics_.decode_ms += call_metrics.decode_ms;
    metrics_.total_ms += call_metrics.total_ms;
    return true;
}

int AliaAsrPipeline::find_split_point(const float* samples,
                                      int sample_count,
                                      int target_sample,
                                      float search_sec) {
    int search_half = static_cast<int>(search_sec * 16000.0f);
    int lo = std::max(0, target_sample - search_half);
    int hi = std::min(sample_count, target_sample + search_half);
    int win_samples = 1600;
    float best_energy = 1e30f;
    int best_center = target_sample;

    for (int pos = lo; pos + win_samples <= hi; pos += win_samples / 2) {
        int end = std::min(sample_count, pos + win_samples);
        float energy = 0.0f;
        for (int i = pos; i < end; ++i) {
            energy += samples[i] * samples[i];
        }
        energy /= static_cast<float>(end - pos);
        if (energy < best_energy) {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
    }

    return best_center;
}

bool AliaAsrPipeline::should_insert_boundary_space(char prev_ch, char next_ch) {
    return should_insert_boundary_space_for_text(prev_ch, next_ch);
}

void parse_asr_output(const std::string& raw,
                      const std::string& forced_language,
                      std::string& language_out,
                      std::string& text_out) {
    language_out.clear();
    text_out.clear();

    std::string value = trim(raw);
    if (value.empty()) {
        return;
    }

    if (!forced_language.empty()) {
        language_out = normalize_language_name(forced_language);
        text_out = value;
        return;
    }

    constexpr const char* kAsrTextTag = "<asr_text>";
    const size_t tag_pos = value.find(kAsrTextTag);
    if (tag_pos == std::string::npos) {
        text_out = value;
        return;
    }

    const std::string meta_part = value.substr(0, tag_pos);
    const std::string text_part = value.substr(tag_pos + std::char_traits<char>::length(kAsrTextTag));

    const std::string meta_lower = to_lowercase(meta_part);
    if (meta_lower.find("language none") != std::string::npos) {
        text_out = trim(text_part);
        return;
    }

    const std::string prefix = "language ";
    size_t line_start = 0;
    while (line_start < meta_part.size()) {
        size_t line_end = meta_part.find('\n', line_start);
        if (line_end == std::string::npos) {
            line_end = meta_part.size();
        }

        const std::string line = trim(meta_part.substr(line_start, line_end - line_start));
        const std::string line_lower = to_lowercase(line);
        if (line_lower.rfind(prefix, 0) == 0) {
            language_out = normalize_language_name(line.substr(prefix.size()));
            break;
        }

        line_start = line_end + 1;
    }

    text_out = trim(text_part);
}

}  // namespace aila::alia
