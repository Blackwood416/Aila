#include "worker/CApiWorkerEngine.hpp"

#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

namespace aila::worker {

CApiWorkerEngine::CApiWorkerEngine() : engine_(aila_engine_create()) {
    if (engine_ == nullptr) {
        throw std::runtime_error("aila_engine_create failed");
    }
}

CApiWorkerEngine::~CApiWorkerEngine() {
    for (auto& [id, stream] : asr_streams_) {
        (void)id;
        aila_transcribe_stream_destroy(stream);
    }
    asr_streams_.clear();
    aila_engine_destroy(engine_);
    engine_ = nullptr;
}

int CApiWorkerEngine::init(const std::string& model, int max_seq_len) {
    return aila_engine_init(engine_, model.c_str(), max_seq_len);
}

void CApiWorkerEngine::reset_context() { aila_engine_reset_context(engine_); }

int CApiWorkerEngine::context_length() const { return aila_engine_context_length(engine_); }

int CApiWorkerEngine::last_error_code() const { return aila_last_error_code(engine_); }

std::string CApiWorkerEngine::last_error_message() const {
    const char* message = aila_last_error_message(engine_);
    return message == nullptr ? std::string{} : std::string(message);
}

void CApiWorkerEngine::set_log_level(int level) { aila_set_log_level(level); }

bool CApiWorkerEngine::generate_text(
    const aila::worker::TextGenerationRequest& request,
    std::string& output) {
    char* result = nullptr;
    switch (request.method) {
        case aila::worker::TextGenerationMethod::Generate:
            result = aila_generate(
                engine_,
                request.input.c_str(),
                request.has_config ? &request.config : nullptr);
            break;
        case aila::worker::TextGenerationMethod::GenerateMessages:
            result = aila_generate_messages(
                engine_,
                request.input.c_str(),
                request.has_config ? &request.config : nullptr);
            break;
        case aila::worker::TextGenerationMethod::GenerateChatJson:
            result = aila_generate_chat_json(
                engine_,
                request.input.c_str(),
                request.has_config ? &request.config : nullptr);
            break;
        case aila::worker::TextGenerationMethod::GenerateChatJsonEx:
            result = aila_generate_chat_json_ex(
                engine_,
                request.input.c_str(),
                request.has_v2_config ? &request.config_v2 : nullptr);
            break;
        case aila::worker::TextGenerationMethod::GenerateStream:
        case aila::worker::TextGenerationMethod::GenerateMessagesStream:
        case aila::worker::TextGenerationMethod::GenerateChatJsonStreamEx:
            return false;
    }
    if (!result) {
        output.clear();
        return false;
    }
    try {
        output.assign(result);
    } catch (...) {
        aila_free_string(result);
        throw;
    }
    aila_free_string(result);
    return true;
}

int CApiWorkerEngine::generate_stream(
    const aila::worker::TextGenerationRequest& request,
    const aila::worker::TokenStreamCallback& token_callback,
    const aila::worker::StructuredStreamCallback& structured_callback) {
    struct TokenContext {
        const aila::worker::TokenStreamCallback* callback;
    } token_context{&token_callback};
    struct StructuredContext {
        const aila::worker::StructuredStreamCallback* callback;
    } structured_context{&structured_callback};
    const auto token_adapter = [](const char* text, void* opaque) -> int {
        auto* context = static_cast<TokenContext*>(opaque);
        if (text == nullptr || !(*context->callback)(text)) {
            // The legacy in-process token API only suppresses later callbacks after
            // abort. Unwinding here also stops the underlying generation promptly;
            // dispatch_stream converts this caught C-API error to status 1 when the
            // control pipe cancellation flag is set.
            throw std::runtime_error("stream generation cancelled");
        }
        return 0;
    };
    const auto structured_adapter = [](const AilaChatStreamEvent* event, void* opaque) -> int {
        auto* context = static_cast<StructuredContext*>(opaque);
        return event != nullptr && (*context->callback)(*event) ? 0 : 1;
    };
    switch (request.method) {
        case aila::worker::TextGenerationMethod::GenerateStream:
            return aila_generate_stream(
                engine_, request.input.c_str(),
                request.has_config ? &request.config : nullptr,
                token_adapter, &token_context);
        case aila::worker::TextGenerationMethod::GenerateMessagesStream:
            return aila_generate_messages_stream(
                engine_, request.input.c_str(),
                request.has_config ? &request.config : nullptr,
                token_adapter, &token_context);
        case aila::worker::TextGenerationMethod::GenerateChatJsonStreamEx:
            return aila_generate_chat_json_stream_ex(
                engine_, request.input.c_str(),
                request.has_v2_config ? &request.config_v2 : nullptr,
                structured_adapter, &structured_context);
        default:
            return -1;
    }
}

bool CApiWorkerEngine::transcribe(
    const aila::worker::AsrRequest& request,
    const std::function<void(std::string_view)>& token_callback,
    std::string& transcript,
    std::string& language) {
    struct Context {
        const std::function<void(std::string_view)>* callback;
    } context{&token_callback};
    const auto adapter = [](const char* text, void* opaque) -> int {
        auto* context = static_cast<Context*>(opaque);
        if (text != nullptr) (*context->callback)(text);
        return 0; // The public offline-ASR callback return is intentionally ignored.
    };
    char* language_result = nullptr;
    char* text = aila_transcribe(
        engine_, request.wav_path.c_str(), request.has_config ? &request.config : nullptr,
        request.has_forced_language ? request.forced_language.c_str() : nullptr,
        request.has_system_prompt ? request.system_prompt.c_str() : nullptr,
        request.segment_sec, request.past_text_conditioning,
        token_callback ? adapter : nullptr, &context, &language_result);
    if (!text) {
        if (language_result) aila_free_string(language_result);
        transcript.clear();
        language.clear();
        return false;
    }
    try {
        transcript.assign(text);
        language = language_result ? std::string(language_result) : std::string{};
    } catch (...) {
        aila_free_string(text);
        if (language_result) aila_free_string(language_result);
        throw;
    }
    aila_free_string(text);
    if (language_result) aila_free_string(language_result);
    return true;
}

bool CApiWorkerEngine::transcribe_stream_create(
    uint64_t id, const aila::worker::AsrStreamConfig& config) {
    if (asr_streams_.find(id) != asr_streams_.end()) return false;
    AilaTranscribeStream* stream = aila_transcribe_stream_create(
        engine_, config.has_config ? &config.config : nullptr,
        config.has_forced_language ? config.forced_language.c_str() : nullptr,
        config.has_system_prompt ? config.system_prompt.c_str() : nullptr);
    if (!stream) return false;
    asr_streams_.emplace(id, stream);
    return true;
}

bool CApiWorkerEngine::transcribe_stream_feed(uint64_t id, const float* samples, size_t count) {
    const auto found = asr_streams_.find(id);
    if (found == asr_streams_.end() || count > (std::numeric_limits<int>::max)()) return false;
    return aila_transcribe_stream_feed(
               found->second, samples, static_cast<int>(count)) == AILA_OK;
}

bool CApiWorkerEngine::transcribe_stream_get_text(
    uint64_t id, std::string& stable, std::string& partial) {
    const auto found = asr_streams_.find(id);
    if (found == asr_streams_.end()) return false;
    char* stable_result = nullptr;
    char* partial_result = nullptr;
    const int status = aila_transcribe_stream_get_text(
        found->second, &stable_result, &partial_result);
    if (status != AILA_OK) {
        if (stable_result) aila_free_string(stable_result);
        if (partial_result) aila_free_string(partial_result);
        return false;
    }
    try {
        stable = stable_result ? std::string(stable_result) : std::string{};
        partial = partial_result ? std::string(partial_result) : std::string{};
    } catch (...) {
        if (stable_result) aila_free_string(stable_result);
        if (partial_result) aila_free_string(partial_result);
        throw;
    }
    if (stable_result) aila_free_string(stable_result);
    if (partial_result) aila_free_string(partial_result);
    return true;
}

bool CApiWorkerEngine::transcribe_stream_destroy(uint64_t id) noexcept {
    const auto found = asr_streams_.find(id);
    if (found == asr_streams_.end()) return false;
    aila_transcribe_stream_destroy(found->second);
    asr_streams_.erase(found);
    return true;
}

bool CApiWorkerEngine::process_audio(
    const aila::worker::AudioRequest& request,
    std::vector<float>& output) {
    float* values = nullptr;
    int count = 0;
    int status = AILA_ERR_RUNTIME;
    switch (request.method) {
        case aila::worker::AudioMethod::SynthesizeWav:
            status = aila_synthesize_wav(
                engine_, reinterpret_cast<const int*>(request.text_tokens.data()),
                static_cast<int>(request.text_tokens.size()),
                request.embedding.empty() ? nullptr : request.embedding.data(),
                static_cast<int>(request.embedding.size()),
                request.has_config ? &request.config : nullptr, &values, &count);
            break;
        case aila::worker::AudioMethod::SynthesizeTextToWav:
            status = aila_synthesize_text_to_wav(
                engine_, request.text.c_str(),
                request.embedding.empty() ? nullptr : request.embedding.data(),
                static_cast<int>(request.embedding.size()),
                request.has_config ? &request.config : nullptr, &values, &count);
            break;
        case aila::worker::AudioMethod::SynthesizeFile: {
            output.clear();
            AilaTTSOptions options = aila_tts_options_default();
            options.reference_text = request.has_reference_text ? request.reference_text.c_str() : nullptr;
            options.voice_clone_mode = static_cast<AilaVoiceCloneMode>(request.voice_clone_mode);
            return aila_synthesize_ex(
                engine_, request.text.c_str(),
                request.has_reference_audio_path ? request.reference_audio_path.c_str() : nullptr,
                request.has_speaker_name ? request.speaker_name.c_str() : nullptr,
                request.has_instruct_text ? request.instruct_text.c_str() : nullptr,
                request.has_language ? request.language.c_str() : nullptr,
                request.has_config ? &request.config : nullptr,
                &options,
                request.output_wav_path.c_str()) == AILA_OK;
        }
        case aila::worker::AudioMethod::DecodeMimi:
            status = aila_decode_mimi_vocoder(
                engine_, request.codes.data(), request.frame_count, &values, &count);
            break;
        case aila::worker::AudioMethod::ExtractEmbedding:
            status = aila_extract_speaker_embedding(
                engine_, request.audio_path.c_str(), &values, &count);
            break;
        case aila::worker::AudioMethod::SynthesizeStream:
            return false;
    }
    if (status != AILA_OK || count < 0 || (count > 0 && values == nullptr)) {
        if (values) aila_free_samples(values);
        output.clear();
        return false;
    }
    try {
        if (count == 0) output.clear();
        else output.assign(values, values + count);
    } catch (...) {
        if (values) aila_free_samples(values);
        throw;
    }
    if (values) aila_free_samples(values);
    return true;
}

int CApiWorkerEngine::synthesize_stream(
    const aila::worker::AudioRequest& request,
    const aila::worker::AudioStreamCallback& callback) {
    struct Context {
        const aila::worker::AudioStreamCallback* callback;
        std::atomic_bool accepted{true};
    } context{&callback};
    const auto adapter = [](const float* samples, int count, void* opaque) {
        auto* context = static_cast<Context*>(opaque);
        if (!context->accepted.load(std::memory_order_acquire)) return;
        bool accepted = false;
        try {
            accepted = samples != nullptr && count > 0 &&
                (*context->callback)(samples, static_cast<size_t>(count));
        } catch (...) {
            accepted = false;
        }
        context->accepted.store(accepted, std::memory_order_release);
    };
    AilaTTSOptions options = aila_tts_options_default();
    options.reference_text = request.has_reference_text ? request.reference_text.c_str() : nullptr;
    options.voice_clone_mode = static_cast<AilaVoiceCloneMode>(request.voice_clone_mode);
    AilaTTSStream* stream = aila_synthesize_stream_ex(
        engine_, request.text.c_str(),
        request.has_reference_audio_path ? request.reference_audio_path.c_str() : nullptr,
        request.has_speaker_name ? request.speaker_name.c_str() : nullptr,
        request.has_instruct_text ? request.instruct_text.c_str() : nullptr,
        request.has_language ? request.language.c_str() : nullptr,
        request.has_config ? &request.config : nullptr,
        &options,
        adapter, &context);
    if (!stream) return -1;
    const int status = aila_stream_wait(stream);
    aila_stream_destroy(stream);
    if (status != AILA_OK) return -1;
    return context.accepted.load(std::memory_order_acquire) ? 0 : 1;
}

bool CApiWorkerEngine::align(
    const aila::worker::AlignmentRequest& request,
    std::vector<aila::worker::AlignedWordResult>& output) {
    AilaAlignedWord* words = nullptr;
    int count = 0;
    int status = AILA_ERR_RUNTIME;
    if (request.method == aila::worker::AlignmentMethod::Text) {
        status = aila_align(
            engine_, request.samples.data(), static_cast<int>(request.samples.size()),
            request.sample_rate, request.text.c_str(), request.language.c_str(),
            &words, &count);
    } else {
        std::vector<const char*> pointers;
        pointers.reserve(request.words.size());
        for (const std::string& word : request.words) pointers.push_back(word.c_str());
        status = aila_align_words(
            engine_, request.samples.data(), static_cast<int>(request.samples.size()),
            request.sample_rate, pointers.data(), static_cast<int>(pointers.size()),
            &words, &count);
    }
    if (status != AILA_OK || count < 0 || (count > 0 && words == nullptr)) {
        if (words) aila_free_aligned_words(words, count > 0 ? count : 0);
        output.clear();
        return false;
    }
    try {
        output.clear();
        output.reserve(static_cast<size_t>(count));
        for (int index = 0; index < count; ++index) {
            if (!words[index].text) {
                aila_free_aligned_words(words, count);
                output.clear();
                return false;
            }
            output.push_back({words[index].text, words[index].start_ms, words[index].end_ms});
        }
    } catch (...) {
        if (words) aila_free_aligned_words(words, count);
        throw;
    }
    if (words) aila_free_aligned_words(words, count);
    return true;
}

bool CApiWorkerEngine::detect(const aila::worker::DetectionRequest& request,
            std::vector<aila::worker::DetectionResult>& output) {
    AilaDetection* detections = nullptr;
    int count = 0;
    int status = AILA_ERR_INVALID_ARGUMENT;
    if (request.method == aila::worker::DetectionMethod::File) {
        status = aila_detect_file(engine_, request.path.c_str(), &request.config,
                                  &detections, &count);
    } else if (request.method == aila::worker::DetectionMethod::Encoded) {
        status = aila_detect_encoded(engine_, request.bytes.data(), request.bytes.size(),
                                     &request.config, &detections, &count);
    } else {
        status = aila_detect_pixels(engine_, request.bytes.data(), request.bytes.size(),
                                    request.width, request.height, request.row_stride,
                                    request.pixel_format, &request.config,
                                    &detections, &count);
    }
    if (status != AILA_OK) {
        if (detections) aila_free_detections(detections, count);
        return false;
    }
    try {
        output.clear();
        output.reserve(static_cast<size_t>(count));
        for (int index = 0; index < count; ++index) {
            aila::worker::DetectionResult value;
            value.x1 = detections[index].x1; value.y1 = detections[index].y1;
            value.x2 = detections[index].x2; value.y2 = detections[index].y2;
            value.confidence = detections[index].confidence;
            value.class_id = detections[index].class_id;
            value.class_name = detections[index].class_name ? detections[index].class_name : "";
            output.push_back(std::move(value));
        }
    } catch (...) {
        aila_free_detections(detections, count);
        throw;
    }
    aila_free_detections(detections, count);
    return true;
}

} // namespace aila::worker
