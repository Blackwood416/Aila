#include "aila_api.h"

#include "proxy/ProxyEngine.hpp"
#include "proxy/ProxyLogging.hpp"
#include "simdjson.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

struct AilaEngine {
    AilaEngine() : proxy(std::make_shared<aila::proxy::ProxyEngine>()) {}
    ~AilaEngine() {
        if (proxy) proxy->prepare_destroy();
    }
    std::shared_ptr<aila::proxy::ProxyEngine> proxy;
};

struct AilaTranscribeStream {
    std::shared_ptr<aila::proxy::ProxyEngine> owner;
    uint64_t worker_session = 0;
    uint64_t remote_id = 0;
};

struct AilaTTSStream {
    std::shared_ptr<aila::proxy::ProxyEngine> owner;
    AilaAudioCallback callback = nullptr;
    void* user_data = nullptr;
    std::atomic_bool cancel_requested{false};
    std::mutex mutex;
    std::condition_variable condition;
    std::thread worker;
    bool started = false;
    bool done = false;
    bool joining = false;
    bool joined = false;
    bool destroy_requested = false;
    int status = AILA_ERR_RUNTIME;
};

namespace {

constexpr const char* kVersion = "0.2.0";
constexpr const char* kEmpty = "";

std::mutex engine_registry_mutex;
std::vector<std::weak_ptr<aila::proxy::ProxyEngine>> engine_registry;

void register_engine(const std::shared_ptr<aila::proxy::ProxyEngine>& engine) {
    std::lock_guard<std::mutex> lock(engine_registry_mutex);
    engine_registry.erase(
        std::remove_if(
            engine_registry.begin(), engine_registry.end(),
            [](const auto& entry) { return entry.expired(); }),
        engine_registry.end());
    engine_registry.emplace_back(engine);
}

std::vector<std::shared_ptr<aila::proxy::ProxyEngine>> live_engines() {
    std::vector<std::shared_ptr<aila::proxy::ProxyEngine>> result;
    std::lock_guard<std::mutex> lock(engine_registry_mutex);
    auto iterator = engine_registry.begin();
    while (iterator != engine_registry.end()) {
        if (auto engine = iterator->lock()) {
            result.push_back(std::move(engine));
            ++iterator;
        } else {
            iterator = engine_registry.erase(iterator);
        }
    }
    return result;
}

void record_boundary_failure(AilaEngine* engine, const char* operation) noexcept {
    if (!engine) {
        return;
    }
    try {
        engine->proxy->record_runtime_error(
            std::string(operation) + " failed at the C ABI boundary");
    } catch (...) {
    }
}

char* allocate_result(const std::string& value) {
    if (value.size() == (std::numeric_limits<size_t>::max)()) {
        return nullptr;
    }
    char* result = static_cast<char*>(std::malloc(value.size() + 1));
    if (!result) {
        return nullptr;
    }
    if (!value.empty()) {
        std::memcpy(result, value.data(), value.size());
    }
    result[value.size()] = '\0';
    return result;
}

int allocate_float_result(
    const std::vector<float>& values, float** out_values, int* out_count) {
    if (values.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        return AILA_ERR_RUNTIME;
    }
    if (values.empty()) {
        *out_values = nullptr;
        *out_count = 0;
        return AILA_OK;
    }
    if (values.size() > (std::numeric_limits<size_t>::max)() / sizeof(float)) {
        return AILA_ERR_RUNTIME;
    }
    auto* allocated = static_cast<float*>(std::malloc(values.size() * sizeof(float)));
    if (!allocated) return AILA_ERR_RUNTIME;
    std::memcpy(allocated, values.data(), values.size() * sizeof(float));
    *out_values = allocated;
    *out_count = static_cast<int>(values.size());
    return AILA_OK;
}

int allocate_alignment_result(
    const std::vector<aila::proxy::AlignmentWord>& values,
    AilaAlignedWord** out_words,
    int* out_count) {
    if (values.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        return AILA_ERR_RUNTIME;
    }
    if (values.empty()) {
        *out_words = nullptr;
        *out_count = 0;
        return AILA_OK;
    }
    if (values.size() > (std::numeric_limits<size_t>::max)() / sizeof(AilaAlignedWord)) {
        return AILA_ERR_RUNTIME;
    }
    auto* allocated = static_cast<AilaAlignedWord*>(
        std::calloc(values.size(), sizeof(AilaAlignedWord)));
    if (!allocated) return AILA_ERR_RUNTIME;
    for (size_t index = 0; index < values.size(); ++index) {
        char* text = allocate_result(values[index].text);
        if (!text) {
            for (size_t prior = 0; prior < index; ++prior) {
                std::free(const_cast<char*>(allocated[prior].text));
            }
            std::free(allocated);
            return AILA_ERR_RUNTIME;
        }
        allocated[index].text = text;
        allocated[index].start_ms = values[index].start_ms;
        allocated[index].end_ms = values[index].end_ms;
    }
    *out_words = allocated;
    *out_count = static_cast<int>(values.size());
    return AILA_OK;
}

int allocate_detection_result(
    const std::vector<aila::proxy::DetectionResult>& values,
    AilaDetection** out_detections, int* out_count) {
    if (values.empty()) return AILA_OK;
    auto* allocated = static_cast<AilaDetection*>(
        std::calloc(values.size(), sizeof(AilaDetection)));
    if (!allocated) return AILA_ERR_RUNTIME;
    for (size_t index = 0; index < values.size(); ++index) {
        char* name = allocate_result(values[index].class_name);
        if (!name) {
            for (size_t prior = 0; prior < index; ++prior) {
                std::free(const_cast<char*>(allocated[prior].class_name));
            }
            std::free(allocated);
            return AILA_ERR_RUNTIME;
        }
        allocated[index].struct_size = sizeof(AilaDetection);
        allocated[index].x1 = values[index].x1; allocated[index].y1 = values[index].y1;
        allocated[index].x2 = values[index].x2; allocated[index].y2 = values[index].y2;
        allocated[index].confidence = values[index].confidence;
        allocated[index].class_id = values[index].class_id;
        allocated[index].class_name = name;
    }
    *out_detections = allocated;
    *out_count = static_cast<int>(values.size());
    return AILA_OK;
}

char* generate_legacy(
    AilaEngine* engine,
    const char* input,
    const AilaGenConfig* config,
    const char* method,
    const char* operation) noexcept {
    if (!engine) {
        return nullptr;
    }
    if (!input) {
        try {
            engine->proxy->record_invalid_argument("generation input must not be NULL");
        } catch (...) {
        }
        return nullptr;
    }
    try {
        std::string output;
        if (!engine->proxy->generate_text(method, input, config, output)) {
            return nullptr;
        }
        char* result = allocate_result(output);
        if (!result) {
            engine->proxy->record_runtime_error("could not allocate generation result");
        }
        return result;
    } catch (...) {
        record_boundary_failure(engine, operation);
        return nullptr;
    }
}

char* generate_v2(
    AilaEngine* engine,
    const char* input,
    const AilaGenConfigV2* config,
    const char* method,
    const char* operation) noexcept {
    if (!engine) {
        return nullptr;
    }
    if (!input) {
        try {
            engine->proxy->record_invalid_argument("generation input must not be NULL");
        } catch (...) {
        }
        return nullptr;
    }
    try {
        std::string output;
        if (!engine->proxy->generate_text_v2(method, input, config, output)) {
            return nullptr;
        }
        char* result = allocate_result(output);
        if (!result) {
            engine->proxy->record_runtime_error("could not allocate generation result");
        }
        return result;
    } catch (...) {
        record_boundary_failure(engine, operation);
        return nullptr;
    }
}

} // namespace

AILA_API const char* aila_version(void) {
    return kVersion;
}

AILA_API AilaEngine* aila_engine_create(void) {
    try {
        std::unique_ptr<AilaEngine> engine(new AilaEngine());
        register_engine(engine->proxy);
        return engine.release();
    } catch (...) {
        return nullptr;
    }
}

AILA_API int aila_engine_init(AilaEngine* engine, const char* model_dir, int max_seq_len) {
    if (!engine) {
        return -1;
    }
    if (!model_dir) {
        try {
            engine->proxy->record_invalid_argument("model_dir must not be NULL");
        } catch (...) {
        }
        return -1;
    }
    try {
        return engine->proxy->init(model_dir, max_seq_len) ? 0 : -1;
    } catch (...) {
        record_boundary_failure(engine, "aila_engine_init");
        return -1;
    }
}

AILA_API void aila_engine_destroy(AilaEngine* engine) {
    try {
        delete engine;
    } catch (...) {
    }
}

AILA_API AilaGenConfig aila_default_gen_config(void) {
    AilaGenConfig config{};
    config.max_new_tokens = 512;
    config.temperature = 0.6f;
    config.top_k = 20;
    config.top_p = 0.95f;
    config.repetition_penalty = 1.0f;
    config.presence_penalty = 0.0f;
    config.frequency_penalty = 0.0f;
    config.do_sample = 1;
    config.decode_chunk_size = 12;
    config.stream_chunk_size = 4;
    return config;
}

AILA_API AilaGenConfigV2 aila_default_gen_config_v2(void) {
    const AilaGenConfig base = aila_default_gen_config();
    AilaGenConfigV2 config{};
    config.struct_size = sizeof(AilaGenConfigV2);
    config.max_new_tokens = base.max_new_tokens;
    config.temperature = base.temperature;
    config.top_k = base.top_k;
    config.top_p = base.top_p;
    config.repetition_penalty = base.repetition_penalty;
    config.presence_penalty = base.presence_penalty;
    config.frequency_penalty = base.frequency_penalty;
    config.do_sample = base.do_sample;
    config.decode_chunk_size = base.decode_chunk_size;
    config.stream_chunk_size = base.stream_chunk_size;
    config.thinking_budget_tokens = -1;
    config.sampling_seed = 42;
    config.use_fixed_seed = 0;
    return config;
}

AILA_API AilaDetectionConfig aila_default_detection_config(void) {
    AilaDetectionConfig config{};
    config.struct_size = sizeof(AilaDetectionConfig);
    config.confidence_threshold = 0.25f;
    config.max_detections = 300;
    return config;
}

AILA_API int aila_detect_file(AilaEngine* engine, const char* image_path,
                              const AilaDetectionConfig* config,
                              AilaDetection** out_detections, int* out_count) {
    if (out_detections) *out_detections = nullptr;
    if (out_count) *out_count = 0;
    if (!engine || !image_path || !out_detections || !out_count) return AILA_ERR_INVALID_ARGUMENT;
    try {
        std::vector<aila::proxy::DetectionResult> values;
        const int status = engine->proxy->detect_file(image_path, config, values);
        return status == AILA_OK ? allocate_detection_result(values, out_detections, out_count) : status;
    } catch (...) { record_boundary_failure(engine, "aila_detect_file"); return AILA_ERR_RUNTIME; }
}

AILA_API int aila_detect_encoded(AilaEngine* engine, const void* bytes, size_t size,
                                 const AilaDetectionConfig* config,
                                 AilaDetection** out_detections, int* out_count) {
    if (out_detections) *out_detections = nullptr;
    if (out_count) *out_count = 0;
    if (!engine || !bytes || !size || !out_detections || !out_count) return AILA_ERR_INVALID_ARGUMENT;
    try {
        std::vector<aila::proxy::DetectionResult> values;
        const int status = engine->proxy->detect_encoded(bytes, size, config, values);
        return status == AILA_OK ? allocate_detection_result(values, out_detections, out_count) : status;
    } catch (...) { record_boundary_failure(engine, "aila_detect_encoded"); return AILA_ERR_RUNTIME; }
}

AILA_API int aila_detect_pixels(AilaEngine* engine, const void* pixels, size_t size,
                                int width, int height, int row_stride, AilaPixelFormat format,
                                const AilaDetectionConfig* config,
                                AilaDetection** out_detections, int* out_count) {
    if (out_detections) *out_detections = nullptr;
    if (out_count) *out_count = 0;
    if (!engine || !pixels || !size || !out_detections || !out_count) return AILA_ERR_INVALID_ARGUMENT;
    try {
        std::vector<aila::proxy::DetectionResult> values;
        const int status = engine->proxy->detect_pixels(
            pixels, size, width, height, row_stride, format, config, values);
        return status == AILA_OK ? allocate_detection_result(values, out_detections, out_count) : status;
    } catch (...) { record_boundary_failure(engine, "aila_detect_pixels"); return AILA_ERR_RUNTIME; }
}

AILA_API void aila_free_detections(AilaDetection* detections, int count) {
    if (!detections) return;
    for (int index = 0; index < count; ++index) {
        std::free(const_cast<char*>(detections[index].class_name));
    }
    std::free(detections);
}

AILA_API char* aila_generate(
    AilaEngine* engine,
    const char* prompt,
    const AilaGenConfig* config) {
    return generate_legacy(engine, prompt, config, "generate", "aila_generate");
}

AILA_API char* aila_generate_messages(
    AilaEngine* engine,
    const char* messages_json,
    const AilaGenConfig* config) {
    return generate_legacy(
        engine,
        messages_json,
        config,
        "generate.messages",
        "aila_generate_messages");
}

AILA_API char* aila_generate_chat_json(
    AilaEngine* engine,
    const char* chat_request_json,
    const AilaGenConfig* config) {
    return generate_legacy(
        engine,
        chat_request_json,
        config,
        "generate.chat_json",
        "aila_generate_chat_json");
}

AILA_API char* aila_generate_chat_json_ex(
    AilaEngine* engine,
    const char* chat_request_json,
    const AilaGenConfigV2* config) {
    return generate_v2(
        engine,
        chat_request_json,
        config,
        "generate.chat_json_ex",
        "aila_generate_chat_json_ex");
}

AILA_API int aila_generate_stream(
    AilaEngine* engine,
    const char* prompt,
    const AilaGenConfig* config,
    AilaTokenCallback callback,
    void* user_data) {
    if (!engine) return -1;
    if (!prompt || !callback) {
        try { engine->proxy->record_invalid_argument("stream input and callback must not be NULL"); }
        catch (...) {}
        return -1;
    }
    try {
        return engine->proxy->generate_stream(
            "generate.stream", prompt, config, callback, user_data);
    } catch (...) {
        record_boundary_failure(engine, "aila_generate_stream");
        return -1;
    }
}

AILA_API int aila_generate_messages_stream(
    AilaEngine* engine,
    const char* messages_json,
    const AilaGenConfig* config,
    AilaTokenCallback callback,
    void* user_data) {
    if (!engine) return -1;
    if (!messages_json || !callback) {
        try { engine->proxy->record_invalid_argument("stream input and callback must not be NULL"); }
        catch (...) {}
        return -1;
    }
    try {
        return engine->proxy->generate_stream(
            "generate.messages_stream", messages_json, config, callback, user_data);
    } catch (...) {
        record_boundary_failure(engine, "aila_generate_messages_stream");
        return -1;
    }
}

AILA_API int aila_generate_chat_json_stream_ex(
    AilaEngine* engine,
    const char* chat_request_json,
    const AilaGenConfigV2* config,
    AilaChatStreamCallback callback,
    void* user_data) {
    if (!engine) return -1;
    if (!chat_request_json || !callback) {
        try { engine->proxy->record_invalid_argument("stream input and callback must not be NULL"); }
        catch (...) {}
        return -1;
    }
    try {
        return engine->proxy->generate_stream_v2(
            "generate.chat_json_stream_ex", chat_request_json, config, callback, user_data);
    } catch (...) {
        record_boundary_failure(engine, "aila_generate_chat_json_stream_ex");
        return -1;
    }
}

AILA_API void aila_engine_reset_context(AilaEngine* engine) {
    if (!engine) {
        return;
    }
    try {
        engine->proxy->reset_context();
    } catch (...) {
        record_boundary_failure(engine, "aila_engine_reset_context");
    }
}

AILA_API int aila_engine_context_length(AilaEngine* engine) {
    if (!engine) {
        return 0;
    }
    try {
        return engine->proxy->context_length();
    } catch (...) {
        record_boundary_failure(engine, "aila_engine_context_length");
        return 0;
    }
}

AILA_API int aila_last_error_code(AilaEngine* engine) {
    if (!engine) {
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        return engine->proxy->last_error_code();
    } catch (...) {
        return AILA_ERR_RUNTIME;
    }
}

AILA_API const char* aila_last_error_message(AilaEngine* engine) {
    if (!engine) {
        return kEmpty;
    }
    try {
        return engine->proxy->last_error_message();
    } catch (...) {
        return kEmpty;
    }
}

AILA_API char* aila_transcribe(
    AilaEngine* engine,
    const char* wav_path,
    const AilaGenConfig* config,
    const char* forced_language,
    const char* system_prompt,
    float segment_sec,
    int past_text_conditioning,
    AilaTokenCallback token_callback,
    void* user_data,
    char** language_out) {
    if (language_out) *language_out = nullptr;
    if (!engine) return nullptr;
    if (!wav_path) {
        try { engine->proxy->record_invalid_argument("wav_path must not be NULL"); } catch (...) {}
        return nullptr;
    }
    try {
        std::string transcript;
        std::string language;
        if (!engine->proxy->transcribe(
                wav_path, config, forced_language, system_prompt, segment_sec,
                past_text_conditioning, token_callback, user_data, transcript, language)) {
            return nullptr;
        }
        char* allocated_language = nullptr;
        if (language_out && !language.empty()) {
            allocated_language = allocate_result(language);
            if (!allocated_language) {
                engine->proxy->record_runtime_error("could not allocate ASR language result");
                return nullptr;
            }
        }
        char* allocated_transcript = allocate_result(transcript);
        if (!allocated_transcript) {
            std::free(allocated_language);
            engine->proxy->record_runtime_error("could not allocate ASR transcript result");
            return nullptr;
        }
        if (language_out) *language_out = allocated_language;
        return allocated_transcript;
    } catch (...) {
        record_boundary_failure(engine, "aila_transcribe");
        return nullptr;
    }
}

AILA_API AilaTranscribeStream* aila_transcribe_stream_create(
    AilaEngine* engine,
    const AilaGenConfig* config,
    const char* forced_language,
    const char* system_prompt) {
    if (!engine) return nullptr;
    try {
        uint64_t session = 0;
        uint64_t id = 0;
        if (!engine->proxy->transcribe_stream_create(
                config, forced_language, system_prompt, session, id)) return nullptr;
        try {
            return new AilaTranscribeStream{engine->proxy, session, id};
        } catch (...) {
            engine->proxy->transcribe_stream_destroy(session, id);
            throw;
        }
    } catch (...) {
        record_boundary_failure(engine, "aila_transcribe_stream_create");
        return nullptr;
    }
}

AILA_API int aila_transcribe_stream_feed(
    AilaTranscribeStream* stream,
    const float* samples,
    int sample_count) {
    if (!stream || !samples || sample_count <= 0) return AILA_ERR_INVALID_ARGUMENT;
    try {
        return stream->owner->transcribe_stream_feed(
            stream->worker_session, stream->remote_id, samples, sample_count);
    } catch (...) {
        return AILA_ERR_RUNTIME;
    }
}

AILA_API int aila_transcribe_stream_get_text(
    AilaTranscribeStream* stream,
    char** out_stable,
    char** out_partial) {
    if (out_stable) *out_stable = nullptr;
    if (out_partial) *out_partial = nullptr;
    if (!stream) return AILA_ERR_INVALID_ARGUMENT;
    try {
        std::string stable;
        std::string partial;
        const int status = stream->owner->transcribe_stream_get_text(
            stream->worker_session, stream->remote_id, stable, partial);
        if (status != AILA_OK) return status;
        char* stable_result = nullptr;
        char* partial_result = nullptr;
        if (out_stable && !stable.empty()) {
            stable_result = allocate_result(stable);
            if (!stable_result) {
                stream->owner->record_runtime_error("could not allocate stable ASR text");
                return AILA_ERR_RUNTIME;
            }
        }
        if (out_partial && !partial.empty()) {
            partial_result = allocate_result(partial);
            if (!partial_result) {
                std::free(stable_result);
                stream->owner->record_runtime_error("could not allocate partial ASR text");
                return AILA_ERR_RUNTIME;
            }
        }
        if (out_stable) *out_stable = stable_result;
        if (out_partial) *out_partial = partial_result;
        return AILA_OK;
    } catch (...) {
        return AILA_ERR_RUNTIME;
    }
}

AILA_API void aila_transcribe_stream_destroy(AilaTranscribeStream* stream) {
    if (!stream) return;
    try {
        stream->owner->transcribe_stream_destroy(stream->worker_session, stream->remote_id);
    } catch (...) {}
    try { delete stream; } catch (...) {}
}

AILA_API int aila_synthesize_wav(
    AilaEngine* engine, const int* text_tokens, int text_tokens_len,
    const float* speaker_embedding, int speaker_embedding_len,
    const AilaGenConfig* config, float** out_samples, int* out_sample_count) {
    if (out_samples) *out_samples = nullptr;
    if (out_sample_count) *out_sample_count = 0;
    if (!engine || !text_tokens || text_tokens_len <= 0 || !out_samples || !out_sample_count) {
        if (engine) try { engine->proxy->record_invalid_argument("TTS synthesis arguments are invalid"); } catch (...) {}
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<float> values;
        const int status = engine->proxy->synthesize_wav(
            text_tokens, text_tokens_len, speaker_embedding, speaker_embedding_len, config, values);
        if (status != AILA_OK) return status;
        const int allocated = allocate_float_result(values, out_samples, out_sample_count);
        if (allocated != AILA_OK) engine->proxy->record_runtime_error("could not allocate TTS samples");
        return allocated;
    } catch (...) { record_boundary_failure(engine, "aila_synthesize_wav"); return AILA_ERR_RUNTIME; }
}

AILA_API int aila_synthesize_text_to_wav(
    AilaEngine* engine, const char* text, const float* speaker_embedding,
    int speaker_embedding_len, const AilaGenConfig* config,
    float** out_samples, int* out_sample_count) {
    if (out_samples) *out_samples = nullptr;
    if (out_sample_count) *out_sample_count = 0;
    if (!engine || !text || !out_samples || !out_sample_count) {
        if (engine) try { engine->proxy->record_invalid_argument("TTS text synthesis arguments are invalid"); } catch (...) {}
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<float> values;
        const int status = engine->proxy->synthesize_text_to_wav(
            text, speaker_embedding, speaker_embedding_len, config, values);
        if (status != AILA_OK) return status;
        const int allocated = allocate_float_result(values, out_samples, out_sample_count);
        if (allocated != AILA_OK) engine->proxy->record_runtime_error("could not allocate TTS samples");
        return allocated;
    } catch (...) { record_boundary_failure(engine, "aila_synthesize_text_to_wav"); return AILA_ERR_RUNTIME; }
}

AILA_API AilaTTSOptions aila_tts_options_default(void) {
    AilaTTSOptions opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.struct_size = static_cast<uint32_t>(sizeof(AilaTTSOptions));
    opts.voice_clone_mode = AILA_VOICE_CLONE_AUTO;
    return opts;
}

AILA_API int aila_synthesize(
    AilaEngine* engine, const char* text, const char* reference_audio_path,
    const char* speaker_name, const char* instruct_text, const char* language,
    const AilaGenConfig* config, const char* output_wav_path) {
    return aila_synthesize_ex(
        engine, text, reference_audio_path, speaker_name, instruct_text,
        language, config, nullptr, output_wav_path
    );
}

AILA_API int aila_synthesize_ex(
    AilaEngine* engine, const char* text, const char* reference_audio_path,
    const char* speaker_name, const char* instruct_text, const char* language,
    const AilaGenConfig* config, const AilaTTSOptions* options, const char* output_wav_path) {
    if (!engine || !text || !output_wav_path) {
        if (engine) try { engine->proxy->record_invalid_argument("TTS file arguments are invalid"); } catch (...) {}
        return AILA_ERR_INVALID_ARGUMENT;
    }
    AilaTTSOptions safe_options = aila_tts_options_default();
    const AilaTTSOptions* p_options = nullptr;
    if (options) {
        uint32_t sz = options->struct_size == 0 ? static_cast<uint32_t>(sizeof(AilaTTSOptions)) : options->struct_size;
        if (sz < sizeof(uint32_t)) {
            if (engine) try { engine->proxy->record_invalid_argument("AilaTTSOptions struct_size is invalid"); } catch (...) {}
            return AILA_ERR_INVALID_ARGUMENT;
        }
        if (sz >= offsetof(AilaTTSOptions, reference_text) + sizeof(options->reference_text)) {
            safe_options.reference_text = options->reference_text;
        }
        if (sz >= offsetof(AilaTTSOptions, voice_clone_mode) + sizeof(options->voice_clone_mode)) {
            safe_options.voice_clone_mode = options->voice_clone_mode;
        }
        if (sz >= sizeof(AilaTTSOptions)) {
            for (int i = 0; i < 6; ++i) {
                if (options->reserved[i] != 0) {
                    if (engine) try { engine->proxy->record_invalid_argument("AilaTTSOptions reserved fields must be 0"); } catch (...) {}
                    return AILA_ERR_INVALID_ARGUMENT;
                }
            }
        }
        p_options = &safe_options;
    }
    try {
        return engine->proxy->synthesize_file(
            text, reference_audio_path, speaker_name, instruct_text, language, config,
            output_wav_path, p_options);
    } catch (...) { record_boundary_failure(engine, "aila_synthesize_ex"); return AILA_ERR_RUNTIME; }
}

AILA_API int aila_decode_mimi_vocoder(
    AilaEngine* engine, const int32_t* codes, int n_frames,
    float** out_samples, int* out_sample_count) {
    if (out_samples) *out_samples = nullptr;
    if (out_sample_count) *out_sample_count = 0;
    if (!engine || !codes || n_frames <= 0 || !out_samples || !out_sample_count) {
        if (engine) try { engine->proxy->record_invalid_argument("Mimi decode arguments are invalid"); } catch (...) {}
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<float> values;
        const int status = engine->proxy->decode_mimi(codes, n_frames, values);
        if (status != AILA_OK) return status;
        const int allocated = allocate_float_result(values, out_samples, out_sample_count);
        if (allocated != AILA_OK) engine->proxy->record_runtime_error("could not allocate Mimi samples");
        return allocated;
    } catch (...) { record_boundary_failure(engine, "aila_decode_mimi_vocoder"); return AILA_ERR_RUNTIME; }
}

AILA_API int aila_extract_speaker_embedding(
    AilaEngine* engine, const char* audio_path,
    float** out_embedding, int* out_embedding_dim) {
    if (out_embedding) *out_embedding = nullptr;
    if (out_embedding_dim) *out_embedding_dim = 0;
    if (!engine || !audio_path || !out_embedding || !out_embedding_dim) {
        if (engine) try { engine->proxy->record_invalid_argument("speaker embedding arguments are invalid"); } catch (...) {}
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<float> values;
        const int status = engine->proxy->extract_speaker_embedding(audio_path, values);
        if (status != AILA_OK) return status;
        const int allocated = allocate_float_result(values, out_embedding, out_embedding_dim);
        if (allocated != AILA_OK) engine->proxy->record_runtime_error("could not allocate speaker embedding");
        return allocated;
    } catch (...) { record_boundary_failure(engine, "aila_extract_speaker_embedding"); return AILA_ERR_RUNTIME; }
}

AILA_API AilaTTSStream* aila_synthesize_stream(
    AilaEngine* engine, const char* text, const char* reference_audio_path,
    const char* speaker_name, const char* instruct_text, const char* language,
    const AilaGenConfig* config, AilaAudioCallback callback, void* user_data) {
    return aila_synthesize_stream_ex(
        engine, text, reference_audio_path, speaker_name, instruct_text,
        language, config, nullptr, callback, user_data
    );
}

AILA_API AilaTTSStream* aila_synthesize_stream_ex(
    AilaEngine* engine, const char* text, const char* reference_audio_path,
    const char* speaker_name, const char* instruct_text, const char* language,
    const AilaGenConfig* config, const AilaTTSOptions* options,
    AilaAudioCallback callback, void* user_data) {
    if (!engine || !text || !callback) {
        if (engine) try { engine->proxy->record_invalid_argument("TTS stream arguments are invalid"); } catch (...) {}
        return nullptr;
    }
    const bool has_options = options != nullptr;
    AilaTTSOptions options_copy = aila_tts_options_default();
    std::string ref_text_copy;
    if (has_options) {
        uint32_t sz = options->struct_size == 0 ? static_cast<uint32_t>(sizeof(AilaTTSOptions)) : options->struct_size;
        if (sz < sizeof(uint32_t)) {
            if (engine) try { engine->proxy->record_invalid_argument("AilaTTSOptions struct_size is invalid"); } catch (...) {}
            return nullptr;
        }
        if (sz >= offsetof(AilaTTSOptions, reference_text) + sizeof(options->reference_text)) {
            if (options->reference_text) {
                ref_text_copy = options->reference_text;
                options_copy.reference_text = ref_text_copy.c_str();
            }
        }
        if (sz >= offsetof(AilaTTSOptions, voice_clone_mode) + sizeof(options->voice_clone_mode)) {
            options_copy.voice_clone_mode = options->voice_clone_mode;
        }
        if (sz >= sizeof(AilaTTSOptions)) {
            for (int i = 0; i < 6; ++i) {
                if (options->reserved[i] != 0) {
                    if (engine) try { engine->proxy->record_invalid_argument("AilaTTSOptions reserved fields must be 0"); } catch (...) {}
                    return nullptr;
                }
            }
        }
    }
    try {
        auto stream = std::make_unique<AilaTTSStream>();
        stream->owner = engine->proxy;
        stream->callback = callback;
        stream->user_data = user_data;
        const std::string text_copy(text);
        const bool has_reference = reference_audio_path != nullptr;
        const bool has_speaker = speaker_name != nullptr;
        const bool has_instruct = instruct_text != nullptr;
        const bool has_language = language != nullptr;
        const std::string reference_copy = has_reference ? reference_audio_path : "";
        const std::string speaker_copy = has_speaker ? speaker_name : "";
        const std::string instruct_copy = has_instruct ? instruct_text : "";
        const std::string language_copy = has_language ? language : "";
        const bool has_config = config != nullptr;
        const AilaGenConfig config_copy = has_config ? *config : AilaGenConfig{};
        AilaTTSStream* raw = stream.get();
        raw->worker = std::thread([
            raw, text_copy, has_reference, reference_copy, has_speaker, speaker_copy,
            has_instruct, instruct_copy, has_language, language_copy, has_config, config_copy,
            has_options, options_copy, ref_text_copy] {
            int status = AILA_ERR_RUNTIME;
            try {
                AilaTTSOptions options_val{};
            if (has_options) options_val = options_copy;
            status = raw->owner->synthesize_stream(
                    text_copy,
                    has_reference ? reference_copy.c_str() : nullptr,
                    has_speaker ? speaker_copy.c_str() : nullptr,
                    has_instruct ? instruct_copy.c_str() : nullptr,
                    has_language ? language_copy.c_str() : nullptr,
                    has_config ? &config_copy : nullptr,
                    has_options ? &options_val : nullptr,
                    raw->callback, raw->user_data, raw->cancel_requested,
                    [raw] {
                        {
                            std::lock_guard<std::mutex> lock(raw->mutex);
                            raw->started = true;
                        }
                        raw->condition.notify_all();
                    });
            } catch (...) {
                try { raw->owner->record_runtime_error("TTS stream failed at background boundary"); } catch (...) {}
            }
            bool delete_self = false;
            {
                std::lock_guard<std::mutex> lock(raw->mutex);
                raw->status = status;
                raw->done = true;
                delete_self = raw->destroy_requested;
            }
            raw->condition.notify_all();
            if (delete_self) {
                try {
                    if (raw->worker.joinable()) raw->worker.detach();
                } catch (...) {
                    return;
                }
                delete raw;
                return;
            }
        });
        {
            std::unique_lock<std::mutex> lock(raw->mutex);
            raw->condition.wait(lock, [&] { return raw->started || raw->done; });
            if (!raw->started) {
                lock.unlock();
                if (raw->worker.joinable()) raw->worker.join();
                return nullptr;
            }
        }
        return stream.release();
    } catch (...) {
        record_boundary_failure(engine, "aila_synthesize_stream");
        return nullptr;
    }
}

AILA_API int aila_stream_wait(AilaTTSStream* stream) {
    if (!stream) return AILA_ERR_INVALID_ARGUMENT;
    try {
        std::unique_lock<std::mutex> lock(stream->mutex);
        if (!stream->joined && stream->worker.get_id() == std::this_thread::get_id()) {
            return AILA_ERR_RUNTIME;
        }
        while (stream->joining && !stream->joined) stream->condition.wait(lock);
        if (!stream->joined) {
            stream->joining = true;
            lock.unlock();
            if (stream->worker.joinable()) stream->worker.join();
            lock.lock();
            stream->joined = true;
            stream->joining = false;
            stream->condition.notify_all();
        }
        return stream->status;
    } catch (...) {
        return AILA_ERR_RUNTIME;
    }
}

AILA_API void aila_stream_destroy(AilaTTSStream* stream) {
    if (!stream) return;
    stream->cancel_requested.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(stream->mutex);
        if (!stream->joined && stream->worker.get_id() == std::this_thread::get_id()) {
            stream->destroy_requested = true;
            return;
        }
    }
    (void)aila_stream_wait(stream);
    try { delete stream; } catch (...) {}
}

AILA_API void aila_free_string(char* string) {
    std::free(string);
}

AILA_API void aila_free_samples(float* samples) {
    std::free(samples);
}

AILA_API int aila_align(
    AilaEngine* engine,
    const float* audio_samples, int num_samples, int sample_rate,
    const char* text, const char* language,
    AilaAlignedWord** out_words, int* out_count) {
    if (out_words) *out_words = nullptr;
    if (out_count) *out_count = 0;
    if (!engine || !audio_samples || num_samples <= 0 || !text || !language ||
        !out_words || !out_count) {
        if (engine) try { engine->proxy->record_invalid_argument("alignment arguments are invalid"); }
        catch (...) {}
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        const std::string_view text_view(text);
        const std::string_view language_view(language);
        if (!simdjson::validate_utf8(text_view) || !simdjson::validate_utf8(language_view)) {
            engine->proxy->record_invalid_argument("alignment text and language must be valid UTF-8");
            return AILA_ERR_INVALID_ARGUMENT;
        }
        std::vector<aila::proxy::AlignmentWord> values;
        const int status = engine->proxy->align(
            audio_samples, num_samples, sample_rate, text_view, language_view, values);
        if (status != AILA_OK) return status;
        const int allocated = allocate_alignment_result(values, out_words, out_count);
        if (allocated != AILA_OK) {
            engine->proxy->record_runtime_error("could not allocate aligned words");
        }
        return allocated;
    } catch (...) {
        record_boundary_failure(engine, "aila_align");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API int aila_align_words(
    AilaEngine* engine,
    const float* audio_samples, int num_samples, int sample_rate,
    const char* const* words, int num_words,
    AilaAlignedWord** out_words, int* out_count) {
    if (out_words) *out_words = nullptr;
    if (out_count) *out_count = 0;
    if (!engine || !audio_samples || num_samples <= 0 || !words || num_words <= 0 ||
        !out_words || !out_count) {
        if (engine) try { engine->proxy->record_invalid_argument("pre-tokenized alignment arguments are invalid"); }
        catch (...) {}
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        std::vector<std::string> input_words;
        input_words.reserve(static_cast<size_t>(num_words));
        for (int index = 0; index < num_words; ++index) {
            const char* word = words[index] ? words[index] : "";
            const std::string_view view(word);
            if (!simdjson::validate_utf8(view)) {
                engine->proxy->record_invalid_argument("alignment words must be valid UTF-8");
                return AILA_ERR_INVALID_ARGUMENT;
            }
            input_words.emplace_back(view);
        }
        std::vector<aila::proxy::AlignmentWord> values;
        const int status = engine->proxy->align_words(
            audio_samples, num_samples, sample_rate, input_words, values);
        if (status != AILA_OK) return status;
        const int allocated = allocate_alignment_result(values, out_words, out_count);
        if (allocated != AILA_OK) {
            engine->proxy->record_runtime_error("could not allocate pre-tokenized aligned words");
        }
        return allocated;
    } catch (...) {
        record_boundary_failure(engine, "aila_align_words");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API void aila_free_aligned_words(AilaAlignedWord* words, int count) {
    if (!words) {
        return;
    }
    for (int index = 0; index < count; ++index) {
        std::free(const_cast<char*>(words[index].text));
    }
    std::free(words);
}

AILA_API void aila_set_log_callback(AilaLogCallback callback, void* user_data) {
    aila::proxy::logging::set_callback(callback, user_data);
}

AILA_API void aila_set_log_level(int level) {
    if (level < 0 || level > 3) {
        return;
    }
    aila::proxy::logging::set_level(level);
    try {
        const auto engines = live_engines();
        for (const auto& engine : engines) engine->set_log_level(level);
    } catch (...) {}
}
