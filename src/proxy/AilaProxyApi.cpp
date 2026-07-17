#include "aila_api.h"

#include "proxy/ProxyEngine.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <memory>
#include <new>
#include <string>

struct AilaEngine {
    AilaEngine() : proxy(std::make_shared<aila::proxy::ProxyEngine>()) {}
    std::shared_ptr<aila::proxy::ProxyEngine> proxy;
};

struct AilaTranscribeStream {
    std::shared_ptr<aila::proxy::ProxyEngine> owner;
    uint64_t remote_id = 0;
};

namespace {

constexpr const char* kVersion = "0.1.6";
constexpr const char* kEmpty = "";

std::mutex log_mutex;
AilaLogCallback log_callback = nullptr;
void* log_user_data = nullptr;
int log_level = 1;

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
        return new AilaEngine();
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
        uint64_t id = 0;
        if (!engine->proxy->transcribe_stream_create(
                config, forced_language, system_prompt, id)) return nullptr;
        try {
            return new AilaTranscribeStream{engine->proxy, id};
        } catch (...) {
            engine->proxy->transcribe_stream_destroy(id);
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
        return stream->owner->transcribe_stream_feed(stream->remote_id, samples, sample_count);
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
            stream->remote_id, stable, partial);
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
    try { stream->owner->transcribe_stream_destroy(stream->remote_id); } catch (...) {}
    try { delete stream; } catch (...) {}
}

AILA_API void aila_free_string(char* string) {
    std::free(string);
}

AILA_API void aila_free_samples(float* samples) {
    std::free(samples);
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
    try {
        std::lock_guard<std::mutex> lock(log_mutex);
        log_callback = callback;
        log_user_data = user_data;
    } catch (...) {
    }
}

AILA_API void aila_set_log_level(int level) {
    if (level < 0 || level > 3) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(log_mutex);
        log_level = level;
    } catch (...) {
    }
}
