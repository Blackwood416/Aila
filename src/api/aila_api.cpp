#include "aila_api.h"
#include "engine/Engine.hpp"
#include "profile/Profiling.hpp"
#include <cstring>
#include <string>

// ============================================================
// Version
// ============================================================
static const char* AILA_VERSION_STRING = "0.1.0";

// ============================================================
// Opaque handle wraps InferenceEngine
// ============================================================
struct AilaEngine {
    InferenceEngine engine;
};

// ============================================================
// Helper: Convert C config to C++ config
// ============================================================
static GenerationConfig to_cpp_config(const AilaGenConfig* c_config) {
    GenerationConfig cfg;
    if (!c_config) return cfg;

    cfg.max_new_tokens     = c_config->max_new_tokens;
    cfg.temperature        = c_config->temperature;
    cfg.top_k              = c_config->top_k;
    cfg.top_p              = c_config->top_p;
    cfg.repetition_penalty = c_config->repetition_penalty;
    cfg.presence_penalty   = c_config->presence_penalty;
    cfg.frequency_penalty  = c_config->frequency_penalty;
    cfg.do_sample          = (c_config->do_sample != 0);
    cfg.decode_chunk_size  = c_config->decode_chunk_size;
    cfg.stream_chunk_size  = c_config->stream_chunk_size;
    return cfg;
}

// ============================================================
// API Implementation
// ============================================================

AILA_API const char* aila_version(void) {
    return AILA_VERSION_STRING;
}

AILA_API AilaEngine* aila_engine_create(void) {
    try {
        return new AilaEngine();
    } catch (...) {
        return nullptr;
    }
}

AILA_API int aila_engine_init(AilaEngine* engine, const char* model_dir, int max_seq_len) {
    if (!engine || !model_dir) return -1;
    try {
        bool ok = engine->engine.init(std::string(model_dir), max_seq_len);
        return ok ? 0 : -1;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Init failed: %s", e.what());
        return -1;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Init failed: unknown exception");
        return -1;
    }
}

AILA_API void aila_engine_destroy(AilaEngine* engine) {
    delete engine;
}

AILA_API AilaGenConfig aila_default_gen_config(void) {
    AilaGenConfig cfg;
    cfg.max_new_tokens     = 512;
    cfg.temperature        = 0.6f;
    cfg.top_k              = 20;
    cfg.top_p              = 0.95f;
    cfg.repetition_penalty = 1.0f;
    cfg.presence_penalty   = 0.0f;
    cfg.frequency_penalty  = 0.0f;
    cfg.do_sample          = 1;
    cfg.decode_chunk_size  = 1;
    cfg.stream_chunk_size  = 1;
    return cfg;
}

AILA_API char* aila_generate(AilaEngine* engine, const char* prompt, const AilaGenConfig* config) {
    if (!engine || !prompt) return nullptr;

    try {
        GenerationConfig cfg = to_cpp_config(config);
        std::string result = engine->engine.generate(std::string(prompt), cfg, nullptr);

        // Allocate and copy result
        char* out = static_cast<char*>(malloc(result.size() + 1));
        if (!out) return nullptr;
        memcpy(out, result.c_str(), result.size() + 1);
        return out;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Generate failed: %s", e.what());
        return nullptr;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Generate failed: unknown exception");
        return nullptr;
    }
}

AILA_API int aila_generate_stream(AilaEngine* engine, const char* prompt,
                                   const AilaGenConfig* config,
                                   AilaTokenCallback callback, void* user_data) {
    if (!engine || !prompt || !callback) return -1;

    try {
        GenerationConfig cfg = to_cpp_config(config);
        bool aborted = false;

        engine->engine.generate(std::string(prompt), cfg,
            [&](const std::string& token_text) {
                if (!aborted) {
                    int ret = callback(token_text.c_str(), user_data);
                    if (ret != 0) aborted = true;
                }
            });

        return aborted ? 1 : 0;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Stream generate failed: %s", e.what());
        return -1;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Stream generate failed: unknown exception");
        return -1;
    }
}

AILA_API void aila_free_string(char* str) {
    free(str);
}

AILA_API void aila_set_log_callback(AilaLogCallback callback, void* user_data) {
    aila::set_log_callback(callback, user_data);
}

AILA_API void aila_set_log_level(int level) {
    if (level >= 0 && level <= 3) {
        aila::set_log_level(static_cast<aila::LogLevel>(level));
    }
}

AILA_API void aila_engine_reset_context(AilaEngine* engine) {
    if (engine) {
        engine->engine.reset_context();
    }
}

AILA_API int aila_engine_context_length(AilaEngine* engine) {
    if (!engine) return 0;
    return engine->engine.context_length();
}
