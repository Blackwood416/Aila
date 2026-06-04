#include "aila_api.h"
#include "engine/Engine.hpp"
#include "profile/Profiling.hpp"
#include "AudioPreprocessor.hpp"
#include <cstring>
#include <vector>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

// ============================================================
// Version
// ============================================================
static const char* AILA_VERSION_STRING = "0.1.5";

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

static int to_c_error_code(EngineErrorCode code) {
    switch (code) {
        case EngineErrorCode::Ok: return AILA_OK;
        case EngineErrorCode::InvalidArgument: return AILA_ERR_INVALID_ARGUMENT;
        case EngineErrorCode::TemplateError: return AILA_ERR_TEMPLATE;
        case EngineErrorCode::JsonParseError: return AILA_ERR_JSON_PARSE;
        case EngineErrorCode::VisionNotEnabled: return AILA_ERR_VISION_NOT_ENABLED;
        case EngineErrorCode::ContextOverflow: return AILA_ERR_CONTEXT_OVERFLOW;
        default: return AILA_ERR_RUNTIME;
    }
}

static int run_streaming_call(AilaEngine* engine,
                              AilaTokenCallback callback,
                              void* user_data,
                              const std::function<void(bool&)>& invoke) {
    if (!engine || !callback) return -1;
    try {
        bool aborted = false;
        invoke(aborted);
        if (engine->engine.last_error_code() != EngineErrorCode::Ok) {
            return -1;
        }
        return aborted ? 1 : 0;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Stream generate failed: %s", e.what());
        return -1;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Stream generate failed: unknown exception");
        return -1;
    }
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
    cfg.decode_chunk_size  = 12;
    cfg.stream_chunk_size  = 4;
    return cfg;
}

AILA_API char* aila_generate(AilaEngine* engine, const char* prompt, const AilaGenConfig* config) {
    if (!engine || !prompt) return nullptr;

    try {
        GenerationConfig cfg = to_cpp_config(config);
        std::string result = engine->engine.generate(std::string(prompt), cfg, nullptr);
        if (engine->engine.last_error_code() != EngineErrorCode::Ok) {
            return nullptr;
        }

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

AILA_API char* aila_generate_messages(AilaEngine* engine, const char* messages_json,
                                      const AilaGenConfig* config) {
    if (!engine || !messages_json) return nullptr;
    try {
        GenerationConfig cfg = to_cpp_config(config);
        std::string result = engine->engine.generate_messages_json(std::string(messages_json), cfg, nullptr);
        if (engine->engine.last_error_code() != EngineErrorCode::Ok) {
            return nullptr;
        }
        char* out = static_cast<char*>(malloc(result.size() + 1));
        if (!out) return nullptr;
        memcpy(out, result.c_str(), result.size() + 1);
        return out;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Generate messages failed: %s", e.what());
        return nullptr;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Generate messages failed: unknown exception");
        return nullptr;
    }
}

AILA_API int aila_generate_messages_stream(AilaEngine* engine, const char* messages_json,
                                           const AilaGenConfig* config,
                                           AilaTokenCallback callback, void* user_data) {
    if (!engine || !messages_json || !callback) return -1;

    GenerationConfig cfg = to_cpp_config(config);
    return run_streaming_call(engine, callback, user_data, [&](bool& aborted) {
        engine->engine.generate_messages_json(std::string(messages_json), cfg,
            [&](const std::string& token_text) {
                if (!aborted) {
                    int ret = callback(token_text.c_str(), user_data);
                    if (ret != 0) aborted = true;
                }
            });
    });
}

AILA_API int aila_generate_stream(AilaEngine* engine, const char* prompt,
                                   const AilaGenConfig* config,
                                   AilaTokenCallback callback, void* user_data) {
    if (!engine || !prompt || !callback) return -1;

    GenerationConfig cfg = to_cpp_config(config);
    return run_streaming_call(engine, callback, user_data, [&](bool& aborted) {
        engine->engine.generate(std::string(prompt), cfg,
            [&](const std::string& token_text) {
                if (!aborted) {
                    int ret = callback(token_text.c_str(), user_data);
                    if (ret != 0) aborted = true;
                }
            });
    });
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

AILA_API int aila_last_error_code(AilaEngine* engine) {
    if (!engine) return AILA_ERR_INVALID_ARGUMENT;
    return to_c_error_code(engine->engine.last_error_code());
}

AILA_API const char* aila_last_error_message(AilaEngine* engine) {
    static const char* kEmpty = "";
    if (!engine) return kEmpty;
    return engine->engine.last_error_message().c_str();
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
    char** language_out
) {
    if (language_out) {
        *language_out = nullptr;
    }
    if (!engine || !wav_path) {
        return nullptr;
    }

    try {
        GenerationConfig cfg = to_cpp_config(config);
        std::string cpp_lang;
        std::string cpp_forced = forced_language ? forced_language : "";
        std::string cpp_sys = system_prompt ? system_prompt : "";
        bool cpp_past = (past_text_conditioning != 0);

        std::function<void(const std::string&)> token_cb = nullptr;
        if (token_callback) {
            token_cb = [=](const std::string& token_text) {
                token_callback(token_text.c_str(), user_data);
            };
        }

        std::string result = engine->engine.transcribe(
            std::string(wav_path),
            cfg,
            &cpp_lang,
            cpp_forced,
            cpp_sys,
            segment_sec,
            cpp_past,
            token_cb
        );

        if (engine->engine.last_error_code() != EngineErrorCode::Ok) {
            return nullptr;
        }

        // Return recognized language if requested
        if (language_out && !cpp_lang.empty()) {
            char* out_lang = static_cast<char*>(malloc(cpp_lang.size() + 1));
            if (out_lang) {
                memcpy(out_lang, cpp_lang.c_str(), cpp_lang.size() + 1);
                *language_out = out_lang;
            }
        }

        // Return transcript text
        char* out_text = static_cast<char*>(malloc(result.size() + 1));
        if (!out_text) {
            // Cleanup language string if allocated
            if (language_out && *language_out) {
                free(*language_out);
                *language_out = nullptr;
            }
            return nullptr;
        }
        memcpy(out_text, result.c_str(), result.size() + 1);
        return out_text;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Transcribe failed: %s", e.what());
        if (language_out && *language_out) {
            free(*language_out);
            *language_out = nullptr;
        }
        return nullptr;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Transcribe failed: unknown exception");
        if (language_out && *language_out) {
            free(*language_out);
            *language_out = nullptr;
        }
        return nullptr;
    }
}

struct AilaTranscribeStream {
    AilaTranscribeStream(
        InferenceEngine* engine,
        const GenerationConfig& gen_config,
        const std::string& forced_language,
        const std::string& system_prompt
    ) : cpp_stream(engine, gen_config, forced_language, system_prompt) {}

    InferenceEngine::TranscribeStream cpp_stream;
};

AILA_API AilaTranscribeStream* aila_transcribe_stream_create(
    AilaEngine* engine,
    const AilaGenConfig* config,
    const char* forced_language,
    const char* system_prompt
) {
    if (!engine) return nullptr;

    try {
        engine->engine.clear_error();

        if (engine->engine.model_spec().family != ModelFamily::Qwen3ASR) {
            engine->engine.set_error(EngineErrorCode::RuntimeError, "Model does not support ASR");
            return nullptr;
        }

        GenerationConfig cfg = to_cpp_config(config);
        std::string cpp_forced = forced_language ? forced_language : "";
        std::string cpp_sys = system_prompt ? system_prompt : "";

        return new AilaTranscribeStream(&(engine->engine), cfg, cpp_forced, cpp_sys);
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Create transcribe stream failed: %s", e.what());
        engine->engine.set_error(EngineErrorCode::RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Create transcribe stream failed: unknown exception");
        engine->engine.set_error(EngineErrorCode::RuntimeError, "Unknown exception in stream creation");
        return nullptr;
    }
}

AILA_API int aila_transcribe_stream_feed(
    AilaTranscribeStream* stream,
    const float* samples,
    int sample_count
) {
    if (!stream || !samples || sample_count <= 0) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        stream->cpp_stream.feed_audio(samples, static_cast<size_t>(sample_count));
        return AILA_OK;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Stream feed failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Stream feed failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API int aila_transcribe_stream_get_text(
    AilaTranscribeStream* stream,
    char** out_stable,
    char** out_partial
) {
    if (out_stable) *out_stable = nullptr;
    if (out_partial) *out_partial = nullptr;

    if (!stream) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        std::string stable;
        std::string partial;
        stream->cpp_stream.get_text(stable, partial);

        if (out_stable && !stable.empty()) {
            char* s = static_cast<char*>(malloc(stable.size() + 1));
            if (s) {
                memcpy(s, stable.c_str(), stable.size() + 1);
                *out_stable = s;
            }
        }

        if (out_partial && !partial.empty()) {
            char* p = static_cast<char*>(malloc(partial.size() + 1));
            if (p) {
                memcpy(p, partial.c_str(), partial.size() + 1);
                *out_partial = p;
            }
        }

        return AILA_OK;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Stream get text failed: %s", e.what());
        if (out_stable && *out_stable) { free(*out_stable); *out_stable = nullptr; }
        if (out_partial && *out_partial) { free(*out_partial); *out_partial = nullptr; }
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Stream get text failed: unknown exception");
        if (out_stable && *out_stable) { free(*out_stable); *out_stable = nullptr; }
        if (out_partial && *out_partial) { free(*out_partial); *out_partial = nullptr; }
        return AILA_ERR_RUNTIME;
    }
}

AILA_API void aila_transcribe_stream_destroy(AilaTranscribeStream* stream) {
    if (stream) {
        delete stream;
    }
}


AILA_API int aila_synthesize_wav(
    AilaEngine* engine,
    const int* text_tokens,
    int text_tokens_len,
    const float* speaker_embedding,
    int speaker_embedding_len,
    const AilaGenConfig* config,
    float** out_samples,
    int* out_sample_count
) {
    if (!engine || !text_tokens || text_tokens_len <= 0 || !out_samples || !out_sample_count) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        GenerationConfig cpp_cfg = to_cpp_config(config);
        
        std::vector<int> tokens(text_tokens, text_tokens + text_tokens_len);
        std::vector<float> spk_emb;
        if (speaker_embedding && speaker_embedding_len > 0) {
            spk_emb.assign(speaker_embedding, speaker_embedding + speaker_embedding_len);
        }

        std::vector<float> samples;
        bool ok = engine->engine.synthesize_wav(tokens, spk_emb, cpp_cfg, samples);
        if (!ok) {
            return AILA_ERR_RUNTIME;
        }

        float* c_arr = (float*)malloc(samples.size() * sizeof(float));
        if (!c_arr) {
            return AILA_ERR_RUNTIME;
        }
        std::copy(samples.begin(), samples.end(), c_arr);

        *out_samples = c_arr;
        *out_sample_count = static_cast<int>(samples.size());

        return AILA_OK;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Synthesis WAV failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Synthesis WAV failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API int aila_synthesize_text_to_wav(
    AilaEngine* engine,
    const char* text,
    const float* speaker_embedding,
    int speaker_embedding_len,
    const AilaGenConfig* config,
    float** out_samples,
    int* out_sample_count
) {
    if (!engine || !text || !out_samples || !out_sample_count) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        GenerationConfig cpp_cfg = to_cpp_config(config);
        
        std::vector<float> spk_emb;
        if (speaker_embedding && speaker_embedding_len > 0) {
            spk_emb.assign(speaker_embedding, speaker_embedding + speaker_embedding_len);
        }

        std::vector<float> samples;
        bool ok = engine->engine.synthesize_text_to_wav(text, spk_emb, cpp_cfg, samples);
        if (!ok) {
            return AILA_ERR_RUNTIME;
        }

        float* c_arr = (float*)malloc(samples.size() * sizeof(float));
        if (!c_arr) {
            return AILA_ERR_RUNTIME;
        }
        std::copy(samples.begin(), samples.end(), c_arr);

        *out_samples = c_arr;
        *out_sample_count = static_cast<int>(samples.size());

        return AILA_OK;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Synthesis text to WAV failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Synthesis text to WAV failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API void aila_free_samples(float* samples) {
    if (samples) {
        free(samples);
    }
}

AILA_API int aila_decode_mimi_vocoder(
    AilaEngine* engine,
    const int32_t* codes,
    int n_frames,
    float** out_samples,
    int* out_sample_count
) {
    if (!engine || !codes || n_frames <= 0 || !out_samples || !out_sample_count) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        std::vector<int32_t> codes_vec(codes, codes + n_frames * 16);
        std::vector<float> samples;
        
        bool ok = engine->engine.decode_mimi_vocoder(codes_vec, n_frames, samples);
        if (!ok) {
            return AILA_ERR_RUNTIME;
        }

        float* c_arr = (float*)malloc(samples.size() * sizeof(float));
        if (!c_arr) {
            return AILA_ERR_RUNTIME;
        }
        std::copy(samples.begin(), samples.end(), c_arr);

        *out_samples = c_arr;
        *out_sample_count = static_cast<int>(samples.size());

        return AILA_OK;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Decode Mimi Vocoder failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Decode Mimi Vocoder failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API int aila_synthesize(
    AilaEngine* engine,
    const char* text,
    const char* reference_audio_path,
    const char* speaker_name,
    const char* instruct_text,
    const char* language,
    const AilaGenConfig* config,
    const char* output_wav_path
) {
    if (!engine || !text || !output_wav_path) {
        return AILA_ERR_INVALID_ARGUMENT;
    }
    try {
        GenerationConfig cpp_cfg = to_cpp_config(config);

        std::vector<float> samples;
        bool ok = engine->engine.synthesizeSpeech(
            std::string(text),
            reference_audio_path ? std::string(reference_audio_path) : std::string(),
            speaker_name ? std::string(speaker_name) : std::string(),
            instruct_text ? std::string(instruct_text) : std::string(),
            language ? std::string(language) : std::string(),
            cpp_cfg,
            samples
        );

        if (!ok) {
            return AILA_ERR_RUNTIME;
        }

        if (!save_wav(std::string(output_wav_path), samples, 24000)) {
            return AILA_ERR_RUNTIME;
        }

        return AILA_OK;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] aila_synthesize failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] aila_synthesize failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}

// ============================================================
// Streaming TTS API
// ============================================================

struct AilaTTSStream {
    std::thread worker;
    std::mutex mutex;
    bool done = false;
};

AILA_API AilaTTSStream* aila_synthesize_stream(
    AilaEngine* engine,
    const char* text,
    const char* reference_audio_path,
    const char* speaker_name,
    const char* instruct_text,
    const char* language,
    const AilaGenConfig* config,
    AilaAudioCallback callback,
    void* user_data
) {
    if (!engine || !text || !callback) return nullptr;
    auto* stream = new AilaTTSStream();

    GenerationConfig cpp_cfg = to_cpp_config(config);
    stream->worker = engine->engine.synthesizeSpeechStream(
        std::string(text),
        reference_audio_path ? std::string(reference_audio_path) : std::string(),
        speaker_name ? std::string(speaker_name) : std::string(),
        instruct_text ? std::string(instruct_text) : std::string(),
        language ? std::string(language) : std::string(),
        cpp_cfg,
        [callback, user_data](const float* samples, int count) {
            callback(samples, count, user_data);
        }, 4);

    return stream;
}

AILA_API int aila_stream_wait(AilaTTSStream* stream) {
    if (!stream) return AILA_ERR_INVALID_ARGUMENT;
    if (stream->worker.joinable()) {
        stream->worker.join();
        stream->done = true;
    }
    return AILA_OK;
}

AILA_API void aila_stream_destroy(AilaTTSStream* stream) {
    if (!stream) return;
    if (stream->worker.joinable()) stream->worker.join();
    delete stream;
}

AILA_API int aila_extract_speaker_embedding(
    AilaEngine* engine,
    const char* audio_path,
    float** out_embedding,
    int* out_embedding_dim
) {
    if (!engine || !audio_path || !out_embedding || !out_embedding_dim) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        std::vector<float> embedding;
        bool ok = engine->engine.extractSpeakerEmbedding(audio_path, embedding);
        if (!ok) {
            return AILA_ERR_RUNTIME;
        }

        float* c_arr = (float*)malloc(embedding.size() * sizeof(float));
        if (!c_arr) {
            return AILA_ERR_RUNTIME;
        }
        std::copy(embedding.begin(), embedding.end(), c_arr);

        *out_embedding = c_arr;
        *out_embedding_dim = static_cast<int>(embedding.size());

        return AILA_OK;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Extract speaker embedding failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Extract speaker embedding failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API int aila_align(
    AilaEngine* engine,
    const float* audio_samples, int num_samples, int sample_rate,
    const char* text, const char* language,
    AilaAlignedWord** out_words, int* out_count
) {
    if (out_words) *out_words = nullptr;
    if (out_count) *out_count = 0;

    if (!engine || !audio_samples || num_samples <= 0 || !text || !language) {
        return AILA_ERR_INVALID_ARGUMENT;
    }
    if (!out_words || !out_count) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        std::vector<float> samples(audio_samples, audio_samples + num_samples);
        auto result = engine->engine.align(samples, sample_rate,
                                           std::string(text),
                                           std::string(language));

        if (engine->engine.last_error_code() != EngineErrorCode::Ok) {
            return AILA_ERR_RUNTIME;
        }

        int count = static_cast<int>(result.size());
        if (count == 0) {
            *out_count = 0;
            *out_words = nullptr;
            return 0;
        }

        auto* words = static_cast<AilaAlignedWord*>(malloc(count * sizeof(AilaAlignedWord)));
        if (!words) {
            return AILA_ERR_RUNTIME;
        }
        memset(words, 0, count * sizeof(AilaAlignedWord));

        for (int i = 0; i < count; ++i) {
            char* txt = static_cast<char*>(malloc(result[i].text.size() + 1));
            if (txt) {
                memcpy(txt, result[i].text.c_str(), result[i].text.size() + 1);
            }
            words[i].text = txt;
            words[i].start_ms = result[i].start_ms;
            words[i].end_ms = result[i].end_ms;
        }

        *out_words = words;
        *out_count = count;
        return 0;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] Align failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] Align failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}

AILA_API void aila_free_aligned_words(AilaAlignedWord* words, int count) {
    if (words) {
        for (int i = 0; i < count; ++i) {
            free(const_cast<char*>(words[i].text));
        }
        free(words);
    }
}

AILA_API int aila_align_words(
    AilaEngine* engine,
    const float* audio_samples, int num_samples, int sample_rate,
    const char* const* words, int num_words,
    AilaAlignedWord** out_words, int* out_count
) {
    if (out_words) *out_words = nullptr;
    if (out_count) *out_count = 0;

    if (!engine || !audio_samples || num_samples <= 0 || !words || num_words <= 0) {
        return AILA_ERR_INVALID_ARGUMENT;
    }
    if (!out_words || !out_count) {
        return AILA_ERR_INVALID_ARGUMENT;
    }

    try {
        std::vector<float> samples(audio_samples, audio_samples + num_samples);
        std::vector<std::string> word_list;
        word_list.reserve(num_words);
        for (int i = 0; i < num_words; ++i) {
            word_list.emplace_back(words[i] ? words[i] : "");
        }

        auto result = engine->engine.align_words(samples, sample_rate, word_list);

        if (engine->engine.last_error_code() != EngineErrorCode::Ok) {
            return AILA_ERR_RUNTIME;
        }

        int count = static_cast<int>(result.size());
        if (count == 0) {
            *out_count = 0;
            *out_words = nullptr;
            return 0;
        }

        auto* out = static_cast<AilaAlignedWord*>(malloc(count * sizeof(AilaAlignedWord)));
        if (!out) return AILA_ERR_RUNTIME;
        memset(out, 0, count * sizeof(AilaAlignedWord));

        for (int i = 0; i < count; ++i) {
            char* txt = static_cast<char*>(malloc(result[i].text.size() + 1));
            if (txt) memcpy(txt, result[i].text.c_str(), result[i].text.size() + 1);
            out[i].text = txt;
            out[i].start_ms = result[i].start_ms;
            out[i].end_ms = result[i].end_ms;
        }

        *out_words = out;
        *out_count = count;
        return 0;
    } catch (const std::exception& e) {
        AILA_LOG_ERROR("[C-API] align_words failed: %s", e.what());
        return AILA_ERR_RUNTIME;
    } catch (...) {
        AILA_LOG_ERROR("[C-API] align_words failed: unknown exception");
        return AILA_ERR_RUNTIME;
    }
}
