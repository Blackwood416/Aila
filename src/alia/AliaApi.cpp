#include "AliaContext.hpp"

#include "../models/IModelBackend.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string safe_string(const char* value) {
    return value ? std::string(value) : std::string();
}

char* duplicate_c_string(const std::string& value) {
    char* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    const char* hex = "0123456789abcdef";
                    out << "\\u00" << hex[(ch >> 4) & 0x0F] << hex[ch & 0x0F];
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

std::string string_array_to_json(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]";
    return out.str();
}

int map_foreground_state(aila::alia::ForegroundTurnState state) {
    switch (state) {
        case aila::alia::ForegroundTurnState::Idle:
            return ALIA_ASYNC_IDLE;
        case aila::alia::ForegroundTurnState::Running:
            return ALIA_ASYNC_RUNNING;
        case aila::alia::ForegroundTurnState::Aborting:
            return ALIA_ASYNC_ABORTING;
        case aila::alia::ForegroundTurnState::Aborted:
            return ALIA_ASYNC_ABORTED;
        case aila::alia::ForegroundTurnState::Completed:
            return ALIA_ASYNC_COMPLETED;
        case aila::alia::ForegroundTurnState::Failed:
            return ALIA_ASYNC_FAILED;
    }
    return ALIA_ASYNC_FAILED;
}

int map_background_state(aila::alia::BackgroundJobState state) {
    switch (state) {
        case aila::alia::BackgroundJobState::Idle:
            return ALIA_ASYNC_IDLE;
        case aila::alia::BackgroundJobState::Running:
            return ALIA_ASYNC_RUNNING;
        case aila::alia::BackgroundJobState::Aborting:
            return ALIA_ASYNC_ABORTING;
        case aila::alia::BackgroundJobState::Completed:
            return ALIA_ASYNC_COMPLETED;
        case aila::alia::BackgroundJobState::Failed:
            return ALIA_ASYNC_FAILED;
    }
    return ALIA_ASYNC_FAILED;
}

bool set_process_env(const char* name, const std::string& value) {
#if defined(_WIN32) || defined(_WIN64)
    return _putenv_s(name, value.c_str()) == 0;
#else
    return setenv(name, value.c_str(), 1) == 0;
#endif
}

int init_context_from_config(AliaContext** out_ctx, const AliaContextConfig& config) {
    if (!out_ctx) {
        return ALIA_ERR_INVALID_ARGUMENT;
    }

    *out_ctx = nullptr;
    if (config.max_seq_len <= 0) {
        return ALIA_ERR_INVALID_ARGUMENT;
    }

    if (config.tts_reference_audio_path && config.tts_reference_audio_path[0] != '\0' &&
        !set_process_env("AILA_TTS_REF_AUDIO", config.tts_reference_audio_path)) {
        return ALIA_ERR_RUNTIME;
    }

    auto ctx = std::make_unique<AliaContext>(config.max_seq_len);
    ctx->asr_model_dir = safe_string(config.asr_model_dir);
    ctx->vlm_4b_model_dir = safe_string(config.vlm_4b_model_dir);
    ctx->vlm_4b_lora_dir = safe_string(config.vlm_4b_lora_dir);
    ctx->vlm_0_8b_model_dir = safe_string(config.vlm_0_8b_model_dir);
    ctx->tts_model_dir = safe_string(config.tts_model_dir);
    ctx->configure_model_slots();
    if (!ctx->load_model_slots()) {
        return ALIA_ERR_MODEL_LOAD;
    }
    *out_ctx = ctx.release();
    return ALIA_OK;
}

template <typename Fn>
int guarded_alia_call(Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const ModelBackendCancelled&) {
        return ALIA_ERR_ABORTED;
    } catch (const std::bad_alloc&) {
        return ALIA_ERR_RUNTIME;
    } catch (const std::exception&) {
        return ALIA_ERR_RUNTIME;
    } catch (...) {
        return ALIA_ERR_RUNTIME;
    }
}

template <typename Fn>
void guarded_alia_void(Fn&& fn) noexcept {
    try {
        fn();
    } catch (...) {
    }
}

}  // namespace

ALIA_API int alia_context_init(
    AliaContext** out_ctx,
    const char* asr_model_dir,
    const char* vlm_4b_model_dir,
    const char* vlm_0_8b_model_dir,
    const char* tts_model_dir,
    int max_seq_len) {
    return guarded_alia_call([&]() -> int {
        AliaContextConfig config;
        config.asr_model_dir = asr_model_dir;
        config.vlm_4b_model_dir = vlm_4b_model_dir;
        config.vlm_4b_lora_dir = nullptr;
        config.vlm_0_8b_model_dir = vlm_0_8b_model_dir;
        config.tts_model_dir = tts_model_dir;
        config.tts_reference_audio_path = nullptr;
        config.max_seq_len = max_seq_len;
        return init_context_from_config(out_ctx, config);
    });
}

ALIA_API int alia_context_init_ex(AliaContext** out_ctx, const AliaContextConfig* config) {
    return guarded_alia_call([&]() -> int {
        if (!config) {
            if (out_ctx) {
                *out_ctx = nullptr;
            }
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        return init_context_from_config(out_ctx, *config);
    });
}

ALIA_API void alia_context_destroy(AliaContext* ctx) {
    guarded_alia_void([&]() {
        delete ctx;
    });
}

ALIA_API int alia_abort_inference(AliaContext* ctx, int pipeline_mask) {
    return guarded_alia_call([&]() -> int {
        if (!ctx) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        ctx->abort_mask.fetch_or(pipeline_mask, std::memory_order_relaxed);
        if (ctx->speculative_endpoint &&
            (pipeline_mask == ALIA_PIPELINE_ALL ||
             (pipeline_mask & (ALIA_PIPELINE_ASR |
                               ALIA_PIPELINE_VLM_FOREGROUND |
                               ALIA_PIPELINE_TTS)) != 0)) {
            ctx->speculative_endpoint->cancel();
        }
        if (ctx->foreground_pipeline &&
            (pipeline_mask == ALIA_PIPELINE_ALL ||
             (pipeline_mask & (ALIA_PIPELINE_VLM_FOREGROUND | ALIA_PIPELINE_TTS)) != 0)) {
            ctx->foreground_pipeline->request_abort();
        }
        if (ctx->background_pipeline &&
            (pipeline_mask == ALIA_PIPELINE_ALL ||
             (pipeline_mask & ALIA_PIPELINE_VLM_BACKGROUND) != 0)) {
            ctx->background_pipeline->request_abort();
        }
        return ALIA_OK;
    });
}

ALIA_API int alia_get_last_error(AliaContext* ctx, int pipeline_mask, char** out_error) {
    return guarded_alia_call([&]() -> int {
        if (out_error) {
            *out_error = nullptr;
        }
        if (!ctx || !out_error) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }

        std::ostringstream message;
        auto append = [&](const char* prefix, const std::string& value) {
            if (value.empty()) {
                return;
            }
            if (message.tellp() > 0) {
                message << "\n";
            }
            message << prefix << value;
        };

        append("", ctx->last_error);
        if (ctx->foreground_pipeline &&
            (pipeline_mask == ALIA_PIPELINE_ALL ||
             (pipeline_mask & (ALIA_PIPELINE_VLM_FOREGROUND | ALIA_PIPELINE_TTS)) != 0)) {
            append("foreground: ", ctx->foreground_pipeline->last_error_text());
        }
        if (ctx->background_pipeline &&
            (pipeline_mask == ALIA_PIPELINE_ALL ||
             (pipeline_mask & ALIA_PIPELINE_VLM_BACKGROUND) != 0)) {
            append("background: ", ctx->background_pipeline->last_error_text());
        }

        *out_error = duplicate_c_string(message.str());
        return *out_error ? ALIA_OK : ALIA_ERR_RUNTIME;
    });
}

ALIA_API int alia_vlm_rollback_kv_cache(AliaContext* ctx, int rollback_tokens) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || rollback_tokens < 0) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (rollback_tokens == 0) {
            return ALIA_OK;
        }
        if (!ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        return ctx->foreground_pipeline->rollback_kv_cache(rollback_tokens);
    });
}

ALIA_API int alia_vlm_prefill_asr_text(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text) {
    return guarded_alia_call([&]() -> int {
        if (!ctx) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        return ctx->foreground_pipeline->prefill_asr_text(
            safe_string(stable_text),
            safe_string(partial_text));
    });
}

ALIA_API int alia_speculative_endpoint_begin(AliaContext* ctx) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->speculative_endpoint) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        ctx->abort_mask.store(0, std::memory_order_relaxed);
        return ctx->speculative_endpoint->begin();
    });
}

ALIA_API int alia_speculative_endpoint_observe_vad(
    AliaContext* ctx,
    float speech_probability) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->speculative_endpoint) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        return ctx->speculative_endpoint->observe_vad(speech_probability);
    });
}

ALIA_API int alia_speculative_endpoint_commit(
    AliaContext* ctx,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->speculative_endpoint) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (config && !aila::alia::is_valid_generation_config(*config)) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        ctx->abort_mask.store(0, std::memory_order_relaxed);
        return ctx->speculative_endpoint->commit(
            config, tool_cb, audio_cb, user_data);
    });
}

ALIA_API int alia_speculative_endpoint_cancel(AliaContext* ctx) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->speculative_endpoint) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        ctx->speculative_endpoint->cancel();
        return ALIA_OK;
    });
}

ALIA_API int alia_speculative_endpoint_get_metrics(
    AliaContext* ctx,
    AliaSpeculativeEndpointMetrics* out_metrics) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->speculative_endpoint || !out_metrics) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        *out_metrics = ctx->speculative_endpoint->metrics();
        return ALIA_OK;
    });
}

ALIA_API int alia_start_speculative_conversation_turn(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text,
    const AliaGenConfig* config) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        return ctx->foreground_pipeline->start_speculative_turn(
            safe_string(stable_text),
            safe_string(partial_text),
            config)
            ? ALIA_OK
            : ALIA_ERR_INVALID_STATE;
    });
}

ALIA_API int alia_commit_speculative_conversation_turn(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        return ctx->foreground_pipeline->commit_speculative_turn(
            safe_string(stable_text),
            safe_string(partial_text),
            config,
            tool_cb,
            audio_cb,
            user_data)
            ? ALIA_OK
            : ALIA_ERR_INVALID_STATE;
    });
}

ALIA_API int alia_asr_feed_audio(AliaContext* ctx, const float* samples, int sample_count) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !ctx->asr_pipeline || !samples || sample_count <= 0) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->asr_pipeline->feed_audio(samples, sample_count)) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        return ALIA_OK;
    });
}

ALIA_API void alia_asr_reset(AliaContext* ctx) {
    guarded_alia_void([&]() {
        if (!ctx) {
            return;
        }

        if (ctx->asr_pipeline) {
            ctx->asr_pipeline->reset();
        }
    });
}

ALIA_API void alia_free_string(char* s) {
    guarded_alia_void([&]() {
        std::free(s);
    });
}

ALIA_API int alia_asr_get_text(AliaContext* ctx, char** out_stable, char** out_partial) {
    return guarded_alia_call([&]() -> int {
        if (out_stable) {
            *out_stable = nullptr;
        }
        if (out_partial) {
            *out_partial = nullptr;
        }
        if (!ctx) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }

        if (!ctx->asr_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }

        std::string stable;
        std::string partial;
        ctx->asr_pipeline->get_text(stable, partial);
        if (out_stable) {
            *out_stable = duplicate_c_string(stable);
            if (!*out_stable) {
                return ALIA_ERR_RUNTIME;
            }
        }
        if (out_partial) {
            *out_partial = duplicate_c_string(partial);
            if (!*out_partial) {
                if (out_stable && *out_stable) {
                    alia_free_string(*out_stable);
                    *out_stable = nullptr;
                }
                return ALIA_ERR_RUNTIME;
            }
        }
        return ALIA_OK;
    });
}

ALIA_API int alia_asr_get_partial_text(AliaContext* ctx, char** out_stable, char** out_partial) {
    return guarded_alia_call([&]() -> int {
        if (out_stable) {
            *out_stable = nullptr;
        }
        if (out_partial) {
            *out_partial = nullptr;
        }
        if (!ctx) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }

        if (!ctx->asr_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }

        std::string stable;
        std::string partial;
        ctx->asr_pipeline->get_partial_text(stable, partial);
        if (out_stable) {
            *out_stable = duplicate_c_string(stable);
            if (!*out_stable) {
                return ALIA_ERR_RUNTIME;
            }
        }
        if (out_partial) {
            *out_partial = duplicate_c_string(partial);
            if (!*out_partial) {
                if (out_stable && *out_stable) {
                    alia_free_string(*out_stable);
                    *out_stable = nullptr;
                }
                return ALIA_ERR_RUNTIME;
            }
        }
        return ALIA_OK;
    });
}

ALIA_API int alia_foreground_get_state(AliaContext* ctx, int* out_state) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !out_state) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        *out_state = map_foreground_state(ctx->foreground_pipeline->state());
        return ALIA_OK;
    });
}

ALIA_API int alia_foreground_wait(AliaContext* ctx, int timeout_ms, int* out_state) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || timeout_ms < 0) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        const bool idle = ctx->foreground_pipeline->wait_until_idle_for(
            std::chrono::milliseconds(timeout_ms));
        if (out_state) {
            *out_state = map_foreground_state(ctx->foreground_pipeline->state());
        }
        return idle ? ALIA_OK : ALIA_ERR_TIMEOUT;
    });
}

ALIA_API int alia_foreground_get_last_result(
    AliaContext* ctx,
    char** out_user_text,
    char** out_assistant_text,
    char** out_action_tags_json) {
    return guarded_alia_call([&]() -> int {
        if (out_user_text) {
            *out_user_text = nullptr;
        }
        if (out_assistant_text) {
            *out_assistant_text = nullptr;
        }
        if (out_action_tags_json) {
            *out_action_tags_json = nullptr;
        }
        if (!ctx) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }

        if (out_user_text) {
            *out_user_text = duplicate_c_string(ctx->foreground_pipeline->last_user_text());
            if (!*out_user_text) {
                return ALIA_ERR_RUNTIME;
            }
        }
        if (out_assistant_text) {
            *out_assistant_text = duplicate_c_string(
                ctx->foreground_pipeline->last_assistant_text());
            if (!*out_assistant_text) {
                if (out_user_text && *out_user_text) {
                    alia_free_string(*out_user_text);
                    *out_user_text = nullptr;
                }
                return ALIA_ERR_RUNTIME;
            }
        }
        if (out_action_tags_json) {
            const std::string action_tags_json =
                string_array_to_json(ctx->foreground_pipeline->last_action_tags());
            *out_action_tags_json = duplicate_c_string(action_tags_json);
            if (!*out_action_tags_json) {
                if (out_user_text && *out_user_text) {
                    alia_free_string(*out_user_text);
                    *out_user_text = nullptr;
                }
                if (out_assistant_text && *out_assistant_text) {
                    alia_free_string(*out_assistant_text);
                    *out_assistant_text = nullptr;
                }
                return ALIA_ERR_RUNTIME;
            }
        }
        return ALIA_OK;
    });
}

ALIA_API int alia_background_get_state(AliaContext* ctx, int* out_state) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !out_state) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->background_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        *out_state = map_background_state(ctx->background_pipeline->state());
        return ALIA_OK;
    });
}

ALIA_API int alia_background_wait(AliaContext* ctx, int timeout_ms, int* out_state) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || timeout_ms < 0) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->background_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        const bool idle = ctx->background_pipeline->wait_until_idle_for(
            std::chrono::milliseconds(timeout_ms));
        if (out_state) {
            *out_state = map_background_state(ctx->background_pipeline->state());
        }
        return idle ? ALIA_OK : ALIA_ERR_TIMEOUT;
    });
}

ALIA_API int alia_background_get_last_result(AliaContext* ctx, char** out_result_json) {
    return guarded_alia_call([&]() -> int {
        if (out_result_json) {
            *out_result_json = nullptr;
        }
        if (!ctx || !out_result_json) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->background_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        *out_result_json = duplicate_c_string(ctx->background_pipeline->last_result_json());
        return *out_result_json ? ALIA_OK : ALIA_ERR_RUNTIME;
    });
}

ALIA_API void alia_register_background_callback(
    AliaContext* ctx,
    AliaBackgroundResultCallback callback) {
    alia_register_background_callback_ex(ctx, callback, nullptr);
}

ALIA_API void alia_register_background_callback_ex(
    AliaContext* ctx,
    AliaBackgroundResultCallback callback,
    void* user_data) {
    guarded_alia_void([&]() {
        if (!ctx) {
            return;
        }

        if (ctx->background_pipeline) {
            ctx->background_pipeline->register_callback(callback, user_data);
        }
    });
}

ALIA_API int alia_trigger_background_processing(AliaContext* ctx, const char* chat_turn_text) {
    return guarded_alia_call([&]() -> int {
        if (!ctx || !chat_turn_text) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->background_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        return ctx->background_pipeline->trigger(chat_turn_text)
            ? ALIA_OK
            : ALIA_ERR_INVALID_STATE;
    });
}

ALIA_API int alia_start_conversation_turn(
    AliaContext* ctx,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data) {
    return guarded_alia_call([&]() -> int {
        if (!ctx) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (config && !aila::alia::is_valid_generation_config(*config)) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }
        if (!ctx->foreground_pipeline) {
            return ALIA_ERR_INVALID_STATE;
        }
        ctx->abort_mask.store(0, std::memory_order_relaxed);
        return ctx->foreground_pipeline->start_turn(config, tool_cb, audio_cb, user_data)
            ? ALIA_OK
            : ALIA_ERR_INVALID_STATE;
    });
}
