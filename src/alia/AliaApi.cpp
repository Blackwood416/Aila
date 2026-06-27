#include "AliaContext.hpp"

#include "../models/IModelBackend.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>

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
        if (!out_ctx) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }

        *out_ctx = nullptr;
        if (max_seq_len <= 0) {
            return ALIA_ERR_INVALID_ARGUMENT;
        }

        auto ctx = std::make_unique<AliaContext>(max_seq_len);
        ctx->asr_model_dir = safe_string(asr_model_dir);
        ctx->vlm_4b_model_dir = safe_string(vlm_4b_model_dir);
        ctx->vlm_0_8b_model_dir = safe_string(vlm_0_8b_model_dir);
        ctx->tts_model_dir = safe_string(tts_model_dir);
        ctx->configure_model_slots();
        if (!ctx->load_model_slots()) {
            return ALIA_ERR_MODEL_LOAD;
        }
        *out_ctx = ctx.release();
        return ALIA_OK;
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

ALIA_API void alia_register_background_callback(
    AliaContext* ctx,
    AliaBackgroundResultCallback callback) {
    guarded_alia_void([&]() {
        if (!ctx) {
            return;
        }

        if (ctx->background_pipeline) {
            ctx->background_pipeline->register_callback(callback);
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
