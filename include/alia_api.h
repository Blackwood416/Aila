/*
 * alia_api.h - Alia-specific Aila runtime C API
 *
 * This API is the customized ABI consumed by Alia Host. It is intentionally
 * separate from the generic one-model Aila API during the migration.
 */

#ifndef ALIA_API_H
#define ALIA_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define ALIA_EXTERN extern "C"
#else
#define ALIA_EXTERN extern
#endif

#if defined(_WIN32) || defined(_WIN64)
    #ifdef AILA_BUILDING_DLL
        #define ALIA_API ALIA_EXTERN __declspec(dllexport)
    #elif defined(AILA_STATIC_LIB)
        #define ALIA_API ALIA_EXTERN
    #else
        #define ALIA_API ALIA_EXTERN __declspec(dllimport)
    #endif
#else
    #define ALIA_API ALIA_EXTERN __attribute__((visibility("default")))
#endif

typedef struct AliaContext AliaContext;

typedef enum AliaErrorCode {
    ALIA_OK = 0,
    ALIA_ERR_INVALID_ARGUMENT = 1,
    ALIA_ERR_INVALID_STATE = 2,
    ALIA_ERR_MODEL_LOAD = 3,
    ALIA_ERR_RUNTIME = 4,
    ALIA_ERR_ABORTED = 5,
    ALIA_ERR_CONTEXT_OVERFLOW = 6,
    ALIA_ERR_CALLBACK = 7
} AliaErrorCode;

typedef enum AliaPipelineMask {
    ALIA_PIPELINE_ASR = 1 << 0,
    ALIA_PIPELINE_VLM_FOREGROUND = 1 << 1,
    ALIA_PIPELINE_TTS = 1 << 2,
    ALIA_PIPELINE_VLM_BACKGROUND = 1 << 3,
    ALIA_PIPELINE_ALL = 0xFFFF
} AliaPipelineMask;

typedef struct AliaGenConfig {
    float temperature;
    float top_p;
    int max_tokens;
} AliaGenConfig;

typedef int (*AliaToolCallCallback)(
    const char* tool_json,
    char* out_result_buf,
    int max_result_len,
    void* user_data);

typedef void (*AliaAudioCallback)(
    const float* samples,
    int sample_count,
    void* user_data);

typedef void (*AliaBackgroundResultCallback)(
    const char* extracted_json,
    void* user_data);

ALIA_API int alia_context_init(
    AliaContext** out_ctx,
    const char* asr_model_dir,
    const char* vlm_4b_model_dir,
    const char* vlm_0_8b_model_dir,
    const char* tts_model_dir,
    int max_seq_len);

ALIA_API void alia_context_destroy(AliaContext* ctx);
ALIA_API int alia_abort_inference(AliaContext* ctx, int pipeline_mask);
ALIA_API int alia_vlm_rollback_kv_cache(AliaContext* ctx, int rollback_tokens);
ALIA_API int alia_asr_feed_audio(AliaContext* ctx, const float* samples, int sample_count);
ALIA_API void alia_asr_reset(AliaContext* ctx);
ALIA_API int alia_asr_get_text(AliaContext* ctx, char** out_stable, char** out_partial);
ALIA_API void alia_register_background_callback(AliaContext* ctx, AliaBackgroundResultCallback callback);
ALIA_API int alia_trigger_background_processing(AliaContext* ctx, const char* chat_turn_text);
ALIA_API int alia_start_conversation_turn(
    AliaContext* ctx,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data);

#endif /* ALIA_API_H */
