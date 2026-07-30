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
    ALIA_ERR_CALLBACK = 7,
    ALIA_ERR_TIMEOUT = 8
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

typedef struct AliaContextConfig {
    const char* asr_model_dir;
    const char* vlm_4b_model_dir;
    const char* vlm_4b_lora_dir;
    const char* vlm_0_8b_model_dir;
    const char* tts_model_dir;
    const char* tts_reference_audio_path;
    int max_seq_len;
} AliaContextConfig;

typedef enum AliaAsyncState {
    ALIA_ASYNC_IDLE = 0,
    ALIA_ASYNC_RUNNING = 1,
    ALIA_ASYNC_ABORTING = 2,
    ALIA_ASYNC_ABORTED = 3,
    ALIA_ASYNC_COMPLETED = 4,
    ALIA_ASYNC_FAILED = 5
} AliaAsyncState;

typedef enum AliaSpeculativeEndpointState {
    ALIA_SPECULATIVE_ENDPOINT_IDLE = 0,
    ALIA_SPECULATIVE_ENDPOINT_LISTENING = 1,
    ALIA_SPECULATIVE_ENDPOINT_PREFILLING = 2,
    ALIA_SPECULATIVE_ENDPOINT_READY = 3,
    ALIA_SPECULATIVE_ENDPOINT_INVALIDATED = 4,
    ALIA_SPECULATIVE_ENDPOINT_COMMITTING = 5
} AliaSpeculativeEndpointState;

typedef struct AliaSpeculativeEndpointMetrics {
    int state;
    int enabled;
    int trigger_count;
    int cold_trigger_count;
    int resume_count;
    int stale_completion_count;
    int last_silence_frames;
    int candidate_prefill_rc;
    int candidate_prefill_tokens;
    int final_prefill_rc;
    int final_prefill_tokens;
    int final_reused_tokens;
    int final_suffix_tokens;
    int commit_prefill_hit;
    int commit_accepted;
    int final_asr_reused_candidate;
    int64_t candidate_silence_ms;
    int64_t candidate_asr_ms;
    int64_t candidate_prefill_ms;
    int64_t commit_wait_ms;
    int64_t final_asr_ms;
    int64_t final_prefill_ms;
} AliaSpeculativeEndpointMetrics;

typedef struct AliaInterruptionMetrics {
    int64_t requested_played_audio_samples;
    int64_t retained_audio_samples;
    int completed_segments;
    int discarded_segments;
    int64_t abort_to_quiescent_ms;
    int late_callback_count;
    int interruption_chain_size;
} AliaInterruptionMetrics;

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
ALIA_API int alia_context_init_ex(AliaContext** out_ctx, const AliaContextConfig* config);

ALIA_API void alia_context_destroy(AliaContext* ctx);
ALIA_API int alia_abort_inference(AliaContext* ctx, int pipeline_mask);
ALIA_API int alia_get_last_error(AliaContext* ctx, int pipeline_mask, char** out_error);
/*
 * Returns a malloc-owned JSON snapshot of the latest ASR/foreground/TTS,
 * speculative endpoint, and background metrics. Release with alia_free_string().
 */
ALIA_API int alia_get_last_turn_metrics_json(AliaContext* ctx, char** out_json);
ALIA_API int alia_vlm_rollback_kv_cache(AliaContext* ctx, int rollback_tokens);
ALIA_API int alia_vlm_prefill_asr_text(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text);
ALIA_API int alia_speculative_endpoint_begin(AliaContext* ctx);
ALIA_API int alia_speculative_endpoint_observe_vad(
    AliaContext* ctx,
    float speech_probability);
ALIA_API int alia_speculative_endpoint_commit(
    AliaContext* ctx,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data);
ALIA_API int alia_speculative_endpoint_cancel(AliaContext* ctx);
ALIA_API int alia_speculative_endpoint_get_metrics(
    AliaContext* ctx,
    AliaSpeculativeEndpointMetrics* out_metrics);
ALIA_API int alia_request_turn_interruption(
    AliaContext* ctx,
    int64_t played_audio_samples);
ALIA_API int alia_get_last_interruption_result(
    AliaContext* ctx,
    char** out_previous_user_text,
    char** out_heard_assistant_text,
    AliaInterruptionMetrics* out_metrics);
ALIA_API int alia_start_speculative_conversation_turn(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text,
    const AliaGenConfig* config);
ALIA_API int alia_commit_speculative_conversation_turn(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data);
ALIA_API int alia_asr_feed_audio(AliaContext* ctx, const float* samples, int sample_count);
ALIA_API void alia_asr_reset(AliaContext* ctx);

/* Frees strings returned by alia_asr_get_text(). Host allocators must not be used. */
ALIA_API void alia_free_string(char* s);

/* Returns malloc-owned UTF-8 strings; release each non-null output with alia_free_string(). */
ALIA_API int alia_asr_get_text(AliaContext* ctx, char** out_stable, char** out_partial);
/*
 * Returns cached text unless enough new audio has arrived to justify another partial decode.
 * Use alia_asr_get_text() for final/forced turn text.
 */
ALIA_API int alia_asr_get_partial_text(AliaContext* ctx, char** out_stable, char** out_partial);
ALIA_API int alia_foreground_get_state(AliaContext* ctx, int* out_state);
ALIA_API int alia_foreground_wait(AliaContext* ctx, int timeout_ms, int* out_state);
ALIA_API int alia_foreground_get_last_result(
    AliaContext* ctx,
    char** out_user_text,
    char** out_assistant_text,
    char** out_action_tags_json);
ALIA_API int alia_background_get_state(AliaContext* ctx, int* out_state);
ALIA_API int alia_background_wait(AliaContext* ctx, int timeout_ms, int* out_state);
ALIA_API int alia_background_get_last_result(AliaContext* ctx, char** out_result_json);
ALIA_API void alia_register_background_callback(AliaContext* ctx, AliaBackgroundResultCallback callback);
ALIA_API void alia_register_background_callback_ex(
    AliaContext* ctx,
    AliaBackgroundResultCallback callback,
    void* user_data);
ALIA_API int alia_trigger_background_processing(AliaContext* ctx, const char* chat_turn_text);
ALIA_API int alia_start_conversation_turn(
    AliaContext* ctx,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data);

#endif /* ALIA_API_H */
