/*
 * aila_api.h - Aila Inference Engine C API
 *
 * This is the public C interface for the Aila inference engine library.
 * Use this header to integrate Aila into any language that supports C FFI
 * (Python ctypes, C# P/Invoke, Java JNI, Go cgo, Rust FFI, etc.)
 *
 * Thread Safety: AilaEngine instances are NOT thread-safe.
 *                Create one per thread or synchronize access externally.
 */

#ifndef AILA_API_H
#define AILA_API_H

#include <stddef.h>
#include <stdint.h>

/* -------------- Platform export macros -------------- */
#ifdef __cplusplus
#define AILA_EXTERN extern "C"
#else
#define AILA_EXTERN extern
#endif

#if defined(_WIN32) || defined(_WIN64)
    #ifdef AILA_BUILDING_DLL
        #define AILA_API AILA_EXTERN __declspec(dllexport)
    #elif defined(AILA_STATIC_LIB)
        #define AILA_API AILA_EXTERN
    #else
        #define AILA_API AILA_EXTERN __declspec(dllimport)
    #endif
#else
    #define AILA_API AILA_EXTERN __attribute__((visibility("default")))
#endif

/* -------------- Opaque handle -------------- */
typedef struct AilaEngine AilaEngine;

typedef enum AilaErrorCode {
    AILA_OK = 0,
    AILA_ERR_INVALID_ARGUMENT = 1,
    AILA_ERR_TEMPLATE = 2,
    AILA_ERR_JSON_PARSE = 3,
    AILA_ERR_VISION_NOT_ENABLED = 4,
    AILA_ERR_CONTEXT_OVERFLOW = 5,
    AILA_ERR_RUNTIME = 6,
    AILA_ERR_MODEL_CAPABILITY = 7
} AilaErrorCode;

typedef enum AilaPixelFormat {
    AILA_PIXEL_RGB8 = 0,
    AILA_PIXEL_BGR8 = 1,
    AILA_PIXEL_RGBA8 = 2,
    AILA_PIXEL_BGRA8 = 3
} AilaPixelFormat;

typedef struct AilaDetectionConfig {
    uint32_t struct_size;
    float confidence_threshold;
    int max_detections;
    int reserved[8];
} AilaDetectionConfig;

typedef struct AilaDetection {
    uint32_t struct_size;
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
    const char* class_name;
    int reserved[4];
} AilaDetection;

/* -------------- Generation configuration -------------- */
typedef struct AilaGenConfig {
    int   max_new_tokens;       /* default: 512    */
    float temperature;          /* default: 0.6    */
    int   top_k;                /* default: 20     */
    float top_p;                /* default: 0.95   */
    float repetition_penalty;   /* default: 1.0    */
    float presence_penalty;     /* default: 0.0    */
    float frequency_penalty;    /* default: 0.0    */
    int   do_sample;            /* 0=greedy, 1=sampling */
    int   decode_chunk_size;    /* default: 12     */
    int   stream_chunk_size;    /* default: 4      */
} AilaGenConfig;

typedef struct AilaGenConfigV2 {
    uint32_t struct_size;       /* set to sizeof(AilaGenConfigV2) */
    int   max_new_tokens;       /* default: 512    */
    float temperature;          /* default: 0.6    */
    int   top_k;                /* default: 20     */
    float top_p;                /* default: 0.95   */
    float repetition_penalty;   /* default: 1.0    */
    float presence_penalty;     /* default: 0.0    */
    float frequency_penalty;    /* default: 0.0    */
    int   do_sample;            /* 0=greedy, 1=sampling */
    int   decode_chunk_size;    /* default: 12     */
    int   stream_chunk_size;    /* default: 4      */
    int   thinking_budget_tokens; /* -1=disabled, 0=no-think, >0 budget */
    uint64_t sampling_seed;     /* valid when use_fixed_seed != 0 */
    int   use_fixed_seed;       /* 0=random/default, 1=use sampling_seed */
    int   reserved[8];          /* must be zero */
} AilaGenConfigV2;

typedef enum AilaChatStreamEventType {
    AILA_CHAT_STREAM_REASONING_DELTA = 0,
    AILA_CHAT_STREAM_CONTENT_DELTA = 1,
    AILA_CHAT_STREAM_TOOL_CALL_DELTA = 2,
    AILA_CHAT_STREAM_WARNING = 3,
    AILA_CHAT_STREAM_FINAL = 4
} AilaChatStreamEventType;

typedef struct AilaChatStreamEvent {
    uint32_t struct_size;
    int type;
    const char* text;
    const char* tool_call_id;
    const char* tool_name;
    const char* arguments_delta;
    const char* finish_reason;
    const char* warnings_json;
    const char* tool_calls_json;
} AilaChatStreamEvent;

/* -------------- Callback types -------------- */

/**
 * Token streaming callback.
 * @param token_text  UTF-8 encoded token text (null-terminated, valid only during call)
 * @param user_data   Opaque pointer passed to aila_generate_stream
 * @return 0 to continue, non-zero to abort generation
 */
typedef int (*AilaTokenCallback)(const char* token_text, void* user_data);

/**
 * Structured chat streaming callback.
 * All string pointers in event are UTF-8 encoded and valid only during this call.
 * @param event      Structured chat stream event
 * @param user_data  Opaque pointer passed to aila_generate_chat_json_stream_ex
 * @return 0 to continue, non-zero to abort generation
 */
typedef int (*AilaChatStreamCallback)(const AilaChatStreamEvent* event, void* user_data);

/**
 * Log callback.
 * @param level  0=Debug, 1=Info, 2=Warning, 3=Error
 * @param message  UTF-8 log message (null-terminated)
 * @param user_data Opaque pointer passed to aila_set_log_callback
 */
typedef void (*AilaLogCallback)(int level, const char* message, void* user_data);

/* -------------- API functions -------------- */

/** Get library version string (e.g. "0.2.0"). The returned pointer is static. */
AILA_API const char* aila_version(void);

/**
 * Create a new engine instance.
 * @return Engine handle, or NULL on allocation failure.
 */
AILA_API AilaEngine* aila_engine_create(void);

/**
 * Initialize the engine: loads model weights and tokenizer.
 * @param engine      Handle from aila_engine_create
 * @param model_dir   Path to model directory (single-file or sharded safetensors + tokenizer files)
 * @param max_seq_len Maximum context window length (e.g. 4096)
 * @return 0 on success, non-zero on failure
 */
AILA_API int aila_engine_init(AilaEngine* engine, const char* model_dir, int max_seq_len);

/**
 * Destroy engine and free all resources.
 * @param engine  Handle to destroy. NULL is safe.
 */
AILA_API void aila_engine_destroy(AilaEngine* engine);

/**
 * Get default generation config with sensible defaults.
 */
AILA_API AilaGenConfig aila_default_gen_config(void);

/**
 * Get ABI-safe v2 generation config defaults.
 * Set struct_size to sizeof(AilaGenConfigV2) before passing custom configs.
 */
AILA_API AilaGenConfigV2 aila_default_gen_config_v2(void);

/**
 * Generate response (blocking, non-streaming).
 * @param engine   Initialized engine handle
 * @param prompt   User message (UTF-8 null-terminated)
 * @param config   Generation config (NULL for defaults)
 * @return  Newly allocated UTF-8 string. Caller must free with aila_free_string().
 *          Returns NULL on error.
 */
AILA_API char* aila_generate(AilaEngine* engine, const char* prompt, const AilaGenConfig* config);

/**
 * Generate response from OpenAI-style messages JSON (blocking, non-streaming).
 * @param engine         Initialized engine handle
 * @param messages_json  UTF-8 JSON array string. Each item should contain role/content.
 * @param config         Generation config (NULL for defaults)
 * @return Newly allocated UTF-8 string. Caller must free with aila_free_string().
 *         Returns NULL on error.
 */
AILA_API char* aila_generate_messages(AilaEngine* engine, const char* messages_json,
                                      const AilaGenConfig* config);

/**
 * Generate a structured assistant result from OpenAI-style chat request JSON.
 *
 * The input may be either a messages array or an object containing `messages`,
 * `tools`, `tool_choice`, `chat_template_kwargs`, and generation parameters.
 * The returned JSON contains `role`, `content`, `reasoning_content`,
 * `tool_calls`, `raw_text`, `finish_reason`, and `warnings`. Aila formats and
 * parses tool calls but does not execute tools; callers should execute returned
 * tool calls externally and feed tool results back as `tool` messages.
 * `finish_reason` may be "stop", "length", "loop_guard", "tool_calls", or
 * "tool_policy" for a strict tool-policy violation.
 *
 * @param engine            Initialized engine handle
 * @param chat_request_json UTF-8 JSON chat request string
 * @param config            Generation config defaults/overrides (NULL for defaults)
 * @return Newly allocated UTF-8 JSON string. Caller must free with aila_free_string().
 *         Returns NULL on error.
 */
AILA_API char* aila_generate_chat_json(AilaEngine* engine, const char* chat_request_json,
                                       const AilaGenConfig* config);

/**
 * Generate a structured assistant result using ABI-safe v2 generation config.
 * @param config  V2 config (NULL for defaults). struct_size must be non-zero.
 */
AILA_API char* aila_generate_chat_json_ex(AilaEngine* engine, const char* chat_request_json,
                                          const AilaGenConfigV2* config);

/**
 * Generate structured chat stream events using ABI-safe v2 generation config.
 * Aila formats and parses tool calls but does not execute tools.
 * @return 0 on success, 1 when aborted by callback, -1 on error
 */
AILA_API int aila_generate_chat_json_stream_ex(AilaEngine* engine,
                                               const char* chat_request_json,
                                               const AilaGenConfigV2* config,
                                               AilaChatStreamCallback callback,
                                               void* user_data);

/**
 * Generate response from OpenAI-style messages JSON with streaming token callback.
 * @param engine         Initialized engine handle
 * @param messages_json  UTF-8 JSON array string. Each item should contain role/content.
 * @param config         Generation config (NULL for defaults)
 * @param callback       Called for each generated token chunk
 * @param user_data      Passed through to callback
 * @return 0 on success, 1 when aborted by callback, non-zero on error
 */
AILA_API int aila_generate_messages_stream(AilaEngine* engine, const char* messages_json,
                                           const AilaGenConfig* config,
                                           AilaTokenCallback callback, void* user_data);

/**
 * Generate response with streaming token callback.
 * @param engine     Initialized engine handle
 * @param prompt     User message (UTF-8 null-terminated)
 * @param config     Generation config (NULL for defaults)
 * @param callback   Called for each generated token
 * @param user_data  Passed through to callback
 * @return 0 on success, non-zero on error
 */
AILA_API int aila_generate_stream(AilaEngine* engine, const char* prompt,
                                   const AilaGenConfig* config,
                                   AilaTokenCallback callback, void* user_data);

/**
 * Free a string returned by aila_generate.
 * @param str  String to free. NULL is safe.
 */
AILA_API void aila_free_string(char* str);

/* -------------- YOLO26 object detection -------------- */
AILA_API AilaDetectionConfig aila_default_detection_config(void);

AILA_API int aila_detect_file(
    AilaEngine* engine, const char* image_path,
    const AilaDetectionConfig* config,
    AilaDetection** out_detections, int* out_count);

AILA_API int aila_detect_encoded(
    AilaEngine* engine, const void* encoded_bytes, size_t encoded_size,
    const AilaDetectionConfig* config,
    AilaDetection** out_detections, int* out_count);

AILA_API int aila_detect_pixels(
    AilaEngine* engine, const void* pixels, size_t size_bytes,
    int width, int height, int row_stride, AilaPixelFormat format,
    const AilaDetectionConfig* config,
    AilaDetection** out_detections, int* out_count);

AILA_API void aila_free_detections(AilaDetection* detections, int count);

/**
 * Set global log callback.
 * @param callback  Log handler, or NULL to restore default (stdout/stderr) logging
 * @param user_data Opaque pointer passed to callback
 */
AILA_API void aila_set_log_callback(AilaLogCallback callback, void* user_data);

/**
 * Set minimum log level. Messages below this level are suppressed.
 * @param level  0=Debug, 1=Info, 2=Warning, 3=Error
 */
AILA_API void aila_set_log_level(int level);

/**
 * Reset conversation context (clear history and KV cache).
 */
AILA_API void aila_engine_reset_context(AilaEngine* engine);

/**
 * Get current context length in tokens.
 */
AILA_API int aila_engine_context_length(AilaEngine* engine);

/**
 * Get last error code on this engine.
 * @return AILA_OK when last call succeeded.
 */
AILA_API int aila_last_error_code(AilaEngine* engine);

/**
 * Get last error message on this engine.
 * The returned pointer is valid until next API call on the same engine.
 */
AILA_API const char* aila_last_error_message(AilaEngine* engine);

/**
 * Transcribe audio file (blocking).
 * Supports WAV, MP3, FLAC, and other formats handled by the engine.
 * @param engine           Initialized engine handle (configured with Qwen3-ASR model)
 * @param wav_path         Path to the audio file
 * @param config           Generation config (NULL for defaults)
 * @param forced_language  Optional language name to force (e.g. "Chinese", "English").
 *                         Set to NULL or "" to enable auto-detection.
 * @param language_out     If not NULL, receives a newly allocated UTF-8 string containing the
 *                         recognized language name (e.g. "Chinese", "English").
 *                         Caller must free this string with aila_free_string().
 *                         If an error occurs, sets *language_out to NULL.
 * @return Newly allocated UTF-8 string containing the clean transcription text.
 *         Caller must free the returned string with aila_free_string().
 *         Returns NULL on error.
 */
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
);

typedef struct AilaTranscribeStream AilaTranscribeStream;

/**
 * Create a real-time streaming ASR context.
 * Returns NULL on error or if the model does not support ASR.
 */
AILA_API AilaTranscribeStream* aila_transcribe_stream_create(
    AilaEngine* engine,
    const AilaGenConfig* config,
    const char* forced_language,
    const char* system_prompt
);

/**
 * Feed mono 16kHz Float PCM audio samples into the ASR stream.
 * Returns 0 on success, non-zero on error.
 */
AILA_API int aila_transcribe_stream_feed(
    AilaTranscribeStream* stream,
    const float* samples,
    int sample_count
);

/**
 * Get the current transcription text from the stream.
 * @param stream       ASR stream context
 * @param out_stable   [out] Receives a newly allocated UTF-8 string containing the stable text.
 *                     Must be freed with aila_free_string() by the caller. Can be NULL.
 * @param out_partial  [out] Receives a newly allocated UTF-8 string containing the temporary text.
 *                     Must be freed with aila_free_string() by the caller. Can be NULL.
 * Returns 0 on success, non-zero on error.
 */
AILA_API int aila_transcribe_stream_get_text(
    AilaTranscribeStream* stream,
    char** out_stable,
    char** out_partial
);

/**
 * Destroy the ASR stream context and free all allocated resources.
 */
AILA_API void aila_transcribe_stream_destroy(AilaTranscribeStream* stream);



/**
 * Synthesize raw float audio samples from pre-tokenized text tokens (blocking).
 * Supports zero-shot voice cloning when speaker_embedding is provided.
 *
 * Most users should prefer aila_synthesize_text_to_wav() for automatic tokenization.
 *
 * @param engine             Initialized engine handle (configured with Qwen3-TTS model)
 * @param text_tokens        Array of text token IDs
 * @param text_tokens_len    Number of tokens in text_tokens array
 * @param speaker_embedding  Optional array of speaker clone embeddings (can be NULL for default voice)
 * @param speaker_embedding_len Length of speaker clone embedding (1024 for 0.6B, 2048 for 1.7B; 0 if NULL)
 * @param config             Generation config (NULL for defaults)
 * @param out_samples        [out] Receives a newly allocated array of float audio samples (24000Hz PCM)
 *                           Must be freed with aila_free_samples() by the caller.
 * @param out_sample_count   [out] Receives the number of generated audio samples.
 * @return 0 on success, non-zero on error.
 */
AILA_API int aila_synthesize_wav(
    AilaEngine* engine,
    const int* text_tokens,
    int text_tokens_len,
    const float* speaker_embedding,
    int speaker_embedding_len,
    const AilaGenConfig* config,
    float** out_samples,
    int* out_sample_count
);

/**
 * Synthesize raw float audio samples directly from a UTF-8 text string (blocking).
 * Automatically handles ChatML layout and tokenization for Qwen3-TTS.
 * Supports zero-shot voice cloning when speaker_embedding is provided.
 *
 * To obtain a speaker embedding, call aila_extract_speaker_embedding() with a reference audio file.
 *
 * @param engine             Initialized engine handle (configured with Qwen3-TTS model)
 * @param text               UTF-8 text prompt to synthesize
 * @param speaker_embedding  Optional array of speaker clone embeddings (can be NULL for default voice)
 * @param speaker_embedding_len Length of speaker clone embedding (1024 for 0.6B, 2048 for 1.7B; 0 if NULL)
 * @param config             Generation config (NULL for defaults). TTS auto-sets repetition_penalty to 1.1
 *                           if not explicitly configured.
 * @param out_samples        [out] Receives a newly allocated array of float audio samples (24000Hz PCM)
 *                           Must be freed with aila_free_samples() by the caller.
 * @param out_sample_count   [out] Receives the number of generated audio samples.
 * @return 0 on success, non-zero on error.
 */
AILA_API int aila_synthesize_text_to_wav(
    AilaEngine* engine,
    const char* text,
    const float* speaker_embedding,
    int speaker_embedding_len,
    const AilaGenConfig* config,
    float** out_samples,
    int* out_sample_count
);

/**
 * One-shot TTS synthesis: text + voice identity → WAV file.
 * This is the recommended high-level API for most use cases.
 *
 * Supports three voice identity modes:
 *   - Base (voice cloning): provide reference_audio_path for ECAPA-TDNN extraction
 *   - CustomVoice: provide speaker_name (e.g. "vivian", "ryan") for pre-trained voices
 *   - VoiceDesign: provide instruct_text for style description
 *   - Default voice: all identity params NULL (or empty strings)
 *
 * Internally handles speaker embedding extraction (with automatic caching),
 * tokenization, synthesis, and WAV file writing in a single call.
 * No manual memory management required.
 *
 * @param engine              Initialized engine handle (Qwen3-TTS model)
 * @param text                UTF-8 text to synthesize
 * @param reference_audio_path  Base: reference audio path for voice cloning (NULL for none)
 * @param speaker_name        CustomVoice: speaker name e.g. "vivian" (NULL for none)
 * @param instruct_text       VoiceDesign: style description text (NULL for none)
 * @param language            Language code: "chinese", "english", "japanese", "korean" (NULL for auto)
 * @param config              Generation config (NULL for defaults)
 * @param output_wav_path     Output WAV file path (24kHz, mono, 16-bit PCM)
 * @return 0 on success, non-zero on error (check aila_last_error_code/aila_last_error_message)
 */
AILA_API int aila_synthesize(
    AilaEngine* engine,
    const char* text,
    const char* reference_audio_path,
    const char* speaker_name,
    const char* instruct_text,
    const char* language,
    const AilaGenConfig* config,
    const char* output_wav_path
);

/**
 * Audio streaming callback for TTS.
 * @param samples   PCM float audio samples (24kHz mono)
 * @param count     Number of samples in this chunk
 * @param user_data Opaque pointer passed to aila_synthesize_stream
 */
typedef void (*AilaAudioCallback)(const float* samples, int count, void* user_data);

typedef struct AilaTTSStream AilaTTSStream;

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
);

AILA_API int aila_stream_wait(AilaTTSStream* stream);
AILA_API void aila_stream_destroy(AilaTTSStream* stream);

/**
 * Free audio samples array returned by aila_synthesize_wav or aila_extract_speaker_embedding.
 */
AILA_API void aila_free_samples(float* samples);

/**
 * Decode discrete codes to float audio samples using Mimi Vocoder.
 *
 * @param engine             Engine handle
 * @param codes              Array of discrete codes of shape [n_frames, 16]
 * @param n_frames           Number of frames
 * @param out_samples        [out] Receives newly allocated audio samples array
 * @param out_sample_count   [out] Receives size of audio samples array
 * @return 0 on success, non-zero on error.
 */
AILA_API int aila_decode_mimi_vocoder(
    AilaEngine* engine,
    const int32_t* codes,
    int n_frames,
    float** out_samples,
    int* out_sample_count
);

/**
 * Extract speaker embedding from a reference audio file for TTS voice cloning.
 * Uses the native C++ ECAPA-TDNN speaker encoder (CPU-side, no Python dependency).
 *
 * Supported audio formats: WAV, MP3, FLAC.
 * Audio requirements: mono (multi-channel is averaged), any sample rate (auto-resampled to 24kHz).
 * Recommended: ≥3 seconds of clear speech, 24kHz, mono, float32 or int16, normalized to [-1, 1].
 *
 * @param engine             Initialized engine handle (must be configured with a Qwen3-TTS model,
 *                           as the speaker encoder weights reside in the TTS model's safetensors file).
 * @param audio_path         Path to the reference audio file.
 * @param out_embedding      [out] Receives a newly allocated float array containing the speaker embedding.
 *                           The caller must free this array with aila_free_samples().
 * @param out_embedding_dim  [out] Receives the dimension of the speaker embedding (1024 for 0.6B, 2048 for 1.7B).
 * @return 0 on success, non-zero on error.
 */
AILA_API int aila_extract_speaker_embedding(
    AilaEngine* engine,
    const char* audio_path,
    float** out_embedding,
    int* out_embedding_dim
);

// ============================================================
// Forced Alignment API
// ============================================================

/** Represents one aligned word with time boundaries. */
typedef struct {
    const char* text;
    int start_ms;
    int end_ms;
} AilaAlignedWord;

/**
 * Run forced alignment: audio samples + text → word-level timestamps.
 *
 * @param engine         Initialized ForceAligner engine.
 * @param audio_samples  Mono float32 audio samples.
 * @param num_samples    Number of audio samples.
 * @param sample_rate    Audio sample rate in Hz (will be resampled to 16kHz).
 * @param text           Transcript text to align.
 * @param language       Language name (e.g. "Chinese", "English").
 * @param out_words      Output: pointer to array of AilaAlignedWord (caller frees with aila_free_aligned_words).
 * @param out_count      Output: number of aligned words.
 * @return 0 on success, non-zero on error.
 */
AILA_API int aila_align(
    AilaEngine* engine,
    const float* audio_samples, int num_samples, int sample_rate,
    const char* text, const char* language,
    AilaAlignedWord** out_words, int* out_count);

/**
 * Free memory allocated by aila_align() or aila_align_words().
 */
AILA_API void aila_free_aligned_words(AilaAlignedWord* words, int count);

/**
 * Run forced alignment with pre-tokenized word list.
 * Bypasses the built-in tokenizer — caller provides already-tokenized words
 * (e.g. from Python-side nagisa for Japanese or soynlp for Korean).
 *
 * @param engine         Initialized ForceAligner engine.
 * @param audio_samples  Mono float32 audio samples.
 * @param num_samples    Number of audio samples.
 * @param sample_rate    Audio sample rate in Hz (resampled to 16kHz).
 * @param words          Pre-tokenized word strings (UTF-8).
 * @param num_words      Number of words.
 * @param out_words      Output: pointer to array of AilaAlignedWord.
 * @param out_count      Output: number of aligned words.
 * @return 0 on success, non-zero on error.
 */
AILA_API int aila_align_words(
    AilaEngine* engine,
    const float* audio_samples, int num_samples, int sample_rate,
    const char* const* words, int num_words,
    AilaAlignedWord** out_words, int* out_count);

#endif /* AILA_API_H */
