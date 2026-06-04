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
    AILA_ERR_RUNTIME = 6
} AilaErrorCode;

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

/* -------------- Callback types -------------- */

/**
 * Token streaming callback.
 * @param token_text  UTF-8 encoded token text (null-terminated, valid only during call)
 * @param user_data   Opaque pointer passed to aila_generate_stream
 * @return 0 to continue, non-zero to abort generation
 */
typedef int (*AilaTokenCallback)(const char* token_text, void* user_data);

/**
 * Log callback.
 * @param level  0=Debug, 1=Info, 2=Warning, 3=Error
 * @param message  UTF-8 log message (null-terminated)
 * @param user_data Opaque pointer passed to aila_set_log_callback
 */
typedef void (*AilaLogCallback)(int level, const char* message, void* user_data);

/* -------------- API functions -------------- */

/** Get library version string (e.g. "0.1.0"). The returned pointer is static. */
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
