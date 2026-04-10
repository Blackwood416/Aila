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
    int   decode_chunk_size;    /* default: 1      */
    int   stream_chunk_size;    /* default: 1      */
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

#endif /* AILA_API_H */
