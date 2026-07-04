# Alia C API

This document covers the Alia-specific C ABI declared in `include/alia_api.h`.
It is separate from the generic `aila_*` API documented in `docs/C_API.md`.

Verified against:

- `include/alia_api.h`
- `src/alia/AliaApi.cpp`
- `src/alia/AliaContext.cpp`
- `src/alia/AliaAsrPipeline.cpp`
- `src/alia/AliaForegroundPipeline.cpp`
- `src/alia/AliaTtsPipeline.cpp`
- `src/alia/AliaBackgroundPipeline.cpp`

## Header and linkage

```c
#include "alia_api.h"
```

The shared-library export macro is `ALIA_API`.

- On Windows, symbols are exported when `AILA_BUILDING_DLL` is defined.
- When `AILA_STATIC_LIB` is defined, no `dllimport` or `dllexport` attribute is used.
- On non-Windows platforms, exported symbols use default visibility.
- All exported functions use `extern "C"` when compiled as C++.

The primary handle is opaque:

```c
typedef struct AliaContext AliaContext;
```

Callers must not allocate, free, or inspect `AliaContext` directly.

## Return codes

```c
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
```

Current implementation behavior:

| Code | Meaning in current code |
| --- | --- |
| `ALIA_OK` | The call completed or an async job was accepted. |
| `ALIA_ERR_INVALID_ARGUMENT` | Null required pointer, invalid count, invalid generation config, or negative rollback. |
| `ALIA_ERR_INVALID_STATE` | Required model slot or pipeline is unavailable, a foreground/background job is busy, or rollback has no replayable generation state. |
| `ALIA_ERR_MODEL_LOAD` | `alia_context_init` failed while loading or warming model slots. |
| `ALIA_ERR_RUNTIME` | C++ exception, allocation failure, KV replay failure, or other runtime failure caught by the C wrapper. |
| `ALIA_ERR_ABORTED` | A `ModelBackendCancelled` exception escaped into the C wrapper. |
| `ALIA_ERR_CONTEXT_OVERFLOW` | Foreground ASR prefill would exceed the loaded VLM context length. |
| `ALIA_ERR_CALLBACK` | Declared in the ABI, but no exported wrapper currently returns it directly. Callback failures normally surface as async foreground failure after the start call has already returned `ALIA_OK`. |

All exported non-void functions are guarded against C++ exceptions. Exceptions
are converted to `ALIA_ERR_RUNTIME`, except backend cancellation, which maps to
`ALIA_ERR_ABORTED`. Exported void functions swallow exceptions.

## Pipeline masks

```c
typedef enum AliaPipelineMask {
    ALIA_PIPELINE_ASR = 1 << 0,
    ALIA_PIPELINE_VLM_FOREGROUND = 1 << 1,
    ALIA_PIPELINE_TTS = 1 << 2,
    ALIA_PIPELINE_VLM_BACKGROUND = 1 << 3,
    ALIA_PIPELINE_ALL = 0xFFFF
} AliaPipelineMask;
```

`alia_abort_inference` stores the supplied mask in `ctx->abort_mask`.
It actively requests abort on:

- foreground pipeline when the mask is `ALIA_PIPELINE_ALL`, includes `ALIA_PIPELINE_VLM_FOREGROUND`, or includes `ALIA_PIPELINE_TTS`;
- background pipeline when the mask is `ALIA_PIPELINE_ALL` or includes `ALIA_PIPELINE_VLM_BACKGROUND`.

`ALIA_PIPELINE_ASR` is recorded in the mask but does not currently trigger a
dedicated ASR abort path in `alia_abort_inference`.

## Generation config

```c
typedef struct AliaGenConfig {
    float temperature;
    float top_p;
    int max_tokens;
} AliaGenConfig;
```

Valid config values:

- `temperature` must be finite and `>= 0.0f`.
- `top_p` must be finite and in `(0.0f, 1.0f]`.
- `max_tokens` must be `> 0`.

When `config == NULL`, the foreground path uses:

```c
temperature = 0.6f;
top_p = 0.9f;
max_tokens = 256;
```

The config is translated to the internal generation config as:

- `max_new_tokens = max_tokens`
- `temperature = temperature`
- `top_p = top_p`
- `do_sample = (temperature > 0.0f)`

So `temperature == 0.0f` selects greedy-style decoding in the current code.

## Callbacks

### Tool callback

```c
typedef int (*AliaToolCallCallback)(
    const char* tool_json,
    char* out_result_buf,
    int max_result_len,
    void* user_data);
```

The foreground VLM only allows tool calls when the user text explicitly asks
for a host tool or tool call. When a tool call is parsed:

- `tool_json` is a UTF-8 JSON string for one parsed tool call.
- `out_result_buf` is an 8192-byte buffer in the current implementation.
- The callback must write a NUL-terminated result into `out_result_buf`.
- Return `0` for success. Any non-zero return marks the async foreground turn as failed.
- `user_data` is the pointer passed to the foreground start or commit call.

If the model emits tool calls and `tool_cb == NULL`, the async foreground turn
fails with the internal error text "foreground VLM emitted a tool call but no
Alia tool callback was registered".

The start function has already returned by the time tool callbacks run, so a
tool callback failure is not returned synchronously from
`alia_start_conversation_turn`.

### Audio callback

```c
typedef void (*AliaAudioCallback)(
    const float* samples,
    int sample_count,
    void* user_data);
```

Audio callbacks are emitted by the TTS worker during foreground turns.

- `samples` points to float PCM samples.
- The TTS path produces 24 kHz mono float PCM.
- The pointer is valid only during the callback. Copy the samples if they must outlive the callback.
- `sample_count` is the number of float samples in this callback.
- `user_data` is the pointer passed to the foreground start or commit call.

If `audio_cb == NULL`, foreground VLM generation can still run, but no TTS
worker is started and no audio is emitted.

### Background result callback

```c
typedef void (*AliaBackgroundResultCallback)(
    const char* extracted_json,
    void* user_data);
```

Background extraction invokes this callback from the background worker thread.

- `extracted_json` is a UTF-8 JSON object string.
- The current implementation always passes `NULL` as `user_data` because
  `alia_register_background_callback` stores only the function pointer.
- The pointer is valid only during the callback.

The background pipeline attempts to return a JSON object with:

```json
{
  "summary": "string",
  "memory_candidates": [],
  "preferences": [],
  "tasks": []
}
```

If the model output is fenced or contains extra text, the implementation tries
to normalize it. If schema validation still fails after one repair generation,
it wraps the result in a schema-compatible JSON object and includes
`raw_model_output` and `source`.

## Lifecycle

### `alia_context_init`

```c
ALIA_API int alia_context_init(
    AliaContext** out_ctx,
    const char* asr_model_dir,
    const char* vlm_4b_model_dir,
    const char* vlm_0_8b_model_dir,
    const char* tts_model_dir,
    int max_seq_len);
```

Creates an `AliaContext`, configures four model slots, loads the slots, and
performs default startup warmups/preloads.

Arguments:

- `out_ctx`: required output pointer. Set to `NULL` before loading begins.
- `asr_model_dir`: Qwen3-ASR or Qwen3 force-aligner compatible model directory.
- `vlm_4b_model_dir`: foreground Qwen3.5 Hybrid bitsandbytes NF4 model directory.
- `vlm_0_8b_model_dir`: background Qwen3.5 Hybrid bitsandbytes NF4 model directory.
- `tts_model_dir`: Qwen3-TTS model directory.
- `max_seq_len`: maximum context length, must be positive.

Return behavior:

- `ALIA_OK`: all required load and startup steps succeeded and `*out_ctx` owns the new context.
- `ALIA_ERR_INVALID_ARGUMENT`: `out_ctx == NULL` or `max_seq_len <= 0`.
- `ALIA_ERR_MODEL_LOAD`: model metadata, tokenizer, weights, backend, warmup, LoRA application, audio encoder, vision encoder, Mimi vocoder, ASR GPU mel warmup, foreground VLM warmup, or TTS reference voice preload failed.
- `ALIA_ERR_RUNTIME`: allocation or unexpected exception.

Important implementation details:

- Null model directory pointers are treated as empty strings.
- `ModelSlot::load_model` treats an empty model directory as a no-op success.
  However, default `alia_context_init` still runs ASR GPU mel warmup,
  foreground VLM warmup, and TTS reference voice preload. With default env vars,
  an empty ASR, foreground VLM, or TTS slot will therefore fail initialization.
- The foreground and ASR/TTS slots share the foreground runtime context. The
  background VLM uses a separate background runtime context.
- The current C API does not expose a `vlm_4b_lora_dir` parameter. The internal
  `AliaContext` has such a field, but it is not populated by `alia_context_init`.

### `alia_context_destroy`

```c
ALIA_API void alia_context_destroy(AliaContext* ctx);
```

Destroys the context and all owned pipelines/model slots. Passing `NULL` is
safe because C++ `delete nullptr` is safe.

Pipeline destructors request abort and join their worker threads. Therefore,
destroying a context with active foreground, background, or TTS work can block
while the worker exits.

### `alia_free_string`

```c
ALIA_API void alia_free_string(char* s);
```

Frees strings allocated by Alia C API functions. Passing `NULL` is safe.

Use this for strings returned through:

- `alia_asr_get_text`
- `alia_asr_get_partial_text`

Do not free those strings with host language allocators.

## ASR API

The ASR path accepts 16 kHz mono float PCM samples and maintains two text
buffers:

- stable text: finalized text from completed audio chunks;
- partial text: current non-final text from the active tail.

### `alia_asr_feed_audio`

```c
ALIA_API int alia_asr_feed_audio(
    AliaContext* ctx,
    const float* samples,
    int sample_count);
```

Appends audio samples to the ASR buffer.

Return behavior:

- `ALIA_OK`: samples were appended.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL`, ASR pipeline missing, `samples == NULL`, `sample_count <= 0`, or the pipeline rejected the input.
- `ALIA_ERR_RUNTIME`: unexpected exception.

This function only appends audio. Decoding happens when text is requested by
`alia_asr_get_text` or `alia_asr_get_partial_text`, or when internal callers
invoke ASR pipeline processing.

### `alia_asr_get_text`

```c
ALIA_API int alia_asr_get_text(
    AliaContext* ctx,
    char** out_stable,
    char** out_partial);
```

Forces pending ASR processing and returns newly allocated UTF-8 copies of the
current stable and partial text.

Behavior:

- Calls `process_pending(true)`, which allows a forced partial decode.
- If `out_stable` is non-null, it is first set to `NULL`, then receives a
  malloc-owned string on success.
- If `out_partial` is non-null, it is first set to `NULL`, then receives a
  malloc-owned string on success.
- Either output pointer may be `NULL`.
- Empty text is returned as an allocated empty string, not as `NULL`.
- Each non-null output must be released with `alia_free_string`.

Return behavior:

- `ALIA_OK`: outputs were produced.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL`.
- `ALIA_ERR_INVALID_STATE`: ASR pipeline is missing.
- `ALIA_ERR_RUNTIME`: string allocation failed or an exception was caught.

If allocation of `out_partial` fails after `out_stable` succeeded, the function
frees the stable string and resets it to `NULL` before returning
`ALIA_ERR_RUNTIME`.

### `alia_asr_get_partial_text`

```c
ALIA_API int alia_asr_get_partial_text(
    AliaContext* ctx,
    char** out_stable,
    char** out_partial);
```

Returns current ASR text using the partial decode throttle path.

Behavior is the same as `alia_asr_get_text`, except it calls
`process_pending(false)`. In current code that means:

- stable chunks are still processed when enough audio is available;
- partial decode may be skipped when too little new audio has arrived since the
  last partial decode;
- the minimum partial advance defaults to at least 500 ms and is controlled by
  `AILA_ASR_PARTIAL_MIN_ADVANCE_MS`.

Use this for frequent UI polling. Use `alia_asr_get_text` when finalizing a
turn and you want to force the active partial tail.

### `alia_asr_reset`

```c
ALIA_API void alia_asr_reset(AliaContext* ctx);
```

Clears ASR audio buffer, stable/partial text, past text, prefix caches, mel
caches, counters, and metrics. Passing `NULL` is safe. If the ASR pipeline is
missing, the function does nothing.

## Foreground VLM and TTS API

The foreground path combines current ASR stable and partial text into a user
message, generates a concise spoken response with the foreground VLM, optionally
handles explicit host tool calls, and optionally streams TTS audio through
`AliaAudioCallback`.

Foreground work is asynchronous. `alia_start_conversation_turn`,
`alia_start_speculative_conversation_turn`, and
`alia_commit_speculative_conversation_turn` return after the worker is started
or rejected, not after the generated answer has completed.

The public C ABI currently has no exported wait/status function. The smoke
tool uses internal C++ pipeline methods to wait for completion.

### `alia_vlm_prefill_asr_text`

```c
ALIA_API int alia_vlm_prefill_asr_text(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text);
```

Prefills the foreground VLM KV cache from the supplied ASR stable/partial text.
This is an optimization for low-latency voice turns.

Behavior:

- Null text pointers are treated as empty strings.
- Stable and partial text are trimmed and combined with a space when needed.
- Empty combined text is a no-op and returns `ALIA_OK`.
- The call fails with `ALIA_ERR_INVALID_STATE` if a foreground turn is running
  or aborting.
- The prompt keeps guard tokens un-prefilled to reduce retokenization and final
  suffix mismatch risk.
- Small incremental suffixes can be skipped according to
  `AILA_FOREGROUND_MIN_INCREMENTAL_PREFILL_SUFFIX_TOKENS`.
- Small cached suffixes may use decode-path forwarding according to
  `AILA_FOREGROUND_DECODE_SUFFIX_TOKENS`.

Return behavior:

- `ALIA_OK`: prefill succeeded, was skipped as a no-op, or was skipped as too small.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL`.
- `ALIA_ERR_INVALID_STATE`: foreground pipeline/model/backend unavailable or busy.
- `ALIA_ERR_CONTEXT_OVERFLOW`: target prefill token count exceeds backend max sequence length.
- `ALIA_ERR_RUNTIME`: exception caught by wrapper.

### `alia_start_conversation_turn`

```c
ALIA_API int alia_start_conversation_turn(
    AliaContext* ctx,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data);
```

Starts an asynchronous foreground turn using text currently held by the ASR
pipeline.

Behavior:

- Validates `config` synchronously when non-null.
- Reads stable and partial ASR text inside the worker by calling
  `asr_pipeline->get_text`.
- If the combined ASR text is empty, the VLM prompt text becomes
  `"Continue the current conversation."`.
- If `audio_cb` is non-null and TTS is ready, starts an async TTS worker before
  VLM decoding.
- Spoken content deltas are chunked and enqueued to TTS while decoding.
- Action tags enclosed in ASCII or full-width parentheses are removed from
  spoken text and stored internally as action tags.
- Structured artifacts such as `<think>` and `<tool_call>` are stripped from
  spoken text before the final assistant text is stored internally.
- Explicit tool calls are parsed and delivered to `tool_cb`. Tool results are
  fed back to the VLM for a concise continuation.

Return behavior:

- `ALIA_OK`: the foreground worker was started.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL`, foreground pipeline missing, or invalid `config`.
- `ALIA_ERR_INVALID_STATE`: foreground worker is already running/aborting or start was rejected.
- `ALIA_ERR_RUNTIME`: exception caught by wrapper.

Async failures after `ALIA_OK` include unloaded VLM slot, TTS worker start
failure, VLM generation failure, tool callback failure, TTS worker failure, and
backend cancellation without an abort request.

### `alia_start_speculative_conversation_turn`

```c
ALIA_API int alia_start_speculative_conversation_turn(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text,
    const AliaGenConfig* config);
```

Starts an asynchronous text-only speculative foreground generation using the
provided stable/partial ASR text. No TTS audio callback is accepted by this
function, so it generates and caches spoken text for a later commit.

Behavior:

- Null text pointers are treated as empty strings.
- Stable and partial text are trimmed and combined.
- Empty combined text is rejected synchronously.
- Any existing completed worker is joined before starting the new speculative worker.
- A busy foreground worker causes rejection.
- The speculative worker generates with the loaded foreground VLM and stores
  speculative text only if the stripped spoken text is non-empty.

Return behavior:

- `ALIA_OK`: speculative worker was started.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL` or foreground pipeline missing.
- `ALIA_ERR_INVALID_STATE`: foreground pipeline rejected the start. This includes busy state, invalid `config`, or empty combined ASR text.
- `ALIA_ERR_RUNTIME`: exception caught by wrapper.

### `alia_commit_speculative_conversation_turn`

```c
ALIA_API int alia_commit_speculative_conversation_turn(
    AliaContext* ctx,
    const char* stable_text,
    const char* partial_text,
    const AliaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data);
```

Commits a previously started speculative foreground turn when the final ASR text
matches, otherwise falls back to a normal foreground turn.

Behavior:

- If a foreground worker is currently running, this call requests abort, marks
  it as aborting, notifies waiters, and joins it before continuing.
- If the speculative result is ready and final ASR text exactly matches the
  speculative user text, the cached spoken text is sent through TTS without
  regenerating the VLM response.
- If `AILA_FOREGROUND_SPECULATIVE_PREFIX_COMMIT=1`, a final ASR text that has
  the speculative user text as a prefix can also commit.
- If commit is impossible or speculative TTS commit fails, the function falls
  back to the same worker path used by `alia_start_conversation_turn`.

Return behavior:

- `ALIA_OK`: commit/fallback worker was started.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL` or foreground pipeline missing.
- `ALIA_ERR_INVALID_STATE`: pipeline rejected the commit/fallback start. This includes invalid `config`.
- `ALIA_ERR_RUNTIME`: exception caught by wrapper.

Note that `ALIA_OK` means the commit/fallback worker was accepted. It does not
prove that the speculative cache was used or that the eventual async turn
succeeded.

### `alia_vlm_rollback_kv_cache`

```c
ALIA_API int alia_vlm_rollback_kv_cache(
    AliaContext* ctx,
    int rollback_tokens);
```

Rolls back generated foreground VLM KV cache tokens from the most recent
replayable generation.

Behavior:

- `rollback_tokens == 0` is a no-op success.
- Negative rollback is invalid.
- The foreground pipeline must be idle.
- A previous generation anchor and generated token list must exist.
- The backend current context length must match the replayable length.
- The implementation first attempts backend KV truncation. If truncation fails,
  it resets and replays the prompt plus generated tokens up to the target length.

Return behavior:

- `ALIA_OK`: rollback succeeded or was a zero-token no-op.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL` or `rollback_tokens < 0`.
- `ALIA_ERR_INVALID_STATE`: no foreground pipeline, busy pipeline, missing replay state, unavailable backend, or current context is not replayable.
- `ALIA_ERR_RUNTIME`: replay fallback failed or exception caught.

## Abort API

### `alia_abort_inference`

```c
ALIA_API int alia_abort_inference(
    AliaContext* ctx,
    int pipeline_mask);
```

Requests cancellation for selected pipelines and returns quickly.

Return behavior:

- `ALIA_OK`: abort was requested.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL`.
- `ALIA_ERR_RUNTIME`: exception caught by wrapper.

This function does not join worker threads. Context destruction and internal
pipeline cleanup perform joins.

## Background API

### `alia_register_background_callback`

```c
ALIA_API void alia_register_background_callback(
    AliaContext* ctx,
    AliaBackgroundResultCallback callback);
```

Registers the background extraction callback. Passing `ctx == NULL` is safe and
does nothing. Passing `callback == NULL` clears the callback.

There is no public `user_data` parameter. The callback receives `NULL` as its
second argument in current code.

### `alia_trigger_background_processing`

```c
ALIA_API int alia_trigger_background_processing(
    AliaContext* ctx,
    const char* chat_turn_text);
```

Starts an asynchronous background memory extraction job.

Arguments:

- `chat_turn_text`: required UTF-8 text. Smoke tooling passes a string shaped
  like `User: ...\nAssistant: ...`.

Behavior:

- Rejects the call if another background job is running or aborting.
- Joins a previous completed worker before starting the new one.
- Requires a registered callback.
- Generates strict JSON with the background VLM. If initial output does not
  satisfy schema, performs one repair generation, then applies a schema wrapper
  if still invalid.
- Invokes the registered callback on the background worker thread with the final
  JSON string.

Return behavior:

- `ALIA_OK`: background worker was started.
- `ALIA_ERR_INVALID_ARGUMENT`: `ctx == NULL` or `chat_turn_text == NULL`.
- `ALIA_ERR_INVALID_STATE`: background pipeline missing, background worker busy, or callback not registered.
- `ALIA_ERR_RUNTIME`: exception caught by wrapper.

As with foreground work, `ALIA_OK` means the worker was accepted. Generation or
schema repair can still fail asynchronously.

## Threading and lifetime notes

- The Alia C API starts foreground and background work on internal `std::thread`
  workers.
- The C ABI currently has no exported wait/status/error accessor for Alia async
  work. Internal smoke tools use C++ pipeline methods such as
  `wait_until_idle_for`, `join`, and `last_error`.
- Do not call foreground start/prefill/rollback APIs concurrently from multiple
  host threads without external synchronization.
- `alia_abort_inference` requests cancellation, but does not guarantee that
  callbacks have stopped before it returns.
- Callback pointers and `user_data` must remain valid until the corresponding
  async worker has finished or the context has been destroyed.
- Audio and background callback string/sample pointers are valid only during the
  callback.

## Minimal host flow

```c
#include "alia_api.h"

static void on_audio(const float* samples, int sample_count, void* user_data) {
    (void)user_data;
    /* Copy samples here if needed. samples is valid only during the callback. */
}

static void on_background(const char* extracted_json, void* user_data) {
    (void)user_data; /* Current implementation passes NULL. */
    /* Copy extracted_json here if needed. */
}

int main(void) {
    AliaContext* ctx = NULL;
    int rc = alia_context_init(
        &ctx,
        "models/Qwen3-ASR-1.7B-BNB-NF4",
        "models/qwen3.5-4B-bnb-nf4-offline-visiondense",
        "models/qwen3.5-0.8B-bnb-nf4-offline",
        "models/Qwen3-TTS-12Hz-0.6B-Base",
        2048);
    if (rc != ALIA_OK) {
        return rc;
    }

    /* Feed 16 kHz mono float PCM. */
    /* alia_asr_feed_audio(ctx, samples, sample_count); */

    char* stable = NULL;
    char* partial = NULL;
    rc = alia_asr_get_text(ctx, &stable, &partial);
    if (rc == ALIA_OK) {
        alia_free_string(stable);
        alia_free_string(partial);
    }

    AliaGenConfig gen;
    gen.temperature = 0.6f;
    gen.top_p = 0.9f;
    gen.max_tokens = 48;

    rc = alia_start_conversation_turn(ctx, &gen, NULL, on_audio, NULL);
    if (rc != ALIA_OK) {
        alia_context_destroy(ctx);
        return rc;
    }

    alia_register_background_callback(ctx, on_background);
    (void)alia_trigger_background_processing(
        ctx,
        "User: hello\nAssistant: hello");

    alia_abort_inference(ctx, ALIA_PIPELINE_ALL);
    alia_context_destroy(ctx);
    return 0;
}
```

## Current limitations visible from the C ABI

- No exported wait, async status, or last-error accessor exists for `alia_*`
  async foreground/background work.
- `AliaBackgroundResultCallback` has a `user_data` parameter, but the C API does
  not expose registration of a user data pointer and currently passes `NULL`.
- `ALIA_ERR_CALLBACK` is declared but not returned directly by the exported
  wrappers in current code.
- The internal foreground LoRA directory field is not configurable through
  `alia_context_init`.
- The API currently exposes ASR audio input, foreground text/TTS output,
  background extraction, abort, KV rollback, and speculative foreground commit.
  It does not expose image input or public foreground result strings through C.
