# Aila C API

The C API (`aila_api.h`) is the stable public interface for integrating Aila into any language with C FFI support. The shared library is `AilaShared.dll` (Windows).

## Quick Example

```c
#include "aila_api.h"

int main() {
    // Create engine
    AilaEngine* engine = aila_engine_create();
    if (!engine) return 1;

    // Load model
    if (aila_engine_init(engine, "./models/qwen3.5-0.8B-bnb-nf4-offline", 4096) != 0) {
        printf("Init failed: %s\n", aila_last_error_message(engine));
        aila_engine_destroy(engine);
        return 1;
    }

    // Generate (blocking)
    AilaGenConfig cfg = aila_default_gen_config();
    cfg.max_new_tokens = 64;
    cfg.do_sample = 0;  // greedy

    char* response = aila_generate(engine, "Hello!", &cfg);
    if (response) {
        printf("%s\n", response);
        aila_free_string(response);
    }

    // Cleanup
    aila_engine_destroy(engine);
    return 0;
}
```

## API Reference

### Lifecycle

#### `aila_engine_create`

```c
AilaEngine* aila_engine_create(void);
```

Creates a new engine instance. Returns `NULL` on allocation failure. The engine is not yet initialized — call `aila_engine_init` before generation.

#### `aila_engine_init`

```c
int aila_engine_init(AilaEngine* engine, const char* model_dir, int max_seq_len);
```

Loads model weights and tokenizer from `model_dir`. `max_seq_len` sets the maximum context window length (e.g. 4096).

Returns `0` on success, non-zero on failure. On failure, call `aila_last_error_code` / `aila_last_error_message` for diagnostics.

The model directory must contain:
- `config.json` — model architecture configuration
- `tokenizer.json` (or `tokenizer_config.json` + vocab files) — tokenizer data
- `model.safetensors` or sharded `model-*.safetensors` — weights

#### `aila_engine_destroy`

```c
void aila_engine_destroy(AilaEngine* engine);
```

Destroys the engine and frees all resources. Passing `NULL` is safe.

### Generation (Blocking)

#### `aila_generate`

```c
char* aila_generate(AilaEngine* engine, const char* prompt, const AilaGenConfig* config);
```

Generates a response for a single user message. Returns a newly allocated UTF-8 string. The caller must free it with `aila_free_string`. Returns `NULL` on error — check `aila_last_error_code` / `aila_last_error_message`.

`config` may be `NULL` to use defaults.

#### `aila_generate_messages`

```c
char* aila_generate_messages(AilaEngine* engine, const char* messages_json, const AilaGenConfig* config);
```

Generates a response from an OpenAI-style messages JSON. Same return/free semantics as `aila_generate`.

`messages_json` can be either a standard OpenAI-compatible JSON object with `"messages"` as the top-level key (which also allows passing generation/sampling parameters directly like `temperature`, `max_tokens` etc.), or a raw JSON array:

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user",   "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 128
}
```

Or backward-compatibly:

```json
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user",   "content": "Hello!"}
]
```

Each message has a `role` (`"system"`, `"user"`, or `"assistant"`) and `content` (string or array of content parts). Content parts may include `text`, `image`, `audio`, and `video` types (including base64 encoded image Data URIs and input_audio payload, see README for details).

### Generation (Streaming)

#### `aila_generate_stream`

```c
int aila_generate_stream(AilaEngine* engine, const char* prompt,
                         const AilaGenConfig* config,
                         AilaTokenCallback callback, void* user_data);
```

Generates with token-level streaming. The callback receives each token as a null-terminated UTF-8 string.

Returns:
- `0` — success
- `1` — aborted by callback (callback returned non-zero)
- `-1` — error

#### `aila_generate_messages_stream`

```c
int aila_generate_messages_stream(AilaEngine* engine, const char* messages_json,
                                  const AilaGenConfig* config,
                                  AilaTokenCallback callback, void* user_data);
```

Streaming version of `aila_generate_messages`. Same return values as `aila_generate_stream`.

#### `AilaTokenCallback`

```c
typedef int (*AilaTokenCallback)(const char* token_text, void* user_data);
```

- `token_text` — UTF-8 token string, valid only during the callback
- Return `0` to continue, non-zero to abort generation

### Audio Transcription (ASR)

#### `aila_transcribe`

```c
char* aila_transcribe(
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
```

Transcribes an audio file (blocking). Supports WAV, MP3, FLAC, and other audio formats.
If the loaded model does not support ASR (i.e. not configured with Qwen3-ASR model), the call will be gracefully intercepted, returning `NULL` and setting the error code to `AILA_ERR_RUNTIME` (6).

- `wav_path` — Path to the audio file.
- `config` — Generation configuration (can be `NULL` for defaults).
- `forced_language` — Optional language name to force (e.g., `"Chinese"`, `"English"`). Set to `NULL` or `""` for auto-detection.
- `system_prompt` — Optional system prompt to guide/bias transcription (e.g., spelling/role bias). Can be `NULL`.
- `segment_sec` — Segment split duration in seconds. Set to `<= 0.0` to disable segmentation and transcribe as a single chunk.
- `past_text_conditioning` — `1` to enable past-text conditioning (uses previous transcript segment as historical context for next segment), `0` to disable.
- `token_callback` — Optional real-time streaming callback receiving segment pieces as they are decoded.
- `user_data` — Optional user-defined pointer passed back to `token_callback`.
- `language_out` — If not `NULL`, receives a newly allocated UTF-8 string containing the recognized/forced language name (e.g., `"Chinese"`, `"English"`). The caller **must** free this string with `aila_free_string()`.

Returns a newly allocated UTF-8 string containing the clean transcription text (stripped of control tags like `<asr_text>` and language prefixes). The caller **must** free the returned string with `aila_free_string()`. Returns `NULL` on error.

### Real-Time Streaming Input ASR (Online ASR)

Aila supports real-time streaming input ASR for live audio transcription (e.g., from a microphone). The user feeds mono 16kHz float PCM chunks, and Aila returns the incremental stable text and the temporary partial text (which updates as more audio is received).

#### `AilaTranscribeStream`

An opaque struct representing a real-time streaming ASR session.

#### `aila_transcribe_stream_create`

```c
AilaTranscribeStream* aila_transcribe_stream_create(
    AilaEngine* engine,
    const AilaGenConfig* config,
    const char* forced_language,
    const char* system_prompt
);
```

Creates a streaming ASR context. Returns `NULL` on error or if the loaded model does not support ASR.

- `engine` — Initialized engine handle.
- `config` — Generation configuration (can be `NULL` for defaults).
- `forced_language` — Optional language name to force (e.g., `"Chinese"`, `"English"`). Set to `NULL` or `""` for auto-detection.
- `system_prompt` — Optional system prompt to bias the stream transcription. Can be `NULL`.

#### `aila_transcribe_stream_feed`

```c
int aila_transcribe_stream_feed(
    AilaTranscribeStream* stream,
    const float* samples,
    int sample_count
);
```

Feeds raw PCM float 16kHz mono samples into the streaming ASR buffer.

- `stream` — Active stream context.
- `samples` — Array of float audio samples (16kHz, mono).
- `sample_count` — Number of float samples in the array.

Returns `0` (`AILA_OK`) on success, non-zero on error.

#### `aila_transcribe_stream_get_text`

```c
int aila_transcribe_stream_get_text(
    AilaTranscribeStream* stream,
    char** out_stable,
    char** out_partial
);
```

Retrieves the current transcribed text.

- `stream` — Active stream context.
- `out_stable` — [out] Receives a newly allocated UTF-8 string containing the stable (finalized) transcription text. The caller **must** free this string with `aila_free_string()`. Can be `NULL`.
- `out_partial` — [out] Receives a newly allocated UTF-8 string containing the temporary (unstable) transcription text representing the latest active speech chunk. The caller **must** free this string with `aila_free_string()`. Can be `NULL`.

Returns `0` (`AILA_OK`) on success, non-zero on error.

#### `aila_transcribe_stream_destroy`

```c
void aila_transcribe_stream_destroy(AilaTranscribeStream* stream);
```

Destroys the streaming ASR context and frees all associated memory. Passing `NULL` is safe.

### Text-to-Speech (TTS)

Aila supports Text-to-Speech (TTS) synthesis when loaded with a Qwen3-TTS model, with optional **zero-shot voice cloning**.

#### Quick Start: `aila_synthesize` (Recommended)

The simplest API — one call handles everything: tokenization, optional voice cloning, synthesis, and WAV file output. No manual memory management.

```c
// Default voice
aila_synthesize(engine, "Hello world!", NULL, NULL, NULL, NULL, NULL, "output.wav");

// Voice cloning from reference audio
aila_synthesize(engine, "你好世界", "./reference_speaker.wav", NULL, NULL, NULL, NULL, "cloned.wav");

// CustomVoice (named speaker)
aila_synthesize(engine, "Hello!", NULL, "vivian", NULL, NULL, NULL, "output.wav");

// VoiceDesign (style description)
aila_synthesize(engine, "Hello!", NULL, NULL, "A deep, warm voice", NULL, NULL, "output.wav");
```

#### `aila_synthesize`

```c
int aila_synthesize(
    AilaEngine* engine,
    const char* text,
    const char* speaker_audio_path,   // NULL for default voice
    const char* speaker_name,         // NULL or named voice (e.g., "vivian", "ryan")
    const char* instruct_text,        // NULL or VoiceDesign style description
    const char* language,             // NULL/"auto" or language (chinese, english, japanese, korean)
    const AilaGenConfig* config,      // NULL for defaults
    const char* output_wav_path       // output WAV file (24kHz, mono, f32 PCM)
);
```

One-shot TTS synthesis: text + optional voice cloning/style → WAV file. Internally handles speaker embedding extraction (with automatic caching), tokenization, synthesis, and WAV writing in a single call. No `malloc`/`free` needed.

- `text` — UTF-8 text to synthesize.
- `speaker_audio_path` — Optional reference audio (WAV/MP3/FLAC) for voice cloning. Pass `NULL` for default voice. Speaker embeddings are automatically cached in memory and on disk.
- `speaker_name` — Optional named speaker preset for CustomVoice (e.g., `"vivian"`, `"ryan"`). Pass `NULL` to use default or reference audio.
- `instruct_text` — Optional VoiceDesign style description (e.g., `"A deep, warm voice with a slow pace"`). Pass `NULL` if not using VoiceDesign.
- `language` — Optional language override (`"chinese"`, `"english"`, `"japanese"`, `"korean"`, `"auto"`). Pass `NULL` for auto-detection.
- `config` — Generation config (`NULL` for defaults).
- `output_wav_path` — Destination path for the output WAV file.

Returns `0` on success, non-zero on error.

#### Low-Level APIs

For advanced use cases (custom audio post-processing, pre-tokenized input, separate embedding management):

| Function | Description |
|----------|-------------|
| `aila_extract_speaker_embedding` | Extract speaker embedding from reference audio |
| `aila_synthesize_text_to_wav` | Synthesize text → float PCM samples (caller frees with `aila_free_samples`) |
| `aila_synthesize_wav` | Synthesize pre-tokenized tokens → float PCM samples |
| `aila_decode_mimi_vocoder` | Decode discrete Mimi codes → float PCM samples |
| `aila_free_samples` | Free sample/embedding arrays |

#### Streaming TTS API

For real-time streaming speech synthesis with low-latency audio output.

##### `AilaAudioCallback`

```c
typedef void (*AilaAudioCallback)(const float* samples, int sample_count, void* user_data);
```

Callback invoked whenever a chunk of audio samples is ready. The `samples` pointer is valid only during the callback — copy the data if you need to keep it.

- `samples` — Float PCM audio samples (24kHz, mono).
- `sample_count` — Number of samples in this chunk.
- `user_data` — Opaque pointer passed through from the API call.

##### `aila_synthesize_stream`

```c
int aila_synthesize_stream(
    AilaEngine* engine,
    const char* text,
    const char* speaker_audio_path,
    const char* speaker_name,
    const char* instruct_text,
    const char* language,
    const AilaGenConfig* config,
    AilaAudioCallback callback,
    void* user_data
);
```

Starts asynchronous streaming TTS synthesis. Audio chunks are delivered via `callback` as they become available. The function returns immediately — use `aila_stream_wait` to block until synthesis completes.

Parameters match `aila_synthesize` with two additions:
- `callback` — Called for each audio chunk as it is generated. Must not be `NULL`.
- `user_data` — Opaque pointer passed to each callback invocation.

Returns `0` on success (synthesis started), non-zero on error.

##### `aila_stream_wait`

```c
int aila_stream_wait(AilaEngine* engine);
```

Blocks until the current streaming TTS synthesis completes. Returns `0` on success, non-zero on error. Must be called after `aila_synthesize_stream` before starting another stream or destroying the engine.

##### `aila_stream_destroy`

```c
void aila_stream_destroy(AilaEngine* engine);
```

Cancels any in-progress streaming synthesis and frees associated resources. Safe to call at any time. Does not destroy the engine itself.

#### Streaming Example

```c
#include "aila_api.h"

void audio_chunk(const float* samples, int count, void* user_data) {
    // Write PCM samples to audio device or file
    FILE* out = (FILE*)user_data;
    fwrite(samples, sizeof(float), count, out);
}

int main() {
    AilaEngine* engine = aila_engine_create();
    aila_engine_init(engine, "./models/Qwen3-TTS-12Hz-0.6B-CustomVoice", 4096);

    FILE* pcm_out = fopen("output.pcm", "wb");

    int rc = aila_synthesize_stream(engine,
        "Hello world!",        // text
        NULL,                  // speaker_audio_path (default voice)
        "vivian",              // speaker_name (CustomVoice)
        NULL,                  // instruct_text
        NULL,                  // language (auto)
        NULL,                  // config (defaults)
        audio_chunk,           // callback
        pcm_out);              // user_data

    if (rc == 0) {
        aila_stream_wait(engine); // block until done
    }

    fclose(pcm_out);
    aila_stream_destroy(engine);
    aila_engine_destroy(engine);
    return 0;
}
```

#### Reference Audio Requirements

| Requirement | Value |
|-------------|-------|
| **Format** | WAV, MP3, or FLAC |
| **Sample rate** | Any (auto-resampled to 24kHz) |
| **Channels** | Mono recommended (multi-channel is averaged) |
| **Duration** | ≥ 3 seconds of clear speech recommended |
| **Content** | Clean speech from the target speaker (minimal background noise) |
| **Encoding** | 16-bit PCM or 32-bit float |
| **Normalization** | [-1, 1] range, automatically handled |

#### `aila_extract_speaker_embedding`

```c
int aila_extract_speaker_embedding(
    AilaEngine* engine,
    const char* audio_path,
    float** out_embedding,
    int* out_embedding_dim
);
```

Extracts a speaker embedding from a reference audio file for TTS voice cloning. Uses the native C++ ECAPA-TDNN speaker encoder running on CPU — no Python or external dependencies required.

The engine must be initialized with a Qwen3-TTS model (the speaker encoder weights are loaded from the TTS model's `model.safetensors` file). The embedding dimension depends on the model: 1024 for 0.6B, 2048 for 1.7B.

- `engine` — Initialized engine handle (Qwen3-TTS model).
- `audio_path` — Path to the reference audio file (WAV, MP3, or FLAC).
- `out_embedding` — [out] Receives a newly allocated float array containing the speaker embedding. The caller **must** free this array with `aila_free_samples()`.
- `out_embedding_dim` — [out] Receives the dimension of the speaker embedding (1024 for 0.6B, 2048 for 1.7B).

Returns `0` (`AILA_OK`) on success, non-zero on error.

#### `aila_synthesize_wav`

```c
int aila_synthesize_wav(
    AilaEngine* engine,
    const int* text_tokens,
    int text_tokens_len,
    const float* speaker_embedding,
    int speaker_embedding_len,
    const AilaGenConfig* config,
    float** out_samples,
    int* out_sample_count
);
```

Synthesizes raw audio samples from pre-tokenized text token IDs (blocking). Supports zero-shot voice cloning when `speaker_embedding` is provided.

Most users should prefer `aila_synthesize_text_to_wav()` for automatic tokenization.

- `engine` — Initialized engine handle.
- `text_tokens` — Array of text token IDs.
- `text_tokens_len` — Number of token IDs in the array.
- `speaker_embedding` — Optional float array of speaker clone embeddings. Pass the array returned by `aila_extract_speaker_embedding()`, or `NULL` for default voice.
- `speaker_embedding_len` — Length of speaker embedding array (1024 for 0.6B, 2048 for 1.7B, or `0` if `speaker_embedding` is `NULL`).
- `config` — Generation configuration (can be `NULL` for defaults). TTS auto-sets `repetition_penalty` to 1.1 if not explicitly configured to prevent autoregressive collapse.
- `out_samples` — [out] Receives a newly allocated array of float audio samples (24kHz PCM). The caller **must** free this array with `aila_free_samples()`.
- `out_sample_count` — [out] Receives the number of generated audio samples in `out_samples`.

Returns `0` (`AILA_OK`) on success, non-zero on error.

#### `aila_synthesize_text_to_wav`

```c
int aila_synthesize_text_to_wav(
    AilaEngine* engine,
    const char* text,
    const float* speaker_embedding,
    int speaker_embedding_len,
    const AilaGenConfig* config,
    float** out_samples,
    int* out_sample_count
);
```

Synthesizes raw audio samples directly from a UTF-8 text string (blocking). The API automatically handles ChatML layout and tokenization. Supports zero-shot voice cloning.

- `engine` — Initialized engine handle.
- `text` — Null-terminated UTF-8 text prompt to synthesize.
- `speaker_embedding` — Optional float array of speaker clone embeddings (pass `NULL` for default voice). To clone a voice, pass the embedding from `aila_extract_speaker_embedding()`.
- `speaker_embedding_len` — Length of speaker embedding array (1024 for 0.6B, 2048 for 1.7B, or `0` if `speaker_embedding` is `NULL`).
- `config` — Generation configuration (can be `NULL` for defaults).
- `out_samples` — [out] Receives a newly allocated array of float audio samples (24000Hz PCM). The caller **must** free this array with `aila_free_samples()`.
- `out_sample_count` — [out] Receives the number of generated audio samples in `out_samples`.

Returns `0` (`AILA_OK`) on success, non-zero on error.

#### `aila_free_samples`

```c
void aila_free_samples(float* samples);
```

Frees the float audio sample array returned by `aila_synthesize_wav`, `aila_decode_mimi_vocoder`, or `aila_extract_speaker_embedding`. Passing `NULL` is safe.

#### `aila_decode_mimi_vocoder`

```c
int aila_decode_mimi_vocoder(
    AilaEngine* engine,
    const int32_t* codes,
    int n_frames,
    float** out_samples,
    int* out_sample_count
);
```

Decodes discrete acoustic codes (shape `[n_frames, 16]`) back to raw audio samples using the Mimi Vocoder.

- `engine` — Initialized engine handle.
- `codes` — Flat array of discrete codes of shape `[n_frames, 16]`.
- `n_frames` — Number of audio frames.
- `out_samples` — [out] Receives a newly allocated array of decoded float audio samples. The caller **must** free this array with `aila_free_samples()`.
- `out_sample_count` — [out] Receives the number of decoded audio samples.

Returns `0` (`AILA_OK`) on success, non-zero on error.

### Memory Management

#### `aila_free_string`

```c
void aila_free_string(char* str);
```

Frees a string returned by `aila_generate` or `aila_generate_messages`. Passing `NULL` is safe.

### Configuration

#### `AilaGenConfig`

```c
typedef struct {
    int   max_new_tokens;       // default: 512
    float temperature;          // default: 0.6
    int   top_k;                // default: 20
    float top_p;                // default: 0.95
    float repetition_penalty;   // default: 1.0
    float presence_penalty;     // default: 0.0
    float frequency_penalty;    // default: 0.0
    int   do_sample;            // 0 = greedy, 1 = sampling
    int   decode_chunk_size;    // default: 12
    int   stream_chunk_size;    // default: 4
} AilaGenConfig;
```

#### `aila_default_gen_config`

```c
AilaGenConfig aila_default_gen_config(void);
```

Returns a config struct with sensible defaults. All fields can be overridden before passing to generation functions.

### Context Management

#### `aila_engine_reset_context`

```c
void aila_engine_reset_context(AilaEngine* engine);
```

Clears conversation history and KV cache. Use to start a fresh conversation without destroying/recreating the engine.

#### `aila_engine_context_length`

```c
int aila_engine_context_length(AilaEngine* engine);
```

Returns the current context length in tokens (prompt + generated). Returns `0` if engine is `NULL`.

### Error Handling

#### `aila_last_error_code`

```c
int aila_last_error_code(AilaEngine* engine);
```

Returns the error code from the last API call on this engine. Returns `AILA_ERR_INVALID_ARGUMENT` if engine is `NULL`.

#### `aila_last_error_message`

```c
const char* aila_last_error_message(AilaEngine* engine);
```

Returns a human-readable error message. The pointer is valid until the next API call on the same engine. Returns an empty string if engine is `NULL`.

### Error Codes

| Code | Constant | Description |
|------|----------|-------------|
| 0 | `AILA_OK` | Success |
| 1 | `AILA_ERR_INVALID_ARGUMENT` | Invalid argument (NULL pointer, etc.) |
| 2 | `AILA_ERR_TEMPLATE` | Chat template rendering failed |
| 3 | `AILA_ERR_JSON_PARSE` | JSON parse error |
| 4 | `AILA_ERR_VISION_NOT_ENABLED` | Vision content in prompt but vision not enabled |
| 5 | `AILA_ERR_CONTEXT_OVERFLOW` | Prompt exceeds context window |
| 6 | `AILA_ERR_RUNTIME` | Generic runtime error |

### Logging

#### `aila_set_log_callback`

```c
void aila_set_log_callback(AilaLogCallback callback, void* user_data);
```

Sets a global log callback. Pass `NULL` for `callback` to restore default (stdout/stderr) logging.

#### `aila_set_log_level`

```c
void aila_set_log_level(int level);
```

Sets the minimum log level. Messages below this level are suppressed.

| Level | Description |
|-------|-------------|
| 0 | Debug |
| 1 | Info |
| 2 | Warning |
| 3 | Error |

#### `AilaLogCallback`

```c
typedef void (*AilaLogCallback)(int level, const char* message, void* user_data);
```

### Version

#### `aila_version`

```c
const char* aila_version(void);
```

Returns the library version string (e.g. `"0.1.0"`). The pointer is static and must not be freed.

## Language Bindings

### Python (ctypes)

```python
import ctypes, json, os

os.add_dll_directory("./build")
lib = ctypes.cdll.LoadLibrary("./build/AilaShared.dll")

# Define AilaGenConfig
class AilaGenConfig(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("do_sample", ctypes.c_int),
        ("decode_chunk_size", ctypes.c_int),
        ("stream_chunk_size", ctypes.c_int),
    ]

# Bind functions
lib.aila_engine_create.restype = ctypes.c_void_p
lib.aila_engine_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.aila_engine_init.restype = ctypes.c_int
lib.aila_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
lib.aila_generate.restype = ctypes.c_void_p
lib.aila_transcribe.argtypes = [
    ctypes.c_void_p,          # engine
    ctypes.c_char_p,          # wav_path
    ctypes.c_void_p,          # config
    ctypes.c_char_p,          # forced_language
    ctypes.c_char_p,          # system_prompt
    ctypes.c_float,           # segment_sec
    ctypes.c_int,             # past_text_conditioning
    TokenCallback,            # token_callback
    ctypes.c_void_p,          # user_data
    ctypes.POINTER(ctypes.c_char_p) # language_out
]
lib.aila_transcribe.restype = ctypes.c_void_p
lib.aila_transcribe_stream_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lib.aila_transcribe_stream_create.restype = ctypes.c_void_p
lib.aila_transcribe_stream_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.aila_transcribe_stream_feed.restype = ctypes.c_int
lib.aila_transcribe_stream_get_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
lib.aila_transcribe_stream_get_text.restype = ctypes.c_int
lib.aila_transcribe_stream_destroy.argtypes = [ctypes.c_void_p]
lib.aila_transcribe_stream_destroy.restype = None
lib.aila_free_string.argtypes = [ctypes.c_void_p]
lib.aila_engine_destroy.argtypes = [ctypes.c_void_p]

# TTS (simplified)
lib.aila_synthesize.argtypes = [
    ctypes.c_void_p,          # engine
    ctypes.c_char_p,          # text
    ctypes.c_char_p,          # speaker_audio_path
    ctypes.c_char_p,          # speaker_name
    ctypes.c_char_p,          # instruct_text
    ctypes.c_char_p,          # language
    ctypes.c_void_p,          # config
    ctypes.c_char_p,          # output_wav_path
]
lib.aila_synthesize.restype = ctypes.c_int

# TTS streaming
AilaAudioCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p)
lib.aila_synthesize_stream.argtypes = [
    ctypes.c_void_p,          # engine
    ctypes.c_char_p,          # text
    ctypes.c_char_p,          # speaker_audio_path
    ctypes.c_char_p,          # speaker_name
    ctypes.c_char_p,          # instruct_text
    ctypes.c_char_p,          # language
    ctypes.c_void_p,          # config
    AilaAudioCallback,        # callback
    ctypes.c_void_p,          # user_data
]
lib.aila_synthesize_stream.restype = ctypes.c_int
lib.aila_stream_wait.argtypes = [ctypes.c_void_p]
lib.aila_stream_wait.restype = ctypes.c_int
lib.aila_stream_destroy.argtypes = [ctypes.c_void_p]
lib.aila_stream_destroy.restype = None

# TTS (low-level)
lib.aila_extract_speaker_embedding.argtypes = [
    ctypes.c_void_p,          # engine
    ctypes.c_char_p,          # audio_path
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), # out_embedding
    ctypes.POINTER(ctypes.c_int)     # out_embedding_dim
]
lib.aila_extract_speaker_embedding.restype = ctypes.c_int
lib.aila_synthesize_wav.argtypes = [
    ctypes.c_void_p,          # engine
    ctypes.POINTER(ctypes.c_int),    # text_tokens
    ctypes.c_int,             # text_tokens_len
    ctypes.POINTER(ctypes.c_float),  # speaker_embedding
    ctypes.c_int,             # speaker_embedding_len
    ctypes.c_void_p,          # config
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), # out_samples
    ctypes.POINTER(ctypes.c_int)     # out_sample_count
]
lib.aila_synthesize_wav.restype = ctypes.c_int
lib.aila_free_samples.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.aila_free_samples.restype = None
lib.aila_decode_mimi_vocoder.argtypes = [
    ctypes.c_void_p,          # engine
    ctypes.POINTER(ctypes.c_int32), # codes
    ctypes.c_int,             # n_frames
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), # out_samples
    ctypes.POINTER(ctypes.c_int)     # out_sample_count
]
lib.aila_decode_mimi_vocoder.restype = ctypes.c_int

# Use
engine = lib.aila_engine_create()
lib.aila_engine_init(engine, b"./models/qwen3.5-0.8B-bnb-nf4-offline", 4096)

cfg = AilaGenConfig()
lib.aila_default_gen_config(ctypes.byref(cfg))  # or set fields manually
cfg.max_new_tokens = 32
cfg.do_sample = 0

result = lib.aila_generate(engine, b"Hello!", ctypes.byref(cfg))
if result:
    print(ctypes.string_at(result).decode())
    lib.aila_free_string(result)

lib.aila_engine_destroy(engine)
```

##### TTS Voice Cloning (Python)

```python
# 1. Init engine
engine = lib.aila_engine_create()
lib.aila_engine_init(engine, b"./models/Qwen3-TTS-12Hz-0.6B-Base", 4096)

# 2. One-shot synthesis with voice cloning
cfg = AilaGenConfig()
lib.aila_default_gen_config(ctypes.byref(cfg))
rc = lib.aila_synthesize(engine,
    b"Hello world!",                     # text
    b"./reference_speaker.wav",          # speaker_audio_path (NULL for default)
    None,                                # speaker_name (NULL for none)
    None,                                # instruct_text (NULL for none)
    None,                                # language (NULL for auto)
    ctypes.byref(cfg),                   # config
    b"output.wav")                       # output_wav_path
if rc != 0:
    raise RuntimeError(f"TTS failed: {rc}")

# 3. Done — output.wav is ready
lib.aila_engine_destroy(engine)
```

##### Streaming TTS (Python)

```python
pcm_data = bytearray()

@AilaAudioCallback
def on_audio(samples_ptr, sample_count, user_data):
    # Accumulate PCM samples
    buf = ctypes.cast(samples_ptr, ctypes.POINTER(ctypes.c_float * sample_count))
    pcm_data.extend(bytearray(buf.contents))

# Start streaming synthesis
engine = lib.aila_engine_create()
lib.aila_engine_init(engine, b"./models/Qwen3-TTS-12Hz-0.6B-CustomVoice", 4096)

rc = lib.aila_synthesize_stream(engine,
    b"Hello world!",       # text
    None,                  # speaker_audio_path
    b"vivian",             # speaker_name (CustomVoice)
    None,                  # instruct_text
    None,                  # language
    None,                  # config
    on_audio,              # callback
    None)                  # user_data
if rc == 0:
    lib.aila_stream_wait(engine)   # block until complete

lib.aila_stream_destroy(engine)
lib.aila_engine_destroy(engine)
```

See `test_api.py` in the repository root for a complete Python test script.

### C# (P/Invoke)

```csharp
using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
struct AilaGenConfig {
    public int max_new_tokens;
    public float temperature;
    public int top_k;
    public float top_p;
    public float repetition_penalty;
    public float presence_penalty;
    public float frequency_penalty;
    public int do_sample;
    public int decode_chunk_size;
    public int stream_chunk_size;
}

class Aila {
    [DllImport("AilaShared.dll")] static extern IntPtr aila_engine_create();
    [DllImport("AilaShared.dll")] static extern int aila_engine_init(IntPtr e, string dir, int maxSeq);
    [DllImport("AilaShared.dll")] static extern IntPtr aila_generate(IntPtr e, string prompt, ref AilaGenConfig cfg);
    [DllImport("AilaShared.dll")] static extern IntPtr aila_transcribe(
        IntPtr e,
        string wavPath,
        ref AilaGenConfig cfg,
        string forcedLang,
        string systemPrompt,
        float segmentSec,
        int pastTextConditioning,
        AilaTokenCallback tokenCallback,
        IntPtr userData,
        out IntPtr langOut
    );
    [DllImport("AilaShared.dll")] static extern IntPtr aila_transcribe_stream_create(IntPtr e, ref AilaGenConfig cfg, string forcedLang, string systemPrompt);
    [DllImport("AilaShared.dll")] static extern int aila_transcribe_stream_feed(IntPtr s, float[] samples, int sampleCount);
    [DllImport("AilaShared.dll")] static extern int aila_transcribe_stream_get_text(IntPtr s, out IntPtr stableOut, out IntPtr partialOut);
    [DllImport("AilaShared.dll")] static extern void aila_transcribe_stream_destroy(IntPtr s);
    [DllImport("AilaShared.dll")] static extern void aila_free_string(IntPtr s);
    [DllImport("AilaShared.dll")] static extern void aila_engine_destroy(IntPtr e);

    // TTS APIs
    [DllImport("AilaShared.dll")] static extern int aila_synthesize(IntPtr e, string text, string speakerAudioPath, string speakerName, string instructText, string language, ref AilaGenConfig cfg, string outputWavPath);
    [DllImport("AilaShared.dll")] static extern int aila_extract_speaker_embedding(IntPtr e, string audioPath, out IntPtr outEmbedding, out int outEmbeddingDim);
    [DllImport("AilaShared.dll")] static extern int aila_synthesize_wav(IntPtr e, int[] textTokens, int textTokensLen, float[] speakerEmbedding, int speakerEmbeddingLen, ref AilaGenConfig cfg, out IntPtr outSamples, out int outSampleCount);
    [DllImport("AilaShared.dll")] static extern void aila_free_samples(IntPtr samples);
    [DllImport("AilaShared.dll")] static extern int aila_decode_mimi_vocoder(IntPtr e, int[] codes, int nFrames, out IntPtr outSamples, out int outSampleCount);

    // TTS Streaming
    delegate void AilaAudioCallback(IntPtr samples, int sampleCount, IntPtr userData);
    [DllImport("AilaShared.dll")] static extern int aila_synthesize_stream(IntPtr e, string text, string speakerAudioPath, string speakerName, string instructText, string language, ref AilaGenConfig cfg, AilaAudioCallback callback, IntPtr userData);
    [DllImport("AilaShared.dll")] static extern int aila_stream_wait(IntPtr e);
    [DllImport("AilaShared.dll")] static extern void aila_stream_destroy(IntPtr e);
    // ... etc
}
```

### Rust (FFI)

```rust
use std::ffi::{c_char, c_int, c_void, CStr, CString};

#[repr(C)]
struct AilaGenConfig { /* ... same fields ... */ }

extern "C" {
    fn aila_engine_create() -> *mut c_void;
    fn aila_engine_init(engine: *mut c_void, model_dir: *const c_char, max_seq: c_int) -> c_int;
    fn aila_generate(engine: *mut c_void, prompt: *const c_char, config: *const AilaGenConfig) -> *mut c_char;
    fn aila_transcribe(
        engine: *mut c_void,
        wav_path: *const c_char,
        config: *const AilaGenConfig,
        forced_language: *const c_char,
        system_prompt: *const c_char,
        segment_sec: f32,
        past_text_conditioning: c_int,
        token_callback: AilaTokenCallback,
        user_data: *mut c_void,
        language_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn aila_transcribe_stream_create(
        engine: *mut c_void,
        config: *const AilaGenConfig,
        forced_language: *const c_char,
        system_prompt: *const c_char,
    ) -> *mut c_void;
    fn aila_transcribe_stream_feed(stream: *mut c_void, samples: *const f32, sample_count: c_int) -> c_int;
    fn aila_transcribe_stream_get_text(stream: *mut c_void, out_stable: *mut *mut c_char, out_partial: *mut *mut c_char) -> c_int;
    fn aila_transcribe_stream_destroy(stream: *mut c_void);
    fn aila_free_string(s: *mut c_char);
    fn aila_engine_destroy(engine: *mut c_void);

    // TTS APIs
    fn aila_synthesize(
        engine: *mut c_void,
        text: *const c_char,
        speaker_audio_path: *const c_char,
        speaker_name: *const c_char,
        instruct_text: *const c_char,
        language: *const c_char,
        config: *const AilaGenConfig,
        output_wav_path: *const c_char,
    ) -> c_int;
    fn aila_extract_speaker_embedding(
        engine: *mut c_void,
        audio_path: *const c_char,
        out_embedding: *mut *mut f32,
        out_embedding_dim: *mut c_int,
    ) -> c_int;
    fn aila_synthesize_wav(
        engine: *mut c_void,
        text_tokens: *const c_int,
        text_tokens_len: c_int,
        speaker_embedding: *const f32,
        speaker_embedding_len: c_int,
        config: *const AilaGenConfig,
        out_samples: *mut *mut f32,
        out_sample_count: *mut c_int,
    ) -> c_int;
    fn aila_free_samples(samples: *mut f32);
    fn aila_decode_mimi_vocoder(
        engine: *mut c_void,
        codes: *const i32,
        n_frames: c_int,
        out_samples: *mut *mut f32,
        out_sample_count: *mut c_int,
    ) -> c_int;

    // TTS Streaming
    type AilaAudioCallback = extern "C" fn(samples: *const f32, sample_count: c_int, user_data: *mut c_void);
    fn aila_synthesize_stream(
        engine: *mut c_void,
        text: *const c_char,
        speaker_audio_path: *const c_char,
        speaker_name: *const c_char,
        instruct_text: *const c_char,
        language: *const c_char,
        config: *const AilaGenConfig,
        callback: AilaAudioCallback,
        user_data: *mut c_void,
    ) -> c_int;
    fn aila_stream_wait(engine: *mut c_void) -> c_int;
    fn aila_stream_destroy(engine: *mut c_void);
}
```

## Thread Safety

AilaEngine instances are **not thread-safe**. Each thread should create its own engine, or external synchronization must be provided. Multiple engines can operate independently in parallel.
