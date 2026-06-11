# Alia-Specific Aila Engine Architecture Design

Date: 2026-06-11

Branch: `codex/alia-custom-engine`

## 1. Purpose

Alia Host already binds to a dedicated `alia_*` C ABI and currently uses
`E:\RiderProjects\Alia\scratch\mock_aila.cpp` as a simulated `AilaShared.dll`.
The purpose of this design is to make the real Aila C++ engine satisfy the
Alia PRD as a fully customized local companion runtime. This branch is allowed
to leave the generic, one-model Aila API behind as implementation progresses.

The target is not a thin wrapper around the current single-model engine. The
target is an Alia-specific multi-model runtime inside `AilaShared.dll`:

- ASR pipeline: Qwen3-ASR, fed by 16 kHz mono f32 PCM chunks.
- Foreground VLM pipeline: Qwen3.5 4B, NF4, vision-capable, low-latency turn
  handling and tool-call interception.
- Background VLM pipeline: Qwen3.5 0.8B, NF4, asynchronous memory extraction.
- TTS pipeline: Qwen3-TTS plus Mimi vocoder, returning 24 kHz mono f32 PCM.
- Shared SYCL device/context with separated foreground and background queues.

## 2. Source Inputs

Primary product requirement:

- `E:\RiderProjects\Alia\docs\design\aila_engine_prd.md`

Related Alia architecture references:

- `E:\RiderProjects\Alia\docs\design\architecture.md`
- `E:\RiderProjects\Alia\docs\design\detailed_design.md`
- `E:\RiderProjects\Alia\docs\design\adr.md`
- `E:\RiderProjects\Alia\docs\dev\current_status.md`

Current Aila implementation references:

- `include/aila_api.h`
- `src/api/aila_api.cpp`
- `include/engine/Engine.hpp`
- `src/core/Context.hpp`
- `src/memory/KVCache.hpp`
- `src/models/Qwen35HybridBnb4Backend.*`
- `src/models/Qwen3ASRBackend.*`
- `src/models/Qwen3TTSBackend.*`
- `src/chat/StructuredStreamParser.*`

## 3. Current-State Summary

The existing Aila engine is optimized and useful, but its ownership model is
single-model-centric:

- `InferenceEngine` loads one model directory and owns one `Context`.
- `Context` creates one default in-order `sycl::queue`, one oneDNN engine, and
  one oneDNN stream.
- The public C ABI exposes a generic `AilaEngine*` handle and generic functions
  such as `aila_engine_init`, `aila_generate_chat_json_stream_ex`,
  `aila_transcribe_stream_*`, and `aila_synthesize_stream`.
- Qwen3.5 Hybrid, Qwen3-ASR, and Qwen3-TTS backends already exist.
- Tool-call parsing exists for the generic chat stream path, but the engine does
  not execute or pause for tools.
- KV cache truncation exists. Qwen3 dense/ASR use simple length truncation.
  Qwen3.5 Hybrid has recurrent state snapshots for DeltaNet/linear layers and
  can restore to a checkpoint or force reset.
- TTS "streaming" currently generates all codec tokens first, then decodes Mimi
  audio in batches. This is useful but does not yet satisfy the PRD's direct VLM
  token to TTS streaming pipeline.

The PRD requires a different top-level runtime shape: one Alia context owns all
models and orchestrates them as pipelines.

## 4. Non-Goals For The First Usable Runtime

The first usable Alia runtime should not attempt every PRD item at once.

Excluded from the first usable runtime:

- YOLO/SAM screen entity extraction.
- WGC GPU texture direct injection.
- Fully tuned queue priority hints beyond separate queue ownership.
- Final TTFT tuning to the 400 ms target across all model assets.
- New model formats beyond those already supported by Aila.

These are not rejected requirements. They are later stages after the `alia_*`
ABI, multi-model loading, interruption, and foreground/background turn flow are
working.

## 5. Target Runtime Shape

Add a new Alia-specific runtime as the primary engine shape:

```text
AliaContext
  |
  +-- RuntimeContext
  |     +-- sycl::device
  |     +-- sycl::context
  |     +-- foreground queue + oneDNN stream
  |     +-- background queue + oneDNN stream
  |     +-- allocator / memory accounting
  |
  +-- AliaAsrPipeline
  |     +-- model slot: Qwen3-ASR
  |     +-- streaming audio buffer
  |     +-- stable/partial text state
  |
  +-- AliaForegroundPipeline
  |     +-- model slot: Qwen3.5-4B
  |     +-- prompt/session state
  |     +-- turn rollback anchors
  |     +-- tool-call parser and callback bridge
  |
  +-- AliaTtsPipeline
  |     +-- model slot: Qwen3-TTS
  |     +-- text queue
  |     +-- audio callback bridge
  |
  +-- AliaBackgroundPipeline
        +-- model slot: Qwen3.5-0.8B
        +-- async worker
        +-- background result callback bridge
```

The Alia runtime should reuse lower-level components where they are stable, but
it should not force Alia's lifecycle through the generic one-model handle. The
generic `InferenceEngine` path can remain during migration to keep the project
buildable, but it is not a long-term architecture constraint on this branch.

## 6. Public ABI

The real `AilaShared.dll` should export the Alia ABI already used by Alia Host:

```c
typedef struct AliaContext AliaContext;

typedef struct {
    float temperature;
    float top_p;
    int max_tokens;
} AilaGenConfig;

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

int alia_context_init(
    AliaContext** out_ctx,
    const char* asr_model_dir,
    const char* vlm_4b_model_dir,
    const char* vlm_0_8b_model_dir,
    const char* tts_model_dir,
    int max_seq_len);

void alia_context_destroy(AliaContext* ctx);
int alia_abort_inference(AliaContext* ctx, int pipeline_mask);
int alia_vlm_rollback_kv_cache(AliaContext* ctx, int rollback_tokens);
int alia_asr_feed_audio(AliaContext* ctx, const float* samples, int sample_count);
void alia_asr_reset(AliaContext* ctx);
int alia_asr_get_text(AliaContext* ctx, char** out_stable, char** out_partial);
void alia_register_background_callback(AliaContext* ctx, AliaBackgroundResultCallback callback);
int alia_trigger_background_processing(AliaContext* ctx, const char* chat_turn_text);
int alia_start_conversation_turn(
    AliaContext* ctx,
    const AilaGenConfig* config,
    AliaToolCallCallback tool_cb,
    AliaAudioCallback audio_cb,
    void* user_data);
```

The Alia ABI should live in the same public header or a new included header.
During early migration it may coexist with generic Aila API symbols. Once the
Alia runtime covers the Host requirements, this branch can remove or demote the
generic API surface.

The existing `AilaGenConfig` in `include/aila_api.h` has a different field order
and richer fields. The Alia ABI must define an ABI-stable Alia-facing config and
translate it internally to `GenerationConfig`.

## 7. Runtime Context And Queues

Current `Context` constructs its own device, context, queue, oneDNN engine, and
oneDNN stream. For Alia, this needs to become reusable.

Recommended direction:

- Introduce a shared `RuntimeContext` that owns one selected `sycl::device` and
  one `sycl::context`.
- Create two `ExecutionLane` objects over that shared context:
  - foreground lane: ASR, 4B VLM, TTS
  - background lane: 0.8B memory extraction
- Each lane owns its own `sycl::queue` and oneDNN stream.
- Add a compatibility wrapper so existing backend code can continue accepting a
  `Context&` while the underlying queue/context is supplied by the Alia runtime.

Queue priority should be treated as an optimization, not the first correctness
requirement. If Intel SYCL priority hints are available and stable in the target
oneAPI toolchain, foreground/background lanes can use them. Otherwise the engine
still gets meaningful isolation from separate queues plus cooperative scheduling
and cancellation.

## 8. Memory Model

Aila already uses SYCL USM allocation APIs. The current `Context::alloc_device`
uses `sycl::malloc_device`, which is USM device allocation. The PRD's robustness
goal also calls for unified memory behavior under pressure.

Recommended allocator policy:

- Weights and high-throughput kernels default to device USM for performance.
- Cross-pipeline buffers, host-observed state, and image/audio staging buffers
  use shared or host USM where zero-copy or CPU visibility matters.
- KV cache allocation remains arena-style and contiguous per model, with an
  explicit dtype policy.
- Memory accounting must be per `AliaContext` and per model slot so load failure
  and cleanup are deterministic.

FP8 KV cache is partially present. `KVCache` and Qwen3.5 Hybrid cache allocation
already support `dnnl::memory::data_type::f8_e4m3` through `AILA_KV_QUANT`, and
attention/copy paths contain f8 handling. The Alia runtime should expose this as
an explicit runtime option rather than relying only on an environment variable.

## 9. Pipeline Responsibilities

### 9.1 ASR Pipeline

Responsibilities:

- Own the ASR model slot and stream state.
- Accept `alia_asr_feed_audio` PCM chunks.
- Maintain stable and partial text.
- Provide `alia_asr_get_text`.
- Reset cleanly on `alia_asr_reset` and interruption.

Existing reuse:

- `InferenceEngine::TranscribeStream`
- `Qwen3ASRBackend`
- `Qwen3ASRBnb4Backend`
- `Qwen3ASRAudioEncoder`

Implementation note:

The current ASR streaming state processes text when `get_text` is called and
decodes larger chunks. That is acceptable for an initial ABI-compatible version,
but lower-latency ASR-to-VLM prefill should later move toward a worker-driven
incremental pipeline.

### 9.2 Foreground VLM Pipeline

Responsibilities:

- Own Qwen3.5 4B model state and tokenizer/formatter.
- Maintain the foreground conversation cache.
- Record a rollback anchor immediately before assistant generation begins.
- Stream decoded text into the tool-call parser and TTS input queue.
- If a tool call is detected, pause generation, call `AliaToolCallCallback`, add
  the tool result back into the prompt/session state, and resume generation.
- Keep normal assistant text inside C++ without FFI marshalling.

Existing reuse:

- Qwen3.5 Hybrid text/BNB4 backend.
- Chat formatter and structured stream parser.
- Tool-call JSON conversion helpers.
- Incremental prefill logic and `cached_ids_` concept.

Implementation note:

The existing generic stream parser emits tool-call events, but the generic API
returns them to the caller. The Alia runtime must turn those events into a
synchronous pause/resume inside the native turn worker.

### 9.3 TTS Pipeline

Responsibilities:

- Own Qwen3-TTS and Mimi state.
- Consume text emitted by the foreground VLM.
- Produce 24 kHz f32 mono PCM chunks via `AliaAudioCallback`.
- Stop quickly on abort.

Existing reuse:

- `Qwen3TTSBackend`
- Mimi incremental decoder state.
- `synthesize_codes_stream` as a stepping stone.

Implementation note:

The current TTS streaming method first generates all codec tokens, then streams
Mimi audio. This is not sufficient for the final PRD. The first version can use
sentence or clause chunking as an integration milestone, but final behavior
needs a TTS text queue that begins synthesis while VLM generation is still in
progress.

### 9.4 Background VLM Pipeline

Responsibilities:

- Own Qwen3.5 0.8B model state.
- Run memory extraction on the background lane.
- Enforce JSON-only output using a dedicated prompt.
- Invoke `AliaBackgroundResultCallback` when complete.
- Avoid touching foreground KV/session state.

Existing reuse:

- Qwen3.5 Hybrid backend.
- Generic chat JSON generation.
- Structured output helpers from the chat layer.

## 10. Interruption And Rollback

Interruption is a first-class runtime contract, not an afterthought.

Recommended model:

- `AliaContext` owns atomic cancellation state per pipeline mask.
- `alia_abort_inference(ctx, mask)` sets cancellation flags immediately and
  wakes any text/audio queues.
- Foreground generation checks cancellation between decode steps and before
  invoking callbacks.
- TTS checks cancellation between text chunks, codec batches, and Mimi batches.
- Background processing checks cancellation before starting and between decode
  chunks. Foreground abort should not destroy the background worker unless the
  mask requests it.

KV rollback:

- On turn start, record `turn_prompt_len` and `turn_generation_start_len`.
- For normal text models, rollback truncates cache and `cached_ids_` to the
  generation start anchor.
- For Qwen3.5 Hybrid, rollback must account for DeltaNet recurrent state. The
  existing snapshot/restore mechanism should be reused. If no exact checkpoint
  exists, restore the nearest safe checkpoint and replay tokens up to the anchor.
- If replay fails, fall back to full reset and prefill of retained history. That
  is slower but correct.

The exported `alia_vlm_rollback_kv_cache(ctx, rollback_tokens)` should support
the PRD's explicit request, but the interruption path should also be able to
rollback to the stored generation-start anchor without trusting the host to
calculate token counts perfectly.

## 11. Error Handling

All `alia_*` functions should return stable integer error codes and catch C++
exceptions at the FFI boundary.

Recommended error categories:

- success
- invalid argument
- invalid state
- model load failure
- runtime failure
- aborted
- context overflow
- callback failure

The existing `AILA_ERR_*` values can be reused where compatible, with additional
Alia-specific values if needed. No exception should escape from `AilaShared.dll`
through P/Invoke.

## 12. Migration Strategy

The final branch target is Alia-specific. Compatibility is useful only as a
short migration aid:

- `AliaContext*` is the primary opaque handle.
- Existing CLI and chat tests may continue to build while the runtime is being
  replaced, but generic behavior is not a product requirement for this branch.
- `src/api/aila_api.cpp` can host both generic and Alia exports initially, but
  larger
  implementation should move into focused files under `src/alia/`.

Recommended new source layout:

```text
src/alia/
  AliaApi.cpp
  AliaContext.hpp
  AliaContext.cpp
  RuntimeContext.hpp
  RuntimeContext.cpp
  AliaAsrPipeline.hpp
  AliaAsrPipeline.cpp
  AliaForegroundPipeline.hpp
  AliaForegroundPipeline.cpp
  AliaTtsPipeline.hpp
  AliaTtsPipeline.cpp
  AliaBackgroundPipeline.hpp
  AliaBackgroundPipeline.cpp
  AliaError.hpp
```

Public declarations can be added to `include/aila_api.h` or split into
`include/alia_api.h` and included by `include/aila_api.h`.

## 13. Test Strategy

Initial tests should focus on ABI and orchestration because full model tests are
asset-heavy.

Recommended test layers:

- Lightweight C++ ABI tests using fake pipeline/model slots.
- Chat/tool parser tests using existing `tests/chat`.
- Alia API lifecycle tests: init failure, destroy, invalid args, callback
  retention, abort idempotence.
- Rollback unit tests using a fake backend that records context length.
- Optional smoke tests gated by model paths for real ASR, 4B, 0.8B, and TTS.
- Alia Host integration test replacing `scratch/mock_aila.cpp` with the real
  DLL once the ABI skeleton is present.

## 14. Major Risks

### Shared context refactor risk

Backends currently assume a `Context&` with a single queue. Refactoring too much
at once can destabilize kernels. Mitigation: introduce compatibility wrappers
and move model slots one at a time.

### True VLM-to-TTS streaming risk

Qwen3-TTS currently prefers complete text/token input. Full token-level
streaming may require sentence buffering or backend-level incremental text
conditioning. Mitigation: ship a clause-buffered milestone first, then optimize.

### Qwen3.5 Hybrid rollback risk

Linear recurrent state means cache rollback is not only a pointer update.
Mitigation: use existing snapshots, add explicit turn anchors, and implement
replay fallback.

### GPU concurrency risk

Separate queues do not automatically guarantee foreground priority. Mitigation:
keep background jobs cancellable, chunk background decode, and add queue priority
hints only after correctness is stable.

### ABI mismatch risk

Alia Host already uses `AilaGenConfig { Temperature, TopP, MaxNewTokens }`, while
Aila's generic config has a different layout. Mitigation: keep a dedicated Alia
config translation path and add ABI tests.

## 15. Recommended First Milestone

The first milestone should prove that the real `AilaShared.dll` can replace the
mock DLL at the ABI level:

- `alia_context_init` loads or validates all requested model slots.
- `alia_context_destroy` releases every slot cleanly.
- ASR feed/get/reset exports exist.
- `alia_start_conversation_turn` starts a worker and can emit audio callback
  chunks.
- `alia_abort_inference` stops the worker without crashing.
- `alia_register_background_callback` and
  `alia_trigger_background_processing` run an async callback path.

This milestone may use conservative internal sequencing. It must establish the
public contract and cleanup discipline before deeper latency optimization.
