# Background Memory Extractor Split Design

Date: 2026-07-09

## Goal

Move Alia background memory extraction out of the realtime GPU model-slot path
so ASR, foreground VLM, and TTS keep GPU memory and compute priority. Background
memory extraction may complete slowly, but it must not block the next voice
turn.

The long-term target is a native in-process CPU backend for the fixed
Qwen3.5 0.8B background model. The initial architecture should make that backend
possible without exposing an external protocol to Alia Host.

## Current State

`AliaBackgroundPipeline` currently owns the background worker thread and also
directly calls the loaded background `ModelSlot`. That slot loads the 0.8B
Qwen3.5 Hybrid NF4 model into a SYCL/oneDNN GPU `Context`.

`RuntimeContext` creates separate foreground and background queues, but both
queues target the same SYCL device and context. This isolates queue ordering, not
GPU residency. The background model therefore still consumes GPU memory,
runtime buffers, KV/cache memory, and GPU scheduling bandwidth.

There is no existing CPU-only Qwen3.5 backend. `Qwen35HybridTextBackend` is also
a SYCL/oneDNN backend and is not a CPU-resident path.

## Architecture

Introduce an internal extraction boundary:

```text
AliaBackgroundPipeline
  -> IBackgroundMemoryExtractor
       -> GpuVlmBackgroundExtractor
       -> NativeCpuQ35BackgroundExtractor
```

`AliaBackgroundPipeline` owns async job scheduling, state, callback delivery,
and last-result diagnostics. It no longer contains tokenization or generation
loops.

`IBackgroundMemoryExtractor` owns the model-specific extraction operation:

```cpp
struct BackgroundExtractionRequest {
    std::string chat_turn_text;
};

struct BackgroundExtractionResult {
    bool ok = false;
    std::string result_json;
    std::string prompt_text;
    std::string error;
    int schema_retry_count = 0;
    bool schema_repair_applied = false;
    std::string schema_diagnostic;
};

class IBackgroundMemoryExtractor {
public:
    virtual ~IBackgroundMemoryExtractor() = default;
    virtual bool ready() const = 0;
    virtual const char* backend_name() const = 0;
    virtual BackgroundExtractionResult extract(
        const BackgroundExtractionRequest& request,
        const std::atomic_bool& abort_requested) = 0;
};
```

The existing prompt construction, schema repair, JSON normalization, and cleanup
logic should move behind a shared helper used by extractor implementations. That
keeps output behavior stable when switching from GPU to CPU.

## Initial Implementation

The first implementation should be structural and behavior-preserving:

- Add `IBackgroundMemoryExtractor`.
- Add `GpuVlmBackgroundExtractor` that wraps the existing `ModelSlot` path.
- Move the current `generate_with_loaded_vlm` logic into the GPU extractor.
- Keep current C API behavior unless explicitly changed by config.
- Keep the current output schema and cleanup behavior unchanged.
- Add focused tests for state transitions and extractor error propagation using
  a fake extractor.

This phase should not implement the native CPU inference kernels.

## Queue Semantics

Replace the current "busy means reject" model with a bounded internal queue.

- `alia_trigger_background_processing` enqueues and returns quickly.
- The worker processes one job at a time.
- The default queue capacity should be small, such as 8 or 16.
- If the queue is full, the trigger call returns an error and records a
  diagnostic. It must not block waiting for capacity.
- `request_abort` cancels the active job and clears queued work.
- `join` waits for the worker to exit as it does today.

Callback delivery remains asynchronous. The callback should receive the final
schema-valid JSON for each completed job.

## Load And Failure Semantics

Realtime slots remain required:

- ASR load failure fails context initialization.
- Foreground VLM load failure fails context initialization.
- TTS load failure fails context initialization.

Background extraction should become optional once the native CPU extractor path
exists:

- If the background extractor fails to load, context initialization can still
  succeed.
- Triggering background work while no extractor is ready should return an
  unavailable error and update diagnostics.
- Background extraction failures must not affect foreground turn state.

During the initial GPU-adapter phase, the existing init behavior may remain
strict to avoid changing host behavior before the optional-background contract is
documented in the C API.

## Native CPU Backend Scope

`NativeCpuQ35BackgroundExtractor` should be purpose-built for background memory
extraction:

- Fixed Qwen3.5 0.8B text-only model.
- Native in-process C++ implementation.
- CPU-resident weights and KV/state.
- No vision, tool calls, LoRA, or foreground streaming.
- Deterministic greedy generation.
- Background extraction token cap around the current 384-token behavior.
- Internal Alia API only; no external inference protocol.

The CPU backend may use model-specific simplifications that would be wrong for
a general chatbot backend, as long as extraction output quality is preserved.

## Testing And Validation

Initial extractor split:

- Unit tests with a fake extractor for enqueue, queue-full, abort, callback, and
  error propagation.
- Existing background schema tests should keep passing through the GPU adapter.
- Real short Chinese smoke should pass with GPU adapter enabled and should show
  unchanged background JSON validity.

Native CPU backend:

- CPU extractor smoke on several Chinese turn texts.
- Compare GPU adapter and CPU backend JSON schema validity and rough content on
  the voice scenario matrix.
- Run voice pipeline matrix to confirm foreground TTFA and playback gaps do not
  regress while CPU background work is active.
- Measure GPU memory at load time to confirm the 0.8B background GPU model is no
  longer resident.

## Non-Goals

- Do not pursue Computer Use, visual input, or tool-call behavior for the
  background path.
- Do not make background extraction block the foreground turn.
- Do not expose a new external inference protocol to Alia Host.
- Do not generalize the CPU backend beyond the fixed background model until the
  memory-extraction path is stable.
