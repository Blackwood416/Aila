# Alia-Specific Aila Engine Implementation Roadmap

Date: 2026-06-11

Branch: `codex/alia-custom-engine`

## Goal

Build the Alia-specific multi-model runtime inside Aila. This branch can shed
generic one-model API constraints as the Alia runtime takes over.

## Implementation Principles

- Make `AliaContext*` the primary runtime handle.
- Keep current CLI and generic C API behavior only as a temporary migration aid
  when that helps keep the branch buildable.
- Land the ABI skeleton early so Alia Host can stop depending on the mock DLL.
- Prefer small, testable changes over a large runtime rewrite.
- Treat hard interruption and cleanup as correctness requirements.
- Treat final latency numbers as performance milestones after correctness.

## Phase 0: Baseline And Build Guard

Purpose: ensure the branch starts from a known state.

Expected files touched:

- `CMakeLists.txt`
- `tests/chat/*` only if the test target needs include/link updates

Actions:

1. Run the existing lightweight chat tests.
2. Run a DLL build of `AilaShared`.
3. Record any baseline failures before Alia runtime changes.
4. Add future `src/alia` files to `LIB_SOURCES` only when each file is created.

Acceptance:

- Existing generic API still compiles until the Alia runtime has enough coverage
  to replace it deliberately.
- `AilaChatTests` still passes.
- `AilaShared` builds.

## Phase 1: Alia ABI Skeleton

Purpose: export the `alia_*` functions used by Alia Host with deterministic
argument validation and cleanup.

Expected files touched:

- Create `include/alia_api.h`
- Modify `include/aila_api.h`
- Create `src/alia/AliaError.hpp`
- Create `src/alia/AliaContext.hpp`
- Create `src/alia/AliaContext.cpp`
- Create `src/alia/AliaApi.cpp`
- Modify `CMakeLists.txt`

Behavior:

- `alia_context_init` allocates an opaque `AliaContext`.
- `alia_context_destroy` is null-safe and idempotent for partially initialized
  contexts.
- All other `alia_*` functions validate arguments and return stable error codes.
- Callback pointers can be registered and stored but are not invoked until later
  phases.

Tests:

- Add a lightweight C++ ABI test target or extend the existing test target with
  no-model Alia lifecycle tests.
- Validate null input paths:
  - null `out_ctx`
  - null `ctx`
  - invalid sample count
  - repeated destroy path through helper cleanup

Acceptance:

- The real DLL exports the same symbols as the Alia mock.
- Alia Host can load the DLL far enough to call `alia_context_init` and destroy
  the handle in a no-model or fake-model test mode.

## Phase 2: Shared Runtime Context

Purpose: provide one selected device/context with foreground and background
execution lanes.

Expected files touched:

- Create `src/alia/RuntimeContext.hpp`
- Create `src/alia/RuntimeContext.cpp`
- Modify `src/core/Context.hpp`
- Modify `src/core/Tensor.hpp` only if constructor/accessor changes are needed

Behavior:

- `RuntimeContext` selects one SYCL device.
- Foreground and background lanes share one `sycl::context`.
- Each lane owns its own queue and oneDNN stream.
- Existing backend calls can still receive a `Context&` compatibility object.
- Allocation accounting is per lane and aggregated at `AliaContext`.

Tests:

- Construct and destroy `RuntimeContext`.
- Verify foreground/background queues report the same SYCL context.
- Allocate/free a small tensor through each lane.

Acceptance:

- No accidental regression in generic engine creation during this migration
  phase.
- The Alia runtime can create both lanes without loading models.

## Phase 3: Model Slot Loading

Purpose: load ASR, foreground VLM, background VLM, and TTS as separate slots
under one Alia context.

Expected files touched:

- Create `src/alia/ModelSlot.hpp`
- Create `src/alia/ModelSlot.cpp`
- Modify `src/alia/AliaContext.*`
- Modify `include/engine/Engine.hpp` only if reusable loading helpers must be
  extracted from `InferenceEngine`
- Potentially create `src/engine/EngineLoadHelpers.*` if extraction becomes too
  large for header-only code

Behavior:

- Each model slot owns tokenizer, weights, backend, and model-specific helpers.
- Slots use the shared runtime lanes instead of creating independent devices.
- ASR and TTS load on the foreground lane.
- 4B VLM loads on the foreground lane.
- 0.8B VLM loads on the background lane.
- Partial initialization cleans up loaded slots if a later model fails.

Tests:

- Fake path failure reports the model slot that failed.
- In a model-asset environment, load each slot individually.
- In a full asset environment, initialize all slots in one `AliaContext`.

Acceptance:

- `alia_context_init` can load all requested models or fail cleanly.
- Generic `aila_engine_init` remains unchanged for existing callers.

## Phase 4: ASR Pipeline

Purpose: implement `alia_asr_feed_audio`, `alia_asr_get_text`, and
`alia_asr_reset`.

Expected files touched:

- Create `src/alia/AliaAsrPipeline.hpp`
- Create `src/alia/AliaAsrPipeline.cpp`
- Modify `src/alia/AliaContext.*`
- Modify `src/alia/AliaApi.cpp`

Behavior:

- `alia_asr_feed_audio` appends f32 16 kHz mono samples.
- `alia_asr_get_text` returns newly allocated UTF-8 stable and partial strings.
- Returned strings use the existing `aila_free_string` allocator contract or a
  documented Alia-compatible free path.
- `alia_asr_reset` clears audio and text state.

Tests:

- Feed silence and reset without crashing.
- Confirm returned text pointers are null or valid allocated strings.
- Use existing ASR model smoke tests when model assets are available.

Acceptance:

- Alia Host can call ASR feed/get/reset against the real DLL.
- No C++ exception crosses the ABI boundary.

## Phase 5: Foreground Turn Worker

Purpose: implement `alia_start_conversation_turn` as a native worker that drives
VLM generation and TTS output.

Expected files touched:

- Create `src/alia/AliaForegroundPipeline.hpp`
- Create `src/alia/AliaForegroundPipeline.cpp`
- Create `src/alia/AliaTtsPipeline.hpp`
- Create `src/alia/AliaTtsPipeline.cpp`
- Modify `src/alia/AliaContext.*`
- Modify `src/alia/AliaApi.cpp`
- Potentially extract generation stepping helpers from `InferenceEngine`

Behavior:

- A conversation turn runs on a controlled worker thread.
- The turn reads the latest ASR stable text or an internal prompt buffer.
- The 4B VLM generates text with `GenerationConfig` translated from Alia config.
- Normal assistant text stays inside C++.
- Initial TTS milestone may synthesize sentence/clause chunks.
- Audio chunks are returned through `AliaAudioCallback`.

Tests:

- Fake VLM/TTS pipeline can emit deterministic audio chunks.
- Starting a second turn while one is active returns invalid-state.
- Callback exceptions are impossible at C ABI level; callback return/absence is
  handled defensively.

Acceptance:

- Alia Host can receive audio callbacks from the real runtime path.
- Turn lifecycle has clear states: idle, running, aborting, completed, failed.

## Phase 6: Tool-Call Pause And Resume

Purpose: let the foreground VLM invoke host tools without exposing ordinary text
over FFI.

Expected files touched:

- Modify `src/alia/AliaForegroundPipeline.*`
- Reuse `src/chat/StructuredStreamParser.*`
- Reuse `src/chat/ChatJson.*`
- Potentially add `src/alia/AliaToolBridge.*`

Behavior:

- Streamed model text is fed into `StructuredStreamParser`.
- Tool-call events are serialized as JSON and passed to `AliaToolCallCallback`.
- The callback writes result JSON/text into the provided output buffer.
- The native turn appends tool results to the model session and resumes.
- Malformed tool calls become model-visible tool errors, not DLL crashes.

Tests:

- Simulated tool call invokes callback exactly once.
- Oversized callback result is truncated safely and marked as error.
- Null tool callback returns a stable error to the model or aborts the turn based
  on policy.

Acceptance:

- Tool calls no longer require Alia Host to parse normal assistant text.
- The 4B model can produce a final spoken response after tool result injection.

## Phase 7: Hard Interruption And KV Rollback

Purpose: make `alia_abort_inference` and `alia_vlm_rollback_kv_cache` correct.

Expected files touched:

- Modify `src/alia/AliaContext.*`
- Modify `src/alia/AliaForegroundPipeline.*`
- Modify `src/alia/AliaTtsPipeline.*`
- Modify `src/models/IModelBackend.hpp` only if a clearer rollback API is needed
- Modify Qwen3.5 backend files only if replay-to-anchor needs new hooks

Behavior:

- Abort flags are atomic and per pipeline mask.
- Foreground decode checks abort between token steps.
- TTS checks abort between text chunks, codec batches, and Mimi batches.
- Turn start records a generation-start rollback anchor.
- Rollback truncates VLM state to the anchor.
- Qwen3.5 Hybrid uses snapshot restore plus replay if necessary.

Tests:

- Abort before turn start is harmless.
- Abort during fake decode stops audio callbacks quickly.
- Rollback after generated tokens restores expected context length.
- Qwen3.5 snapshot fallback path is covered with a deterministic fake or small
  model-backed test.

Acceptance:

- `alia_abort_inference` returns quickly.
- No worker thread is left running after destroy.
- Interrupted turns do not pollute foreground conversation state.

## Phase 8: Background 0.8B Pipeline

Purpose: implement asynchronous memory extraction without disturbing the
foreground VLM.

Expected files touched:

- Create `src/alia/AliaBackgroundPipeline.hpp`
- Create `src/alia/AliaBackgroundPipeline.cpp`
- Modify `src/alia/AliaContext.*`
- Modify `src/alia/AliaApi.cpp`

Behavior:

- `alia_register_background_callback` stores callback and user data.
- `alia_trigger_background_processing` starts a background job if one is not
  already running.
- The job uses the background lane and 0.8B model slot.
- Output is constrained to JSON text and sent through the registered callback.
- Destroy waits for or cancels background work.

Tests:

- Trigger without callback returns invalid-state or success-with-no-callback
  according to the final API policy.
- Trigger while busy returns invalid-state.
- Fake background output invokes callback once.

Acceptance:

- Alia Host can receive graph/preference JSON from the real callback path.
- Foreground turn state is untouched by background extraction.

## Phase 9: Performance And Soak Validation

Purpose: measure the real system and tune after correctness.

Expected files touched:

- Add or extend perf scripts under `perf/`
- Add Alia-specific smoke script if useful, for example
  `tests/alia_runtime_smoke.ps1`
- Update `docs/Environment_Variables.md` for new Alia runtime knobs

Metrics:

- TTFT from VAD fall to first audio callback.
- VLM 4B decode tokens per second.
- Abort call latency.
- Time from abort to last audio callback.
- Background extraction duration.
- Peak allocation by slot and lane.

Acceptance:

- Baseline metrics are recorded in repeatable scripts.
- Regressions can be compared before deeper optimization work.

## Later Stage: Computer Use Vision Pipeline

Purpose: satisfy the PRD's second-phase screen understanding requirements.

Likely additions:

- `alia_vlm_feed_image`
- GPU texture or shared image buffer abstraction
- WGC interop contract with Alia Host
- YOLO/SAM or separate native vision extension
- Entity list serialization into VLM prompt context

This stage should start only after the core audio conversation runtime is
stable, because it adds external graphics interop and a new model family.

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Shared context refactor destabilizes kernels | High | Keep compatibility `Context` wrapper and migrate one slot at a time. |
| TTS cannot consume tiny text chunks naturally | High | Start with clause-buffered synthesis, then add deeper incremental TTS support. |
| Qwen3.5 rollback misses recurrent state | High | Use existing snapshots and replay fallback; add explicit anchor tests. |
| Background queue still contends with foreground | Medium | Chunk background decode, keep it cancellable, add queue priority hints after correctness. |
| Alia and generic `AilaGenConfig` layouts diverge | Medium | Use a dedicated Alia ABI config translation path and ABI tests. |
| Full model tests are too asset-heavy for CI | Medium | Keep fake-pipeline tests in CI and gate model smoke tests behind explicit paths. |

## Documentation Updates Required During Implementation

- Update `docs/C_API.md` once `alia_*` exports are real.
- Update `docs/Environment_Variables.md` for Alia runtime options.
- Keep this directory as the design and roadmap home for the branch.
- Do not place branch planning material under `docs/superpowers/`.
