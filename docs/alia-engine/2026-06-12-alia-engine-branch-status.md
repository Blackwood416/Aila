# Alia Engine Branch Status

Date: 2026-06-12

Branch: `codex/alia-custom-engine`

## Purpose

This document records the current handoff state for the Alia-specific Aila
engine branch. The design and roadmap documents in this directory define the
target architecture; this status note identifies what is already present in the
worktree, what remains intentionally incomplete, and which verification commands
were run.

## Document Location

All branch-local planning material for this work lives under:

- `docs/alia-engine/`

Do not place Alia engine planning material under `docs/superpowers/`. That
directory is ignored on the main branch and is reserved for separate planning
material.

## Current Worktree Additions

Public ABI:

- `include/alia_api.h`

Alia runtime implementation files:

- `src/alia/RuntimeContext.hpp`
- `src/alia/RuntimeContext.cpp`
- `src/alia/ModelSlot.hpp`
- `src/alia/ModelSlot.cpp`
- `src/alia/AliaContext.hpp`
- `src/alia/AliaContext.cpp`
- `src/alia/AliaApi.cpp`
- `src/alia/AliaAsrPipeline.hpp`
- `src/alia/AliaAsrPipeline.cpp`
- `src/alia/AliaForegroundPipeline.hpp`
- `src/alia/AliaForegroundPipeline.cpp`
- `src/alia/AliaTtsPipeline.hpp`
- `src/alia/AliaTtsPipeline.cpp`
- `src/alia/AliaBackgroundPipeline.hpp`
- `src/alia/AliaBackgroundPipeline.cpp`

Build and test integration:

- `CMakeLists.txt`
- `tests/alia/AliaApiTestMain.cpp`
- `src/core/Context.hpp`

## Implemented Foundation

The branch currently contains a real Alia ABI skeleton and an initial
multi-pipeline runtime shape:

- `alia_*` C ABI declarations are separated into `include/alia_api.h`.
- `AliaContext` owns the Alia runtime state and four model slots.
- `RuntimeContext` creates foreground and background execution lanes over one
  shared SYCL context/device.
- `ModelSlot` loads metadata, selects supported backend kinds, and fails cleanly
  when required assets are absent.
- ASR, foreground, TTS, and background pipeline classes exist with lifecycle and
  validation behavior.
- ASR can commit stable UTF-8 text into the native pipeline state for the
  foreground turn to consume without FFI text marshalling.
- Foreground turns validate the compact Alia generation config and translate it
  into the internal `GenerationConfig` shape.
- Foreground turns record the current user text and translated generation config
  as native session state, establishing the boundary where the real VLM decode
  path will attach.
- Foreground turns now prefer the loaded Qwen3.5 Hybrid VLM slot when it is
  ready: the pipeline builds an Alia-specific system prompt, applies the
  tokenizer chat template, performs a prefill plus token decode loop through
  `IModelBackend::forward`, and decodes assistant text natively.
- Foreground assistant text is parsed natively for `<tool_call>` blocks. Tool
  calls are serialized through the existing chat JSON helpers and passed to
  `AliaToolCallCallback`; spoken text is kept separate so tool JSON is not sent
  into TTS.
- Loaded foreground VLM generation now also feeds decoded token deltas through
  `StructuredStreamParser` during the initial decode pass. As soon as a complete
  tool-call block is detected, the pipeline advances backend state for that
  emitted token and pauses before sampling further ordinary text.
- Foreground pipeline state records the last tool-call JSON and host tool
  result text, creating the native state boundary needed for the later
  pause/resume decode loop.
- Tool callback results are promoted into a native resume prompt containing the
  captured user request, assistant tool-call JSON, and host tool result for
  diagnostics. When a loaded foreground VLM slot is available, the actual
  resume pass now appends compact `<tool_result>` continuation tokens instead
  of re-encoding a fresh chat scaffold. The loaded backend is reset for the
  initial turn prefill only, and the resume pass keeps the current VLM session
  and original generation-start rollback anchor.
- Loaded foreground VLM content deltas are now streamed toward TTS during the
  decode loop. Sentence-like content chunks are enqueued and synthesized as soon
  as they are complete, before later assistant tokens are sampled; the no-model
  fallback still uses the deterministic whole-text chunking path.
- Foreground abort now has an explicit `Aborted` terminal state. If abort is
  requested while a TTS chunk callback is in flight, the pipeline stops before
  synthesizing remaining spoken chunks after the callback returns.
- `alia_vlm_rollback_kv_cache` now has stateful behavior instead of acting as a
  no-op: positive rollbacks require a loaded foreground VLM generation anchor,
  and loaded backends are asked to truncate KV state through
  `IModelBackend::truncate_kv_cache`.
- Background processing now requires a registered result callback, records an
  Alia-specific JSON extraction prompt, reports whether it used no-model
  fallback or a loaded VLM slot, and has a real loaded background VLM decode
  entry point for 0.8B memory extraction.
- Background results now use a stable Alia memory-extraction JSON shape with
  `summary`, `memory_candidates`, `preferences`, and `tasks`. Fallback output
  uses that schema, and loaded-model output is parsed with `simdjson` before it
  is accepted: malformed JSON, missing fields, or wrong required field types are
  wrapped into a schema-repair result that preserves the raw model output.
- Loaded background VLM extraction now gets one guided retry before schema
  repair wrapping. If the first 0.8B output is malformed or has wrong required
  field types, the pipeline builds a repair prompt containing the invalid output
  and asks the loaded background slot for strict schema JSON again.
- Background schema decisions are now recorded as native diagnostics: retry
  count, whether the final callback result used the schema-repair wrapper, and a
  short diagnostic string for initial acceptance, retry acceptance, or retry
  failure followed by repair wrapping.
- The deterministic no-model foreground response is now explicitly marked as
  `NoModelFallback` state for ABI/lightweight tests, not treated as normal
  model inference.
- The no-model path can initialize, destroy, feed/reset ASR state, run
  deterministic foreground/TTS callback behavior, and invoke background
  callbacks.
- `AilaShared.dll` exports the Alia ABI symbols expected by Alia Host.

## Known Incomplete Areas

The current implementation is not yet the full PRD runtime. The remaining
product work is concentrated in these areas:

- Foreground VLM generation still needs full multi-turn prompt/session
  ownership, deeper multi-tool continuation coverage with real model assets, and
  async overlap between VLM decode and real TTS synthesis.
- TTS still needs real Qwen3-TTS plus Mimi streaming synthesis from queued text.
- Background processing still needs model-asset smoke validation for real 0.8B
  extraction output.
- Abort handling is wired at the API, worker lifecycle, and TTS chunk boundary,
  but hard latency guarantees still require model-step cancellation checks and
  timing tests.
- Selective KV rollback still needs full Qwen3.5 Hybrid recurrent-state
  snapshot/replay validation and exact generation-start restoration tests with
  real model assets.
- Computer Use, WGC texture injection, YOLO/SAM entity extraction, and related
  low-latency vision routing remain later-stage work.

## Fresh Verification

Run from `E:\RiderProjects\Aila` with the oneAPI environment initialized through
`perf/PerfCommon.ps1`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; cmake --build build --target AilaAliaApiTests AilaChatTests AilaShared --config Release"
```

Result:

- `AilaShared.dll` was rebuilt and relinked after the foreground
  token-time tool-call pause, tool-result continuation-token resume,
  token-time foreground-to-TTS forwarding, foreground abort-state, rollback
  state, and background VLM prompt/type-level schema validation plus guided
  retry/diagnostic changes.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; .\build\AilaAliaApiTests.exe"
```

Result:

- `Alia API tests passed`, including background schema accept/repair coverage
  for valid JSON, malformed JSON, wrong required field types, loaded background
  VLM guided retry before schema repair, retry/repair diagnostics, and loaded
  foreground VLM tool-result resume coverage that verifies only the initial turn
  prefill resets the backend session, resume prefill avoids the chat scaffold,
  initial decode stops when a complete tool call is emitted, and a complete
  spoken sentence reaches TTS before later assistant tokens are sampled.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; .\build\AilaChatTests.exe"
```

Result:

- `AilaChatTests: 225 passed, 0 failed`

```powershell
dumpbin /exports build\AilaShared.dll | Select-String 'alia_'
```

Result:

- `alia_abort_inference`
- `alia_asr_feed_audio`
- `alia_asr_get_text`
- `alia_asr_reset`
- `alia_context_destroy`
- `alia_context_init`
- `alia_register_background_callback`
- `alia_start_conversation_turn`
- `alia_trigger_background_processing`
- `alia_vlm_rollback_kv_cache`

```powershell
git diff --check
```

Result:

- No whitespace errors were reported.
- Git warned that `CMakeLists.txt` and `src/core/Context.hpp` will be converted
  from LF to CRLF the next time Git touches them.

## Recommended Next Implementation Order

1. Add model-asset smoke validation for background 0.8B JSON extraction output.
2. Wire real TTS synthesis behind the existing text queue and callback boundary.
3. Add hard-interruption timing tests and model-step cancellation checks.
4. Expand selective KV rollback from backend truncation to Qwen3.5 Hybrid
   recurrent-state snapshot/replay validation.

## Review Notes

The architecture direction is intentionally Alia-specific. The generic Aila API
can remain temporarily while this branch migrates, but it should not constrain
the final runtime shape.
