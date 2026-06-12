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
- `tools/alia/AliaRealModelSmoke.cpp`
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
- The TTS pipeline now prefers a loaded TTS backend streaming path for queued
  spoken text. `AliaTtsPipeline` formats assistant text for Qwen3-TTS,
  tokenizes it through the loaded TTS slot, and forwards backend audio chunks to
  `AliaAudioCallback`; no-model, unsupported-backend, and empty-token cases
  keep the deterministic fallback audio behavior for lightweight tests.
- `Qwen3TTSBackend` now exposes the Alia TTS streaming hook by delegating to its
  existing `synthesize_codes_stream` plus Mimi incremental decoder path using
  the default voice, no instruct prompt, and automatic language mode.
- Foreground abort now has an explicit `Aborted` terminal state. If abort is
  requested while a TTS chunk callback is in flight, the pipeline stops before
  synthesizing remaining spoken chunks after the callback returns.
- Foreground abort is now also propagated into the loaded TTS backend streaming
  path. `AliaTtsPipeline::synthesize_pending` accepts a cancellation predicate,
  foreground generation passes `abort_requested()`, and `Qwen3TTSBackend`
  checks cancellation during codec generation and between Mimi streaming
  batches so long synthesis work can unwind before emitting fallback audio.
- Foreground abort is now propagated into loaded VLM backend `forward` calls.
  `IModelBackend` exposes a cancellation checker plus `ModelBackendCancelled`,
  `AliaForegroundPipeline` installs `abort_requested()` while a loaded VLM turn
  is active, and `Qwen35HybridBnb4Backend` checks cancellation at forward entry,
  layer boundaries, before LM head projection, and before recurrent-state
  snapshots.
- `alia_vlm_rollback_kv_cache` now has stateful behavior instead of acting as a
  no-op: positive rollbacks require a loaded foreground VLM generation anchor,
  and loaded backends are asked to truncate KV state through
  `IModelBackend::truncate_kv_cache`.
- Foreground rollback now records the initial loaded-VLM prompt tokens and
  generated token IDs from the anchored turn. If a backend cannot truncate
  exactly, or restores an earlier checkpoint than requested, the pipeline resets
  and replays the prompt plus the required generated-token prefix to restore the
  requested context length before reporting rollback success.
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
- Loaded background VLM output now normalizes common real-model Markdown JSON
  fences before schema validation. This allows strict JSON returned inside
  Markdown code fences to be accepted without falling into the repair wrapper
  path.
- Background schema decisions are now recorded as native diagnostics: retry
  count, whether the final callback result used the schema-repair wrapper, and a
  short diagnostic string for initial acceptance, retry acceptance, or retry
  failure followed by repair wrapping.
- Foreground and background Qwen3.5 prompts now append the same closed-think
  prefix used by the general 0.8B chat path (`<think>...</think>`) instead of
  relying on the lightweight `/no_think` suffix. Real 0.8B/4B runs showed that
  the suffix path can emit only reasoning, malformed tool-call scaffolds, or
  immediate EOS, while the closed-think prompt produces spoken content and
  strict background JSON.
- `AilaAliaRealSmoke` is a model-asset smoke executable for the Alia runtime.
  It loads ASR, foreground VLM, background VLM, and TTS slots in one
  `AliaContext`, feeds real audio, waits for async foreground/background work,
  records ASR text, foreground response, TTS chunk timing/samples, background
  JSON diagnostics, and writes the resulting TTS WAV.
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
  better async overlap between VLM decode and real TTS synthesis.
- TTS has real Qwen3-TTS plus Mimi streaming smoke coverage, but callback
  cadence/TTFT is still too slow for the PRD target and needs calibration.
  Host voice control inputs and real-asset cancellation timing proof remain.
- Background processing has real 0.8B extraction smoke coverage, including
  Markdown-fence normalization, but schema quality still needs prompt tuning so
  extracted `preferences` and `tasks` avoid echoing schema instructions.
- Abort handling is wired at the API, worker lifecycle, TTS callback boundary,
  loaded TTS backend streaming path, and loaded VLM backend forward path, but
  hard latency guarantees still require timing tests with real assets.
- Selective KV rollback is only partially validated with real assets. A
  foreground 0.8B + background 0.8B smoke survived a one-token rollback, but a
  foreground 4B + background 0.8B run crashed during subsequent background
  generation after rollback. The same 4B full pipeline passes when rollback is
  skipped, so rollback must remain a separate investigation item.
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
  state plus prompt replay fallback, loaded TTS backend streaming hook plus
  cancellation propagation, loaded VLM backend forward cancellation propagation,
  and background VLM prompt/type-level schema validation plus guided
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
  spoken sentence reaches TTS before later assistant tokens are sampled. The
  tests also cover a loaded TTS slot using the backend streaming hook rather
  than deterministic fallback audio, plus foreground abort propagation into a
  loaded TTS backend stream before any audio callback is emitted, and foreground
  abort propagation into a loaded VLM backend `forward` call. Rollback coverage
  includes backend reset fallback that replays the saved prompt to the
  generation anchor.

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
pwsh -NoProfile -ExecutionPolicy Bypass -File .\smoke.ps1 -Preset phase_gate_q35_text -CaseNames q35_08_bnb_nf4_smoke -OutputDir tmp\alia-real-smoke\baseline-q35
```

Result:

- Passed with `models/qwen3.5-0.8B-bnb-nf4-offline`, returning
  `1+1 等于 2。`.
- The run exercised the real Qwen3.5 Hybrid BnB4 backend and recurrent-state
  replay path (`KV decode overrun ... replaying accepted tail`).

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; .\build\Aila.exe -m .\models\Qwen3-ASR-0.6B-BNB-NF4 --transcribe '.\This is an English test.wav' --forced-lang English --log-level info"
```

Result:

- Passed with `This is an English test.` and `[Language] English`.
- Reported `Audio: 2.1s`, `Latency: 1635ms`, `Speed: 1.3x`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; .\build\Aila.exe -m .\models\Qwen3-TTS-12Hz-0.6B-Base --synthesize 'Hello from Alia real model smoke.' --output-wav .\tmp\alia-real-smoke\tts_base_smoke.wav --language english --log-level info --max-tokens 48 --stream-tts --stream-batch 6"
```

Result:

- Passed. Qwen3-TTS Base and Mimi vocoder loaded successfully.
- Streaming output was saved to
  `tmp\alia-real-smoke\tts_base_smoke.wav` (`161324` bytes).

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command '. .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; $env:AILA_INIT_WARMUP="0"; .\build\Aila.exe -m .\models\Qwen3-TTS-12Hz-0.6B-Base --synthesize "Alia, please say hello in one short sentence." --output-wav .\tmp\alia-real-smoke\alia_request.wav --language english --log-level error --max-tokens 48'
```

Result:

- Passed. This generated the real audio request used by the Alia full-pipeline
  smoke (`tmp\alia-real-smoke\alia_request.wav`, `307244` bytes).

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command '. .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; $env:AILA_INIT_WARMUP="0"; $env:AILA_LOG_LEVEL="info"; .\build\AilaAliaRealSmoke.exe --asr-model .\models\Qwen3-ASR-0.6B-BNB-NF4 --foreground-model .\models\qwen3.5-0.8B-bnb-nf4-offline --background-model .\models\qwen3.5-0.8B-bnb-nf4-offline --tts-model .\models\Qwen3-TTS-12Hz-0.6B-Base --audio ".\tmp\alia-real-smoke\alia_request.wav" --output-wav .\tmp\alia-real-smoke\alia_full_pipeline_0p8.wav --max-seq 2048 --max-tokens 48 --timeout-sec 900'
```

Result:

- Passed with `ALIA_REAL_MODEL_SMOKE_PASS`.
- ASR partial text: `Alia, please say hello in one short sentence.`
- Foreground decode mode: `LoadedVlm`
- Foreground assistant text:
  `Hello! I'm Alia, and I'm here to help you with your needs.`
- TTS emitted `9` callbacks, first audio at about `4149ms`, `99840` total
  samples, `99425` nonzero samples.
- Background decode mode: `LoadedVlm`
- Background schema valid: `true`, retry count `0`, repair applied `false`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command '. .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; $env:AILA_INIT_WARMUP="0"; $env:AILA_LOG_LEVEL="info"; .\build\AilaAliaRealSmoke.exe --asr-model .\models\Qwen3-ASR-0.6B-BNB-NF4 --foreground-model .\models\qwen3.5-4B-bnb-nf4-offline --background-model .\models\qwen3.5-0.8B-bnb-nf4-offline --tts-model .\models\Qwen3-TTS-12Hz-0.6B-Base --audio ".\tmp\alia-real-smoke\alia_request.wav" --output-wav .\tmp\alia-real-smoke\alia_full_pipeline_4b_no_rollback.wav --max-seq 2048 --max-tokens 48 --timeout-sec 1200 --rollback-tokens 0'
```

Result:

- Passed with `ALIA_REAL_MODEL_SMOKE_PASS`.
- ASR partial text: `Alia, please say hello in one short sentence.`
- Foreground 4B decode mode: `LoadedVlm`
- Foreground assistant text: `Hello!`
- TTS emitted `2` callbacks, first audio at about `4339ms`, `21120` total
  samples, `21099` nonzero samples.
- Background 0.8B decode mode: `LoadedVlm`
- Background schema valid: `true`, retry count `0`, repair applied `false`.

4B rollback probe:

- The same 4B full-pipeline command with an explicit rollback probe
  (`--rollback-tokens 1`) completed ASR, foreground 4B, TTS, and rollback, then
  crashed with access violation while background 0.8B generation was starting.
- The successful `--rollback-tokens 0` run above isolates this as a
  rollback-after-4B issue, not a general 4B full-pipeline load or generation
  issue.

```powershell
git diff --check
```

Result:

- No whitespace errors were reported.
- Git warned that the modified branch-status doc, foreground pipeline files,
  and Alia API test file will be converted from LF to CRLF the next time Git
  touches them.

## Recommended Next Implementation Order

1. Investigate and fix the foreground 4B rollback crash before treating
   `alia_vlm_rollback_kv_cache` as production-ready with Qwen3.5 Hybrid 4B.
2. Calibrate Qwen3-TTS callback cadence/TTFT against the PRD 100ms audio-chunk
   target; the current full-pipeline smoke emits real audio but first audio is
   still roughly 4.1-4.3s after foreground turn start.
3. Add hard-interruption timing tests with real VLM/TTS assets.
4. Improve background extraction prompt quality so `preferences` and `tasks`
   contain user/product facts instead of schema-task echoes.
5. Expand real full-pipeline smoke to multi-turn and tool-result continuation
   once the voice path stays stable.

## Review Notes

The architecture direction is intentionally Alia-specific. The generic Aila API
can remain temporarily while this branch migrates, but it should not constrain
the final runtime shape.
