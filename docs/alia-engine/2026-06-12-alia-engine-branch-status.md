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
- `tests/alia/AliaApiTestMain.cpp` (optional diagnostic target, not the default
  branch verification path)
- `tools/alia/AliaRealModelSmoke.cpp`
- `tools/alia/RunAliaTargetPipeline.ps1`
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
- `AilaAliaRealSmoke` now defaults to the fixed Alia target model set:
  `Qwen3-ASR-1.7B-BNB-NF4`,
  `qwen3.5-4B-bnb-nf4-offline-visiondense`,
  `qwen3.5-0.8B-bnb-nf4-offline`, and
  `Qwen3-TTS-12Hz-0.6B-Base`. The smoke enforces those model directory names by
  default and requires `--allow-non-target-models` for exploratory probes.
- If the request WAV is missing, `AilaAliaRealSmoke` now uses the target
  Qwen3-TTS slot to synthesize `--request-text` into that WAV path before
  feeding the generated audio into ASR. This keeps the full-pipeline smoke
  self-contained after `tmp/` cleanup.
- `tools/alia/RunAliaTargetPipeline.ps1` is the branch-local verification
  entrypoint. It configures/builds `AliaEngine` with generic CLI, generic API,
  generic chat tests, and lightweight Alia API tests disabled, then runs the
  fixed four-model full pipeline.
- `CMakeLists.txt` now treats `AliaEngine` as the custom branch aggregate
  target. Default configure builds `AilaShared` plus `AilaAliaRealSmoke`; the
  generic CLI, generic chat tests, lightweight Alia API tests, and generic
  `aila_*` shared API surface are opt-in.
- The default `AilaShared.dll` export surface is now Alia ABI only when
  `AILA_BUILD_GENERIC_API=OFF`.
- Noisy TTS/Mimi debug tensor dumps, `mimi_output.wav` debug emission, and a
  Qwen3.5 debug load log were moved behind explicit debug-level logging or
  `AILA_TTS_DEBUG` / `AILA_TTS_DEBUG_WAV`.
- Loaded background JSON gets a narrow post-validation cleanup pass for the
  fixed flow: array values are deduplicated, one-shot completed hello requests
  are removed from `memory_candidates` and `tasks`, and summaries that mistake
  `Alia,` as the speaker are normalized back to `User`.
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
  Markdown-fence normalization and a narrow cleanup pass for duplicate/completed
  one-shot items. Broader memory quality should still be tuned with real
  multi-turn prompts instead of API-only tests.
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
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; cmake -S . -B build -DAILA_BUILD_GENERIC_API=OFF -DAILA_BUILD_GENERIC_CLI=OFF -DAILA_BUILD_GENERIC_CHAT_TESTS=OFF -DAILA_BUILD_ALIA_API_TESTS=OFF -DAILA_BUILD_ALIA_REAL_SMOKE=ON; cmake --build build --target AliaEngine --config Release"
```

Result:

- Passed.
- The default branch build rebuilt `AilaLib`, `AilaAliaRealSmoke`, and
  `AilaShared`; it did not build the generic CLI, generic chat tests, or the
  lightweight Alia API tests.
- After the build directory was reconfigured, a full rebuild completed. It still
  reports unrelated pre-existing Jinja warnings from `src/templates/jinja`, but
  the Alia/TTS changes compile cleanly.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; dumpbin /exports build\AilaShared.dll | Select-String 'alia_|aila_'"
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
- No generic `aila_*` exports were present with `AILA_BUILD_GENERIC_API=OFF`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -AudioPath tmp\alia-real-smoke\alia_autogen_request.wav -OutputWav tmp\alia-real-smoke\alia_full_pipeline_autogen.wav -LogPath tmp\alia-real-smoke\alia_full_pipeline_autogen.log
```

Result:

- Passed with `ALIA_REAL_MODEL_SMOKE_PASS`.
- Target model enforcement: `true`.
- Loaded ASR `Qwen3-ASR-1.7B-BNB-NF4`, foreground
  `qwen3.5-4B-bnb-nf4-offline-visiondense`, background
  `qwen3.5-0.8B-bnb-nf4-offline`, and TTS
  `Qwen3-TTS-12Hz-0.6B-Base`.
- The request WAV was intentionally absent before the run. The smoke generated
  it through the target TTS model:
  `request_audio_generated=true`, callback count `8`, samples `90240`.
- Model load time: `27236ms`.
- ASR time: `1700ms`.
- ASR partial text: `Alia, please say hello in one short sentence.`
- Foreground time: `5015ms`.
- Foreground decode mode: `LoadedVlm`.
- Foreground assistant text: `Hello!`
- TTS emitted `2` callbacks, first audio at `4767ms`, `21120` total samples,
  `21099` nonzero samples, chunk sizes `11520,9600`.
- Background time: `902ms`.
- Background decode mode: `LoadedVlm`.
- Background schema valid: `true`, retry count `0`, repair applied `false`.
- Background diagnostic: `initial background JSON accepted; post cleanup applied`.
- Background JSON:
  `{"summary":"User asked to say hello in one short sentence, and the assistant replied with 'Hello!'.","memory_candidates":[],"preferences":[],"tasks":[]}`
- Full log: `tmp\alia-real-smoke\alia_full_pipeline_autogen.log`.

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
- Git warned that modified files will be converted from LF to CRLF the next
  time Git touches them.

## Recommended Next Implementation Order

1. Add one more real-model flow smoke for a non-trivial user turn that should
   produce a durable preference or unresolved task, so background cleanup is
   proven on something richer than the hello request.
2. Calibrate Qwen3-TTS callback cadence/TTFT against the PRD 100ms audio-chunk
   target. The fixed four-model smoke emits real audio, but first audio is still
   about `4.7s` after foreground turn start.
3. Add hard-interruption timing probes with the fixed 4B foreground and TTS
   assets.
4. Investigate and fix the foreground 4B rollback crash before treating
   `alia_vlm_rollback_kv_cache` as production-ready with Qwen3.5 Hybrid 4B.
5. Migrate the 4B visiondense image path into the Alia foreground pipeline once
   the audio/text full pipeline latency is under control.

## Review Notes

The architecture direction is intentionally Alia-specific. The generic Aila API
can remain temporarily while this branch migrates, but it should not constrain
the final runtime shape.
