# Alia Engine Branch Status

Date: 2026-06-16

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
  as they are complete, before later assistant tokens are sampled.
- Foreground turns now record real voice-pipeline timing markers for the main
  loaded 4B generation: prompt token count, generated token count, first
  decoded content delta, and first TTS enqueue.
- Ordinary voice turns now guard against malformed structured-control leakage
  from the identity LoRA. If a non-tool user turn starts emitting
  `<tool_call>` markup, the foreground decode stops early, the markup is
  removed from spoken text, and the background memory turn receives only
  natural-language assistant text. Explicit host-tool requests still use the
  normal tool-call path.
- The TTS pipeline now prefers a loaded TTS backend streaming path for queued
  spoken text. `AliaTtsPipeline` formats assistant text for Qwen3-TTS,
  tokenizes it through the loaded TTS slot, and forwards backend audio chunks
  to `AliaAudioCallback`. Missing or non-emitting TTS backends fail the product
  path instead of emitting deterministic placeholder audio.
- Foreground and TTS now overlap on the real product path. A per-turn TTS
  worker starts before foreground decode, consumes spoken chunks as the 4B model
  emits sentence-like content, and is drained before background extraction.
  Synchronous TTS remains available for request-audio generation.
- TTS chunk boundaries now handle UTF-8 CJK sentence punctuation in addition to
  ASCII punctuation, reducing unnecessary first-chunk size for Chinese persona
  replies.
- The TTS path now records real first-chunk backend timings: codec generation,
  Mimi init, first audio, total backend time, emitted frames, callback count,
  and first audio sample count. `AilaAliaRealSmoke` and the voice matrix CSV
  both report these metrics.
- `Qwen3TTSBackend` now exposes the Alia TTS streaming hook by delegating to its
  existing `synthesize_codes_stream` plus Mimi incremental decoder path using
  the default voice, no instruct prompt, and automatic language mode.
- Qwen3-TTS codec generation now has an optional frame callback. The streaming
  path initializes Mimi before codec generation and feeds completed codec frame
  batches directly into `decode_mimi_incremental`, so first audio no longer
  waits for the full spoken chunk's codec sequence.
- The Alia TTS stream schedule now uses uniform `12`-frame codec/Mimi batches.
  A previous first-small/steady-large schedule improved measured TTFA, but was
  rejected because the short first audio buffer can finish playback before the
  second batch is ready. The product path now favors playback continuity until
  Mimi incremental-state work or an explicit playback buffer scheduler can
  reduce latency without under-run risk.
- Foreground abort now has an explicit `Aborted` terminal state. If abort is
  requested while a TTS chunk callback is in flight, the pipeline stops before
  synthesizing remaining spoken chunks after the callback returns.
- Foreground abort is now also propagated into the loaded TTS backend streaming
  path. `AliaTtsPipeline::synthesize_pending` accepts a cancellation predicate,
  foreground generation passes `abort_requested()`, and `Qwen3TTSBackend`
  checks cancellation during codec generation and between Mimi streaming
  batches so long synthesis work can unwind cleanly.
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
  Alia-specific JSON extraction prompt, and uses the real loaded background VLM
  decode entry point for 0.8B memory extraction.
- Background results now use a stable Alia memory-extraction JSON shape with
  `summary`, `memory_candidates`, `preferences`, and `tasks`. Loaded-model
  output is parsed with `simdjson` before it is accepted: malformed JSON,
  missing fields, or wrong required field types are wrapped into a schema-repair
  result that preserves the raw model output.
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
  default.
- The foreground Qwen3.5 4B slot can now load the Alia identity LoRA from
  `F:\unsloth\qwen35_4b_alia_identity_r16_lr1e5\checkpoint-1400` when
  configured through `AliaContext::vlm_4b_lora_dir` or
  `AilaAliaRealSmoke --foreground-lora`. The Qwen3.5 Hybrid NF4 backend applies
  PEFT q/k/v/o adapters to full-attention layers, including fused QKV row
  offsets and the gated full-attention `o_proj` input shape.
- `LoraLoader` now accepts the PEFT prefix shape emitted by the identity LoRA
  (`base_model.model.model.language_model.layers...`) and normalizes it to the
  local `model.layers...` weight names used by backend LoRA attachment code.
- Foreground tool-call parsing now tolerates one real-model near miss observed
  during LoRA smoke: `<parameter=id=42</parameter>` is recovered as
  `{"id":"42"}` when it appears inside an otherwise valid function tool block.
- If the request WAV is missing, `AilaAliaRealSmoke` now uses the target
  Qwen3-TTS slot to synthesize `--request-text` into that WAV path before
  feeding the generated audio into ASR. This keeps the full-pipeline smoke
  self-contained after `tmp/` cleanup.
- `tools/alia/RunAliaTargetPipeline.ps1` is the branch-local verification
  entrypoint. It configures/builds `AliaEngine`, then runs the fixed four-model
  full pipeline with the identity LoRA and real foreground tool-call probe.
- `tools/alia/RunAliaVoiceScenarioMatrix.ps1` runs the fixed real voice
  scenarios and writes `tmp\alia-real-smoke\voice_matrix\summary.csv`. It
  skips the dedicated tool probe by default so voice timing regressions are not
  hidden by tool-call stochasticity; pass `-IncludeToolProbe` when intentionally
  stress-testing that path per scenario.
- `CMakeLists.txt` now treats `AliaEngine` as the custom branch aggregate
  target. Default configure builds only `AilaShared` plus
  `AilaAliaRealSmoke`; generic CLI/API/test targets and the lightweight Alia
  API unit-test executable have been removed from this custom branch.
- The default `AilaShared.dll` export surface is now Alia ABI only.
- Noisy TTS/Mimi debug tensor dumps, `mimi_output.wav` debug emission, and a
  Qwen3.5 debug load log were moved behind explicit debug-level logging or
  `AILA_TTS_DEBUG` / `AILA_TTS_DEBUG_WAV`.
- Loaded background JSON gets a narrow post-validation cleanup pass for the
  fixed flow: array values are deduplicated, one-shot completed hello requests
  are removed from `memory_candidates` and `tasks`, and summaries that mistake
  `Alia,` as the speaker are normalized back to `User`.
- `AilaShared.dll` exports the Alia ABI symbols expected by Alia Host.

## Known Incomplete Areas

The current implementation is not yet the full PRD runtime. The remaining
product work is concentrated in these areas:

- Foreground VLM generation still needs full multi-turn prompt/session
  ownership, deeper multi-tool continuation coverage with real model assets, and
  faster first-token/content latency. The current real voice matrix shows first
  content at roughly `3.6s` to `3.8s` after foreground turn start.
- TTS has real Qwen3-TTS plus Mimi streaming smoke coverage, foreground decode
  overlaps TTS synthesis, codec frames stream into Mimi before the full chunk is
  complete, and stream batches now use uniform `12`-frame batches for playback
  continuity. The current matrix shows first TTS enqueue around `3.8s` to
  `4.5s`, while first audio arrives around `4.9s` to `5.4s`. This is still too
  slow for the PRD target because foreground first content remains around
  `3.7s` to `3.9s`.
  The next TTS-side bottleneck is true Mimi incremental-state efficiency: the
  boundary is streaming, but the current Mimi path still recomputes
  full-history pre-transformer/conv work per batch. Host voice control inputs
  and real-asset cancellation timing proof remain.
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
- The identity LoRA can execute the current smoke tool probe after parser
  recovery, but its raw output is less protocol-clean than the base model. Keep
  the LoRA tool probe in smoke so regressions stay visible.

## Fresh Verification

Run from `E:\RiderProjects\Aila` with the oneAPI environment initialized through
`perf/PerfCommon.ps1`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; cmake -S . -B build; cmake --build build --target AliaEngine --config Release"
```

Result:

- Passed.
- The default branch build generated `AilaLib`, `AilaAliaRealSmoke`, and
  `AilaShared`.
- Generic CLI/API/test targets and lightweight API unit tests are no longer part
  of this branch build graph.

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
- No generic `aila_*` exports were present.

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -AudioPath "tmp\alia-real-smoke\alia_clean_request.wav" -OutputWav "tmp\alia-real-smoke\alia_full_pipeline_clean.wav" -LogPath "tmp\alia-real-smoke\alia_full_pipeline_clean.log" -TimeoutSec 1500
```

Result:

- Passed with `ALIA_REAL_MODEL_SMOKE_PASS`.
- Target model enforcement: `true`.
- Loaded ASR `Qwen3-ASR-1.7B-BNB-NF4`, foreground
  `qwen3.5-4B-bnb-nf4-offline-visiondense`, background
  `qwen3.5-0.8B-bnb-nf4-offline`, and TTS
  `Qwen3-TTS-12Hz-0.6B-Base`.
- Foreground LoRA:
  `F:\unsloth\qwen35_4b_alia_identity_r16_lr1e5\checkpoint-1400`.
- LoRA load/apply evidence: `Parsed 32 LoRA pairs`,
  `LoRA adapter applied: r=16 alpha=32 scaling=2.00 applied=32 skipped=0`,
  `foreground_lora_applied=true`, `foreground_lora_pair_count=32`.
- The request WAV was intentionally absent before the run. The smoke generated
  it through the target TTS model:
  `request_audio_generated=true`, callback count `8`, samples `90240`.
- Model load time: `25178ms`.
- ASR time: `1634ms`.
- ASR partial text: `Alia, please say hello in one short sentence.`
- Foreground time: `11220ms`.
- Foreground decode mode: `LoadedVlm`.
- Foreground assistant text:
  `父亲大人教过我，要简短回应。我……我可以用“Hello”和“我是Alia”回答，虽然有点害羞，但试试看吧。`
- TTS emitted `8` callbacks, first audio at `7559ms`, `92160` total samples,
  `91557` nonzero samples.
- Background time: `1515ms`.
- Background decode mode: `LoadedVlm`.
- Background schema valid: `true`, retry count `0`, repair applied `false`.
- Background diagnostic: `initial background JSON accepted; post cleanup applied`.
- Tool probe user text:
  `Call the host tool inspect_window with parameter id equal to 42 now. Return only the tool call.`
- Tool probe callback executed:
  `tool_probe_tool_call_json={"id":"call_0","type":"function","function":{"name":"inspect_window","arguments":"{\"id\":\"42\"}"}}`
  and `tool_probe_tool_result_text={"ok":true,"result":"real smoke tool callback executed"}`.
- The LoRA raw assistant text still contained spoken preface plus an orphan
  think close artifact, so the parser recovery is part of the smoke coverage
  rather than a reason to remove the probe.
- Full log: `tmp\alia-real-smoke\alia_full_pipeline_clean.log`.

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500
```

Result:

- Passed with `ALIA_VOICE_SCENARIO_MATRIX_PASS`.
- Summary CSV:
  `tmp\alia-real-smoke\voice_matrix\summary.csv`.
- Every scenario used the fixed ASR, foreground 4B + identity LoRA, background
  0.8B, and TTS model set.
- Every scenario reported `foreground_lora_applied=true` and
  `foreground_lora_pair_count=32`.
- Tool probe was intentionally disabled for the voice matrix:
  `tool_probe=false`.
- Scenario metrics:

```text
scenario          asr_ms  prompt_tokens  generated_tokens  first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_codes_ms  first_backend_audio_ms  first_audio_samples  backend_total_ms  foreground_ms  background_ms
short_hello       1648    120            15                3584              3842                  4948            850.632         1105.47                 23040                8100.67           11946          1043
persona_chat      1752    123            23                3656              3927                  5155            992.303         1228.05                 23040                9086.63           13017          1335
preference_memory 1720    123            20                3712              4247                  5186            689.095         938.956                 23040                6031.53           10281          1067
task_memory       1817    122            21                3658              3917                  5169            1002.79         1251.63                 23040                12178.2           16098          1061
long_answer       1693    125            28                3731              4090                  5406            1065.1          1315.44                 23040                10229.3           14322          655
```

- A prior `task_memory` run reproduced an access violation after foreground/TTS
  because the LoRA model emitted a half-open `<tool_call>` fragment in an
  ordinary voice answer. The foreground structured-artifact guard fixed that
  failure and reduced the reproduced `task_memory` generation from `64` tokens
  to `21` tokens.
- The current batch-scheduled codec streaming run uses uniform `12`-frame
  batches. This moves first backend audio later than the rejected `3`-frame
  first batch, but the first callback now carries `23040` samples, reducing the
  risk that playback drains before the second batch is available. The latest
  matrix landed first backend audio around `1.1s` after TTS starts.

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio -SkipToolProbe -AudioPath "tmp\alia-real-smoke\voice_matrix\short_hello_request.wav" -OutputWav "tmp\alia-real-smoke\alia_tts_uniform_batch_short.wav" -LogPath "tmp\alia-real-smoke\alia_tts_uniform_batch_short.log" -RequestText "Alia, please say hello in one short sentence." -MaxTokens 48 -TimeoutSec 1500
```

Result:

- Passed with `ALIA_REAL_MODEL_SMOKE_PASS`.
- Tool probe was skipped for this TTS schedule verification:
  `tool_probe=false`.
- Uniform `12`-frame stream batching produced first backend audio samples of
  `23040`; the observed output chunks were
  `23040,11520,23040,7680`.
- A natural-language tool probe miss under the identity LoRA produced persona
  text instead of a tool call. Tool-call robustness should be tracked as its
  own foreground-model task rather than folded into the TTS batch schedule
  change.
- Full log: `tmp\alia-real-smoke\alia_tts_uniform_batch_short.log`.

## 2026-06-16 TTS TTFA Inner-Loop Pass

The next TTS-focused pass keeps the uniform `12`-frame playback-continuity
schedule and reduces codec-generation overhead instead of shrinking the first
audio buffer.

Changes:

- Qwen3-TTS load now runs a tiny codec decode warmup after fixed embeddings are
  precomputed, so predictor/talker decode kernels are exercised before the
  first product utterance.
- `synthesize_codes` reuses fixed-shape decode tensors across codec frames
  instead of allocating them inside every frame/codebook step.
- Trailing text hidden states stay on GPU, and the frame loop adds them directly
  from GPU memory instead of copying small hidden vectors through CPU memory.
- Tool-call robustness under the identity LoRA remains a foreground-model TODO;
  it is intentionally not part of the TTS TTFA gate.

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500
```

Result:

- Passed with `ALIA_VOICE_SCENARIO_MATRIX_PASS`.
- Summary CSV:
  `tmp\alia-real-smoke\voice_matrix\summary.csv`.
- Every scenario kept `tts_first_backend_audio_samples=23040`, so the first
  audio callback still carries the continuity-first buffer.
- Scenario metrics:

```text
scenario          asr_ms  first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_text_tokens  first_frames  first_samples  first_codes_ms  first_backend_audio_ms  backend_total_ms  foreground_ms  background_ms
short_hello       923     3646              3897                  4938            19                 44            23040          796.032         1040.16                 7811.55           11712          1033
persona_chat      1033    3714              3987                  5189            19                 39            23040          969.469         1201.37                 9195.24           13186          1584
preference_memory 1022    3547              3919                  5288            23                 43            23040          1132.98         1368.79                 9365.56           13287          1143
task_memory       963     3498              3746                  5524            18                 42            23040          1543.68         1777.94                 21246             24996          1529
long_answer       1036    3583              4004                  5288            24                 56            23040          1051.66         1283.34                 10555.4           14563          669
```

Interpretation:

- The comparable `short_hello` run improved first backend audio from about
  `1105ms` to `1040ms` while keeping the first callback at `23040` samples.
- The larger remaining TTS variance is codec generation for the first spoken
  chunk. When the first chunk shape grows, first-code timing still ranges near
  `1.0s` to `1.5s`.
- Next useful TTS work is deeper predictor/talker decode profiling or reducing
  first spoken chunk shape without creating a too-short audio buffer.

```powershell
git diff --check
```

Result:

- No whitespace errors were reported.
- Git warned that modified files will be converted from LF to CRLF the next
  time Git touches them.

## Recommended Next Implementation Order

1. Reduce foreground first-content latency. The prompt is only about `120` to
   `125` tokens, so the next useful measurement is prefill time versus first
   sampled-token time inside the 4B backend.
2. Reduce TTS first-chunk latency. The matrix now shows a separate gap between
   first TTS enqueue and first audio, so the next optimization should inspect
   fixed Qwen3-TTS prefill/setup and first Mimi batch emission.
3. Improve background memory quality with real scenarios. It is functional and
   isolated, but some outputs still include schema-instruction artifacts or
   require repair wrapping.
4. Add hard-interruption timing probes with the fixed 4B foreground and TTS
   assets.
5. Investigate and fix the foreground 4B rollback crash before treating
   `alia_vlm_rollback_kv_cache` as production-ready with Qwen3.5 Hybrid 4B.
6. Migrate the 4B visiondense image path into the Alia foreground pipeline once
   the audio/text full pipeline latency is under control.

## Review Notes

The architecture direction is intentionally Alia-specific. This branch now keeps
the build and verification surface focused on the fixed Alia product pipeline
instead of carrying generic API, generic CLI, or lightweight API-test entry
points forward.
