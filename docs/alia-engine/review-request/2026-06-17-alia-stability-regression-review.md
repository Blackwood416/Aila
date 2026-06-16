# Review Request: Alia Stability Fixes and Lane-Lock Regression Follow-up

Date: 2026-06-17
Branch: `codex/alia-custom-engine`
Repo: `E:\RiderProjects\Aila`

## Scope

Review the final branch diff for the Alia SYCL inference engine stability fixes.
The intended review range is:

```powershell
git diff 4585303..codex/alia-custom-engine
```

Do not read or modify code under `E:\RiderProjects\Alia`; that repository is out
of scope. The product requirement may be consulted only at
`E:\RiderProjects\Alia\docs\design\aila_engine_prd.md` if needed.

## What Changed

The branch addresses the high-priority findings from
`docs/alia-engine/review/2026-06-17-branch-quality-review.md`:

- Finding #1: shared foreground `Context` access is serialized at pipeline
  boundaries. `Context` now exposes `lock_execution()` and protects allocation
  bookkeeping with a mutex. ASR transcription, foreground VLM prefill/decode and
  rollback, and TTS chunk synthesis take the lane lock around model-level GPU
  submission windows. TTS remains on the foreground lane. The earlier separate
  TTS lane and per-oneDNN-primitive stream locks were removed because they caused
  severe GPU slowdown/high copy utilization.
- Finding #2: `alia_free_string(char* s)` was added to the Alia ABI and is
  implemented with `std::free`, matching the `std::malloc` used by
  `alia_asr_get_text`.
- Finding #3: every `alia_*` C ABI entry point in `src/alia/AliaApi.cpp` is
  wrapped so C++ exceptions do not escape the DLL boundary.
  `ModelBackendCancelled` maps to `ALIA_ERR_ABORTED`; standard/unknown failures
  map to `ALIA_ERR_RUNTIME`.
- Finding #4: rollback replay now records tool-result continuation tokens and
  refuses to report success unless the backend context length matches the
  replayable sequence.

## Regression Diagnosis

An initial Finding #1 implementation passed functionally but regressed badly:

```text
model_load_ms=135933
asr_ms=11069
foreground_first_content_delta_ms=17334
tts_first_audio_ms=40341
```

A parent-control worktree at `HEAD~1` on the same GPU was fast:

```text
model_load_ms=25496
asr_ms=895
foreground_ms=4077
tts_first_audio_ms=3593
ALIA_REAL_MODEL_SMOKE_PASS
```

After reverting the per-primitive stream locking and separate TTS queue, then
using pipeline-level lane locking instead, timings returned to the expected
range.

## Verification Evidence

Forced clean build:

```powershell
$build = Resolve-Path -LiteralPath 'E:\RiderProjects\Aila\build'
Remove-Item -LiteralPath $build.Path -Recurse -Force
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; cmake -S . -B build -G Ninja; cmake --build build --target AliaEngine --config Release"
```

Result: exit code `0`, completed through:

```text
[124/124] Linking CXX shared library AilaShared.dll; Copying runtime DLLs for AilaShared
```

Short smoke:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe -MaxTokens 8 -NoGenerateAudio -LogPath 'tmp\alia-real-smoke\lane_lock_short.log' -OutputWav 'tmp\alia-real-smoke\lane_lock_short.wav'
```

Result:

```text
ALIA_REAL_MODEL_SMOKE_PASS
model_load_ms=26811
asr_ms=945
foreground_ms=3809
foreground_first_content_delta_ms=3067
foreground_first_tts_enqueue_ms=3312
tts_first_audio_ms=3807
```

Full target smoke with tool probe:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -AudioPath 'tmp\alia-real-smoke\alia_request.wav' -OutputWav 'tmp\alia-real-smoke\lane_lock_full.wav' -LogPath 'tmp\alia-real-smoke\lane_lock_full.log' -TimeoutSec 1500
```

Result:

```text
ALIA_REAL_MODEL_SMOKE_PASS
model_load_ms=25762
asr_ms=897
foreground_first_content_delta_ms=2927
foreground_first_tts_enqueue_ms=3185
tts_first_audio_ms=4063
tool_probe_ms=2081
```

Voice scenario matrix:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500
```

Result:

```text
ALIA_VOICE_SCENARIO_MATRIX_PASS summary=E:\RiderProjects\Aila\tmp\alia-real-smoke\voice_matrix\summary.csv
```

Matrix summary:

```text
scenario           model_load_ms  asr_ms  first_content_ms  first_tts_enqueue_ms  first_audio_ms
short_hello        25750          929     3085              3344                  4231
persona_chat       25638          1059    3143              3508                  4234
preference_memory  26174          1130    3163              3843                  4744
task_memory        25963          1016    3142              3403                  4480
long_answer        25877          1086    3090              3377                  4516
```

The matrix completed without the previous display/GPU black-screen crash.

## Requested Review Focus

Please prioritize:

- Whether `Context::lock_execution()` is used consistently enough to prevent the
  shared foreground `Context` allocator/queue/oneDNN stream race among ASR, VLM,
  and TTS without locking the NF4 decode hot path.
- Whether any shared-foreground-lane GPU submission remains reachable from
  multiple threads without the lane lock.
- Whether the lock ordering can deadlock with foreground `mutex_`, TTS callbacks,
  ASR `get_text`, abort, or rollback.
- Whether the ABI exception wrappers cover every `alia_*` export and preserve
  existing error-code contracts.
- Whether `alia_free_string` is documented and safe for C# P/Invoke ownership.
- Whether rollback replay can still report `ALIA_OK` with a KV sequence that does
  not match the recorded forwarded tokens.
- Whether any part of the fix violates `CLAUDE.md` Known Regressions, especially
  by modifying NF4 decode kernels or changing decode hot-path behavior.

## Reviewer Prompt

```text
You are a senior code reviewer for the Aila/Alia SYCL inference engine.

Repo: E:\RiderProjects\Aila
Branch: codex/alia-custom-engine
Review range: git diff 4585303..codex/alia-custom-engine

Do not read or modify code under E:\RiderProjects\Alia. You may read only
E:\RiderProjects\Alia\docs\design\aila_engine_prd.md if product context is
needed.

Please review the stability/correctness fixes from
docs/alia-engine/review/2026-06-17-branch-quality-review.md, focusing on:

1. Shared foreground Context thread safety after the regression fix:
   Context::lock_execution(), allocator bookkeeping mutex, ASR/VLM/TTS pipeline
   lock coverage, lock ordering, and whether any shared queue/stream/backend
   mutation remains concurrently reachable.
2. Performance risk: verify the fix does not reintroduce per-layer/per-primitive
   locks in src/ops/Bnb4BitLinear.cpp or src/ops/Linear.cpp and does not violate
   CLAUDE.md Known Regressions.
3. Alia ABI contract: alia_free_string ownership, every alia_* entry point
   wrapped in try/catch, and correct ALIA_ERR_ABORTED/ALIA_ERR_RUNTIME mapping.
4. Rollback correctness after tool-result resume: replay must not return ALIA_OK
   with inconsistent KV content/length.
5. Documentation accuracy in docs/alia-engine/2026-06-12-alia-engine-branch-status.md.

Verification already run on this machine:
- Forced clean configure/build with Ninja: exit 0, AliaEngine linked.
- RunAliaTargetPipeline.ps1 short smoke: ALIA_REAL_MODEL_SMOKE_PASS,
  model_load_ms=26811, asr_ms=945, foreground_ms=3809,
  tts_first_audio_ms=3807.
- RunAliaTargetPipeline.ps1 full smoke with tool probe:
  ALIA_REAL_MODEL_SMOKE_PASS, model_load_ms=25762, asr_ms=897,
  foreground_first_content_delta_ms=2927, tts_first_audio_ms=4063,
  tool_probe_ms=2081.
- RunAliaVoiceScenarioMatrix.ps1: ALIA_VOICE_SCENARIO_MATRIX_PASS, no
  black-screen crash observed.

Output findings first, ordered by severity, with file:line references. Include
open questions/assumptions and a short final assessment: ready to merge, ready
with fixes, or not ready.
```
