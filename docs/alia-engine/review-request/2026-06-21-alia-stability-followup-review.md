# Review Request: Alia Stability Follow-up Fixes

Date: 2026-06-21
Branch: `codex/alia-custom-engine`
Repo: `E:\RiderProjects\Aila`

## Scope

Please review the follow-up fixes for
`docs/alia-engine/review/2026-06-17-alia-stability-regression-fix-review.md`.
The relevant new diff is the current working tree after commit `56c21be`
(`docs: add Alia engine review reports`).

Do not read or modify code under `E:\RiderProjects\Alia`. If product context is
needed, read only `E:\RiderProjects\Alia\docs\design\aila_engine_prd.md`.

## Review Feedback Addressed

### P1: Host audio callback under foreground lane lock

`AliaTtsPipeline::synthesize_text()` previously invoked the host
`AliaAudioCallback` while holding `Context::ExecutionLock`. That created a
possible self-deadlock if the callback re-entered an Alia ABI path that also
needed the shared foreground lane, such as `alia_asr_get_text()`.

Fix:

- `Context::ExecutionLock` now has `scoped_unlock()`.
- The TTS stream callback releases the foreground lane immediately before
  invoking `audio_cb`, then reacquires it when the callback returns.
- GPU submission windows remain serialized; only the host callback executes
  outside the lane lock.

Please scrutinize whether this narrow unlock window is safe with the current
TTS backend call stack and whether any backend/context mutation can now
interleave unsafely.

### P2: ABI ownership comment

`include/alia_api.h` now documents that strings returned by
`alia_asr_get_text()` are owned by the DLL and must be released with
`alia_free_string()`, not host-side allocators.

### P3: Stale export list

`docs/alia-engine/2026-06-12-alia-engine-branch-status.md` now includes
`alia_free_string` in the earlier dumpbin export list.

## Verification Evidence

Build:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -Command ". .\perf\PerfCommon.ps1; Initialize-AilaOneApiEnvironment; cmake --build build --target AliaEngine --config Release"
```

Result:

```text
[72/72] Linking CXX shared library AilaShared.dll; Copying runtime DLLs for AilaShared
```

Whitespace:

```powershell
git diff --check
```

Result: no whitespace errors; only existing LF-to-CRLF working-copy warnings.

Export check:

```powershell
dumpbin /exports build\AilaShared.dll | Select-String 'alia_free_string|alia_'
```

Result includes:

```text
alia_free_string
alia_asr_get_text
alia_start_conversation_turn
alia_vlm_prefill_asr_text
alia_vlm_rollback_kv_cache
```

Short smoke:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe -MaxTokens 8 -NoGenerateAudio -LogPath 'tmp\alia-real-smoke\callback_unlock_short.log' -OutputWav 'tmp\alia-real-smoke\callback_unlock_short.wav'
```

Result:

```text
ALIA_REAL_MODEL_SMOKE_PASS
model_load_ms=47072
asr_ms=1027
foreground_ms=3784
foreground_first_content_delta_ms=3024
foreground_first_tts_enqueue_ms=3270
tts_first_audio_ms=3782
```

Full target smoke with tool probe:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -AudioPath 'tmp\alia-real-smoke\alia_request.wav' -OutputWav 'tmp\alia-real-smoke\callback_unlock_full.wav' -LogPath 'tmp\alia-real-smoke\callback_unlock_full.log' -TimeoutSec 1500
```

Result:

```text
ALIA_REAL_MODEL_SMOKE_PASS
model_load_ms=26228
asr_ms=905
foreground_first_content_delta_ms=2865
foreground_first_tts_enqueue_ms=2930
tts_first_audio_ms=3756
tool_probe_ms=2018
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
short_hello        26141          955     3054              3314                  4218
persona_chat       26625          1087    3022              3112                  4389
preference_memory  26917          1060    2910              2974                  4689
task_memory        29359          1071    3300              3561                  4598
long_answer        27779          1105    3113              3449                  4548
```

Note: the real smokes exercise normal host audio callbacks, but they do not
explicitly re-enter Alia ABI calls from inside the audio callback. Please review
the lock-ordering fix structurally.

## Requested Review Focus

- Confirm `Context::ExecutionLock::scoped_unlock()` cannot leave the lane
  permanently unlocked if `audio_cb` throws or returns early.
- Confirm releasing the lane around `audio_cb` does not permit unsafe
  interleaving of TTS backend state, foreground VLM decode, ASR transcription,
  or allocator bookkeeping.
- Confirm no host callback is invoked while holding `Context::ExecutionLock`.
- Confirm the ABI ownership comments are clear enough for C# P/Invoke callers.
- Confirm the branch-status export list is internally consistent.
- Re-check that no per-primitive/per-layer locks were reintroduced in
  `src/ops/Bnb4BitLinear.cpp` or `src/ops/Linear.cpp`.

## Prompt For Reviewer

```text
You are a senior code reviewer for the Aila/Alia SYCL inference engine.

Repo: E:\RiderProjects\Aila
Branch: codex/alia-custom-engine

Please review the follow-up fixes for:
docs/alia-engine/review/2026-06-17-alia-stability-regression-fix-review.md

Review the current working tree diff after commit 56c21be. Do not read or
modify code under E:\RiderProjects\Alia. You may read only
E:\RiderProjects\Alia\docs\design\aila_engine_prd.md if product context is
needed.

Focus on:
1. P1 deadlock fix: Context::ExecutionLock::scoped_unlock() and
   AliaTtsPipeline::synthesize_text() must not invoke host audio callbacks while
   holding the foreground lane lock.
2. Lock correctness: the temporary unlock around audio_cb must not permit unsafe
   interleaving of TTS backend state, VLM decode, ASR transcription, queue/stream
   use, or Context allocator bookkeeping.
3. ABI contract: include/alia_api.h must clearly document alia_asr_get_text
   string ownership and alia_free_string usage.
4. Documentation: docs/alia-engine/2026-06-12-alia-engine-branch-status.md
   should consistently list alia_free_string in the export surface.
5. Performance risk: no per-primitive/per-layer locking should be reintroduced
   in src/ops/Bnb4BitLinear.cpp or src/ops/Linear.cpp.

Verification already run:
- cmake --build build --target AliaEngine --config Release: passed, linked
  AilaShared.dll.
- git diff --check: no whitespace errors.
- dumpbin export check: alia_free_string present.
- RunAliaTargetPipeline.ps1 short smoke: ALIA_REAL_MODEL_SMOKE_PASS,
  asr_ms=1027, foreground_ms=3784, tts_first_audio_ms=3782.
- RunAliaTargetPipeline.ps1 full smoke with tool probe:
  ALIA_REAL_MODEL_SMOKE_PASS, model_load_ms=26228, asr_ms=905,
  foreground_first_content_delta_ms=2865, tts_first_audio_ms=3756,
  tool_probe_ms=2018.
- RunAliaVoiceScenarioMatrix.ps1: ALIA_VOICE_SCENARIO_MATRIX_PASS.

Output findings first, ordered by severity, with file:line references. Include
open questions/assumptions and a final assessment: ready to merge, ready with
fixes, or not ready.
```
