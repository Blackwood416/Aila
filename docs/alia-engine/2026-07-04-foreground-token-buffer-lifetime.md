# 2026-07-04 Foreground Token Buffer Lifetime Stability Fix

## Symptom

After the short multi-turn matrix update, repeated real-model smoke runs began
failing with Windows exit code `-1073741819` (`0xc0000005`). Windows Event Log
showed the faulting module as Intel Level Zero:

```text
AilaAliaRealSmoke.exe
ze_intel_gpu64.dll
exception: 0xc0000005
fault offset before fix: 0x00000000002fd735
```

The crash happened during `AliaContext::load_model_slots()`, after TTS/Mimi
warmup and around foreground VLM warmup. `model_load_ms` was not printed in the
failing runs.

## Root Cause

`AliaForegroundPipeline::forward_token_span()` uploads prompt token ids into a
temporary device allocation and immediately returns after calling
`backend.forward()`. The hybrid foreground backend queues SYCL/Level Zero work
and returns without a full queue wait, so the temporary token-id USM allocation
could be freed while queued kernels still referenced it.

This was easy to miss because earlier host waits often masked the lifetime
hazard. Recent performance work reduced blocking waits, exposing the bug as a
driver access violation rather than a C++ exception.

## Fix

Synchronize the foreground queue before `forward_token_span()` returns and lets
the temporary token-id allocation destruct. The synchronization is local to this
helper, not a global `free_device()` wait, so it protects this specific
lifetime boundary without making all device frees blocking.

This helper is used for foreground prompt/prefill spans. It is not the per-token
foreground decode loop, so the added wait did not materially regress first-token
or prompt-prefill timing in the real-model matrix.

## Validation

Before the fix:

- Default foreground warmup repeatedly crashed in `ze_intel_gpu64.dll`.
- Setting `AILA_FOREGROUND_VLM_WARMUP=0` avoided the initialization crash and
  completed a real smoke, but prompt prefill regressed badly:
  `foreground_asr_prefill_ms=5695`.

After the fix:

```powershell
git diff --check
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --target AliaEngine --config Release

.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -VerifyOutputAsr `
  -OutputDir 'tmp\alia-real-smoke\voice_matrix_foreground_token_lifetime_fix'
```

Result:

```text
ALIA_VOICE_SCENARIO_MATRIX_PASS

scenario             TTFA  fg_prefill_ms  fg_tokens  tts_backend_ms  buffer_gap_ms
short_hello          973   328            26         5331.03         59
persona_chat         1061  325            17         3786.61         0
preference_memory    1049  323            12         3235.08         12
task_memory          1157  328            48         10220           246
multi_turn_followup  1118  324            41         7901.59         0
```

No new `AilaAliaRealSmoke` APPCRASH events were present in the Application log
after the matrix completed.

Content note: the matrix still shows `checkpoint-500` role drift and repetition
on some short Chinese prompts. That is a LoRA/model behavior issue, separate
from this foreground buffer lifetime stability fix.
