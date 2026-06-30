# 2026-06-30 Alia turn scheduler design

## Goal

Introduce an explicit turn-level scheduler for the real voice pipeline so ASR
partial decoding, foreground VLM prefill/speculation, and TTS first/steady audio
work are assigned by policy instead of by scattered cadence and threshold checks.

The target metric is short-prompt TTFA, measured as
`simulated_vad_to_first_audio_ms`, while preserving output ASR quality and
playback continuity.

## Current evidence

Latest backend work moved first TTS backend audio to about `210-230 ms`.
The visible remaining short-prompt path is:

```text
ASR final/tail -> foreground prompt prefill -> first spoken delta ->
first TTS enqueue -> TTS first backend audio -> audio callback
```

Recent short real probes on the current build:

```text
path                                  vad_to_audio  notes
default matrix short_hello             994 ms       ASR 357 ms, VLM full prompt prefill 329 ms
stream ASR prefill, 1000 ms cadence    1144 ms      87 cached tokens, but final 31-token suffix cost 451 ms
stream ASR prefill, guard=4            1311 ms      deeper prefill, still slow final suffix and later first enqueue
stream ASR prefill, 500 ms cadence     1089 ms      more ASR calls; useful partial text only near final
speculative foreground, 1000 ms        1308 ms      did not start; candidate stayed too short before final
```

This says that simply increasing partial frequency or prefix reuse can make the
critical path worse. The scheduler must know whether work is hidden under audio
time or is now delaying first audio.

## Non-goals

- No Computer Use, vision input, or tool-call implementation in this phase.
  Tool-call scheduling remains deferred.
- No TTS backend sampling changes or device sampling changes.
- No API-only validation as a replacement for real model smoke/matrix runs.
- No first-audio shrink-only strategy. Any chunk policy must preserve playback
  continuity.

## Scheduler model

The scheduler owns policy decisions for one voice turn. It is deterministic and
does not own model execution. Pipelines still execute work; the scheduler decides
when work is worth doing and records why.

### Inputs

- ASR event:
  - chunk end time in audio milliseconds
  - partial/final flag
  - stable text and partial text
  - whether text changed from the last scheduler event
  - ASR decode wall time for this event
- Foreground prefix state:
  - cached prompt token count
  - reused token count
  - suffix token count
  - last prefill wall time
  - current foreground lane wait/hold statistics
- Speculative state:
  - candidate text
  - candidate stable tick count
  - minimum character and word thresholds
  - whether a speculative turn is already running or ready
- TTS state:
  - first text enqueue seen
  - first audio callback emitted
  - playback buffer gap metrics

### Outputs

- ASR text decision:
  - skip unchanged text
  - decode partial
  - force final decode
- VLM prefill decision:
  - skip because text is too short
  - skip because only a tiny suffix was added
  - skip final stream prefill because final text is handled by the foreground
    turn on the critical path
  - prefill because it can be hidden under remaining audio time
  - reject cached prefix on final commit when its suffix is likely slower than a
    fresh full prompt prefill
- Speculative decision:
  - skip with reason
  - start speculative foreground turn
  - commit exact match
  - fall back to normal foreground turn
- TTS decision:
  - first-audio priority wait budget
  - steady backend batch size after first audio
  - playback-facing callback split size

## First implementation slice

Start with a conservative scheduler probe instead of a broad runtime rewrite.

1. Add a small scheduler policy object under `src/alia/`.
2. Hook the real smoke stream-ASR path through the scheduler and emit one
   machine-readable decision log per ASR tick.
3. Keep the default non-stream pipeline behavior unchanged.
4. Add foreground final cached-prefix reuse policy:
   - If cached ASR prefix reuse leaves a prompt suffix larger than the fast
     decode-suffix threshold, allow the scheduler to reject the cached prefix
     and run a fresh full prompt prefill.
   - Gate this behind an environment flag for the first probe.
5. Run short real smoke and then matrix with output ASR before making any policy
   default-on.

## Expected wins

- Prevent stream ASR prefill from worsening TTFA when the final suffix is large.
- Make ASR partial/speculative foreground failures diagnosable by reason rather
  than by reading scattered logs.
- Provide the structure needed for later speculative silence window integration
  once real VAD is available.
- Keep backend TTS improvements independent from turn scheduling policy.

## Validation

Each meaningful change must run:

```powershell
git diff --check
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --target AliaEngine --config Release
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe -NoGenerateAudio `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix_fused_conv2_default_retry\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\<run>\short.wav' `
  -LogPath 'tmp\alia-real-smoke\<run>\short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500
```

For default-on scheduler behavior, also run:

```powershell
if (-not $env:MIMO_API_KEY) {
  $env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable('MIMO_API_KEY', 'Machine')
}
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\<matrix-dir>'
```

## Deferred work

- Real-VAD speculative silence window: commit ASR text earlier than the final VAD
  silence window, then rollback/recompute suffix if new speech arrives.
- More aggressive speculative foreground starts from stable partial text.
- Host-facing scheduler API after the probe has matrix-grade evidence.
- Tool-call scheduling, vision input, and Computer Use integration.

## 2026-06-30 short smoke results

Implemented the first scheduler slice:

- `AliaTurnScheduler` policy helper.
- Stream-ASR scheduler decision logs in real smoke.
- Matrix CSV fields for scheduler decisions.
- Foreground final cached-prefix gate.

The first scheduler-on probe exposed a bad policy: doing VLM prefill on the
final ASR stream tick added about `325 ms` to the simulated ASR tail, and the
foreground then rejected the cached prefix and paid the full prompt prefill
again. The scheduler now skips final stream prefill and lets the foreground turn
handle final text.

Short smoke comparison, `short_hello_request.wav`, stream ASR prefill cadence
`1000 ms`:

```text
set                 vad_to_audio  asr_tail  stream_prefill  prompt_prefill  first_enqueue
scheduler on        1007 ms       357 ms    0 ms            328 ms          425 ms
scheduler off       1275 ms       358 ms    324 ms          443 ms          703 ms
```

Scheduler-on details:

```text
scheduler_prefill_allowed=0
scheduler_prefill_skipped=2
scheduler_last_reason="final text handled by foreground turn"
foreground_profile_prefilled_prompt_tokens=0
foreground_profile_final_cached_prefix_rejected=0
foreground_profile_final_cached_prefix_reject_reason="no cached prefix"
```

Scheduler-off fallback details:

```text
scheduler_prefill_allowed=2
foreground_profile_prefilled_prompt_tokens=87
foreground_profile_prompt_suffix_tokens=31
foreground_profile_final_cached_prefix_reject_reason="scheduler disabled"
```

External Mimo-ASR on scheduler-on output:

```text
Hello, I'm Alia, a local companion from the Isla Engine. I can be helpful if you ask.
```

This matches the foreground response closely enough for the smoke. The next
validation step is a full matrix with `-StreamAsrPrefill` and output ASR before
making host-facing scheduling defaults broader than the smoke/probe path.

## 2026-06-30 scheduler phase and budget results

The scheduler has been expanded from a prefill-only gate into two explicit
decisions per stream tick:

- ASR decode decision:
  - `phase`
  - `action`
  - `lane`
  - `reason`
- VLM prefill decision:
  - `phase`
  - `action`
  - `lane`
  - `reason`

Two hidden-work budget guards are now exposed:

- `AILA_TURN_SCHEDULER_MIN_HIDDEN_ASR_DECODE_AUDIO_MS`, default `450`.
- `AILA_TURN_SCHEDULER_MIN_HIDDEN_PREFILL_AUDIO_MS`, default `400`.

The ASR decode guard skips non-final partial decode near the end of a turn when
the remaining audio window cannot hide the decode. Final ASR decode is always
forced. The VLM prefill guard now uses `prefill_audio_budget_ms`, which is the
remaining audio window after the ASR decode for that tick has already completed.
This prevents a tick from allowing ASR and VLM work independently when their
combined cost no longer fits under the remaining audio.

Real-model matrix with ASR-decode budget only:

```text
path: tmp/alia-real-smoke/voice_matrix_scheduler_asr_decode_budget/summary.csv
pass: 5/5
avg simulated_vad_to_first_audio_ms: 1192.4
avg simulated_vad_asr_tail_ms: 423.2
avg foreground_profile_prompt_prefill_ms: 323.8
near-final ASR decode skips: 2
near-final VLM prefill skips: 0
output ASR: all scenarios returned transcripts
```

Real-model matrix with combined prefill budget:

```text
path: tmp/alia-real-smoke/voice_matrix_scheduler_combined_budget/summary.csv
pass: 5/5
avg simulated_vad_to_first_audio_ms: 1158.6
avg simulated_vad_asr_tail_ms: 435.2
avg foreground_profile_prompt_prefill_ms: 330.8
near-final ASR decode skips: 2
near-final VLM prefill skips: 2
output ASR: all scenarios returned transcripts
```

Per-scenario comparison, combined budget versus ASR-decode budget only:

```text
scenario           old_ttfa  new_ttfa  delta  decode_skips  prefill_skips
short_hello        997       1002      +5     0             0
persona_chat       1046      1081      +35    0             1
preference_memory  1262      1263      +1     0             1
task_memory        1366      1112      -254   1             0
long_answer        1291      1335      +44    1             0
```

The combined-budget policy is kept because it makes the scheduler's lane
allocation explicit and avoids known wasted near-final work. The TTFA effect is
modest and noisy because foreground first-content timing and TTS first-text
chunk shape now dominate many scenarios. The short prompt is around `1.0s` in
the best current matrix runs.

Next scheduler work:

- Move this policy out of the smoke harness into a host-facing turn scheduler
  API once real VAD exposes live silence-window timing.
- Track per-tick predicted ASR and VLM work cost rather than using fixed hidden
  budget thresholds.
- Add the real-VAD speculative silence window TODO: commit ASR text before the
  full VAD silence window, then rollback/recompute suffix if new speech arrives.
- Keep tool-call, vision input, and Computer Use scheduling deferred.

## 2026-06-30 first TTS soft-boundary flush

The foreground streaming path now allows the first TTS text chunk to flush at
the last soft boundary once the first soft minimum is met, instead of waiting
for `AILA_TTS_STREAM_TEXT_FIRST_SOFT_MAX_CHARS`. The behavior is controlled by:

```text
AILA_TTS_STREAM_TEXT_FIRST_SOFT_BOUNDARY_FLUSH=1
```

This only changes the first low-latency TTS enqueue path under the default
steady chunking policy. It is intended to reduce the gap between first
foreground content and first TTS enqueue when the model emits a useful comma or
other soft punctuation early.

Real-model matrix:

```text
path: tmp/alia-real-smoke/voice_matrix_tts_first_soft_boundary/summary.csv
pass: 5/5
avg simulated_vad_to_first_audio_ms: 1110.6
avg foreground_profile_first_tts_enqueue_ms: 462.4
total playback buffer gap ms: 595
output ASR: all scenarios returned transcripts
```

Baseline:

```text
path: tmp/alia-real-smoke/voice_matrix_scheduler_combined_budget/summary.csv
avg simulated_vad_to_first_audio_ms: 1158.6
avg foreground_profile_first_tts_enqueue_ms: 504.6
total playback buffer gap ms: 864
```

Per-scenario comparison:

```text
scenario           old_ttfa  new_ttfa  delta  old_enqueue  new_enqueue  gap_total
short_hello        1002      1000      -2     427          425          54
persona_chat       1081      1063      -18    436          431          0
preference_memory  1263      1098      -165   612          467          197
task_memory        1112      1137      +25    492          506          0
long_answer        1335      1255      -80    556          483          344
```

The average TTFA and first enqueue improved, and total playback-buffer gap
decreased on this matrix. Individual scenarios still show generation-dependent
gap variance, especially when the foreground text shape changes. Treat this as
a first-enqueue policy win rather than a final playback-continuity solution;
future scheduler work should still reason about TTS chunk duration, pending
audio buffer, and steady chunk coalescing together.
