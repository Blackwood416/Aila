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
  - prefill because it can be hidden under remaining audio time
  - prefill because it is final-critical
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
