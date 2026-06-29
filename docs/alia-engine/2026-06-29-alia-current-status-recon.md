# 2026-06-29 Alia custom-engine current status recon

Scope: current-state reconnaissance after the 2026-06-28 Mimi/TTS backend
optimization pass. This document intentionally does not change the main path.
It summarizes what is now default, what remains default-off, and where the next
latency work should go.

## Branch state

Worktree: `E:\RiderProjects\Aila\.worktrees\alia-custom-engine`

Branch: `codex/alia-custom-engine`

Current local HEAD:

```text
558e0e6 feat: filter Alia foreground action tags
```

Recent commits after the first Mimi recon:

```text
2cd163a perf: profile Alia Mimi conv allocations
80e3e8f perf: probe Alia Mimi pre-transformer fusion
613aa02 perf: probe Alia Mimi decoder conv2 fusion
2a5a05a perf: reduce Alia Mimi transpose conv work
1681f97 perf: tune Alia TTS stream chunking
3eb9e76 perf: lower Alia TTS stream latency
558e0e6 feat: filter Alia foreground action tags
```

`origin/alia-custom-engine` is at `3eb9e76`; local HEAD adds the action-tag
filter on top.

## Current default path

The product/default path has changed materially since the first recon:

- Mimi decoder transposed convolution is optimized by default. The kernel now
  iterates only stride-compatible taps instead of scanning every kernel tap and
  testing `rem % stride`.
- TTS streaming defaults to uniform 4-frame callbacks:
  `AILA_TTS_STREAM_BATCH_FRAMES=4`, `AILA_TTS_FIRST_AUDIO_FRAMES=4`, and the
  first audio packet is `7680` samples, about 320 ms of 24 kHz mono audio.
- Foreground-to-TTS text flushing defaults to low-latency first chunk and more
  conservative steady chunks:
  first soft `18/48`, steady soft `24/120`, first hard min `0`, steady hard min
  `96`.
- `AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY` now defaults to `true`, with a
  `250 ms` timeout. The foreground decode yields briefly after first TTS enqueue
  so TTS can produce first audio without waiting behind continued VLM work.
- Parenthesized assistant action tags are filtered out of spoken text and
  exposed separately through smoke/matrix metrics.

The tool-call path still exists in the foreground pipeline and smoke harness,
but this recon does not pursue Computer Use, visual, or tool-call work. Keep
pipeline runs on `-SkipToolProbe` unless a separate tool-call task explicitly
asks otherwise.

## Evidence snapshot

Key source/logs inspected:

- `src/models/Qwen3TTSBackend.cpp`
- `src/ops/ConvOps.cpp`
- `src/alia/AliaForegroundPipeline.cpp`
- `src/alia/AliaTtsPipeline.cpp`
- `docs/alia-engine/2026-06-28-tts-mimi-backend-optimization-recon.md`
- `tmp/alia-real-smoke/voice_matrix_uniform4_default/summary.csv`
- `tmp/alia-real-smoke/voice_matrix_action_filter/summary.csv`
- `tmp/alia-real-smoke/tts_uniform4_default_short.log`
- `tmp/alia-real-smoke/action_filter_ascii_output_fresh.log`

Latest uniform-4 matrix:

```text
scenario           vad_to_first_audio  first_audio  first_enqueue  backend_first_audio
short_hello        989 ms              634 ms       414 ms         219.519 ms
persona_chat       993 ms              572 ms       345 ms         227.314 ms
preference_memory  1245 ms             827 ms       597 ms         229.048 ms
task_memory        970 ms              564 ms       341 ms         222.403 ms
long_answer        1495 ms             948 ms       724 ms         223.114 ms
```

Average in that matrix:

```text
tts_first_backend_audio_ms      224.3 ms
tts_first_audio_ms              709.0 ms
simulated_vad_to_first_audio    1138.4 ms
```

Latest action-filter matrix:

```text
scenario           vad_to_first_audio  first_audio  first_enqueue  backend_first_audio
short_hello        1031 ms             674 ms       450 ms         224.368 ms
persona_chat       1012 ms             588 ms       349 ms         238.468 ms
preference_memory  1253 ms             832 ms       600 ms         230.896 ms
task_memory        1115 ms             706 ms       484 ms         221.167 ms
long_answer        1422 ms             845 ms       602 ms         242.695 ms
```

Average in that matrix:

```text
tts_first_backend_audio_ms      231.5 ms
tts_first_audio_ms              729.0 ms
simulated_vad_to_first_audio    1166.6 ms
```

All 5 scenarios passed in both matrices. Output ASR matched the foreground
assistant text closely enough for the tested samples. In the action-filter
matrix, `preference_memory` extracted one action tag, `尾巴轻轻摆动`, while the
output ASR did not include that tag in spoken audio.

## What the probes changed

### Allocation profile

`AILA_TTS_MIMI_ALLOC_PROFILE=1` showed `mimi_conv_stages` allocates and frees
about 33 tensors per call, but allocator wall time was only a few milliseconds:

```text
8 frames   33 alloc/free   30.89 MB   2-5 ms alloc   ~97-99 ms conv total
15 frames  33 alloc/free   57.92 MB   ~5 ms alloc    ~184 ms conv total
32 frames  33 alloc/free   123.55 MB  ~5.6 ms alloc  ~366 ms conv total
```

Conclusion: scratch reuse is no longer the leading TTS backend lever. It can
still reduce variance and simplify lifetime/sync cleanup, but it should not
displace bigger wins.

### Pre-transformer residual fusion

`AILA_TTS_MIMI_PTFM_FUSED_RESIDUAL=1` replaces scale + residual add + copy with
`x = x + update * scale` in the incremental pre-transformer. It improved cached
pre-transformer slices by roughly 3-5 ms per Mimi callback in short smoke.

It remains default-off because it changes bf16 write ordering and has only
short-smoke plus output-ASR coverage.

### Decoder conv2 residual fusion

`AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL=1` fuses residual-block kernel-size
1 `conv2` with residual add. It removes one temporary and one launch per
residual block, but the measured gain was small/noisy. It remains default-off.

### Decoder transpose-conv cleanup

The decoder block profile showed transpose convolution was about 60-65% of the
decoder block time. The default kernel cleanup reduced the transpose substage
substantially:

```text
window frames    before transpose    after transpose
8                ~63 ms              ~27 ms
15/16            ~118-125 ms         ~47-51 ms
32               ~248 ms             ~96-99 ms
```

This is the most important backend win from the pass and is now on the default
path.

### Uniform 4-frame streaming

The branch now accepts the smaller first audio packet because continuity is
protected by uniform early callbacks. First backend audio is now consistently
around 220-240 ms for the first 4-frame packet. The cost is more callbacks and
more TTS backend invocations, which can still inflate total backend time for
long or repetitive outputs.

### Action tag filter

The action-tag filter is a speech cleanliness change, not a tool/vision feature.
It prevents parenthesized action text from being spoken while keeping the tags
visible in metrics. The final assistant text used by smoke/background memory is
the spoken text, not the action label.

## Current bottleneck read

The original short-prompt target is still not fully solved, but the dominant
shape has moved:

1. TTS first backend audio is no longer the largest visible cost. It is roughly
   220-240 ms for a 4-frame callback.
2. `tts_first_audio_ms` is usually 560-850 ms. That includes first enqueue delay
   plus backend first audio and scheduling contention.
3. `simulated_vad_to_first_audio_ms` still ranges about 970-1495 ms in the
   inspected matrices. The remaining gap is mostly before or beside TTS:
   ASR tail/partial cadence, foreground first content, and first TTS enqueue.
4. Long-tail TTS RTF is still a concern. Some scenarios have high
   `tts_backend_total_ms` because 4-frame streaming and fragmented/repetitive
   text can trigger many callbacks. This is a throughput/continuity risk, not
   the first-audio bottleneck.

## Ranked next candidates

### 1. Foreground first-enqueue reduction

Why: first backend audio is now near 225 ms; slow cases are gated by
`foreground_first_tts_enqueue_ms` and upstream ASR/foreground timing.

Candidate directions:

- Tune the first spoken text boundary with semantic guardrails, not just shorter
  character thresholds.
- Add more precise timing around first usable spoken delta, first punctuation,
  and first chunk acceptance/rejection.
- Keep action tags and tool markup out of the spoken/TTS buffer before boundary
  decisions.

Expected benefit: 100-300 ms on scenarios currently enqueuing around 600-724 ms.

Risk: medium. Too-aggressive flushing can produce unnatural fragments, more
backend invocations, or speaker/audio discontinuities.

Validation: build, short smoke, full matrix, output ASR, and callback interval
inspection.

Default: only small, conservative chunking changes should be default-on. More
aggressive policies should be default-off probes first.

### 2. 4-frame throughput/continuity guard

Why: uniform 4-frame streaming helps first audio, but some logs still show long
total backend time and occasional callback intervals above the 320 ms audio
payload.

Candidate directions:

- Track underrun risk explicitly: callback interval minus previous callback
  audio duration.
- Add a playback-aware steady-state coalescing rule after first audio has been
  delivered.
- Consider returning to 6 or 8 steady frames only after first audio, but keep
  first callback at 4 frames.

Expected benefit: lower total backend time and fewer playback-risk gaps on long
answers; little or no first-audio gain.

Risk: low to medium. Chunk changes can alter TTS text shape and audio cadence.

Validation: matrix plus callback intervals, output ASR, and long-answer RTF.

Default: default-off probe first unless the rule only affects post-first
coalescing when backlog exists.

### 3. Make pre-transformer residual fusion matrix-grade

Why: the opt-in path has a real 3-5 ms per-callback backend gain and simple
local semantics.

Candidate directions:

- Run full matrix with `AILA_TTS_MIMI_PTFM_FUSED_RESIDUAL=1`.
- Compare output ASR and key callback metrics against default.
- If clean, consider default-on.

Expected benefit: about 3-5 ms per Mimi callback.

Risk: low to medium due to bf16 write-order changes.

Validation: build, matrix, output ASR, at least one long-answer run.

Default: possible default-on only after matrix coverage.

### 4. Revisit decoder conv2 residual fusion only after matrix A/B

Why: it removes work, but measured benefit was modest and noisy.

Expected benefit: probably under 10-20 ms per larger conv window.

Risk: low to medium. It preserves the conv2 bf16 rounding point by writing the
conv output as bf16 before residual add, but still changes kernel structure.

Validation: matrix A/B with output ASR and backend total.

Default: keep default-off until it proves stable.

### 5. Decoder conv1 or exact conv state carry

Why: after transpose cleanup, decoder blocks are still the largest backend RTF
component for later windows. `conv1` is the next meaningful substage.

Risk: high. Fusing `act1 + conv1` naively recomputes SnakeBeta per output
channel and can regress. Exact state carry is more promising for long outputs
but is correctness-sensitive and must preserve causal receptive fields.

Expected benefit: potentially large for long-answer RTF; limited first-audio
gain.

Validation: offline equivalence against full-history decode, smoke, matrix,
Mimo-ASR/output ASR, and waveform sanity checks.

Default: research/default-off only.

## Do not spend next effort on

- SYCL Graph on the current oneAPI/Arc stack. Prior probe showed only
  `ext_oneapi_limited_graph` and slower replay.
- Large scratch-buffer rewrite as the next primary task. Allocation profile does
  not justify making it the first move.
- Tool-call, Computer Use, or visual work. Keep tool calls as TODO unless the
  task is explicitly scoped to them.
- API-only tests for TTS/vocoder changes. Real model smoke/matrix and output ASR
  are required.

## Suggested next experiment

The highest-signal next probe is not another Mimi kernel tweak. It is a
playback-aware first/steady chunking experiment:

1. Keep first callback at 4 frames.
2. After first audio is delivered, coalesce steady chunks when the previous
   callback interval threatens to exceed buffered audio.
3. Report `max(callback_interval - previous_audio_ms)` and count positive gaps.
4. Run short smoke and matrix with output ASR.

This directly tests whether the current 4-frame default can keep the TTFA win
while reducing long-tail backend time and playback risk.

## Playback-aware chunking probe results

Implemented observability:

- Smoke output now reports `tts_playback_gap_count`,
  `tts_playback_max_gap_ms`, `tts_playback_total_gap_ms`, and
  `tts_playback_gap_ms`.
- Matrix summaries now carry those fields plus backend stream-batch metadata:
  `tts_backend_stream_batch_frames`,
  `tts_backend_initial_stream_batch_frames`,
  `tts_backend_steady_stream_batch_frames`,
  `tts_backend_steady_batch_callback_count`, and
  `tts_backend_playback_aware_steady_batch`.

At this checkpoint the default path remained unchanged: first audio was still
4 frames / `7680` samples, steady backend batch was still 4 frames, and
playback-aware steady batching was off by default.

Default full matrix with output ASR:

```text
command:
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\voice_matrix_playback_gap_default'

summary:
tmp\alia-real-smoke\voice_matrix_playback_gap_default\summary.csv

avg simulated_vad_to_first_audio_ms  1085.0
avg tts_first_audio_ms                650.4
avg tts_chunks_synthesized              5.0
avg tts_backend_total_ms            16094.1
avg tts_playback_gap_count             28.6
avg tts_playback_max_gap_ms           345.2
avg tts_playback_total_gap_ms        1740.6
```

Probe matrix used:

```text
AILA_TTS_PLAYBACK_AWARE_STEADY_BATCH=1
AILA_TTS_STEADY_STREAM_BATCH_FRAMES=8
AILA_TTS_PLAYBACK_GAP_TRIGGER_MS=0
AILA_TTS_COALESCE_STEADY_TEXT_CHUNKS=1
```

The probe keeps the first backend callback at 4 frames. The foreground text
coalescing path is gated so it does not coalesce steady text until the first
audio callback has actually been emitted.

Probe full matrix with output ASR:

```text
command:
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\voice_matrix_playback_gap_probe8_after_first_audio'

summary:
tmp\alia-real-smoke\voice_matrix_playback_gap_probe8_after_first_audio\summary.csv

avg simulated_vad_to_first_audio_ms  1273.0
avg tts_first_audio_ms                839.8
avg tts_chunks_synthesized              2.6
avg tts_backend_total_ms             7333.5
avg tts_playback_gap_count              5.6
avg tts_playback_max_gap_ms           335.6
avg tts_playback_total_gap_ms         804.2
```

Compared with the default matrix, this probe changed the averages by:

```text
simulated_vad_to_first_audio_ms  +188.0
tts_first_audio_ms               +189.4
tts_chunks_synthesized             -2.4
tts_backend_total_ms            -8760.6
tts_playback_gap_count            -23.0
tts_playback_max_gap_ms            -9.6
tts_playback_total_gap_ms        -936.4
```

Interpretation:

- The probe substantially lowers backend total time and gap count by reducing
  TTS text fragmentation and using larger steady Mimi callback windows.
- It is not ready for default-on. Average TTFA regressed by about `188-189 ms`,
  and output ASR suggests possible tail truncation on several scenarios. The
  targeted external Mimo-ASR short probe only transcribed the first sentence:

```text
Hello, I'm Alia, a local companion.
```

- The full post-gating matrix also showed truncation-like output ASR for
  `preference_memory` and a shortened `long_answer`.
- Keep this path default-off until a follow-up can tune the transition policy
  and verify output ASR does not truncate or lose later spoken content.

## Backend-only split default follow-up

The combined probe was split into single-variable matrices:

```text
backend-only:
AILA_TTS_PLAYBACK_AWARE_STEADY_BATCH=1
AILA_TTS_STEADY_STREAM_BATCH_FRAMES=8
AILA_TTS_PLAYBACK_GAP_TRIGGER_MS=0

text-only:
AILA_TTS_COALESCE_STEADY_TEXT_CHUNKS=1
```

Average results:

```text
set          vad_to_first_audio  first_audio  first_enqueue  chunks  callbacks  backend_total  buffer_gap_total
oldDefault  1085.0              650.4        414.6          5.0     50.0       16094.1        981.6
backend8    1128.8              696.0        461.2          5.2     36.4       12474.2        296.8
textOnly    1137.2              704.2        465.8          3.8     46.8       15835.7        1460.2
combo       1273.0              839.8        601.4          2.6     19.2       7333.5         358.8
```

`textOnly` is not viable. It did not materially improve backend total time, made
playback gaps worse, and output ASR showed truncation-like results in
`preference_memory` and `task_memory`.

`backend8` is the useful variable. It reduced average backend total time by
about `3.6 s` and buffer-aware playback gap total by about `685 ms`, while first
backend audio stayed flat:

```text
set          first_backend_audio  first_backend_total  first_backend_frames  first_backend_callbacks
oldDefault  235.4                2811.8               30.0                  7.8
backend8    234.5                1329.1               17.8                  3.8
```

The follow-up implementation therefore makes playback-aware backend steady
batching default-on, with a default steady backend batch of 8 frames, but keeps
the foreground text coalescing path default-off. To preserve the playback-facing
chunk contract, `AliaTtsPipeline` now splits backend audio into at most
`AILA_TTS_AUDIO_CALLBACK_MAX_FRAMES` frames per audio callback, defaulting to the
normal stream batch size of 4 frames.

New observability:

- `tts_audio_callback_max_frames`
- `tts_playback_buffer_gap_count`
- `tts_playback_buffer_max_gap_ms`
- `tts_playback_buffer_total_gap_ms`
- `tts_playback_buffer_gap_ms`

Default split matrix:

```text
command:
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\voice_matrix_backend8_split_default'

summary:
tmp\alia-real-smoke\voice_matrix_backend8_split_default\summary.csv

avg simulated_vad_to_first_audio_ms       1133.6
avg tts_first_audio_ms                     700.4
avg foreground_first_tts_enqueue_ms        465.6
avg tts_chunks_synthesized                   5.0
avg tts_callback_count                      45.8
avg tts_backend_total_ms                 12908.9
avg tts_playback_buffer_gap_count            2.8
avg tts_playback_buffer_max_gap_ms         254.6
avg tts_playback_buffer_total_gap_ms       388.6
```

Interpretation:

- First backend audio remains about `235 ms`; the average TTFA difference versus
  old default comes from foreground first-enqueue variation, not from backend
  first-audio cost.
- The playback-facing chunks remain 4 frames or smaller, while the backend can
  amortize Mimi decode over 8-frame steady windows once playback debt appears.
- Output ASR did not show the systematic tail truncation seen in text coalescing
  probes, though foreground text sampling still varies noticeably between
  matrices.
