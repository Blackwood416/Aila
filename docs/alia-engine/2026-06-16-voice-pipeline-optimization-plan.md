# Voice Pipeline Optimization Plan

> For agentic workers: execute this plan task-by-task against the Alia custom
> branch. Keep verification on real models; do not reintroduce API-only unit
> tests or generic engine paths.

**Goal:** Improve the fixed Alia voice pipeline from real user speech input to
first TTS audio callback, while preserving identity LoRA behavior and background
memory extraction.

**Architecture:** Keep the product path fixed to Qwen3-ASR 1.7B NF4,
Qwen3.5 4B NF4 visiondense + identity LoRA, Qwen3.5 0.8B NF4 background, and
Qwen3-TTS 0.6B BF16. Use real scenario smoke runs to measure stage latency and
quality, then optimize the slow foreground/TTS overlap and TTS first-audio path.

**Tech Stack:** C++17, SYCL/oneAPI, oneDNN, PowerShell smoke runners, real model
assets under `models/`, identity LoRA under
`F:\unsloth\qwen35_4b_alia_identity_r16_lr1e5\checkpoint-1400`.

---

## Current Baseline

Latest product smoke (`tmp\alia-real-smoke\alia_full_pipeline_clean.log`) passed
with:

- ASR: `1634ms`
- foreground 4B + LoRA + streamed TTS turn: `11220ms`
- first audio callback from turn start: `7559ms`
- TTS callbacks: `8`
- background 0.8B extraction: `1515ms`
- tool probe: `1874ms`

The branch is functionally alive, but the PRD target is much more aggressive:
VAD fall to first audio should approach `400ms` in the final product. The first
optimization phase is therefore about measurement and pipeline overlap, not
Computer Use or visual input.

## Scenario Matrix

Use these fixed text prompts to synthesize request audio through the target TTS,
feed that audio through ASR, then run foreground/TTS/background:

- `short_hello`: `Alia, please say hello in one short sentence.`
- `persona_chat`: `Alia, I feel tired today. Say something gentle and brief.`
- `preference_memory`: `Alia, please remember that I prefer short Chinese replies at night.`
- `task_memory`: `Alia, remind me later to stretch my shoulders after work.`
- `long_answer`: `Alia, explain in three short sentences how you will help me focus tonight.`

Each run must write:

- a request wav
- an output wav
- a full log
- one CSV row containing stage metrics and validation flags

The voice matrix is for ASR -> foreground -> TTS -> background timing and
quality. It skips the dedicated LoRA tool probe by default so tool-call
stochasticity does not hide voice pipeline regressions. The full target smoke
still runs the tool probe unless `-SkipToolProbe` is explicitly passed.

## Task 1: Build Real Scenario Matrix Runner

**Files:**

- Create: `tools/alia/RunAliaVoiceScenarioMatrix.ps1`
- Modify: `tools/alia/RunAliaTargetPipeline.ps1`
- Verify with: `tools/alia/RunAliaVoiceScenarioMatrix.ps1 -SkipBuild`

Steps:

- Add a PowerShell matrix runner that builds once, invokes
  `RunAliaTargetPipeline.ps1` once per scenario, and writes
  `tmp\alia-real-smoke\voice_matrix\summary.csv`.
- Parse key-value log lines for `model_load_ms`, `asr_ms`, `foreground_ms`,
  `foreground_prompt_tokens`, `foreground_generated_tokens`,
  `foreground_first_content_delta_ms`, `foreground_first_tts_enqueue_ms`,
  `tts_first_audio_ms`, `tts_callback_count`, `tts_total_samples`,
  `background_ms`, `background_schema_valid`, `tool_probe_enabled`,
  `tool_probe_ms`, and pass/fail.
- Keep every scenario on the fixed model set and identity LoRA.

## Task 2: Add Foreground/TTS Timing Markers

**Files:**

- Modify: `src/alia/AliaForegroundPipeline.hpp`
- Modify: `src/alia/AliaForegroundPipeline.cpp`
- Modify: `tools/alia/AliaRealModelSmoke.cpp`

Steps:

- Record elapsed milliseconds from foreground turn start to first decoded
  content delta.
- Record elapsed milliseconds from foreground turn start to first TTS enqueue.
- Record prompt token count and generated token count for the main foreground
  turn.
- Print these metrics in `AilaAliaRealSmoke`.

## Task 3: Measure Before Optimizing

**Files:**

- Modify: `docs/alia-engine/2026-06-12-alia-engine-branch-status.md`

Steps:

- Run the full matrix once.
- Save the command and summary CSV path in the status document.
- Identify the top two bottlenecks from real data.

## Task 4: Optimize TTS First Audio

**Files:**

- Modify: `src/alia/AliaForegroundPipeline.cpp`
- Modify: `src/alia/AliaTtsPipeline.cpp`
- Inspect: `src/models/Qwen3TTSBackend.cpp`

Steps:

- Use Task 2 markers to distinguish VLM text delay from TTS synthesis delay.
- If TTS enqueue is early but first audio is late, reduce repeated TTS setup or
  prefill work on the fixed Qwen3-TTS path.
- If TTS enqueue is late, make the foreground chunk boundary more aggressive for
  short Chinese/persona replies without sending malformed text to TTS.

## Task 4A: Overlap Foreground Decode and TTS Synthesis

**Files:**

- Modify: `src/alia/AliaForegroundPipeline.cpp`
- Modify: `src/alia/AliaTtsPipeline.hpp`
- Modify: `src/alia/AliaTtsPipeline.cpp`
- Modify: `src/models/IModelBackend.hpp`
- Modify: `src/models/Qwen3TTSBackend.cpp`
- Modify: `tools/alia/AliaRealModelSmoke.cpp`
- Modify: `tools/alia/RunAliaVoiceScenarioMatrix.ps1`

Steps:

- Record first-chunk TTS backend timing separately from host callback timing:
  codec generation, Mimi init, first audio, total backend time, frames, callback
  count, and first audio sample count.
- Start a per-turn TTS worker before foreground decode, enqueue spoken chunks
  from the decode loop, and drain the worker before background extraction.
- Keep the synchronous `synthesize_pending` path for request-audio generation.
- Use UTF-8-safe chunk boundaries for ASCII and CJK sentence punctuation.
- Guard asynchronous worker cleanup across normal returns, failures, aborts, and
  exceptions.

## Task 4B: Stream Codec Frames Into Mimi

**Files:**

- Modify: `src/models/Qwen3TTSBackend.hpp`
- Modify: `src/models/Qwen3TTSBackend.cpp`

Steps:

- Keep the existing `synthesize_codes` API usable for full-code generation.
- Add an optional codec-frame callback to the existing Qwen3-TTS generation
  loop so generated frames can be consumed before the whole utterance finishes.
- Initialize Mimi before codec generation in `synthesize_codes_stream`.
- Feed each completed codec frame batch directly into `decode_mimi_incremental`.
- Preserve cancellation checks before codec generation, inside predictor/talker
  loops, before Mimi decode, and before host audio callbacks.

## Task 4C: Tune TTS Stream Batch Schedule

**Files:**

- Modify: `src/alia/AliaTtsPipeline.cpp`
- Modify: `src/models/Qwen3TTSBackend.hpp`
- Modify: `src/models/Qwen3TTSBackend.cpp`

Steps:

- Use uniform `12`-frame Mimi batches for the product path so the first emitted
  audio buffer is long enough to cover likely generation time for the next
  batch.
- Keep smaller first-batch experiments out of the product path unless paired
  with playback-aware buffering that can prove no under-run.
- Preserve the full-code generation API and default behavior for non-streaming
  callers.

## Task 5: Optimize Foreground Prompt/Decode

**Files:**

- Modify: `src/alia/AliaForegroundPipeline.cpp`
- Inspect: `src/models/Qwen35HybridBnb4Backend.cpp`

Steps:

- Keep the identity LoRA enabled.
- Reduce unnecessary prompt tokens in the foreground system prompt while keeping
  tool-call behavior covered by smoke.
- Use existing Qwen3.5 profile flags only after the scenario matrix shows VLM
  decode or prefill is the dominant bottleneck.

## Task 6: Background Isolation Check

**Files:**

- Modify: `tools/alia/AliaRealModelSmoke.cpp`
- Modify: `tools/alia/RunAliaVoiceScenarioMatrix.ps1`

Steps:

- Add a scenario mode that starts background extraction while a subsequent
  foreground turn begins.
- Verify foreground first-audio metrics do not regress under background load.

## Verification Commands

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake -S . -B build
cmake --build build --target AliaEngine --config Release
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild
git diff --check
```

Success requires:

- Every matrix scenario exits with `ALIA_REAL_MODEL_SMOKE_PASS`.
- Every run uses the target four-model set and applies 32 foreground LoRA pairs.
- The matrix keeps `tool_probe=false` by default; run
  `.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild` for the real
  `inspect_window` tool probe.
- Summary CSV is written with numeric timing data.

## 2026-06-16 Measurement Update

After adding the timing markers and ordinary-turn structured-artifact guard,
the voice matrix passed:

```text
scenario          asr_ms  first_content_ms  first_tts_enqueue_ms  first_audio_ms  foreground_ms
short_hello       1589    3626              3794                  5692            9477
persona_chat      1716    3714              3803                  4955            13959
preference_memory 1784    3704              4261                  8008            14637
task_memory       1679    3670              4257                  8034            14702
long_answer       1769    3790              4125                  5510            12024
```

Summary CSV:
`tmp\alia-real-smoke\voice_matrix\summary.csv`.

Initial bottleneck split:

- The foreground 4B path needs about `3.6s` to `3.8s` before the first
  content delta on these real scenarios.
- TTS receives the first chunk around `3.8s` to `4.3s`, but first audio arrives
  at `5.0s` to `8.0s`, so TTS first-chunk synthesis still adds roughly `1.2s`
  to `3.8s`.

## 2026-06-16 Async TTS Update

After adding UTF-8 chunk boundaries, first-chunk backend metrics, and a
per-turn asynchronous TTS worker, the voice matrix passed again:

```text
scenario          asr_ms  first_content_ms  first_tts_enqueue_ms  first_audio_ms  tts_first_codes_ms  tts_first_audio_backend_ms  tts_backend_total_ms  foreground_ms
short_hello       1710    3760              4073                  7402            3176.86             3328.15                     14712.7               18790
persona_chat      1829    3870              3960                  5848            1739.50             1887.76                     12124.9               16088
preference_memory 1784    3697              4248                  7522            3125.40             3273.52                     7898.39               12149
task_memory       1698    3642              3900                  7022            2976.00             3120.76                     18040.5               21944
long_answer       1780    3798              4137                  6854            2568.91             2716.31                     25685.1               29826
```

Summary CSV:
`tmp\alia-real-smoke\voice_matrix\summary.csv`.

Interpretation:

- Foreground decode now overlaps TTS synthesis, so VLM sampling no longer
  blocks on each completed spoken chunk.
- `foreground_ms` still includes final TTS drain by design because the smoke
  waits for full speech before background extraction.
- Mimi stream initialization is not the first-audio bottleneck; it is about
  `3ms` on the measured runs.
- First audio is dominated by full first-chunk codec generation before Mimi can
  emit samples. The next high-value optimization is real codec-token streaming
  into Mimi, or an equivalent first-batch path that does not wait for all frames
  of the chunk.

## 2026-06-16 Codec Frame Streaming Update

After adding the codec-frame callback and feeding generated frame batches into
Mimi during Qwen3-TTS generation, the voice matrix passed again:

```text
scenario          asr_ms  first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_codes_ms  first_backend_audio_ms  first_backend_total_ms  tts_backend_total_ms  foreground_ms
short_hello       1699    3714              3971                  4617            502.655         645.291                 6336.79                 10539.7               14514
persona_chat      1807    3869              4155                  5032            718.912         876.405                 5701.58                 12819.8               16978
preference_memory 1819    3690              3776                  4849            927.459         1073.02                 2786.95                 20506.9               24287
task_memory       1735    3632              3885                  5218            1184.03         1332.59                 6238.73                 28728.4               32618
long_answer       1783    3738              4174                  5262            933.657         1086.64                 9428.29                 18816.4               22994
```

Summary CSV:
`tmp\alia-real-smoke\voice_matrix\summary.csv`.

Interpretation:

- First audio moved from the previous `5.8s` to `7.5s` range into roughly
  `4.6s` to `5.3s` across the matrix.
- First codec batch readiness moved from full-chunk generation (`1.7s` to
  `3.2s`) to frame-batch generation (`0.5s` to `1.2s`).
- The next bottleneck is Mimi incremental decode efficiency. It is incremental
  at the API boundary, but currently recomputes full-history pre-transformer
  and conv stages per batch, so long answers still have high total TTS drain.
- Foreground first content remains about `3.6s` to `3.9s`, so further
  end-to-end TTFA work needs both foreground first-token profiling and Mimi
  incremental-state optimization.

## 2026-06-16 TTS Batch Schedule Update

After adding a two-stage stream schedule, the voice matrix passed again, but
the strategy was rejected for product playback. A `3`-frame first batch lowers
the first callback timestamp, yet it also makes the first audio buffer short
enough that playback can drain before the second batch is generated. The Alia
product path therefore returned to uniform `12`-frame batches:

```text
scenario          asr_ms  first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_codes_ms  first_backend_audio_ms  first_backend_total_ms  tts_backend_total_ms  foreground_ms
short_hello       1691    3697              3838                  4944            996.176         1105.24                 2907.38                 15287.7               19130
persona_chat      1823    3853              4219                  4517            194.839         298.012                 8331.75                 8331.75               12553
preference_memory 1781    3856              4463                  4766            195.586         302.900                 6833.15                 6833.15               11299
task_memory       1725    3791              4052                  5103            946.233         1049.55                 5483.24                 19418.3               23475
long_answer       1758    3827              4164                  4601            333.946         436.832                 3701.61                 4834.66               9002
```

Rejected two-stage run:

- Single-chunk replies now show first backend audio around `300ms` to `440ms`
  after TTS starts, while multi-chunk replies still depend on the first spoken
  chunk shape.
- Steady-state batching reduces Mimi call count on longer chunks, improving
  total backend drain compared with the previous fixed `6`-frame schedule.
- The remaining hard floor is still foreground first content at roughly
  `3.7s` to `3.9s`; TTS can now often be ready soon after the first chunk is
  enqueued.

Current uniform-batch verification:

```text
scenario          asr_ms  first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_codes_ms  first_backend_audio_ms  tts_backend_total_ms  foreground_ms
short_hello       1648    3584              3842                  4948            850.632         1105.47                 8100.67               11946
persona_chat      1752    3656              3927                  5155            992.303         1228.05                 9086.63               13017
preference_memory 1720    3712              4247                  5186            689.095         938.956                 6031.53               10281
task_memory       1817    3658              3917                  5169            1002.79         1251.63                 12178.2               16098
long_answer       1693    3731              4090                  5406            1065.10         1315.44                 10229.3               14322
```

Interpretation:

- The first callback is later than the rejected `3`-frame first-batch run, but
  now contains `23040` samples, which gives playback materially more buffer.
- This is a continuity-first schedule. Further latency work should target true
  Mimi incremental-state efficiency and playback-aware buffering instead of
  shortening only the first chunk.

## 2026-06-16 TTS Inner-Loop TTFA Update

The next pass kept the uniform `12`-frame first audio buffer and attacked
codec-generation overhead directly:

- Added a tiny codec decode warmup during Qwen3-TTS load, after fixed embeddings
  are precomputed, to exercise predictor/talker decode kernels before the first
  product utterance.
- Reused fixed-shape decode tensors in `synthesize_codes` instead of allocating
  them for every codec frame and codebook step.
- Kept trailing text hidden states on GPU and added them directly in the frame
  loop, removing small CPU round trips and synchronization points.

Verification:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500
```

```text
scenario          first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_text_tokens  first_frames  first_samples  first_codes_ms  first_backend_audio_ms  backend_total_ms
short_hello       3646              3897                  4938            19                 44            23040          796.032         1040.16                 7811.55
persona_chat      3714              3987                  5189            19                 39            23040          969.469         1201.37                 9195.24
preference_memory 3547              3919                  5288            23                 43            23040          1132.98         1368.79                 9365.56
task_memory       3498              3746                  5524            18                 42            23040          1543.68         1777.94                 21246
long_answer       3583              4004                  5288            24                 56            23040          1051.66         1283.34                 10555.4
```

Interpretation:

- `short_hello` is the closest apples-to-apples comparison with the previous
  uniform-batch matrix: first backend audio moved from about `1105ms` to
  `1040ms` while the first callback stayed at `23040` samples.
- The main TTS-side TTFA ceiling is still first-chunk codec generation, not
  Mimi init. Next work should profile predictor/talker decode per substage and
  reduce first spoken chunk shape without reducing playback buffer duration.
- Identity LoRA tool-call misses are recorded as a foreground TODO and should
  not gate TTS TTFA optimization.

## 2026-06-16 TTS Wait Cleanup Update

This pass kept the uniform `12`-frame callback contract and removed host-side
blocking from low-risk in-order queue paths inside `synthesize_codes`:

- Device-to-device copies for predictor inputs, hidden slots, and
  `past_hidden_talker` are now queued without immediate host waits.
- Talker-frame embedding accumulation no longer waits after every vector add.
- The 16 generated code ids for talker decode are uploaded as one frame array,
  and sampled predictor tokens use a stable host upload buffer so the queue can
  consume them asynchronously.

Verification:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500
```

```text
scenario          first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_text_tokens  first_frames  first_codes_ms  first_backend_audio_ms  backend_total_ms
short_hello       3830              4099                  5065            19                 44            703.279         965.217                 7296.87
persona_chat      3930              4299                  5125            21                 64            562.527         825.118                 7156.94
preference_memory 3951              4123                  5049            14                 23            664.119         925.609                 5111.15
task_memory       3929              4192                  5307            18                 42            854.988         1115.02                 11370.3
long_answer       4142              4326                  5559            15                 22            966.582         1231.96                 7603.42
```

Interpretation:

- The change is intentionally conservative: it reduces unnecessary host
  synchronization while keeping sampling semantics, first audio size, and the
  streaming callback contract unchanged.
- A device-sampling experiment was tried and rejected for this pass. It passed
  smoke, but changed sampled text/chunk shapes enough that matrix TTFA was not
  reliably better. Revisit only with a tighter deterministic comparison or a
  TTS-specific sampling contract.
- Next high-value TTS work is still true Mimi incremental-state efficiency and
  predictor/talker substage profiling, not shrinking the first audio chunk.

## 2026-06-16 Mimi Streaming Decode Update

This pass kept the uniform `12`-frame first audio callback and reduced overhead
inside Mimi streaming decode:

- Cached fixed Mimi `Linear` wrappers at vocoder load time instead of rebuilding
  them inside every full/incremental decode call. Persistent reshape views are
  kept for the VQ output projection weights so cached wrappers do not point at
  temporary tensors.
- Replaced the incremental pre-transformer's full-history K/V cache copy with
  direct prefill attention. The path already recomputes full-history Q/K/V, so
  copying all K/V into a stream cache before attention only added many small
  queued copies.
- Removed now-unused Mimi stream K/V/pre-conv cache allocations after direct
  prefill made those buffers dead state.
- Added `AILA_TTS_PROFILE=1` breakdowns for incremental VQ/projection,
  pre-conv, pre-transformer, conv/readback, and total Mimi chunk time.

Profile check on `short_hello`:

```text
first 12-frame Mimi chunk before direct prefill:
  VQ+proj 1.8 ms, pre-conv 0.4 ms, pre-transformer 76.8 ms,
  conv/readback 164.9 ms, total 244.1 ms

first 12-frame Mimi chunk after direct prefill:
  VQ+proj 2.1 ms, pre-conv 0.2 ms, pre-transformer 27.8 ms,
  conv/readback 166.1 ms, total 196.3 ms
```

Verification:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500
```

```text
scenario          first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_text_tokens  first_frames  first_samples  first_codes_ms  mimi_init_ms  first_backend_audio_ms  backend_total_ms
short_hello       3656              3911                  5100            19                 46            23040          999.592         0.0092        1188.63                 9129.08
persona_chat      3771              4052                  5115            19                 39            23040          868.054         0.0098        1062.41                 7358.56
preference_memory 3745              4028                  5464            19                 30            23040          1238.77         0.0106        1435.22                 11370.6
task_memory       3735              3991                  5391            18                 42            23040          1207.64         0.0098        1398.91                 12129.2
long_answer       3831              4269                  5792            24                 27            23040          1333.83         0.0083        1522.89                 9919.51
```

Interpretation:

- The fixed first audio buffer stayed at `23040` samples in every scenario.
- Removing unused stream K/V/pre-conv allocations reduced measured Mimi stream
  init from roughly `5ms` to near-zero.
- The profile run shows a real Mimi pre-transformer reduction, but the final
  full matrix is not an end-to-end TTFA win because this sampled run produced
  heavier first chunks and larger codec generation times. The pipeline remains
  functionally healthy, and the fixed first audio callback contract is intact.
- The remaining Mimi-side first-chunk cost is now mostly conv/readback
  (`~160ms` for the 12-frame first batch). Deeper conv-stage synchronization
  cleanup is possible, but should be done with care because the current code
  relies on local tensor lifetimes around queued device work.

## Next TODO: ASR Partial-to-VLM Prefill

Pause deeper TTS surgery unless profiling shows a clear low-risk win. Qwen3-TTS
has nested transformer generation, so codec sampling speed will naturally vary
with first spoken chunk shape. The current TTS path is good enough to shift the
main TTFA effort to ASR/VLM overlap.

High-value next step:

- Use ASR streaming `stable_text` / `partial_text` to start foreground VLM
  prefill before the ASR utterance is fully finalized.
- Maintain a foreground prefill state keyed by the stable ASR prefix. Stable
  text can be committed into the VLM prompt/KV state; partial text can be
  speculatively prefed only if the implementation can discard or rewind it when
  ASR revises the suffix.
- Keep the handoff stateful rather than repeatedly rebuilding the full
  foreground prompt. The expected TTFA gain comes from overlapping ASR decode
  tail time with VLM prompt prefill.
- First implementation should only bridge ASR -> foreground text prefill.
  Computer Use, visual input, and tool-call execution remain TODOs outside this
  phase.
- Verify with the real voice matrix, comparing `asr_ms`,
  `foreground_first_content_delta_ms`, `foreground_first_tts_enqueue_ms`, and
  `tts_first_audio_ms`. Do not replace this with API-only unit tests.

## 2026-06-16 ASR Partial-to-VLM Prefill Prototype

First prototype implemented an opt-in ASR/VLM overlap path:

- Added `alia_vlm_prefill_asr_text(stable, partial)` for the Alia product API.
- Foreground VLM now supports speculative prompt-prefix prefill from
  `stable_text + partial_text`. The prefetched prefix is validated against the
  final full prompt before generation; if ASR revised the prefix, generation
  falls back to a full prompt prefill.
- The prefill path only commits a token prefix of the final ChatML prompt and
  leaves a suffix guard so turn start can still forward at least one prompt
  token and obtain logits.
- `RunAliaTargetPipeline.ps1 -StreamAsrPrefill` feeds ASR in chunks and calls
  the VLM prefill hook after each ASR update. The default matrix path remains
  unchanged unless this flag is set.

Verification:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio `
  -SkipToolProbe -StreamAsrPrefill `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\alia_asr_prefill_short.wav' `
  -LogPath 'tmp\alia-real-smoke\alia_asr_prefill_short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500
```

```text
asr_ms=6875
asr_stream_prefill_enabled=true
asr_stream_prefill_calls=4
foreground_prompt_tokens=120
foreground_asr_prefill_tokens=79
foreground_asr_prefill_ms=330
foreground_first_content_delta_ms=920
foreground_first_tts_enqueue_ms=1306
tts_first_audio_ms=2240
```

Default path sanity check:

```text
asr_stream_prefill_enabled=false
foreground_asr_prefill_tokens=0
foreground_first_content_delta_ms=3459
foreground_first_tts_enqueue_ms=3818
tts_first_audio_ms=4691
```

Interpretation:

- The prototype proves the desired state handoff: most foreground prompt
  prefill can be completed before `alia_start_conversation_turn`.
- The smoke runner is still sequential, so `asr_ms` grows because it repeatedly
  invokes ASR partial transcription after each chunk. In the product, these
  prefill calls should run during live capture / ASR tail time, not after the
  whole utterance is already available.
- Next step is a more realistic streaming harness that records wall-clock from
  simulated VAD fall while feeding chunks over time, and then a matrix run with
  `-StreamAsrPrefill` once the harness reflects product overlap.

## 2026-06-16 ASR/VLM Live-Stream Harness Update

The ASR prefill prototype now has a more useful real-model measurement mode:

- `AliaRealModelSmoke` accepts `--stream-chunk-ms` for the ASR partial cadence
  used by `--stream-asr-prefill`; the PowerShell wrapper exposes this as
  `-StreamChunkMs`.
- The runner still executes deterministically in one process, but it now
  computes a simulated live timeline: each ASR/prefill operation can start only
  once its audio chunk would have arrived, and any compute before VAD fall is
  treated as hidden behind microphone capture.
- New log fields:
  - `asr_audio_duration_ms`
  - `asr_stream_simulated_tail_ms`
  - `simulated_vad_asr_tail_ms`
  - `simulated_vad_to_first_content_ms`
  - `simulated_vad_to_first_tts_enqueue_ms`
  - `simulated_vad_to_first_audio_ms`
- The default non-streaming path reports `simulated_vad_asr_tail_ms=asr_ms`,
  making it a conservative VAD-fall baseline where ASR work starts after the
  utterance is complete.

This is still a harness-level estimate, not a replacement for a real capture
thread. It is good enough to answer the next product question: whether ASR
partial text plus VLM prefix prefill materially reduces VAD-fall-to-first-audio
once capture overlap is accounted for.

Short real-smoke checks with the existing `short_hello_request.wav`:

```text
mode                  asr_ms  prefill_calls  vad_tail_ms  vad_to_first_content_ms  vad_to_first_audio_ms
default               943     0              943          4476                     5685
stream 1000ms chunks  6478    4              3717         4631                     6077
stream 2000ms chunks  4901    2              3140         4051                     5753
```

Interpretation:

- Prefix prefill is doing the useful foreground work: streaming runs reduce
  `foreground_first_content_delta_ms` from roughly `3.5s` to roughly `0.9s`.
- Running ASR partial after every `1000ms` of audio is too expensive for this
  implementation and leaves a large simulated VAD tail. `2000ms` cadence is
  better for first content, but the sampled TTS first chunk still made first
  audio roughly equal to the default run.
- A `4000ms` exploratory run is not a good product shape: it effectively waits
  until the utterance is over before prefill, and in one run produced a long
  max-token foreground output followed by a background-stage access violation.
  Keep the next pass focused on partial cadence and ASR compute reuse, not on
  end-of-utterance-only prefill.

## 2026-06-16 ASR Partial Cache and Cadence Split

Follow-up changes:

- Split streaming audio feed cadence from ASR partial/VLM prefill cadence.
  `--stream-chunk-ms` controls audio feed size, while
  `--stream-prefill-interval-ms` controls when the runner calls
  `alia_asr_get_text` and `alia_vlm_prefill_asr_text`.
- Added runner counters for `asr_stream_text_calls`,
  `asr_stream_prefill_calls`, and
  `asr_stream_prefill_skipped_unchanged`.
- Cached the last ASR partial result inside `AliaAsrPipeline` when the audio
  buffer size and stable offset are unchanged. This avoids a duplicate full
  partial transcription when a streaming prefill pass has already finalized
  ASR text before `alia_start_conversation_turn`.

Short smoke result on `short_hello_request.wav` with `-StreamAsrPrefill
-StreamChunkMs 1000 -StreamPrefillIntervalMs 3000`:

```text
before ASR partial cache:
foreground_first_content_delta_ms=929
foreground_first_tts_enqueue_ms=1173
simulated_vad_to_first_audio_ms=5407

after ASR partial cache:
foreground_first_content_delta_ms=354
foreground_first_tts_enqueue_ms=600
simulated_vad_to_first_audio_ms=4847
```

Streaming-prefill matrix:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 3000 `
  -OutputDir 'tmp\alia-real-smoke\voice_matrix_stream_prefill_3s' `
  -TimeoutSec 1500
```

```text
scenario          asr_ms  text_calls  prefill_calls  vad_tail_ms  fg_first_ms  vad_to_first_content_ms  tts_first_audio_ms  vad_to_first_audio_ms
short_hello       4526    2           1              3766         349          4115                     1870                5636
persona_chat      5165    2           2              3283         351          3634                     1719                5002
preference_memory 5109    2           2              3388         353          3741                     1882                5270
task_memory       4959    2           2              3798         350          4148                     1692                5490
long_answer       5159    2           2              2799         347          3146                     2522                5321
```

Interpretation:

- The foreground part of TTFA is now mostly solved for this prefill path:
  first content after turn start is around `350ms` across the matrix.
- The remaining live-simulated gap is dominated by ASR partial recompute tail
  plus TTS first-chunk variance. The ASR pipeline still transcribes the full
  non-stable suffix for each partial; the cache only removes repeated work when
  no new audio has arrived.
- Next ASR work should target incremental or cadence-aware partial decode:
  either reuse audio encoder/model work for the growing suffix, or run partial
  only near likely VAD fall while keeping the last good partial available to
  the foreground prompt cache.

## 2026-06-17 ASR Tail-Only Partial Experiment

Tested an experimental `AILA_ASR_TAIL_PARTIAL=1` path that tries to avoid
full suffix recompute by decoding only newly arrived audio after the previous
partial, using the previous partial text as ASR past text. The implementation
keeps a guard: if the tail decode does not produce a changed partial, it falls
back to the full non-stable suffix decode.

Short `short_hello_request.wav` checks:

```text
mode                         full_decodes  tail_decodes  vad_tail_ms  asr_partial_text
tail-only without full guard  1             3             2319         "Alia. Please say hello."
tail-only with full guard     2             2             3643         "Alia, please say hello in one short sentence."
```

Conclusion:

- Fine audio chunks are still the right product direction, but text-level
  tail stitching is not reliable enough to enable by default. Without the full
  guard it can drop the end of the utterance; with the guard it preserves ASR
  correctness but recovers little latency because it falls back to full suffix
  decode.
- `AILA_ASR_TAIL_PARTIAL` is therefore default-off and should stay an
  experiment only.
- The next useful ASR optimization is below the text stitching layer: profile
  and reuse mel/audio encoder work for growing partial suffixes, or add a real
  ASR decoder prefill/KV reuse path. The existing VLM prefill path is ready to
  consume finer ASR partials once they are cheap enough.

## 2026-06-17 ASR Partial Profile Probe

Added `AILA_ASR_PROFILE=1` stage metrics for each raw ASR segment
transcription. The smoke logs now include:

- `asr_profile_transcribe_calls`
- `asr_profile_generated_tokens`
- `asr_profile_input_audio_ms`
- `asr_profile_mel_ms`
- `asr_profile_upload_ms`
- `asr_profile_encoder_ms`
- `asr_profile_readback_ms`
- `asr_profile_prompt_ms`
- `asr_profile_prefill_ms`
- `asr_profile_decode_ms`
- `asr_profile_total_ms`

Short `short_hello_request.wav`, `-StreamAsrPrefill -StreamChunkMs 1000
-StreamPrefillIntervalMs 1000`, with `AILA_ASR_PROFILE=1`:

```text
asr_stream_text_calls=4
asr_partial_full_decode_count=4
asr_partial_tail_decode_count=0
asr_profile_transcribe_calls=4
asr_profile_generated_tokens=44
asr_profile_input_audio_ms=9760
asr_profile_mel_ms=429.948
asr_profile_upload_ms=2.9577
asr_profile_encoder_ms=260.048
asr_profile_readback_ms=3.1643
asr_profile_prompt_ms=4.7312
asr_profile_prefill_ms=1157.05
asr_profile_decode_ms=571.751
asr_profile_total_ms=2429.88
```

Interpretation:

- The largest measured ASR-internal stage is the text model prompt prefill over
  the growing audio-token prompt, not the audio encoder.
- Audio encoder reuse can still help, especially for longer utterances, but it
  is not the whole TTFA problem for short partials.
- KV reuse for the ASR text model is awkward with the current prompt shape
  because new audio tokens are inserted before `<audio_end>`, so the next
  partial is not a simple token-prefix extension of the previous partial.
- A promising next design is a streaming-specific ASR prompt contract with a
  stable append-only prefix, or an audio-token reservation/window scheme that
  preserves prefix reuse without changing recognized text.

## 2026-06-17 VLM Partial Suffix Recompute Reduction

The safe suffix-reuse target is the foreground VLM ASR-text prefill, not the
ASR audio-token prompt:

- The ASR audio encoder uses block-wise bidirectional attention
  (`n_window=50`, `n_window_infer=800`, giving 104 encoder tokens per block).
  Short utterances are usually inside one bidirectional block, so adding suffix
  audio can change earlier audio features. Reusing old audio-token KV would not
  be equivalent to a full ASR recompute.
- The foreground VLM prefill is text-only and causal, so unchanged token
  prefixes are safe to reuse.

Implementation:

- `AliaForegroundPipeline::prefill_asr_text` now handles ASR partial text
  revisions by finding the longest common token prefix between the previous
  prefill prompt and the new guarded target prompt.
- If the previous backend context still matches the previous prefill length, it
  truncates KV to that common prefix and only forwards the changed suffix.
- Pure append partials keep the existing fast path. Full reset is only used
  when the backend state cannot be trusted or truncation fails.
- Smoke/matrix logs now include
  `foreground_asr_prefill_reused_tokens` and
  `foreground_asr_prefill_suffix_tokens`.

## 2026-06-21 Foreground First-Content Profile Probe

Added foreground-stage metrics to the real smoke and matrix logs so future
TTFA work can separate VLM prompt construction, cached-prefix reuse, suffix
prefill, first token, first content, and decode drain time:

- `foreground_profile_prompt_tokens`
- `foreground_profile_prefilled_prompt_tokens`
- `foreground_profile_prompt_suffix_tokens`
- `foreground_profile_generated_tokens`
- `foreground_profile_prompt_build_ms`
- `foreground_profile_prompt_prefill_ms`
- `foreground_profile_first_token_delta_ms`
- `foreground_profile_first_content_delta_ms`
- `foreground_profile_first_tts_enqueue_ms`
- `foreground_profile_decode_ms`
- `foreground_profile_model_ms`

The metrics are wall-clock timings on the real product path and do not insert
extra GPU synchronizations. They are meant to explain TTFA behavior without
turning the profile itself into a new latency regression.

Short no-stream smoke (`short_hello_request.wav`) showed the baseline
foreground cost clearly:

```text
foreground_profile_prompt_tokens=120
foreground_profile_prefilled_prompt_tokens=0
foreground_profile_prompt_suffix_tokens=120
foreground_profile_prompt_prefill_ms=3279
foreground_profile_first_token_delta_ms=3294
foreground_profile_first_content_delta_ms=3294
foreground_profile_decode_ms=573
foreground_profile_model_ms=3860
```

With 3s ASR streaming prefill, the same short scenario reused 79 prompt tokens
and only forwarded the final 41-token chat suffix during `start_turn`:

```text
foreground_profile_prefilled_prompt_tokens=79
foreground_profile_prompt_suffix_tokens=41
foreground_profile_prompt_prefill_ms=347
foreground_profile_first_token_delta_ms=355
foreground_profile_first_content_delta_ms=355
foreground_profile_model_ms=1289
```

3s streaming-prefill matrix:

```text
scenario          prefilled  suffix  suffix_prefill_ms  first_token_ms  first_tts_enqueue_ms  vad_tail_ms  vad_to_first_content_ms  vad_to_first_audio_ms
short_hello       79         41      348                356             603                   3823         4179                     5153
persona_chat      82         41      349                356             823                   3515         3871                     5532
preference_memory 82         41      350                357             853                   3709         4067                     5875
task_memory       81         41      347                355             745                   4007         4362                     5671
long_answer       84         41      347                355             641                   2916         3271                     4865
```

Interpretation:

- The VLM-side streaming prefill path is doing its job: foreground turn-time
  prompt work is consistently reduced from the full 120-token prompt to a
  41-token suffix, and first content lands around 355ms.
- The remaining simulated TTFA is dominated by ASR partial cadence/recompute
  tail and TTS first-audio variance, not foreground first token generation.
- The next optimization should therefore return to ASR streaming partial cost
  and cadence, while preserving the VLM suffix-prefill contract measured here.

## 2026-06-25 Alia Reference Voice

Alia TTS now uses the worktree-local `alia_ref.wav` as the default Qwen3-TTS
Base speaker reference. `AILA_TTS_REF_AUDIO` can override the path; relative
paths are resolved from the current working directory, and the Alia smoke
runners set the default to the worktree root copy when present.

Implementation notes:

- `AliaTtsPipeline` extracts an ECAPA speaker embedding through the existing
  CPU `SpeakerEncoder`, caches it in memory, and passes it into
  `Qwen3TTSBackend::synthesize_codes_stream`.
- Reference voice loading happens immediately after the TTS model slot is
  loaded, so the roughly 8.5s extraction cost is accounted as model load time
  instead of first-turn TTFA.
- Missing or invalid reference audio is a setup failure for this custom Alia
  branch rather than silently falling back to the default TTS voice.
- Real smoke/matrix logs now report `tts_reference_audio_enabled`,
  `tts_reference_embedding_dim`, `tts_reference_embedding_ms`,
  `tts_reference_audio_path`, and `tts_reference_audio_error`.
- `RunAliaTargetPipeline.ps1` and `RunAliaVoiceScenarioMatrix.ps1` can run from
  the migrated worktree: they prefer `./models` and fall back to
  `../../models`, with `-ModelRoot` available for explicit overrides.

Validation:

```text
cmake --build build --target AliaEngine --config Release

RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio -SkipToolProbe ...
ALIA_REAL_MODEL_SMOKE_PASS
tts_reference_audio_enabled=1
tts_reference_embedding_dim=1024
tts_reference_embedding_ms=8537.83
tts_first_backend_audio_samples=23040
tts_first_backend_audio_ms=724.553

RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -OutputDir tmp\alia-real-smoke\voice_matrix_ref_tts
ALIA_VOICE_SCENARIO_MATRIX_PASS
```

Reference-voice matrix summary:

```text
scenario          ref  dim   first_content_ms  first_tts_enqueue_ms  first_audio_ms  first_samples  first_codes_ms  first_backend_audio_ms  first_backend_total_ms  backend_total_ms
short_hello       1    1024  4560              4622                  5841            23040          447.418         695.506                 1049.68                 7147.66
persona_chat      1    1024  4590              4860                  5901            23040          452.315         698.937                 3237.91                 12817.1
preference_memory 1    1024  4585              5105                  5829            23040          464.3           708.98                  4594.63                 4594.63
task_memory       1    1024  4620              4870                  6756            23040          456.502         704.117                 2959.63                 17301.8
long_answer       1    1024  4556              4876                  5974            23040          447.046         630.776                 2803.69                 6884.43
```

The first audio callback remains the fixed 12-frame / 23040-sample packet.
Reference extraction is now visible but no longer part of turn-time TTFA.

## 2026-06-25 VLM Tiny Partial Suffix Skip

With 1s ASR streaming cadence, the last few partial updates can append only a
handful of foreground prompt tokens. Forwarding those tiny suffixes through the
4B foreground VLM still costs roughly the same scheduling/launch overhead as a
larger useful suffix, and the final turn has to validate the full prompt prefix
again anyway.

Implementation:

- `AliaForegroundPipeline::prefill_asr_text` now skips pure append suffixes
  smaller than 16 tokens when the backend already matches the previous cached
  ASR prompt prefix.
- The skip deliberately keeps the old cached prompt prefix instead of updating
  `asr_prefill_prompt_ids_`; the skipped tokens are then included in the next
  non-tiny suffix or in the final `start_turn` prompt suffix.
- This only affects foreground text prefill. It does not change ASR decoding,
  TTS sampling, or the fixed 12-frame / 23040-sample first audio packet.
- Smoke/matrix logs now include
  `foreground_asr_prefill_skipped_small_suffix`.

Short 1s streaming smoke before the skip:

```text
asr_stream_simulated_tail_ms=8485
foreground_asr_prefill_tokens=79
foreground_asr_prefill_reused_tokens=75
foreground_asr_prefill_suffix_tokens=4
foreground_asr_prefill_ms=2176
foreground_profile_prefilled_prompt_tokens=79
foreground_profile_prompt_suffix_tokens=41
foreground_profile_prompt_prefill_ms=2212
tts_first_audio_ms=3698
simulated_vad_to_first_audio_ms=12183
```

Short 1s streaming smoke after the skip:

```text
asr_stream_simulated_tail_ms=4040
foreground_asr_prefill_tokens=71
foreground_asr_prefill_reused_tokens=71
foreground_asr_prefill_suffix_tokens=8
foreground_asr_prefill_skipped_small_suffix=2
foreground_asr_prefill_ms=0
foreground_profile_prefilled_prompt_tokens=71
foreground_profile_prompt_suffix_tokens=49
foreground_profile_prompt_prefill_ms=2259
tts_first_audio_ms=3583
simulated_vad_to_first_audio_ms=7623
```

1s streaming-prefill matrix with reference voice:

```text
scenario          pass  skipped  vad_tail_ms  prefilled  final_suffix  final_prefill_ms  tts_first_audio_ms  vad_to_first_audio_ms  first_samples
short_hello       true  2        4102         71         47            2267              3755                7857                   23040
persona_chat      true  4        3824         71         52            2265              3458                7282                   23040
preference_memory true  3        3653         71         52            2249              4180                7833                   23040
task_memory       true  3        3928         71         51            2236              3188                7116                   23040
long_answer       true  4        3829         71         54            2159              3567                7396                   23040
```

Interpretation:

- The optimization removes wasted VLM prefill work for tiny late partials and
  keeps correctness anchored on the final full prompt-prefix check.
- The final foreground suffix is larger because skipped partial tokens are
  carried into `start_turn`, but this moves the work out of the live ASR tail
  path where it was repeatedly delaying the simulated VAD-to-first-audio clock.
- The remaining gap is still dominated by ASR partial recompute/cadence plus
  the unavoidable foreground/TTS first-chunk work. The next high-value work is
  ASR partial cost reduction or a cadence policy that avoids expensive partial
  calls when the recognized text is unlikely to change enough.

## 2026-06-25 ASR/VLM Prefill Guard Tightening

The ASR partial-to-VLM prefill path was still leaving a large final foreground
suffix because it kept a 32-token retokenization guard on the speculative user
prefix, plus the separate 16-token final-prompt guard. The final turn already
checks that the cached prompt ids are a prefix of the real final ChatML prompt,
so the retokenization guard can be tighter without risking an incorrect
generation state: if the prefix ever stops matching, the final turn falls back
to normal prompt prefill.

Implementation:

- Reduced `kRetokenizeGuardTokens` in
  `AliaForegroundPipeline::prefill_asr_text` from 32 to 16.
- Kept the final 16-token prompt suffix guard unchanged.
- Cleaned up ASR runtime overhead by reading the valid audio encoder prefix
  directly from `audio_tmp` and reusing the ASR decode-loop argmax device
  buffer.

1s streaming-prefill matrix with reference voice:

```text
scenario          pass  prefilled  suffix  fg_prefill_ms  vad_tail_ms  vad_to_first_audio_ms  first_samples
short_hello       true  87         31      2402           4136         7627                   23040
persona_chat      true  87         36      2031           3478         6304                   23040
preference_memory true  87         36      2166           3584         7329                   23040
task_memory       true  87         35      2161           4110         7339                   23040
long_answer       true  87         38      2142           4103         7869                   23040
```

Compared with the previous 1s tiny-suffix matrix:

```text
scenario          prefilled_delta  suffix_delta  vad_to_first_audio_delta_ms
short_hello       +16              -16           -230
persona_chat      +16              -16           -978
preference_memory +16              -16           -504
task_memory       +16              -16           +223
long_answer       +16              -16           +473
```

Interpretation:

- The guard change consistently moves 16 more foreground prompt tokens into the
  speculative prefill cache and leaves a smaller final suffix.
- End-to-end TTFA is still noisy because foreground sampling and TTS first chunk
  shape vary, but the matrix average improved by roughly 200ms while preserving
  the fixed 23040-sample first audio packet.
- Foreground prompt prefill remains about 2.0-2.4s even with a 31-38-token final
  suffix, so the next VLM-side target is the incremental prefill backend path
  itself rather than only reducing suffix length.

## 2026-06-25 Decode-Path Foreground Suffix Prefill

Profiling `AILA_PROFILE_Q35_PREFILL=1` on the 31-token final foreground suffix
showed that incremental batch prefill was dominated by the BnB matmul/FFN
stages, not attention:

```text
[Q35PrefillProfile] calls=1 tokens=31 total=2366.777 total_tok=76.347655
linear_proj=14.191884 linear_o=10.661868 ffn_proj=28.987668 down=15.159871
attn=0.093300 lm_head=0.104871
```

The existing Qwen3.5 decode path has faster single-token kernels. For cached
foreground prefixes, a small suffix is causally equivalent whether it is
forwarded as one batch or token-by-token through decode. The product path now
uses decode-path forwarding for cached prompt suffixes up to 64 tokens, while
initial full prompt prefill still uses the normal batch path.

Implementation:

- Added a small `forward_token_span` helper in `AliaForegroundPipeline`.
- `prefill_asr_text` uses decode-path forwarding for non-initial ASR prefill
  suffixes up to 64 tokens.
- `generate_with_loaded_vlm` uses decode-path forwarding for final cached prompt
  suffixes up to 64 tokens.

1s streaming-prefill matrix with reference voice:

```text
scenario          pass  suffix  prompt_prefill_ms  fg_first_ms  vad_tail_ms  vad_to_first_content_ms  tts_first_audio_ms  vad_to_first_audio_ms  first_samples
short_hello       true  31      739                755          4027         4782                     1548                5575                   23040
persona_chat      true  36      882                897          3674         4571                     1936                5610                   23040
preference_memory true  36      883                898          3621         4519                     1989                5610                   23040
task_memory       true  35      861                876          3829         4705                     1966                5795                   23040
long_answer       true  38      928                944          3775         4719                     2428                6203                   23040
```

Compared with the guard-tightening matrix:

```text
scenario          prompt_prefill_delta_ms  vad_to_first_content_delta_ms  vad_to_first_audio_delta_ms
short_hello       -1663                    -1762                          -2052
persona_chat      -1149                    -943                           -694
preference_memory -1283                    -1236                          -1719
task_memory       -1300                    -1572                          -1544
long_answer       -1214                    -1532                          -1666
```

Interpretation:

- The foreground/VLM side of the streaming path is now consistently under 1s
  from `start_turn` to first content for these scenarios.
- The remaining VAD-to-first-audio budget is mostly ASR partial tail plus TTS
  first audio. Further VLM suffix length trimming is less valuable than ASR
  partial recompute work or TTS first-chunk variance reduction.

## 2026-06-27 ASR Partial Decode Throttle

After the foreground suffix path moved under 1s, the main avoidable cost in the
streaming path became repeated ASR partial recompute. A 500ms stream chunk can
still drive frequent VLM prefill checks, but the ASR model should not rerun a
partial decode for every small appended slice when the cached partial text is
already good enough for speculative prefill.

Short-scenario cadence probe:

```text
partial cadence  text calls  ASR transcribes  ASR total ms  ASR tail ms  VAD->audio ms
500ms            8           8                3415.9        3006         4651
1500ms           3           3                1691.3        2127         3653
2000ms           2           2                1299.4        2224         3848
```

Implementation:

- Added `alia_asr_get_partial_text()` for non-final streaming reads. It reuses
  cached stable/partial text unless enough new audio has arrived.
- Kept `alia_asr_get_text()` as the final/forced decode path so final turn text
  still flushes pending audio.
- Added `AILA_ASR_PARTIAL_MIN_ADVANCE_MS` with a default of 1500ms and a 500ms
  floor.
- Added `asr_partial_throttled_count` to the smoke/matrix output.
- Added TTS pipeline first-callback aggregation so backend short chunks cannot
  expose an underfilled first host callback. The first host audio callback
  remains fixed at 12 frames / 23040 samples; only later callbacks may be
  smaller.

500ms stream chunk + 500ms VLM prefill tick matrix, with 1500ms ASR partial
advance gate and reference voice:

```text
scenario          pass  text_calls  throttled  transcribes  asr_total_ms  vad_tail_ms  fg_prefill_ms  vad_to_first_audio_ms  first_samples
short_hello       true  8           5          3            1755.96       2238         757            4117                   23040
persona_chat      true  10          6          4            2517.71       2141         752            5128                   23040
preference_memory true  10          6          4            2476.72       2148         825            4134                   23040
task_memory       true  9           6          3            1848.03       1839         858            3848                   23040
long_answer       true  11          7          4            2623.49       1684         890            4363                   23040
```

Compared with the previous 1s streaming-prefill decode-suffix matrix:

```text
scenario          old_vad_to_audio_ms  new_vad_to_audio_ms  delta_ms
short_hello       5575                 4117                 -1458
persona_chat      5610                 5128                 -482
preference_memory 5610                 4134                 -1476
task_memory       5795                 3848                 -1947
long_answer       6203                 4363                 -1840
```

Interpretation:

- The stream can poll text/prefill at 500ms granularity without paying a full
  ASR partial decode on every poll.
- The final forced decode keeps correctness anchored on the actual final audio.
- End-to-end TTFA is still shaped by TTS first audio and generation variance,
  but the ASR-side tail is now roughly 1.7-2.2s in this matrix instead of the
  previous 3.6-4.0s range.
- Next ASR work should look at reusing encoder/prompt state inside real partial
  transcribes, not increasing the partial call cadence again.

## 2026-06-27 ASR Device Embedding Override

The ASR partial path used the audio encoder output by copying the encoded audio
features from GPU to host and then uploading them again as transformer embedding
overrides. That host bounce also creates an implicit synchronization point in
the streaming path. The Qwen3-ASR BnB4 backend now accepts device-resident
embedding overrides, so Alia ASR can scatter `audio_tmp` directly into the token
hidden states.

Implementation:

- Added `Qwen3ASRBnb4Backend::set_embedding_overrides_device(...)`.
- Kept the existing host `set_embedding_overrides(...)` path as fallback.
- Added `AILA_ASR_DEVICE_EMBEDDING_OVERRIDES`; default is enabled.
- Alia ASR uses the device path only for the Qwen3-ASR BnB4 backend.

Short prompt A/B, same build and audio, no ASR profiling sync:

```text
mode                 pass  transcribes  asr_tail_ms  vad_to_first_audio_ms  first_samples  asr_text
device override on   true  3            2004         3576                   23040          "Alia, please say hello in one."
device override off  true  3            3838         5476                   23040          "Alia, please say hello in one."
```

500ms stream chunk matrix with device override enabled, no ASR profiling sync:

```text
scenario          pass  asr_tail_ms  vad_to_first_audio_ms  first_samples
short_hello       true  2004         3576                   23040
persona_chat      true  3918         6663                   23040
preference_memory true  3775         5658                   23040
task_memory       true  3475         5455                   23040
long_answer       true  3247         5545                   23040
```

Interpretation:

- The device override removes a real GPU-host-GPU sync/copy in the ASR path and
  preserves ASR text on the short A/B.
- The full matrix still shows substantial runtime and generation variance, so
  this should be treated as a low-risk plumbing improvement rather than the main
  path to sub-1s TTFA.
- Tail-partial ASR and final cached/contextual decode experiments were rejected:
  they reduced some ASR work but changed user text shape enough to make VLM/TTS
  behavior worse.

## 2026-06-27 Foreground VLM Warmup

The next profiling pass split the ASR streaming tick into ASR text decode and
foreground ASR-prefill work. The short prompt showed that the largest remaining
tail was not ASR decode itself, but the first foreground Qwen3.5 forward:

```text
metric                              before warmup
asr_stream_get_text_total_ms        1691
asr_stream_vlm_prefill_total_ms     4503
asr_stream_vlm_prefill_max_ms       4503
foreground_profile_prompt_prefill_ms 751
simulated_vad_to_first_audio_ms     5564
```

Two scheduling experiments were rejected:

- Async foreground prefill moved the cost out of `alia_vlm_prefill_asr_text`,
  but GPU contention made ASR `get_text` block instead; short prompt TTFA
  regressed to roughly 6.2s.
- Small per-tick VLM prefill slices still paid the same first-forward/resize
  cost on the first slice and regressed TTFA.

Implementation:

- Added `AliaForegroundPipeline::warmup_loaded_vlm()` with a 128-token dummy
  foreground prompt forward followed by `reset()`.
- `AliaContext::load_model_slots()` runs this warmup after foreground model and
  LoRA load, before real audio pipeline work. It can be disabled with
  `AILA_FOREGROUND_VLM_WARMUP=0`.
- Added smoke/matrix telemetry for ASR stream text, VLM prefill, and total tick
  time so future regressions show up in the real-model matrix.

500ms stream chunk + 500ms VLM prefill tick matrix after foreground warmup:

```text
scenario          pass  asr_tail_ms  vlm_prefill_max_ms  fg_prefill_ms  vad_to_content_ms  vad_to_audio_ms  first_samples
short_hello       true  484          333                 751            1251               2081             23040
persona_chat      true  1091         331                 744            1851               3746             23040
preference_memory true  1001         330                 821            1838               2936             23040
task_memory       true  557          329                 847            1420               2582             23040
long_answer       true  738          332                 869            1623               2791             23040
```

Interpretation:

- The foreground first-forward spike moved to model-load time, outside the
  voice TTFA path.
- Streaming VLM prefill max is now about 330ms instead of about 4.5s on the
  short prompt.
- The remaining short-prompt TTFA is roughly `0.48s ASR tail + 0.76s foreground
  first content + 1.6s TTS first audio`. The next high-value work is therefore
  TTS first-audio latency and ASR final/partial tail, not VLM prefill
  scheduling.

## 2026-06-27 ASR Partial/Final Breakdown

`AILA_ASR_PROFILE=1` now has an optional smoke-level per-call trace via
`AILA_ASR_PROFILE_CALLS=1`. The profile mode synchronizes after ASR stages, so
the resulting TTFA is not a product number; the stage split is useful for root
cause analysis.

Short prompt, 500ms stream chunk, 1500ms partial advance gate:

```text
call  kind     audio_ms  call_ms  mel_ms  encoder_ms  prefill_ms  decode_ms  tokens
0     partial  1500      605      54      124         355         70         6
1     partial  3000      569      109     58          307         94         8
2     final    3840      504      137     36          186         143        12
total          8340      1679     300     218         847         306        26
```

Final-only, same audio:

```text
call  kind   audio_ms  call_ms  mel_ms  encoder_ms  prefill_ms  decode_ms  tokens
0     final  3840      753      138     151         318         144        12
```

Existing throttle A/B without ASR profiling sync:

```text
partial_min_advance_ms  transcribes  asr_total_ms  asr_tail_ms  vad_to_audio_ms
1500                    3            1625          484          2081
2500                    2            1277          625          2738
3000                    2            1190          705          2437
```

Interpretation:

- The stream is paying for repeated full ASR transformer prefill over growing
  audio prefixes. The current short prompt spends about half of ASR model time
  in prefill across partial/final calls.
- Simply raising the partial interval removes one ASR transcribe, but it delays
  useful VLM prefill and does not improve TTFA in real smoke.
- The promising ASR direction is state reuse or cheaper incremental partial
  verification, not a coarser partial cadence. Candidate work: cache/reuse ASR
  prompt/audio prefix state across partials, or add a deterministic low-cost
  stability check before rerunning full ASR prefill.

## 2026-06-27 ASR Prompt Prefix Reuse Experiment

Qwen3-ASR prompt-prefix KV reuse was prototyped behind
`AILA_ASR_PREFIX_REUSE=1` and left disabled by default. The implementation keeps
the cached backend KV up to `<audio_start> + audio_tokens` after each ASR
partial/final call. A later growing partial can truncate back to that prefix,
upload only new audio-token embeddings, and run incremental prefill for the new
audio tokens plus the text suffix. Smoke telemetry now reports
`asr_profile_prefix_reuse_attempts`, `asr_profile_prefix_reuse_hits`,
`asr_profile_prefix_reused_tokens`, and `asr_profile_prefix_appended_tokens`.

Short prompt profile, 500ms stream chunk, 1500ms partial advance gate:

```text
mode          calls  hits  reused_tokens  appended_tokens  prefill_ms  total_ms  final_text
reuse off     3      0     0              0                808.139     1615.63   "Alia, please say hello in one."
reuse on      3      2     102            44               820.209     1636.67   "Alia, please say hello in one."
```

Interpretation:

- Prefix reuse is mechanically viable: the second partial and final call both
  hit the cached KV path.
- It is not a good default optimization for the current cadence. The reusable
  ASR prompt prefix is small, and cached incremental prefill attention still
  attends over the full prefix, so the extra bookkeeping slightly regressed the
  profiled short prompt.
- Reusing audio-token KV is also conceptually risky because the ASR audio
  encoder runs self-attention over encoder blocks; older audio embeddings are
  not guaranteed to be invariant when more audio arrives.
- The next ASR optimization should target the larger repeated costs directly:
  mel/audio encoder reuse for stable audio blocks, or a cheaper partial
  verification path that avoids full ASR prefill when the transcript is unlikely
  to change.

## 2026-06-27 ASR Mel/Audio Encoder Reuse Probe

Added finer ASR profile telemetry for mel and audio encoder work:

- `asr_profile_mel_stft_ms`
- `asr_profile_mel_norm_ms`
- `asr_profile_encoder_conv_ms`
- `asr_profile_encoder_transformer_ms`
- `asr_profile_encoder_proj_ms`

The profile confirmed that short-prompt mel time is almost entirely CPU STFT,
not Whisper normalization:

```text
short profile before mel cache:
asr_profile_mel_ms                   296.562
asr_profile_mel_stft_ms              296.478
asr_profile_mel_norm_ms              0.0747
asr_profile_encoder_ms               232.856
asr_profile_encoder_conv_ms          157.946
asr_profile_encoder_transformer_ms   54.0329
asr_profile_encoder_proj_ms          20.4088
```

Implemented exact raw-log-mel prefix reuse for growing ASR partials:

- `compute_mel_spectrogram_cached` stores unnormalized log-mel frames.
- On a growing partial, it reuses the stable prefix, recomputes the last 3 old
  frames plus new frames to cover reflect-padding right-edge effects, then
  normalizes over the full raw-log-mel buffer. This keeps output equivalent to
  full recompute while avoiding repeated STFT work.
- `AILA_ASR_MEL_CACHE=0` disables the cache.
- `AILA_ASR_MEL_CACHE_VALIDATE=1` computes both cached and full mel and reports
  `asr_profile_mel_cache_max_abs_diff`.
- Smoke/matrix telemetry now includes `asr_profile_mel_cache_hits`,
  `asr_profile_mel_cache_reused_frames`,
  `asr_profile_mel_cache_computed_frames`, and
  `asr_profile_mel_cache_max_abs_diff`.

Short prompt, 500ms stream chunk, 1500ms partial advance gate:

```text
mode                 hits  reused_frames  computed_frames  max_diff  mel_ms   mel_stft_ms  asr_total_ms  asr_tail_ms
validate cache       2     446            391              0         462.145  140.933      1828.52       538
cache on             2     446            391              0         141.811  141.518      1515.08       388
previous no-cache    0     0              0                n/a       296.562  296.478      1663.66       505
```

Interpretation:

- The cached mel path matched full recompute exactly on the real short smoke
  (`max_abs_diff=0`).
- It removes about 155ms of repeated short-prompt mel STFT work in the profiled
  streaming path and reduces the final ASR tail by about 117ms in this run.
- Audio encoder block reuse is still a longer-utterance optimization. With
  `n_window=50` and `n_window_infer=800`, the encoder block is 104 audio tokens.
  The 3.84s short prompt reaches only about 50 audio tokens, so no complete
  bidirectional block can be sealed safely yet.

## 2026-06-27 ASR Conv Frontend Preallocation

The ASR audio encoder conv frontend still allocated several fixed-shape tensors
on every partial/final encode. This pass reuses persistent buffers for the
100-frame mel chunk, the three conv2d intermediates, the flattened conv output,
and the concatenated pre-transformer output. `conv_all_out` is also preallocated
at load time to the configured source-position capacity, with dynamic growth for
larger inputs.

Short prompt profile, 500ms stream chunk, 1500ms partial advance gate:

```text
mode                  encoder_conv_ms  asr_total_ms  asr_tail_ms
before prealloc       155.077          n/a           n/a
after prealloc        146.790          1440.08       380
```

Default short smoke after the change, without ASR profiling sync:

```text
asr_ms                            1796
asr_stream_simulated_tail_ms       381
simulated_vad_to_first_content_ms 1144
simulated_vad_to_first_audio_ms   1977
first_audio_samples              23040
```

Interpretation:

- This removes avoidable frontend allocation churn and is worth keeping, but the
  measured short-prompt gain is small: roughly 8ms in profiled conv time.
- The fixed 12-frame / 23040-sample first TTS buffer remains intact.
- The next high-value ASR optimization should move below this layer: either GPU
  mel/STFT to remove the CPU computation and upload sync, or a cheaper
  partial/final policy that avoids full ASR prefill when the transcript is
  unlikely to change.

## 2026-06-27 ASR GPU Mel/STFT Prototype

Implemented an ASR-specific GPU mel path for the Qwen3-ASR frontend:

- `GpuMelSpectrogram` preloads the ASR Hann window, direct DFT cos/sin table,
  and embedded 201x128 mel filterbank to the ASR context.
- The GPU path computes raw log-mel frames, reuses the stable prefix exactly
  like the CPU mel cache, normalizes on device, and writes the bf16 transposed
  `[1,128,n_frames]` tensor directly for the audio encoder.
- `AILA_ASR_GPU_MEL` is enabled by default on this branch; set
  `AILA_ASR_GPU_MEL=0` to return to the CPU mel path.
- `AILA_ASR_GPU_MEL_VALIDATE=1` compares GPU normalized f32 mel against the CPU
  full mel path and reports the max absolute diff through
  `asr_profile_mel_cache_max_abs_diff`.
- `AILA_ASR_GPU_MEL_WARMUP=1` is enabled by default when GPU mel is enabled.
  This compiles the GPU mel kernels during model load so the first live ASR
  partial does not pay the SYCL JIT cost.

Correctness smoke with validation:

```text
asr_profile_mel_cache_max_abs_diff  3.92199e-05
asr_partial_text                    "Alia, please say hello in one."
result                              ALIA_REAL_MODEL_SMOKE_PASS
```

Short prompt profile, 500ms stream chunk, 1500ms partial advance gate:

```text
mode                    mel_stft_ms  mel_ms   asr_total_ms  asr_tail_ms  vad_to_audio_ms
CPU cached mel           141.321      n/a      1490.17       381          2022
GPU mel + warmup           1.638       17.052  1339.96       356          1966
```

Default voice matrix after enabling GPU mel:

```text
scenario           asr_ms  mel_stft_ms  mel_norm_ms  vad_to_audio_ms  first_audio_samples
short_hello        583     0.652        6.084        1738             23040
persona_chat       656     0.766        7.349        1750             23040
preference_memory  660     1.279        7.500        1818             23040
task_memory        639     0.635        7.016        2209             23040
long_answer        667     0.782        8.380        2221             23040
```

Interpretation:

- The direct GPU DFT is sufficient for the ASR `n_fft=400` case: after warmup,
  the full short-prompt streaming run spends about 1.6ms in GPU STFT instead of
  about 141ms in CPU STFT.
- Warmup matters. Without it, the first GPU mel call paid roughly 400ms of
  kernel compilation and erased the TTFA win.
- End-to-end TTFA improves modestly because the remaining short-prompt path is
  dominated by ASR text prefill/decode, foreground first token, and TTS first
  audio. The mel bottleneck is now mostly removed for ASR.
- TTS reference speaker mel uses a different 24kHz / `n_fft=1024` / magnitude
  mel contract. The shared GPU mel scaffolding can be extended to it later, but
  it should be validated as a separate TTS-speaker-encoder pass.

## 2026-06-27 Foreground Cached Suffix Prefill

The next short-prompt profile showed the foreground final suffix as a larger
turn-time hotspot than ASR mel:

```text
before:
foreground_profile_prefilled_prompt_tokens  87
foreground_profile_prompt_suffix_tokens     31
foreground_profile_prompt_prefill_ms        751
simulated_vad_to_first_content_ms           1124
simulated_vad_to_first_audio_ms             1982
```

The 31-token suffix was still routed through the single-token decode path
because the old decode-path cutoff was 64 tokens. That assumption was wrong for
this shape: cached batch prefill is faster even though it attends over the
prefilled prefix.

Implemented:

- `AILA_FOREGROUND_DECODE_SUFFIX_TOKENS` controls the maximum cached suffix
  length routed through single-token decode kernels.
- The default is now 16 tokens, so the common 31-token final suffix uses cached
  batch prefill.
- `warmup_loaded_vlm` now warms a 64-token prefix followed by a 64-token cached
  suffix, moving the first incremental prefill buffer/JIT cost into model load.

Short prompt, 500ms ASR chunk/prefill cadence:

```text
mode                              prompt_prefill_ms  first_content_ms  first_tts_enqueue_ms  vad_to_audio_ms
decode suffix <=64                751                1124              1210                  1982
batch suffix <=16                 467                827               890                   1879
batch suffix <=16 + cached warmup 307                676               764                   1529
```

Interpretation:

- The fixed 12-frame / 23040-sample first TTS buffer is unchanged.
- The win comes from reducing foreground final prompt suffix latency before the
  first generated token can reach TTS.
- The current short-prompt TTFA hotspot is now the TTS first-audio path after
  enqueue: first code generation plus Mimi incremental decode. ASR tail is about
  360ms, and foreground first content is now below 700ms on the short smoke.
- A 500ms streaming matrix showed final suffixes up to 36 tokens and live
  incremental score growth to `seq_cap=48`, so the cached warmup shape was
  widened to 64 suffix tokens while keeping the same 128-token warmup prompt.

## 2026-06-27 TTS Stream Batch Probe

The next TTS profile split latency into two parts:

- `foreground_first_tts_enqueue_ms`: when foreground text is first ready for
  TTS.
- `tts_first_backend_audio_ms`: time from TTS backend start to the first audio
  callback for that text chunk.

The text side is highly output-dependent. Some turns emit an early punctuation
boundary such as `Kurash...`, producing a very small first text chunk. Other
turns wait for a much longer sentence boundary, producing 70-116 character
first chunks. A soft text splitter was added behind
`AILA_TTS_STREAM_TEXT_SOFT_MAX_CHARS`, but it remains disabled by default
because smaller text chunks can make later chunks larger and increase playback
gap risk.

Added `AILA_TTS_STREAM_BATCH_FRAMES` to test uniform audio frame batches. The
first host audio target follows the same batch size, so experiments do not use a
special smaller first packet. Default remains 12 frames.

Short streaming smoke:

```text
batch_frames  first_audio_samples  first_backend_codes_ms  first_backend_audio_ms  vad_to_audio_ms  chunk_sizes
12            23040                ~451-453                ~639-677                ~1880            23040,...
8             15360                345.619                 481.612                 1501             15360,...,tail smaller
10            19200                431.504                 724.044                 2168             19200,...,tail smaller
```

Interpretation:

- Uniform 8-frame batches can reduce TTFA on short output, and the callback
  sizes stay uniform with smaller tail chunks.
- The 8-frame path increases callback/Mimi decode frequency. A longer persona
  probe showed a first TTS segment with 64 frames taking 7.1s total backend
  time for about 5.1s of audio, which is a gap-risk signal.
- 10-frame batches were worse in the short smoke and are not promising.
- Keep the default at 12 frames for now. The next high-value work is reducing
  Mimi incremental decode cost or pipelining/overlapping TTS segment generation
  so a smaller uniform batch can keep up with playback.
