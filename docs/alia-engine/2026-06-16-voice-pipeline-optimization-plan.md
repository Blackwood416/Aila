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
