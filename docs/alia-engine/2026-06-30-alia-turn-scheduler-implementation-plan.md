# Alia Turn Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conservative turn-level scheduler that makes ASR partial, VLM prefill/speculation, and TTS first-audio decisions observable and prevents cached-prefix reuse from hurting short-prompt TTFA.

**Architecture:** Add a pure policy helper in `src/alia` and integrate it first with the real smoke stream-ASR path and foreground final prompt reuse. The scheduler records decisions and reasons; model execution stays inside the existing ASR, foreground, and TTS pipelines.

**Tech Stack:** C++17, existing Alia native pipeline, PowerShell real-model smoke/matrix scripts, Mimo output ASR script.

---

## File Structure

- Create: `src/alia/AliaTurnScheduler.hpp`
  - Scheduler config, input structs, output decision structs, and compact reason strings.
- Create: `src/alia/AliaTurnScheduler.cpp`
  - Environment-backed config defaults and deterministic decision logic.
- Modify: `tools/alia/AliaRealModelSmoke.cpp`
  - Route stream-ASR prefill/speculative decisions through the scheduler.
  - Emit `scheduler_decision=...` lines and aggregate counters.
- Modify: `tools/alia/RunAliaVoiceScenarioMatrix.ps1`
  - Add scheduler summary columns so matrix CSV can compare policy runs.
- Modify: `src/alia/AliaForegroundPipeline.cpp`
  - Add final cached-prefix reuse policy before `generate_with_loaded_vlm`.
- Modify: `src/alia/AliaForegroundPipeline.hpp`
  - Add metrics for cached-prefix rejection count and reason.
- Modify: `docs/alia-engine/2026-06-30-alia-turn-scheduler-design.md`
  - Append validation results after each real-model round.

## Task 1: Add The Scheduler Policy Helper

**Files:**
- Create: `src/alia/AliaTurnScheduler.hpp`
- Create: `src/alia/AliaTurnScheduler.cpp`

- [ ] **Step 1: Add scheduler declarations**

Create `src/alia/AliaTurnScheduler.hpp`:

```cpp
#pragma once

#include <string>

namespace aila::alia {

struct AliaTurnSchedulerConfig {
    bool enabled = true;
    int min_prefill_text_chars = 12;
    int min_incremental_text_chars = 8;
    int max_cached_final_suffix_tokens = 16;
    int speculative_min_chars = 24;
    int speculative_required_stable_ticks = 1;
    int speculative_min_ascii_words = 3;
};

struct AliaAsrSchedulerEvent {
    double chunk_end_ms = 0.0;
    bool final_chunk = false;
    bool text_changed = false;
    int stable_chars = 0;
    int partial_chars = 0;
    int combined_chars = 0;
    int ascii_words = 0;
};

struct AliaPrefillSchedulerState {
    bool speculative_enabled = false;
    bool speculative_started = false;
    int last_prefill_text_chars = 0;
    int candidate_stable_ticks = 0;
};

struct AliaPrefillDecision {
    bool prefill = false;
    bool start_speculative = false;
    std::string reason;
};

struct AliaFinalPrefixDecision {
    bool use_cached_prefix = true;
    std::string reason;
};

AliaTurnSchedulerConfig read_alia_turn_scheduler_config();

AliaPrefillDecision decide_asr_prefill(
    const AliaTurnSchedulerConfig& config,
    const AliaAsrSchedulerEvent& event,
    const AliaPrefillSchedulerState& state);

AliaFinalPrefixDecision decide_final_cached_prefix(
    const AliaTurnSchedulerConfig& config,
    int prefilled_prompt_tokens,
    int prompt_suffix_tokens);

}  // namespace aila::alia
```

- [ ] **Step 2: Add deterministic policy logic**

Create `src/alia/AliaTurnScheduler.cpp`:

```cpp
#include "AliaTurnScheduler.hpp"

#include "../utils/EnvUtils.hpp"

#include <algorithm>

namespace aila::alia {

AliaTurnSchedulerConfig read_alia_turn_scheduler_config() {
    AliaTurnSchedulerConfig config;
    config.enabled = aila::env::read_flag("AILA_TURN_SCHEDULER", true);
    config.min_prefill_text_chars = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MIN_PREFILL_TEXT_CHARS", 12));
    config.min_incremental_text_chars = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MIN_INCREMENTAL_TEXT_CHARS", 8));
    config.max_cached_final_suffix_tokens = std::max(
        0, aila::env::read_int_raw("AILA_TURN_SCHEDULER_MAX_CACHED_FINAL_SUFFIX_TOKENS", 16));
    config.speculative_min_chars = std::max(
        1, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_MIN_CHARS", 24));
    config.speculative_required_stable_ticks = std::max(
        1, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_STABLE_TICKS", 1));
    config.speculative_min_ascii_words = std::max(
        0, aila::env::read_int_raw("AILA_FOREGROUND_SPECULATIVE_MIN_ASCII_WORDS", 3));
    return config;
}

AliaPrefillDecision decide_asr_prefill(
    const AliaTurnSchedulerConfig& config,
    const AliaAsrSchedulerEvent& event,
    const AliaPrefillSchedulerState& state) {
    AliaPrefillDecision decision;
    if (!config.enabled) {
        decision.prefill = event.text_changed;
        decision.reason = decision.prefill ? "scheduler disabled: changed text" :
                                             "scheduler disabled: unchanged text";
        return decision;
    }
    if (!event.text_changed) {
        decision.reason = "unchanged text";
        return decision;
    }
    if (event.combined_chars < config.min_prefill_text_chars) {
        decision.reason = "text shorter than prefill minimum";
        return decision;
    }
    const int delta_chars = event.combined_chars - state.last_prefill_text_chars;
    if (!event.final_chunk &&
        state.last_prefill_text_chars > 0 &&
        delta_chars >= 0 &&
        delta_chars < config.min_incremental_text_chars) {
        decision.reason = "incremental text delta too small";
        return decision;
    }

    decision.prefill = !state.speculative_started;
    decision.reason = event.final_chunk ? "final text prefill" : "partial text prefill";

    if (state.speculative_enabled &&
        !event.final_chunk &&
        !state.speculative_started &&
        event.combined_chars >= config.speculative_min_chars &&
        state.candidate_stable_ticks >= config.speculative_required_stable_ticks &&
        (event.ascii_words == 0 || event.ascii_words >= config.speculative_min_ascii_words)) {
        decision.start_speculative = true;
        decision.prefill = false;
        decision.reason = "start speculative foreground";
    }
    return decision;
}

AliaFinalPrefixDecision decide_final_cached_prefix(
    const AliaTurnSchedulerConfig& config,
    int prefilled_prompt_tokens,
    int prompt_suffix_tokens) {
    AliaFinalPrefixDecision decision;
    if (!config.enabled) {
        decision.reason = "scheduler disabled";
        return decision;
    }
    if (prefilled_prompt_tokens <= 0) {
        decision.reason = "no cached prefix";
        return decision;
    }
    if (prompt_suffix_tokens <= config.max_cached_final_suffix_tokens) {
        decision.reason = "cached suffix within fast threshold";
        return decision;
    }
    decision.use_cached_prefix = false;
    decision.reason = "cached suffix exceeds fast threshold";
    return decision;
}

}  // namespace aila::alia
```

- [ ] **Step 3: Build**

Run:

```powershell
cmake --build build --target AliaEngine --config Release
```

Expected: build succeeds. If the new `.cpp` is not picked up by the build, add it to the existing Alia source list in `CMakeLists.txt` following neighboring `src/alia/*.cpp` entries.

- [ ] **Step 4: Commit**

```powershell
git add src/alia/AliaTurnScheduler.hpp src/alia/AliaTurnScheduler.cpp CMakeLists.txt
git commit -m "feat: add Alia turn scheduler policy"
```

## Task 2: Add Scheduler Decision Logs To Real Smoke

**Files:**
- Modify: `tools/alia/AliaRealModelSmoke.cpp`
- Modify: `tools/alia/RunAliaVoiceScenarioMatrix.ps1`

- [ ] **Step 1: Include scheduler helper**

In `tools/alia/AliaRealModelSmoke.cpp`, add:

```cpp
#include "../../src/alia/AliaTurnScheduler.hpp"
```

- [ ] **Step 2: Replace ad hoc stream prefill gating**

Inside the `opts.stream_asr_prefill` loop after `get_asr_text(...)`, build an `AliaAsrSchedulerEvent`, call `decide_asr_prefill`, and print:

```cpp
std::cout << "scheduler_decision="
          << "chunk_end_ms:" << chunk_end_ms
          << ",final:" << (final_chunk ? "true" : "false")
          << ",prefill:" << (decision.prefill ? "true" : "false")
          << ",speculative:" << (decision.start_speculative ? "true" : "false")
          << ",reason:" << quote(decision.reason)
          << ",stable_chars:" << stable_text.size()
          << ",partial_chars:" << partial_text.size()
          << "\n";
```

Keep existing counters and behavior equivalent except for scheduler-approved skips.

- [ ] **Step 3: Add aggregate counters**

Print these summary lines near the existing ASR stream metrics:

```cpp
std::cout << "scheduler_prefill_allowed=" << scheduler_prefill_allowed << "\n"
          << "scheduler_prefill_skipped=" << scheduler_prefill_skipped << "\n"
          << "scheduler_speculative_allowed=" << scheduler_speculative_allowed << "\n"
          << "scheduler_last_reason=" << quote(scheduler_last_reason) << "\n";
```

- [ ] **Step 4: Add CSV fields**

In `tools/alia/RunAliaVoiceScenarioMatrix.ps1`, add row fields:

```powershell
scheduler_prefill_allowed = Get-ValueOrEmpty $values "scheduler_prefill_allowed"
scheduler_prefill_skipped = Get-ValueOrEmpty $values "scheduler_prefill_skipped"
scheduler_speculative_allowed = Get-ValueOrEmpty $values "scheduler_speculative_allowed"
scheduler_last_reason = Get-ValueOrEmpty $values "scheduler_last_reason"
```

- [ ] **Step 5: Build and smoke**

Run:

```powershell
cmake --build build --target AliaEngine --config Release
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe -NoGenerateAudio `
  -StreamAsrPrefill -StreamChunkMs 500 -StreamPrefillIntervalMs 500 `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix_fused_conv2_default_retry\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\scheduler_logs_short\short.wav' `
  -LogPath 'tmp\alia-real-smoke\scheduler_logs_short\short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500
```

Expected: `ALIA_REAL_MODEL_SMOKE_PASS` and at least one `scheduler_decision=` line in the log.

- [ ] **Step 6: Commit**

```powershell
git add tools/alia/AliaRealModelSmoke.cpp tools/alia/RunAliaVoiceScenarioMatrix.ps1
git commit -m "perf: trace Alia turn scheduler decisions"
```

## Task 3: Gate Harmful Final Cached-Prefix Reuse

**Files:**
- Modify: `src/alia/AliaForegroundPipeline.hpp`
- Modify: `src/alia/AliaForegroundPipeline.cpp`

- [ ] **Step 1: Add metrics**

Add fields to `AliaForegroundMetrics`:

```cpp
int final_cached_prefix_rejected = 0;
std::string final_cached_prefix_reject_reason;
```

- [ ] **Step 2: Apply final prefix decision**

In `generate_with_loaded_vlm`, after `prompt_tokens_to_forward` is computed and before the prompt is forwarded, call `decide_final_cached_prefix(...)`. If it returns `use_cached_prefix=false`, reset the backend and set `prefilled_prompt_tokens=0`.

- [ ] **Step 3: Preserve fallback**

Add environment fallback:

```text
AILA_TURN_SCHEDULER=0
```

Expected behavior: with scheduler disabled, cached prefix reuse follows the old path.

- [ ] **Step 4: Build and A/B short smoke**

Run scheduler enabled and disabled:

```powershell
$env:AILA_TURN_SCHEDULER = "1"
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe -NoGenerateAudio `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix_fused_conv2_default_retry\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\scheduler_prefix_gate_on\short.wav' `
  -LogPath 'tmp\alia-real-smoke\scheduler_prefix_gate_on\short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500

$env:AILA_TURN_SCHEDULER = "0"
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe -NoGenerateAudio `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix_fused_conv2_default_retry\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\scheduler_prefix_gate_off\short.wav' `
  -LogPath 'tmp\alia-real-smoke\scheduler_prefix_gate_off\short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500
```

Expected: enabled run should not be slower than disabled due to a large cached final suffix. Compare `foreground_profile_prompt_prefill_ms`, `foreground_profile_prefilled_prompt_tokens`, and `simulated_vad_to_first_audio_ms`.

- [ ] **Step 5: Commit**

```powershell
git add src/alia/AliaForegroundPipeline.hpp src/alia/AliaForegroundPipeline.cpp
git commit -m "perf: gate Alia cached prompt reuse by suffix cost"
```

## Task 4: Matrix And Decide Defaults

**Files:**
- Modify: `docs/alia-engine/2026-06-30-alia-turn-scheduler-design.md`

- [ ] **Step 1: Run full matrix with output ASR**

```powershell
if (-not $env:MIMO_API_KEY) {
  $env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable('MIMO_API_KEY', 'Machine')
}
$env:AILA_TURN_SCHEDULER = "1"
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\voice_matrix_turn_scheduler_probe'
```

- [ ] **Step 2: Compare against latest default**

Use:

```powershell
Import-Csv 'tmp\alia-real-smoke\voice_matrix_turn_scheduler_probe\summary.csv' |
  Format-Table scenario,simulated_vad_to_first_audio_ms,foreground_profile_prompt_prefill_ms,foreground_profile_prefilled_prompt_tokens,tts_first_backend_audio_ms,scheduler_prefill_allowed,scheduler_prefill_skipped -AutoSize
```

Expected: no output ASR failures; short prompt should not regress from the latest default matrix.

- [ ] **Step 3: Document results**

Append a section to `docs/alia-engine/2026-06-30-alia-turn-scheduler-design.md` with:

```text
matrix path
pass count
average simulated_vad_to_first_audio_ms
average foreground_profile_prompt_prefill_ms
output ASR notes
default-on/default-off decision
```

- [ ] **Step 4: Commit**

```powershell
git add docs/alia-engine/2026-06-30-alia-turn-scheduler-design.md
git commit -m "docs: record Alia turn scheduler probe"
```

## Self-Review

- Spec coverage: the plan covers scheduler policy, smoke observability, final cached-prefix gating, and real model validation.
- Placeholder scan: no placeholder implementation steps remain.
- Scope check: tool calls, vision, Computer Use, and real-VAD speculative silence are deferred outside this plan.
