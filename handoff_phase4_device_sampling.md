# Phase 4 Device Sampling Handoff

## Summary

- Current branch/state is on commit `0ebd90e` with a dirty worktree.
- The main validated win in this turn is a new GPU sampling fast path in `SamplingOps.cpp`.
- This improves `sample_main` decode throughput meaningfully while keeping functionality stable.
- The previously attempted `raw weight layout + Linear JM decode` direction was experimentally negative and has been reverted as the default path.

## What Was Implemented

### 1. GPU sampling fast path

Primary file:

- [src/ops/SamplingOps.cpp](e:/RiderProjects/Aila/src/ops/SamplingOps.cpp)

Key changes:

- `can_use_device_sampling(...)` no longer returns `false` unconditionally.
- Device sampling is enabled for the no-penalty sampling path when:
  - `do_sample == true`
  - no repetition/presence/frequency penalties
  - `top_k <= 64`
  - `temperature > 0`
  - `top_p > 0`
- Added env gate:
  - `AILA_DEVICE_SAMPLING=1` by default
  - `AILA_DEVICE_SAMPLING=0` forces fallback to host sampling

Implementation shape:

- `sample_with_config_device(...)` is now a 2-stage GPU path:
  1. Multi-workgroup parallel scan over vocab to produce partial per-group top-k.
  2. Final GPU merge of partial top-k, then top-p filtering and sampling on device.
- Output is written directly to the device token buffer used by the fast decode chain in `Engine.hpp`.

Why this matters:

- Before this, sample mode still effectively paid a host-side per-token logits roundtrip.
- Now benchmark decode for sample mode can stay on the same fast device chain style used by greedy decode.

### 2. Minor kernel experiments that are currently not clearly beneficial

Files:

- [src/ops/NormOps.cpp](e:/RiderProjects/Aila/src/ops/NormOps.cpp)
- [src/ops/ElementwiseOps.cpp](e:/RiderProjects/Aila/src/ops/ElementwiseOps.cpp)

Changes:

- Added `seq_len == 1` fast path to `rms_norm(...)`
- Added `seq_len == 1` fast path to `fused_add_rms_norm(...)`
- Added `vec8` path to `sigmoid_mul(...)`

Important note:

- These changes did not produce a clear validated performance win in full-suite testing.
- They may be worth re-checking or reverting if the next agent wants a cleaner “sampling-only” commit.

### 3. Linear JM experiment remains present but disabled by default

File:

- [src/ops/Linear.cpp](e:/RiderProjects/Aila/src/ops/Linear.cpp)

State:

- There is still an experimental decode JM path in `Linear::forward(...)`.
- It is now gated by:
  - `AILA_LINEAR_DECODE_JM`
- Default is:
  - `AILA_LINEAR_DECODE_JM=0`

Important:

- This is not part of the validated gain for this phase.
- The current Qwen3.5 main path is not intended to rely on this for production performance.

## Validated Results

Main artifacts:

- [tmp/perf/phase4_device_sampling_v1/0ebd90e/summary.json](e:/RiderProjects/Aila/tmp/perf/phase4_device_sampling_v1/0ebd90e/summary.json)
- [tmp/perf/phase4_device_sampling_v1/0ebd90e/compare_vs_0ebd90e.json](e:/RiderProjects/Aila/tmp/perf/phase4_device_sampling_v1/0ebd90e/compare_vs_0ebd90e.json)

Full suite command used:

```powershell
.\perf_suite.ps1 -Preset phase_gate_q35_text -Phase phase4_device_sampling_v1
```

Comparison baseline used:

- [tmp/perf/phase4_cleanup_recheck_v1/0ebd90e](e:/RiderProjects/Aila/tmp/perf/phase4_cleanup_recheck_v1/0ebd90e)

### Bench deltas vs baseline

From `compare_vs_0ebd90e.json`:

- `greedy_main`
  - prefill: `605.69 -> 601.04 tok/s` (`-0.77%`)
  - decode: `90.02 -> 89.29 tok/s` (`-0.81%`)
- `sample_main`
  - prefill: `605.58 -> 600.31 tok/s` (`-0.87%`)
  - decode: `72.61 -> 83.79 tok/s` (`+15.40%`)

Interpretation:

- Prefill is effectively flat within variance.
- Greedy decode is effectively flat within variance.
- Sample decode shows the meaningful validated gain for this phase.

### Full suite summary snapshot

From `summary.json`:

- `greedy_short`
  - `pp 557.09 tok/s`
  - `tg 96.37 tok/s`
- `greedy_main`
  - `pp 601.04 tok/s`
  - `tg 89.29 tok/s`
- `sample_main`
  - `pp 600.31 tok/s`
  - `tg 83.79 tok/s`

Smoke:

- `4 / 4` passed

Decode profile hotspots still dominated by matmul-heavy stages:

- `ffn_proj`
- `down`
- `linear_delta`
- `linear_proj`
- `linear_o`

## Important Caution About Absolute Numbers

Machine state has been noisy across runs.

Observed in this turn:

- A smaller ad hoc same-code benchmark run showed roughly:
  - greedy decode around `100.9 tok/s`
  - sample decode around `95.2 tok/s`
- Full scripted suite later landed around:
  - greedy decode around `89.3 tok/s`
  - sample decode around `83.8 tok/s`

Conclusion:

- Use same-condition A/B inside the scripted workflow.
- Do not compare absolute numbers across thermally or system-disturbed runs.

## Failed / Rejected Direction

### Raw weight layout + Linear JM mainline attempt

This was tested and explicitly rejected.

What was attempted:

- Switch Qwen3.5 text weights back to raw `[out, in]` style layout in the main path.
- Try to leverage experimental Linear decode JM path from `Linear.cpp`.

Observed outcome:

- Memory dropped from about `2.91 GB` to about `2.42 GB`
- But performance regressed badly:
  - decode dropped from about `90 tok/s` to about `66 tok/s`
  - with the specific JM attempt enabled it fell further to about `43 tok/s`

Status:

- This was reverted as the default behavior.
- Do not continue this path as the mainline default unless a new implementation proves otherwise.

## Remaining Main Bottlenecks

Current decode profile still points to the same major hotspots:

- `ffn_proj`
- `down`
- `linear_delta`
- `linear_proj`
- `linear_o`

These are the most promising next optimization targets if the next agent continues toward decode ceiling.

## Recommended Next Steps For The Next Agent

### 1. Decide commit scope

Before committing, decide whether to:

- keep only the validated device sampling improvement, or
- keep device sampling plus the smaller norm/sigmoid experiments

Recommended:

- keep `SamplingOps.cpp` device sampling improvements
- re-check whether `NormOps.cpp` and `ElementwiseOps.cpp` changes should remain

### 2. Re-run a clean A/B for sampling

Recommended command:

```powershell
$env:AILA_DEVICE_SAMPLING='0'
.\bench.ps1 -Preset phase_gate_q35_text -CaseNames @('sample_main') -Phase sample_ab_disable_device
```

Then compare with default `AILA_DEVICE_SAMPLING=1`.

Goal:

- prove that the sample-mode improvement is specifically coming from the new device sampling path

### 3. If continuing performance work, return to decode matmul hotspots

Recommended direction:

- FFN decode matmul optimization
- `linear_delta` decode optimization
- projection matmul optimization

Avoid for now:

- making raw weight layout the default main path again

## Scripted Workflow To Reuse

Build:

```powershell
.\build.ps1 -Config Release
```

Full suite:

```powershell
.\perf_suite.ps1 -Preset phase_gate_q35_text -Phase phase4_device_sampling_v1
```

Compare vs previous stable baseline:

```powershell
.\perf_compare.ps1 -BaselineDir .\tmp\perf\phase4_cleanup_recheck_v1\0ebd90e -CurrentDir .\tmp\perf\phase4_device_sampling_v1\0ebd90e
```

## Current Worktree State

The worktree is dirty and includes more than just this turn’s changes.

Current `git status --short` includes:

- `bench.ps1`
- `build.ps1`
- `include/aila_api.h`
- `include/engine/Engine.hpp`
- `include/engine/Types.hpp`
- `src/api/aila_api.cpp`
- `src/main.cpp`
- `src/models/Qwen35HybridTextBackend.cpp`
- `src/models/Qwen35HybridTextBackend.hpp`
- `src/ops/AttentionOps.cpp`
- `src/ops/ElementwiseOps.cpp`
- `src/ops/Linear.cpp`
- `src/ops/NormOps.cpp`
- `src/ops/Ops.hpp`
- `src/ops/SamplingOps.cpp`
- `perf/`
- `perf_compare.ps1`
- `perf_suite.ps1`
- `profile_decode.ps1`

Implication:

- The next agent should not blindly commit everything.
- It should first narrow the commit to the intended phase scope.

## One-Line Executive Handoff

The validated win from this phase is the GPU sampling fast path in [src/ops/SamplingOps.cpp](e:/RiderProjects/Aila/src/ops/SamplingOps.cpp), which improved `sample_main` decode by about `15.4%` in scripted full-suite testing while keeping smoke tests green; the raw-weight/JM mainline experiment was negative and should not be revived as default.
