# R1 prefill expert-XMX: first integrated smoke (2026-10-16)

Date: 2026-10-16. Base HEAD `c6b516f` + the WS-A/WS-B working-tree changes.
Design: `docs/plans/qwen38-r1-prefill-expert-xmx-plan-2026-10-16.md`;
quality-gate recipe: `docs/plans/qwen38-r1-expert-xmx-quality-gate-recipe-2026-10-16.md`.
Status: **smoke only** - no A/B/B/A, no quality gate yet. The feature is
default-off (`AILA_Q4E_PREFILL_EXPERT_XMX`), so the tree stays safe either way.

## What was built

* `src/ops/MoeExpertXmx.{hpp,cpp}` (new): per-chunk dequantize-once
  NF4->bf16 + activation gather + oneDNN batched bf16 GEMMs (gate_up, down) +
  the existing SwiGLU op + rank-ordered scatter; two bf16 weight banks; prim
  cache keyed by (B, M).
* `tests/ops/MoeExpertXmxTests.cpp` (new): CPU-reference numerics for the full
  chunk pipeline (bank 0 + ready, bank 1, B=1 flat codes).
* `src/models/Qwen4ExpBnb4Backend.{hpp,cpp}`: usage-descending chunk planner
  (B <= 16, B*M <= 2048, single-expert chunks up to 4096 rows, slice fallback),
  double-buffered plan staging, gated dispatch in `RunGroupedExperts`.

## Receipts

### Unit test (GPU, CPU reference)

`build-refactor/AilaMoeExpertXmxTests.exe`: **passed**.
caseA bank0+ready nrmse=0.0038895 max_abs=4.5e-7; caseB bank1 identical;
caseC B=1 flat nrmse=0.0041563 max_abs=2.7e-5.

### Smoke 1 - regression, root cause found (`px-smoke1`)

| metric | value |
|---|---|
| 14K TTFT | 84.6 s (baseline 69.6 s `[dpe-b1]`) |
| steady 4096 blocks | 22.5 / 23.4 s (baseline 19.05-19.89 s) |
| evidence | `[Q4E PrefillExpertXmx] enabled=1 ... plan_bytes=329344` |

Root cause: the oneDNN primitive cache was capped at 32 (B*, M*) shapes and
its eviction path calls `ctx.synchronize()`. A layer visits ~10-30 distinct
shapes, so the cap thrashed and drained the queue repeatedly mid-prefill.

### Smoke 2 - cap raised to 512 (`px-smoke2`)

| metric | XMX (smoke2) | baseline (`dpe-b1`) | delta |
|---|---:|---:|---:|
| 14K TTFT | **65.1 s** | 69.6 s | **-6.5%** |
| steady 4096 blocks | **17.41 / 17.59 / 17.97 s** | 19.05 / 19.64 / 19.89 s | **-8.6..-10.4%** |
| SSE content sha | differs (expected: bf16 weights + new accumulation order) | - | not bit-exact by design |

Both smoke arms ran the standard 14K/`--diag` runner with the hotset + int8 KV
bundle; worker sha `58906760...` for smoke2.

## Reading and next steps

1. The integrated path is now a real (if modest) win: ~-9% per prefill block.
   It is far from the probe's -30% because the probe priced the device pipeline
   only; the production path adds ~200 launches per layer, the SwiGLU pass, plan
   uploads, and (in-order queue) no dequant/GEMM overlap.
2. The remaining gap is a kernel-arm question, not a correctness question:
   schedule the debug knob next - (a) count launches per layer, (b) check the
   (16, 96) GEMM efficiency vs the probe's larger-M chunks, (c) consider a
   persistent/merged chunk pipeline (single launch per chunk with an internal
   state machine).
3. Before any adoption: the section-4 quality gate (teacher-forced KL/top-1)
   and a full A/B/B/A on the 14K arm; the block gate in the plan is <= 13.5 s,
   so the current 17.4 s is not yet at the target.
4. Not affected: decode rates stay ~17-18 tok/s in both smoke arms.

## Evidence

* `tmp/q4e-sprint-r1-20261014/px-smoke1*/` and `px-smoke2*/` (result.json,
  worker logs with the `[Q4E PrefillExpertXmx]`/`[Q4E AdaptiveWork]` lines).
* `build-refactor/AilaMoeExpertXmxTests.exe` output (see above).
