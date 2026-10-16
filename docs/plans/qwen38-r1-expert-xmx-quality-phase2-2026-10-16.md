# R1 expert-XMX quality phase 2: KL measurement, c1-f32, min-M (2026-10-16)

Follows `qwen38-r1-expert-xmx-ab-quality-round1-2026-10-16.md`. Base HEAD
`ab5a2cb`. Discipline: single A/B per round (owner directive 2026-10-16).

## 0. Results

1. **MIN_M=256 + ROWS=4096 (single A/B): win, kept as the tuning baseline.**
   14K TTFT 65.52 -> **64.21 s (-2.0%)**; steady 4096 blocks 17.57/17.93/17.89
   -> **16.98/17.51/17.70 s** (-1.1..-3.4%). Wider (16,256)-shaped batched
   GEMMs; padded A rows are nearly free while the GEMM is weight-bandwidth
   bound. Receipts `tmp/q4e-sprint-r1-20261014/mm-{a1,b1}/`.
2. **KL/NRMSE measured directly from the probe's logits dumps** (the
   drift-tolerant probe now completes; the repo's own
   `q4e_decode_summary.compare()` was called on the dumps, bypassing its
   meta/replay preconditions):

   | pair | top-1 | mean KL | max KL | max NRMSE | contract |
   |---|---:|---:|---:|---:|---|
   | ALU vs XMX (bf16 c1) | 120/128 | **0.0190** | 0.160 | **0.298** | KL <= 0.01, NRMSE <= 0.03 |
   | ALU vs XMX (c1 f32) | 120/128 | **0.0172** | 0.245 | **0.268** | same |
   | XMX bf16-c1 vs XMX c1-f32 (both XMX!) | 123/128 | 0.0099 | 0.061 | 0.164 | - |

   Two XMX configurations that differ only in the gate_up accumulation dtype
   are already 0.0099 KL / 0.164 NRMSE apart, i.e. the deviation is
   **class-level** (bf16 storage of dequantized weights + staging activations),
   not a small last-bit jitter. Row-level for ALU vs XMX: median NRMSE 0.089,
   max 0.298, median per-element |dlogit| ~1.9 (max 6.0).
3. **c1-f32 alone is insufficient**: kernel nrmse 0.00389 -> 0.00331 (unit
   test case D, ~15% tighter) and e2e KL 0.0190 -> 0.0172, but top-1 stays
   120/128. The flag stays opt-in (`AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32`).
4. Therefore the bf16-storage XMX path does **not** meet the repo's numeric
   contract as integrated; it remains default-off. The documented decision
   points are: (a) precision upgrade (fp16 storage keeps 11 mantissa bits at
   the same 2 bytes/element - in progress), or (b) explicit owner-level risk
   acceptance of KL ~0.017 with 94% teacher-forced top-1.

## 1. Measurement plumbing added

* `AILA_Q4E_DECODE_PROBE_ALLOW_REPLAY_DRIFT=1` (commit `ab5a2cb`): the probe
  records `replay_drift` per pass instead of failing the replay assert; pass-0
  logits/top-1 remain the comparable artifact. The XMX path drifts 128/128 on
  the replay pass because its bf16 grouping depends on which cache source
  serves the routing (see round-1 section 2.1).
* `q4e_decode_summary.compare()` can be invoked directly on
  `<prefix>.logits.bf16` pairs; the meta.json/replay preconditions of
  `read_run` are only needed for the stock CLI flow.

## 2. Evidence

* `tmp/q4e-sprint-r1-20261014/mm-{a1,b1}/` (min-M A/B).
* `tmp/q4e-expert-xmx-gate-20261016/{alu-off,xmx-def,xmx-c1f32}.{log,logits.bf16}`.
* `build-refactor/AilaMoeExpertXmxTests.exe` case D output (0.00331 vs 0.00389).

## 3. Next

1. fp16 storage flag (`AILA_Q4E_PREFILL_EXPERT_XMX_W_F16`) - implement, unit
   test, then one drift-tolerant probe round (alu-off reference already on
   disk) to price KL/top-1; then a single A/B for its timing cost.
2. If fp16 still misses the gate, bring the measured trade (KL/agreement vs
   -8..-10% prefill time) to the owner as an explicit decision.
