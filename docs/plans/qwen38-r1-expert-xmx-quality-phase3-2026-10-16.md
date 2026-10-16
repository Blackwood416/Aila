# R1 expert-XMX quality phase 3: fp16 storage is not the lever (2026-10-16)

Follows `qwen38-r1-expert-xmx-quality-phase2-2026-10-16.md`. Single A/B per
round (owner directive). Base HEAD `3a063c6` + the fp16 working-tree changes.

## 0. Result: the XMX deviation is accumulation-order bound

| config | unit-test kernel nrmse | e2e mean KL | e2e max NRMSE | top-1 |
|---|---:|---:|---:|---:|
| XMX bf16 storage (default) | 0.00389 | **0.0190** | 0.298 | 120/128 |
| XMX + c1 f32 | 0.00331 | 0.0172 | 0.268 | 120/128 |
| XMX + fp16 storage (`W_F16`) | **0.00185** | **0.0236** | 0.275 | 120/128 |
| (control) XMX bf16 vs XMX c1-f32, both XMX | - | 0.0099 | 0.164 | 123/128 |

Making the dequantized weights and staging activations 2.1x more precise at the
**kernel** level does not move the **e2e** deviation; the three storage variants
land inside a 0.017-0.024 KL band whose run-to-run spread (driven by
cache-state-dependent chunk grouping, control row) is ~0.010. The e2e floor is
therefore set by the batched-GEMM accumulation order (and its grouping), not by
the storage dtype of the operands. Storage precision is not the lever.

Timing cost of fp16: none (fx-a1 14K TTFT 63.985 / blocks 16.84/17.54/17.65;
fx-b1 63.996 / 16.96/17.41/17.65 - within noise; same 2 bytes per element).

## 1. What this means for the goal

* The prefill XMX path delivers -8..-10% per block and needs tensor-core GEMMs;
  its numeric class sits at ~0.02 mean KL / 120/128 teacher-forced top-1
  (contract: 0.01 KL, >=122/128), and that is irreducible by operand dtype.
* The exact ALU path cannot reach the prefill target on its own (its expert
  kernels are the dominant 7 s/block) - tensor-core grouping is the price of
  the only available 3-4x compute win.
* DECISION REQUIRED (owner): adopt the XMX numeric class with a documented
  quality waiver (KL ~0.02, 94% top-1), or keep the exact path and give up the
  current prefill lever. The feature remains default-off until that call.

## 2. Decode note: LM head Q8 re-screen (same session, single A/B)

`AILA_Q4E_LM_HEAD_Q8=1` on today's bundle: 17.379 -> **17.733 tok/s (+2.0%)**,
SSE content sha identical (bit-exact). Below the +5% default-on gate, so it
stays opt-in, but it is free to stack (no numeric change).
Receipts `tmp/q4e-sprint-r1-20261014/lh-{a1,b1}/`.

## 3. Receipts

* `tmp/q4e-expert-xmx-gate-20261016/{alu-off,xmx-def,xmx-c1f32,xmx-f16}.{log,logits.bf16}`
  (KL/NRMSE computed with `perf/qwen4exp/q4e_decode_summary.compare`).
* `tmp/q4e-sprint-r1-20261014/fx-{a1,b1}/` (fp16 timing A/B).
* `build-refactor/AilaMoeExpertXmxTests.exe` cases A-E (E: 0.00185).
* Flags: `AILA_Q4E_PREFILL_EXPERT_XMX_C1_F32`, `AILA_Q4E_PREFILL_EXPERT_XMX_W_F16`
  (both default 0; log line reports `c1_f32=` and `w_f16=`).
