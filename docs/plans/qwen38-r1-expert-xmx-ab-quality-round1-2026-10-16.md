# R1 prefill expert-XMX: A/B timing + quality screen, round 1 (2026-10-16)

Follows `qwen38-r1-prefill-expert-xmx-smoke-2026-10-16.md` (the smoke that found
and fixed the prim-cache eviction regression). Base HEAD `1eebcde`.
Discipline note: per the owner's 2026-10-16 directive, routine rounds use a
single A/B; A/B/B/A stays reserved for default-on adoption gates.

## 0. Verdict

* **Timing: win.** 14K TTFT 71.16 -> 65.32 s (-8.2%); steady 4096 blocks
  19.57/20.06/20.32 -> 17.70/17.67/17.89 s (-9.6% / -11.9% / -12.0%).
* **Quality: NOT PASSED as integrated.** Teacher-forced top-1 agreement vs the
  ALU path is **119/128 = 93.0%**, below the pre-registered >= 122/128 (95.3%)
  bar. The XMX path stays opt-in; adoption needs a numeric-tightening round.
* The probe's bit-identical replay requirement cannot be met by design (see
  section 2.1); the standard comparator path needs an env-gated tolerance
  before it can produce a KL number for this numeric class.

## 1. Timing A/B (single A/B, 14K arm, --diag both arms)

| metric | xb-a1 (ALU) | xb-b1 (XMX) | delta |
|---|---:|---:|---:|
| 14K TTFT | 71.155 s | 65.322 s | **-8.2%** |
| 6000 TTFT | 120.434 s | (not the gate metric) | - |
| steady 4096 blocks | 19.57 / 20.06 / 20.32 s | 17.70 / 17.67 / 17.89 s | -9.6..-12.0% |
| SSE content sha | `0efc641f...` | `f503834d...` | changed (expected) |

Evidence: `tmp/q4e-sprint-r1-20261014/xb-{a1,b1}/`, worker sha `58906760...`;
the B arm logs `[Q4E PrefillExpertXmx] enabled=1 ... plan_bytes=329344`.

## 2. Quality screen

### 2.1 Probe determinism

`AilaQwen4ExpToleranceTests --decode-model` (prompt 512, rows 1, repeats 2):

* alu-off: exit 0 (passes the identical-replay assert).
* xmx-on: exit 1, `Identical continuation replay changed logits bits`
  (`tests/models/Qwen4ExpDecodeProbe.hpp:103`, hard-required `repeats >= 2`).

Mechanism (by design, not an uninitialised read): each pass re-prefills from a
reset context, but the shared adaptive cache evolves, so the same routing can
be served by different resident/arena/bank segments between passes. XMX chunk
grouping follows the segment, and bf16 GEMM accumulation is grouping sensitive,
so the last bits move; the ALU path is grouping invariant and stays bit exact.

Consequence: the logits/KL comparator (`q4e_decode_summary.py`, which also
requires `step_hashes == passes[0]`) cannot process this class of change until
the probe gains an env-gated replay-drift allowance; its `meta.json` is also
only produced by `perf/qwen4exp/q4e_execution_probe.ps1`, not by the bare exe.

### 2.2 Teacher-forced top-1 agreement (from the pass-0 records)

Both arms printed their pass-0 `top1` arrays (128 teacher-forced positions):
**119/128 = 93.0% agreement** (first divergence at position 6: 6103 vs 66881).
`step_hashes` are identical in 0/128 (last-bit drift everywhere, as expected).

### 2.3 Greedy-continuation proxy (existing 6000/14K receipts)

| request | ALU len | XMX len | common prefix (chars) | identical |
|---|---:|---:|---:|---|
| context-6000 | 154 | 156 | 22 | no |
| context-6000-continue | 197 | 175 | 45 | no |
| context-14000 | 168 | 184 | 41 | no |
| context-14000-continue | 175 | 170 | 45 | no |

Consistent with the 93% teacher-forced figure: the greedy text diverges once
the first near-tie argmax flips (then everything downstream differs).

## 3. Next levers (ordered)

1. **Numeric tightening (quality gate path).** The recipe's open question 2:
   f32 gate_up dst (`c1`) and/or f32 SwiGLU output (`a2`) instead of bf16. The
   kernel unit test sits at nrmse 0.0039; recovering ~2-3 points of top-1 is
   the goal. Cost: ~+126 MB/layer of traffic for f32 c1 (measure on the same
   single A/B; expect a few points of the -9.7% back).
2. **Probe tolerance (measurement path).** Env-gated replay-drift allowance in
   `Qwen4ExpDecodeProbe.hpp` + per-pass logits retention, so the real mean/max
   KL can be computed; use `q4e_execution_probe.ps1` for `meta.json`.
3. **Replay stabilisation (optional).** Planning chunks from the expert id/usage
   set independent of the serving segment would make the path replay-stable
   (and simplify all future gates), at the cost of the bank-lookahead structure.
4. **Timing knobs still on deck:** `AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M` (staged,
   default 0, not yet measured), merging the four plan uploads into one, and
   overlapping dequant with GEMMs (needs an out-of-order path).

## 4. Evidence

* `tmp/q4e-sprint-r1-20261014/xb-{a1,b1}/result.json` + `xb-*-worker.log`.
* `tmp/q4e-expert-xmx-gate-20261016/{alu-off,xmx-on}.log` (+ alu-off logits
  dump; the xmx-on dump is aborted by the replay assert).
* `tmp/q4e-sprint-r1-20261014/run-xmx-ab.ps1`, `run-xmx-gate.ps1`.
