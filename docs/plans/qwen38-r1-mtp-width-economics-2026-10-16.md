# R1-P3: MTP width economics and the per-row marginal cost

Date: 2026-10-16. HEAD `2e25fb8`. Scope: Qwen4Exp decode, the "decode >= 20
tok/s via MTP width / marginal row cost" path.
Method: CPU-only - source reading plus existing receipts and logs. No GPU run,
no build, no src change, no commit. Every number carries a source; numbers
marked [est] show their derivation.

Inputs treated as measured (cross-checked against the receipts listed in
section 9):

- Production width 2 (2 rows = 1 draft), g1-b1/b2: 15.543/15.592 tok/s
  (runner field); continue-window recompute done_at-first_content gives
  15.666 (g1-b1), 15.558 (s2x-a1), 15.116 (s2x-b1).
- Verify window (count=2) submitted median 92.26 ms; host split
  experts_host 53.97 = cpu_compute 29.47 + cpu_join 24.84; staged=0, h2d=0
  (g1-b1-worker.log, n=151).
- Draft residual: W - verify = 1.748/15.543 - 0.09226 = 112.5 - 92.3 =
  19.9 ms/window (G4, `qwen38-r1-mtp-step-budget-2026-10-15.md` sec 1.1).
- Chain acceptance on the long 1833 prompt, n=124 anchors, from the C pass
  of `tmp/q4e-sprint-r1-20261014/mtp-top1/top1-gate.json`: p1 = 92/124 =
  0.7419354839, q2 = 51/92 = 0.5543478261, q3 = 27/51 = 0.5294117647.
  Short 512 prompt: p1 = 114/124 = 0.9193548, q2 = q3 = 1.0.
- Probe-measured verify blocks ([MTP VERIFY] block_ms medians, first block
  trimmed: long1833.utf8.log): width 1: 66.31, width 2: 109.28,
  width 3: 159.39, width 4: 198.67 ms.

## 1. Notation - code width vs the brief's 1+3p model

`width` = rows in the verify forward (Qwen4ExpMtpRuntime.cpp:27). Code
semantics: width w = w rows = w-1 drafts; committed tokens per window follow
the chain

    T(2) = 1 + p1
    T(3) = 1 + p1 + p1*q2
    T(4) = 1 + p1 + p1*q2 + p1*q2*q3

The 1+3p linear model used in the G4 brief corresponds to 3 *drafts* = 4
rows and overstates T(4) (2.371 chain vs 3.226 "linear" at p=.742). All
tables below use the chain. Sanity: the [MTP PASS] accepted_prefix means
reproduce the chain exactly (long: mean 1+E[ap] = 2.3710; short: 3.7581),
and the live [MTP CYCLES] runs give the same totals within their own
trajectories (long live: T2=1.7714, T3=2.2963, T4=2.3846; short live:
T2=1.9683, T3=2.7778, T4=3.8182).

## 2. Width economics (long-prompt chain, primary)

rate = T / W, so 20 tok/s needs W <= 50*T; T1 parity (17.33 tok/s =
57.70 ms/token) needs W <= 57.70*T. W = verify + draft total; current W2 =
92.26 + 19.9 = 112.16 ms (rate 15.53 measured).

| width | rows/drafts | T (long) | W <= for 20 | W <= T1 parity | current W [est] | rate [est] | verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 2 | 2/1 | 1.7419 | 87.10 | 100.52 | 112.16 (measured) | 15.53 (measured) | needs -25.1 ms |
| 3 | 3/2 | 2.1532 | 107.66 | 124.25 | 92.26 + 42.97 + 29.0 = 164.2 | 13.11 | -56.6 ms short |
| 4 | 4/3 | 2.3710 | 118.55 | 136.81 | 92.26 + 42.97 + 50.11 + 38.1 = 223.4 | 10.61 | -104.9 ms short |

Sources: T from section 1; current W3/W4 = measured verify(2) + the measured
probe marginal rows (+42.97, +50.11 ms) + draft totals 29.0 = 19.9+9.1 and
38.1 = 19.9+2*9.1 (G4: each extra draft = one full Step + Select + bookkeeping
= +8.5-9.7 [est]; probe live draft_ms median 6.22 is the Step-only floor).

Short-prompt chain (512, best case): T = 1.9194 / 2.8387 / 3.7581 -> W <=
95.97 / 141.94 / 187.90 at 20 tok/s. Width 4 would need W <= 187.9 but its
[est] W is 219.8 (production marginals) to 236.8 (probe blocks + 38.1), i.e.
16.1-17.4 tok/s. Even the most favorable measured chain does not reach 20.

p1 sensitivity (holding q2, q3; W20 / W_T1parity in ms):

| p1 | T2 | T3 | T4 | W20 2/3/4 | parity 2/3/4 |
|---:|---:|---:|---:|---|---|
| 0.6919 | 1.6919 | 2.0755 | 2.2786 | 84.60 / 103.78 / 113.93 | 97.63 / 119.76 / 131.48 |
| 0.7419 | 1.7419 | 2.1532 | 2.3710 | 87.10 / 107.66 / 118.55 | 100.52 / 124.25 / 136.81 |
| 0.7919 | 1.7919 | 2.2309 | 2.4634 | 89.60 / 111.55 / 123.17 | 103.40 / 128.73 / 142.14 |

Each +0.05 of p1 buys roughly +2.5 ms of window budget at width 2. p1 is a
modest lever; closing a 10-25 ms gap with p1 alone would need +0.10 to +0.20.

## 3. Required per-row marginal cost (the punchline)

Budget = W_target - draft_total - row-1 cost. Row 1 (T1 shape) = 52.5 ms
(same-run count=1 submitted median, g1-b1, n=2) to 55.0 ms (d1-smoke, G4).
Measured marginal row today: +39.76 ms (production medians) and +42.97 /
+50.11 / +39.28 ms (probe block deltas).

| target | width | budget W | rows after row 1 must cost <= | measured | gap |
|---|---:|---:|---:|---:|---:|
| 20 tok/s | 2 | 87.10 | row 2: +14.70 (base 52.5) / +12.20 (base 55.0) | +39.76 | -25 to -27 ms |
| 20 tok/s | 3 | 107.66 | rows 2-3: <= 13.08 each | +42.97 / +50.11 | -30 to -37 per row |
| 20 tok/s | 4 | 118.55 | rows 2-4: <= 9.32 each | +42.97 / +50.11 / +39.28 | -30 to -41 per row |
| T1 parity | 2 | 100.52 | row 2: <= 24.01 | +39.76 | -15.8 ms |
| T1 parity | 3 | 124.25 | rows 2-3: <= 21.37 each | +42.97 / +50.11 | -22 to -29 per row |
| T1 parity | 4 | 136.81 | rows 2-4: <= 15.40 each | ~+44 avg | -28 per row |

Equivalent negative-marginal form: today's 2-row verify (92.26) already
exceeds the width-3 budget verify(3) <= 78.66 ms before any third row, so
width 3 cannot reach 20 tok/s even with a *free* third row until the
2-row window itself loses >= 13.6 ms. Width 3/4 are therefore NO-GO under
every measured marginal-row cost; the revival gate is a marginal row of
~9-13 ms, i.e. a 3-4x cut, plus draft overhead that grows with width.

## 4. Per-row marginal cost decomposition (measured)

Production medians, g1-b1-worker.log (count=1: n=2; count=2: n=151):

| field | count=1 | count=2 | delta (row 2) |
|---|---:|---:|---:|
| submitted_ms | 52.50 | 92.26 | +39.76 |
| router_wait_ms | 12.59 | 12.64 | +0.05 |
| experts_host_ms | 19.18 | 53.97 | +34.79 |
| cpu_compute_ms | 12.77 | 29.47 | +16.70 |
| cpu_join_ms | 9.75 | 24.84 | +15.09 |
| hc/linear/full/router/misc | 20.73 | 25.70 | +4.97 |
| cpu_tasks | 244 | 587 | +343 |
| hits (GPU hot) | 236 | 373 | +137 |

Routed selections per window = 48 layers * 10 topk * 2 rows = 960 =
373 hot + 587 cold (hits+misses, consistent in all four MTP arms). Per-task
wall: cpu_compute 29.47/587 = 50.2 us; the marginal row's 343 extra tasks
cost 48.7 us each (16.70 ms / 343). The marginal row is ~82% host
(compute + join) and ~18% GPU/other (+4.97). Probe cross-check: the same
marginal row measured as verify-block deltas is +43 to +50 ms (which
includes the probe's own accounting), confirming row scaling is ~linear and
expensive.

## 5. Are expert weights reloaded when both rows select the same expert?

YES. The CPU path has no per-expert dedup; the same expert selected by two
rows is executed as two independent tasks, each streaming the expert's full
packed weights.

- Task build: `Qwen4ExpExpertExecutor::StartArenaColdRows`,
  `src/models/Qwen4ExpExpertExecutor.cpp:1064-1117`. Tasks are a strict
  per-row partition: `job.tasks[i] = {source, r}`
  (`src/models/Qwen4ExpExpertExecutor.cpp:1100-1103`); the same
  `MoeExpertSlotHost` can appear once per row.
- Execution: the worker calls `ops::cpu::cpu_expert_batch_forward` with the
  flat task list (`src/models/Qwen4ExpExpertExecutor.cpp:488`;
  declaration `src/ops/Qwen4ExpCpuExpertOps.cpp:1575-1583`). Both the A64
  and AVX2 tile loops index one task per tile:
  `c = tile/(kIntermediateDim/gate_tile)`, `slot = tasks[c].slot`,
  `values = input + tasks[c].token*kHiddenDim`
  (`src/ops/Qwen4ExpCpuExpertOps.cpp:1643-1655`, down at :1660-1675).
  Every task reads its own expert's packed bytes; nothing groups tasks by
  `slot`.
- There is no other dedup: the single-row path
  (`StartArenaCold` :893 / `JoinResidentCold` :926) is per row by
  construction, and the rows-batch path is the same task list with row
  indices.
- In contrast the GPU side already shares weight reads across rows:
  `ExecuteVerifyRows` groups ready slots into quads/pairs/singles under
  `AILA_Q4E_VERIFY_GPU_REUSE` (`src/models/Qwen4ExpExpertExecutor.cpp:1284-1294`,
  comment "a weight read is shared across 2/4 tokens"), and
  `moe_expert_major_prefill_*` take `pairs`/`quads` counts. The CPU
  counterpart of that grouping does not exist.
- Byte accounting: packed gate+up = 2*640*2560/2 = 1.6384 MB, down =
  2560*640/2 = 0.8192 MB, total 2.4576 MB of weight bytes per task
  (kHiddenDim=2560, kIntermediateDim=640:
  `src/ops/Qwen4ExpCpuExpertOps.hpp:12-13`). Window total 587 * 2.4576 =
  1442 MB in 29.47 ms = 48.9 GB/s effective - the CPU batch is dominated
  by weight streaming, which is exactly what dedup removes.
- Dedup saving if tasks are merged by (layer, expert) and the weight pass is
  shared while each row keeps its own dot math (bitwise-safe per row;
  the Row4/nf4_q8 kernels already do multi-row dots within one expert,
  `Qwen4ExpCpuExpertOps.hpp:28-29`): saved = f * 343 * 48.7 us = f * 16.70
  ms of the window (f = fraction of row-2 cold experts also selected by
  row 1). f=0.3: -5.0 ms (16.26 tok/s); f=0.5: -8.4 (16.78); f=0.8: -13.4
  (17.63); f=1.0: -16.7 (18.25). f is not yet measured - it is the single
  unknown that decides the size of this lever.

## 6. Status of `AILA_Q4E_VERIFY_CPU_BATCH` (the arm that was in flight)

Receipts now on disk (same binary 7a8182..., same runner, only the env
differs):

| arm | rate (continue) | submitted med | cpu_compute | cpu_join | sha |
|---|---:|---:|---:|---:|---|
| g1-b1 (A, 10-15) | 15.666 | 92.26 | 29.47 | 24.84 | 2f34f37e |
| s2x-a1 (A, 10-16) | 15.558 | 93.71 | 30.27 | 25.22 | 2f34f37e |
| s2x-b1 (B, batch) | 15.116 | 95.00 | 31.37 | 31.73 | 2f34f37e |

Read: the predicted -15 ms did not materialize. B1 is +1.4% on the submitted
median and -2.8% on the continue rate vs A1 (A/A floor across the two A arms
is -0.7% here, E2's session floor 0.9-2.6%); content sha is identical. The
mechanism is visible in code: `JoinArenaColdRowsRow` gates row 0's publish
on the single `finish()` of the whole 587-task job
(`src/models/Qwen4ExpExpertExecutor.cpp:1125-1129`, publish at
:1159-1174), so the early row-0 rank-span publish that the per-row path had
is lost, and the join wait simply re-absorbs the job duration (cpu_join
24.84 -> 31.73). The batched entry did engage ([Q4E ArenaColdRowsEnabled]
rows=2 tasks=12 fired at prime in s2x-b1-worker.log). b2 is running and a2
is pending; treat "batch = neutral-to-worse" as the current measured
position, not as the predicted win.

Consequence for planning: the "batch lands -> W2 = 76-80 ms" scenario is
contradicted by B1. The arithmetic below uses the measured window
(W2 = 112.2-114.9) as the base; the hypothetical post-batch base is kept
only as an upper bound (if a future fix restores early row-0 publish).

## 7. Two candidates (neither is in charter section 2)

C1 - CPU verify-window expert merge (the CPU analogue of VERIFY_GPU_REUSE).

- Mechanism: group the window's cold tasks by (layer, expert); one packed
  weight pass serves the rows that selected that expert, each row keeping
  its own dot accumulation order.
- Expected: -f*16.70 ms; f=0.3/0.5/0.8 -> W2 107.2/103.8/98.8 ->
  16.26/16.78/17.63 tok/s. Cap 18.25 at f=1. Not enough alone for 20; it is
  the largest in-window CPU lever and it stacks.
- Change points: task build `src/models/Qwen4ExpExpertExecutor.cpp:1064-1117`;
  task consumer `src/ops/Qwen4ExpCpuExpertOps.cpp:1575+` (add a token-list
  variant of the tile loops); mirror the GPU grouping at
  `src/models/Qwen4ExpExpertExecutor.cpp:1284-1294`.
- Probe (after a read-only merged-vs-total counter proves f >= ~0.4):
  GPU mutex check, then `run_mtp_ab.py p3-merge-a1/b1/b2/a2` with
  `AILA_Q4E_VERIFY_CPU_MERGE=1` in the B arms; read submitted median and
  the continue rate; pass = submitted <= 106.7 (>=5% window) with 2/2 pairs
  over the A/A floor and identical content sha (bitwise per-row claim).
- Risks: f may be < 0.3 (candidate dies); kernel change touches the
  bitwise-guarded A64 path (H8/A64 verify batches must stay on); build cost.

C2 - Draft-side window trim (the non-Step half of the 19.9 ms).

- Mechanism: the draft residual outside the draft Step is 19.9 - 6.2
  (probe live draft_ms) ~= 13.7 ms/window: capture/checkpoint 1.3-1.5,
  Select(1)+Select(2)+D2H 2-4, accept commit 0.7-0.9, reject path 1.3-3.8
  avg (25.2% of windows), per-window ctx.synchronize 1-3. Cheap replay
  (state_only_replay exists in the probe), async/fused selects and folding
  the sync target -3 to -6 ms [est].
- Expected: W2 106.2-109.2 -> 15.9-16.4 tok/s; stacks under C1.
- Change points: `src/models/Qwen4ExpMtpRuntime.cpp:99-142` (window steps),
  probe flag `state_only_replay` in `tests/models/Qwen4ExpMtpProbe.hpp`.
- Probe: `run_mtp_ab.py p3-trim-b1` with the candidate env; read window
  split (draft_ms/select/commit) and rate; pass >= +3% window with
  acceptance unchanged.
- Risks: sub-noise gains; needs a clean A/B/B/A; MTP defaults are
  quality-gated (<= 2048 ctx, no Q8 head).

Charter section 2 cross-check (differences from disproven items): "2 rows
per subgroup" is a GPU GEMV subgroup change - C1 is CPU task-level weight
streaming with unchanged per-row arithmetic; "P4 SLM dequant reuse" is a
GPU SLM scheme - C1 reads host RAM; "pair2 e2e" was attention KV reuse -
C1 pairs expert tasks; "CPU cap forms" and "CPU_EARLY_START lazy" are not
touched by either candidate. The GPU-side precedent (VERIFY_GPU_REUSE
pairs/quads) is in production and was never in section 2.

## 8. If width 3/4 cannot reach 20 - the remaining exit

The only surviving path is width 2 with W <= 87.1 ms (row 2 marginal <=
12.2-14.7 ms). Composition from the measured window (112.2 ms):

- C1 CPU dedup: -5.0 to -13.4 (f=0.3-0.8).
- C2 draft-side trim: -3 to -6.
- Kernel bundle (E5): full-attn KV head grouping pair4 -0.5 to -1.0 (cached
  len >= 2048), hc -1.6, linear -1.8, full -1.2 (each below the 3% kernel
  bar alone).
- Host residual family inside the verify window (measured medians):
  shared_submit 5.72 + moe_front 2.30 + topk_submit 1.55 = 9.57 ms - this is
  the "other ~10 ms" in the width-2 window and it is P2's T1-shave family;
  G4 sec 3.1 states verify inherits T1 host shaves 1:1.

Stacking: C1 mid + C2 + host residual ~= -8.4 - 5 - 9.6 = -23 -> W 89.2 ->
19.5 tok/s. Reaching 87.1 needs essentially every item at its optimistic
end. Width 3 remains dead unless a merged window also collapses the join
wait (not demonstrated by B1) - under that scenario width 3 would need
f >= ~0.52, which is a different bet on the same unknown f.

The single highest-leverage unknown is f (row-to-row expert overlap in the
verify window): it sets C1's size, and it is measurable read-only (the cold
ids and row indices already live in `cold_ids_` / `job.tasks`, and
`cache_->TraceColdCpu` / the cpu_observer path already receive
(layer, expert_id, token_idx); a counter can be added behind an existing
gated log with no kernel change).

## 9. Sources

- `tmp/q4e-resident-speed-20261009/g1-b1-worker.log`, `g1-b2-worker.log`
  (count=2 medians, n=151; count=1 medians; windows=151 accepted=113),
  `g1-b1/result.json`, `g1-b2/result.json` (15.543/15.592).
- `tmp/q4e-resident-speed-20261009/s2x-a1{,/result.json}` and
  `s2x-b1/...` (A/B for VERIFY_CPU_BATCH; B1 fired
  [Q4E ArenaColdRowsEnabled] rows=2 tasks=12; rates 15.558 vs 15.116;
  identical content sha 2f34f37e...).
- `tmp/q4e-sprint-r1-20261014/mtp-top1/top1-gate.json` (C section: p1/q2/q3
  for 512 and 1833), `long1833.utf8.log` / `short512.utf8.log`
  ([MTP PASS], [MTP CYCLES], [MTP VERIFY], [MTP LIVE] draft_ms 6.22,
  verify_ms 111.39, cycle_ms 117.86).
- Docs: `qwen38-r1-mtp-step-budget-2026-10-15.md` (G4; width budget 1+3p
  note, draft residual 19.9, marginal row 37, reject path, 10-06 anchors),
  `qwen38-r1-mtp-top1-2026-10-15.md` (E1; C/A/B metric definitions),
  `qwen38-r1-decode-misc-2026-10-15.md` / `qwen38-r1-decode-kernel-roofline-2026-10-15.md`
  (E5; device busy 57%, kernel bundle bound), `qwen38-r1-decode-20-2026-10-15.md`
  (E2; A/A floor, break-even formula), `qwen38-performance-research-charter-2026-10-14.md`
  (section 2 disproven list), `qwen38-server-mtp-2026-10-07.json`
  (production env).
- Code: `src/models/Qwen4ExpExpertExecutor.cpp:488,893-960,1064-1117,1121-1174,1211-1330,1410-1490`;
  `src/models/Qwen4ExpBnb4Backend.cpp:1236-1345,2591`;
  `src/ops/Qwen4ExpCpuExpertOps.cpp:905-1000,1575-1675`;
  `src/ops/Qwen4ExpCpuExpertOps.hpp:12-15,28-29`;
  `src/models/Qwen4ExpMtpRuntime.cpp:99-142`.
