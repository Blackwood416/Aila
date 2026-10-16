# R1 structural phase: OOO answer, submit-cost anatomy, XMX/stage-copy adopted (2026-10-16 night)

Owner directives for this phase: structural changes over tuning; the audit's
out-of-order-queue question and submit/launch overhead are the priority;
Routine rounds = single A/B (A/B/B/A only for default-on gates); XMX admission
gate relaxed where the deviation is provably rounding-order bound.

## Adopted (default-on, committed)

`6f9a41e` flips four defaults (escape hatch `=0` on each; rationale doc
`qwen38-r1-xmx-stagecopy-adoption-2026-10-16.md`):

| flag | effect on the primed 14K TTFT |
|---|---|
| `AILA_Q4E_PREFILL_EXPERT_XMX=1` (+MIN_M 256, ROWS 4096) | 69.6-71.2 s -> 63.99 s |
| `AILA_Q4E_STAGE_COPY_PARALLEL=1` | 63.99 s -> 53.2-54.1 s; steady blocks 17.0 -> 14.6 s; fill 7.4 -> 2.3 s |

Confirmed by two independent B arms (`xpb-b1` 54.06 s, `xpb-b2` 53.20 s);
`xpb-b2` ran the h4/bfloat16-KV bundle, so its decode rate is not comparable,
but the fill/bank/block timings reproduced exactly. Quality: owner waiver
recorded (KL 0.019 class, rounding-order bound; ALU arms stay content-identical,
XMX arms drift run-to-run at 32-token greedy).

## Answered / falsified

1. **Out-of-order queue: no.** Probe (`8f4950e`, `qwen38-r1-ooo-queue-probe`):
   OOO chain + depends_on is 7% slower and costs +27% host submit
   (6.45 -> 8.17 us); two truly independent chains do not overlap more than
   in-order (0.43 vs 0.42); removing the per-layer wait is what collapses the
   synthetic (-40%, 14.4 -> 8.6 ms); the event-status-query release behaves
   identically on both queue types. Hooke's anatomy doc: 203 `.submit(` sites,
   only 33 `depends_on`; decode relies 100% on implicit in-order; OOO blast
   radius 31 TUs + shared scratch + a test assertion. No OOO conversion.
2. **The "5x submit inflation" is really 2.7x** (Hooke): 18.5 us/submit vs
   6.5-6.8 us bare; the 5x was a denominator artifact (291 HC submits over 97
   calls). Absolute gap ~12.8 ms/token.
3. **`Context::queue()` is not the cost** (1 acquire load + 1 null check).
   `Bnb4BitLinear.cpp:2709`'s in-order check is the Q8 LM-head contract; a
   violating queue would silently disable LM_HEAD_Q8.
4. **chunk-8192 stays parked**: `6498159` falsified the min-cap hypothesis -
   the 55.5 s first-block cost is in the topk submit path, unchanged by
   `AILA_Q4E_RUNTIME_MIN_CAP` (133.5 vs 132.9 s TTFT).

## Submit-cost anatomy (new receipts)

| evidence | value |
|---|---|
| production, `hct-a1` (`AILA_Q4E_HC_SUBMIT_TIMING=1`) | HC submits 20.8 us (norm) / 26.0 us (norm-fb) per call; HC host 7.5 ms/token |
| E0 probe, direct production kernel, AilaLib-linked, drain-per-call | 23.8-27.5 us |
| E0 probe, param-table ABI + SYCL submit | 20.3-20.9 us |
| E0 probe, param-table ABI, burst (64 back-to-back) | 5.2-5.4 us |
| E0 probe, single-kernel graph replay | 32.8-49.2 us (no win alone) |
| shape probe (`probe_submit_shape`, wait-every-32, local kernels) | real down kernel 3.7 us; +SLM 5.0; plain 10.7; empty single_task 2.9 |
| shape probe under 18-thread DRAM contention | medians 6.7-15.2 us, p90 33-45 us |

**E0 verdict: the param-table ABI is bit-exact on the real `hc_decode_down_inject`
kernel** (3 shapes x 1000 replays, echo-pointer proof the table is live,
`all_bitexact=1 failures=0`). The graph/parameter-table route is viable for
correctness; its economics depend on the open regime question below.

**Open contradiction (next discriminator):** the same true kernel submits at
3.7 us (local binary, burst) but 23.8-27.5 us (AilaLib binary, drained) and
21-27 us in production. Hypothesis list: (a) kernel-program-cache size effect
in the big `per_kernel`-split binary, (b) drained-vs-batched launch regime,
(c) L0 batch flush behavior at production cadence (8 submits per wait).
The decisive probe: same kernel, same binary, sweep `wait_every` {1,2,4,8,16,32}
on both binaries.

### Sweep result (same night): the regime is the lever, not the binary

`probe_submit_shape.exe --n 128 --wait-every W`, idle, median us per submit
(p10/median/p90; receipts `tmp/q4e-structural-20261016/shape-w{1,2,4,8,16,32}.log`):

| arm | W=1 | W=2 | W=4 | W=8 | W=16 | W=32 |
|---|---:|---:|---:|---:|---:|---:|
| `R_down_real` (production kernel, verbatim) | 19.6/22.2/31.6 | 8.4/20.2/21.6 | 7.5/14.1/20.6 | 3.9/**7.0**/20.9 | 3.7/4.0/15.2 | 3.6/3.7/11.4 |
| `C_down_slm` (same + SLM) | 22.7/24.1/24.9 | -/- | 10.4/17.1/25.3 | 5.1/9.0/23.4 | 5.4/6.0/18.0 | 4.7/4.8/13.7 |
| `A_plain` (tiny scalar kernel) | 30.1/30.3/30.8 | 15.8/27.9/28.6 | 16.2/21.7/29.4 | 10.1/16.2/28.6 | 10.3/11.3/22.4 | 10.1/11.6/30.6 |
| `S_sync` (empty single_task) | 15.3/16.7/17.3 | 6.4/16.3/17.6 | 6.1/11.4/18.8 | 2.9/5.9/18.3 | 2.8/3.2/13.8 | 2.8/2.8/13.3 |

Reading: the real production kernel costs 22.2 us when each submit is followed
by a wait (W=1) but 7.0 us at 8-deep batching and 3.7 us at 32-deep - a 3-6x
regime swing, identical kernel, identical binary. Production measures 21-27 us
per HC submit, i.e. the **W=1 value**, even though a decode layer issues
8-16 submissions between its two host waits. The likeliest mechanism is that
the UR/L0 dynamic batch closes when the host leaves a gap between submissions
(routing/classification CPU work), so production effectively pays the drained
round trip per submit. Under 18-thread DRAM contention even batched submits
collapse (medians 318-382 us), but that arm oversaturates the bus more than the
real worker does.

Actionable forks this opens (queued below):
1. Restructure HC submits into tight back-to-back bursts per layer (if the
   batch stays open, per-submit drops toward 7 us -> ~2-3 ms/token at 288
   submits).
2. Try explicit UR batch knobs (e.g. `UR_L0_BATCH_SIZE`) as a zero-build A/B.
3. Measure the production inter-submit gap directly (extend the HC timing with
   a "time since previous submit" field) to confirm the gap-to-regime link.

## Noise floors (use for gates)

| metric | A/A spread |
|---|---|
| 14K TTFT (primed, single arm) | ~0.9% (53.2/54.1/53.5/54.0) |
| decode, 19-token continue, agent6k+int8, production timing | 1.05% (16.372/16.202) |
| decode, 19-token continue, h4+bf16-KV | 8% (15.02/16.24) - do not gate on this |
| decode, n=128, agent6k+int8 | 17.15 tok/s (baseline 17.38-17.50, unchanged by adoption) |

## Goal status

- Prefill 14K TTFT now 53.1-54.1 s (~258 tok/s; from 69.6-71.2 s ALU).
  400 tok/s needs ~35 s; remaining slices per steady block: attention +
  router wait ~7.5 s, experts (fill 2.3 + H2D/readers ~2.1 + compute) ~6.7 s.
- Decode 17.15 tok/s (n=128, canonical bundle) vs 20 tok/s target (-13%).
  Levers ranked: submission-cost regime (above), per-layer attention-subchain
  graph (multi-kernel graph + param table), fewer sync points.

## Next experiment queue (ranked)

1. ~~Submit-regime discriminator~~ DONE (sweep above): regime, not binary size.
2. Zero-build A/B on UR batch knobs (`UR_L0_BATCH_SIZE`) on the canonical
   decode arm, then (if flat) a code change that bursts the per-layer HC
   submits back-to-back; gate on the existing 1.05% A/A floor.
3. Multi-kernel attention-subchain graph using the proven param-table ABI
   (graph replay only pays with many kernels per graph; single-kernel replay
   measured 32.8-49.2 us).

## Late-night follow-up: batch knob, production-regime HC timing, knob screen

### Static batching is confirmed dead

`UR_L0_BATCH_SIZE=16` (receipt `ub16`): decode 6.423 tok/s vs the 16.2-16.4
baseline (-61%). Static batches hold kernels until the batch fills or a wait
fires, exactly the audit's batch-hold failure. Keep dynamic (0). A second
memory effect was found while screening: the previous arm's ~52 GiB
large-page arena is not always returned immediately, and starting the next
worker too early fails model load (`MEM_LARGE_PAGES`, host_avail=190 MiB).
All remaining scripts now gate on >=45 GiB available physical memory.

### Production-regime HC submit cost (hct-a1, diag arm with the instrument)

Per-token window, exact semantics: `hc_submit_<site>_us` is the window total,
`_n` the call count, `_min_us`/`_max_us` per call.

| site | calls/token | mean us | min us | max us |
|---|---:|---:|---:|---:|
| norm | 47 | 20.3 | 16.2 | 30.9 |
| norm-fallback | 48 | 24.8 | 13.6 | 37.6 |

Every call pays >=13.6 us - production sits in the drained (W=1-like) regime
for *all* HC submits, not just first-in-batch ones. The AdaptiveWork log line
truncates at 1025 chars, so the down/up fields (down total alone was 2856.6 us
in one token) are lost; instrumentation fix in flight.

Note on method: `--production-timing` deliberately strips diagnostic env
(`WORK_SUMMARY`, `AILA_SERVER_WORKER_LOG`) before launching the server, so the
per-site log is only available in diag arms; `hct-p1` (production timing +
the instrument) ran fine and measured 16.694 tok/s, in line with the
production baseline.

### Zero-build knob screen (canonical decode arm, single A/B per owner rule)

| arm | knob | decode tok/s |
|---|---|---:|
| dec-a1 / dec-a2 | - (A/A) | 16.372 / 16.202 |
| hct-p1 | HC timing instrument only | 16.694 |
| k1-inorder | `UR_L0_USE_DRIVER_INORDER_LISTS=1` | 16.821 |
| k2-counter-ev | `UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=1` | 16.125 |
| k3-reuse-ev | `UR_L0_REUSE_DISCARDED_EVENTS=1` (+SYCL_PI) | 17.069 |
| k4-single-thr | `UR_L0_SINGLE_THREAD_MODE=1` | worker crash - NO-GO |
| ub16 | `UR_L0_BATCH_SIZE=16` | 6.423 - NO-GO |

Session drift is ~2% (the instrument-only arm moved +2.5% vs the A pair), so
k1/k3 need paired confirmation against interleaved controls before any claim.
Pending: k3 mechanism check (diag + HC timing: does the per-call cost drop
from ~20 us toward the 5-7 us batched regime?), then A/k3/A/k3 pairing.

### Pairing progress and two operational hazards

- The k3 mechanism check ran (`gap2-k3`, diag + HC timing + reuse-discarded-
  events): norm 46 calls, mean 18.4 us (min 15.0) vs baseline 47/20.3/16.2;
  norm-fallback 48 calls, mean 26.2 (min 20.7) vs 24.8/13.6. Mixed, within
  noise - no mechanism-level confirmation. The k3 e2e +4.7% is more likely
  session drift than a real effect; k3 stays unadopted, pairing would need a
  longer interleaved run.
- Gap buckets still do not reach the log: the AdaptiveWork line truncates at
  1025 chars right after `hc_submit_down_us=2550.100`, so `hc_gap_b0..b3` and
  `hc_gap_max_us` are lost. Fix in flight (print the hc_* block as its own
  line or move it to the front).
- Worker/server identity: rebuilding only the worker makes every arm fail in
  4 s with `Worker handshake identity/path isolation mismatch`
  (receipts `gap-a1`, `gap-k3`). The server must be rebuilt in the same
  window (`tools/server/build.ps1 -SkipDependencies`).
- Host-memory race: back-to-back arms can fail model load with
  `MEM_LARGE_PAGES ... host_avail=107-190 MiB` (receipts `ub8`, `gap2-a1`)
  because the previous arm's ~52 GiB large-page arena is not yet back. Always
  gate on >=45 GiB available physical memory plus the 15 s pool drain before
  launching the next arm. `gap2-k3` succeeded when started ~15 s later.

### The batch breaker, found: memcpy/wait/status-query between submits

Gap buckets (gap4-a1) had already falsified the "batch closes on host idle"
story: 190 of 286 HC submits sit <10 us from the previous HC submit yet still
pay 23.1 us; bucket cost is flat (~22-26 us) across all gap sizes. A dedicated
probe (`tmp/q4e-structural-20261016/probe_chain.cpp`, receipt
`chain-probe3.log`) then isolated what actually breaks the batch:

| per-iteration sequence (k=32 back-to-back submits, no waits) | us per submit |
|---|---:|
| plain, no dependencies | 5.5 |
| explicit `depends_on(previous)` chain | 5.8 |
| submit; host wait; submit; ... | 19.7 |
| submit; event status query; submit; ... | 17.8 |
| submit; 4 B H2D memcpy (stack source); ... | 78.3 |
| submit; 4 B H2D memcpy (USM host source); ... | 22.2 |

Conclusions:

1. `depends_on` on an in-order queue is free (+0.3 us) - not the issue.
2. **Any `queue.memcpy` on the compute queue is a synchronization point**:
   even a 4-byte USM-host->device copy costs ~22 us of host time and closes
   the L0 batch (a non-USM source degrades to a blocking copy at ~78 us).
   Event status queries (17.8 us) and host waits (19.7 us) do the same.
3. Production's per-layer pattern contains roughly 4-6 such flush events
   (2 host waits + 1 status query in the executor + 1-2 compute-queue
   memcpys: the 40 B top-k weight H2D at
   `Qwen4ExpBnb4Backend.cpp:1583/:1654` and the 5 KB top-k D2H readback),
   i.e. ~200-290 flushes/token - the same order as the 286 HC submits/token,
   which is why *every* HC submit measures in the drained 20-28 us regime.
   The audit itself had flagged these two memcpys as "not on copy_q_".

### Next change (scoped, ready to implement)

Route both per-layer copies off the compute queue:

1. The 40 B top-k weight H2D (`RunResidentDecode`) and the 5 KB top-k D2H
   readback -> `copy_q_` (already exists: in-order + copy engine, used by the
   prefill staging path), chained with explicit events; the host wait for the
   readback stays (it is the router dependency), but the compute queue's
   batch no longer breaks.
2. Gate the executor's per-layer status query (`Qwen4ExpExpertExecutor.cpp:2162`)
   behind an env flag and measure.
3. Expected effect: compute-queue submits drop from ~20-28 us toward the
   5-8 us batched regime; if the whole submit ledger (~18.7 ms/token) follows,
   decode moves from 16.2 toward 20+ tok/s.
