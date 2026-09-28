# Qwen3.8-Flash-Next CPU/Fill Overlap

Date: 2026-09-28. Working tree over f78732d, A770 driver 32.0.101.8860,
Windows, oneAPI 2026.1, Xeon E5-2696 v3. Model:
`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`.

## Implementation

`AILA_Q4E_CPU_OVERLAP=1` enables one persistent worker and one bounded job
slot per executor. The coordinator sends an activation D2H event, up to four
immutable host expert views and token indices, and preallocated output/h
buffers. The worker waits for D2H, then computes while the coordinator runs
demand acquire and GPU submission. There are no per-layer threads/futures or
USM allocations. `AILA_Q4E_CPU_THREADS` limits its TBB arena to 1..4 threads,
default 2. This is a local arena budget, not a global CPU thread guarantee.

The coordinator alone mutates cache leases, plans and counters. Host expert
leases remain held until the worker joins. Results are copied to the existing
rank arena and merged in original rank order. Worker exceptions propagate
at join; exceptional coordinator exit drains the worker and GPU before
releasing leases. Teardown stops/joins the worker before freeing USM.
No work survives `execute_hetero_decode` on its successful CPU side.

Overlap defers acquire even when `AILA_Q4E_READY_FIRST` is OFF, so CPU can
start before acquire. Mixed ready/pending GPU subsets reuse the existing
ready-first implementation in those CPU-dispatched layers; it does not
activate GPU-only ready-first on every layer. Next-layer prefetch remains
after current acquire. GPU-only all-hit work keeps its existing path.

New counters are coordinator-owned: asynchronous layer count, join wait and
intersection of host CPU-compute and acquire intervals. These are not GPU idle
or physical I/O time. Worker compute includes TBB scheduling inside the arena;
join wait includes the completion synchronization overhead. Cumulative logs
include initialization warmup and must not be treated as request-only timing.

Explicit environment CPU credits of zero now remain disabled (initialization
previously clamped them to one despite the setter allowing zero). No default
CPU dispatch, GPU-order, overlap or ready-first promotion.

## Failure Found During Screening

The initial async version sampled GPU completion after joining, using the
old synchronous straggler heuristic. This failed to throttle a blocking CPU
join: CPU tasks rose to 1717, compute 4498.01 ms, join wait 3547.20 ms,
acquire overlap 818.12 ms. Story time was 9002.39 ms and token 26 differed
(23221 expected, 3808 actual). Preserve `tmp/q4e_prof_overlap_async_1.*`.
Its real multi-turn acceptance had passed, showing that those tests alone
do not certify all greedy trajectories. High-dispatch numerical exactness
is NOT resolved. The older GPU-order sixteen-sample alignment is not a
universal CPU/GPU equivalence proof.

The async penalty now observes whether the CPU job was incomplete at merge
admission, before joining. Two such waits trigger the existing cooldown;
this is conservative throttling, not an optimal completion-time predictor.
With one credit and two threads, corrected screening passed all golden IDs:
132 tasks, compute 355.77 ms, join 271.17 ms, overlap 69.10 ms, request
5957.21 ms (`tmp/q4e_prof_overlap_throttled_1.*`).

Four-thread screening reduced join wait to 137.45 ms with 104.53 ms overlap
and 171 tasks. Same-binary request times were GPU-only 5663.58 ms, synchronous
CPU 5952.37 ms, async CPU four threads 5874.22 ms, all golden IDs exact.
Logs: `tmp/q4e_prof_overlap_{final_base_1,final_sync_1,threads4_1}.*`.
One sample each: the apparent ~1.3% gain over synchronous CPU is screening,
not a validated overall acceleration.

## Final Paired Comparison

Final binary SHA256:
`6A7617978D601C4B69C0154533A3E46C62EB05B3E53DB2339A917F8239DBBF28`.
Canonical story, 30 prompt tokens, 26+4 reusable-prefix, max sequence 512,
greedy, thinking budget 0, 48 output tokens. ReadFile reader, decode profiling
and token trace enabled on both sides; stage-sync profiling OFF. Candidate:
CPU dispatch + GPU-order + overlap ON, one CPU credit, four TBB arena threads.
All other optimization toggles remain at defaults.

Fresh process and normal warmup per run, warm OS pages without forced cache
clearing. One excluded warm run, then base1/async1, async2/base2, base3/async3.
One GPU workload at a time. GenerateMessages excludes load and warmup, but
is a first-request measurement, not repeated in-process steady state.

| Pair | GPU-only ms | Async CPU ms |
|---|---:|---:|
| 1 | 5682.80 | 5865.32 |
| 2 | 5669.44 | 5932.72 |
| 3 | 5685.58 | 5856.99 |

Medians 5682.80 -> 5865.32 ms (+3.21%). MAD 2.78 vs 8.33 ms. All three
pairs regress; do not enable this candidate by default. All seven traces,
including the excluded warm run, pass full prompt and 48-token comparisons
against `tmp/h6-baseline.log`. These are three request pairs, not a twenty-
sample operator benchmark.

Final async runs report 172/168/164 CPU tasks, 262.58/260.24/248.49 ms
CPU compute, 140.71/140.95/140.75 ms join, and 103.50/96.80/89.56 ms
acquire overlap (cumulative including warmup). CPU work overlaps, but its
remaining wait and scheduling cost exceed the benefit on this workload.
Logs, environment/binary metadata and token comparisons:
`tmp/q4e_prof_overlap_pairs_{warm,base1,async1,async2,base2,base3,async3}.*`.

## Verification and Remaining Work

- Full build passed: `tmp/q4e-cpu-overlap-final-build.log`.
- All eight Q4E CTests passed serially in 62.01 s:
  `tmp/q4e-cpu-overlap-final-tests.log`.
- Final four-thread, one-credit overlap configuration passed explicit real-
  model 248320 BF16 skip-logits/rollback checks, two-turn reuse, edited-
  history rollback and snapshot-policy transitions:
  `tmp/q4e-cpu-overlap-final-real.log`. Known H6c anchor-2/16 diagnostics
  remain unresolved.
- Tests hold a demand fill callback until the CPU job finishes, proving
  progress before acquire returns. Repeated output buffers and synchronous
  versus asynchronous output are bit-exact on the fixture.
- Coordinator callback failure and over-capacity acquire after CPU launch
  exercise join/drain and host-reader release. This is not exhaustive
  cancellation, worker-exception injection or device-loss coverage.
- Original fill-pool exception handling and ReadFile short reads remain
  separate limitations. This worker does not repair those contracts.
- `git diff --check` passed. No commit or default promotion in this delivery.

Historical decision at this checkpoint: retain the path OFF by default and
optimize CPU arithmetic before increasing admission. The user subsequently
requested a twelve-thread, large-expert-batch campaign. That supersedes the
one-credit/four-thread scope above; it does not invalidate these measurements.
See [CPU scaling follow-up](qwen3.8-flash-next-cpu-scale12-2026-09-28.md) for
40-task capacity, flat batching, prepared CPU inputs and full-size experiments.
