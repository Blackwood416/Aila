# Qwen3.8-Flash-Next CPU Dataflow Follow-Up

Date: 2026-09-28. Working tree over f78732d. Windows, oneAPI 2026.1,
A770 driver 32.0.101.8860, Xeon E5-2696 v3 (18 physical/logical cores).
Model: C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2.
Continues the [twelve-thread scaling campaign](qwen3.8-flash-next-cpu-scale12-2026-09-28.md).

## Scope And Implementation

Two independent OFF-by-default flags, with the prepared-input CPU kernel,
twelve compute arena threads, forty task credits and four fill threads.

### Batched Host Preparation

AILA_Q4E_CPU_BATCH_PREPARE=1 applies to forced CPU placement (placement=2).
The coordinator first leases existing ready inputs, then reserves distinct
cold experts, preserving CPU readers and DMA fences. It submits one complete
expert per task to the existing demand fill pool. Duplicates share a fill but
retain independent leases. Exhausted host capacity yields empty output leases
and GPU fallback, never replacement of another task's active CPU input.

The pool callbacks catch read exceptions, finish all admitted callbacks and
return errors to the coordinator. Failed batches invalidate newly prepared
slots, release leases and remain retryable. Existing ready slots remain
valid. This also adds failure cleanup to single-expert prepare_host through
the shared reservation helper. It does NOT repair exceptions in unrelated
GPU-acquire/prefetch callbacks or the reader's existing silent short-read
and cancellation behavior.

This is parallel preparation, NOT asynchronous preparation overlapped with
the same batch's CPU arithmetic: the CPU worker starts after preparation.
Admission snapshots can differ from the serial policy when speculative
prefetch completes during planning. Validate actual routes/tokens, not just
the configured credit limit. The host_prepare timer includes batched ready
lease handling; the serial timer covers individual cold preparations, so
end-to-end request time is the primary comparison.

### Coalesced Result Copies

AILA_Q4E_CPU_COALESCE_RESULTS=1 joins consecutive CPU rank destinations into
one H2D copy. Forty all-CPU tasks can use one copy instead of forty. Gaps
containing GPU ranks start another run. Source order and destination order
are unchanged, no CPU sums are combined, and the existing GPU FP32 ordered
rank merge remains intact. Bytes are unchanged; fewer API submissions are
the hypothesis. No additional USM buffers or hot-path USM allocations.

New counters: prepare_batches, prepared_experts, result_copies, result_bytes,
result_submit. The prepared_experts count is batch-only; zero with batching
OFF does not mean the serial path loaded no experts. CPU compute statistics
are now visible for all-CPU execution even when no ready-first GPU split ran.

## Profiling Before Changes

VTune 2026.2 software Hotspots completed before the rewrite, on runtime hash
EBCC69AD31A4987BE67AC7D7F3EA56C328E282A9052066E745E64BEF8DD1BA98.
Artifacts: tmp/q4e_vtune_dataflow_before and its .log/.hotspots.csv files.
Collection includes model initialization, JIT and generation. Release has
no usable application PDB, so application functions appear as addresses.
The report contains 0.479 s in zeCommandListAppendMemoryCopy, but that is
whole-process sampled CPU time, not request-only copy overhead or a bound
on recoverable latency. Loader/JIT and waits dominate many other entries.
No unsupported source-line, DRAM-utilization or GPU-idle attribution is made.

Use perf/qwen4exp/q4e_cpu_dataflow_profile.ps1 with a run metadata file to
reproduce settings. Profiled timings are excluded from performance pairs.

## Verification And Preparation Microbenchmark

Runtime SHA256:
D883F2B32C31674A21D739E15FE0E6D24D0C7DDD7B9CB7AC944C816E7FEDE4C1.
Build passed: tmp/q4e-dataflow-build.log. All eight Q4E CTests passed serially
in 64.12 s: tmp/q4e-dataflow-tests.log. Independent twelve-thread GPU-order
and prepared-input test passed: tmp/q4e-dataflow-large-ordered.log.

Coverage includes 5/10/40-task mixed/all-CPU plans, each flag separately and
together, unchanged result bytes, exact copy-run counts, sparse GPU-rank
holes, repeated outputs, duplicate experts, pinned host capacity fallback,
reader exception drain, invalidation, retry, and oversized input rejection.
CPU arithmetic and ordered-merge outputs are bit-exact on these fixtures.

Preparation benchmark: AilaQwen4ExpDmaTimingTests --host-batch-bench.
Synthetic NF4 H=2560/I=640/block=64, compact codes, RAM source buffers (not
SSD/ReadFile), two disjoint sets of experts, host capacity equal to one set.
Every call alternates sets to force host-cache misses. Source allocation and
byte verification are outside timing; preparation and returned leases are
inside. Six warm calls, then three repetitions of twenty interleaved pairs.
Four fill threads include the caller. Logs include p10/p90 and CV, flagging
CV > 0.1. All repetitions here are below the threshold.

| Tasks | Serial medians ms (3 reps) | Batch medians ms (3 reps) | Median comparison |
|---|---|---|---|
| 10 | 3.6088 / 3.7756 / 3.7389 | 1.2799 / 1.3040 / 1.3734 | 3.7389 -> 1.3040 |
| 40 | 16.5456 / 15.9949 / 15.6979 | 5.9424 / 5.5815 / 5.5809 | 15.9949 -> 5.5815 |

Both are about 2.87x faster for this preparation microbenchmark, not a claim
about SSD bandwidth or complete inference. See tmp/q4e-dataflow-host-bench.log.

## Model Measurement Protocol

Fresh process per case, normal warmup, one excluded GPU warm run (5699.80 ms),
warm OS pages without forced clearing, one pressure process at a time.
Canonical thirty-token story, 26+4 reusable-prefix semantics, 48 greedy
tokens, thinking budget zero, max sequence 512, ReadFile, profiling and token
trace enabled. Request times exclude load/warmup; counters include warmup
unless explicitly differenced. CPU/credit/small-M policy matches the previous
large-placement baseline. A GPU-only eight-fill-thread reference is included.

Screen: data_base, data_fill, data_copy, data_both; additionally miss-only
prepared CPU with and without coalescing. Each case's metadata and golden
prompt/token integrity comparison are retained.

## Initial Screens And Negative Results

The following measurements were completed in the original chat and recovered
from the saved logs/token-comparison JSONs when work resumed. Times are
GenerateMessages milliseconds, excluding process initialization.

| Configuration | First screen ms | Golden 48 token IDs |
|---|---:|---|
| GPU-only, eight fill threads | 5686.59 | exact |
| CPU data_base | 9192.32 | exact |
| Batch preparation only | 8871.50 | FAIL |
| Coalesced copies only | 8871.72 | exact |
| Both | 8516.93 | exact |
| Miss-only prepared CPU | 6715.25 | FAIL |
| Miss-only plus coalesced copies | 6701.13 | FAIL |

Repetition invalidated promotion of the apparently successful variants:
data_both at 8622.59 ms failed against a passing 9194.89 ms baseline.
Copy-only passed its second pair (9005.69 vs 9360.23 ms), but failed its
third pair (9097.17 vs passing 9203.72 ms). The failed copy-only run first
differed at zero-based token index 31 (836 vs 44265). These timings are
feasibility observations, not accepted speedups. Changes in submission timing
can change CPU/GPU ownership while speculative prefetch is completing;
synthetic bit-exact arithmetic tests do not establish equivalence of those
different real-model placements.

Logs: tmp/q4e_prof_dataflow_screen1_*, dataflow_pair2_*,
dataflow_copy_pair{2,3}_* (same q4e_prof_ prefix).

With prefetch disabled for all cases, one diagnostic screen produced:

| Configuration | Request ms | Golden IDs |
|---|---:|---|
| GPU-only | 5900.82 | exact |
| CPU data_base | 10530.25 | exact |
| Batch preparation only | 9627.75 | exact |
| Coalesced copies only | 10220.61 | exact |
| Both | 9427.08 | exact |

All four CPU configurations dispatched 26400 tasks, including warmup.
This supports investigating ownership timing, but is only one screen and
does not establish a general numerical fix. Disabling prefetch also loses
performance. Logs: tmp/q4e_prof_dataflow_nopf1_*.

## Waiting For Selected Prefetch Inputs

AILA_Q4E_CPU_WAIT_PREFETCH=1 is an additional OFF-by-default experiment,
effective with CPU_BATCH_PREPARE=1 and forced placement=2. Selected ranks
join their existing prefetch fills before taking host leases, instead of
being skipped merely because those fills are still running. The coordinator
can help finish unclaimed parts through the existing prefetch mechanism.
It also waits for host-slot Ready publication before attempting the lease.
The selection remains bounded by credits and available host capacity;
exhausted capacity still falls back to GPU. This is not a universal promise
that every selected rank always executes on CPU.

The runner exposes data_wait and -DataflowWaitScreen. Tests cover joining
existing fills without refilling, valid bytes, and retaining leased inputs
when the old prefetch batch is cleaned up. The initial build caught a test
calling private cancel_prefetch; the test now triggers cleanup through the
public prefetch_async API. The corrected build completed successfully in
tmp/q4e-dataflow-wait-build2.log. On resumption, all eight focused Q4E CTests
passed serially in 63.84 seconds (tmp/q4e-dataflow-resume-tests.log).

## Resumed Real-Model Pairs

Runtime SHA256 for all resumed pairs and the acceptance configuration:
5CA0BAE474C8249C59A94EA84A9FD3C2AC1139F47BB86FFF0F14EBC352A8AB0A.
No runtime changes were made between these pairs. Reader=ReadFile, CPU
threads=12, fill threads=4, fixed credits=40, GPU-order prepared-input CPU
arithmetic, small-M enabled, forced placement=2, normal prefetch enabled.
data_base disables all three dataflow flags; data_wait enables all three.

| Pair / execution order | data_base ms | data_wait ms | Both golden traces |
|---|---:|---:|---|
| 1 / base then wait | 9363.65 | 8733.11 | exact |
| 2 / wait then base | 9191.85 | 8748.65 | exact |
| 3 / base then wait | 9258.08 | 8547.99 | exact |
| Median | 9258.08 | 8733.11 | 6/6 exact |

[MEASURED] The candidate's median request time is 5.67% lower than this
previous CPU configuration on the canonical story. Three pairs support a
limited configuration-level result, not a general shape dispatch rule or
CPU/GPU numerical-equivalence guarantee. The same-build GPU-only reference
was 5695.54 ms with exact tokens, so forced CPU placement remains slower.

The old CPU configuration dispatched 26298 / 26303 / 26304 tasks;
data_wait dispatched exactly 26400 in all three runs. Each candidate performed
2194 cold-expert preparations through batching and submitted 2448
result copies (270336000 bytes), versus one copy per task on data_base.
These counters include warmup. Different task counts are disclosed: this
compares complete policies, not an isolated identical-route copy benchmark.

For pair 1, subtracting the counters immediately before the request from
the final counters gives 24960 CPU tasks, 2352 result copies, 255590400
result bytes, 3134.02 ms CPU compute, 2788.55 ms join wait, 1022.56 ms host
preparation and 107.46 ms copy submission. These intervals can overlap and
must not be added as independent recoverable costs. The CPU/acquire overlap
counter is zero for this all-CPU small-M workload; the worker implementation
alone does not establish effective overlap with GPU expert acquisition.

Logs: tmp/q4e_prof_dataflow_resume_wait{1,2,3}_{data_base,data_wait}.*;
GPU reference: tmp/q4e_prof_dataflow_resume_wait1_gpu8.*.
Compact extracted results: tmp/q4e-dataflow-resume-summary.json.

## Stateful Acceptance And Decision

The candidate's explicit real-model suite passed with the exact environment
from pair 3 (tmp/q4e-dataflow-resume-wait-real.log):

- All 248320 BF16 logits were identical for the 26+4 skip-logits and rollback
  comparisons.
- Two-turn prefix reuse matched fresh replay token IDs.
- Edited history performed an actual recurrent rollback and matched fresh
  replay token IDs; snapshot-policy transitions also passed.
- H6c story anchors 26/28 had zero split-reference mismatches. Existing
  anchor-2/16 differences remained (231920/235541 mismatches, maximum absolute
  differences 0.375/0.304688). The suite reports those as diagnostics; this
  work does not fix or establish equivalence for them.

Keep batch preparation, coalesced results and wait-prefetch OFF by default.
The wait-enabled forty-task experiment now has three matching story traces
and stateful acceptance, while the faster-looking dynamic mixed variants
remain numerically unaccepted. The large CPU kernel and preparation wins
are useful experimental results, but unconditional CPU takeover still loses
to GPU-only end to end. This delivery does not commit the accumulated earlier
campaign changes or promote an experimental configuration.

The next meaningful optimization needs less exposed CPU compute/preparation
or a validated selective ownership policy. Raising the task cap again is
not supported by these measurements: the current all-CPU small-M path already
uses every eligible rank and shows no expert-acquire overlap. General mixed
CPU/GPU numerical alignment remains a separate prerequisite for changing
ownership opportunistically.

Reproduction (use unique tags, one pressure process at a time):

    pwsh -NoProfile -File perf/qwen4exp/q4e_cpu_scale.ps1 -Tag local_wait -DataflowWaitScreen
    pwsh -NoProfile -File perf/qwen4exp/q4e_cpu_scale.ps1 -Tag local_wait_pair -DataflowPair -DataflowCandidate data_wait -ReverseOrder
    pwsh -NoProfile -File perf/qwen4exp/q4e_cpu_scale.ps1 -Tag local_wait_accept -Cases data_wait -RealAcceptance

The runner retains failing token comparisons as feasibility data; inspect
the .tokens.json equal/integrity fields instead of treating its process exit
alone as numerical acceptance. RealAcceptance additionally propagates the
explicit stateful suite's failure.

## Post-Change Profile And Final Checks

VTune 2026.2 software Hotspots completed for data_wait using the pair-3
environment. The profiled run also passed full prompt/token integrity and
the golden 48-token comparison. Artifacts:
tmp/q4e_vtune_dataflow_resume_wait_after, its .log/.hotspots.csv/.tokens.json
sidecars, and tmp/q4e-dataflow-resume-vtune-{command,report,tokens}.log.

zeCommandListAppendMemoryCopy sampled CPU time was 0.396731 seconds versus
0.478677 seconds in the original before profile. Both cover the whole
process, including initialization and unrelated copies; these samples cannot
quantify request-only savings. Module creation and waiting still occupy
large entries, and application PDBs remain unavailable, so no source-level
CPU-kernel or DRAM/GPU-idle conclusion is drawn from unresolved addresses.
The request counters and unprofiled pairs above remain the performance
evidence. Profiled elapsed time is excluded from those pairs.

Final git diff --check passed. The focused eight-test regression, explicit
stateful acceptance, three unprofiled pairs and post-change profile completed
serially; no full-repository CTest run was performed. No build, test, model
or VTune process from this continuation remains running.
