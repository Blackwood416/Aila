# Qwen3.8-Flash-Next: 12-Thread CPU Expert Scaling

Date: 2026-09-28. Working tree over f78732d, Windows, oneAPI 2026.1,
A770 driver 32.0.101.8860. Xeon E5-2696 v3: 18 physical/logical cores,
45 MiB L3 reported by CIM, approximately 64 GiB RAM. Model:
`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`.

This follows the user's explicit request for twelve CPU threads and large
expert batches, superseding the earlier recommendation to stay at one expert
and four threads. Failed golden traces remain feasibility data, not valid
speedups, but do not stop other isolated capacity experiments.

## Implementation

All new policies remain opt-in; no default CPU-dispatch promotion.

- `AILA_Q4E_CPU_THREADS`: worker-local TBB arena budget 1..12, default 2.
  This is not affinity or proof of twelve continuously occupied cores.
- `AILA_Q4E_CPU_MAX_EXPERTS`: 0..40 expert/token tasks per layer, NOT unique
  experts or experts per token. Pinned scratch and host leases support forty.
- `AILA_Q4E_CPU_FIXED_CREDITS=1`: bypass admission throttling, retaining
  straggler measurements. Explicit zero credits still disable dispatch.
- `AILA_Q4E_CPU_PLACEMENT=0`: host-ready GPU misses only. `1`: also accept
  host-ready GPU hits. `2`: prepare selected host weights synchronously.
  Do not hand in-flight GPU copies to CPU. Preparation is timed separately
  and remains inside request time; it is not a parallel weight reader.
- `AILA_Q4E_CPU_BATCH=1`, with overlap: two flat TBB stages over all
  expert/row tiles instead of two parallel loops per expert; convert shared
  input once. No per-layer USM allocations or new worker threads.
- `AILA_Q4E_CPU_PREPARED_INPUT=1`, with batch and GPU-order: transpose each
  64-element input block to eight-lane-major order once. Replace contiguous
  weight gathers with vector loads, and NF4 gathers with register permutation
  and blend. Preserve lane FMA order, scale and BF16 rounding, and reduction.
- `AILA_Q4E_CPU_SMALL_M=1`: allow heterogeneous execution BEFORE GPU batch
  acquire for 2..4 tokens. Without this override, the story's 4-token suffix
  stays GPU-only, so forty configured credits still yield at most ten tasks.

Prepared input is CPU AVX2 work, not a GPU QMAP/SLM change. It does not repack
weights or reuse an expert's weights across different token tasks. Fixed
geometry: H=2560, I=640, block=64, tokens=1..4, tasks=0..40. Invalid inputs
are rejected; prepared arithmetic is rejected by non-batch entry points.
Coordinator-owned leases, worker drain, original ordered rank merge and
preallocated execution buffers are retained. Maximum actual CPU tasks and
host preparation time are exposed. Overlap is a host interval intersection,
not GPU idle time or physical storage throughput.

## Operator Measurement

Harness: `AilaQwen4ExpCpuExpertTests --batch-scale`. Forty separately
allocated, differently seeded synthetic experts, alternating nested and
expanded FP32 scales. 1/4/10 tasks at one token; forty distinct tasks at four
tokens. Forty tasks exceed L3; ten can fit. Warm synthetic results are not
real-model DRAM rooflines. Allocation is outside timing; input conversion,
BF16 intermediates, scheduling and completion are included.

Three warmup pairs, then three repetitions of twenty interleaved
sequential/batch pairs per shape/thread/backend. Below: median of three
sample medians. For backend 5, sequential means original GPU-order backend 4.

| Tasks | GPU-order 4T sequential ms | GPU-order 12T sequential ms | Prepared batch 12T ms |
|---|---:|---:|---:|
| 1 | 1.2172 | 0.5450 | 0.1262 |
| 4 | 4.9663 | 2.3312 | 0.4448 |
| 10 | 12.6982 | 5.8163 | 1.0432 |
| 40 | 51.3209 | 23.4655 | 4.1333 |

At fixed 12T and flat batching, original GPU-order medians are 5.1442 ms
(ten tasks), 20.4528 ms (forty). Prepared input improves these by 4.93x and
4.95x. The 51.3209 -> 4.1333 ms comparison combines threads, scheduling and
kernel changes (12.42x), NOT end-to-end inference acceleration.

Prepared ten-task medians: 1.0432/1.0550/1.0339 ms; p10/p90:
1.0001/1.0694, 0.9950/1.0995, 0.9910/1.0805 ms. Forty-task medians:
4.1467/4.1130/4.1333 ms; p10/p90: 4.0069/4.2200,
4.0346/4.2800, 3.9197/4.3011 ms. Full repetitions are in the logs.

All tested batch BF16 h and FP32 down outputs match the corresponding
per-expert CPU reference bitwise. This is not universal CPU/GPU equivalence:
old GPU-order already had high-dispatch golden-trace differences. Evidence
for the rewrite is code inspection plus equivalent-output timings, not
VTune instruction or DRAM attribution.

Logs: `tmp/q4e-cpu-scale12-micro.log` (initial batching),
`tmp/q4e-cpu-prepared12-micro.log` (prepared input).

## Initial Model Screen

Initial binary SHA256:
`F4B3CD304766A9A1CB017C258C3EAB3874CF24C08F2295832B0597B69B7B76E7`.
No prepared input or small-M override in this build.

One GPU pressure process at a time, fresh process per case, one excluded warm
run after building, normal initialization warmup, warm OS pages without
forced cache clearing. Canonical story: 30 prompt tokens, 26+4 reusable
prefix, greedy, thinking budget zero, 48 outputs, max sequence 512. ReadFile,
profiling and token trace ON; H6c/direct-I/O/ready-first global settings at
defaults. CPU cases: 12 compute arena threads and 4 fill workers. GPU-only
8-worker baseline is also measured. Times are GenerateMessages first-request
totals, EXCLUDING load/warmup. Counters INCLUDE warmup. Single samples are
screening, not paired proof.

| Case | Request ms | Golden IDs | Actual max tasks | Cumulative join ms |
|---|---:|---|---:|---:|
| GPU-only, fill=8 | 5677.24 | exact | 0 | 0 |
| GPU-only, fill=4 | 5902.70 | exact | 0 | 0 |
| Miss-only sequential, credits=10 | 9692.34 | differs at 26 | 10 | 3440.93 |
| Miss-only batch, credits=10 | 9390.40 | differs at 26 | 10 | 3038.00 |
| Host-ready including GPU hits, credits=10 | 16705.50 | differs at 26 | 10 | 10695.39 |
| Forced, credits=5 | 11372.44 | differs at 26 | 5 | 5011.35 |
| Forced, credits=10 | 18462.99 | differs at 26 | 10 | 12389.08 |
| Forced, credits=40 | 18706.57 | differs at 26 | 10 | 12514.13 |
| Original AVX2 arithmetic, forced 5 | 8986.83 | differs | 5 | 2436.13 |
| Original AVX2 arithmetic, forced 10 | 13396.15 | exact | 10 | 7324.40 |

Miss-only batching executed 7615 CPU tasks including warmup, versus roughly
170 in the earlier one-credit campaign. Scale was exercised, but original
gather-heavy CPU arithmetic remained too slow. Taking GPU-resident experts
also sacrifices already-available GPU work. None is a faster valid GPU-only
replacement. Logs/metadata/token checks: `tmp/q4e_prof_scale12_screen1_*.*`;
excluded warm run: `tmp/q4e_prof_scale12_warm_gpu8.*`.

## Verification

- Initial build passed; eight Q4E CTests passed serially in 63.87 s.
  `tmp/q4e-cpu-scale12-{build,tests}.log`.
- Prepared build passed; eight Q4E CTests passed serially in 63.26 s.
  `tmp/q4e-cpu-prepared12-{build,tests}.log`.
- `AilaQwen4ExpDmaTimingTests --large-cpu-ordered` passed with 12 threads
  and prepared input: 5/10/40 tasks, mixed/all-CPU, repeated buffers, fixed
  credits despite penalty, host/GPU-hit placement, ordered merge, all host
  readers released. `tmp/q4e-cpu-prepared12-large-ordered.log`.
- Existing GPU reference, default dispatch and overlap exception tests pass.
  Known H6c diagnostics remain unresolved.

## Prepared-Input Model Screen

Final runtime SHA256:
`EBCC69AD31A4987BE67AC7D7F3EA56C328E282A9052066E745E64BEF8DD1BA98`.
CPU test binary SHA256:
`E3381508243AEBC404EF767F742ED3418C4C5D06405128172C51965DE48EE373`.
Same story/settings/protocol as above. Excluded warm run 6034.13 ms;
subsequent GPU baseline 5689.62 ms. Prepared cases differ only as named.

| Case | Request ms | Golden IDs | Actual max tasks | Cumulative join ms |
|---|---:|---|---:|---:|
| GPU-only, fill=8 | 5689.62 | exact | 0 | 0 |
| GPU-only, fill=4 | 5898.85 | exact | 0 | 0 |
| Old GPU-order batch, miss-only 10 | 9220.40 | differs | 10 | 3056.44 |
| Prepared batch, miss-only 10 | 6719.18 | differs | 10 | 375.40 |
| Prepared batch, host-ready 10 | 7827.19 | differs | 10 | 1706.96 |
| Prepared batch, forced 5 | 7241.44 | differs | 5 | 596.80 |
| Prepared batch, forced 10 | 8948.67 | differs | 10 | 2663.30 |
| Old batch, forced 40 + small-M | 19327.56 | exact | 40 | 13807.92 |
| Prepared batch, forced 40 + small-M | 9144.62 | exact | 40 | 2962.22 |

Miss-only prepared completed 6820 CPU tasks including warmup, with
1019.29 ms CPU compute and 443.40 ms CPU/acquire interval overlap.
Host-ready prepared completed 21758 tasks with 2744.17 ms compute.
Small-M now ACTUALLY executes forty tasks. Other high-volume policies still
show golden differences; matching full placement does not certify them.
Logs/metadata/token checks: `tmp/q4e_prof_prepared12_screen1_*.*`;
excluded warm run: `tmp/q4e_prof_prepared12_warm_gpu8.*`.

## Paired Large-Placement Result

Same final binary, twelve CPU threads, four fill threads, fixed forty credits,
forced placement, small-M ON, batch ON, GPU-order ON. ONLY prepared input
changes. Pair 1 is the adjacent screen cases; pair 2 reverses order; pair 3
returns to old/new. No other pressure workload runs simultaneously.

| Pair | Old batch ms | Prepared batch ms | All 48 golden IDs |
|---|---:|---:|---|
| 1 | 19327.56 | 9144.62 | exact on both |
| 2 (reverse order) | 19717.17 | 9283.33 | exact on both |
| 3 | 19281.85 | 9192.97 | exact on both |

Medians: 19327.56 -> 9192.97 ms, 52.44% lower latency, 2.10x speedup over
the SAME large-CPU-placement baseline. MAD: 45.71 vs 48.35 ms. This is an
end-to-end improvement to that experimental path, not a twenty-sample service
benchmark or an improvement over GPU-only. GPU-only at 5689.62 ms remains
substantially faster; do NOT promote forced placement by default.

For pair 2 prepared, subtracting pre-request counters removes warmup:
2352 async layers, 2837.24 ms join, 39.14 ms CPU/acquire overlap,
1305.85 ms synchronous host preparation. Final preparation is 1868.44 ms;
562.59 ms belongs to warmup. These host intervals are not additive independent
bottlenecks or wholly recoverable request time.

Logs: `tmp/q4e_prof_prepared12_pair{2,3}_{slow,fast}_small40.*`.
An extra acceptance-run trace is exact at 9195.04 ms and is not included
in paired statistics: `tmp/q4e_prof_prepared12_accept_fast_small40.*`.

## Decision And Next Work

Retain experiments OFF by default, with the measured CPU kernel and large
capacity available for subsequent work. Larger resources were worthwhile:
batch scheduling helped modestly; replacing the gather-heavy algorithm gave
much more. It did not make unconditional CPU takeover optimal.

Next structural experiments, not established improvements:

1. Batch/overlap forced host preparation using bounded fill machinery,
   retaining coordinator ownership, exception recovery and host leases.
2. Reduce per-rank result transfers/merge submissions without changing FP32
   rank order. Study contiguous CPU prefixes or all-CPU layers separately;
   arbitrary CPU/GPU rank aggregation changes arithmetic order.
3. Account for future GPU reuse. Current miss-only CPU execution does not
   populate GPU cache and may keep repeatedly used experts on CPU. Test this
   cache-policy hypothesis with matched traces.
4. Isolate high-dispatch CPU/GPU numerical differences. Prepared-vs-old CPU
   bitwise agreement and one matching route do not certify mixed placements.

## Final Acceptance And Reproduction

The prepared forty-task/small-M configuration passed explicit real-model
acceptance: all 248320 BF16 skip-logits/rollback logits identical, two-turn
reuse versus fresh replay, edited-history rollback, and snapshot-policy
transitions. Anchors 26/28 had zero split-reference mismatches. Existing H6c
anchor-2/16 diagnostics remain (231920/235541 mismatches); they are not fixed
or silently accepted as equivalent. Log:
tmp/q4e_prof_prepared12_accept_fast_small40.real.log.

This acceptance covers that configuration, not the mixed-placement cases
that failed the story golden trace. No default promotion or commit. All
build/test/inference processes launched for this campaign completed.
Final git diff --check passed.

Reproduce with unique tags to preserve earlier logs (one pressure process):

    pwsh -NoProfile -File perf/qwen4exp/q4e_cpu_scale.ps1 -Tag local_screen -PreparedScreen
    pwsh -NoProfile -File perf/qwen4exp/q4e_cpu_scale.ps1 -Tag local_pair -ScalePair -ReverseOrder
    pwsh -NoProfile -File perf/qwen4exp/q4e_cpu_scale.ps1 -Tag local_accept -Cases fast_small40 -RealAcceptance
    build-refactor/AilaQwen4ExpCpuExpertTests.exe --batch-scale
    build-refactor/AilaQwen4ExpDmaTimingTests.exe --large-cpu-ordered

The campaign runner saves full environment/binary sidecars and token integrity
comparisons. It retains mismatching runs as feasibility data and checks for
complete 48-token generation. RealAcceptance additionally runs the model
suite and propagates its failure. Do not infer effective CPU task counts
from the configured credits; inspect max_tasks in each trace.
