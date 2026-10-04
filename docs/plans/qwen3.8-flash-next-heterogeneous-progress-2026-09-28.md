# Qwen3.8-Flash-Next: H1-H5 Hardening and Direct Prefill I/O

Latest follow-up (2026-10-04): [H9 measurements and acceptance](qwen3.8-flash-next-heterogeneous-h9-results-2026-10-04.md) continue the validated H8 mapped NF4-Q8/zero-host-pool path. This document retains the earlier H1-H5 baseline and decisions; they do not describe current experimental dispatch or throughput.

Date: 2026-09-28. Working-tree implementation over `f78732d`.
Target: Intel Arc A770, driver 32.0.101.8860, Windows, oneAPI 2026.1.
Model: `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`.
This report supplements the original next-stage design, not its historical baseline.

## Status and Decision Gates

**Later CPU alignment update:** the opt-in `AILA_Q4E_CPU_GPU_ORDER=1`
candidate passes three golden story runs and real multi-turn acceptance,
but negative performance screening keeps it OFF. The original Auto CPU
route still fails the golden gate after scale repair alone. See
[CPU alignment results](qwen3.8-flash-next-cpu-alignment-2026-09-28.md);
the CPU failure below is retained as the preceding experiment's history.

- H6a/b remain completed. H6c stays experimental and OFF; anchor-2/16
  split-reference differences remain unresolved.
- H1/H4/H5 ownership and measurement fixes are implemented and tested.
  This does not complete asynchronous I/O, cancellation, or all failure paths.
- GPU ready-first is independently selectable with `AILA_Q4E_READY_FIRST=1`.
  It remains OFF by default: the corrected screening did not show a gain.
- CPU dispatch remains OFF. Once the one-credit policy actually continued
  running CPU experts, the golden-token gate failed. More CPU workers or
  credits must wait for numerical localization and alignment.
- Direct oneDNN prefill I/O is experimental, default OFF via
  `AILA_Q4E_DIRECT_PREFILL_IO`. Correctness passed; paired measurements below
  determine whether this copy-removal candidate is useful.

## Ownership and Measurement Repairs

`Qwen4ExpExpertReader` uses relaxed atomic counters instead of unsynchronized
worker writes. Staging byte accounting handles uneven partitions and expanded
float scales. A concurrent snapshot is approximate; a joined snapshot is exact.
Reset is permitted only at a quiescent boundary. These bytes are not physical
SSD I/O bytes. Concurrent reader/count tests cover these contracts.

`Qwen4ExpExpertCache` collects per-slot events and batch counts on the
coordinator after fills join. The failure guard conservatively marks possible
DMA before worker dispatch rather than racing on a shared flag. A new batch
resets copy duration/rate and consumer-gap fields. Event profiling uses minimum
copy start and maximum copy end, with profiled/unprofiled batch counters.
Event queries are opt-in through `AILA_Q4E_DMA_PROF` or explicit stats access.

The copy span includes queue gaps. The consumer hook is a host interval, not
GPU idle time or physical PCIe bandwidth. Consecutive-batch tests verify reset,
byte accounting, and idempotent profiling. Timing state moves/resets with cache
ownership. A hit with unfinished DMA retains its dependency instead of being
unconditionally declared Ready.

The executor now acquires the complete unique GPU expert set, protecting
already-ready experts from pending-miss eviction. Plans carry slot generations;
stale plans and over-capacity requests fail instead of consuming another key's
weights. Full-cache tests check ready-slot preservation and generation rejection.
Cleanup drains work; CPU scratch reserves the supported maximum four tasks and
four tokens, with bounded setters and capacity validation.

Final review also reproduced a configuration bug: setting
`AILA_Q4E_CPU_MAX_TOKENS=1` made the old executor silently skip two-token
GPU-only ready-first execution. GPU/merge capacity is now independent of CPU
admission, and invalid execution sizes throw. The multi-token test fixes the
CPU limit at one while executing twenty GPU tasks.
Negative reproduction: `tmp/q4e-ready-first-cpu-cap-before.log`.

## Ready-First and CPU Screening

For mixed ready/pending GPU work, ready kernels are submitted before the
synchronous acquire. Ready slots have consumer fences and remain in acquire's
reservation set. Pending kernels follow the acquire; ordered rank merge retains
the original BF16 boundary. All-hit GPU-only work keeps the original fast path.
Next-layer prefetch is deferred until after current acquire to avoid canceling
useful current-layer prefetch. The unnecessary final host wait is removed only
for ready-first; merge/residual/next-router ordering remains in-order.

This is not a fully asynchronous expert reader or background CPU worker. CPU
expert arithmetic still runs on the calling thread after acquire. Synthetic
tests observe ready submission before fill and compare deferred/all-resident
outputs bitwise, including more than sixteen tasks. Real ready-first acceptance
passed skip-logits/rollback, two-turn reuse, edited-history rollback and policy
transitions (`tmp/hetero-ready-first-real.log`).

The old one-credit CPU penalty could become permanently zero during warmup.
It now retries after eight eligible plan calls; explicit zero stays disabled.
This cooldown is an experimental policy, not an optimal dispatch threshold.
CPU compute timing excludes activation wait. The former exposed-tail metric is
named observed straggler interval because it is not an exact exposed GPU tail.

Corrected screening, same binary, ReadFile, canonical 30-token story, greedy,
48 generated tokens, reusable-prefix snapshots, max sequence 512:

| Variant | GenerateMessages ms | Golden IDs | Decision |
|---|---:|---|---|
| Warm baseline, excluded | 5674.49 | Match | Warmup observation |
| Baseline | 5677.79 | 48/48 match | Reference |
| GPU ready-first | 5816.12 | 48/48 match | +2.4%, one sample; keep OFF |
| Ready-first + CPU credit 1 | 6437.93 | Fail at step 26 | Stop CPU expansion |

CPU run counters include warmup: 158 expert tasks, 82.38 ms CPU arithmetic.
Expected token ID 23221 (` beneath`), actual 3808 (` across`). The first
operator-level divergence has NOT been localized. Tiny synthetic tests with
absolute tolerance 1e-3 do not establish real-model generation equivalence.
The planned later CPU comparison samples were aborted by the token gate.
Logs/metadata: `tmp/q4e_prof_hetero_final_{warm_0,base_1,ready_1,cpu1_1}.*`.
Earlier CPU screening that executed only two warmup tasks is invalid evidence
of sustained heterogeneous execution and must not be used for a speedup claim.

VTune xpu-offload collection: `tmp/hetero-foundation-vtune/` and its `.log`.
The whole-process 45.738 s included 26.430 s `zeModuleCreate`; this is not
steady-state performance. Detailed GPU utilization was unavailable because
of profiling privileges. No utilization or physical-link conclusion is drawn.

## Direct Prefill I/O

`Bnb4BitLinear` optionally binds the caller's BF16 source/destination directly
to the same oneDNN matmul. This removes the two staging copies without changing
weight dequantization, descriptors, arithmetic, or scratch allocation. Only the
normal oneDNN path, rows > 1, empty LoRA attachments and disjoint logical byte
spans qualify. Full/partial alias, decode, LoRA and fused NF4/XMX retain the
existing behavior. Fused-row initialization inherits the flag. The dispatch
counter `direct_io` confirms actual use.

New tests use NF4 block 64, 2560x2560 geometry, rows 1/2/4/26/64, comparing
every BF16 bit against staged oneDNN. They exercise pointer rebinding, full
alias and one-element partial overlap, and assert dispatch counters. This is
implementation-equivalence testing, not an independent CPU numerical reference.

Real model with ReadFile and direct I/O ON passed 248320 BF16 skip-logits/
rollback comparisons, two-turn reuse, edited-history rollback and policy
transitions. H6c diagnostic anchors 2/16 still differ from split reference;
the test's state-restore PASS does not resolve those differences.
Log: `tmp/q4e-direct-io-real.log`.

### Isolated Measurements

`[MEASURED]` Same executable SHA256
`563586F7268FAAAD7A59C6088C915201C3885AEF0DC73387931AADC8D75AE404`.
ReadFile, max sequence 512, greedy, thinking budget 0, 48 output tokens,
`perf/qwen4exp/prompts/story.json`, default reusable-prefix 26+4 splitting.
CPU, ready-first, inline checkpoints and MoE rows are OFF.
`AILA_PROFILE_Q4E_DECODE=1` and `AILA_DEBUG_TOKEN_IDS=1` on both sides.

Order: excluded warm baseline; base1/on1; on2/base2; base3/on3. Each run
started a fresh process with normal model warmup; OS file pages were warm,
not forcibly cleared. No overlapping GPU workload. Timings below exclude
model loading and initialization, but describe the first story request in
each process, not repeated in-process requests.

| Variant | GenerateMessages ms | Prefill 26+4 ms | Decode mean ms |
|---|---:|---:|---:|
| Baseline 1 | 5677.90 | 2168.32 | 73.713 |
| Direct I/O 1 | 5659.37 | 2147.91 | 73.755 |
| Direct I/O 2 | 5656.77 | 2148.25 | 73.686 |
| Baseline 2 | 5687.40 | 2165.81 | 73.947 |
| Baseline 3 | 5725.54 | 2212.71 | 73.797 |
| Direct I/O 3 | 5673.29 | 2159.67 | 73.797 |

GenerateMessages medians: 5687.40 -> 5659.37 ms (-0.493%). Median absolute
deviations: 9.50 and 2.60 ms; ranges 5677.90-5725.54 and 5656.77-5673.29 ms.
Prefill medians: 2168.32 -> 2148.25 ms (-0.926%). Decode uses exactly 47
rows, positions 30..76, count=1, logits=1, excluding warmup and the final
state-only step. The runner's printed 45-row estimate is not used here.

All seven complete prompt/generated traces pass `perf/compare-token-traces.ps1`
against `tmp/h6-baseline.log` (48/48 IDs). Dispatch counters increase by 192
direct calls per prefill slice, removing 384 copy submissions per slice.
Logs, binary/environment metadata and token comparisons:
`tmp/q4e_prof_directio_{warm_0,base_1,on_1,on_2,base_2,base_3,on_3}.*`.

Decision: small favorable screening result, no default promotion. Three paired
request samples are not the skill's >=20-sample operator benchmark, do not
establish a universal improvement, and do not imply a decode speedup.

### Combination With Existing MoE Rows

`AILA_Q4E_PREFILL_MOE_ROWS=1` plus direct I/O passes the explicit real-model
acceptance on the final binary: exact skip-logits/rollback checks, two-turn
reuse, edited-history replay and policy transitions.
Log: `tmp/q4e-direct-io-rows-real.log`. Known H6c diagnostics remain unchanged.

`[MEASURED]` Same binary and story settings as above. Compare rows-only
against rows plus direct I/O, ordered rows1/both1, both2/rows2, rows3/both3.
The real-model acceptance ran before this series; all file pages were under
ordinary warm-cache conditions, not controlled cold-cache measurements.

| Variant | GenerateMessages ms | Prefill 26+4 ms | Decode mean ms |
|---|---:|---:|---:|
| Rows only 1 | 5535.57 | 2010.61 | 74.048 |
| Rows + direct 1 | 5531.46 | 2006.31 | 74.048 |
| Rows + direct 2 | 5562.25 | 2028.67 | 74.215 |
| Rows only 2 | 5514.94 | 2008.51 | 73.659 |
| Rows only 3 | 5518.06 | 2003.56 | 73.823 |
| Rows + direct 3 | 5507.38 | 2005.35 | 73.547 |

GenerateMessages medians: 5518.06 -> 5531.46 ms (+0.243%). Median absolute
deviations: 3.12 and 24.08 ms. Prefill medians: 2008.51 -> 2006.31 ms
(-0.110%). Pairwise differences change sign; no reliable incremental gain
is established. All six runs pass the full golden 48-token comparison.
Logs/metadata/comparisons: `tmp/q4e_prof_directrows_{rows_1,both_1,both_2,rows_2,rows_3,both_3}.*`.

Decision: do not recommend the combination over rows-only and do not promote
direct I/O. Rows-only remains the stronger short-story candidate, with no
claim that a comparison between these separately ordered series is paired.
All thirteen screening runs (including the excluded warm run) have complete,
equal golden traces. These experiments leave all defaults unchanged.

## Verification

- Full build passed: `tmp/q4e-direct-io-final-build.log`.
- Final focused CTest selection: 11/11, 86.44 s, serial GPU execution.
  Command: `ctest --test-dir build-refactor -R 'Aila(Qwen4Exp|Bnb4BitLinear|LoraExecution)' --output-on-failure -j 1`.
  Log: `tmp/q4e-direct-io-final-tests.log`. Includes LoRA fused/oneDNN, shared
  NF4 linear, Q4E kernels/modules/generation/readiness/CPU/reader/DMA tests.
- Earlier pre-direct-I/O selection: 8/8, 62.40 s,
  `tmp/hetero-ready-first-tests-final.log`.
- Real-model acceptance is an explicit `--model` run; the model-free CTest
  generation test alone is not real-model acceptance.
- Final `git diff --check` passes. No commit or default change is part of
  this working-tree delivery. The full repository CTest suite was not run.

## Next Work

1. Localize CPU versus GPU on the same real expert and activation: compare
   scale decode, gate/up, BF16 h, down and ordered merge before tuning threads.
2. Use stable data-supply measurements before bounded asynchronous ReadFile
   or host-ready-first scheduling. Fill-pool exception propagation and ReadFile
   short-read handling remain known limitations, not completed repairs.
3. Revisit small-M projection/dequantization and GDN residency based on
   operator profiles, with strict logits/state gates for changed arithmetic.
4. Keep earlier MoE-row story gains separate: three-pair medians
   5673.68 -> 5507.95 ms (-2.92%), prefill 2179.13 -> 1998.99 ms (-8.27%).
   Those are earlier-binary, short-story measurements, not a universal gain
   or a new default. Long-prompt data supply was unstable.
