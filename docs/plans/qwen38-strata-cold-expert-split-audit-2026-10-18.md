# Qwen3.8 R1: cold-expert CPU/GPU split audit (2026-10-18)

Scope: canonical Qwen4Exp decode, `count == 1`, adaptive pipeline. Read-only
source/receipt audit; no GPU run and no source change. Base revision
`9b12d3dd88ff0571503904f9d42a77748f825cd3`.

Related inputs:

- `tmp/q4e-strata-principle/REPORT.md` section 3
- `tmp/q4e-bottleneck-audit-20261016/REPORT.md`
- `docs/plans/qwen38-performance-research-charter-2026-10-14.md` sections 2, 5, 6
- `docs/plans/qwen38-r1-cost-model-2026-10-14.md`
- `docs/plans/qwen3.8-flash-next-cpu-scale12-2026-09-28.md`
- `docs/plans/qwen3.8-flash-next-hotset-h4-c2-results-2026-10-02.md`
- `docs/plans/qwen3.8-flash-next-heterogeneous-h8-results-2026-10-03.md`
- `docs/plans/qwen38-r1-pinned-arena-2026-10-16.md`
- `docs/plans/qwen38-r1-strata-kernel-transfer-2026-10-17.md`

## 0. Verdict

1. **Current T1 really sends every cold miss to CPU.** This is not an
   accidental fallback or dispatch race. The production bundle sets
   `top_k=10`, `AILA_Q4E_CPU_MAX_EXPERTS=10`, `AILA_Q4E_COLD_POLICY=1` and
   `AILA_Q4E_CPU_THREADS=18`. For each one-token decode layer,
   `miss_count <= 10`; `StartArenaCold` takes
   `min(miss_count, max_cpu_credits_) == miss_count`; therefore
   `gpu_cold_count == 0` and `StageColdExpertsAsync` is not called.
2. **The Strata-style fraction flag is not connected to this path.**
   `AILA_Q4E_COLD_POLICY=3` and `AILA_Q4E_COLD_PCIE_NUM` are consulted only
   by the legacy `StartResidentCold` path. Canonical adaptive decode calls
   `StartArenaCold`, which ignores both. Changing those env vars alone does
   nothing in the current bundle.
3. **A fixed CPU/GPU fraction under the current bounce path is already
   refuted.** The canonical `CPU_MAX_EXPERTS=4` A/B moved about 24% of misses
   to GPU staging and changed the token from **66.006 ms to 95.518 ms**
   (`-37%` rate), despite cutting `cpu_join` by 7.0 ms.
4. **The root mismatch with Strata is lane independence, not the numeric
   ratio.** Aila's current GPU-cold path first performs an arena-to-pinned host
   memcpy on the coordinator, then queues H2D and GPU kernels. The measured
   path is effectively additive and also slows the CPU lane by about 39% per
   task through host-memory contention.
5. **Only one new split A/B is justified.** It must combine a token-scoped
   `40/256` PCIe share (15.6%) with direct DMA from the existing large-page
   arena. A per-layer cap or a plain `COLD_POLICY=3` rerun is prohibited.
   Under perfect lane overlap this split has an e2e upper bound of only about
   `+6%`; it cannot be presented as the path to 20 tok/s.

## 1. Canonical route and exact quantities

Production evidence is
`tmp/q4e-sprint-r1-20261014/cq-a2/result.json` and `cq-b1/result.json`.
Both record:

| env | value |
|---|---:|
| `AILA_Q4E_CPU_MAX_EXPERTS` | 10 |
| `AILA_Q4E_COLD_POLICY` | 1 |
| `AILA_Q4E_COLD_MODE` | 2 |
| `AILA_Q4E_CPU_THREADS` | 18 |

`Qwen4ExpExpertExecutor::init` reads the cap at
`src/models/Qwen4ExpExpertExecutor.cpp:650-651`. The cap is a
**expert/token-task cap per forward**, not a fraction and not unique-expert
count. The old CPU scaling note says the same explicitly.

For each one-token layer:

```text
classify top-k
  hits  -> resident GPU slots
  misses -> miss_experts[miss_count], miss_count <= 10

StartArenaCold(..., miss_count)
  cold_count = min(miss_count, max_cpu_credits)
             = miss_count                    // production cap is 10

gpu_cold_count = miss_count - cpu_count
               = 0

if (gpu_cold_count > 0)
  StageColdExpertsAsync(...)                 // never entered
```

Source order:

1. adaptive branch: `src/models/Qwen4ExpBnb4Backend.cpp:1300`
2. hit/miss classification: `Qwen4ExpBnb4Backend.cpp:1454-1473`
3. CPU start: `Qwen4ExpBnb4Backend.cpp:1511-1517`
4. GPU overflow split: `Qwen4ExpBnb4Backend.cpp:1520-1527`
5. CPU cap: `Qwen4ExpExpertExecutor.cpp:936-998`, especially `:970`
6. CPU join and result H2D: `Qwen4ExpExpertExecutor.cpp:1002-1136`
7. final ranked merge: `Qwen4ExpBnb4Backend.cpp:1592-1615`

Baseline retained20 receipt:

| quantity | median/token |
|---|---:|
| hits | 225 |
| misses | 255 |
| CPU tasks | 255 |
| staged experts | 0 |
| H2D | 0 MB |
| `cpu_compute` | 14.530 ms |
| `cpu_join` | 10.001 ms |
| `experts_host` | 22.113 ms |
| ForwardWall | 66.006 ms |

The 255 misses are not 255 distinct models experts; they are 255
expert/token tasks across 48 layers. With `count == 1`, no multi-row dedup or
weight reuse across rows exists.

## 2. `COLD_POLICY=3` is dead in canonical adaptive decode

The only fraction calculation is here:

```text
Qwen4ExpExpertExecutor.cpp:877-889   StartResidentCold
  if cold_policy_ == 3:
      reserve last floor(cold_count * cold_pcie_num / 256) for GPU
```

`cold_policy_` has no reference in `StartArenaCold` (`:936-998`) or in the
backend's adaptive split (`Qwen4ExpBnb4Backend.cpp:1511-1527`).

Consequences:

- `AILA_Q4E_COLD_PCIE_NUM=141` in the recorded worker environment does not
  create a 55% PCIe split in canonical decode.
- Setting `AILA_Q4E_COLD_POLICY=3` without wiring it into the adaptive path
  still leaves all misses on CPU.
- The old H4/H8 policy3 measurements belong to the legacy resident path. They
  are useful negative priors, not evidence that the adaptive path was tested.

## 3. Existing split/cap evidence and why it must not be repeated

### 3.1 Canonical cap=4: direct refutation

Receipt: `tmp/q4e-sprint-r1-20261014/decode/retained20-dec-cpu4.json`.

| field | all-CPU baseline | cap=4 | delta |
|---|---:|---:|---:|
| `cpu_tasks` | 255 | 173 | -82 |
| staged | 0 | 55 | +55 |
| H2D | 0 | 132.935 MB | +132.935 |
| `fill_ms` | 0 | 24.406 | +24.406 |
| `cpu_compute_ms` | 14.530 | 13.739 | -0.791 |
| `cpu_join_ms` | 10.001 | 2.960 | -7.041 |
| `router_wait_ms` | 12.686 | 25.436 | +12.750 |
| `experts_host_ms` | 22.113 | 40.641 | +18.529 |
| ForwardWall | 66.006 | 95.518 | +29.512 |
| implied rate | 15.15 tok/s | 10.47 tok/s | -30.9% |

This is the minimum-dose-like test that the current architecture had:
CPU tasks remained the majority (173/228 misses), and GPU received 55 tasks
(24.1%). The GPU fraction alone consumed 24.4 ms of host fill and 132.9 MB of
H2D to save only 0.8 ms of measured CPU compute. This is a structural failure
of lane independence, not a tuning miss.

### 3.2 Legacy Strata fraction experiments

- H4/C2 implemented `policy=3` with `141/256` and validated representative
  quality, but the performance evidence was a four-request screening curve
  (`13.4, 12.3, 15.1 tok/s`), not a controlled pair.
- H8 tested `64/256`: against the same-stage mapped/no-PCIe arm, archive
  latency worsened `16.968 -> 23.339 s`, code worsened `4.458 -> 6.065 s`,
  while one short story sample improved `6.131 -> 5.151 s`. That is mixed and
  dominated by regressions.
- These runs used the legacy path. They do not authorize a canonical adaptive
  rerun at the same fractions, and their direction agrees with cap=4.

### 3.3 Direct DMA and pinning are already partly answered

- `c1-e1` already showed that the large-page arena can be read directly by the
  copy engine at **9.05-9.23 GB/s**, versus 9.83 GB/s
  for pinned 32 MiB (probe in `qwen38-r1-pinned-arena-2026-10-16.md`).
- In prefill, removing the fill did not move wall time because the transfer
  remained additive with memory-bound kernels. That is a prefill result.
- Full 51 GiB pinning allocates but the first request fails with
  `UR_OUT_OF_RESOURCES`; do not retry it.

The **decode split + direct DMA** combination has not been tested. It is
different from both the global prefill direct-DMA arm and the bounce-based
cap=4 arm. That is the only admissible new mechanism below.

## 4. Three-way cost model and upper bounds

### 4.1 Constants

| item | value | source |
|---|---:|---|
| expert slot | 2,534,400 B | production geometry |
| CPU effective service | 45-53 GB/s | DDR4-1866 measured range |
| PCIe H2D | 9.9 GB/s | audit probe |
| measured bounce fill | 5.45 GB/s | cap=4: 132.935 MB / 24.406 ms |
| GPU hot/cold compute | tens of us per layer | C1 decode-MoE receipt |

Let:

```text
N = total cold expert tasks/token
P = 2.5344 MB per task
t_cpu = CPU lane aggregate service per task
t_fill = P / bounce-fill bandwidth
t_dma = P / PCIe bandwidth
t_gpu = cold GPU kernel service per task
t_pcie = t_fill + t_dma + t_gpu           // current Aila
t_pcie_direct = t_dma + t_gpu             // large-page direct DMA
```

At the retained20 baseline:

```text
t_cpu = 14.530 / 255            = 0.0570 ms/task
N*P   = 646.3 MB                = 0.646 GB/token
B_cpu = N*P / 14.530 ms         = 44.5 GB/s
t_fill = 2.5344 / 5.45          = 0.465 ms/task
t_dma  = 2.5344 / 9.9           = 0.256 ms/task
```

The CPU number is an aggregate 18-thread service rate, not a single-core
latency.

### 4.2 Ideal Strata split

If CPU and GPU-cold are genuinely independent lanes and `N` can be divided
continuously, the PCIe fraction that equalizes both lanes is:

```text
(1-p) * t_cpu = p * t_pcie

p_pcie* = t_cpu / (t_cpu + t_pcie)
```

Using `t_gpu = 0.05 ms` only as a small placeholder:

| lane model | `t_pcie` | ideal PCIe share | 255-task GPU-lane time |
|---|---:|---:|---:|
| direct large-page DMA | 0.306 ms | **15.7%** | 78.0 ms |
| current bounce | 0.771 ms | **6.9%** | 196.7 ms |

The direct-DMA result is close to Strata's 20% Q2_0 default. That is useful
alignment, not evidence of a speedup on Aila.

Under ideal overlap, the cold critical-path time is:

```text
T_cold* = N * t_cpu * t_pcie / (t_cpu + t_pcie)
```

For direct DMA, `T_cold* = 12.25 ms` versus `14.53 ms` all-CPU: a **2.28 ms**
cold-path saving. If `cpu_join` scales proportionally with CPU work, the total
`cpu_compute + cpu_join` bound is about **20.68 ms**, a **3.85 ms** e2e saving
from the 66.006 ms baseline. That is only about **+6.2%** and is an oracle
bound, not a forecast. The actual gate remains +5%.

### 4.3 Current Aila is additive, so its optimum is p=0

The cap=4 arm shows the current shape is approximately:

```text
T_current =
    (1-p) * t_cpu
  + p * (t_fill + t_dma + t_gpu)
  + synchronization / queue-depth cost
```

Both terms are positive, so increasing `p` increases the critical path.
Measured per-task CPU service also worsened:

```text
baseline: 14.530 / 255 = 57.0 us/task
cap=4:    13.739 / 173 = 79.4 us/task   (+39%)
```

The fill and CPU expert kernels contend for the same host memory and
coordinator. That is why the simple `max(CPU, PCIe)` model does not apply.

Do not search `CPU_MAX_EXPERTS=9,8,...,1` or fractions `141/256,64/256,...`.
Those are dose-response reruns of a mechanism whose current placement is
additive. The branch either becomes lane-independent or stays closed.

### 4.4 "If X were free" bounds

| X removed | resulting ForwardWall | resulting rate |
|---|---:|---:|
| `cpu_compute + cpu_join` | 66.006 - 14.530 - 10.001 = **41.475 ms** | **24.11 tok/s** |
| `cpu_join` only | 66.006 - 10.001 = **56.005 ms** | **17.86 tok/s** |
| all 255 weights over PCIe | transfer alone = 646.3 MB / 9.9 = **65.28 ms** | below CPU compute |

Therefore:

- 20 tok/s is theoretically reachable only by attacking the combined CPU +
  handoff/control budget, not by moving a small cold fraction.
- A perfect direct-DMA split can contribute at most a few milliseconds.
- Sending all misses over PCIe is strictly worse than the 14.5 ms CPU lane.

## 5. Candidate A/B slots

### C1 - GO to one probe: token-scoped `40/256` split + direct large-page DMA

This is the only candidate that differs materially from the refuted cap tests:

- **Share:** 40/256 = 15.625% of cold tasks per token. Use a token-scoped
  Bresenham accumulator (`credit += miss_count*40; n_pcie=credit>>8;
  credit&=255`) rather than `floor(miss*40/256)` per layer. Per-layer flooring
  would send zero on most layers because each layer has only about 5 misses.
- **No bounce:** `StageColdExpertsAsync` should submit per-slot DMA directly
  from `host_arena_.GetExpertPointer(...)` into the staging-device bank. Do
  not copy the selected slots into `staging_host_` first. A direct copy of
  one or two non-contiguous slots per layer is acceptable; the measured
  large-page source rate is 9.05-9.23 GB/s.
- **Order:** submit the GPU-cold copies before starting or immediately after
  starting the CPU job, so H2D can overlap the CPU lane. The current local
  host fill must be absent; otherwise this is just cap=4 again.
- **Scope:** canonical `count == 1` adaptive decode only. Do not combine with
  MTP, layer-major prefill, copy-q routing, or graph capture in the same arm.
- **Single A/B:** current canonical all-CPU vs direct-DMA 40/256. One round is
  sufficient unless this candidate is proposed for default-on.

Mechanism gates before trusting wall time:

| gate | required |
|---|---:|
| actual staged share | 10-18% of misses |
| bounce `fill_ms` | ~0 |
| staged bytes | about 0.08-0.13 GB/token |
| `cpu_compute + cpu_join` | lower than 24.531 ms |
| `router_wait` increase | <= 1.0 ms |

Decision gates:

- **Default-on candidate:** decode e2e median **>= +5%**, pairs won, TTFT
  regression <= 1%, and established cold-quality limits (KL <= 0.01,
  max-logit NRMSE <= 0.15). Save raw tokens because CPU/GPU placement changes
  rounding.
- **Opt-in only:** +3% to +5% with the mechanism gates passing.
- **NO-GO:** any wall regression, `router_wait` > +2 ms, or staged share
  outside the pre-registered band.

Upper-bound warning: this probe can plausibly clear +5% only if the direct
copy lane is genuinely independent. It is not a route to 20 tok/s.

### C2 - deferred, not an A/B now: window-deduplicated split

Strata deduplicates experts across a 2-4 token window before splitting. Aila
canonical T1 has no such dedup. The existing `StartArenaColdRows`/MTP paths
are a different execution regime and current MTP economics are negative.

If MTP later becomes the main decode path, re-evaluate with:

- unique cold experts, not row/task count, as `N`;
- top-1 agreement >= 95%;
- actual unique-dedup factor >= 1.5;
- MTP e2e >= +5% versus current T1 before any default-on discussion.

Do not use T1 split evidence to approve this path or vice versa.

No second immediate candidate is proposed. A fixed cap sweep, legacy policy3,
pinned 51 GiB arena, and prefill direct-DMA rerun are all already closed or
insufficient.

## 6. Disposition

| item | disposition |
|---|---|
| all misses on CPU in canonical T1 | **confirmed and intentional under cap=10** |
| `COLD_POLICY=3` in adaptive decode | **no-op; do not use as an experiment** |
| cap-based fixed split | **NO-GO; cap=4 receipt** |
| legacy PCIe fraction | **closed; mixed/worse evidence** |
| bounce fill under split | **remove before any new split test** |
| token-scoped 40/256 + direct DMA | **one probe candidate, C1** |
| cap/fraction dose matrix | **prohibited** |
| whole-token graph / handoff | still the larger decode lever, unchanged |

## 7. Receipts

- `tmp/q4e-sprint-r1-20261014/decode/retained20-r1-diag.json`
- `tmp/q4e-sprint-r1-20261014/decode/retained20-dec-cpu4.json`
- `tmp/q4e-sprint-r1-20261014/cq-a2/result.json`
- `tmp/q4e-sprint-r1-20261014/cq-b1/result.json`
- `docs/plans/qwen3.8-flash-next-hotset-h4-c2-results-2026-10-02.md`
- `docs/plans/qwen3.8-flash-next-heterogeneous-h8-results-2026-10-03.md`
- `docs/plans/qwen38-r1-pinned-arena-2026-10-16.md`
- `docs/plans/qwen38-r1-cost-model-2026-10-14.md` section 6.1
