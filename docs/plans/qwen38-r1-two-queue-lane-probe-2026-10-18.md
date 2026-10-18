# Qwen3.8 R1: two-queue device-lane probe (2026-10-18)

Scope: A770 Windows/L0 production environment after Hyper-V was enabled.
No model was loaded. This is a relative concurrency probe; absolute timings
are not comparable to pre-Hyper-V model runs.

Probe: `tmp/q4e-structural-20261016/probe_two_queue.cpp`

Receipts:

- `tmp/q4e-structural-20261016/two-queue-long.log`
- `tmp/q4e-structural-20261016/two-queue-immediate.log`
- `tmp/q4e-structural-20261016/two-queue-copy.log`

## 0. Answer

The current driver exposes **no usable independent compute lane** through a
second SYCL in-order compute queue. Two compute kernels on separate queues
have zero measured kernel-time overlap.

The copy engine **is** independent enough to overlap a compute kernel:

```text
compute-only:       1.444 ms
host->device copy:   3.584 ms
serial:              4.959 ms
overlapped:          3.659 ms
overlap interval:    1.261 ms
```

Therefore the feasible device-side lane is a **copy/DMA prefetch lane**, not a
second GPU cold-expert compute lane. A cold-expert executor that requires a
second compute queue is NO-GO on the current path.

## 1. Compute/compute overlap

Production regular command lists:

| arm | median |
|---|---:|
| compute-only | 1.365 ms |
| memory-stream-only | 2.510 ms |
| same-queue serial | 3.785 ms |
| two-queue overlap | 3.797 ms |

The profiling-event intersection was **0.000 ms** for every iteration. The
wall time did not improve (`0.997x`); the second queue was serialized.

Immediate command lists:

| arm | median |
|---|---:|
| compute-only | 1.362 ms |
| memory-stream-only | 2.494 ms |
| same-queue serial | 3.834 ms |
| two-queue overlap | 3.434 ms |

The wall time improved, but the kernel event intersection was still
**0.000 ms**. The gain is from submission/command-list behavior, not kernel
co-execution. This is not a cold-compute lane, and prior production immediate
list tests were regressive; do not use this as an adoption path.

## 2. Copy-engine/compute overlap

Regular command lists, 32 MiB host USM source, 1.4 ms compute kernel:

| arm | median |
|---|---:|
| compute-only | 1.444 ms |
| copy-only | 3.584 ms |
| same-queue serial | 4.958 ms |
| copy on second queue + compute | 3.659 ms |

Measured overlap interval: **1.261 ms** median (36.3% of the union), with a
wall-time speedup of **1.355x**. The copy engine can read host USM concurrently
with a compute kernel, so prefetching cold-expert bytes while attention runs is
possible.

## 3. Consequence for cold experts

The C1 direct-arena split already showed that eliminating the bounce fill is
not sufficient:

```text
fill_ms:       24 ms class -> 0.047 ms
cpu_join_ms:   12.03 -> 0.98
router_wait:   16.84 -> 31.33
bank_wait:     0 -> 16.70
rate:          16.232 -> 12.245 tok/s (-24.6%)
```

The copy engine can prefetch bytes, but it cannot execute the cold-expert
kernels. On the measured path:

- router-dependent staging still lands in the main queue's critical section;
- two compute queues do not co-execute, so a separate cold compute queue does
  not create a second lane;
- the only independently measured lane is copy/DMA.

To get a real device-side cold lane, one of these would be required:

1. a single cooperative/persistent kernel that schedules main and cold work
   inside one launch (not a second SYCL compute queue), or
2. a driver/queue-group primitive that proves independent compute execution
   on A770 Windows (not exposed by the current SYCL path), or
3. cross-request/token pipelining where the cold result is not on the same
   layer dependency chain (not batch-1 single-stream decode).

None is ready to implement as a small delta. Do not add a second compute queue
to the production cold path and expect overlap.

## 4. Next structural direction

If the cold lane is pursued, start with a **copy-engine prefetch** probe on
real layer data:

```text
route known for layer L
-> prefetch future/staged cold bytes on copy queue
-> main attention/compute continues on q0
-> consume staged bytes later, with one explicit dependency
```

The prefetch must be for a later dependency point; copying the current layer's
needed weights immediately before consuming them still leaves the cold
execution in the critical path. If no useful future route can be predicted,
copy-lane overlap alone cannot rescue the split.

The immediate decision is therefore:

- **copy lane: GO for a bounded prefetch probe;**
- **second compute queue cold lane: NO-GO;**
- **C1 direct split: NO-GO;**
- **all-CPU cold misses remain the correct current default.**
