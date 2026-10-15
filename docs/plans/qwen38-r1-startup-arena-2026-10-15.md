# R1/B4 - Worker startup: host-arena fill (51.46 GiB) and F1-style page prefetch

Author: B4 implementation subagent. Date: 2026-10-15. Repo `E:\RiderProjects\Aila`,
HEAD `40239cd` (pair `retained24`). Charter: `docs/plans/qwen38-performance-research-charter-2026-10-14.md`.

Scope discipline: source edit confined to `src/models/Qwen4ExpHostArena.cpp`
(`.hpp` unchanged, no signature changes); no build, no GPU process, no git write.
Only this document and `tmp/q4e-sprint-r1-20261014/startup/` were written.
Measured values cite an artifact path; derived values are tagged `[estimate]`.

## 0. TL;DR

The 51.46 GiB arena fill runs at 578 MB/s because every `memcpy` demand-faults
file-backed SafeTensors pages one after another on a single file section
(C7 mechanism, re-confirmed on `retained24`: `arena RAM copies` 95.6 s =
1992.0 ms/layer, `tmp/q4e-strata-c7/b4-loadtiming2/worker.log`).

Change shipped in this round: F1-style `PrefetchVirtualMemory` hint on the four
per-layer source ranges, **new env `AILA_Q4E_HOST_ARENA_PREFETCH`, default 1**
(`=0` restores the historical byte-identical path). Per layer the ranges are
queued in 1 MiB chunks, at most 64 entries per call, right before the fill and
the GPU-owned staging copies consume them. Advisory only: failures are ignored,
no data-path change, works only while the sources are the mmap `host_view`.

Predicted effect `[estimate]`: fill 95.6 s -> 20-30 s at 1.9-2.8 GB/s, startup
136.5-147.5 s -> 65-85 s, i.e. the coordinator's 65-80 s target band. No GPU arm
was run by B4 (the GPU was held by the peer window); the verification plan with
the coordinator's A-arm numbers is in section 3.

## 1. Where the 51.46 GiB goes (retained24 evidence)

### 1.1 Phase timeline

Source: `tmp/q4e-strata-c7/b4-loadtiming2/worker.log` (AILA_Q4E_LOAD_TIMING=1,
no requests), coordinator-measured startup 136.5 s; A/A reference arms
`b4-base-a1`/`b4-base-a2` = 147.5 / 146.5 s (production profile env, no
LOAD_TIMING).

| Phase | Time | Notes |
|---|---:|---|
| weights+embedding prep | 6.812 s | safetensors refs only, no payload reads |
| adaptive_cache Allocate+InitializeResidency | 0.123 s | bookkeeping only |
| host_arena.Allocate | 3.533 s | 51.46 GiB native large pages (VirtualAlloc + OS zeroing) |
| host_arena.LoadFromModelWeights | 121.275 s | see 1.2 |
| FinishInitialLoad + runtime buffers | 0.026 s | -- |
| launch/handshake remainder | ~4.7 s | 136.5 - sum of the above `[estimate]` |
| **measured startup total** | **136.5 s** | 147.5 / 146.5 s without LOAD_TIMING |

The non-fill part of the whole startup is ~30 s `[estimate]` (6.8 s prep +
3.5 s alloc + 25.7 s inside LoadFromModelWeights + 4.7 s launch), matching the
coordinator's "rest ~30 s (static weight upload / PLE init etc.)".

### 1.2 LoadFromModelWeights per-layer window (48 layers, serial)

| Item | Value | Source |
|---|---:|---|
| layer total (layers 0/8/16/24/32/40/47) | 2.447-2.671 s | `b4-loadtiming2/worker.log` |
| weight refs per layer | 0.000-0.002 s | same (no payload IO) |
| arena RAM copies (fill_range only) | 95.6 s total, 1992.0 ms/layer | same, `arena RAM copies` line |
| GPU-owned staging copies + H2D + waits + metadata | ~25.7 s total (121.275 - 95.6) | same; C7 cross-check 121.241 - 95.7 = 25.5 s in `tmp/q4e-strata-c7/t-diag2-worker.log` |
| per-layer working-set trim | `SetProcessWorkingSetSize(-1,-1)` after each layer | `Qwen4ExpHostArena.cpp:295` |

Effective fill rate: 51.46 GiB (55.25 GB) written in 95.6 s = **578 MB/s**,
while the same file can be read at 2.8 GB/s sequential (single file) and the
resident-page copy runs at 6.7 GB/s (C7 doc section 1.1). The gap is demand
faulting, not the disk and not `memcpy`.

### 1.3 Byte account (why 51.46 GiB arena reads 58.0 GiB of source)

Slot geometry (block-64 compact NF4, one expert):

| Segment | Bytes | Notes |
|---|---:|---|
| gate_up packed | 1,638,400 | 2*640 x 2560 / 2 |
| gate_up absmax codes | 51,200 | 2*640 x 2560 / 64 |
| down packed | 819,200 | 2560 x 640 / 2 |
| down absmax codes | 25,600 | 2560 x 640 / 64 |
| **slot total** | **2,534,400** (2.417 MiB) | `Qwen4ExpHostArena.cpp:87-97` geometry |

- 48 layers x 512 experts = 24,576 slots = 58.0 GiB of source bytes.
- 21,784 RAM-resident experts x 2,534,400 B = **51.46 GiB** arena fill
  (`Loading native RAM complement: 21784 experts, 51.46 GiB; GPU-only
  experts=2792`, `b4-loadtiming2/worker.log`).
- 2,792 GPU-only experts x 2,534,400 B = 6.59 GiB uploaded H2D from the same
  source ranges (its staging `memcpy` also faults the same pages).
- Per layer: 512 slots = 1,237.5 MiB source (gate_up packed 800 MiB, gate_up
  absmax 25 MiB, down packed 400 MiB, down absmax 12.5 MiB).

### 1.4 Pointer provenance (which pages are faultable, which are not)

| Storage | Allocator | Faults? |
|---|---|---|
| `gu/dn.packed_weight.ptr`, `gu/dn.absmax.ptr` | SafeTensors mmap `host_view` (`src/utils/SafeTensors.cpp:83-96, 898-930`) | **yes, the fill bottleneck** |
| `gu/dn.nested_absmax`, `nested_quant_map` (copied into `layer_nested_`) | same mmap, ~0.6 MiB/layer | yes, small |
| arena chunks (`chunks_[]`) | `ctx_->alloc_large_host` / `memory::AllocateLargePages` (`Qwen4ExpHostArena.cpp:112-117`) | no, committed large pages |
| `load_staging_` | `ctx_->alloc_host` (pinned) (`:154`) | no |
| `layer_nested_dev_allocs_` | device allocation (`:279-281`) | n/a |

So the prefetchable window is exactly the four contiguous mmap spans per layer;
the destination is already resident, and the local access order inside every
range is sequential (`fill_range` walks expert-by-expert, `LoadLayerDirect`).
Queuing those spans before the copies lets the cache manager run the reads with
real depth, which is what `PrefetchVirtualMemory` did for the PLE gather
(12.18x, `docs/plans/qwen38-r1-ple-firstchunk-2026-10-15.md`).

### 1.5 Bandwidth references (measured)

| Path | Rate | Source |
|---|---:|---|
| mmap demand faults, one file section (any thread count) | 0.66 GB/s | C7 doc section 1.1; `pf16-worker.log` (source prefetch 62.4 s + copies 8.6 s) |
| resident-page `memcpy` into large pages | 6.7 GB/s | C7 doc section 1.1 (8.6 s for 51.46 GiB) |
| file sequential read, 1 thread / same file 16 threads / many shards | 2.8 / 3.2 / 8.6-9.8 GB/s | C7 doc sections 1.1-1.2 + coordinator note |
| PLE F1 `PrefetchVirtualMemory` on cold random rows | 12.18x (186.9 -> 15.3 us/access) | `qwen38-r1-ple-firstchunk-2026-10-15.md` |

## 2. Change (file:line)

All edits are in `src/models/Qwen4ExpHostArena.cpp`; the header is unchanged.

1. `:16-44` - file-local helper `prefetch_source_range(const uint8_t* base,
   size_t bytes)` (anonymous namespace, `_WIN32` only, no-op elsewhere):
   splits a contiguous range into 1 MiB chunks and calls
   `PrefetchVirtualMemory(GetCurrentProcess(), count, ranges, 0)` in batches of
   at most 64 entries (the documented API limit, same as the PLE F1 pattern).
   Return value deliberately ignored: the hint must never fail the load.
2. `:314-318` - `static const bool s_host_arena_prefetch =
   aila::env::read_flag("AILA_Q4E_HOST_ARENA_PREFETCH", true);` (read-once
   static, house style). Default 1 = hint enabled; `=0` = historical path.
3. `:380-382, :409-428` - `mmap_sources` flag: when the C7 explicit-read path
   (`AILA_Q4E_ARENA_READ_THREADS>0`) succeeds, the sources are the bounce
   buffer and the flag turns false (prefetch skipped, otherwise it would only
   add disk work); on a failed explicit read the fallback keeps the flag true.
   The explicit-read timing branch is unchanged in behaviour (now inside the
   success `else`).
4. `:429-449` - call site inside the layer loop, immediately before
   `LoadLayerDirect` consumes the ranges:
   `prefetch_source_range` on `gu_src.packed_weight`, `gu_src.absmax`,
   `dn_src.packed_weight`, `dn_src.absmax`. Under `AILA_Q4E_LOAD_TIMING=1` it
   accumulates and prints `[Q4E LoadTiming] host arena prefetch (PFV): X s
   total` at layer 47 (call/enqueue time only).

Untouched (kept exactly as C7 left them): `AILA_Q4E_ARENA_READ_THREADS`,
`AILA_Q4E_ARENA_FILL_PREFETCH`, `AILA_Q4E_ARENA_FILL_THREADS`,
`AILA_Q4E_HOST_ARENA_PINNED`, the fill/staging data path, and every other file.

Semantics: the hint only changes when the OS decides to fetch pages, never
their content. Copies remain byte-identical; a failed hint silently degrades to
today's demand-fault behaviour.

## 3. Verification plan and gates

### 3.1 A arm (coordinator-measured, `retained24`, no requests)

| Arm | Startup | Fill | Log |
|---|---:|---:|---|
| `b4-base-a1` | 147.5 s | -- | min avail 3.48 GiB (coordinator) |
| `b4-base-a2` | 146.5 s | -- | A/A delta 0.7% |
| `b4-loadtiming2` | 136.5 s | `arena RAM copies` 95.6 s (1992.0 ms/layer) | `tmp/q4e-strata-c7/b4-loadtiming2/worker.log` |

Cache-state discrimination (required by the sprint rules): the two back-to-back
A arms differ by 0.7%, so on this box consecutive startup arms are both
effectively **cold** - each run writes 51.46 GiB of arena and trims the process
working set, which flushes the model pages out of standby. Practical detector:
if an A arm ever reports `arena RAM copies` far below 95.6 s, that run was warm
and must be discarded or re-run after eviction (section 3.4).

### 3.2 B arm commands (startup_only form, same as the A arms)

```
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # GPU mutex: must be empty
python tmp/q4e-strata-c7/startup_only.py b4-pfv-b1 AILA_Q4E_LOAD_TIMING=1 AILA_Q4E_HOST_ARENA_PREFETCH=1 AILA_SERVER_WORKER_LOG=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/startup/b4-pfv-b1-worker.log
python tmp/q4e-strata-c7/startup_only.py b4-pfv-b2 AILA_Q4E_LOAD_TIMING=1 AILA_Q4E_HOST_ARENA_PREFETCH=1 AILA_SERVER_WORKER_LOG=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/startup/b4-pfv-b2-worker.log
python tmp/q4e-strata-c7/startup_only.py b4-pfv-a3 AILA_Q4E_LOAD_TIMING=1 AILA_Q4E_HOST_ARENA_PREFETCH=0 AILA_SERVER_WORKER_LOG=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/startup/b4-pfv-a3-worker.log
```

(A1 is the existing `b4-base-a2`; A2 is `b4-pfv-a3`; that closes A/B/B/A
without re-running the 2.5-minute A base. Optional automation:
`python tmp/q4e-sprint-r1-20261014/startup/abba_startup.py --label b4-hot`
runs all four arms with the same launcher, parses the logs, and writes
`startup/receipt-b4-hot.json`; it refuses to start while any
AilaWorker/AilaServer is alive.)

Fields to read per arm (worker log / runner output):

- `[Q4E LoadTiming] arena RAM copies: X s total (Y ms/layer)` - primary metric;
- `[Q4E LoadTiming] host arena prefetch (PFV): X s total` - must appear on B
  arms only (presence check of the new code path);
- `[Q4E LoadTiming] host_arena.LoadFromModelWeights: X s`;
- startup seconds printed by `startup_only.py`; min available / worker RSS.

### 3.3 Gates (pair `retained24`, cold arms)

| Verdict | Condition (mean of B1/B2 vs 95.6 s fill / 136.5-147.5 s startup) |
|---|---|
| GO (default stays 1) | fill <= 0.50x (<= 47.8 s) AND startup <= 115 s AND no failures AND PFV line on both B arms |
| Strong GO | fill <= 30 s (>= 3.2x) AND startup <= 90 s; stretch target 65-80 s |
| NO-GO / flip default to 0 | fill ratio > 0.85, OR any startup regression > 2% vs the A band, OR any new failure |

Failure tokens scanned: `UR_OUT_OF_RESOURCES`, `allocation failed`,
`MEM_LARGE_PAGES`. A/A noise on this metric is ~1% (fill) and 0.7% (startup);
anything under 3% count change is a no-op.

Do not stack if PFV already lands <= 30 s: the C7 explicit-read path
(`AILA_Q4E_ARENA_READ_THREADS=16`, 116.6 -> 63.8 s cold measured) is the
fallback composition only when PFV is insufficient (fill > 40 s), because the
two solve the same bottleneck by different means.

Memory sanity: min available must stay >= 2.98 GiB (a1's 3.48 GiB - 0.5) and
worker RSS within the known 5.7-6.4 GiB band; the hint must not increase peak
residency (same pages are demanded either way).

### 3.4 Cache-state control (optional)

If a genuinely warm state must be forced for the regression half of the gate
(e.g. a machine that keeps the model resident), `abba_startup.py --evict-gib 48`
purges the standby list (`NtSetSystemInformation(SystemMemoryListInformation,
MemoryPurgeStandbyList)`, admin) or falls back to a dirty 48 GiB allocation,
before every arm. B4 did not execute this (no exclusive GPU window and the
peer window's arms were live); it is static-checked only (`py_compile` + parser
replay on `t-diag2` / `pf16` logs).

## 4. Risks and rollback

1. **Warm-cache overhead.** On a machine that really keeps the 58 GiB resident,
   the hint adds a residency walk over ~15.2M pages. Expected +1-3 s worst case
   `[estimate]`; this is the only reason the +/-2% startup gate exists. Rollback
   is `AILA_Q4E_HOST_ARENA_PREFETCH=0` (no rebuild), or flip the default.
2. **Hint effectiveness on one large section is not guaranteed.** If the cache
   manager serializes the queued reads like it serializes faults, the ratio
   lands near 1.0 -> NO-GO, compose with the explicit-read opt-in instead.
   The PLE result (12.18x) and the same-file explicit read (3.2 GB/s) bound the
   expectation to roughly 1.5-3 GB/s `[estimate]`.
3. **Chunk size.** 1 MiB chunks x 64 per call = 64 MiB per syscall, ~19 calls
   per layer, ~912 calls per startup `[estimate]`; if the measured win is
   marginal, try 4 MiB chunks (fewer calls) before concluding NO-GO.
4. **Combination with C7 opt-ins.** `READ_THREADS>0` skips the hint by design
   (sources are a bounce buffer). `FILL_PREFETCH>0` touch threads run after the
   hint and merely wait on the queued pages; the combination is untested and
   not part of the gate - set one mechanism at a time.
5. **Startup remains GPU-window exclusive.** Two workers cannot coexist
   (`MEM_LARGE_PAGES ... win32_error=1450`), and the hint does not change that.
6. **Scope-limited risk.** No semantic, ABI, or byte-level change; failures are
   silent; header untouched, so no consumer rebuild coupling.

## 5. Open items / TODO

- TODO(measure): PFV enqueue time per layer (logged); expected << 1 s/layer
  since the call only queues work.
- TODO(design, only if PFV is insufficient): prefetch layer L+1 during layer
  L's GPU-owned staging loop (~25.7 s of host-side time per startup exists
  there `[estimate]`), or raise chunk size to 4 MiB. Do not build either
  before the first B-arm result.
- TODO(verify): B4 did not compile the TU (sprint rule "no build"); the change
  must be built by the coordinator before the B arms. Static review: helper
  mirrors the compiled PLE/ExpertExecutor `PrefetchVirtualMemory` patterns;
  `HostTensorView::ptr` is `const uint8_t*` (`src/utils/SafeTensors.hpp:64`).
- Receipt to write after the arms: `tmp/q4e-sprint-r1-20261014/startup/`
  (worker logs + `receipt-b4-*.json`), referenced from this document.

## 6. Round 2 (C3, retained25): hint overlap, slice knob, residual split

Author: C3 implementation subagent. Date: 2026-10-15. HEAD `fe1bf9e`
(pair `retained25`). Scope: source edit confined to
`src/models/Qwen4ExpHostArena.cpp` (`.hpp` unchanged); this section appended to
the B4 document; runner extended in place at
`tmp/q4e-sprint-r1-20261014/startup/abba_startup.py`. No build, no GPU process,
no git write (sprint rules). Measured values cite an artifact; derived values
are tagged `[estimate]` where they are predictions.

### 6.1 Baseline of record and the residual before the arms

| Item | Value | Source |
|---|---:|---|
| fill (`arena RAM copies`) | 26.9 s total, 561.1 ms/layer | `tmp/q4e-strata-c7/b4-pf-on-a1/worker.log` |
| PFV enqueue (`host arena prefetch (PFV)`) | 22.4 s total (467 ms/layer `[estimate]`) | same log, 22.4/48 layers |
| `host_arena.LoadFromModelWeights` | 63.538 s | same log |
| residual after PFV + fill | 14.2 s total, ~296 ms/layer `[estimate]` | 63.538 - 22.4 - 26.9; layer 8 line 1.324 s - 0.467 - 0.561 |
| layer totals (8/16/24/32/40, 47) | 1.276-1.375 s (layer 0: 1.718 s) | same log |
| worker startup (this pair) | 78.6 s median (A/B/B/A 4/4) | `tmp/q4e-sprint-r1-20261014/receipt-startup-prefetch-abba.json` |
| GPU-only experts | 2,792 (58.2/layer, 147.5 MB/layer) | `b4-pf-on-a1/worker.log` + geometry section 1.3 |

Round-2 gates (sprint brief, ratios against the values above): GO = fill
<= 0.50x (<= 13.5 s) AND startup <= 115 s; revert = fill > 0.85x (> 22.9 s) or
startup regression > 2%; soft target ~19 s (0.71x).

The 14.2 s residual was never split. The candidate sites, all inside
`LoadLayerDirect` after the RAM fill (`Qwen4ExpHostArena.cpp:272-332`):

| Site | Pre-arms expectation | Evidence |
|---|---:|---|
| GPU-owned staging: wait-for-previous-H2D + 4 staging memcpy + enqueue, 58.2 experts/layer | [estimate] 150-250 ms/layer (7-12 s) | C7 doc section 1: ~24 s in the pre-PFV fault regime, 0.5 s/layer; source pages are now hinted |
| nested metadata (0.6 MiB assign) + device upload + `Queue().wait_and_throw()` | [estimate] 10-40 ms/layer | code only, never measured |
| per-layer `SetProcessWorkingSetSize(-1,-1)` trim | unknown, [estimate] 20-80 ms/layer | `Qwen4ExpHostArena.cpp:295` (old numbering) |
| CPU-side per-layer overhead (refs are 0.002 s) | ~2 ms/layer | `b4-pf-on-a1/worker.log` |

Code-level check of the 22.4 s PFV number (the brief asked whether it is
enqueue cost or wait): `prefetch_source_range` (`:26-42`) only fills a 64-entry
`WIN32_MEMORY_RANGE_ENTRY` array and calls `PrefetchVirtualMemory` on the
calling thread. Per layer that is 22 calls (`[estimate]`: 13 for the 800 MiB
gate_up packed range, 7 for the 400 MiB down packed range, 1+1 for the two
absmax ranges), so 467 ms/layer is ~21 ms per 64 MiB call (~3 GB/s of batch
issue). User-space list building costs microseconds; the 21 ms is kernel-side
PTE walk plus read queuing/servicing inside the synchronous syscall. The async
mechanism therefore does not reduce the I/O - it moves that main-thread
kernel time into a helper so it overlaps the previous layer's copies.

### 6.2 Changes (all in `src/models/Qwen4ExpHostArena.cpp`)

1. **Async one-layer-lookahead hint** - `AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC`
   (default 0). With the flag on (and `AILA_Q4E_HOST_ARENA_PREFETCH=1`,
   `AILA_Q4E_ARENA_READ_THREADS=0`), the layer L+1 hint is issued on a helper
   thread right where the sync hint for L used to run, and joined just after
   layer L+1's refs resolve (before its sources are consumed). Layer 0 keeps
   the inline hint, so at most one helper is ever alive and the join bounds the
   overlap to one layer. The helper only calls `PrefetchVirtualMemory`
   (advisory); copy order/bytes/visibility are untouched, and a joinable
   helper cannot escape on exceptions (local guard, `:455-458`).
   Inactive unless explicitly enabled: with the flag unset the code visits the
   identical sync path (`:556-575`).
2. **PFV slice knob** - `AILA_Q4E_HOST_ARENA_PREFETCH_CHUNK_KB`
   (default 1024, clamp 64..16384). Applies to both the sync and async hint;
   64 entries per call stays fixed (API limit). 1024 KiB reproduces the
   fe1bf9e call sequence exactly; 256 / 4096 are the screened alternatives
   (22 calls/layer at 1 MiB, 8 at 4 MiB, 78 at 256 KiB `[estimate]`).
3. **Residual split under `AILA_Q4E_LOAD_TIMING`** (`:258-352`): new print-once
   line `arena residual split: stage memcpy X s, stage wait Y s, stage enqueue
   Z s (N experts), nested+device A s, device drain B s, working-set trim C s`,
   plus the async line `host arena prefetch (PFV async chunk N KiB): inline X s,
   issue Y s, join Z s total` (`:660-665`). When the gate is off there are no
   clock reads (one bool branch per staging expert).

Line map: helper `:26-45`; residual block `:258-352`; knobs `:375-390`;
`resolve_layer_refs` `:438-451`; helper plumbing `:452-491`; loop join
`:497-499`; sync hint `:556-575`; async hint/scheduling `:576-601`; async
summary line `:660-665`.

Rollback: unset the two new envs (or `=0` / `=1024`) reproduces the fe1bf9e
path bit for bit; `AILA_Q4E_HOST_ARENA_PREFETCH=0` disables every hint.

### 6.3 Expected effect and the physical floors [estimate]

- Read floor: the fill consumes 58.0 GiB of source. At the best measured
  single-file rate (3.2 GB/s, 16 threads explicit; C7 doc section 1) that is
  **19.4 s**; at the 1-thread rate (1.97-2.19 GB/s) 26-30 s. The 13.5 s gate
  needs >= 4.6 GB/s sustained on one file, i.e. above every measured
  single-file number - the ~19 s sprint target is exactly this 3.2 GB/s floor.
- Copy floor: resident pages into large pages measured 8.6 s for the same
  51.46 GiB (6.7 GB/s, C7 doc section 1.1); current fill is 3.1x above it.
- If the 22.4 s PFV stall is mostly manager-wait, overlap hides nearly all of
  it: fill -> max(19.4 s read, 8.6 s copy + staging contention) ~~ **19-24 s**
  (ratio 0.71-0.89, layer 0.40-0.50 s). If the manager serializes the helper
  against the copy faults, the win approaches 0 and the ratio lands near 1.0.
  The revert line (0.85x) sits inside the expected band - this arm can go
  either way and the receipt decides.
- Slice knob: if the 467 ms/layer call cost has a per-call component, 4 MiB
  chunks cut 22 calls to 8; if the cost is I/O-proportional, no change. Screen
  after, not together with, the async arm (section 6.4).

### 6.4 Coordinator commands, fields read, verdicts

GPU-exclusive window required; every arm runs with `AILA_Q4E_LOAD_TIMING=1`
(set by the runner) and refuses to start while any AilaWorker/AilaServer lives.
All four arms A/B/B/A; A = current default (PFV 1 MiB sync), B = the new
mechanism.

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # GPU mutex: must be empty

# C1 (primary): async overlap vs current default, cold arms
python tmp/q4e-sprint-r1-20261014/startup/abba_startup.py --label r25-async --evict-gib 48 `
  --extra AILA_Q4E_HOST_ARENA_PREFETCH=1 --b-extra AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC=1

# C2 (only if C1 is below gate or join is large): slice screen, one at a time
python tmp/q4e-sprint-r1-20261014/startup/abba_startup.py --label r25-chunk4m --evict-gib 48 `
  --extra AILA_Q4E_HOST_ARENA_PREFETCH=1 --b-extra AILA_Q4E_HOST_ARENA_PREFETCH_CHUNK_KB=4096
python tmp/q4e-sprint-r1-20261014/startup/abba_startup.py --label r25-async4m --evict-gib 48 `
  --extra AILA_Q4E_HOST_ARENA_PREFETCH=1 --b-extra AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC=1 `
  --b-extra AILA_Q4E_HOST_ARENA_PREFETCH_CHUNK_KB=4096

# C3 (hot no-regression half of the gate; B4 section 3.4 discrimination)
python tmp/q4e-sprint-r1-20261014/startup/abba_startup.py --label r25-hot `
  --extra AILA_Q4E_HOST_ARENA_PREFETCH=1 --b-extra AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC=1
```

Single-arm fallback in the `startup_only.py` shape (when only one datapoint is
needed, e.g. confirming a log line):

```powershell
python tmp/q4e-strata-c7/startup_only.py r25-async-single AILA_Q4E_LOAD_TIMING=1 `
  AILA_Q4E_HOST_ARENA_PREFETCH=1 AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC=1 `
  AILA_SERVER_WORKER_LOG=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/startup/r25-async-single-worker.log
```

Fields to read per arm (`startup/<label>-<arm>/worker.log`, receipt
`startup/receipt-<label>.json`):

- `arena RAM copies: X s total (Y ms/layer)` - the gate metric "fill".
- `arena residual split: ...` - stage memcpy / stage wait / stage enqueue /
  nested+device / device drain / working-set trim (new attribution).
- A arms: `host arena prefetch (PFV): X s total`. B arms:
  `host arena prefetch (PFV async chunk N KiB): inline X s, issue Y s, join Z s`.
- `host_arena.LoadFromModelWeights: X s`; `arena layer N:` totals; startup
  seconds; `min_available_gib` (>= 2.98 GiB) and worker RSS (5.7-6.4 GiB).
- Receipt `summary.r25` (fill_ratio_b_over_a, startup_b_mean_s, gate_go,
  gate_revert) and `summary.failures` (must be empty: `UR_OUT_OF_RESOURCES`,
  `allocation failed`, `MEM_LARGE_PAGES*`).

Verdicts (cold receipt; ratios B/A over the two pairs):

| Verdict | Condition |
|---|---|
| GO (eligible to flip the default on) | fill_ratio <= 0.50 (fill_B <= 13.5 s vs A mean 26.9 s) AND startup_B_mean <= 115 s AND no failures AND PFV/async line present on both B arms |
| Improved, below gate (keep opt-in, NO default flip) | 0.50 < fill_ratio <= 0.85 |
| Revert (drop the mechanism) | fill_ratio > 0.85 OR startup_ratio > 1.02 OR any failure |
| Hot check | `--label r25-hot` startup_ratio <= 1.02 AND fill_ratio <= 1.02 |

Pre-registered reading of the new fields:

- `join` > ~5 s total: the helper trails the layer window (manager-limited) ->
  slice knob (C2) is the next lever; large `stage memcpy` (>= 0.2 s/layer,
  i.e. 147.5 MB at the 0.66 GB/s fault rate, C7 section 1) means the hinted
  pages were not resident yet.
- `stage wait` large: the C7 "GPU upload pipelining" follow-up has headroom.
- `working-set trim` large: candidate for a separate opt-in trim-every-N (new
  env, not in this round).
- Warm runs show up as a materially faster second consecutive same-config arm
  (receipt `noise.a2_over_a1_startup` / `b2_over_b1_startup`, B4 section 3.4);
  discard or re-run with `--evict-gib 48`.

### 6.5 Risks, rollback, TODO

1. **Trim vs lookahead ordering.** The per-layer working-set trim runs at the
   end of `LoadLayerDirect(L)`, i.e. after L+1's hint was queued; prefetched
   but unread pages can be pushed to standby and re-faulted softly during L+1.
   If `join` ~ 0 while fill does not drop, this is the first suspect. Follow-up
   options (not in this round): move the hint issue after the trim (needs a hook
   inside `LoadLayerDirect`) or an opt-in trim-every-N env.
2. **Contention.** The helper shares the cache manager with the copy faults of
   the current layer; if the manager serializes them, the arm measures ~0 win.
   That is the point of the A/B/B/A, not a reason to skip it.
3. **Combinations are out of scope.** ASYNC=1 with `FILL_PREFETCH>0`,
   `FILL_THREADS>1`, or `READ_THREADS>0` is untested (the latter disables
   async by construction). One mechanism at a time, as in B4 section 4.
4. **Error surface moved one layer earlier** in async mode: a corrupt/missing
   tensor in layer L+1 now fails during layer L with the identical error
   string from `LoadBnb4BitHostRef` (refs resolve one iteration early,
   `:597-600`). The load still fails loudly; only the layer index in the log
   can shift.
5. **No build/GPU run by C3** (sprint rules: peer window held the GPU). The
   coordinator must build before C1. Static review only: helper mirrors the
   compiled F1 pattern; `HostTensorView::ptr` is `const uint8_t*` and `bytes`
   is `size_t` (`src/utils/SafeTensors.hpp:64,67`); the new C++ surface is
   `std::thread` + a local RAII guard (already included, `:10`).
6. **Receipts.** C3 wrote no receipts (no GPU). Expected outputs:
   `tmp/q4e-sprint-r1-20261014/startup/receipt-r25-async.json` (+ optional
   `-chunk4m`, `-async4m`, `-hot`) and their per-arm worker logs.
7. **Peak residency.** The lookahead keeps one extra layer of hinted pages in
   flight (~1.2 GiB) relative to the sync schedule; the per-layer trim still
   bounds the working set. Watch the receipt's `min_available_gib` (gate:
   >= 2.98 GiB, B4 section 3.3) and worker RSS (5.7-6.4 GiB band).

## 7. Round 3 (D2, retained26): trim audit + fill-lookahead variant

Author: D2 implementation subagent. Date: 2026-10-15. HEAD `58f7874`
(pair `retained26`). Scope: source edit confined to
`src/models/Qwen4ExpHostArena.cpp` (`.hpp` unchanged, no signature changes);
runner `tmp/q4e-sprint-r1-20261014/startup/abba_startup.py` extended in place
(additive parse/gate block only); this section appended to the B4/C3 document.
No build, no GPU process, no git write (sprint rules; the GPU was held by the
`d0-deferwait-b1` arm). Measured values cite an artifact; predictions are
tagged `[estimate]`.

Two deliverables: (A) the per-layer `SetProcessWorkingSetSize` trim audited
read-only behind a new default-on gate, (B) one opt-in fill variant
(`AILA_Q4E_HOST_ARENA_PREFETCH_LOOKAHEAD=2`). Both new switches leave the
default path semantically identical to `58f7874`; both are flip candidates
that only a receipt may promote.

### 7.1 Baseline of record (retained26 default path)

| Item | Value | Source |
|---|---:|---|
| worker startup | 58.5 s | cost model section 6.21 (A/B/A 79.1/58.5/78.6 s); sprint brief |
| fill (`arena RAM copies`) | 29.0 s total, 603.9 ms/layer | `tmp/q4e-sprint-r1-20261014/startup/r25-async2-b1/worker.log` (b2: 28.9 s) |
| working-set trim | 9.8 s total (~204 ms/layer) | same log, residual split line |
| stage memcpy / wait / enqueue | 3.0 / 0.9 / 0.1 s (2792 experts) | same log |
| nested+device / device drain | 0.2 / 0.0 s | same log |
| `host_arena.LoadFromModelWeights` | 43.6 s | same log |
| layer totals (8/16/24/32/40/47) | 0.857-0.905 s (layer 0: 1.697 s) | same log |
| PFV async (chunk 1024 KiB) | inline 0.4 s, issue 0.0 s, join 0.0 s | same log |
| RAM complement / GPU-only experts | 21784 / 2792 (51.46 GiB arena) | same log (geometry: section 1.3) |
| sync-hint control (same pair) | fill 26.9 s, PFV 22.6 s, load 63.6 s | `r25-async2-a1/worker.log` |

Round-3 gate (sprint brief): fill <= 0.83x of 29.0 s = **24.0 s** AND startup
regression <= 2% (58.5 -> 59.67 s). The 9.8 s trim is the largest removable
load item (cost model section 6.21); the 29.0 s fill runs at ~1.9-2.0 GB/s
against the 2.8 GB/s single-file ceiling (section 1.5).

### 7.2 Trim audit: history, mechanism, and the new read-only evidence

**History (why the trim exists).** The trim was introduced with the large-page
resident arena in `a83cd80` (2026-10-09): "Release consumed mmap source pages
after each layer, before they compete with the next layer." Rationale in
`docs/plans/qwen38-large-page-residency-2026-10-08.md`:

- Without it, one measured load stalled with available physical memory at
  ~0.17 GiB for ~80 s and tripped the Server's 120 s control-RPC deadline; the
  per-layer release was added together with the separate 10-minute load
  deadline (same document, the withdrawn-attempts/allocation-fix section).
  The B8 run with the trim reported startup 79.17 s and min available
  4.00 GiB.
- The trim cannot evict the arena: native large pages are outside the process
  working set (same document, line 22), so `SetProcessWorkingSetSize(-1,-1)`
  only makes trimmable (mostly clean, file-backed mmap source) pages
  reclaimable.

**Mechanism at the code level.** `LoadLayerDirect` ends with
`SetProcessWorkingSetSize(GetCurrentProcess(), -1, -1)`; consumed source pages
leave the working set (standby), so the next layer's reads and the arena's
non-pageable pages do not compete with stale active pages. Documented cost:
9.8 s/startup. Documented interaction (section 6.5 risk 1): the trim runs
after layer L+1's hint was queued, so prefetched-but-unconsumed pages can be
pushed to standby and re-faulted (soft) during L+1. Whether that interaction
taxes the fill is exactly what the audit decides.

**New read-only diagnostics (only under `AILA_Q4E_LOAD_TIMING=1`).** All in the
worker log, no state change:

| Line / field | Meaning |
|---|---|
| `trim audit: trim_on=%d layers=.. ws_before_avg=.. ws_after_avg=.. ws_drop_avg=.. drop_min=.. drop_max=.. private_after_avg=.. faults_avg=..` | `GetProcessMemoryInfo` samples immediately before/after the trim point (MiB); `faults_avg` is `PageFaultCount` across each layer's fill+staging window (entry sample to trim point) |
| `fill ms/layer: 0=.. 1=.. ... 47=..` | per-layer RAM-copy ms; `fill[L+1]` is exactly the fill that ran right after layer L's trim |

The same lines are emitted with the trim disabled (`trim_on=0`), so the two
arms compare field by field. The worker-side numbers complement the runner's
system-wide `min_available_gib` / worker RSS monitors; none of them alone is a
hard/soft-fault split (`PageFaultCount` is a sum -> TODO below).

**Pre-registered reading (before the arms).** Per layer the fill consumes
1,237.5 MiB of source (section 1.3) = ~317k pages of 4 KiB:

| Signal | H-A: trim is load-bearing for memory | H-B: trim taxes the fill |
|---|---|---|
| `trim=1` `ws_drop_avg` | ~1.1-1.3 GiB/layer (the consumed range) | same |
| `trim=1` `ws_after_avg` | small (<= ~200 MiB) [estimate] | same |
| `trim=1` `faults_avg` | `<< 3.17e5` (hinted pages stay resident until used) | ~3.17e5/layer (every page faulted on the fill side even though the hint read it) |
| `trim=0` `ws_before/after_avg` | grows ~1.2 GiB/layer until pressure; `min_available_gib` falls toward the 2.98 GiB gate | same, but fill drops materially in the same arm |
| `trim=0` `min_available_gib` | falls below 2.98 GiB or stalls (2026-10-08 signature) | stays >= 2.98 GiB |

**Default policy (D2 does not flip anything; no commit in this round).**

- The gate ships with default 1 so the round-3 A arms reproduce `58f7874`.
- A later commit may flip the default to 0 only if the `r26-notrim` receipt
  shows, on cold arms: fill_ratio <= 0.83, startup_ratio <= 0.85 (the 9.8 s
  trim alone is 0.17x of startup `[estimate]`), `min_available_gib` >= 2.98 GiB
  in every arm, no failure tokens, and no available-memory stall in the
  runner's 0.5 s monitor.
- Otherwise the default stays 1, and the 2026-10-08 evidence above is the
  stated basis. Fallback line for users either way:
  `AILA_Q4E_ARENA_WORKING_SET_TRIM=1` (default) or `=0` (audit/lean arm).

### 7.3 Fill variant: two-slot hint lookahead (`AILA_Q4E_HOST_ARENA_PREFETCH_LOOKAHEAD=2`)

**Mechanism.** The round-2 async hint keeps exactly one helper reading layer
L+1 while layer L is filled (join at the top of L+1, measured join 0.0 s and
fill 29.0 s). The variant adds a second hint slot so the schedule keeps two
targets in flight: during layer L the helpers read L+1 and L+2; the slot
targeting L is joined at the top of layer L and immediately re-armed for L+2.
Lookahead stays 2 (clamped to 1..2); each helper only calls
`PrefetchVirtualMemory` on the four source ranges (advisory, failures
ignored, data path byte-identical). Unset or `0`/`1` runs the historical
single-slot schedule: same refs resolution, same thread body, same join
points as `58f7874`.

**Why this lever.** The sync-arm PFV time (22.6 s for 58.0 GiB of hints) sits
at the measured 2.8 GB/s single-file sequential rate, i.e. one prefetch stream
already runs at the single-reader ceiling; the fill at 603.9 ms/layer then
demand-faults whatever the hint has not delivered. Multi-stream reads on this
box measured 8.6-9.8 GB/s across shards and 3.2 GB/s same-file/16-thread
(section 1.5), so a second concurrent hint stream is the remaining lever that
does not change the copy path:

| Quantity | Value |
|---|---:|
| read floor, 1 stream at 2.8 GB/s | 22.3 s for 62.3 GB (58.0 GiB) |
| read floor, same-file 16-thread at 3.2 GB/s | 19.4 s |
| copy floor (resident pages into large pages, section 1.5) | 8.6 s for 51.46 GiB |
| expected fill with two effective streams | 20-25 s `[estimate]`, ratio 0.69-0.86 |
| startup if fill lands 24.0 s and nothing else moves | ~53.5 s `[estimate]` (58.5 - 5.0) |

**Falsifiers, pre-registered.** If the cache manager serializes the two
helpers against the current layer's copy faults (the section 6.3 caveat), the
arm measures ~0 and the ratio lands near 1.0 -> NO-GO for this variant at
lookahead 2. If the trim arm (7.2) shows the trim is already discarding the
hinted pages, the lookahead's headroom shrinks by that amount; read the two
receipts together. If join time stays 0.0 s and fill is flat, the binding
constraint is the single read stream, not the lookahead lead.

### 7.4 Changes (file:line, default path unchanged)

All source edits are in `src/models/Qwen4ExpHostArena.cpp`; the header is
untouched. New env knobs (read-once statics, house style):

| Env | Default | `=0` semantics | Effect |
|---|---:|---|---|
| `AILA_Q4E_ARENA_WORKING_SET_TRIM` | 1 | skip the trim syscall | `=1` reproduces `58f7874` exactly; `=0` is the audit/lean arm |
| `AILA_Q4E_HOST_ARENA_PREFETCH_LOOKAHEAD` | 1 | (read_int maps <=0 to the default) | `=1` reproduces `58f7874` exactly; `=2` runs the two-slot window |

Line map: memory sampler `:48-72`; trim gate + layer-entry sample `:228-240`;
per-layer fill list `:284-311`; trim block `:379-395`; audit accumulation and
`trim audit`/`fill ms/layer` print-once lines `:396-446`; lookahead knob
`:485-491`; hint slots + RAII guard + join/launch plumbing `:555-601`; per-layer
join `:607-612`; lookahead scheduling `:707-729`; final joins and the async
summary line (now ending `..., lookahead %d`) `:789-797`.

Runner `tmp/q4e-sprint-r1-20261014/startup/abba_startup.py` (additive):
docstring r26 usage `:34-45`; trim-audit / per-layer-fill / lookahead parsing
`:240-270`; `summary.r26` gate block (fill <= 0.83 GO, fill > 1.0x or
startup > +2% revert) `:394-413`.

Semantics: the trim gate only skips a Windows working-set syscall; the
lookahead only changes how many advisory hint helpers exist. Copies keep the
same bytes and order; a failed hint silently degrades to demand faulting
(unchanged). Rollback without a rebuild: unset both envs (or set
`LOOKAHEAD=1` / `WORKING_SET_TRIM=1`).

### 7.5 Coordinator commands, fields, verdicts

GPU-exclusive window required; both runners refuse to start while any
AilaWorker/AilaServer lives. Every arm needs `AILA_Q4E_LOAD_TIMING=1` for the
new fields (`abba_startup.py` sets it; `startup_only.py` does not).

Primary automation, cold A/B/B/A with standby eviction (A = retained26
default; the `--extra` prefetch=1 is mandatory because the runner's hardcoded
A arms still default to prefetch 0):

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # GPU mutex: must be empty

# D1 (primary): fill variant, lookahead 2
python tmp/q4e-sprint-r1-20261014/startup/abba_startup.py --label r26-look2 --evict-gib 48 `
  --extra AILA_Q4E_HOST_ARENA_PREFETCH=1 --b-extra AILA_Q4E_HOST_ARENA_PREFETCH_LOOKAHEAD=2

# D2 (trim audit): trim off
python tmp/q4e-sprint-r1-20261014/startup/abba_startup.py --label r26-notrim --evict-gib 48 `
  --extra AILA_Q4E_HOST_ARENA_PREFETCH=1 --b-extra AILA_Q4E_ARENA_WORKING_SET_TRIM=0
```

Single-arm fallback in the `startup_only.py` shape (no eviction; one datapoint
per call; the log path is required so the artifacts land in the sprint dir):

```powershell
python tmp/q4e-strata-c7/startup_only.py r26-base-single AILA_Q4E_LOAD_TIMING=1 `
  AILA_Q4E_HOST_ARENA_PREFETCH=1 `
  AILA_SERVER_WORKER_LOG=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/startup/r26-base-single-worker.log
python tmp/q4e-strata-c7/startup_only.py r26-look2-single AILA_Q4E_LOAD_TIMING=1 `
  AILA_Q4E_HOST_ARENA_PREFETCH=1 AILA_Q4E_HOST_ARENA_PREFETCH_LOOKAHEAD=2 `
  AILA_SERVER_WORKER_LOG=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/startup/r26-look2-single-worker.log
python tmp/q4e-strata-c7/startup_only.py r26-notrim-single AILA_Q4E_LOAD_TIMING=1 `
  AILA_Q4E_HOST_ARENA_PREFETCH=1 AILA_Q4E_ARENA_WORKING_SET_TRIM=0 `
  AILA_SERVER_WORKER_LOG=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/startup/r26-notrim-single-worker.log
```

Fields to read per arm (`startup/<label>-<arm>/worker.log`, receipt
`startup/receipt-<label>.json`):

- `arena RAM copies: X s total (Y ms/layer)` - the gate metric "fill".
- `fill ms/layer: ...` - per-layer list; compare A vs B entry by entry
  (`fill[L+1]` follows layer L's trim in both arms).
- `trim audit: ...` - `trim_on`, `ws_before_avg`, `ws_after_avg`,
  `ws_drop_avg`, `drop_min/max`, `private_after_avg`, `faults_avg`.
- `arena residual split: ...` and `host arena prefetch (PFV async chunk
  N KiB): ..., lookahead N` (presence check for the variant value).
- startup seconds; `min_available_gib` (gate >= 2.98 GiB, section 3.3); worker
  RSS (5.7-6.4 GiB band); failure tokens (`UR_OUT_OF_RESOURCES`,
  `allocation failed`, `MEM_LARGE_PAGES*`).
- Receipt `summary.r26` (`fill_ratio_b_over_a`, `startup_ratio_b_over_a`,
  `gate_go`, `gate_revert`, `trim_on_b`, `lookahead_b`) and `noise.*`.

Verdicts (cold receipt; B mean vs A mean over the two pairs):

| Verdict | Condition |
|---|---|
| GO for the variant (eligible for a later default flip) | fill_ratio <= 0.83 AND startup_ratio <= 1.02 AND no failures AND `lookahead_b == [2, 2]` |
| Improved, below gate | 0.83 < fill_ratio <= 1.00 AND startup_ratio <= 1.02 -> keep opt-in, no flip |
| Revert | fill_ratio > 1.00 OR startup_ratio > 1.02 OR any failure |
| Trim default flip (separate, later commit) | trim=0 arm meets ALL of: fill_ratio <= 0.83, startup_ratio <= 0.85, min_available_gib >= 2.98 GiB in every arm, no failures, and the trim=1 arm's `ws_drop_avg` >= 1.0 GiB/layer (proves what the trim reclaims) |

### 7.6 Hot-state discrimination (required by the sprint rules)

- Consecutive same-config arms: if the second arm is materially faster
  (startup ratio < 0.95 or fill < 0.85x of the first), the run is warm ->
  discard it or re-run with `--evict-gib 48` (purges the standby list; falls
  back to a dirty 48 GiB allocation).
- Metric-level detector for this journey: an A arm with `arena RAM copies`
  materially below the 29.0 s band (< ~20 s) is warm; a fill near 26.9 s with
  a sync `PFV: 22.6 s total` line means the arm ran the sync path - keep
  `AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC` at its default 1 in every arm; never
  compare a sync-mode arm against the async baseline.
- Also compare `min_available_gib` and the `trim audit` fields across arms of
  the same config; divergence means the machine state, not the variant,
  moved the numbers.
- `startup_only.py` runs do not evict. Use them for presence checks and
  single-arm probes only; the ABBA receipts are the gate evidence.

### 7.7 Risks, rollback, TODO

1. **Default path.** Both knobs default to the `58f7874` behaviour; all new
   sampling sits behind `AILA_Q4E_LOAD_TIMING`, which was already opt-in.
   Rollback without a rebuild: unset the two envs.
2. **Lookahead 2 error surface.** Refs for L+2 resolve during layer L, so a
   corrupt/missing tensor fails one layer earlier than in section 6.5 risk 4
   (same error string from `LoadBnb4BitHostRef`; only the log layer index
   shifts). The load still fails loudly.
3. **Peak residency.** The extra slot keeps ~one more layer of hinted pages in
   flight (~1.2 GiB `[estimate]`); the per-layer trim still bounds the working
   set. Watch `min_available_gib` and RSS (7.5 gates).
4. **Trim-off memory risk.** The 2026-10-08 stall (0.17 GiB available for
   ~80 s) is the reason the default stays 1 until a receipt clears the 7.5
   trim-flip row. If the trim-off arm reproduces low available memory, keep
   the default and cite the arm in the follow-up decision.
5. **No build/GPU run by D2.** The coordinator must build before the arms.
   Static review only: the `psapi.h` usage mirrors the already-compiled
   `src/models/Qwen4ExpExpertExecutor.cpp:41-51` (`GetProcessMemoryInfo` via
   kernel32 K32 export, no new link input); `std::thread`/`std::clamp`/
   `<vector>` are already included; the depth-1 scheduling branch is the
   round-2 call sequence (verified by replaying the slot algorithm: exactly
   targets 1..47 launched once each at both depths).
6. **Warm-cache overhead of the variant.** Two helpers on a fully resident
   cache walk the same pages twice; bounded by the existing +2% startup gate
   (section 6.4).
7. **TODO(round 4):** if the audit shows the trim is the fill's soft-fault
   tax while `trim=0` fails the memory gate, prototype a range-scoped release
   (`DiscardVirtualMemory` on the four consumed source ranges at the trim
   site) so the consumed pages become reclaimable without discarding L+1's
   prefetch. Do not stack it with this round's arms.
8. **TODO(measure):** `PageFaultCount` cannot separate hard from soft faults;
   if the trim audit is inconclusive, add a PDH/ETW hard-fault counter before
   any further trim work.
9. **Receipts to write after the arms:**
   `tmp/q4e-sprint-r1-20261014/startup/receipt-r26-look2.json` and
   `receipt-r26-notrim.json` (+ per-arm worker logs), referenced from here.
