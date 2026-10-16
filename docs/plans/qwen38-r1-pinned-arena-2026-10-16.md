# P8: pinned expert arena -> staging-free DMA (archaeology + verdict)

Date: 2026-10-16. Role: P8. Scope: decide whether the host expert arena can be
read by the copy engine directly (pinning it first if needed), which would
delete the `fill` host memcpy and the pinned bounce bank from the prefill
supply path. No GPU was run by this role; every number below is a receipt or a
source line. All files ASCII.

Verdict first: the premise is already answered by two closed receipts.

1. The 51 GiB `AILA_Q4E_HOST_ARENA_PINNED=1` experiment did *not* fail to
   allocate - it allocated 208 pinned blocks and loaded the model. It died on
   the first request with `UR_RESULT_ERROR_OUT_OF_RESOURCES`. So multi-block
   pinning works at allocation granularity; the platform cannot sustain the
   pinned set once request-time allocations arrive.
2. The `c1-e1` direct-DMA arm already read the large-page arena straight from
   the copy queue. Its source was pageable large pages, but there is no hidden
   UR staging tax to blame: the probe measures 9.05-9.23 GB/s from large pages
   vs 9.83 GB/s from pinned (32 MiB), and e1's cold production run sustained
   ~9.7-9.8 GB/s over 46.8-47.6 GB per chunk. The +2.9% TTFT regression is a
   wall that did not move when `fill`/`bank_wait` were driven to ~0.

Remaining engineering upside of the whole P8 branch: <=1-8% on the transfer
rate, and 0 s/block observed in the e1 precedent. Recommended: close, unless
the new bench shows a >5% pinned-vs-large-page gap.

## 1. Receipt 1: full 51 GiB pin (charter section 2)

### 1.1 What it did

- Gate: `AILA_Q4E_HOST_ARENA_PINNED=1` (default 0), read once in
  `src/models/Qwen4ExpHostArena.cpp:135-138`.
- Allocation API: the same chunk loop that allocates the large-page arena
  (`src/models/Qwen4ExpHostArena.cpp:139-146`) calls
  `ctx_->alloc_host(bytes)` when the gate is on, else
  `ctx_->alloc_large_host(bytes)`. `Context::alloc_host`
  (`src/core/Context.hpp:66-68`) routes to `allocate(bytes, true)`
  (`src/core/Context.hpp:238-256`), which calls
  `sycl::malloc_host(bytes, q_)` - i.e. the Level Zero USM host allocator via
  UR (`zeMemAllocHost`). It is *not* `MapUserPhysicalPages` and not a raw
  `urUSMHostAlloc` call; `src/memory/LargePages.cpp:50` still calls
  `enable_lock_pages()` on the large-page path only (privilege setup, no data
  locking).
- Granularity: `per_chunk = max(1, min(chunk_bytes, 256 MiB) / slot_bytes)`
  (`src/models/Qwen4ExpHostArena.cpp:134`) with `chunk_bytes = 256 MiB`
  (`src/models/Qwen4ExpHostArena.hpp:20`). Slot bytes = 2,534,400
  (`perf/qwen4exp/q4e_prefill_block_bench.cpp:44`), so 105 experts per chunk;
  each chunk is rounded by `LargePageAllocationSize` to a 2 MiB multiple
  (`src/memory/LargePages.cpp:40-45`), i.e. 266,338,304 B = 254.0 MiB.
- Total: the receipt run logs `Loading native RAM complement: 21784 experts,
  51.46 GiB; GPU-only experts=2792`
  (`tmp/q4e-resident-speed-20261009/b3-diag-a1-worker.log:9`, same arena as
  the pinned arm). 21,784 = 207 x 105 + 49 -> 207 blocks of 254.0 MiB plus one
  partial block of ~120 MiB = 208 pinned blocks, 51.46 GiB.
- Failure handling: any allocation throw is caught at
  `src/models/Qwen4ExpHostArena.cpp:147-173`, printed as `[Q4E Arena]
  allocation failed: ... pinned=%d err=%s`, and makes
  `Qwen4ExpBnb4Backend.cpp:874-877` abort the model load. A load that reaches
  `ready` therefore proves that all 208 pinned blocks were allocated.

### 1.2 What happened

Receipt: `tmp/q4e-resident-speed-20261009/b3-pinned/result.json`.

- `worker_environment.AILA_Q4E_HOST_ARENA_PINNED = "1"`.
- `worker_startup_seconds = 171.34`, `health_at_load.status = ok`,
  `model_resident_device_bytes_at_load = 11,963,686,138` (~11.1 GiB VRAM).
  The pinned arena was live before the request.
- First request: `counters.failed = 1`, request `status = failed`, then
  `ipc.error = "Inference shutdown grace period expired"`, cleanup
  `exit_code = 2`, `shutdown.graceful = false`.
- Error text as delivered to the client:
  `tmp/q4e-resident-speed-20261009/b3-pinned/pi/aila-pi-smoke-xcbwqq/events.jsonl`
  carries `"errorMessage":"level_zero backend failed with error: 40
  (UR_RESULT_ERROR_OUT_OF_RESOURCES)"`.

Docs that record the same experiment: `docs/plans/qwen38-optimization-roadmap-2026-10-10.md:98-101`
("the worker loads under ~51 GB of locked pages, but the first request fails
with level_zero error 40 ... conclusion: a 51 GB host complement cannot replace
the native large-page arena") and the charter
`docs/plans/qwen38-performance-research-charter-2026-10-14.md:58`.

Note: the worker log of the pinned arm was not archived (only `result.json`,
the pi stream and `stderr.log`/"Worker shutdown: Inference shutdown grace
period expired"), so *which* request-time allocation threw is not localized.
What is settled: allocation of the full pinned set succeeded; the first
request could not co-exist with it.

## 2. Receipt 2: c1-e1 direct DMA (2026-10-13)

- Source allocation: the arena arm's source is the private large-page arena
  (`ctx.alloc_large_host`, `src/models/Qwen4ExpHostArena.cpp:145` default
  branch -> `src/memory/LargePages.cpp:47-58`,
  `VirtualAlloc(..., MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, ...)`). It is
  *not* pinned or locked; the flag `AILA_Q4E_HOST_ARENA_PINNED` was off.
- What "direct" did: the staging path skipped the pinned bounce. The copy
  queue reads the arena pointer directly; the equivalent committed shape is
  `tests/ops/Qwen4ExpKernelTests.cpp:4542-4591` (`time_copy_src(large)` next to
  pinned and D2D, comment: "can the staging H2D read straight from the
  large-page arena memory instead of the pinned bounce buffer"). The e1 arm's
  production patch was `AILA_Q4E_STAGE_DIRECT_DMA=1`; it was rolled back and
  never committed (`git log -S STAGE_DIRECT_DMA` finds only 31410f9, which
  archived the doc and the probe; the C1 doc section 5 says the arm code was
  rolled back).
- Probe numbers, same file/commit (`docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md:43`):
  large-page direct 9.05-9.23 GB/s, pinned 9.83 GB/s, D2D 117-129 GB/s at
  32 MiB. The large-page source is therefore at DMA speed; a runtime staging
  pass would not survive this measurement.
- Cold production rate, `tmp/q4e-strata-c1/c1-e1-worker.log:33-39` (chunks of
  4096 rows): `fill_ms = 0.639-0.753`, `bank_wait_ms = 0.044-0.051`,
  `experts_host_ms = 4296-4909`, `h2d_mb = 42130-47595` -> 9.6-9.8 GB/s,
  `router_wait_ms = 6856-15432`, `ForwardWall` 11.40-20.65 s.
- Baseline arm a04 (same doc section 2): `fill` 7.20-7.32 s, `bank_wait`
  3.77-3.85 s, chunk wall ~20.3 s, TTFT 70.23 s. e1 TTFT 72.23 s (+2.9%),
  i.e. *no improvement* even though fill+bank_wait collapsed. The time moved
  into `router_wait`/queue depth, not into the wall.

Consequence for the charter's "if fill is free" bound: the fill is already
masked by device work in the banks2 regime of that experiment, and the
transfer itself is additive with memory-bound kernels (four independent
receipts in `docs/plans/qwen38-r1-prefill-overlap-design-2026-10-16.md`
section 1). Removing the fill is therefore not worth 2.47-7.2 s/block of wall
in the model that today's P1/P6 ledgers use.

## 3. Feasibility of multi-block pinning (64 x 512 MiB / 256 x 256 MiB)

### 3.1 Code changes that would be needed

- Block size: `src/models/Qwen4ExpHostArena.cpp:134` caps each chunk at
  `min(chunk_bytes, 256 MiB)`. Raising the 256 MiB literal to 512 MiB is a
  one-line change; 64 x 512 MiB would then be 64 chunks.
- Allocation: the pinned branch already exists (`:145`); no new API is needed.
- Accounting: `host_budget_` is a plain `AllocationBudget` with `limit_ = 0`
  by default, and 0 means "no limit"
  (`src/memory/AllocationBudget.hpp:19-48`, check at `:26-31`). Limits are
  only installed if a caller passes a nonzero value to
  `Context::set_allocation_limits` (`src/core/Context.hpp:112-119`); the
  b3-pinned health report shows `max_pinned_host_bytes = 0`. The ledger does
  not block the arena and does not need to change.
- Failure path: unchanged (`:147-173`); a partial allocation already fails the
  load, which is why the b3-pinned run proves the full set fit.

So the code side is trivial. The blocker is the platform.

### 3.2 Platform and RAM judgment

- Physical RAM on this box: 68,622,086,144 B = 63.9 GiB
  (`Win32_ComputerSystem.TotalPhysicalMemory`, queried 2026-10-16). Arena:
  51.46 GiB = 55.3 GB, i.e. 86% of physical. Worker private usage during a
  production run is ~64.9 GB (coordinator's monitor), i.e. the process already
  commits more than physical and leans on the pagefile for the tail.
- Pinning the whole arena removes its pageability. What remains for the OS,
  the WDDM driver, the logits/PLE host buffers, KV/scratch host mirrors and the
  existing pinned staging banks is ~8.6 GiB of physical. The measured result
  fits this picture exactly: the 51 GiB pinned set allocates and the model
  loads, then the first request's next allocation cannot be satisfied and UR
  returns `OUT_OF_RESOURCES`.
- External corroboration for the same class of machine:
  `tmp/Strata/docs/MULTI_GPU.md:217-220` - "Under WDDM (Windows, and WSL2)
  only 8 GiB of the expert arena is pinned (more, mapped into two GPU
  contexts, leaves WDDM refusing allocations); the rest streams through the
  pinned staging ring ... `STRATA_ARENA_PIN_GIB=N`". Strata's production
  answer on Windows is a bounded pinned slice, not a fully pinned arena.

Judgment: multi-block pinning is *possible at load* (measured) but *not viable
in steady state* on this 64 GiB / A770 / WDDM box. A bounded <=8 GiB slice is
the only pinning shape with external precedent; see candidate C2.

## 4. Minimal verification experiment (new, not yet run)

New file: `perf/qwen4exp/q4e_arena_dma_bench.cpp`; new CMake target appended in
`CMakeLists.txt:798-801` (`AilaQwen4ExpArenaDmaBench`); existing targets
untouched. It allocates two ~1 GiB sources in `--block-mib` blocks (default
256 MiB, so the multi-block pattern is exercised), fills them with different
deterministic patterns, and runs three arms through `cgh.memcpy` (the same
runtime shape as the committed probe):

- `bank`: large-page source -> pinned bank memcpy -> H2D (today's path, single
  threaded fill, ping-pong banks; `bankwait_ms` is the reader fence wait).
- `lg`: large-page source -> H2D directly (c1-e1's mechanism).
- `pin`: pinned `sycl::malloc_host` source -> H2D directly.

Every arm is verified by full D2H readback + memcmp against its own source and
a mismatch check against the other source. Readout per arm: `wall_ms`,
`fill_ms`, `bankwait_ms`, `dma_ms` (event spans), `wall_gbps`, `dma_gbps`, plus
an extrapolation to the production 49.07 GB/block.

Command (exclusive GPU slot; fill+verify of 1 GiB per arm, ~15-40 s total):

```powershell
build-refactor\AilaQwen4ExpArenaDmaBench.exe --total-mib 1024 --block-mib 256 --group-slots 16 --reps 3
```

Criteria:

- Validity: every `[P8 VERIFY]` line `match_src=1`, `[P8 GUARD] bitwise=pass`.
- Source tax: let `d = pin.wall_gbps - lg.wall_gbps`. `d <= 3%` -> pinning buys
  nothing, close P8. `d > 5%` and `pin.extrap_block_s <= bank.extrap_block_s -
  1.0` -> a bounded pinned slice becomes worth one production arm.
- Direct path health: `lg.wall_gbps >= 9.0` reproduces the probe (do not cite
  c1-e1 as already closing the source question); `lg.wall_gbps < 8.0` closes
  both P8 and overlap-design candidate Z2.
- `bank` is the single-threaded-fill control: its `fill_ms` should scale to the
  serial 7.2 s/block number, and its wall is not comparable to the parallel
  fill that production now defaults to (2.47 s/block); do not read the bank
  extrapolation as a regression.

## 5. Two candidates

C1 - direct DMA from the arena in the current banks4 pipeline (per 16-slot
group, bounce removed). Where: `src/models/Qwen4ExpAdaptiveCache.cpp:606-660`
(`StageColdExpertsAsync`) plus the pump in `Qwen4ExpBnb4Backend.cpp` around the
staging/reader fence. Mechanism: the copy queue reads
`host_arena_.GetExpertPointer(layer, expert)` (large pages) directly and the
fill disappears; the destination bank still carries the reader fence.
Expected: fill -2.47 s/block at banks4 (P1 supply breakdown section 1), host
write traffic -49 GB/block; wall upside bounded by the e1 precedent (0 s,
banks2) with a model-side ceiling of -2.5 s/block [est]. Risks: e1's wall did
not move; the fence/slot-reuse bookkeeping changes. Not the section-2 entry:
that one pins 51 GiB; this reads the existing large-page arena with zero new
locked memory. Cheapest form: an opt-in flag that keeps the default path
bit-identical.

C2 - bounded pinned slice (<=8 GiB, Strata's WDDM number) plus direct reads for
that slice. Where: `src/models/Qwen4ExpHostArena.cpp:134-146` (pin the first N
chunks) and the staging selection. Expected: <=16% of the 49.07 GB/block
covered -> <=0.4 s/block [est] at banks4 fill scale, under the 3%/5% default
gates. Risks: WDDM cap for real; complexity across load/runtime selection. It
is a bounded version of the section-2 denial, not a re-run of it, but my
recommendation is NO-GO unless the section-4 bench shows a >5% pinned gap.

## 6. Uncertainties and needed commands

U1. Which allocation threw UR 40 in b3-pinned is not localized (no worker log
was archived; the ledger limit is 0, so it is not `host_budget_`). Needed if
the coordinator wants that localized: `python tmp/q4e-sprint-r1-20261014/run.py
pinned-repro --prime --diag AILA_Q4E_HOST_ARENA_PINNED=1` (today's build, ~1-3
min with load), then read the `[Q4E Arena]`/first-failure lines. Expected
outcome: the same UR 40 at request time; if it passes, the 2026-10-12 result was
build-specific and pinning would deserve one more look.

U2. The b3-pinned receipt does not contain the `native RAM complement` INFO
line, so "all 208 blocks were pinned" is inferred from the load gate at
`Qwen4ExpBnb4Backend.cpp:877` (which aborts load on arena failure) plus the
reached `ready` state, not from a direct log line.

U3. e1's +2.9% TTFT is within that era's A/A spread (1-3%); treat it as "not an
improvement", not as a precise cost.

U4. Whether UR internally stages `cgh.memcpy` from a large-page pointer is not
observable from outside. The probe (32 MiB: 9.05-9.23 GB/s) and e1 (cold 46.8
GB: ~9.7-9.8 GB/s) bound any such staging to near zero; the section-4 `lg` arm
at `--total-mib 2048` is the direct cold test.
