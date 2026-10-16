# Role-P1: prefill supply breakdown -- fill / H2D / per-chunk latch

Date: 2026-10-16. Repo HEAD `2e25fb8`, worktree clean at start. Scope: source +
existing receipts only. ZERO GPU actions, ZERO builds, no src edits, no commit.
Every number carries a source (`file:line` or receipt path). [est] = derived.

Unit of account: the steady-state 4096-token prefill block = 19,360 cold experts
= 49.07 GB (19,360 x 2,534,400 B; `h2d_mb=46792.969` MiB in
`tmp/q4e-sprint-r1-20261014/g5-a1-worker.log`) streamed as 1,210 chunks of 16
slots (`staged=19360`, `[Q4E PrefillStage] ... slots_per_bank=16`,
`Qwen4ExpBnb4Backend.cpp:1673,1685`).

## 1. Supply three-way split (ms per block)

| term | default (g5-a1/a2: banks2, serial fill) | e6-banks4-b1 (banks4, TBB fill) | isolated floor (BlockBench A1) | source |
|---|---:|---:|---:|---|
| (1) host fill | 7,189-7,331 | 2,472 | 4,931 | `fill_ms`; `block-bench-A1.log` |
| (2) H2D pure DMA | 4,961 [est: 49.07 GB / 9.89 GB/s] | 4,961 [est] | 4,956 measured (9.892 GB/s) | `copy_gbps`; supply-bench 9.77 GB/s |
| (3) fence/queue latch + submit residual | 0 exposed + ~523 submit | 6,283-4,961 = 1,322 + 611 | 96 (bankwait) + ~680 | `bank_wait_ms`; `expert_host - fill - bank` |
| supply total (`experts_host_ms`) | 11,487 / 11,600 | 9,366 | 5,731 (wall, fence arm 5,773) | arm logs |

Facts that anchor the table:
- fill is a per-slot `std::memcpy` (see section 2), measured 6.83 GB/s serial
  (49.07 GB / 7.19 s) vs 19.85 GB/s with TBB (e6: 2.47 s).
- pure H2D = 4,961 ms; exposed part differs per arm because fill and DMA
  overlap: default hides 1,186 ms of DMA under the fill (bank_wait 3,775), then
  pays it back as latch cost when the fill is shortened.
- "if X were free" upper bounds on top of the best measured arm (9,366 ms
  supply / 18.08 s block / TTFT 65.9 s):

| free term | supply [est] | block [est] | note |
|---|---:|---:|---|
| fill (2,472) | 6,894 | 15.7 s (261 tok/s) | upper bound only: banks2 shows freed fill is absorbed into bank_wait (fill 2,401 -> bank 7,921, C1 doc) |
| H2D (4,961) | 4,405 | 13.0 s (315 tok/s) | bytes are irreducible without changing the routed set |
| fence/latch (1,322) | 8,044 | 16.6 s (246 tok/s) | per-chunk latch, 1.09 ms/chunk |
| supply hidden under drain | 0 | 8.58 s router_wait + ~0.6 kernel = 9.2 s (446 tok/s) | needs the route dependence gone: this is G5's claim |

## 2. What the fill actually does (source level)

- `Qwen4ExpAdaptiveCache::StageColdExpertsAsync`: `Qwen4ExpAdaptiveCache.cpp:606`.
  Bank wait `:614-617`; plain `std::memcpy` of the whole compact slot
  `:640-644`; parallel branch `:627-639` -> `CopyColdStagingHostSlots`
  (`Qwen4ExpExpertExecutor.cpp:547`, TBB `parallel_for` `:566-571`, serial
  fallback gate `:558-562`); H2D submit `:652-655`; byte counter `:657`.
- Slot = one contiguous compact NF4 blob, no repack, no scale math, no per-expert
  concatenation: gu_packed 1,638,400 + gu_absmax 51,200 + dn_packed 819,200 +
  dn_absmax 25,600 = 2,534,400 B (`Qwen4ExpHostArena.cpp:116-126`).
- Source = arena (native large pages by default, pinned only under
  `AILA_Q4E_HOST_ARENA_PINNED=1`; `Qwen4ExpHostArena.cpp:135-146`). Dest = USM
  host (pinned) bank (`Qwen4ExpAdaptiveCache.cpp:167,237`). Both use plain CPU
  stores; the copy engine never reads the arena.
- 6.8 GB/s hypothesis ranking (falsifiable):
  1. single-thread memcpy ceiling -- supported: same bytes, same buffers, TBB
     fill = 19.85-20.4 GB/s (2.4 s), and the isolated block bench does the same
     serial fill in 4.93 s (9.94 GB/s) with a concurrent DMA reader.
  2. NOT pinned-bank write bandwidth -- the parallel path writes the same pinned
     bank at 19.85 GB/s.
  3. residual production penalty (7.19 vs 4.93 s, +46%) -- unattributed; the only
     remaining candidates are (a) arena large-page source vs heap source,
     (b) CPU/DRAM contention with interleaved kernel submissions, (c) 48
     per-layer pipeline pauses (see section 4 C1).
  4. NOT repack/CPU math and NOT page faults -- pure memcpy, locked large pages.
- "ids D2H" is per layer, not per chunk: top-k writes straight into pinned host
  and the route wait is one event (`Qwen4ExpBnb4Backend.cpp:2547-2562,2646`).
  Per-chunk H2D descriptor traffic (3 int arrays + tiles) is small:
  `Qwen4ExpBnb4Backend.cpp:1789-1793`.

## 3. Levers

L1 (threads; measured, ready to flip): defaults
`AILA_Q4E_STAGE_COPY_PARALLEL=1` + `AILA_Q4E_PREFILL_STAGING_BANKS=4`.
Mechanism: 16-slot host copy via TBB (2.47 s vs 7.19 s) + deeper bank ring.
Measured: block 18.08 s vs 19.05 s (-5.1%), fill -4.7 s, TTFT 69.59 -> 65.9 s
(-4.8%), bit-identical. Charter S2 status: not listed (the "ring 4" entry needs
an owner check against the newer banks4 receipts, see section 5).

L2 (fence removal + chunk bytes; in flight, G5): layer arena of 454 slots =
1.15 GB/chunk, 48 chunks/block instead of 1,210. Mechanism: each layer's cold
set is written into a per-layer arena once, so a staging bank is never reused
inside the layer -- the 1,210 per-bank reader fences (`AdaptiveCache.cpp:652-655`
+ `RecordStagingConsumer`, call site `Qwen4ExpBnb4Backend.cpp:1702`) collapse to 48
layer boundaries, and the H2D chain no longer waits on the compute consumer.
Cost: +12.0% H2D bytes (51.2 GiB vs 46.8 GiB, g5-b1 receipt) = +0.6 s DMA.
Expected: supply -> max(6-thread fill ~2.6 s, H2D 5.6-6.1 s) [est], and if
issued ahead of the route wait it hides under the 8.58 s drain: block 9.5-10.5 s
= 400-430 tok/s [est]. Charter S2 difference: "chunk 8192" is the prompt-token
chunk (route window), "staging 64"/slots32 keep the route-driven per-bank fence
and measured worse (18,496 ms, e6-b4s32); L2 changes the fence topology, not
just the chunk size.

L3 (remove the copy; needs a new mechanism): DMA straight out of the arena is
already denied (c1-e1 +2.9%; S2 "51 GiB pinned -> UR_OUT_OF_RESOURCES"). A
different mechanism would be a pinned *active slice* (one layer = 1.15 GB, not
51 GiB) that the copy engine can read directly, making the fill disappear
(-2.47 to -7.19 s of host time). Without the L2 fence removal this only moves
the time into bank_wait (measured conservation), so it is only worth it as L2's
supply engine.

## 4. Judging commands (all CPU-free of the sprint arms; each <= 3 min)

C1 (isolated floor + fence cost): `build-refactor\AilaQwen4ExpPrefillBlockBench.exe --arms copy,block --reps 3`
  Read: `[BLOCK MEDIAN]` fill ~4.93 s, copy ~4.96 s (9.89 GB/s), bankwait
  ~0.10 s, wall copy 5.73 s / block 5.77 s. Criterion: block-copy <= 0.3 s =
  fence is free when the queue is solo (then production's +3.6..5.7 s over this
  floor is context, not the fence). Falsifier: block-copy > 1 s. Runtime ~60 s.

C2 (does the bench model the production kernel load?): same exe
`--arms kernel --cold-acts 593 --reps 2` then `--cold-acts 806 --reps 2`.
  Read: delta wall/gu/down over 10,224 extra acts (213/layer x 48). Criterion:
  >= 20 us/act means production's ~33,339 cold acts/layer (1.6M/block,
  `misses=1600267` in g5-a1) cannot fit the observed 19 s block, so the
  `misses` counter is not act-level and C1's shape is faithful; ~1-2 us/act
  means production kernels cost 1.6-3.2 s/block and C1 understates the block
  arm. Runtime ~10 s.

C3 (can the layer arena hide under the drain?): `build-refactor\AilaQwen4ExpLayerStageBench.exe --arms stage,overlap --experts 454 --rows 4096 --samples 3`
  Read: `[LS MEDIAN]` stage_ms (fill+H2D for 454 slots alone) vs overlap, and
  overlap vs the load arm (GDN 2 calls at 4096 rows). Criterion: overlap <=
  max(load, stage) x 1.15 => L2's overlap is real; overlap ~ load+stage => the
  arena path serializes and L2 must be re-pipelined. Runtime ~30-60 s.

Also confirmed CLI/sources: all three exes exist in `build-refactor`;
`perf/qwen4exp/q4e_prefill_block_bench.cpp` (fill/copy/bankwait buckets :8-13,
fence arm :172-177, reader :229), `q4e_prefill_supply_bench.cpp` (:70-105 modes,
:306-315 CLI; measures DMA only -- 648.8 MB/batch at 9.77 GB/s,
`supply-bench-2026-10-15.log`), `q4e_layer_stage_bench.cpp` (:93-105 params,
:186-210 CLI). No `[LS MEDIAN]` receipt exists yet.

## 5. Uncertainties

- Charter S2 "ring 4" vs the newer banks4 receipts (`d7a5f6f`: -2.4% block,
  -4.8% TTFT): the names may collide; reconcile before flipping L1 defaults.
- The block bench caps `cold_acts <= 2 x staged_per_layer` (806), so it cannot
  represent production's ~33k cold acts/layer -- C2 prices the gap.
- The +46% fill gap (7.19 s production vs 4.93 s isolated) is unattributed;
  C1 reproduces or refutes the isolated side only.
- LayerStageBench is a synthetic GDN fixture (no attention/router/hc), so C3
  bounds the overlap opportunity, not the production block.
- H2D rates differ across receipts: 9.89 GB/s (block bench) vs 9.77 GB/s
  (supply bench, 2026-10-15). All derived numbers use 9.89.
