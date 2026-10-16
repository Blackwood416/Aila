# R1 P6: why the production prefill fill is ~2x the isolated bench

Date: 2026-10-16. Scope: explain the gap between
`build-refactor/AilaQwen4PrefillBlockBench.exe --arms copy,block` (fill 4.95 s /
copy 4.96 s / bankwait 0.095 s, `tmp/q4e-sprint-r1-20261014/block-bench-A1.log`)
and the production arms (`tmp/q4e-sprint-r1-20261014/g5-a{1,2}-worker.log`:
fill 7.19-7.33 s on six clean 4096-row blocks (6.58 s on the 1656-row tail),
bank_wait 3.74-3.91 s, h2d 42.1-47.6 GB per block).
No source changes, no build, no GPU run. All numbers are existing receipts.

## 1. Submission paths: what is identical and what is not

Identical on both sides:

| item | bench | production |
|---|---|---|
| host memcpy | `perf/qwen4exp/q4e_prefill_block_bench.cpp:167-170` | `src/models/Qwen4ExpAdaptiveCache.cpp:642-643` |
| bank reuse wait before fill | `:159-163` | `:614-615` |
| H2D fence on the bank's last consumer | `:172-181` (`arm=1`) | `:652-654` via `RecordStagingConsumer` (`Qwen4ExpBnb4Backend.cpp:1702`) |
| slot size / batch shape | 2,534,400 B, 16 slots/batch, 403/layer | same (`slot_bytes`, `staging_slots_per_buffer`) |
| source bytes per pass | 49,025,433,600 | 46,793-47,595 MB (+tail) |

Differences:

1. Source identity. Bench: an 81.1 MB pageable heap vector, two 40.55 MB halves
   recycled for every batch (`:362` `b.source.resize(host_bank_bytes)`, `:168`).
   Production: the 51.46 GiB native host arena, 49 GB of *unique* bytes per
   block (`AdaptiveCache.cpp:638` `arena.GetExpertPointer`).
2. Thread. Bench fills on the run thread with nothing else alive. Production
   fills on the worker thread; `AILA_Q4E_STAGE_COPY_PARALLEL=1` moves it to the
   shared TBB arena (`Qwen4ExpExpertExecutor.cpp:561-579`, 16 slots over the
   `AILA_Q4E_CPU_THREADS=18` arena from the production profile env).
3. Consumer weight. Bench consumer = gu+down of 593 acts/layer = 719 ms per
   pass = 0.58 ms/chunk (median gu 138.6 + down 580.4, `block-bench-A1.log`).
   Production consumer = the chunk's real MoE kernels behind the whole layer's
   compute queue (device drain ~8.6 s/block).
4. Queue properties. Bench copy queue adds `immediate_command_list` +
   `enable_profiling` (`:323-327`); production `copy_q_` is plain in-order
   (`Qwen4ExpBnb4Backend.cpp:885-886`). Doorbell/timestamp overhead only; not a
   bandwidth factor at 40.5 MB per submit.

## 2. Source residency: the page-fault hypothesis is refuted

Runtime fill source = private large pages, not the file mmap:

- `Qwen4ExpHostArena.cpp:141-146` allocates with `ctx_->alloc_large_host(bytes)`
  (`AILA_Q4E_HOST_ARENA_PINNED` default false; not set by the sprint profile).
- `Context.hpp:72-86` -> `memory/LargePages.cpp:44-52` =
  `VirtualAlloc(MEM_LARGE_PAGES | MEM_COMMIT)`. Large pages are non-pageable by
  construction and cannot enter the standby/trim path.
- `AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC` (default 1) only issues
  `PrefetchVirtualMemory` over the *SafeTensors mmap* ranges and only from
  `LoadFromModelWeights` (`HostArena.cpp:478-493`, call sites `:592-595`,
  `:671-700`). It is startup-only and does not touch the arena.
- `[Qwen4Exp] Adaptive residency ready: expert_mmap_fallback=0` in the g5 logs:
  no runtime mmap fallback exists.

The recorded "working set 6.6 GiB / available RAM 0.13 GiB" thrash
(`qwen38-r1-cost-model-2026-10-14.md` 6.24) is a *startup* arm whose source was
the mmap (`RAM copies 179.3 s`), i.e. a different mechanism. At runtime the
49 GB read cannot produce page faults, disk reads or pagefile traffic.
Confidence: high (source-level; a probe is still listed in section 5).

## 3. Unique expert bytes and cross-block repetition

Arena = 21,784 experts x 2,534,400 B = 55.2 GB (51.46 GiB), GPU-only experts
2,792 (`g5-a1-worker.log` arena line). Per 4096-row block:

| block | staged experts | GB | share of arena bytes |
|---|---:|---:|---:|
| 1 (rows 0-4095) | 19,360 | 49.07 | 88.9% |
| 2 | 19,577 | 49.62 | 89.9% |
| 3 | 19,692 | 49.91 | 90.4% |
| 4 (1656 rows) | 17,431 | 44.18 | 80.1% |

The 14K prompt stages 76,060 slots = 192.8 GB, so every arena byte is streamed
3.49x-3.93x per prompt (bound: union is between 19,360 and 21,784 experts).
Reuse distance is the arena itself (55 GB), i.e. ~1200x the 46 MB L3, so no
block can be served from cache: each block is a genuine cold 49 GB DRAM read.
Consequence: "keep the hotset warm" cannot reduce fill bytes; only a smaller
active set (GPU residency) can. Confidence: high (counters, not estimates).

## 4. Root cause (two independent factors)

### F1 - the bench number is not a host-fill measurement, and the production
fill is single-core latency-bound

`copy_gbps` is `bytes / sum(event spans)` (`q4e_prefill_block_bench.cpp:436-437`)
= 9.890 GB/s, and the bench fill (4.953 s) matches copy+wait (4.956 + 0.095) to
2%: the host is *paced* by the DMA of the same bank, so the bench only bounds
the host at ~9.9 GB/s for an 81 MB recycled source (46 MB L3 partially serving
it). Production fill is host-paced and reads unique bytes:

| arm | source | fill | rate |
|---|---|---:|---:|
| bench block | 81 MB recycled | 4.953 s | 9.90 GB/s |
| g5-a1 default (serial memcpy) | 49 GB cold arena | 7.19-7.33 s | 6.4-6.5 GB/s |
| e6-banks4-b1 (`STAGE_COPY_PARALLEL=1`) | same arena | 2.47-2.52 s | 18.9 GB/s |

Same binary, same arena, same bytes: 6.5 -> 18.9 GB/s with 16 TBB workers.
18.9 GB/s of copy = 37.8 GB/s of DRAM traffic, ~57 GB/s with RFO, i.e. ~95% of
the 4x16 GB DDR4-1866 quad-channel peak (59.7 GB/s) on this Xeon E5-2696 v3
(18C, 46 MB L3). So the single-thread rate is a per-core MLP/latency limit, not
a property of the arena, and the machine-wide ceiling for this copy shape is
~18.9 GB/s = 2.5 s per 49 GB block.
The P1 model ("production supply is 1.6-2.0x the isolated bench", "3.1-3.6 ms
fixed cost per chunk") is therefore an artifact of comparing a DMA-paced,
L3-assisted source with a host-paced 49 GB stream. [est for the L3 share;
the pacing argument and the 6.5/18.9 GB/s endpoints are measured.]

### F2 - bank_wait is the compute queue's consumer fence, not the fill

Production bank_wait 3.74-3.78 s = the worker blocking in
`AdaptiveCache.cpp:614-615` before it may overwrite a bank whose H2D was fenced
on that bank's previous consumer (`:652-654` <- `RecordStagingConsumer`
`Qwen4ExpBnb4Backend.cpp:1702`). Those consumers are the chunk's real kernels that
sit behind the rest of the layer in the in-order compute queue, so the DMA
cannot start when the host is ready. The bench's consumers are 0.58 ms/chunk on
an otherwise idle queue, so its wait is 0.095 s (2 µs/chunk).
Both regimes appear in one production log: 4096-row chunks (device-heavy) show
bank_wait 3.77 s with fill 7.2 s; the 24-row continuation chunks show bank_wait
0.93 ms with fill 950 ms / h2d 6.1 GB - i.e. when the host is the slow side the
wait vanishes and the supply rate is exactly the 6.4 GB/s fill rate.
Confidence: high.

## 5. Probes (2, both need a quiet slot: they saturate DRAM)

P6-A, CPU-only, no GPU at all, ~90 s. Source and binary are in the tree:
`tmp/q4e-sprint-r1-20261014/fillrate/fill_rate_probe.cpp` (compiled to
`fill_rate_probe.exe`). It replays the bench batch loop unchanged against
(a) the 81 MB recycled source, (b) a 40.55 MB half, (c) a 16-24 GiB stream,
(d) c + a concurrent reader thread, (e) c with a VirtualLock'ed destination,
(f) c split over 4/8/16 threads.
Command: `tmp\q4e-sprint-r1-20261014\fillrate\fill_rate_probe.exe 16 2 4,8,16`
Smoke receipt (1 layer, arms 0 and m only, ~1 s of DRAM traffic):
`fill_rate_probe.exe 1 1 2 1 0` -> `bench ... fill_rate=8.14 GB/s`;
`fill_rate_probe.exe 1 1 2 1 m` -> `stream_mt2 ... 14.80 GB/s`. The full 48-layer
run was NOT executed: an AilaWorker arm was live during this analysis and the
probe saturates DRAM for ~90 s, so it needs the operator's quiet slot.
Criteria: (a) 9.5-10.5 GB/s AND (c) 6.2-7.0 GB/s => F1 is source locality and
the bench number must not be used as a production floor; (c) near 9.9 => F1 is
wrong and the production-only slowdown must be thread/queue contention (then
run P6-B); (f) 16 threads >= 15 GB/s => LR1 candidate below is real.

P6-B, GPU arm, ~5 min, proves the fill fix in situ at fixed banks:
`python tmp/q4e-sprint-r1-20261014/run.py p6-par-b2 --prime --diag --generate=8 AILA_Q4E_STAGE_COPY_PARALLEL=1`
Criteria: `fill_ms` <= 4.0 s (vs 7.19/7.28/7.33) and `bank_wait_ms` >= 5.0 s
while the block wall stays within +-1% => the host fill was the only thing the
flag changed and the freed time is conserved into bank_wait (P2 conservation),
so the flag alone is not a wall win; `fill_ms` >= 6 s => the production worker
cannot use the TBB path (check `max_concurrency() > 1` and the CpuWorker log).

## 6. Fixes for the fill gap

LR1 (zero new mechanism, already measured): make the parallel fill the default
beside `AILA_Q4E_PREFILL_STAGING_BANKS=4`. Receipt: fill 7.19-7.33 -> 2.47-2.52 s,
block wall 19.05 -> 18.08 s, TTFT 69.6 -> 65.9 s, bit-exact (P1 L1 /
`e6-banks4-b1`). Expected: -4.7 s of host fill, ~-1.0 s of block wall.
Not a section-2 item: section 2 forbids pinning the 51 GiB source
(UR_OUT_OF_RESOURCES) and the copy-stream-ahead ideas; this only parallelizes
the existing memcpy into the existing 77.3 MiB pinned banks.

LR2 (the fill becomes free, but only with LR1): the layer-stage / ArenaCold
path must fill its 454 slots/layer through the TBB path. Arithmetic: 454 x
2,534,400 B = 1.151 GB/layer; at 6.5 GB/s single-thread that is 177 ms/layer =
8.5 s/block - the whole 178.8 ms/layer device budget and no slack; at
18.9 GB/s it is 61 ms/layer = 2.9 s/block, which fits inside the device drain.
So a layer-stage arm measured with the serial fill can never show overlap.
Expected: up to -7 s/block in the extreme (fill hidden), realistically the
`max(device, supply)` floor of ~10-12 s/block = 340-400+ tok/s once the
consumer fence (F2) is also removed. Not a section-2 item for the same reason
as LR1 (no new pinning; same bytes, same order, bit-exact).

## 7. Open items

U1 The 45% hot-source advantage is attributed to L3/DDIO [est] until P6-A runs;
the pacing argument does not depend on it.
U2 P6-A's `stream_dma` arm cannot emulate Intel DDIO (PCIe reads are served from
L3 when the 81 MB staging ring is warm), so treat that arm as an upper bound on
DMA interference.
U3 Whether the production worker's TBB arena actually reaches 16-way for
`CopyColdStagingHostSlots` (16 tasks) in the P6-B arm - read
`[Q4E CpuTilePool] threads=` and `[Q4E PrefillStage]` lines.
U4 The 24-row continuation chunks fill at 6.4 GB/s with bank_wait ~0; if a
future arm shows fill near 9.9 GB/s there, F1 needs re-opening.
