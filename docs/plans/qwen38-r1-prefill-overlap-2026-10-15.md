# Role-E2: prefill supply/overlap audit (router_wait + bank_wait attribution)

Scope: source + existing receipts only, no GPU runs, HEAD `df1f45a`.
All block numbers are the steady-state 4096-token prefill chunk (14K prompt,
`[Q4E AdaptiveWork] submitted_ms`, pos=0 second request), unless noted.

Path correction: the production prefill is the **chunk-major** path
(`RunAdaptivePrefill`, `src/models/Qwen4ExpBnb4Backend.cpp:1540-1643`), not
layer-major (`AILA_Q4E_LAYER_MAJOR` defaults off and also requires count==8192;
`src/models/Qwen4ExpLayerMajor.cpp:112,116`). The chunk-major AdaptiveWork
line carries the `ple_reused=` field; the layer-major line ends `layer_major=1`.

## 1. Emission sites (what each marker actually measures)

| marker | code | measured interval |
|---|---|---|
| `router_wait_ms` | `src/models/Qwen4ExpBnb4Backend.cpp:2505-2515` (`wait_actual_route`, called for prefill at `:2646`), summed into `prof_router_sync_ms`, printed at `:2901-2926` | one `actual_route_event.wait_and_throw()` per layer = 48 sync points/block |
| `fill_ms` | `src/models/Qwen4ExpAdaptiveCache.cpp:614-621` (`fill_begin` -> `stats_.staging_fill_ns` at `:649`) | host memcpy arena -> pinned bank, per staging chunk |
| `bank_wait_ms` | same function `:613-617` (`staging_copy_[pp].wait_and_throw()` -> `stats_.staging_wait_ns`) | wait for the previous H2D of that staging bank |

## 2. router_wait attribution (8,583 ms/block, banks=4)

Host-side terms are all emitted in the same log line and sum to 0.05% of it:

| component | ms/block | source |
|---|---:|---|
| host router GEMM submit (`dense_rows` `:2462-2470`) | 1.787 | `moe_front_ms` |
| top-k kernel submit (`:2482-2493`) | 2.038 | `topk_submit_ms` |
| shared-expert submit (inside `kStageRouter`) | 8.693 | `shared_submit_ms` |
| route-stage residual | 0.096 | `router_other_ms` |
| router prediction compute / pred D2H / pred staging wait | **0 / 0 / 0** | prefill disables prediction: `predict_next` requires count==1 (`:2564-2567`); arms ran `AILA_Q4E_PREFETCH_PREFILL=0` (receipt env) |
| **device queue residual (queue depth)** | **~8,579 (99.9%)** | remainder of `router_wait_ms` |

Mechanism: `wait_actual_route()` waits on an event whose position in the in-order
`ctx.queue()` is after attention(L) + HC(L) + router GEMM + topk; nothing between
the submit block and the wait consumes device work, so the wait is the drain of
the layer's device stream (plus the previous layer's expert-kernel tail, still
queued). Cross-check `docs/plans/qwen38-r1-cost-model-2026-10-14.md` sections 3
and 6.25: with per-stage sync the marker collapses to 3-21 ms; 7,727-8,389 ms
error bars there match the banks=2 arms in section 4 below.
Per-layer: 8,583/48 = 178.8 ms.

## 3. bank_wait attribution (6,283 ms/block, banks=4)

1,210 staging chunks/block (19,360 staged experts / 16 slots; `staged=` and
`[Q4E PrefillStage] slots_per_bank=16` in the arm log). 40.55 MB per chunk
(16 x 2,534,400 B compact slot).

| component | per chunk | per block | source |
|---|---:|---:|---|
| H2D transfer at the measured bench rate (9.89 GB/s) | 4.10 | 4,962 | `qwen38-r1-cost-model-2026-10-14.md` 6.29 supply bench |
| command/queue/fence residual | 1.09 | 1,321 | 6,283 - 4,962 |
| **bank_wait total** | **5.19** | **6,283** | arm `e6-banks4-b1` |

The residual sits in `StageColdExpertsAsync`: the memcpy starts after the
cross-queue dependency `cgh.depends_on(staging_reader_[pp])`
(`Qwen4ExpAdaptiveCache.cpp:652-654`), where the reader event is the compute-queue
consumption of that bank recorded by `RecordStagingConsumer` from the chunk loop
(`Qwen4ExpAdaptiveCache.hpp:219-222`, call site `Qwen4ExpBnb4Backend.cpp:1628`).
The same ledger section records a 95 ms bank wait in the isolated supply bench,
so the residual is production compute-context fence/queue latency, not intrinsic
copy cost.

Disjointness vs router_wait: the two are disjoint host intervals per layer
(`wait_actual_route` at `:2646` precedes `RunAdaptivePrefill` at `:2701`), so the
sums are real in wall time. But there is a measured **conservation channel**
between the markers: deepening the pump moves staged-tail time out of bank_wait
and into the next layer's router_wait.

Block closure (`e6-banks4-b1`, submitted 18,076.8 ms): router_wait 8,583
(47.5%) + `experts_host_ms` 9,366 (51.8%; fill 2,472 + bank_wait 6,283 + ~600 ms
of submissions/merges) + attention/HC host submits 113 + moe_front 1.8 +
topk_submit 2.0 + shared_submit 8.7 + shared_add 0.8 + hc_inject 0.7 + ple 0.5 =
18,076.5 ms (residual 0.3 ms). There is no third hidden term: the block is
device-drain + supply, and the two are serial because the host only learns the
route at the end of the drain.

| arm | banks | slots | chunks | submitted | router_wait | fill | bank_wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| `e11-default-frontfused` | 2 | 16 | 1210 | 18,626 | 7,569 | 2,408 | 7,921 |
| `f1-b1` | 2 | 16 | 1210 | 18,632 | 7,580 | 2,401 | 7,938 |
| `e6-banks4-b1` | 4 | 16 | 1210 | 18,077 | 8,583 | 2,472 | 6,283 |
| `e6-banks4-b2` | 4 | 16 | 1210 | 18,384 | 8,724 | 2,456 | 6,490 |
| `e6-b4s32` | 4 | 32 | 605 | 18,496 | 10,278 | 2,592 | 4,551 |

All five arms carry `AILA_Q4E_STAGE_COPY_PARALLEL=1` (receipt env). Readings:
banks 2->4 moves -1,655 ms out of bank_wait and +1,003..+1,144 ms into
router_wait; slots 16->32 moves -1,732 ms and +1,695..+2,922 ms. Net block gain
is 556/242 ms (banks4) and 0/-491 ms (slots32) -- the tail is deferred, not
saved. The section 6.25 six-bucket table must therefore not be summed:
GDN 4.1 + full attn 2.8 + hc/router 0.6 + expert kernel 0.55 are inside
router_wait, and the 5.3 s H2D is inside bank_wait (25.6 s vs a 19.4 s block).

## 4. Supply cycle model (new, falsifiable)

Per-chunk supply cycle = (fill + bank_wait)/chunks; effective H2D rate =
40.55 MB / cycle (81.1 MB for slots=32):

| config | cycle/chunk ms | effective GB/s | implied fixed cost over 9.89 GB/s |
|---|---:|---:|---:|
| banks 2, slots 16 | 8.55 | 4.74 | 4.45 ms |
| banks 4, slots 16 | 7.24 | 5.60 | 3.14 ms |
| banks 4, slots 32 | 11.80 | 6.87 | 3.60 ms |

Fixed cost ~3.1-3.6 ms/chunk at banks=4 is 3.8-4.4 s/block, comparable to the
5.0 s of pure DMA, and it is what separates the 8.75 s production supply from
the 5.7 s supply bench. It is per-chunk (scales with chunk count, not bytes),
which is why "parallel fill" is conserved into bank_wait (section 6.5 of the
cost model: fill -4.9 s, bank_wait +4.0 s) and why hiding the fill is not a
lever. Falsification probe in section 6.

## 5. Two overlap candidates

### C1: whole-layer speculative staging on the non-resident set (extend E7/G5)

Mechanism: the layer's *non-resident* expert set is known before the layer runs
(resident table frozen, `AILA_Q4E_HOTSET_SWAPS=0`), so its supply can be issued
before `wait_actual_route` and overlap the 178.8 ms/layer device drain instead of
following it. Cost proof from receipts: route-driven staging already uploads
19,360 experts/block = 403.3/layer, and the loader reports 21,784 non-resident
experts (2,792 GPU-only; `[Qwen4Exp] Loading native RAM complement` in the arm
log) = 453.8/layer -> the speculative superset is only **+12.5% H2D
(+5.8 GB/block, +0.6 s DMA)** for 100% coverage.
Expected: block ~10 s (router_wait 8.58 + expert kernels 0.55 + slack) =
400+ tok/s, TTFT ~35 s; needs the layer's supply to fit into the drain window.
Not a duplicate of the denied items: strata C1/C2 (copy-only queue / stream-ahead
thread), staging 64 (deeper banks) and chunk 8192 (wider route window) all keep
the supply route-dependent or move it between existing queues; this removes the
route dependency itself. E7/G5 is the in-flight implementation: this section is
its byte budget and trigger (non-resident set), not a parallel design.

### C2: split route publication into two 2048-row top-k launches

Mechanism: today `router_softmax_topk_device` publishes all 4096 rows in one
launch (`Qwen4ExpBnb4Backend.cpp:2486-2493`) and the single wait at `:2646` holds
the host until the entire device drain finishes; two launches over row halves let
the host start the layer's staging after the first half drains, overlapping the
second half's attention with the supply loop. The per-half use-count machinery
already exists (`Qwen4ExpLayerMajor.cpp:531-548`). Expected: up to
min(supply/2, second-half drain) ~2-4 s/block at best, ~40-80 lines, no new
buffers, bit-identical (the top-k kernel is row-independent).
Difference vs denied: no speculative bytes (unlike C1/E7), no new queue or
thread (unlike strata C1/C2), no chunk-size change; difference vs C1 is the
smaller overlap window (half the drain) with zero extra H2D.

## 6. Probe (one GPU arm, falsifies the section 4 model)

    python tmp/q4e-sprint-r1-20261014/run.py e2-slots8 --prime --diag --generate=8 \
        AILA_Q4E_STAGE_COPY_PARALLEL=1 AILA_Q4E_PREFILL_STAGING_BANKS=4 \
        AILA_Q4E_STAGING_SLOTS=8

`AILA_Q4E_STAGING_SLOTS` is the chunk-size knob for the chunk-major path
(`Qwen4ExpBnb4Backend.cpp:786-793`; the layer-major alias is inert because
layer-major is off). 2,420 chunks of 20.3 MB. If the section 4 fixed cost is real
the cycle is ~5.2 ms/chunk -> supply ~12.6 s (+44% vs e6-banks4-b1); if it is
per-byte the cycle is ~2.05 + fence and the supply stays ~5-8 s. Read
`[Q4E PrefillStage] slots_per_bank=8`, then `fill_ms`, `bank_wait_ms`,
`router_wait_ms`, `submitted_ms`, and `staged=`/`h2d_mb=` for the 4096 block.
A positive result also prices C2's window and closes the "bigger chunks" idea;
a negative result kills the fixed-cost term and leaves C1/G5 as the only lever.

## 7. Files touched

None (audit only). Forbidden files untouched: `Qwen4ExpLayerStage.*`,
`Qwen4ExpExpertExecutor.*`, and the G5-owned consumption path in
`Qwen4ExpBnb4Backend.cpp`.
