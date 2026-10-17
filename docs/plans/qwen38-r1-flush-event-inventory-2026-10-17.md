# Qwen3.8 R1 canonical decode: L0 batch-breaker (flush/event) inventory

Date: 2026-10-17
Scope: read-only source enumeration. No source edits, no GPU run, no commit.
Author role: performance research sub-agent (source audit only).

Canonical basis used here (same arm as the 2026-10-16 structural phase and
`tmp/q4e-structural-20261016/run-knob-screen.ps1`):

- `docs/plans/qwen38-server-mtp-2026-10-07.json` -> `worker_environment`
  (adaptive pipeline is compiled on, `AILA_Q4E_ROUTER_RESIDENT_FUSED=0`,
  `AILA_Q4E_COLD_MODE=2`, `AILA_Q4E_CPU_MAX_EXPERTS=10`,
  `AILA_Q4E_CPU_MMAP_COLD=1`, `AILA_Q4E_CPU_RESULT_HOST_READ=0`,
  `AILA_Q4E_CPU_RESULT_PACKET=0`, `AILA_Q4E_CPU_DEFER_ARENA_WAIT=0`,
  `AILA_Q4E_RESIDENT_PREFILL=1`, hotset profile enabled).
- `tmp/q4e-sprint-r1-20261014/run.py:36` overrides MTP=0, VRAM reserve 3584,
  prefill chunk 4096.
- `tmp/q4e-structural-20261016/run-knob-screen.ps1:28-29` adds
  `AILA_Q4E_HOTSET_PROFILE=perf/qwen4exp/profiles/qwen38-nf4-agent6k-v1.profile`,
  `AILA_KV_QUANT=1`, `AILA_KV_QUANT_FORMAT=int8`, `AILA_SERVER_KV_TYPE=int8`.
- `use_adaptive_pipeline_` is a compile-time `const bool = true`
  (`src/models/Qwen4ExpBnb4Backend.hpp:541`).

Evidence for the breaker taxonomy (probe, same UR/L0 environment):

| arm (32 submits per iteration, no per-submit wait except where stated) | us/call |
|---|---:|
| plain, back-to-back submits | 5.6 (`chain-probe4`) / 5.1 (`chain-probe5`) |
| explicit `depends_on(previous)` | 5.9 / 5.6 |
| submit; host `wait`; next submit | 19.8 / 20.1 |
| submit; `command_execution_status`; next submit | 17.6 / 16.6 |
| 4 B H2D from stack (non-USM) on compute queue | 79.0 / 71.7 |
| 4 B H2D from USM host on compute queue | 22.1 / 22.3 |
| memcpy issued on a second in-order queue | 8.0 / 7.9 |
| device-to-device memcpy on the compute queue | 7.9 / 7.7 |
| compute submit `depends_on` a copy-queue event | - / 8.5 |
| deferred wait on a previous copy-queue event | - / 12.8 |

Sources: `tmp/q4e-structural-20261016/chain-probe4.log` and
`chain-probe5.log`; analysis in
`docs/plans/qwen38-r1-structural-phase-2026-10-16.md:200-270`.

Important scope caveat: the probe varies one operation at a time on one
thread. The inventory below therefore marks cross-thread waits and waits on
the second queue explicitly as "mechanism-implied / not directly measured"
rather than folding them into the measured 20-28 us class.

---

## 0. TL;DR

1. The canonical decode never calls the `device_lease` overload of
   `RunResidentDecode`. `ForwardImpl` calls the 5-argument overload at
   `Qwen4ExpBnb4Backend.cpp:3021`; the lease overload at `:1599` has **no
   call site anywhere in `src/`**. `:1654` is therefore dead in the canonical
   path. The real per-layer 40 B weights H2D is `:1583`, and it runs exactly
   once per layer at `count == 1`.
2. The canonical path has **no per-layer status query**. `:1570`
   (`AILA_Q4E_JOIN_RELEASE`) defaults to 0 and is absent from the canonical
   env; `Qwen4ExpExpertExecutor.cpp:2162` lives in `execute_hetero_decode`,
   whose only callers are `tests/models/Qwen4ExpDmaTimingTests.cpp`. All
   `is_event_complete` sites in `Qwen4ExpExpertCache.cpp` are on the
   compile-time-dead legacy path (`!use_adaptive_pipeline_`) or on the
   non-canonical verify packet/host-read modes.
3. Distinct main-thread flush **sites** per layer: 6 (F1-F6: 3 waits, 1 D2H,
   2 H2D families), plus one worker-thread wait (F7) and one copy-queue wait.
   Distinct flush **operations** per layer is higher: `JoinResidentCold`
   uploads one host-pointer H2D per *contiguous run of CPU-served ranks*
   (`Qwen4ExpExpertExecutor.cpp:1062-1066`), i.e. up to
   `min(miss_count, AILA_Q4E_CPU_MAX_EXPERTS=10)` copies per layer.
   With the audit's ~199 CPU misses/token this is ~150-200 copies/token, so
   the queue is flushed far more often than the "4-6 sites/layer" estimate.
4. The 5 KB copy in the audit's wording is not a "top-k D2H readback": it is
   the resident-input activation D2H (`Qwen4ExpExpertExecutor.cpp:746`,
   2560 x 2 B = 5120 B). Top-k is written by the router kernel directly into
   pinned host memory; its readback is the event wait at `:2822`, not a copy.
5. The only per-layer flush *on the submit thread* that is semantically
   unavoidable today is the router wait at
   `Qwen4ExpBnb4Backend.cpp:2822` (host classification needs the top-k ids).
   Everything else in the list has a known dependency graph that permits
   either elimination (the 40 B weights copy) or a move to `copy_q_` with
   explicit events (activation D2H, CPU rank-row uploads, and the
   corresponding waits).

---

## 1. Why the canonical topology is what it is

| fact | evidence | consequence |
|---|---|---|
| `use_adaptive_pipeline_ = true` | `Qwen4ExpBnb4Backend.hpp:541` | every `!use_adaptive_pipeline_` branch is dead at compile time |
| `resident_epoch` requires `!use_adaptive_pipeline_` | `Qwen4ExpBnb4Backend.cpp:2716-2717` | no `GpuResidentLayerLease`, no `moe_resident_classify`, no `BeginResidentLayer` in canonical |
| dispatch: `count > 4` prefill else resident decode | `Qwen4ExpBnb4Backend.cpp:3008-3021` | `count == 1` -> `RunResidentDecode(ctx,L,fuse,count,verify_rows)` at `:1292` |
| the lease overload has no caller | `rg RunResidentDecode src` shows only definitions `:1292`, `:1599` and the call `:3021` | `:1599-1671` (BindReady/AcquireCold/`:1654`) is dead code in this tree |
| `cpu_overlap_enabled()` = `cpu_worker_ != nullptr` | `Qwen4ExpExpertExecutor.hpp:179`; worker created when `AILA_Q4E_CPU_OVERLAP=1 || AILA_Q4E_COLD_MODE>0` (`Qwen4ExpExpertExecutor.cpp:733`) | canonical `COLD_MODE=2` -> CPU worker exists -> `StartArenaCold` path is live |
| result host-read / result packet off | `Qwen4ExpExpertExecutor.cpp:669-683`, canonical `...=0` | `JoinResidentCold` takes the rank-span H2D branch (`:1062-1066`), not packet/host-read |
| `copy_q_` always created at load | `Qwen4ExpBnb4Backend.cpp:892-893` | `StageColdExpertsAsync` uses the second queue; the `copy_q_ ? : ctx.queue()` fallback at `:1517` never triggers |
| `AILA_Q4E_STAGEIN_HOIST` default false | `Qwen4ExpBnb4Backend.cpp:2521`, canonical env absent | `StageResidentInput` is called from `:2729-2731`, after the attention/MLP-HC submits |
| `AILA_Q4E_JOIN_RELEASE` default false | `Qwen4ExpBnb4Backend.cpp:1317-1318` | the `:1570` status query is off in canonical |
| `AILA_Q4E_CPU_ASYNC_ROUTE=1` is inert | `Qwen4ExpBnb4Backend.cpp:2718` requires `!use_adaptive_pipeline_` | the worker-side route wait (`Qwen4ExpExpertExecutor.cpp:431`) does not replace the main-thread router wait |

Canonical per-layer call chain (the only path this document enumerates):

```
ForwardImpl (count == 1, decode)
  L==1: ple_.consume()                         -> PLE copy_q_ event wait (second queue)
  RunHyperConnectionCombined(attn_hc)          -> submits on ctx.queue()
  RunAttentionWindow / RunAttentionLayerTile   -> submits on ctx.queue()
  RunHyperConnectionCombined(mlp_hc)           -> submits on ctx.queue()
  StageResidentInput (2729-2731)
    745 wait on previous resident_input_event_  (ctx.queue() event)
    746 D2H memcpy x_host_ <- x_dev (5 KB)      (ctx.queue())
  bf16_moe_front_fused (2746)                  -> submit on ctx.queue()
  router_softmax_topk_device (2779-2783)       -> submit on ctx.queue()
  wait_actual_route (2822, called from 2961)   -> host wait on router event
  RunResidentDecode overload :1292 (3021)
    1471 adaptive_cache_.ObserveRouting        -> host only
    1505-1511 StartArenaCold
       927 wait on prev_cpu_h2d_ev_            (ctx.queue() event)
       936-937 job.activation = resident_input_event_ (no new copy)
    1517-1520 StageColdExpertsAsync            -> copy_q_ (615 wait, 656-659 H2D)
    moe_decode_gate_up_swiglu_grouped          -> submit(s) on ctx.queue()
    moe_decode_down_to_rank_f32                -> submit(s) on ctx.queue()
    1570 status query                          -> OFF (JOIN_RELEASE=0)
    1579 JoinResidentCold
       956 cpu_worker_->finish()               -> condvar wait (not a SYCL wait)
       1062-1066 rank-span H2D uploads         -> ctx.queue(), n_runs copies
    1583 40 B weights H2D                      -> ctx.queue()
    1586-1589 moe_ordered_accumulate_down_rows -> submit on ctx.queue()
    1590 FinishResidentCpuRead                 -> no-op (host-read off)
  add_shared_expert_to_y (3040)                -> submit on ctx.queue()
  pending_mlp.Defer() (3047)                   -> host only
```

---

## 2. Per-layer inventory (canonical, `count == 1`)

Legend for queue: **C** = `ctx.queue()` (compute, in-order);
**Q2** = `copy_q_` (second in-order queue, same context/device);
**PLE-Q** = `Qwen4ExpPle::copy_q_` (third in-order queue).

### 2.1 Main-thread batch breakers and their exact wiring

| # | site (file:line) | operation | queue | count / layer at `count==1` | condition / env that changes it | who depends on it | safe to move to Q2 / D2D? |
|---|---|---|---|---|---|---|---|
| F1 | `src/models/Qwen4ExpExpertExecutor.cpp:745` | host wait on the previous `resident_input_event_` (D2H event) | C | 1 (all layers after the first; 48/token steady state) | `!AILA_Q4E_CPU_DEFER_ARENA_WAIT` (canonical 0). `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1` removes it; `AILA_Q4E_STAGEIN_HOIST=1` moves the whole call earlier but keeps both ops | protects the pinned `x_host_` buffer from being overwritten before the previous D2H completed; the CPU worker also waits the same event at `:458` | Not movable as-is (it is a C event). It is redundant when the previous layer ran a CPU job: `JoinResidentCold` -> `finish()` -> worker waited `activation`, and `cold_count_` is already 0. Removing/guarding it is a small, local change; see S3 |
| F2 | `src/models/Qwen4ExpExpertExecutor.cpp:746` | **memcpy-hostptr** D2H `x_host_ <- x_dev`, `rows*hidden*2` = 5120 B | C | 1 (48/token) | same gate as F1 (`cpu_overlap_enabled`, `count==1`, adaptive) | `StartArenaCold` stores this event as `job.activation` (`:936-937`); the CPU worker waits it at `:458`; F1 protects reuse | **Q2 yes** with an explicit C->Q2 dependency (source `buf_.x` is produced by the layer's attention/MLP-HC submits on C). Consumer is the worker, which already takes an event. Blocker: `RunHyperConnectionCombined` returns `void`, so there is no last-event handle today; capture it via a marker or a return-value change. D2D is N/A (host destination). A kernel writing directly to USM host memory is an unprobed alternative |
| F3 | `src/models/Qwen4ExpBnb4Backend.cpp:2822` (`wait_actual_route`, reached from `:2961`) | host wait `actual_route_event.wait_and_throw()` | C | 1 (48/token) | always in adaptive/`count==1`; `resident_epoch` (dead) would wait `moe_resident_classify` instead (`:2784-2788`) | publishes `router_host_ids_` / `router_host_weights_` to the host; gates hit/miss classification, `StartArenaCold`, `StageColdExpertsAsync`, and `bind_actual_route` | **Cannot be removed today**: the kernel writes the top-k ids straight into pinned host memory. Moving it off the main thread requires the legacy async-route design (worker waits `route_event`, `Qwen4ExpExpertExecutor.cpp:431`, started from `Qwen4ExpBnb4Backend.cpp:2834/2837`), which is gated on `!use_adaptive_pipeline_`. This is the one semantically required same-queue wait per layer |
| F4 | `src/models/Qwen4ExpBnb4Backend.cpp:1505-1511` -> `src/models/Qwen4ExpExpertExecutor.cpp:927` | host wait on `prev_cpu_h2d_ev_` (previous layer's CPU rank-row upload event) | C | 1 iff previous layer had `cold_count_>0` and this layer has `miss_count>0` (~48/token in steady state; 199 misses/token per audit) | `count==1 && cpu_overlap_enabled()`; `max_cpu_credits_ = AILA_Q4E_CPU_MAX_EXPERTS` (canonical 10); `AILA_Q4E_COLD_MODE=0` and `AILA_Q4E_CPU_OVERLAP=0` disable the whole path | protects `cpu_out_host_` and the previous upload's destination before reuse; `prev_cpu_h2d_ev_` is also the merge dependency at `:1588` | Once F5 is on Q2 this becomes a **Q2 wait** (chain-probe5 deferred cross-queue wait = 12.8 us/call, not the 20-28 us class), or better: delete it and hand the upload event to the next job's `transfer_reader` (`Qwen4ExpExpertExecutor.cpp:464` already supports it, used by the async-route path). Not a D2D candidate |
| F5 | `src/models/Qwen4ExpExpertExecutor.cpp:1062-1066` (`JoinResidentCold`, canonical else-branch) | **memcpy-hostptr** H2D `cpu_out_host_[run] -> resident_down_[rank]`, 10 KiB per rank, one copy per contiguous rank run | C | `n_runs` where `1 <= n_runs <= min(miss_count, 10)` per layer with CPU-served misses; ranks are ascending route ranks, so isolated misses give `n_runs == cpu_count` (~199 copies/token at the audit's miss rate) | `cpu_result_host_read_ == false` and `cpu_result_packet_host_ == nullptr` (both canonical); `AILA_Q4E_CPU_RESULT_PACKET=1` replaces this with one packet copy at `:1059`; `AILA_Q4E_CPU_RESULT_HOST_READ=1` replaces it with `moe_ordered_accumulate_cpu_host_rows` (no copy, kernel reads host rows) | the merge kernel `moe_ordered_accumulate_down_rows` reads `resident_down_` (`:1588`); the next layer's `StartArenaCold` waits `prev_cpu_h2d_ev_` (`:927`); the same `resident_down_` buffer is reused every layer | **Q2 yes**, two dependencies: (i) the C-side merge event that last read `resident_down_` must gate the Q2 copy (cross-queue `depends_on` is 8.5 us/call per probe, no C batch break); (ii) the merge kernel must `depends_on` the Q2 upload event (it already accepts `&last`). The merge is the next C command after the upload and every later C writer is in-order behind it, so (ii) also protects the next layer's `resident_down_` reuse. Then delete/convert F4. This is the highest-count breaker and the best second target after S1 |
| F6 | `src/models/Qwen4ExpBnb4Backend.cpp:1583` | **memcpy-hostptr** H2D 40 B `router_host_weights_ -> resident_weights_` | C | 1 (48/token, always) | the canonical (`:1292`) overload `count==1` loop `t=0`. The dead `:1654` is the same copy in the lease overload | the merge kernel `moe_ordered_accumulate_down_rows` reads `resident_weights_.data_as<float>()` (`:1586-1589`); nothing else in the `count==1` adaptive path consumes `router_host_weights_` | **Eliminate rather than move** (S1): pass `resident_weights_.data_as<float>()` as the router's weights output for `count==1` at `:2781-2782`, and delete/guard `:1583`. The merge then reads device weights directly. No D2D needed (the source is host only because the router was pointed at the host mirror). If not eliminated, Q2 with a cross-queue dep is safe but keeps 48 copies/token |
| F7 | `src/models/Qwen4ExpExpertExecutor.cpp:458` (`CpuWorker::run`) | host wait `current.activation.wait_and_throw()` | C event, waited on the CPU worker thread | 1 per layer with a CPU job (~48/token) | `job.activation` is `resident_input_event_` (`:936-937`); the worker exists when `COLD_MODE>0`/`CPU_OVERLAP=1` | serialises CPU expert work behind the current layer's D2H; result publication `done` gates `JoinResidentCold` | Mechanism-implied same-queue flush, **not** covered by the single-thread probe. If F2 moves to Q2 this becomes a Q2 wait (12.8 us class, off the submit thread). If F2 stays on C, this wait should be considered part of F2's cost, not a separate fix |

### 2.2 Conditional points present in canonical but off / alternate mode

| site | operation | queue | canonical? | what enables it | note |
|---|---|---|---|---|---|
| `Qwen4ExpBnb4Backend.cpp:1569-1572` | `command_execution_status` query (JOIN_RELEASE) | C | **OFF** | `AILA_Q4E_JOIN_RELEASE != 0` | the explicit "release the batch before the CPU join" workaround; 16.6-17.6 us/call class. Do not turn it on as a fix; S1-S3 remove the need |
| `Qwen4ExpExpertExecutor.cpp:1049-1061` | packet H2D (16 ints + up to 16 rows) | C | **OFF** | `AILA_Q4E_CPU_RESULT_PACKET=1` | same ordering problem as F5; one copy per layer instead of `n_runs`, but still a compute-queue host-pointer copy |
| `Qwen4ExpExpertExecutor.cpp:1044-1048` | host-read result mode (`MoeCpuHostDownRows`) | C (kernel reads host) | **OFF** | `AILA_Q4E_CPU_RESULT_HOST_READ=1` | no memcpy flush, but the merge kernel reads pinned host memory; `FinishResidentCpuRead` (`:1273-1278`) then records the merge event as `cpu_output_reader_event_` |
| `Qwen4ExpExpertExecutor.cpp:463` | wait on `cpu_output_reader_event_` in the worker | C event | **OFF** | only when host-read mode produced an active host result | becomes live if `CPU_RESULT_HOST_READ=1` |
| `Qwen4ExpExpertExecutor.cpp:464` | wait on `transfer_reader` | C event | **OFF** | set by the async-route job (`:777-786`, only with `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1`) and the async verify job (`:1311`) | already the right hook for S2's "retire F4 into the worker" |
| `Qwen4ExpExpertExecutor.cpp:431` | wait on `route_event` | C event | **OFF** | `job.route_ids != nullptr`; only `StartAsyncResidentRoute` (`:770`) / `StartAsyncVerifyRoute` (`:1303`), both gated off by `!use_adaptive_pipeline_` or `count==2` | this is the legacy mechanism that does the router wait off the main thread; a model for S4 |
| `Qwen4ExpBnb4Backend.cpp:2280-2285` | `BeginHotsetForward` / `CpuPackedPromptRefreshPending` | C | **OFF** | whole block under `!use_adaptive_pipeline_` | includes legacy hotset publish/sync logic |
| `Qwen4ExpExpertCache.cpp:1589-1606` (`PublishHotsetPending`) and callers `:1649/:1657/:1676/:1751` | `is_event_complete` polling to gate hotset publish/rebalance | Q2/C | **OFF** | legacy `expert_cache_` hotset only; `BeginHotsetForward`/`RebalanceHotset` are under `!use_adaptive_pipeline_` | see section 3 |
| `Qwen4ExpBnb4Backend.cpp:3084-3089` | `PlanSwaps` + `ExecuteSwapsAsync` | Q2 | **OFF** | `AILA_Q4E_ADAPTIVE_SWAP_FREQ > 0` (default 0) | per-token, not per-layer; `ExecuteSwapsAsync` itself does `resident_ready_.wait_and_throw()` and per-slot waits (`Qwen4ExpAdaptiveCache.cpp:569-593`) if ever enabled |
| `Qwen4ExpBnb4Backend.cpp:2669-2680` / `3015` | layer-stage begin / `WaitReady` | Q2 + C | **OFF** | `AILA_Q4E_LAYER_STAGE=1` and `is_prefill && count>4` | prefill only |

### 2.3 Sites that were checked and are NOT breakers in canonical decode

| site | why not |
|---|---|
| `Qwen4ExpBnb4Backend.cpp:2779-2783` `router_softmax_topk_device` | pure submit; the op writes ids/weights into USM host memory from the kernel. No host copy, no status query, no internal wait (`src/ops/MoeExpertOps.cpp:89-285`) |
| `Qwen4ExpBnb4Backend.cpp:2784-2788` `moe_resident_classify` | requires `resident_epoch`, which requires `!use_adaptive_pipeline_`; unreachable here |
| `Qwen4ExpBnb4Backend.cpp:1471` `adaptive_cache_.ObserveRouting` | host-only counters (`Qwen4ExpAdaptiveCache.cpp:430-450`) |
| `Qwen4ExpBnb4Backend.cpp:1517-1520` -> `Qwen4ExpAdaptiveCache.cpp:656-659` | H2D host-pointer copy but issued on `copy_q_`; 8.0 us class per probe. Caveat: the `copy_q_ ? : ctx.queue()` fallback at `:1517` would make it a C-queue breaker, but `copy_q_` is created unconditionally at load (`:892-893`) |
| `Qwen4ExpAdaptiveCache.cpp:615` | wait on `staging_copy_[pp]`, a Q2 event. Cross-queue; 12.8 us class deferred wait per `chain-probe5`, not the C-queue 20-28 us class. One per layer that re-enters the same ping-pong bank |
| `Qwen4ExpExpertExecutor.cpp:956` `cpu_worker_->finish()` | condvar wait; no SYCL call (`Qwen4ExpExpertExecutor.cpp:186-199`) |
| `Qwen4ExpBnb4Backend.cpp:1590` -> `Qwen4ExpExpertExecutor.cpp:1273-1278` `FinishResidentCpuRead` | early-returns because `cpu_host_result_active_ == false` (host-read off); no device op |
| `Qwen4ExpBnb4Backend.cpp:1586-1589` | `moe_ordered_accumulate_*` are pure submits (`src/ops/MoeExpertOps.cpp:6357-6451`); `depends_on` is free (+0.3 us in probe) |
| `Qwen4ExpBnb4Backend.cpp:2746` `bf16_moe_front_fused` | pure submit (`src/ops/Bnb4BitLinear.cpp:2793`) |
| `Qwen4ExpBnb4Backend.cpp:3040` `add_shared_expert_to_y` | host-side wrapper around a submit (`:125`) |
| `Qwen4ExpBnb4Backend.cpp:2126-2245` HC (attn/mlp/mixer) | `HyperConnectionOps.cpp`, `HcDecodeVariant.cpp` contain no `memcpy`/`wait`/`get_info`/`synchronize` (checked); all submits |
| `Qwen4ExpPrefill.cpp:63-91` linear decode, `:143-...` full-attention decode | all submits (`packed_nf4_*`, `run_linear_delta_decode_gpu`, `decode_prepare_qgkv_packed_partial`, QSA selects, `attention_decode`); `AttentionOps.cpp`, `GatedDeltaNetOps.cpp`, `QsaIndexerOps.cpp`, `ElementwiseOps.cpp`, `Qwen4ExpDecodeFusion.cpp` contain no `memcpy`/`wait`/`get_info`/`synchronize`. `GdnChunkOps.cpp:169` has one `ctx.synchronize()`, but it is `GdnChunkScratch::Reserve` growth-only and the chunk path itself is prefill-only (`AILA_Q4E_GDN_CHUNK_MODE=0` in canonical, `Qwen4ExpPrefill.cpp:111-119`) |
| `Bnb4BitLinear.cpp:2306-2380`, `:1419-1616`, `:2165-2207` | decode projections are pure submits; the only `ctx.synchronize()` calls in the file are load-time (`:1787/1995/2002/2034/2211`), prefill dequant fallback (`:2525`), and LoRA scratch growth (`:2111`), none live in canonical decode |
| `Qwen4ExpBnb4Backend.cpp:1017-1030` `ensure_runtime_buffers` | early-returns at `:1024` when capacity is unchanged; the `ctx.synchronize()` at `:1030` is growth-only (startup/prefill) |
| `Qwen4ExpBnb4Backend.cpp:3170-3172` `CaptureSnapshot` | only when `SnapshotPolicy::ReusablePrefix`; for decode it also early-returns unless `nextLength % AILA_Q4E_PREFILL_STEP(64) == 0` (`:3566-3568`). Not a per-layer point; see section 6 |

---

## 3. `is_event_complete` call-site audit (`Qwen4ExpExpertCache.cpp`)

`is_event_complete` itself is `:46-53`; every use is a
`command_execution_status` query. The canonical decode does **not** execute
any of the 1209/1231/1270/1591/2231/2744/3318/3386 sites:

| line | enclosing function | reached from canonical decode? | why / which gate |
|---:|---|---|---|
| 1209, 1211 | `pick_victim_hslot` (`:1171-1265`) | **No** | only called from `acquire_impl` (`:2528`, `:2538`, `:2838`) and `reserve_host_preparation` (`:3928`); all legacy `expert_cache_` acquisition paths |
| 1231, 1238 | `pick_victim_hslot` | **No** | same |
| 1270 | `host_has_dma_in_flight` (`:1266-1272`) | **No** | helper has no caller in the tree (dead legacy helper) |
| 1591 | `PublishHotsetPending` (`:1589-1606`) | **No** | called from `InstallHotset`/`RebalanceHotset`/`HandoffPromptHotset`/`EndHotsetForward`; those are under `!use_adaptive_pipeline_` (`Qwen4ExpBnb4Backend.cpp:2280-2285`, `:3174-3178`) or legacy `expert_cache_` use |
| 2231 | `acquire_impl` (`:2092-2787`) | **No** | legacy `Qwen4ExpExpertCache::acquire`/`BindReadyImpl` only |
| 2744 | `acquire_impl` | **No** | same |
| 3318 | `get_slot_state` (`:3315-3323`) | **No** | only inspect/debug callers; not on the decode submit path |
| 3386 | `query_residency` (`:3367-`) | **No** in canonical | `query_residency` is called by `StartResidentCold` (`Qwen4ExpExpertExecutor.cpp:854`) and `StartAsyncResidentRoute` (`:798`), both non-canonical (dead lease overload / async route gated by `!use_adaptive_pipeline_`, hotset swaps == 0) |
| 4078, 4091 | `ensure_gpu` (`:4064-4093`) | **No** | legacy `expert_cache_` demand path only |
| 4113 | `update_dma_profiling` (`:4110-4132`) | **No** | diagnostics accessor only |

The same conclusion holds for the two named "suspects":

- `Qwen4ExpBnb4Backend.cpp:1570` is **off by default** (`AILA_Q4E_JOIN_RELEASE=0`)
  and lives in the *canonical* `:1292` overload.
- `Qwen4ExpExpertExecutor.cpp:2162` is inside
  `execute_hetero_decode` (`:1817-2191`), whose only callers in the repo are
  `tests/models/Qwen4ExpDmaTimingTests.cpp`; `src/` has no caller.

---

## 4. Router / attention / HC / linear segment: breaker check

| segment | entry point | submits per layer (count==1, typical) | any memcpy / wait / status? |
|---|---|---:|---|
| attn-HC | `RunHyperConnectionCombined(ctx, layer.attn_hc, 1, true, ...)` (`Qwen4ExpBnb4Backend.cpp:2633-2636`) | 2-3 (norm/inject fused, down, up) | **none** (`HyperConnectionOps.cpp`, `HcDecodeVariant.cpp` clean) |
| linear attention (36 layers) | `RunAttentionLayerTile` `count==1` (`Qwen4ExpPrefill.cpp:63-91`) | 2 projections + `run_linear_delta_decode_gpu` (+ optional history commit fused) | **none** (`GatedDeltaNetOps.cpp` clean) |
| full attention (12 layers) | `RunAttentionLayerTile` `count==1` (`Qwen4ExpPrefill.cpp:143-...`) | qkv GEMV + `decode_prepare_qgkv_packed_partial` + QSA selects + `attention_decode` | **none** (`AttentionOps.cpp`, `QsaIndexerOps.cpp` clean); KV-int8 writes are kernel-side |
| MLP-HC | `RunHyperConnectionCombined(ctx, layer.mlp_hc, 1, true, y, inject)` (`:2686-2688`) | 2-3 | **none** |
| resident-input staging | `StageResidentInput` (`:2729-2731`) | 1 wait + 1 D2H | **F1/F2 above** |
| MoE front | `bf16_moe_front_fused` (`:2746`) | 1 | none |
| router | `router_softmax_topk_device` (`:2779`) | 1 | none (the wait is F3) |
| resident/cold experts | `moe_decode_gate_up_swiglu_grouped`, `moe_decode_down_to_rank_f32` (`:1526-1564`) | 2 + 2 (hits and/or cold) | none (`MoeExpertOps.cpp` launch functions clean) |
| rank merge | `moe_ordered_accumulate_down_rows` (`:1586-1589`) | 1 | none (submit only) |
| shared expert | `try_forward_decode_gate_up_swiglu` + `ForwardDecodeRows` (`:2907-2914`) | 2 | none (`Bnb4BitLinear.cpp` packed GEMV / fused SwiGLU are submits) |
| LM head (per token, not per layer) | `lm_head_.forward` (`:3123`) | 1 | no per-call sync; `Linear::init` sync is load-time (`Linear.cpp:46`) |

Conclusion for the requested segments: **no additional L0 batch breaker is
hiding in attention, HC, or Bnb4BitLinear decode submission**. The per-layer
cost is entirely in the MoE coordination / staging / router readback described
above.

---

## 5. Answers to the two questions

### (a) Does the adaptive pipeline always call `RunResidentDecode` with a `device_lease`? Does the 40 B H2D at `:1654` happen every layer in canonical?

**No, and no.**

- `ForwardImpl` has exactly one `RunResidentDecode` call:
  `Qwen4ExpBnb4Backend.cpp:3021`:
  `RunResidentDecode(ctx, L, s_fuse_shared, count, options.decode_verification)`.
  That argument list binds to the 5-parameter overload at `:1292`
  `(Context&, int, bool, int, bool)`, not to the lease overload at `:1599`
  `(Context&, int, bool, GpuReadyLease*, ResidentColdWork*)`.
- The lease overload `:1599-1671` has **no call site** in `src/`
  (only its declaration at `Qwen4ExpBnb4Backend.hpp:561-563` and the test
  harness does not use it). `Qwen4ExpExpertCache::BindReady`
  (`Qwen4ExpExpertCache.cpp:1471`) is therefore not called from decode
  either; its only `src/` caller is the lease overload's `:1604`, plus the
  non-canonical `ExecuteVerifyRows` (`Qwen4ExpExpertExecutor.cpp:1334`).
- Consequently `:1654` never executes in canonical, and the 40 B weights
  H2D that *does* execute is `:1583`, once per layer at `count == 1`
  (48 copies/token).

### (b) In the `resident_epoch` path, is `resident_weights_` (device) already written correctly by the router? Can `:1654` be skipped or turned into D2D?

**Yes for the legacy `resident_epoch` path; the canonical adaptive path is different.**

- Legacy `resident_epoch` (`resident_decode_mode_ == 2` and
  `!use_adaptive_pipeline_`): the top-k kernel is pointed at
  `resident_weights_.data_as<float>()` (`Qwen4ExpBnb4Backend.cpp:2772-2783`):
  - `AILA_Q4E_ROUTER_RESIDENT_FUSED=1`: `router_softmax_topk_resident`
    (`MoeExpertOps.cpp:287-300`) writes ids, weights and slot classification
    in one kernel; `resident_weights_` is a device output.
  - `...=0`: the split path calls `router_softmax_topk_device` with
    `resident_epoch ? resident_weights_ : router_host_weights_` and then
    `moe_resident_classify` publishes the host mirrors.
  In both cases the device buffer is correct before `RunResidentDecode`, so
  the `!device_lease` guard at `:1654` correctly skips a redundant H2D.
- Canonical adaptive: `resident_epoch` is null by construction
  (`:2716` requires `!use_adaptive_pipeline_`), and the router is called with
  `resident_epoch ? resident_weights_ : router_host_weights_` = 
  `router_host_weights_` (host, `:2781-2782`). `resident_weights_` (device)
  is **not** written by the router, so `:1583` is load-bearing. It cannot be
  simply deleted, and there is nothing to D2D from (the source is host).
  The clean fix is to pass `resident_weights_.data_as<float>()` as the weights
  output for `count==1` (one-line change at `:2782`) and drop `:1583`.
  `resident_weights_` is already allocated in the adaptive pipeline
  (`:938-940`), is `top_k` floats, and the only consumer is the merge kernel
  (`:1586-1589`). Host weights are still needed by the `count>4` prefill path
  (`:1779`) and by MTP, so gate the redirect to `count==1` (or write both via
  a second pointer argument).

---

## 6. Per-token points outside the layer loop (context, not counted per layer)

These do not change the per-layer table but do flush the same compute queue
once per generated token:

| site | operation | queue | condition |
|---|---|---|---|
| `Qwen4ExpBnb4Backend.cpp:2047` (`EmbedTokens`) | blocking **memcpy-hostptr** D2H token ids (4 B) + wait | C | `AILA_Q4E_HOST_EMBED=1` (default; canonical) |
| `Qwen4ExpBnb4Backend.cpp:2058` (`EmbedTokens`) | **memcpy-hostptr** H2D embedding row (5 KB) | C | same |
| `Qwen4ExpPle.cpp:692-694` (`prepare`) | wait on the PLE's own `copy_q_` event | PLE-Q | once per decode forward (lookahead is prefill-only) |
| `Qwen4ExpPle.cpp:752-759` (`consume`, `L==1`) | worker-task condvar + wait on the PLE `copy_q_` event | PLE-Q | once per decode forward |
| `Qwen4ExpPle.cpp:587-590` | PLE H2D on its own `copy_q_` | PLE-Q | once per decode forward |
| `Qwen4ExpBnb4Backend.cpp:3170-3172` -> `3566-3589` | `ctx.synchronize()` + linear-state D2H snapshot | C | only `SnapshotPolicy::ReusablePrefix` and `cached_len % 64 == 0` (every 64th token); not the steady per-layer path |
| sampling / head readback (`src/ops/SamplingOps.cpp:378/391/459`) | blocking D2H of logits/token | C | once per token, after the final mixer/head |

Together with the per-layer list, this means the canonical decode currently
has on the order of **400 flush-class queue operations per token**
(~3 waits + 1 D2H + ~1 weights H2D + ~4 CPU-row H2Ds per layer, plus the
router wait), not 4-6. That is consistent with every measured HC submit
landing in the 17-29 us drained regime even when consecutive HC submits are
<10 us apart.

---

## 7. Suggested minimal change sequence (for construction, not applied here)

Order is chosen by (effect x evidence) / risk. Each step is independently
measurable on the canonical arm; do not combine more than one step per A/B.

### S1. Delete the per-layer 40 B weights H2D (bit-identical, smallest change)

Site: `Qwen4ExpBnb4Backend.cpp:2781-2782` (router weights target) and
`Qwen4ExpBnb4Backend.cpp:1583` (the H2D).

Change sketch:

- For the adaptive `count==1` decode call, pass
  `resident_weights_.data_as<float>()` as the `expert_weights_dev` argument
  of `router_softmax_topk_device`.
- Delete (or guard to `count>1`) the `ctx.queue().memcpy(...)` at `:1583`.
- Keep `router_host_weights_` for the prefill path (`:1779`) and MTP; gate the
  redirect on `count == 1`.

Why it is safe: same kernel writes the same float values, just into the device
buffer the merge kernel actually reads; no arithmetic change; no dependency
change (the merge already runs after the topk submit on the same in-order
queue). It removes 48 host-pointer memcpys/token on the compute queue.

Verification: `AILA_Q4E_HC_SUBMIT_TIMING=1` window should show the per-call
min moving down toward the batched regime; bit-identical greedy output on the
19-token continue arm; 1.05% A/A noise floor from the 2026-10-16 doc.

### S2. Move the CPU rank-row uploads to `copy_q_` and retire the next-layer wait

Site: `Qwen4ExpExpertExecutor.cpp:1062-1066` (uploads), `:927` (next-layer
wait), `:1588` (merge dependency).

Change sketch:

- Keep a member event `resident_down_reader_` = the merge event returned by
  `RunResidentDecode` (the caller at `:3021` currently ignores the return
  value).
- In `JoinResidentCold`, submit the per-run copies on `copy_q_`, each
  `depends_on(resident_down_reader_)`; return the last upload event.
- Keep `prev_cpu_h2d_ev_` as the upload event, but pass it to the next job as
  `job.transfer_reader` (the worker already waits this at `:464`) instead of
  waiting on the main thread at `:927`. That moves F4 off the submit thread.
- The merge kernel at `:1588` already takes `&last`; no signature change. Because the merge is the next C command after the upload and every later C writer is in-order behind the merge, `depends_on(upload)` also protects the next layer's `resident_down_` reuse.

Why it is safe: the only cross-queue edges are (previous merge -> upload) and
(upload -> merge), both expressible with `depends_on`; the probe measured a
cross-queue `depends_on` at 8.5 us/call with no compute-batch break. This
removes `n_runs` (1..10) host copies and the F4 wait from the compute queue
per layer. Highest operation-count reduction after S1.

### S3. Move the resident-input D2H to `copy_q_` and make its reuse wait redundant

Site: `Qwen4ExpExpertExecutor.cpp:745-746`.

Change sketch:

- Track the last compute-queue event of the layer's attention/MLP-HC chain
  (make `RunHyperConnectionCombined` return its last event, or submit one
  marker event) and issue the 5 KB D2H on `copy_q_` with
  `depends_on(last_compute_event)`.
- The CPU worker already waits the same event at `:458`; it will simply wait a
  Q2 event, which is the cheaper flavor (12.8 us class, off the submit
  thread).
- The F1 wait at `:745` becomes unnecessary whenever the previous layer's CPU
  job completed (which is guaranteed by the `cold_count_` guard at `:741`);
  make it conditional on `resident_input_event_consumed_` or, if the previous
  layer had no CPU job, leave it.

Expected: removes 1 host wait + 1 host-pointer D2H per layer from the compute
queue; leaves only cross-queue waits/deps.

### S4. Keep exactly one router wait; only then consider async routing

`Qwen4ExpBnb4Backend.cpp:2822` is the route-publication point. Do not try to
move it to `copy_q_` (it is a host wait on a compute-queue kernel by design).
The only way to take it off the main thread is to port the legacy
`AILA_Q4E_CPU_ASYNC_ROUTE` mechanism (`Qwen4ExpExpertExecutor.cpp:770-810`,
`:431`) into the adaptive path: the worker waits the router event and
classifies the cold set, while the main thread submits the resident hit
kernels. That is a larger, separate change; S1-S3 should be measured first.

### S5. Do not use `AILA_Q4E_JOIN_RELEASE=1` as the fix

The status query at `:1570` is 16.6-17.6 us/call class and its only purpose is
to force the L0 batch to execute during the CPU join. Once S1-S3 remove the
other flushes, the batch can stay open without it. Keep it off; it is a
diagnostic, not a production path.

Operational notes for any of these runs (from the 2026-10-16 doc):
one GPU occupant at a time; wait for `AilaWorker`/`AilaServer` to exit; gate
on >=45 GiB free physical memory because the previous ~52 GiB large-page
arena can still be held; sleep ~15 s between arms; rebuild both the worker and
the server for handshake identity. Use `--production-timing` for the
production arm and `AILA_Q4E_HC_SUBMIT_TIMING=1` for the mechanism arm.

---

## 8. Verification plan for the future GPU runs (not executed here)

1. **Mechanism gate (per step):** canonical arm + `AILA_Q4E_HC_SUBMIT_TIMING=1`.
   Success = `hc_submit_*_min_us` and `mean_us` move toward 5-8 us, and the
   gap-bucket distribution no longer has every bucket at ~20 us.
2. **E2E gate:** 19-token continue, agent6k + int8 KV, at least A/A pairs
   around the new arm; the documented A/A floor is 1.05% (`16.372/16.202`
   tok/s) and the n=128 baseline is 17.15 tok/s.
3. **Correctness:** greedy 32-token output must be bit-identical for S1
   (weights pointer only) and for S2/S3 (same bytes, different queue); for
   S2/S3 also verify the cancel/timeout path drains `copy_q_` before
   `resident_down_` / `x_host_` reuse (`AilaWorker` cancellation suite).
4. **Safety:** S2/S3 add cross-queue dependencies; run the existing
   cancellation and expert-readiness tests before any GPU acceptance.

---

## 9. Mechanical reproduction of this audit

```powershell
# canonical bundle (what the structural-phase knob arm used)
Get-Content docs/plans/qwen38-server-mtp-2026-10-07.json -Raw |
  ConvertFrom-Json | Select-Object -ExpandProperty worker_environment
Get-Content tmp/q4e-structural-20261016/run-knob-screen.ps1 | Select-Object -Skip 27 -First 5

# dead/canonical overload resolution
rg -n "RunResidentDecode" src
# -> definitions Qwen4ExpBnb4Backend.cpp:1292 / :1599, call :3021

# all compute-queue copies/waits in the backend/executor
rg -n "memcpy|wait_and_throw|get_info<sycl::info::event|command_execution_status|synchronize" `
   src/models/Qwen4ExpBnb4Backend.cpp src/models/Qwen4ExpExpertExecutor.cpp `
   src/models/Qwen4ExpAdaptiveCache.cpp

# prove the ops segments are submit-only
rg -n "memcpy|wait_and_throw|command_execution_status|synchronize" `
   src/ops/HyperConnectionOps.cpp src/ops/HcDecodeVariant.cpp `
   src/ops/AttentionOps.cpp src/ops/GatedDeltaNetOps.cpp `
   src/ops/QsaIndexerOps.cpp src/ops/ElementwiseOps.cpp `
   src/models/Qwen4ExpDecodeFusion.cpp
```

Receipts used:
`tmp/q4e-structural-20261016/chain-probe4.log`,
`tmp/q4e-structural-20261016/chain-probe5.log`,
`tmp/q4e-bottleneck-audit-20261016/REPORT.md` (decode ledger and 199
misses/token).
