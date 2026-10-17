# Qwen3.8 R1 canonical decode: L0 flush inventory + copy-q residual breakers

Date: 2026-10-17
Base: HEAD `5a5138d` plus the uncommitted copy-q worktree changes present at
audit time. Read-only audit: no source edits by this audit, no GPU run.

Gate: `AILA_Q4E_MOE_COPY_VIA_COPYQ`, read once, default **off**
(`src/models/Qwen4ExpMoeCopyQueue.hpp:23-27`). Sections 1-2 are the
canonical bundle (gate off); section 3 is the gate-on residual.

Canonical bundle: `docs/plans/qwen38-server-mtp-2026-10-07.json`
`worker_environment` (`COLD_MODE=2`, `CPU_MAX_EXPERTS=10`,
`CPU_MMAP_COLD=1`, `CPU_RESULT_HOST_READ=0`, `CPU_RESULT_PACKET=0`,
`CPU_DEFER_ARENA_WAIT=0`, `ROUTER_RESIDENT_FUSED=0`,
`RESIDENT_PREFILL=1`, hotset on) plus
`tmp/q4e-sprint-r1-20261014/run.py:36` (MTP=0, VRAM 3584, chunk 4096)
and `tmp/q4e-structural-20261016/run-knob-screen.ps1:28-29`
(agent6k hotset profile, int8 KV). `use_adaptive_pipeline_` is
`const bool = true` (`src/models/Qwen4ExpBnb4Backend.hpp:547`).

Probe evidence (`tmp/q4e-structural-20261016/chain-probe4.log`,
`chain-probe5.log`; 32 back-to-back submits per iteration):

| arm | us/call |
|---|---:|
| plain | 5.6 / 5.1 |
| `depends_on(previous)` | 5.9 / 5.6 |
| submit; host wait | 19.8 / 20.1 |
| submit; `command_execution_status` | 17.6 / 16.6 |
| 4 B H2D, stack source, compute queue | 79.0 / 71.7 |
| 4 B H2D, USM host source, compute queue | 22.1 / 22.3 |
| memcpy on second in-order queue | 8.0 / 7.9 |
| D2D memcpy on compute queue | 7.9 / 7.7 |
| compute submit `depends_on` copy-queue event | - / 8.5 |
| deferred wait on copy-queue event | - / 12.8 |

The probe covers same-thread same-queue ops. Cross-thread waits and
cross-queue waits are marked "mechanism-implied" below, not folded into the
20-28 us class.

---

## 1. Canonical topology (gate off)

| fact | current evidence | consequence |
|---|---|---|
| adaptive pipeline is compile-time on | `Qwen4ExpBnb4Backend.hpp:547` | all `!use_adaptive_pipeline_` branches are dead |
| one decode dispatch | `Qwen4ExpBnb4Backend.cpp:3056` -> `RunResidentDecode` at `:1298` | the lease overload at `:1624` is not called |
| lease overload has no caller | `src/` grep | `BindReady` and the `:1679` 40 B H2D are dead |
| `resident_epoch` needs `!use_adaptive_pipeline_` | `Qwen4ExpBnb4Backend.cpp:2751` | no `moe_resident_classify` in canonical |
| CPU worker exists | `Qwen4ExpExpertExecutor.cpp:734` (`COLD_MODE=2`) | `StartArenaCold`/CPU complement is live |
| result host-read/packet off | `Qwen4ExpExpertExecutor.cpp:670-684`, canonical 0 | `JoinResidentCold` uses rank-span H2D |
| `copy_q_` always created | `Qwen4ExpBnb4Backend.cpp:893` | staging DMA uses the second queue |
| `STAGEIN_HOIST` off | `Qwen4ExpBnb4Backend.cpp:2549` | `StageResidentInput` is the `:2765` call |
| `JOIN_RELEASE` off | `Qwen4ExpBnb4Backend.cpp:1324` | the `:1575-1578` status query is not executed |
| router wait | `Qwen4ExpBnb4Backend.cpp:2857`, called at `:2996` | one same-queue host wait per layer |

Per-layer chain (canonical, `count == 1`):

```
attn HC -> attention -> mlp HC (RunHyperConnectionCombined, :2719)
 -> StageResidentInput (:2765) -> :766 wait + :773 D2H
 -> MoE front -> router -> :2857 wait
 -> RunResidentDecode (:1298)
    :1477 ObserveRouting; :1516 StartArenaCold -> :955 wait
    :1524 StageColdExpertsAsync (copy_q_)
    :1594 JoinResidentCold -> :981 finish -> :1099-1108 rank-span H2D
    :1604/1607 weights H2D -> :1613 merge
 -> shared add (:3075)
```

---

## 2. Per-layer flush inventory (gate off, `count == 1`)

Queue: **C** = `ctx.queue()`; **Q2** = `copy_q_`.

| # | current site | operation | queue | count/layer | condition | depends on / consumer | safe to Q2 or D2D? |
|---|---|---|---|---|---|---|---|
| F1 | `Qwen4ExpExpertExecutor.cpp:766` | host wait on previous `resident_input_event_` | C | 1 (48/token steady) | `CPU_DEFER_ARENA_WAIT=0`; hoist does not remove it | protects `x_host_` reuse; worker also waits the same event at `:459` | not movable as-is; redundant when the previous layer ran a CPU job (worker waited `:459`, join `:981`); keep on all-hit layers |
| F2 | `Qwen4ExpExpertExecutor.cpp:773-774` | **memcpy-hostptr** D2H `x_host_ <- x_dev`, 5120 B | C (gate on: Q2) | 1 (48/token) | CPU overlap + `count==1`; called at `Qwen4ExpBnb4Backend.cpp:2765`, hoisted at `:2740` | worker `job.activation` (`:964-966`), then `:459` | Q2 yes with the mlp-HC producer event; see section 3 |
| F3 | `Qwen4ExpBnb4Backend.cpp:2857` | host wait `actual_route_event` | C | 1 (48/token) | always adaptive/`count==1` | publishes `router_host_ids_`; gates classification/`StartArenaCold` | **cannot remove**: topk writes host ids; the only off-thread alternative is the legacy async-route path (`Qwen4ExpExpertExecutor.cpp:432`) |
| F4 | `Qwen4ExpExpertExecutor.cpp:955` | host wait on `prev_cpu_h2d_ev_` | C (gate on: Q2) | 1 iff previous layer had CPU misses and this layer has `miss_count>0` | called from `Qwen4ExpBnb4Backend.cpp:1516` | protects `cpu_out_host_`; previous upload event | becomes Q2 wait with gate on; semantically required until moved to the worker (section 3.4) |
| F5 | `Qwen4ExpExpertExecutor.cpp:1099-1108` | **memcpy-hostptr** rank-span H2D, 10 KiB/rank, one per contiguous run | C (gate on: Q2) | `1..min(miss,10)` per CPU layer (~199 copies/token at the audit's rate) | result host-read/packet off; call `Qwen4ExpBnb4Backend.cpp:1594`; packet `:1088-1089` | merge `:1613`; next `StartArenaCold` wait `:955`; same buffer every layer | Q2 yes; gate-on already uses one coalesced `host_copy_fence` (`:1095-1098`) from `device_last` (`Qwen4ExpBnb4Backend.cpp:1588-1594`) |
| F6 | `Qwen4ExpBnb4Backend.cpp:1607` (gate off) / `:1604-1605` (gate on) | **memcpy-hostptr** 40 B weights H2D | C (gate on: Q2) | 1 (48/token) | canonical `:1298` loop `t=0`; `:1679` is the dead lease copy | merge reads `resident_weights_` (`:1613`) | best: eliminate by pointing the router weights output at device `resident_weights_` for `count==1` (`:2817`) and drop the copy; no D2D source |
| F7 | `Qwen4ExpExpertExecutor.cpp:459` | host wait on `job.activation` | C event, worker thread | 1 per CPU layer (~48/token) | worker exists (`:734`) | serializes CPU work behind F2; `finish()` (`:981`) gates join | mechanism-implied same-queue flush; gate on makes it a Q2 wait (12.8 us class, off the submit thread) |

Off / non-canonical per-layer points:

| site | operation | why off |
|---|---|---|
| `Qwen4ExpBnb4Backend.cpp:1575-1578` | JOIN_RELEASE status query | `AILA_Q4E_JOIN_RELEASE=0` |
| `Qwen4ExpExpertExecutor.cpp:1088-1089` | result-packet H2D | `CPU_RESULT_PACKET=0` |
| `Qwen4ExpExpertExecutor.cpp:1073-1077` | host-read merge | `CPU_RESULT_HOST_READ=0`; `FinishResidentCpuRead` `:1323-1328` is then a no-op |
| `Qwen4ExpExpertExecutor.cpp:464-465` | worker output/transfer-reader waits | host-read/async-route only |
| `Qwen4ExpExpertExecutor.cpp:432` | route-event wait | async-route only, gated by `!use_adaptive_pipeline_` |
| `Qwen4ExpExpertExecutor.cpp:2233` | `command_execution_status` | inside `execute_hetero_decode` (`:1868`); only `tests/models/Qwen4ExpDmaTimingTests.cpp` calls it |

`Qwen4ExpExpertCache.cpp` `is_event_complete` sites (`:46-53`): 1209/1211,
1231/1238 (`pick_victim_hslot`), 1270 (`host_has_dma_in_flight`, no caller),
1591 (`PublishHotsetPending`), 2231/2744 (`acquire_impl`), 3318
(`get_slot_state`), 3386 (`query_residency`), 4078/4091 (`ensure_gpu`),
4113 (`update_dma_profiling`) are all on the dead legacy path or the
non-canonical verify/host-read modes. None runs in canonical decode.

### Questions (a)/(b)

**(a)** `ForwardImpl` calls `RunResidentDecode(ctx,L,s_fuse_shared,count,
options.decode_verification)` at `Qwen4ExpBnb4Backend.cpp:3056`. That binds to
the 5-argument overload at `:1298`, never to the lease overload at `:1624`
(no call site in `src/`). `:1654` is now `:1679` and is dead. The real
per-layer 40 B H2D is `:1607` (gate off) / `:1604-1605` (gate on).

**(b)** In the legacy `resident_epoch` path (`:2751`), the router writes device
`resident_weights_` directly (`:2806-2822`), so `:1679` is redundant and is
correctly guarded by `!device_lease`. In canonical adaptive, `resident_epoch`
is null and the router target is the host mirror (`:2817`); device
`resident_weights_` is stale, so `:1607` is load-bearing. It can be eliminated
by passing `resident_weights_.data_as<float>()` as the weights target for
`count==1` and dropping the copy; `resident_weights_` is allocated at `:941`;
the prefill host path still uses `:1804`.

---

## 3. `copy_q` 后残余断点

This section is `AILA_Q4E_MOE_COPY_VIA_COPYQ=1`. Helper:
`src/models/Qwen4ExpMoeCopyQueue.hpp:42-52`; with an `order_fence` it uses
that event (`:46-48`); without one it submits
`ctx.queue().ext_oneapi_submit_barrier()` (`:49-51`). `host_copy_fence`
(`:57-59`) coalesces one barrier per batch.

### 3.1 What actually moved

| point | gate off | gate on |
|---|---|---|
| StageResidentInput x D2H | C memcpy `Qwen4ExpExpertExecutor.cpp:773-774` | Q2 copy with the **mlp-HC producer event**, no barrier (`:773-774` + `MoeCopyQueue.hpp:46-48`) |
| CPU rank-span H2D | C copies `Qwen4ExpExpertExecutor.cpp:1099-1108` | Q2 copies; one coalesced C fence per batch (`:1095-1098`), fed by the expert `device_last` (`Qwen4ExpBnb4Backend.cpp:1588-1594`) |
| weights H2D | C copy `Qwen4ExpBnb4Backend.cpp:1607` | Q2 copy `:1604-1605` with `copy_order` = expert `device_last`; if no GPU expert kernel (all CPU-served), `copy_order=nullptr` -> one barrier |
| packet H2D | C copy `Qwen4ExpExpertExecutor.cpp:1088-1089` | Q2 copy with `order_dep`; off in canonical |
| F1 wait | C event | Q2 event (12.8 us deferred class) |
| F4 wait | C event | Q2 event (12.8 us deferred class) |
| F3 router wait | C event | unchanged C event |

### 3.2 Can the StageResidentInput barrier be removed?

**Yes, and the current worktree already removes it for canonical
`count==1`.**

- Producer chain: `ForwardImpl` calls the mlp-HC
  `RunHyperConnectionCombined` at `Qwen4ExpBnb4Backend.cpp:2719`. In the
  canonical branch (`count==1`, `AILA_Q4E_HC_FUSED=1`,
  `AILA_Q4E_DECODE_FUSED_HC=0`, `:2237-2252`), the last kernel writing
  `buf_.x` is `ops::hc_decode_up_mix` at `:2247-2251`.
- `hc_decode_up_mix` now returns `sycl::event`
  (`src/ops/HyperConnectionOps.cpp:742-748`, decl
  `src/ops/HyperConnectionOps.hpp:86`).
- `RunHyperConnectionCombined` returns `std::optional<sycl::event>`
  (`Qwen4ExpBnb4Backend.cpp:2151`, decl `Qwen4ExpBnb4Backend.hpp:479`);
  `RunHyperConnection` mirrors it (`:2275-2277`, `hpp:476`).
- `ForwardImpl` captures `mlp_hc_ev` at `:2717-2727` and passes
  `mlp_hc_ev ? &*mlp_hc_ev : nullptr` to `StageResidentInput` at
  `:2740-2741` (hoisted) and `:2765-2766` (normal).
- `StageResidentInput` takes `const sycl::event* producer`
  (`Qwen4ExpExpertExecutor.cpp:760-761`, decl
  `Qwen4ExpExpertExecutor.hpp:239-240`) and forwards it to
  `submit_host_copy` at `:773-774`. With a producer, the helper takes the
  `order_fence` path (`MoeCopyQueue.hpp:46-48`), so **no
  `ext_oneapi_submit_barrier()` is submitted** for the canonical x D2H.

Remaining barrier fallbacks (not canonical):

- `DECODE_FUSED_HC=1` returns `std::nullopt` at
  `Qwen4ExpBnb4Backend.cpp:2197`; the small-rows branch returns nullopt at
  `:2235`; the generic branch returns nullopt at `:2272`. In those modes the
  x D2H still falls back to one barrier.
- `StartArenaCold`/`StartArenaColdRows`/`StartResidentCold`/`ExecuteVerifyRows`
  fallback copies (`Qwen4ExpExpertExecutor.cpp:915`, `:964-966`, `:1153`,
  `:1410`) call `submit_host_copy` without a producer. In canonical they
  do not fire because `resident_input_event_` is set at `:773` first.
- The weights copy at `Qwen4ExpBnb4Backend.cpp:1604-1605` uses
  `copy_order`; when `device_event_live` is false (`:1588`), `copy_order`
  is null and the helper submits one barrier (all-CPU-miss edge case).

Evidence gap: neither `chain-probe4` nor `chain-probe5` tests
`ext_oneapi_submit_barrier()` itself. The closest evidence is
`xqueue_dep` = 8.5 us and `deferred_xqueue_wait` = 12.8 us. The helper comment
(`MoeCopyQueue.hpp:32-41`) asserts the barrier is device-side and does not
close the batch, but that is not measured. Therefore the canonical barrier is
already avoided by construction; for the remaining nullopt branches, whether
replacing the barrier is a win is unproven until a dedicated probe arm
(`submit; ext_oneapi_submit_barrier; submit`) runs.

### 3.3 Minimal signature list (what was needed / what remains)

Already implemented in the current worktree:

| # | signature / site | change |
|---|---|---|
| 1 | `ops::hc_decode_up_mix` `HyperConnectionOps.cpp:742`, decl `hpp:86` | returns `sycl::event` (canonical producer) |
| 2 | `RunHyperConnectionCombined` `Qwen4ExpBnb4Backend.cpp:2151`, decl `hpp:479` | returns `std::optional<sycl::event>`; canonical branch returns `hc_decode_up_mix` at `:2247-2251` |
| 3 | `RunHyperConnection` `Qwen4ExpBnb4Backend.cpp:2275`, decl `hpp:476` | returns `std::optional<sycl::event>` |
| 4 | `StageResidentInput` `Qwen4ExpExpertExecutor.cpp:760`, decl `hpp:239-240` | takes `const sycl::event* producer`; forwards to `submit_host_copy` `:773-774` |
| 5 | `ForwardImpl` `Qwen4ExpBnb4Backend.cpp:2717-2741`, `:2765-2766` | captures `mlp_hc_ev`; passes it to both StageResidentInput calls |

Still needed only if the non-canonical HC branches must also avoid the
barrier:

| # | signature / site | change |
|---|---|---|
| 6 | `ops::hc_mix_reduce` `HyperConnectionOps.cpp:207`, decl `hpp:32` | `void` -> `sycl::event`; return it from the generic branch `Qwen4ExpBnb4Backend.cpp:2262-2272` |
| 7 | `q4e_fusion::run_hc_up_mix_with_hidden_update` `Qwen4ExpDecodeFusion.cpp:200`, decl `hpp:81` | `void` -> `sycl::event`; return it from the `DECODE_FUSED_HC` branch `Qwen4ExpBnb4Backend.cpp:2193-2197` |
| 8 | `ops::hc_decode_up_mix_rows` `HyperConnectionOps.cpp:758` | `void` -> `sycl::event` if the 2..4-row branch (`Qwen4ExpBnb4Backend.cpp:2233-2235`) must report a producer |

### 3.4 F1 / F3 / F4 and the two prev-copy waits with `copy_q=1`

| wait | current line | event queue | semantics with copy_q=1 | verdict |
|---|---|---|---|---|
| F1, `resident_input_event_` | `Qwen4ExpExpertExecutor.cpp:766` | Q2 | protects `x_host_` from the next D2H; redundant when the previous layer ran a CPU job (worker waited `:459`, join `:981`), necessary on all-hit layers | keep as Q2 wait or make conditional; no longer a C batch breaker; 12.8 us class |
| F4, `prev_cpu_h2d_ev_` | `Qwen4ExpExpertExecutor.cpp:955` | Q2 | protects `cpu_out_host_` and the previous H2D's source; the next CPU worker can overwrite it | **semantically necessary as written**; move it to `job.transfer_reader` (`StartArenaCold` `:964-966`, worker wait `:465`), then delete `:955` |
| F3, router | `Qwen4ExpBnb4Backend.cpp:2857` | C | top-k host publication required by classification | **necessary**; this is the remaining 20 us-class per-layer breaker |

The two prev-copy waits both become Q2 waits under the gate. They are still
real host time (12.8 us class), but no longer drain the compute batch. Worst
case 2 x 12.8 us/layer ~= 1.2 ms/token [estimate]. F4 is the safe one to
retire first (worker `transfer_reader` already exists); F1 should be made
conditional, not deleted unconditionally.

---

## 4. Minimal construction order

1. **The canonical x-D2H barrier is already gone** in the current worktree:
   `mlp_hc_ev` (`Qwen4ExpBnb4Backend.cpp:2717-2727`) -> `StageResidentInput`
   (`Qwen4ExpExpertExecutor.cpp:760-774`) -> `submit_host_copy` with
   `order_fence` (`MoeCopyQueue.hpp:46-48`). Keep this path.
2. **Measure the barrier arm** (`submit; ext_oneapi_submit_barrier; submit`)
   before extending the producer event to the nullopt branches (fused HC,
   generic HC, multi-row). If the barrier is batch-safe, the extension is a
   command-count win only; if it flushes, it is a major win.
3. **F4 -> worker `transfer_reader`**, then delete the main-thread wait at
   `Qwen4ExpExpertExecutor.cpp:955`; make F1 conditional on "previous D2H
   consumed by worker".
4. Leave F3 (router) and the dead `execute_hetero_decode` status query alone.

Verification for future GPU runs: canonical arm + `AILA_Q4E_HC_SUBMIT_TIMING=1`;
success = per-call min/mean toward 5-8 us and the gap buckets losing the flat
~20 us floor. E2E gate = 19-token continue, agent6k + int8 KV, A/A pairs,
1.05% floor (16.372/16.202 tok/s), n=128 baseline 17.15 tok/s. Correctness:
greedy 32-token bit-identical; cancellation must drain Q2 before
`x_host_`/`resident_down_` reuse.

Receipts: `tmp/q4e-structural-20261016/chain-probe4.log`,
`chain-probe5.log`, `tmp/q4e-bottleneck-audit-20261016/REPORT.md`
(~199 CPU misses/token).
