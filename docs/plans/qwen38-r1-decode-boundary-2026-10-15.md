# D1: decode layer-boundary attribution (inter_dev / stagein_dev) and two
#     opt-in variants to shrink the boundary exposure

Date: 2026-10-15.  Owner: D1 sub-agent.  Base revision: 58f7874 (retained26).
Scope of code changes: `src/models/Qwen4ExpBnb4Backend.cpp` only (+82/-3).
No build, no GPU, no commit in this delivery (per role discipline).

--------------------------------------------------------------------------------
## 0. TL;DR

1. `inter_dev` (median 8.86 ms/token, C2 round 2) is NOT a generic per-layer
   queue stall.  Over 256 count=1 tokens from `c2-round2-b{1,2}-worker.log`,
   `corr(inter_dev, ple_host) = 0.9982` and `inter_dev - ple_host` = 0.31 ms
   [0.15, 0.48] p10/p90.  Both numbers measure the SAME interval: the stage()
   span that closes at the L==1 PLE marker, which contains exactly one
   `ple_.consume()` per token.
2. Root cause of that interval: `consume()` blocks on the PLE background
   worker (`Qwen4ExpPle.cpp:747 wait_task`) and its staging H2D
   (`:752`).  For decode (1 token) the worker's gather is NOT prefetched
   (`prefetch_gather_pages` returns early for `num_tokens < 64`,
   `Qwen4ExpPle.cpp:527`), so the worker pays ~16-32 serial mmap faults:
   slow mode 9.05 ms vs fast mode 2.50 ms of `ple_host` (144 vs 90 tokens).
   The exposed part is worker time minus the layer-0 submission lead.
3. Observer effect (new): the `AILA_Q4E_DECODE_STAGE_EVENTS=1` protocol
   itself costs +11-13 ms/token (74.5/72.9 instrumented vs 60.2/61.7 in
   clean arms of the same shape).  All dev-bucket ms numbers in the D1 brief
   live in that +18% regime; rank candidates by mechanism, not by those
   absolute ms.

Two opt-in variants are implemented (default off, bit-identical semantics):

| env | what it changes | expected [estimate] |
|---|---|---|
| `AILA_Q4E_CPU_ROUTE_EARLY=1` | classify the layer's cold experts and start the CPU batch right after the router wait (instead of inside `RunResidentDecode`) | 1-4 ms/token |
| `AILA_Q4E_STAGEIN_HOIST=1` | submit the per-layer resident-input D2H right after the MLP-HC kernels, ahead of the lease/branch bookkeeping | 0.5-2 ms/token |

--------------------------------------------------------------------------------
## 1. Evidence and attribution

### 1.1 What each dev bucket actually covers

`dev_mark(bucket)` = `ctx.queue().ext_oneapi_submit_barrier()`
(`Qwen4ExpBnb4Backend.cpp:2089-2092`), one marker per phase boundary; the
readback is `bucket(i) += start(i) - end(i-1)` (`:2597-2604`).  Marker order
inside one layer (all line numbers pre-D1):

```
kDevInter(:2102) kDevPle(:2122) kDevHc(:2141) kDevLinear|Full(:2157)
kDevHc(:2173) kDevStageIn(:2200) kDevPreRouter(:2209) kDevRouter(:2257)
kDevShared(:2385) kDevExperts(:2429) kDevResid(:2462) -> next kDevInter
```

Consequences:

* `kDevInter` = everything submitted between the previous layer's `kDevResid`
  and this layer's `kDevInter` marker.  In this loop that is *no GPU work at
  all*: the host reaches the next `kDevInter` within a handful of statements.
  A non-zero `inter_dev` therefore reads the host-side interval that STARTS
  at `kDevResid` and ENDS when the next marker touches the queue -- and the
  only big host work in that window is the L==1 PLE block.
* `stage(kStagePle)` (`:2034-2038`, called at `:2119`) closes the span that
  began at the previous `stage(kStageShared)` call (`:2461` of layer 0), i.e.
  the SAME window: tail of layer 0 + loop back-edge + `kDevResid` +
  `kDevInter` + PLE block + `kDevPle`.  So `ple_host_ms` and `inter_dev_ms`
  are two instruments on one interval, and they agree to 0.31 ms.

### 1.2 Token-paired numbers (256 count=1 tokens, b1+b2)

| field | median | p10 | p90 |
|---|---:|---:|---:|
| `ple_host_ms` | 8.522 | 2.396 | 9.502 |
| `inter_dev_ms` | 8.858 | 2.722 | 9.947 |
| `inter_dev - ple_host` | 0.313 | 0.154 | 0.481 |

Two modes track each other exactly:

| mode | n | `ple_host` med | `inter_dev` med |
|---|---:|---:|---:|
| slow (`ple_host>7`) | 144 | 9.053 | 9.381 |
| fast (`ple_host<5`) | 90 | 2.497 | 2.766 |

### 1.3 Mechanism inside the PLE block

* `ForwardImpl` prepares the PLE ticket once per forward
  (`Qwen4ExpBnb4Backend.cpp:1841-1843`); for decode `reuse_ple` is false
  (input ids change every token), so `prepare()` runs every token.
* `prepare()` hands the gather to the PLE worker thread
  (`Qwen4ExpPle.cpp:696-711`) and returns; the worker runs
  `compute_ngram_ids -> prefetch_gather_pages -> dequant_to_staging -> H2D`
  (`:573-584`).
* `prefetch_gather_pages` is a no-op below 64 tokens (`:527`).  Decode
  touches `NUM_HEADS` packed rows + absmax rows per token; at 210 us per cold
  mmap fault (A2 `ple_gather_io_probe.py`) the cold floor is ~7 ms/token,
  which matches the slow mode.
* The L==1 block then blocks the coordinator: `consume()` waits for the
  worker (`wait_task`, `:747`) and the staging H2D (`:752`).  Exposure =
  worker time minus the layer-0 submission lead (~1.5-2 ms).

So the D1 target is a PLE-worker latency, not a queue-depth or ordering
problem: the host is already as far ahead as the dependency chain allows
(`prepare` is issued at the forward top).

### 1.4 Observer effect of the instrumentation itself

| arm | dev events | `submitted_ms` med | request rate |
|---|---|---:|---:|
| c2-round2-b1 | on | 74.492 | ~13.5 tok/s |
| c2-round2-b2 | on | 72.903 | ~13.7 tok/s |
| c2-earlystart-b1 | off | 60.209 | 15.57-15.65 tok/s |
| d0-deferwait-b1 | off | 61.668 | ~16.0 tok/s |

528 marker submissions/token at ~24-28 us each plus the single end-of-forward
drain account for most of the +11-13 ms; the host-stage fields inflate the
same way (`hc_host_ms` 9.24 -> 6.17 because two `kDevHc` markers sit inside
each HC stage).  Reading recipe: only compare arms with the SAME gate state,
and prefer the clean (`--diag` without `DECODE_STAGE_EVENTS`) rate for
decisions.

### 1.5 `stagein_dev` = the per-layer resident-input D2H window

`stagein_dev` 2.106 [1.854, 2.782] ms/token = 44 us/layer, for
`StageResidentInput` (`Qwen4ExpExpertExecutor.cpp:723-730`): one 5 KB D2H per
layer plus the wait for the previous copy.  The wait is already skippable via
`AILA_Q4E_CPU_DEFER_ARENA_WAIT=1`; the clean-arm A/B (d0 vs earlystart) shows
61.7 vs 60.2 ms/token, i.e. inside noise -> the defer knob is not a rate
lever by itself.  The in-scope lever is to submit the copy earlier (variant
V2), which shortens both the D2H execution path and the CPU job's activation
wait.

### 1.6 Host sub-step ledger (clean arms, median ms/token)

| sub-step | d0/earlystart | c2-round2 (instrumented) |
|---|---:|---:|
| `ple_host` | 6.43-6.45 | 8.52 |
| `hc_host` | 6.17-6.32 | 9.24 |
| `linear_host` + `full_host` | 3.46+2.93=6.39 | 4.80+3.44=8.24 |
| `router_wait` | 11.95-12.00 | 5.94-6.84 |
| `shared_submit` + `moe_front` + `topk_submit` | ~6.4 | ~7.6 |
| `experts_host` (incl. `cpu_join` 9.5 / `cpu_compute` 14.4) | 20.9-22.8 | 23.9 |
| `tail` | ~1.6-4.6 | ~4.5 |

Note `router_wait` roughly doubles in the clean arms: with the instrument
off the host reaches the wait earlier and absorbs the device backlog there
instead of paying it as marker gaps.

--------------------------------------------------------------------------------
## 2. Variants (default off; semantics must stay bit-identical)

### 2.1 V1 `AILA_Q4E_CPU_ROUTE_EARLY=1`

Code: stash block at `Qwen4ExpBnb4Backend.cpp:211-222`, gate read at
`:2054-2055`, classification + `StartArenaCold` right after the route wait
at `:2453-2483`, consumer in `RunResidentDecode` at `:1305-1325`.

* Moves [miss classification + `StartArenaCold`] from inside
  `RunResidentDecode` to immediately after `wait_actual_route()`, i.e. ahead
  of the `stage(kStageRouter)` bookkeeping, the prediction-stats loop and the
  `RunResidentDecode` entry.
* Dependency argument: the classification needs only `router_host_ids_` and
  `adaptive_cache_.GetSlot`; `ObserveRouting` (`Qwen4ExpAdaptiveCache.cpp:418-438`)
  touches only `usage_`/`access_count_`/stats and never `slot_table_`, so the
  slot map seen by the early classification is the same one the kernels use.
* Safety: the stash is written once per layer and consumed once; the consumer
  re-derives the miss arrays and throws unless the stashed CPU prefix matches
  element-wise (`:1311-1319`).  Verification / HC-reference paths and
  non-adaptive pipelines are gated out.
* Ceiling: the host work skipped is `router_other`-class time (6.0-7.5
  ms/token in the instrumented arms) plus the classification itself; the CPU
  job additionally gets the shared-expert submission block as lead time.
  Expect 1-4 ms/token [estimate], mostly visible as `cpu_join_ms` and
  `experts_host_ms` drops.

### 2.2 V2 `AILA_Q4E_STAGEIN_HOIST=1`

Code: `:2212-2220` (early `StageResidentInput`) and the original call site
gated at `:2243`; gate read at `:2055`.

* Submits the adaptive count==1 D2H right after the MLP-HC kernels
  (`:2209-2210`), before `BeginResidentLayer`/lease/branch bookkeeping.
* Dependency argument: the D2H only needs the in-order queue position after
  the kernels that finalize `buf_.x`; nothing submitted in between writes
  `buf_.x`.  Bytes and the executor state machine are unchanged.
* Ceiling ~0.5-2 ms/token [estimate]; compounds with V1 because the CPU
  job's `activation` event is published earlier.

Both variants can be used together (they do not touch the same decision).

--------------------------------------------------------------------------------
## 3. Arms, read order and gates

GPU mutex precondition (both must be empty):

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
```

A/B/B/A on the retained runner (clean regime, no stage events):

```powershell
python tmp/q4e-sprint-r1-20261014/run.py d1-a1 --prime --diag
python tmp/q4e-sprint-r1-20261014/run.py d1-b1 --prime --diag --AILA_Q4E_CPU_ROUTE_EARLY=1 --AILA_Q4E_STAGEIN_HOIST=1
python tmp/q4e-sprint-r1-20261014/run.py d1-b2 --prime --diag --AILA_Q4E_CPU_ROUTE_EARLY=1 --AILA_Q4E_STAGEIN_HOIST=1
python tmp/q4e-sprint-r1-20261014/run.py d1-a2 --prime --diag
```

Single-knob arms (only if the combined arm passes):

```powershell
python tmp/q4e-sprint-r1-20261014/run.py d1-v1 --prime --diag --AILA_Q4E_CPU_ROUTE_EARLY=1
python tmp/q4e-sprint-r1-20261014/run.py d1-v2 --prime --diag --AILA_Q4E_STAGEIN_HOIST=1
```

Boundary readback (optional, instrumented regime; all arms must carry the
flag to stay comparable):

```powershell
python tmp/q4e-sprint-r1-20261014/run.py d1-i-a1 --prime --diag --AILA_Q4E_DECODE_STAGE_EVENTS=1
python tmp/q4e-sprint-r1-20261014/run.py d1-i-b1 --prime --diag --AILA_Q4E_DECODE_STAGE_EVENTS=1 --AILA_Q4E_CPU_ROUTE_EARLY=1
python tmp/q4e-sprint-r1-20261014/run.py d1-i-a2 --prime --diag --AILA_Q4E_DECODE_STAGE_EVENTS=1
```

Fields to read, in order:

1. Parse correctness: zero `[Q4E AdaptiveWork]` exception lines, `misc_dev_ms
   == 0.000`, `dev_events` as configured.
2. Bit-exactness: the SSE text of `d1-*` must equal the `d1-a1` reply for the
   same request (`responses/*.sse`); the runner reuses sessions, so compare
   the `context-14000` and `context-14000-continue` bodies.
3. Rate: `tokens_s` of the `continue` rows (the decode-dominated rows).
4. Host fields: `cpu_join_ms`, `experts_host_ms`, `router_other_ms`,
   `stagein_dev_ms` (instrumented arms only).

Gates (charter section 5 defaults):

* GO if median decode rate improves >= 5% on both prompt shapes with 2/2
  pairs, A/A noise <= 1%, and the SSE bodies are byte-identical.
* In the instrumented pair, require `inter_dev + stagein_dev` median to drop
  >= 50% for V2-only arms; V1 is judged by rate and `cpu_join_ms` (its
  effect lands before the router marker).
* Any `UR_OUT_OF_RESOURCES`, `MEM_LARGE_PAGES`, or
  "early CPU route" exception = automatic NO-GO for that arm.

--------------------------------------------------------------------------------
## 4. Risks and open items

* Both variants are unbuilt and unmeasured (role discipline).  First action
  on the GPU slot is the `d1-a1` smoke, then the B1 arm.
* V1 depends on `ObserveRouting` staying counter-only; if a future change
  makes it mutate `slot_table_`, the fail-closed compare throws instead of
  silently changing rows -- re-check then.
* The stash uses file-scope `thread_local` state because the D1 file scope
  excludes `Qwen4ExpBnb4Backend.hpp`; it is reset at every forward entry
  (`:2128-2131`) and is per scheduler thread by construction.
* V2 widens the window in which the D2H is queued before
  `BeginResidentLayer`; if a future `BeginResidentLayer` starts writing
  `buf_.x`, the hoist must be revisited.

### Out of D1 file scope, but the largest measured lever

`inter_dev` is PLE-worker-bound.  The direct fix is to prefetch the decode
gather rows too: the guard `if (!s_enabled || num_tokens < 64 ...) return;`
at `Qwen4ExpPle.cpp:527` excludes the 1-token decode row, whose ~32 page
contacts then serialize at ~210 us each.  Suggested probe for the PLE owner
(A2/R1): env `AILA_Q4E_PLE_PREFETCH_MIN` (default 64) and an arm with
`=1`; predicted 4-7 ms/token on the slow-mode tokens [estimate].  D1 does
not touch that file.

--------------------------------------------------------------------------------
## 5. Sources

* `tmp/q4e-sprint-r1-20261014/c2-round2-b{1,2}-worker.log`,
  `c2_round2_parse.py` (256 count=1 tokens; medians, correlation computed in
  this task).
* `tmp/q4e-sprint-r1-20261014/{c2-earlystart-b1,d0-deferwait-b1}-worker.log`
  and `-run.log` (clean-regime rates; d0 = defer-wait arm).
* `docs/plans/qwen38-r1-cost-model-2026-10-14.md` 6.19-6.21.
* `docs/plans/qwen38-r1-ple-firstchunk-2026-10-15.md` (mmap fault cost
  27.6 s cold vs 0.048 s warm per 131k touches; F1 prefetch).
* Code: `Qwen4ExpBnb4Backend.cpp:2089-2092, 2034-2038, 2102-2122, 2429,
  2462, 2597-2604`; `Qwen4ExpPle.cpp:527, 558-585, 587-610, 696-711, 723-753`;
  `Qwen4ExpExpertExecutor.cpp:723-730, 891-921`;
  `Qwen4ExpAdaptiveCache.cpp:418-438`.
