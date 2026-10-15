# Qwen38 R1 E7: speculative whole-layer staging (V3) design + prototype

Date: 2026-10-15. Role: E7 (prefill 400+ track). Status: module and bench
delivered, never compiled into a binary, never wired into production, zero
GPU runs in this task. All numbers carry their source; estimates are marked.

Scope of this document: (1) the speculation window and the dependency proof,
(2) the V3 design: `src/models/Qwen4ExpLayerStage.{hpp,cpp}`, the integration
diff draft (for coordinator review, not applied) and the arena VRAM options,
(3) the overlap bench `perf/qwen4exp/q4e_layer_stage_bench.cpp` with arms and
gates, (4) risks and TODOs.

## 1. Speculation window and dependency argument

### 1.1 Current per-layer order (post-E6 worktree)

`Qwen4ExpBnb4Backend.cpp` layer loop:

- `:2207` `RunAttentionWindow(...)` submits this layer's attention kernels.
- `:2225` `RunHyperConnectionCombined(layer.mlp_hc, ...)` folds the attention
  output into `buf_.x`.
- `:~2290` `dense_rows(layer.router_gate, layer.router_w, buf_.x, ...)`
  computes router scores from the post-attention hidden state.
- `:2337` `wait_actual_route` waits the top-k event; the host is blocked while
  the GPU finishes attention + HC + router (this is the observed
  `router_wait`).
- `:2527` `RunAdaptivePrefill(...)` builds `prefill_unique_` and stages cold
  experts through the ping-pong banks.

E6 proved (`docs/plans/qwen38-r1-prefill-pipeline-2026-10-15.md` section 1)
that routing depends on post-attention hidden, so the *set* of experts for
layer L is unknowable before L's own attention completes. Cross-layer
speculation of the routing-dependent bank traffic is therefore impossible.

### 1.2 Why whole-layer weight staging is still legal

The speculative set does not need routing: it is the layer's entire
nonresident weight set, and expert *weights* are static for the model
lifetime. Three facts make this exact:

- Resident membership is a pure cache decision (`GetSlot(layer, e)`), so
  "all experts of layer L minus the resident ones" is computable at any time
  before the router returns (`Qwen4ExpLayerStage::ColdExperts`).
- The payloads are bit-identical to the bank path: both copy the same host
  arena slots (large-page RAM owners) with no transformation.
- The arena map is keyed by expert id, so a swap that lands between `Begin`
  and the route either makes an expert resident (staged copy unused) or cold
  (already staged).

In-scope v1 window: attention(L) submitted -> `wait_actual_route` returns.
Because expert weights do not depend on L's compute, staging could legally be
launched even earlier (L-1's MoE window). v1 keeps the same-layer window so
the single arena is never overwritten while L-1's kernels still read it
(arena reuse needs a consumer fence, which the module accepts via
`RecordArenaConsumer`/the `arena_reader` argument and the module gates the
next epoch's first H2D on). Earlier launch is a v2 extension (section 4).

### 1.3 Supply vs window arithmetic (production arm, 4096-token block)

Measured per-layer supply (`docs/plans/qwen38-r1-prefill-400-2026-10-15.md`
section 1, `docs/plans/qwen38-r1-cost-model-2026-10-14.md` section 6.29):

- host fill 2.4 s / 48 = 50 ms; copy-engine H2D 5.0-5.3 s / 48 = 104-110 ms;
  supply = 154 ms per layer for the routed cold set (~433 of 454).
- device attention window: GDN 4.1 s + full attention 2.7 s = 6.8 s block =
  141 ms per layer; plus HC/router ~0.6 s = 12 ms. Window = ~153 ms.
- block expert kernels: 50.26 GB at 116.5 GB/s = 0.43-0.55 s per block.

Supply (154 ms) and window (153 ms) are nearly balanced, so if the copy
engine and the GPU do not interfere, the block wall should collapse from
19.5 s to roughly

    max(supply 7.4 s, device 6.8 s) + kernel 0.55 s + overhead = 8.5-10 s

which is 410-480 tok/s for a 4096-token block. This is the one premise the
bench exists to falsify: concurrent DMA writes plus host fill must not inflate
the VRAM-bound device load beyond 30% of the staging duration.

## 2. V3 design

### 2.1 Module `src/models/Qwen4ExpLayerStage.{hpp,cpp}`

Public surface (all in `aila`, no production call sites):

- `Allocate(ctx, geom, {num_experts_per_layer, max_slots, chunk_slots,
  fill_banks})`: one contiguous device arena (`max_slots * slot_bytes`) plus
  `fill_banks * chunk_slots` pinned host banks. Geometry is the host arena's
  `ExpertSlotGeometry`, so descriptors match cache slots exactly.
- `Begin(layer, expert_ids, sources, copy_q, arena_reader)`: builds the epoch
  map on the calling thread, hands the job to the background fill thread.
  `copy_q` must be in order and must be the module's only writer during the
  epoch. `arena_reader` (or the stored `RecordArenaConsumer` event) gates the
  first H2D, which is what makes arena *reuse* safe.
- Worker: per chunk (default 16 slots), waits the bank's previous event,
  memcpys the slot bytes into a pinned host bank, submits one in-order H2D
  into the arena. Failures are fail-closed: the epoch map is cleared, so
  `MissedExperts` reports everything as missed and the caller falls back to
  the bank path; `WaitReady` rethrows.
- `IsReady` / `WaitReady` / `ReadyEvent` / `ChunkEvent` / `ChunkCount`:
  host-wait or kernel-gate on the copy events.
- `MissedExperts(epoch, route_ids, count)`: arena hits (occurrences) and
  unique missed ids (bank fallback set).
- `GetSlotDev(epoch, expert, nested)` / `SlotOf`: same descriptor math as
  `Qwen4ExpAdaptiveCache::GetStagingSlotDev` (compact nested layout).
- `RecordArenaConsumer`, `WaitIdle`, `Reset`, `Release`, `Stats`.

RS-1 discipline: the module issues no allocation after `Allocate`; the
per-epoch path only clears/reuses member vectors. Consumer helpers run on the
`Begin` thread only (documented in the header).

### 2.2 Integration diff draft (NOT applied; for coordinator review)

All hunks below are opt-in behind `AILA_Q4E_LAYER_STAGE=1`, default off.

`src/models/Qwen4ExpBnb4Backend.hpp` (near the adaptive cache members):

    #include "models/Qwen4ExpLayerStage.hpp"
    aila::Qwen4ExpLayerStage layer_stage_;
    std::vector<int> layer_stage_ids_;
    uint64_t layer_stage_epoch_ = 0;

Load path (`Qwen4ExpBnb4Backend.cpp`, where `adaptive_cache_.Allocate` runs):

    // after adaptive cache allocation and host arena load; env-gated
    if (aila::env::read_flag("AILA_Q4E_LAYER_STAGE", false)) {
        aila::LayerStageConfig sc;
        sc.num_experts_per_layer = cfg_.num_experts;
        sc.max_slots = cfg_.num_experts;         // 512 covers the worst layer
        sc.chunk_slots = 16;
        sc.fill_banks = 4;
        std::string err;
        if (!layer_stage_.Allocate(ctx, host_arena_.Geometry(), sc, &err))
            AILA_LOG_WARN("[Q4E LayerStage] disabled: %s", err.c_str());
    }

Layer loop, right after `:2207 RunAttentionWindow(...)`:

    uint64_t stage_epoch = 0;
    if (layer_stage_.IsAllocated() && is_prefill && count > 4) {
        layer_stage_ids_ = Qwen4ExpLayerStage::ColdExperts(adaptive_cache_, L);
        // The previous epoch's arena consumer (recorded below) gates this
        // H2D; an explicit reader may be passed instead.
        stage_epoch = layer_stage_.BeginFromArena(
            L, layer_stage_ids_, host_arena_, *copy_q_);
    }

Consume, immediately before `:2527`'s `RunAdaptivePrefill` call (the router
has returned by then):

    if (stage_epoch) {
        layer_stage_.WaitReady(stage_epoch);   // v1: wait the staging tail
    }

Inside `RunAdaptivePrefill`, replace the cold-list loop head with a split
(keep the resident-first order):

    // staged cold experts first (arena descriptors), then any bank misses
    for (int e : layer_stage_ids_) {
        if (!prefill_expert_uses_[e]) continue;
        const int slot = layer_stage_.SlotOf(stage_epoch, e);
        if (slot >= 0) {
            prefill_unique_.push_back(e);
            adaptive_prefill_slots_[e] =
                layer_stage_.GetSlotDev(stage_epoch, e, &metadata);
        }
    }
    const int stage_hits = int(prefill_unique_.size()) - hits;
    // existing bank pump runs on prefill_unique_[hits + stage_hits ..)

and after the MoE kernels for the layer:

    layer_stage_.RecordArenaConsumer(stage_epoch, last);

Notes for the reviewer:

- `WaitReady` before consuming is the conservative v1: kernels gate on
  `ReadyEvent`/`ChunkEvent` in v2 if the measured tail matters.
- The bank pump must stay for the no-stage path and for stage misses; with a
  full-layer arena the bank chunk count is normally zero.
- MTP verify (`count==2` decode path) never enters this code; checkpoints and
  cancellation are unaffected, but the cancel path must drain `copy_q_`
  before `layer_stage_.Release()` (the module waits its own last event; the
  worker thread must be joined through `Release`, never destroyed twice).

### 2.3 VRAM options (arena = 454 x 2,534,400 B = 1,150.6 MB = 1.071 GiB)

Measured admission accounting (`tmp/q4e-sprint-r1-20261014/
e6-banks4-b1-worker.log` "Memory Admission Accounting"): Total VRAM 15933 MB,
static weights 4522 MB, session state 1000 MB (4K context), reserve headroom
3584 MB, expert cache 6826 MB -> 2824 slots of budget, minus 32 staging
slots = 2792 resident slots. Per layer: 58-59 resident, so the worst layer
has 454 nonresident experts (48 x 58 + 8 = 2792).

| Option | Mechanism | Decode impact | Complexity | Status |
|---|---|---|---|---|
| A: permanent cut | `AILA_Q4E_MAX_GPU_SLOTS=2337` (`-455` slots = -1097 MiB, exactly the arena) | permanent; hit-rate drop must be measured (decode log mean hits 80-287/token at 2792 slots) | S (one env) | measurement arm |
| B: phase/borrow | during prefill, evict ~455 resident victims (table updates), arena = borrowed slot memory, restore + re-upload (~1.15 GB H2D, hideable in a later window) after | none when no decode interleaves; interleaved decode tokens hit the cold path for victims until restore | M-L (scatter arena, two table syncs, restore job) | roadmap |
| C: reserve re-budget | lower `AILA_Q4E_VRAM_RESERVE_MB` 3584 -> 2450 (Delta 1134 MiB = arena 1097 + slack); keep `AILA_Q4E_MAX_GPU_SLOTS` at 2800 so the pool tops out at 2800 slots (+8) and the arena allocates from the freed headroom | none | S (one env + one allocation site) | first probe |

Arithmetic check for C: today used = 4522 static + 1000 session + 6826 expert
budget + 3584 reserve = 15932 of 15933 MB; with reserve 2450 the pool grows
to the 2800-slot cap, and 4522 + 1000 + 6767 (2800 slots) + 1097 (arena) +
2450 = 15836 MB, leaving ~97 MB of accounting slack.

Recommendation: validate with C (no slot churn, no decode coupling), keep A
only as the decode-impact measurement arm, and put B on the roadmap if C
shows allocation instability (`UR_OUT_OF_RESOURCES`) or if the reserve cannot
be lowered operationally. The bench below needs none of this: it runs
standalone on an idle GPU.

## 3. Overlap bench and validation arms

`perf/qwen4exp/q4e_layer_stage_bench.cpp` (target
`AilaQwen4ExpLayerStageBench`). One round is `--gdn-calls` production GDN
prefill calls (real layer geometry: vh=48, qh=16, rows=4096; K4 anchor
74.8 ms per call) submitted on the compute queue, plus one staging pass of
`--experts` slots (default 454) on the copy queue with the background fill
thread.

Arms and gate:

- `load`: GDN batch alone.
- `stage`: `Begin` -> `WaitReady` alone.
- `overlap`: GDN batch submitted, staging kicked off while it runs; the
  round's load barrier is recorded as the arena consumer so the next epoch's
  H2D gate is exercised.
- Gate: `inflation = median(overlap load) - median(load)` must be <= 30% of
  `median(stage)`; the printed `[LS GATE]` JSON carries `ratio` and
  `verdict` (`overlap_ok` / `no_go`). `stage_slow_ratio` is informational.

Guards: GDN output finite check, arena payload bit-compare against the source
pool (first/last 4 KiB of the last slot, gate_up + down regions), slot-map and
descriptor offset checks, and a route-delta check (`staged_hits`/`missed`).
`[LS GUARD]` failing means stop; never interpret numbers past it.

Coordinator commands (GPU mutex first, no fixed sleeps; A/A pair):

    Get-Process AilaWorker,AilaServer,Aila -ErrorAction SilentlyContinue   # must be empty
    pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
    build-refactor/AilaQwen4ExpLayerStageBench.exe --experts 454 --gdn-calls 2 --samples 5 *> tmp/q4e-sprint-r1-20261014/layer-stage/ls-A1.log
    build-refactor/AilaQwen4ExpLayerStageBench.exe --experts 454 --gdn-calls 2 --samples 5 *> tmp/q4e-sprint-r1-20261014/layer-stage/ls-A2.log
    build-refactor/AilaQwen4ExpLayerStageBench.exe --experts 227 --fill-banks 4 --gdn-calls 2 --samples 5 *> tmp/q4e-sprint-r1-20261014/layer-stage/ls-half.log

Read order: `[LS DEVICE]` -> `[LS GUARD]` -> `[LS TIME]` -> `[LS MEDIAN]` ->
`[LS GATE]`. Expected magnitudes `[estimate]`: stage 130-170 ms per 454-slot
pass (1.15 GB at 9.2-9.6 GB/s H2D + 55 ms fill, bank waits included), load
~150 ms per 2-call batch; a passing run should show inflation in the tens of
milliseconds, not hundreds.

E2E arms (after the coordinator wires the draft, v1 scope):

1. `AILA_Q4E_LAYER_STAGE=0` vs `=1`, A/B/B/A on the same 14K request,
   gate: block wall <= 15 s (from 19.5 s), staging bytes reported by the
   module stats in the worker log.
2. Bit-exact greedy continuation between both arms; any diff = fail closed.
3. A/A <= 1% on both arms; `UR_OUT_OF_RESOURCES`/`MEM_LARGE_PAGES` absent.
4. Decode non-regression in the same session (>= 15.4 tok/s quiet).

## 4. Risks and TODOs

- R1 (premise): copy-engine H2D + host fill may steal internal bandwidth from
  the GDN/full-attention kernels. This is exactly the bench gate; a `no_go`
  verdict closes V3 without further work (write the negative result first,
  per charter discipline).
- R2 (tail): v1 waits the whole staging tail before consuming; if the window
  is shorter than the supply the tail leaks into the critical path. v2:
  `ChunkEvent`-gated kernels (module already exposes per-chunk events; a
  per-chunk ready handshake is the only missing piece).
- R3 (arena reuse): overwriting the arena while the previous layer's MoE
  kernels read it is only safe through the recorded consumer gate. The
  integration draft must set `RecordArenaConsumer` on every consume path,
  including the fallback bank path (the returned event is the layer's last
  MoE event).
- R4 (VRAM): option C trades reserve headroom for the arena; watch
  `UR_OUT_OF_RESOURCES`, memcpy failures and the accounting line drift.
  Option B is the fallback and needs a scatter-slot variant of the module.
- R5 (concurrency): `copy_q_` also serves swaps and bank staging; the module
  requires being its only writer during an epoch. Guard the draft with an
  assert or a counter in review.
- TODO-1: two-arena ping-pong so staging of L+1 can start in L's MoE window
  (the earlier-launch extension). Do not attempt before R2 is resolved.
- TODO-2: per-chunk kernel gating (R2) once the e2e arm confirms the tail is
  real; measure with `AILA_Q4E_DECODE_STAGE_EVENTS`-style markers only if
  needed (they cost +11-13 ms/token on decode; prefill markers are untested).
- TODO-3: bench `--gdn-calls 1`/`4` sweep to bracket the interference curve;
  the current default (2) matches the production window ratio.
- Not verified in this task: the two new TUs were never compiled into a
  binary (only `icx-cl -fsyntax-only -fsycl` clean) and the bench never ran;
  the coordinator's build+GPU window owns both.

## 5. Sources

- `docs/plans/qwen38-r1-prefill-pipeline-2026-10-15.md` (E6: router reads
  post-attention hidden; bank pump with `staging_banks`).
- `docs/plans/qwen38-r1-prefill-400-2026-10-15.md` (E1: 19.5 s block split,
  bank_wait 8.1 s with H2D 5.3 + reader lag 2.8, fill 2.4 s, expert kernel
  0.43-0.55 s).
- `docs/plans/qwen38-r1-cost-model-2026-10-14.md` section 6.29.
- `docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md` (K4: 74.8172 ms anchor,
  516.7 GB/s effective).
- `tmp/q4e-sprint-r1-20261014/e6-banks4-b1-worker.log` (admission accounting,
  2792 resident slots).
- `src/models/Qwen4ExpBnb4Backend.cpp` `:2207/:2225/:2337/:2527` (window).
- `src/models/Qwen4ExpAdaptiveCache.cpp` `:606-665` (bank staging semantics
  the module mirrors), `:321-352` (descriptor math).

---

## 6. G5 consumer close-out (2026-10-16)

### 6.1 Root cause of the g3 zero-hit: a stale object, not the source

The g3 arm (`tmp/q4e-sprint-r1-20261014/g3-layerstage-b1*`, 23:32) ran a worker
whose `Qwen4ExpBnb4Backend.cpp` object predates the final wiring edit:

- g3 versus the banks=2 default (block 2): submitted 16511.451 vs 11177.727 ms
  (+5.33 s) and the entire delta is `router_wait_ms` 9108.757 vs 3815.620
  (+5.29 s); `experts_host_ms` 7133.1 vs 7139.8, `bank_wait_ms` 4276.1 vs
  4304.2, `fill_ms` 2316.6 vs 2318.6 and `staged=18130 / h2d_mb=43820.068` are
  bit-identical (block 1: 19360 / 46792.969). The arena H2D therefore ran
  (~55.3 GB per block ~= 5.3 s at the measured ~10.4 GB/s) while the consumer
  served zero experts and the banks re-staged the full routed cold set.
- Executable: `g3-layerstage-b1/launch.json` records worker sha256 `da4d47b6...`;
  `build-refactor/AilaWorker.exe` still carried that sha and mtime 23:15:27 when
  checked before the 00:01 rebuild, so the run used that binary.
- Build arithmetic from `build-refactor/.ninja_log` (that build): obj entries
  [166, 37769] ms and the `AilaWorker.exe` link end at [104852] ms; taking the
  exe mtime 23:15:27 as the link end gives T0 = 23:13:42.2 and an object compile
  window of 23:13:42.2 .. 23:14:19.9. The final edit to
  `src/models/Qwen4ExpBnb4Backend.cpp` is timestamped 23:14:34, i.e. ~14 s
  AFTER the object finished; the 23:15:27 link used that stale object.
- Symptom match: the pre-edit consumer merged arena slots into
  `adaptive_prefill_slots_` but never added them to `prefill_unique_`, so the
  bank pump staged every routed cold expert and overwrote those slots: arena
  copies run, banks bit-identical to default, no `act_start` mismatch. That is
  exactly the measured g3 behavior.

Conclusion: the zero-hit was a stale-binary artifact. The 00:01 rebuild
recompiled the TU (obj mtime 23:59:55 > source 23:14:34) and already contains
the completed consumer code at HEAD.

### 6.2 G5 changes (post-patch line numbers)

- `src/models/Qwen4ExpLayerStage.hpp` `:52-55` stats fields, `:127-130`
  `NoteConsume` declaration.
- `src/models/Qwen4ExpLayerStage.cpp` `:290-297` `NoteConsume` implementation.
- `src/models/Qwen4ExpBnb4Backend.cpp`:
  - `:1569-1572` arena-hit dedupe: an expert admitted by a swap after Begin is
    already counted as a resident hit, so it is never listed twice off the arena
    map (prevents a double `act_start` -> `Incomplete Qwen4Exp prefill routing
    map`).
  - `:1620-1623` `layer_stage_.NoteConsume(staged_count, bank_fallback)` per
    consumed epoch.
  - `:2197-2199` `s_layer_stage_log` plus the per-forward `LayerStageStats`
    baseline.
  - `:2940-2960` one `[Q4E LayerStage] forward pos=.. epochs=.. arena_experts=..
    bank_experts=.. arena_mb=..` line per prefill forward, plus a WARN when an
    epoch ran with zero arena consumption, so a silent zero-hit cannot recur.
- Unchanged by design: gate default 0 (all new code sits inside the gate/counter
  paths), bit-exact slots (verbatim host-arena copies), `RecordArenaConsumer`
  still called after the MoE submit, VRAM option C untouched. Both TUs pass
  `icx-cl -fsyntax-only -fsycl` (exit 0).

### 6.3 Arms and gates (coordinator)

```
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # must be empty
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
# MANDATORY build-race guard: the TU must be newer than its source
(Get-Item build-refactor\CMakeFiles\AilaLib.dir\src\models\Qwen4ExpBnb4Backend.cpp.obj).LastWriteTime -gt (Get-Item src\models\Qwen4ExpBnb4Backend.cpp).LastWriteTime
python tmp/q4e-sprint-r1-20261014/run.py g5-a1 --prime --diag --generate=8
python tmp/q4e-sprint-r1-20261014/run.py g5-b1 --prime --diag --generate=8 AILA_Q4E_LAYER_STAGE=1
python tmp/q4e-sprint-r1-20261014/run.py g5-b2 --prime --diag --generate=8 AILA_Q4E_LAYER_STAGE=1
python tmp/q4e-sprint-r1-20261014/run.py g5-a2 --prime --diag --generate=8
```

Read in this order: `[Q4E LayerStage] enabled`, then per count=4096 block
`[Q4E LayerStage] forward ...`, then `[Q4E AdaptiveWork]` `h2d_mb`/`bank_wait_ms`,
then the block wall and the SSE bodies.

Gates for the b arms: (1) `arena_experts` ~19k per block (48 layers x ~403 routed
cold experts) and `bank_experts` <= 5% with no WARN; (2) `h2d_mb` <= 2 GB and
`bank_wait_ms` <= 1.5 s per block (the arena's own 55 GB is not part of the bank
counter); (3) block wall <= 15 s (target ~10 s); (4) 2/2 pairs and A/A <= 1%;
(5) SSE bodies byte-identical to the a arms; (6) no `UR_OUT_OF_RESOURCES`; decode
continuation >= 15.4 tok/s. This arm is also the verdict arm for 6.1: if
`arena_experts` is ~0 and the WARN fires on a freshly built exe, the defect is in
the source, not the build.

## 7. Risks (G5)

- R1 build/edit race (hit for real this round): ninja silently linked a stale TU,
  so a GPU arm measured pre-edit code. Always assert `obj mtime > src mtime`
  before an arm (6.3) and keep the build log next to the receipt.
- R2 VRAM: the 1097 MB arena is paid from reserve headroom (run.py pins 3584 MB,
  leaving 2487 MB, above the option-C 2450 MB threshold). On
  `UR_OUT_OF_RESOURCES`, fall back to option A
  (`AILA_Q4E_MAX_GPU_SLOTS=2337`) or reserve 2450.
- R3 bit-exactness: arena slots are verbatim host-arena copies and descriptors
  mirror the staging math (section 2.2); gate 5 is the byte check.
- R4 counter cost: one mutex per epoch plus one log line per prefill forward,
  only when the gate is on; no hot-path change for the default.
- R5 if, on a freshly built exe, `bank_wait_ms` stays ~4.3 s without the WARN, a
  second staging path exists (for example the layer-major pump at
  `src/models/Qwen4ExpLayerMajor.cpp:617`); check that file next.

## 8. Coordinator verdict (2026-10-16 ~00:53) - NO-GO as implemented

Arms ran on the freshly built pair (worker `7a81823169ad`, server `79ac6ce953`,
obj mtime 00:33:11 > source 00:32:16, so the 6.1 stale-object trap is excluded).
Receipts: `tmp/q4e-sprint-r1-20261014/g5-{a1,b1,b2,a2}`.

Measured (14K prompt, `--prime --diag --generate=8`):

| arm | AILA_Q4E_LAYER_STAGE | ttft 6000 | ttft 14000 | decode 14K cont | block wall | bank_wait | fill | h2d (bank) |
|-----|----------------------|-----------|------------|-----------------|------------|-----------|------|------------|
| a1  | off                  | 112.0 s   | 69.6 s     | 15.24 tok/s     | 19.05/19.64/19.90 s | 3.60-3.77 s | 7.19-8.66 s | 46.8-47.6 GB |
| b1  | on                   | 112.1 s   | 73.4 s     | 6.83 tok/s      | 19.87/20.43/20.59 s | 0.05 s | 31-44 ms | 0.218 GB |
| b2  | on                   | 111.0 s   | 71.5 s     | 10.81 tok/s     | 19.38/19.89/20.10 s | 0.05 s | 30-43 ms | 0.218 GB |
| a2  | off                  | 110.7 s   | 70.2 s     | 15.26 tok/s     | -                   | -     | -      | -           |

What the arm proved (all positive): the consumer path works, not the g3
zero-hit. Per block `epochs=48`, `arena_experts=19,270-19,604`,
`bank_experts=83-91` (0.5%, no WARN), `arena_mb=52,402.8`. The bank pump is
genuinely retired: bank_wait 3.6 s -> 0.05 s, fill 7.2-8.7 s -> ~35 ms,
bank-path h2d 46.8 GB -> 0.218 GB. SSE bodies of all four arms are
byte-identical (`06100f64b407`, `8a64a37d9143`).

Why it is still NO-GO:

1. The supply cost is conserved, not hidden. The arena moves 52.4 GB/block
   (+12% over the 46.8 GB bank path, exactly the superset the audit predicted)
   and the wait reappears as `router_wait` 6.9-8.2 s -> 18.4-20.3 s. Block wall
   19.87-20.59 s vs 19.05-19.90 s baseline; TTFT 14000 is 2-5% worse.
2. Decode regresses hard: the 14K continuation drops 15.25 -> 6.83/10.81 tok/s,
   so the gate (>= 15.4 tok/s) fails independently of prefill.
3. Mechanism: the router must see post-attention hidden before layer L's expert
   ids exist (E6), so an epoch cannot be prefetched across layers, and both the
   arena H2D and the consumer sit on the same in-order queue. The arena changes
   who copies and where the wait is accounted, not whether supply and device
   compute overlap.

Consequence for the 400 tok/s goal: the only remaining lever is a real
overlap - start supplying layer L's experts while layer L's non-MoE device work
still runs and/or split the route launch so the first half of the block can
begin staging early (`tmp/q4e-sprint-r1-20261014/prefill-topk-split.patch`,
bit-exact, unapplied). Knob stays default-off; the counters and the per-forward
log line are kept because they made this verdict cheap to read.
