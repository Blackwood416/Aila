# R1 A1-v4: layer-major cross-layer staging pipeline (2026-10-15)

Status: v4 implemented by A1-v4 on top of HEAD `9abb858` (pair `retained23`).
One source file touched (`src/models/Qwen4ExpLayerMajor.cpp`). NOT built and NOT
run by A1-v4: the task forbade any build or GPU action, so everything below is
either a source reference, a number read from an existing receipt, or marked
`[estimate]`.

Gate this arm is measured against (unchanged): 8192-token prompt, TTFT
<= 35.2 s and H2D <= 54.7 GiB (`qwen38-r1-layer-major-proto-2026-10-15.md:8.4`).

## 0. Verdict in one line

The staging path is not the binding constraint of the layer-major window: the
48 per-layer `bank_wait_ms` sum to 32.1 s while the staged H2D volume is only
4.95 s of DMA and 2.37 s of worker fill (section 3.2). Cross-layer staging can
therefore recover at most the per-layer head exposure, ~6 ms x 48 `[estimate]`.
v4 implements that plus the missing module split; the gate will not be met by
v4 (section 5) and the reason is structural, not a shape choice (section 2).

## 1. Task-1 findings: the fill, its thread, and its overlap

| Question | Finding | Source |
|---|---|---|
| Which thread runs the host fill? | The **calling worker thread**, synchronously: `StageColdExpertsAsync` does wait -> fill -> submit in one call. | `src/models/Qwen4ExpAdaptiveCache.cpp:594-648` |
| What does it wait on first? | `staging_pending_[pp].wait_and_throw()` - the **previous H2D of the same ping-pong bank**. | `src/models/Qwen4ExpAdaptiveCache.cpp:603` |
| What does that H2D depend on? | `cgh.depends_on(staging_reader_[pp])`, i.e. the **previous consumer kernels of that bank**. | `src/models/Qwen4ExpAdaptiveCache.cpp:640-643` |
| Is the fill parallel? | Only with `AILA_Q4E_STAGE_COPY_PARALLEL=1`: the executor TBB copy is bounded to 16 slots/call and **joins before returning**. Otherwise it is a plain per-slot `std::memcpy` loop. | `src/models/Qwen4ExpAdaptiveCache.cpp:615-632`, `src/models/Qwen4ExpExpertExecutor.cpp:547-573` |
| Can the fill move off the worker thread? | Not with the current object: there is exactly **one host bank per ping-pong buffer** (`staging_host_[2]`), so the next fill must wait for the in-flight H2D of that bank. Moving it off-thread needs >= 2 host banks per device bank (v5 item 2). | `src/models/Qwen4ExpAdaptiveCache.hpp:326-331` |
| Is chunk-major's async staging reusable? | It **is the same code**: chunk-major and layer-major both call `StageColdExpertsAsync` with the same global `ping_pong_index_`; only the union source (4096-row chunk vs 8192-row window) and the batch size differ. There is nothing chunk-major-specific left to port. | `src/models/Qwen4ExpBnb4Backend.cpp:1477-1487`, `src/models/Qwen4ExpLayerMajor.cpp:606-632` |

So the honest answer to task 1 is: the fill is worker-thread synchronous **and
gated by the previous consumer of the same bank**, not merely "on the CPU".
That gate is what the 32.1 s of `bank_wait` measures.

## 2. The barrier that actually exists

The pre-registered NEXT ACTION was "stage layer L+1 while layer L computes".
That cannot be implemented for the routing-derived set:

1. Layer L+1's cold union comes from L+1's router top-k
   (`src/models/Qwen4ExpLayerMajor.cpp:342-346`, v4 tree).
2. L+1's router input is the **MLP-HC output of L+1**, which is computed from
   `lm_y_` / `lm_hidden_` of layer L after **all** of L's expert batches were
   merged (`src/models/Qwen4ExpLayerMajor.cpp:405-438`, v4 tree).
3. Therefore the identity of L+1's cold experts is not known until L's expert
   pass has already finished. A true cross-layer staging prefetch of that set
   is impossible; only a *speculative* set is available (v5 item 3), and
   section 3.2 shows the DMA path has ~7x slack, so speculation cannot pay.

What *is* available, and what v4 implements, is the head overlap: the union of
the **current** layer is known before its resident-hit phase, so the head cold
batches can be staged (fill + H2D) while the hit kernels run, instead of being
issued after them and sitting on the critical path of the first cold
consumption.

## 3. Evidence used by this revision

### 3.1 v3 (lm-b5) is 42.4 GiB of H2D, not 19.7 GiB

The 48 `[Q4E LayerMajorLayer]` rows of `tmp/q4e-sprint-r1-20261014/lm-b5-worker.log`
sum to (sums recomputed from the raw log for this doc):

| quantity | v3 (lm-b5) sum | chunk-major (lm-a1) |
|---|---:|---:|
| window wall (per-layer sum) | 61,569.992 ms | TTFT 37.06 s |
| TTFT (run log, `context-8268`) | 62.2996 s | 37.06 s |
| `experts_ms` | 35,549.914 ms | not logged per layer |
| `bank_wait_ms` | 32,092.489 ms | 7,506 + 7,471 ms |
| `fill_ms` | 2,368.982 ms | 2,054 + 2,001 ms |
| staged experts | 17,951 | 33,108 uploads |
| `h2d_mb` | 43,387.432 MiB = 42.37 GiB | 78.15 GiB |
| wall - `experts_ms` (non-expert modules) | 26,020.078 ms | ~18,028 ms |

Sources: `tmp/q4e-sprint-r1-20261014/lm-b5-run.log` (`context-8268` ->
`first_content_seconds`), `tmp/q4e-sprint-r1-20261014/lm-b5-worker.log`
(48 per-layer rows, one `[Q4E LayerMajor] begin pos=0 count=8192`), and
`docs/plans/qwen38-r1-cost-model-2026-10-14.md:504` + `:7.1` table for the
chunk-major column.

Correction for the sibling docs: `qwen38-r1-layer-major-proto-2026-10-15.md`
S8.1 and `qwen38-r1-cost-model-2026-10-14.md:511` list v3 as "19.7 GiB /
8,349"; those are the **v2 (`lm-b3`)** totals (same sums in `lm-b3-worker.log`).
v3 includes `EnsureLayerMajorMaps`, which fixes the v2 host-map overrun, so the
true 8192-row union is staged: 17,951 experts / 42.37 GiB. The H2D gate still
passes (42.37 vs 78.15 GiB = -45.8%), but the "-75% H2D" headline belongs to
v1/v2 only and must not be quoted for v3.

### 3.2 Why the staging path is not the constraint

| interval | value | source |
|---|---:|---|
| staged H2D volume | 42.37 GiB (= 45.5 GB) | lm-b5 per-layer `h2d_mb` sum |
| same at the measured link rate 9.2 GB/s | 4.95 s | link rate: `qwen38-r1-cost-model-2026-10-14.md:273` |
| worker fill | 2.37 s | lm-b5 per-layer `fill_ms` sum |
| `bank_wait` (host blocked on the same-bank copy) | 32.09 s | lm-b5 per-layer `bank_wait_ms` sum |
| everything else inside `experts_ms` (host self-time) | 1.09 s | 35,549.914 - 32,092.489 - 2,368.982 |

The wait is 6.5x the DMA floor and 13x the host fill, and the copy it waits for
transitively depends on the previous consumer kernels of that bank
(`Qwen4ExpAdaptiveCache.cpp:640-643`). Conclusion: **v3's expert pass is
consumer-bound on the compute queue**: of the 35.55 s pass the host blocks on a
same-bank copy for 32.09 s (90%), and spends only 2.37 s filling plus 1.09 s of
other host work. Chunk-major, for comparison, is 14.98 s of `bank_wait` against
a 9.1 s DMA floor (1.65x), i.e. partly copy-bound.

The same decomposition says the expert phase is the larger regression: v3
spends 35.5 s of host-blocking time where chunk-major spends 19.0 s
(14.98 + 4.06) for the same 8192 x 10 x 48 activation work, and v3's
non-expert modules add 26.0 s against chunk-major's ~18.0 s.

## 4. v4 change list

All of it is in `src/models/Qwen4ExpLayerMajor.cpp`; no header, CMake, tests or
tools change. Line numbers are the v4 working tree.

| File:line | Change |
|---|---|
| `src/models/Qwen4ExpLayerMajor.cpp:253-256` | read-once `AILA_Q4E_LAYER_MAJOR_PIPELINE` gate + module-split accumulators |
| `...:299,309` | PLE interval (layer 1) |
| `...:313,348` | per-half prologue interval (attn HC + attention/GDN + MLP HC + carry copies + router + topk submit) |
| `...:352,375` | shared-expert interval |
| `...:408,430` | post-expert mix interval (shared add + `hc_inject_add`) |
| `...:433-438` | `[Q4E LayerMajorPipeline] progress` cumulative row, one per layer |
| `...:443,465` | final mixer + LM head interval |
| `...:505` | `[Q4E LayerMajorPipeline] window` row (module split) |
| `...:586-591` | read-once pipeline flag + one-time `pipeline=1 depth=2` line |
| `...:642-657` | `PendingStage fifo[2]` + `push_stage`; when gated, prime **both** banks before the resident phase |
| `...:658-670` | hit phase, then the v3-style stage/consume loop driven from the ring |
| `...:693` | `[Q4E LayerMajorPipeline] layer=... stage_head_ms=...` per layer |

### 4.1 Schedule (gated arm)

```
router -> route wait -> union -> [prime bank A + bank B]  <- new: fill + H2D
                              -> resident-hit microbatches   (overlaps the 2 copies)
                              -> consume A, push C, consume B, push D, ...
                              -> merges (unchanged)
```

### 4.2 Why the ungated path is unchanged

When `AILA_Q4E_LAYER_MAJOR_PIPELINE` is unset the ring is not primed
(`:652` is behind the flag), so the first `push_stage` happens inside the loop
at `:663`; tracing both versions gives the identical call sequence
`stage(b0), stage(b1), consume(b0), stage(b2), consume(b1), ...` with the same
`ping_pong_index_` alternation, the same `adaptive_prefill_slots_` write order
and the same `RunGroupedExperts` arguments. The added `read_flag` calls are
side-effect free (`src/utils/EnvUtils.hpp:18-37`), the timers are `steady_clock`
reads behind the flag, and the new log rows are emitted only when the flag is
set. So `AILA_Q4E_LAYER_MAJOR=1` alone must reproduce lm-b5 bit for bit.

## 5. Expected effect `[estimate]`

- Recoverable per layer = head fill + head H2D + the exposure of the first cold
  consumption = ~2 ms + ~4 ms + ~0 = ~6 ms, i.e. **-0.29 s of the 61.57 s
  window (-0.5%)** `[estimate]`. Inputs: lm-b5 fill 2,368.982 ms / 48 layers /
  ~25 batches = ~2 ms/batch; 16 slots x 2.417 MiB (`Qwen4ExpAdaptiveCache.hpp:36`)
  at 9.2 GB/s = 4.2 ms; the previous layer's last bank copy is 2 consumers old
  at the layer boundary, so its wait should already be ~0.
- `bank_wait` sum: unchanged to slightly lower; if the coordinator instead sees
  a drop far larger than 0.3 s, the wait had a non-consumer component and v5
  item 2 (deeper bank ring) becomes the right lever `[estimate]`.
- TTFT: 62.30 s -> 62.0..62.3 s `[estimate]`. **The 35.2 s gate will not be met
  by v4.** H2D stays at 42.37 GiB, already inside the 54.7 GiB gate.
- v4's value is the two diagnostics: it tests the consumer-bound reading
  directly, and the `progress`/`window` rows split the previously unattributed
  26.0 s of non-expert module time.

## 6. Risks

1. **Barrier relocation, not removal.** The head stage of layer L still waits
   for the last bank copy of layer L-1 (`Qwen4ExpAdaptiveCache.cpp:603`). That
   wait used to be paid inside the resident phase; it is now paid before it.
   Net ~0 by construction; wall regression is possible if that copy is still in
   flight (`-0.2 s` worst case `[estimate]`).
2. **Log tail loss.** `lm-b5`'s worker log contains neither the window
   `[Q4E LayerMajor] done` row nor the `layer_major=1` `[Q4E AdaptiveWork]` row
   (0 hits for both strings in the raw file), i.e. the tail of that log was
   lost at teardown. The v4 module split is therefore emitted both as a
   per-layer cumulative `progress` row and as the window row, so 47 of 48 rows
   are enough to reconstruct every module total.
3. **Correctness.** No arithmetic, routing, checkpoint, snapshot or cancellation
   change: only the *host-side order* of two `StageColdExpertsAsync` calls is
   moved ahead of the hit phase. `adaptive_prefill_slots_` entries written by
   priming belong to cold experts, the hit phase reads only hit ids
   (`prefill_unique_[0, hits)` vs `[hits, total_unique)` are disjoint), and the
   copies still gate their consumers through `StagingBatch::ready_event`.
4. **MTP, checkpoint, chunk-major, decode**: untouched. The gate lives inside
   the already-gated layer-major path only.
5. Working tree note: `tests/ops/Qwen4ExpKernelTests.cpp` is dirty in the tree
   but was **not** touched by A1-v4.

## 7. Static self-check (no build)

- Brace/paren/bracket balance of the edited file: 0/0/0 (comment- and
  string-stripped count); 697 lines.
- `AILA_LOG_INFO` format/arg counts checked pairwise: progress (8), window (8),
  layer (5).
- `std::chrono::steady_clock` used inside `RunAdaptivePrefillLayerMajor`
  (`<chrono>` included at `:7`), `aila::env::read_flag` via the existing
  `../utils/EnvUtils.hpp` include, `PendingStage`/`StagingBatch` copied by value
  (POD: two fixed `std::array`s + `sycl::event`).
- No new signatures, no header/CMake change, no new file besides this doc.
- Not verified by A1-v4: compilation, VRAM headroom, runtime logs.

## 8. Verification (coordinator, GPU mutex first)

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # must be empty

# B arm: layer-major + pipeline
python tmp/q4e-sprint-r1-20261014/run.py lm-v4-a --diag --chunk=8192 --prompts=6000,8192 `
  AILA_Q4E_LAYER_MAJOR=1 AILA_Q4E_LAYER_MAJOR_PIPELINE=1

# A0 control: layer-major, no pipeline (must match lm-b5: 62.30 s / 42.37 GiB)
python tmp/q4e-sprint-r1-20261014/run.py lm-v4-a0 --diag --chunk=8192 --prompts=6000,8192 `
  AILA_Q4E_LAYER_MAJOR=1
```

`run.py` accepts both `KEY=VALUE` and `--KEY=VALUE` (`tmp/q4e-sprint-r1-20261014/run.py:56-63`);
`--diag` is required, it sets `AILA_Q4E_WORK_SUMMARY=1` and the per-label worker
log (`run.py:39-42`). Prefer A0/B/B/A0 ordering; the A/B pair should differ only
in the pipeline flag.

Fields to read:

- `lm-v4-*-run.log`: `context-8192` -> `first_content_seconds` (gate).
- worker log, B arm: `[Q4E LayerMajor] pipeline=1 depth=2` once.
- worker log: 48 `[Q4E LayerMajorPipeline] layer=... stage_head_ms=...` rows;
  sum it - if it stays <= ~0.3 s the head exposure was never the cost.
- worker log: 48 cumulative `[Q4E LayerMajorPipeline] progress` rows; per-layer
  deltas give `ple_ms/halves_ms/shared_ms/route_wait_ms/experts_ms/mix_ms`, and
  the last row gives the totals of the 26.0 s non-expert block.
- worker log: 48 `[Q4E LayerMajorLayer]` rows; sum `wall_ms`, `bank_wait_ms`,
  `fill_ms`, `h2d_mb`; compare with the lm-b5 sums in section 3.1.
- window totals: `[Q4E LayerMajor] done ... h2d_mb=...` (if the log tail
  survives) or the 48-row sums, which is what lm-b5 required.
- VRAM: monitor samples, absence of `UR_OUT_OF_RESOURCES` / `USM allocation
  failed`; A0 must be byte-identical output to lm-b5, B must match A0.

## 9. v5 directions (ordered by expected value)

1. **Consumer side first (16.5 s scale).** v3 blocks the host on the same-bank
   copy once per microbatch, and the copy it waits for is 2 batches behind the
   consumer chain. Two candidate levers, both measurable with the new
   `progress` rows: (a) issue the resident hits of a layer as **one** batch, as
   chunk-major does (`src/models/Qwen4ExpBnb4Backend.cpp:1470-1474`) instead of
   <= 16-expert microbatches - this needs the tile/act mirrors grown to
   `count * top_k` first, because `expert_tiles_host_`/`dev_` stay sized
   `cap * top_k` (`src/models/Qwen4ExpBnb4Backend.cpp:1116-1118`); (b) increase
   `AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS` only together with (a), since with 2
   banks the batch size does not change the consumer-bound steady state.
2. **Deeper bank ring with >= 2 host banks per device bank**
   (`Qwen4ExpAdaptiveCache` change: `staging_dev_/host_/copy_/reader_` from 2 to
   N, `GetStagingSlotDevicePtr`/`RecordStagingConsumer` modulus fix). This is
   the only change that lets the worker run ahead of the consumer chain; VRAM
   cost 16 x 2.417 MiB = 38.7 MiB per extra bank pair, paid from the same
   staging reserve (v3's 32-slot run cost 32 resident slots, `proto:8.2`).
   Do it only if v4 shows the wait is not purely consumer-bound.
3. **Speculative cross-layer union.** Stage the next layer's most likely cold
   experts from the adaptive cache's `usage_` table during that layer's
   attention phase. Bound by section 3.2: the DMA path has ~7x slack, so this
   can only remove the head exposure (already done by v4) - expect <= 1%
   `[estimate]`. Not worth the bank-arbitration complexity before (1) and (2).
4. **Attribute the 26.0 s non-expert block** with the v4 `progress` rows before
   changing any of it: `halves_ms` (2 attention windows + 2 HC + router),
   `route_wait_ms`, `mix_ms`, `head_ms`, `ple_ms`. Chunk-major runs the same
   operations for ~18.0 s inside a 37.06 s window, so this block holds ~8 s of
   unexplained cost that no staging change can recover.

Do not re-run shape sweeps on the layer-major window until (1) and (4) are
answered; v1-v3 already showed shape changes do not move this wall
(`qwen38-r1-cost-model-2026-10-14.md:511-529`).
