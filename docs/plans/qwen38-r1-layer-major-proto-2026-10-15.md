# R1 A1: Qwen4Exp layer-major two-chunk prefill prototype (2026-10-15)

Status: v1 built and measured by the coordinator (lm-b2, wall 163.3 s vs
37.1 s, H2D 20.2 GiB vs 78.2 GiB); v2 changes are appended in Section 7 and
have not been built or run by A1. HEAD a3ee21e, pair retained22. Everything
below is either a source reference or a marked [estimate].

## 1. What this prototype does

When `AILA_Q4E_LAYER_MAJOR=1` and a prefill is exactly two full 4096-token
chunks (8192 rows), the backend processes the window layer-major:

- PLE (layer 1) and both hyper-connections run once per layer for the whole
  8192-row window.
- Attention/GDN still run as two independent 4096-row windows in causal order
  (chunk 0 at start_pos, chunk 1 at start_pos + 4096).
- Router and shared-expert work keep the per-chunk 4096 cadence.
- Cold experts are staged once per layer for the union of both chunks, and one
  expert-major `RunAdaptivePrefill` pass consumes all 8192 rows.

Default is fully unchanged: with the env unset, `ForwardImpl` takes the old
chunk-major path (`src/models/Qwen4ExpBnb4Backend.cpp:1760`).

## 2. Change list

| File:line | Change |
|---|---|
| `src/engine/EngineGeneration.cpp:86` | `layer_major_merge` flag: Qwen4Exp + `AILA_Q4E_LAYER_MAJOR=1` + chunk 4096 |
| `src/engine/EngineGeneration.cpp:90` | one-time `[Q4E LayerMajor] shape fallback` log for non-8192 shapes |
| `src/engine/EngineGeneration.cpp:114` | merge exactly two remaining full chunks into one backend forward; skipped when a ReusablePrefix anchor lies inside the window |
| `src/engine/EngineGeneration.cpp:119` | `[Q4E LayerMajor] merged two full chunks start=... count=8192` |
| `src/models/Qwen4ExpBnb4Backend.hpp:405` | `LayerMajorPrefillEligible()` declaration |
| `src/models/Qwen4ExpBnb4Backend.hpp:407` | `ForwardLayerMajorPrefill()` declaration |
| `src/models/Qwen4ExpBnb4Backend.hpp:420` | `RunAttentionWindowAt()` declaration |
| `src/models/Qwen4ExpBnb4Backend.cpp:1760` | dispatch into the layer-major path; one-time fallback log otherwise |
| `src/models/Qwen4ExpLayerMajor.cpp:71` | eligibility gate (count == 8192, prefill, no decode-verification, no interior checkpoint, no mrope/vision, adaptive cache present, num_experts == 512) |
| `src/models/Qwen4ExpLayerMajor.cpp:96` | attention tiling identical to `RunAttentionWindow` but with a row offset into `buf_.x`/`buf_.y` |
| `src/models/Qwen4ExpLayerMajor.cpp:122` | the layer-major forward |
| `src/models/Qwen4ExpLayerMajor.cpp:183` | PLE once for the 8192-row window |
| `src/models/Qwen4ExpLayerMajor.cpp:191` | attn HC once per layer |
| `src/models/Qwen4ExpLayerMajor.cpp:196` | two 4096-row attention/GDN windows, causal order |
| `src/models/Qwen4ExpLayerMajor.cpp:203` | MLP HC once per layer (produces the combined MoE input) |
| `src/models/Qwen4ExpLayerMajor.cpp:222` | router per chunk into offset host arrays |
| `src/models/Qwen4ExpLayerMajor.cpp:271` | one `RunAdaptivePrefill(count=8192)` per layer |
| `src/models/Qwen4ExpLayerMajor.cpp:317` | final snapshot + `currentLength_` publication |
| `CMakeLists.txt:201` | register `src/models/Qwen4ExpLayerMajor.cpp` |

No changes to `src/inference/Model.cpp`, `src/inference/Request.cpp`, `src/worker/*`,
`src/server/*`, `tests/*`, `tools/*`, or `docs/Environment_Variables.md`.

## 3. Correctness argument

### 3.1 KV causality and GDN state

`RunAttentionWindowAt` (`src/models/Qwen4ExpLayerMajor.cpp:96`) calls the
existing `RunAttentionLayerTile` with `start_pos + c*4096` and
`row_offset == c*4096`. Chunk 0 writes K/V/indexer/GDN state for positions
[start, start+4096); chunk 1 reads that state through the same
`session.layerCaches_[L]` and writes [start+4096, start+8192). This is the same
order and the same arithmetic as two sequential chunk-major forwards
(`src/engine/EngineGeneration.cpp:72` before this patch,
`src/models/Qwen4ExpPrefill.cpp:557`).

### 3.2 PLE / HC ordering

PLE layer 1 runs once over 8192 tokens; the dilated conv and recurrent state
are token-ordered inside `ple_dilated_conv_silu_add`
(`src/models/Qwen4ExpPle.cpp:689`). HC kernels are row-independent, so running
them over 8192 rows is mathematically identical to two 4096-row calls.

### 3.3 MoE

Router top-k is still computed per 4096-row chunk and written into
`router_host_ids_` at row offsets (`src/models/Qwen4ExpLayerMajor.cpp:222`).
`RunAdaptivePrefill` then observes the union for the full window and stages
each cold expert once per layer (`src/models/Qwen4ExpBnb4Backend.cpp:1435`).
Per-token down accumulation stays rank-ordered
(`moe_ordered_accumulate_down_rows`), while gate/up arithmetic per activation
is unchanged; only tile membership of the expert-major pass changes.

### 3.4 Checkpoint / prefix reuse

- Eligibility rejects `checkpoint_after != 0`
  (`src/models/Qwen4ExpLayerMajor.cpp:85`).
- The Engine merge is skipped if a ReusablePrefix anchor lies inside the
  8192-row window (`src/engine/EngineGeneration.cpp:114`).
- Final-state snapshot at `cached_len` is still captured
  (`src/models/Qwen4ExpLayerMajor.cpp:317`).
- Gap: a layer-major forward loses the interior 4096 snapshot that the
  chunk-major path would have taken after its first chunk. The production
  14072 prefix-reuse prompt never takes the merged path (remaining != 8192),
  so P1 is expected to be unaffected; a single 8192-token session can only
  truncate back to the 8192 boundary. TODO before production.

### 3.5 Cancellation

Cancellation is only observed by `RequestState::StepLocked` between backend
calls (`src/inference/Request.cpp:264`). The merged forward therefore doubles
the uninterruptible host window from ~4096 to ~8192 rows. No GPU work is freed
early and the existing drain-before-free rules are untouched. TODO: per-layer
cancel check for a long-running window.

### 3.6 MTP / vision

- MTP: the A/B arm keeps `AILA_Q4E_SERVER_MTP=0`. `Model::ForwardImpl` calls
  `mtp->AfterForward` once per backend forward; with MTP enabled the
  8192-token cadence differs from the 4096 cadence. TODO: gate layer-major on
  the MTP runtime (the flag lives outside the A1 file scope).
- Vision/mrope: rejected by eligibility
  (`src/models/Qwen4ExpLayerMajor.cpp:90`).
- Numerical note: attn HC, MLP HC, shared-expert GEMMs stay at 4096, but the
  two HC calls and the final mixer run at 8192 rows; bf16 GEMM last bits may
  differ from chunk-major. Attention/GDN/PLE are the same calls as chunk-major.

## 4. Expected gain and the main risk

- H2D floor (role-P export): 8192 window = 20761 key touches / 49.0 GiB vs
  measured chunk-major 78.0 GiB for the 8192 prompt
  (`tmp/q4e-sprint-r1-20261014/prefill/union8192.profile`,
  `layer_major_floor.py`). Gate: H2D
  reduction >= 30%, wall improvement >= 5%, noise +/-2%.
- This v1 still lets `ensure_runtime_buffers(ctx, 8192)` grow every
  capacity-sized scratch: `moe_down_acc` +336 MiB, PLE +~420 MiB,
  hidden/HC +~320 MiB, attn/linear phase +~250 MiB [estimate]. This is close
  to the documented "+4.6 GiB forward reserve for 8192 rows" failure mode
  (`docs/plans/qwen38-optimization-roadmap-2026-10-10.md:103`). If the arm
  shows `UR_OUT_OF_RESOURCES` or the bank_wait regression repeats, v2 is the
  role-P mitigation: keep 4096-row runtime buffers and add a dedicated 8192-row
  carry (hidden/x/y + pointer-parameterized HC and `RunGroupedExperts`).
- The per-layer log exists specifically to distinguish "H2D down, wall flat"
  (NO-GO) from a real win.

## 5. Verification arms (coordinator runs these; GPU mutex first)

GPU mutex check, then one arm at a time:

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # must be empty
```

A0 = pre-registered baseline (chunk 4096, flag off):

```powershell
python tmp/q4e-sprint-r1-20261014/run.py w-lm-a0 --no-fixture --prompts=8192 --generate=1 --diag AILA_Q4E_LAYER_MAJOR=0
```

`run.py` hardcodes `--chunk 4096` (`tmp/q4e-sprint-r1-20261014/run.py:61`),
which is the A0 shape. The merged path needs a single 8192-token prefill
command; the server accepts `--prefill-chunk-size 8192`
(`src/server/ServerMain.cpp:82`), but `run.py`/`qwen38_context_probe.py` pass
4096 (`tools/server/qwen38_context_probe.py:55`). Two options, both outside
the A1 file scope:

1. Direct probe arm (recommended, no source change):

```powershell
$config = Get-Content docs/plans/qwen38-server-mtp-2026-10-07.json -Raw | ConvertFrom-Json
foreach ($e in $config.worker_environment.PSObject.Properties) {
  [Environment]::SetEnvironmentVariable($e.Name, [string]$e.Value, 'Process')
}
$env:AILA_Q4E_SERVER_MTP='0'
$env:AILA_Q4E_VRAM_RESERVE_MB='3584'
$env:AILA_Q35_PREFILL_CHUNK='4096'
$env:AILA_Q4E_LAYER_MAJOR='1'
$env:AILA_Q4E_WORK_SUMMARY='1'
$env:AILA_SERVER_WORKER_LOG='E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/w-lm-b-worker.log'
python tools/server/qwen38_context_probe.py --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 `
  --output tmp/q4e-sprint-r1-20261014/w-lm-b --context 32768 --chunk 8192 `
  --generate 1 --reuse 1 --prompt-tokens 8192
```

2. If the coordinator prefers the run.py receipt form, add one scratch-side
   env lookup to `tmp/q4e-sprint-r1-20261014/run.py` (replace the hardcoded
   `--chunk 4096`) or to `qwen38_context_probe.py`, then use the run.py-shaped
   command with `AILA_Q4E_LAYER_MAJOR=1`.

Recommended receipts: A0/B pair plus B/B repeat and A0/A0 repeat (A/B/B/A), and
optionally a caller-only control (chunk 8192, flag 0) to separate the caller
merge from the backend layer-major logic.

### 5.1 Fields to read

- `w-lm-*-run.log`: `context-8192` -> `first_content_seconds` (TTFT gate).
- worker log, chunk-major arms: two `[Q4E AdaptiveWork]` rows with
  `count=4096`; sum `staged`, `h2d_mb`, `fill_ms`, `bank_wait_ms`.
- worker log, layer-major arm: one `[Q4E AdaptiveWork]` row with `count=8192`
  and `layer_major=1`; the same fields are already summed for the window.
- worker log: 48 `[Q4E LayerMajorLayer]` rows with per-layer
  `staged/h2d_mb/fill_ms/bank_wait_ms` (the union evidence per layer).
- worker log: `[Q4E LayerMajor] begin/done` (wall) and
  `[Q4E LayerMajor] fallback chunk-major` (any shape that did not merge).
- VRAM: run monitor samples + absence of `UR_OUT_OF_RESOURCES` /
  `USM allocation failed`.
- Output parity: compare first generated token / first content text between A0
  and B; expect identical text, with possible last-bit bf16 differences in the
  8192-row HC/mixer GEMMs (Section 3.6).

## 6. Open TODOs

- v2 carry-buffer variant if v1 VRAM/wall fails (Section 4).
- Interior checkpoint support inside the 8192 window (Section 3.4).
- Per-layer cancellation (Section 3.5).
- MTP cadence guard (Section 3.6).
- PLE lookahead across the next outer chunk is disabled on this path
  (`src/models/Qwen4ExpLayerMajor.cpp:160`).
- Server/worker path must forward an 8192-token prefill command before this
  prototype can be measured through `AilaWorker.exe` (Section 5); the CLI
  (`EngineGeneration`) path already merges.
- First build and first GPU run are intentionally left to the coordinator.

## 7. v2: pipelined, half-split expert supply (2026-10-15)

### 7.1 Measured v1 evidence (coordinator, `tmp/q4e-sprint-r1-20261014/lm-b2`)

| Metric | chunk-major (lm-a1) | layer-major v1 (lm-b2) |
|---|---:|---:|
| TTFT (prompt 8192) | 37.06 s | 163.34 s |
| H2D | 78.15 GiB | 20.18 GiB |
| staged experts | 33,108 | 8,349 |
| `bank_wait_ms` | 7,506 + 7,471 | 107,008 |
| `router_wait_ms` | 7,748 + 8,513 | 46,711 |
| `fill_ms` | 2,054 + 2,001 | 1,051 |

v1 reached `[Q4E LayerMajor] begin count=8192` and all 48
`[Q4E LayerMajorLayer]` rows, so the H2D mechanism is proven. The two large
waits are both queue-drain style intervals: `bank_wait` is the wait for the
previous copy on the same ping-pong bank, whose memcpy depends on the previous
consumer's kernels (`src/models/Qwen4ExpAdaptiveCache.cpp:594-640`). In effect
v1's per-layer expert GPU time was ~6-8x the chunk-major kernel shapes, not a
missing ping-pong (the shared `RunAdaptivePrefill` loop already issues copy
i+1 after submitting compute i).

### 7.2 v2 changes

| File:line | Change |
|---|---|
| `src/models/Qwen4ExpBnb4Backend.cpp:755` | `AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS` overrides `AILA_Q4E_STAGING_SLOTS` only when `AILA_Q4E_LAYER_MAJOR=1`; default 16, VRAM cost 2 x N x 2.417 MiB (N=32 -> ~155 MiB, and 32 fewer resident slots from the staging reserve) |
| `src/models/Qwen4ExpBnb4Backend.hpp:437` / `src/models/Qwen4ExpBnb4Backend.cpp:1494` | `RunGroupedExperts` gains defaulted `route_base`, `token_base`, `use_counts`; chunk-major call sites are unchanged |
| `src/models/Qwen4ExpLayerMajor.cpp:361` | `RunAdaptivePrefillLayerMajor()` |
| `src/models/Qwen4ExpLayerMajor.cpp:407` | expert batch size = min(env, cache staging depth), default 16 |
| `src/models/Qwen4ExpLayerMajor.cpp:421` | every staged bank is consumed by two 4096-row halves (`route_base`/`token_base`), so kernel shapes match the fast chunk-major path while the bank is staged once |
| `src/models/Qwen4ExpLayerMajor.cpp:441` | resident hits are processed in bounded microbatches instead of one large launch |
| `src/models/Qwen4ExpLayerMajor.cpp:469` | lookahead: batch i+1 is staged before batch i is consumed |
| `src/models/Qwen4ExpLayerMajor.cpp:488` | one rank-ordered merge per 4096-row half, after all banks are consumed (identical arithmetic to chunk-major) |
| `src/models/Qwen4ExpLayerMajor.cpp:275` | layer-major forward now calls the v2 expert pass |
| `src/models/Qwen4ExpLayerMajor.cpp:280` | `[Q4E LayerMajorLayer]` now reports `experts_ms` and `wall_ms` in addition to staged/h2d/fill/bank_wait |
| `src/models/Qwen4ExpLayerMajor.cpp:337` | `[Q4E AdaptiveWork]` now reports the real `experts_host_ms` instead of 0 |

The staging lookahead order was already correct in the shared loop; the v2
levers are (a) half-split kernel shapes, (b) bounded resident batches, and
(c) optional deeper banks. All are inside the existing `AILA_Q4E_LAYER_MAJOR=1`
gate, so the default path is untouched.

### 7.3 Expected effect and residual risks

- H2D stays at the v1 level (one union per layer): the measured 20.2 GiB /
  8,349 staged is already far below the 54.7 GiB gate.
- `bank_wait` should drop toward the chunk-major order of magnitude if the
  8192-row expert composition was the cause; `[Q4E LayerMajorLayer]
  experts_ms/wall_ms` now gives the direct per-layer attribution.
- Residual risk 1: v2 still calls `ensure_runtime_buffers(ctx, 8192)`, so all
  capacity-sized scratch stays doubled (same VRAM risk as v1, Section 4).
- Residual risk 2: the final mixer, PLE and both hyper-connections still run at
  8192 rows; if `experts_ms` drops but `wall_ms` does not, those GEMMs are the
  next suspect and need a per-half call variant.
- Residual risk 3: merging at half granularity changes nothing numerically
  (per-token rank order unchanged), but the final output parity check against
  the chunk-major arm still applies.

### 7.4 Verification command (coordinator)

```powershell
python tmp/q4e-sprint-r1-20261014/run.py lm-b3 --no-fixture --prompts=6000,8268 `
  --generate=1 --diag --chunk=8192 --AILA_Q4E_LAYER_MAJOR=1 `
  --AILA_Q4E_STAGE_COPY_PARALLEL=1 --AILA_Q4E_4TOKEN_WGSHARE=4
```

Gate unchanged: TTFT <= 35.2 s, H2D <= 54.7 GiB. Read
`[Q4E LayerMajor] expert batch=...`,
all 48 `[Q4E LayerMajorLayer]` `experts_ms/wall_ms/bank_wait_ms`, and the
single `count=8192 layer_major=1` `[Q4E AdaptiveWork]` row.
To test the deeper banks, add
`--AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS=32` (batch becomes 32, VRAM cost in
Section 7.2).

## 8. v3: per-half modules + 8192-row carries (2026-10-15)

### 8.1 Measured v2 evidence (`tmp/q4e-sprint-r1-20261014/lm-b3`)

| Metric | v2 |
|---|---:|
| TTFT (prompt 8192) | 61.95 s (v1 163.3, baseline 37.06) |
| H2D / staged | 19.7 GiB / 8,349 |
| fill / bank_wait | 1,052 ms / 20,772 ms |
| sum `experts_ms` | 22,374 ms |
| sum `wall_ms` | 61,054 ms |

`experts_ms` is measured around `RunAdaptivePrefillLayerMajor()` and therefore
already contains the fill and bank_wait intervals. The real non-expert
residual is wall - experts = 38.7 s, dominated by layer 1 (7.45 s, the
8192-row PLE) and ~0.66 s/layer elsewhere. Everything else in v2 still ran
with capacity-sized scratch at 8192 rows.

### 8.2 v3 changes

| File:line | Change |
|---|---|
| `src/models/Qwen4ExpLayerMajor.cpp:161` | `EnsureLayerMajorCarries()`: hidden/x/y/inject/moe_h/down/shared carries, allocated once, ~1.0 GiB at count=8192 |
| `src/models/Qwen4ExpLayerMajor.cpp:192` | `EnsureLayerMajorMaps()`: grows the route and expert-map host/device mirrors to count*top_k because the shared-staging pass indexes the full 8192-row window globally |
| `src/models/Qwen4ExpLayerMajor.cpp:72` | embedding replication into the 8192-row hidden carry |
| `src/models/Qwen4ExpLayerMajor.cpp:266` | forward prologue stays at `ensure_runtime_buffers(ctx, 4096)`; carries/mirrors are the only 8192-row allocations |
| `src/models/Qwen4ExpLayerMajor.cpp:292` | PLE per 4096-token half on the carry (was one 8192-row call) |
| `src/models/Qwen4ExpLayerMajor.cpp:309` | attn HC + attention + MLP HC per 4096-row half using the standard scratch, results copied to the carries |
| `src/models/Qwen4ExpLayerMajor.cpp:328` | router + shared expert per 4096-row half |
| `src/models/Qwen4ExpLayerMajor.cpp:377` | expert pass unchanged from v2 but reads/writes the carries |
| `src/models/Qwen4ExpLayerMajor.cpp:396` | shared add + residual injection per half |
| `src/models/Qwen4ExpLayerMajor.cpp:419` | mixer and LM head only for the last 4096-row half |
| `src/models/Qwen4ExpBnb4Backend.hpp:437` | `RunGroupedExperts` optional x/moe_h/down bases (defaults keep chunk-major unchanged) |
| `src/models/Qwen4ExpBnb4Backend.hpp:446` | `RunAdaptivePrefillLayerMajor` optional carry pointers |
| `src/models/Qwen4ExpBnb4Backend.cpp:1070` | map-capacity reset when the standard runtime scratch is reallocated |

32-slot staging bug fix (reported from `lm-b4`, runtime fault
`Invalid Qwen4Exp staging host-copy batch`):

| File:line | Change |
|---|---|
| `src/models/Qwen4ExpAdaptiveCache.cpp:621` | the parallel host copy is split into <= 16-slot calls, so `AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS=32` no longer trips the executor's input validation |

Cost of 32 slots (recorded from `lm-b4`): `staging_experts=32`,
`slots=2760` instead of 2792 (2 x 32 x 2.417 MiB staging, -32 resident slots).

### 8.3 Expected effect and residual risks

- Every non-expert module is back to the chunk-major 4096-row shape, so the
  38.7 s residual should collapse toward the baseline's ~2 s.
- The expert pass keeps one staging union per layer and the v2 pipelining, so
  H2D/staged should stay at ~19.7 GiB / 8,349 and `bank_wait` should not
  regress.
- Residual risk A: the carries are ~1.0 GiB; `lm_down_` alone is 671 MiB
  because the shared-staging pass needs rank rows for the full window. If
  VRAM pressure itself was the problem, this only removes part of it.
- Residual risk B: per-expert activation counts are still half-window counts
  (~188 vs ~95 in chunk-major), so `experts_ms` may stay ~22 s; the next
  lever is the deeper-bank arm and, if needed, a per-half staging schedule.
- Residual risk C: output parity still needs the chunk-major comparison; all
  attention/GDN/PLE/HC/router/shared arithmetic is now 4096-row identical.

### 8.4 Verification (coordinator)

```powershell
python tmp/q4e-sprint-r1-20261014/run.py lm-b5 --no-fixture --prompts=6000,8268 `
  --generate=1 --diag --chunk=8192 --AILA_Q4E_LAYER_MAJOR=1 `
  --AILA_Q4E_STAGE_COPY_PARALLEL=1 --AILA_Q4E_4TOKEN_WGSHARE=4
```

Gate unchanged: TTFT <= 35.2 s, H2D <= 54.7 GiB. Read
`[Q4E LayerMajor] carries ...`, the 48 `[Q4E LayerMajorLayer]` rows
(`experts_ms` vs `wall_ms` split), and the now self-contained
`[Q4E LayerMajor] done ... experts_ms/router_wait_ms/fill_ms/bank_wait_ms/
staged/h2d_mb` line (the lm-b3 worker log did not contain the window-level
`layer_major=1` AdaptiveWork row, so v3 repeats the totals on `done`).
