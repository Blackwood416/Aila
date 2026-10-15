# Qwen3.8-Flash-Next / A770 Sprint R1 -- P (prefill & supply) report

Date: 2026-10-14. Role: P (prefill & supply, charter S4). Repo HEAD 946c644,
build-refactor pair retained20 (worker sha 4d75c59c...).

Scope: per-4096-block prefill cost model, "if X were free" bounds, layer-major
feasibility, and probe requests. Hard discipline: this document contains ZERO
GPU actions (no Worker/Server/KernelTests run, no build, no git writes); every
number carries a source (`file:line` or artifact path); predictions are tagged
[estimate]; nothing from charter S2 (denied list) is re-run. Documents and
scripts only: this file plus `tmp/q4e-sprint-r1-20261014/prefill/`.

---

## 0. Executive summary

1. Current baseline per 4096-token block (adaptive pipeline, reduced 14K
   runner, serial staging fill arm "d0"): wall 19.24 / 19.83 / 20.07 s for the
   three full blocks; counters: experts_host 11.60-11.73 s (host fill
   7.21-7.32 + bank_wait 3.79-3.85), router_wait 7.52-8.22 s; H2D 45.7-46.5
   GiB per block. Evidence: `tmp/q4e-strata-c1/c1-d0-worker.log:32,34,36`;
   totals 70.03 s submitted vs 70.174/70.229 s first-content in
   `c1-d0-run.log` / `c1-a04-run.log`.
2. Host-side supply levers are measured zero-sum: parallel fill moves time
   from fill into bank_wait (11.0 s conserved); direct-DMA zeroes fill and
   bank_wait but router_wait absorbs it and wall is +2.9% (72.23 vs 70.23 s).
   Evidence: `docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md`
   (C1 doc) lines 24-33, 51-63; `tmp/q4e-strata-c1/c1-d2-worker.log:31-37`,
   `c1-e1-worker.log:33-39`.
3. The only >5% lever in P scope is layer-major (upload each cold key once per
   prompt). Recomputed floor from the real workload export: 24,352 of 24,576
   keys touched, 61.72 GB / 57.48 GiB "once", ~51-52 GiB after resident
   credit. Saving vs the observed single-14K chunk-major volume (179.53 GiB /
   192.74 GB) is 128.1 GiB = 137.6 GB = 14.95 s at 9.2 GB/s -> wall upper
   bound 55.3 s (-21%) [estimate]. Charter's -13..-18% is the loss-adjusted
   claim (`docs/plans/qwen38-roadmap-status-2026-10-12.md:39-40`).
4. Layer-major structural facts: the per-layer expert union is effectively the
   whole layer (min 491 / median 510 / max 512 of 512), so a retention-window
   design needs ~512 slots x 2.417 MiB = 1.21 GiB; the alternative "layer-wide
   single-pass MoE" avoids the window at the cost of ~0.15-0.3 GB scratch and
   changed row bookkeeping [estimate].
5. The tempting cheap variant -- "cross-chunk cold window" inside chunk-major
   -- is NO-GO by residency arithmetic: 48 layers x W slots must stay live for
   a whole chunk; W=16/layer costs 1.81 GiB for <=4% of a block's uploads,
   W=11 (1.25 GiB, roughly the reserve headroom) saves ~0.4 s. Section 3.3.
6. Resident-capacity oracle (extra aligned >=2,792 slots) is bounded at
   <=2.6 s / -3.7% [estimate] -- below the 5% gate, not probed.
7. GO/NO-GO for the layer-major minimal prototype is in Section 3.6; the two
   probe requests (both config-only, no src change) are in Section 4.

---

## 1. Per-4096-block cost table

### 1.1 Instrumentation that exists today (what can be printed, and where)

| Field group | Log tag / env | Source |
|---|---|---|
| submitted_ms, router_wait_ms, ple/hc/linear/full/experts/head_host_ms, fill_ms, bank_wait_ms, swap_ms, cpu_compute/join_ms, cpu_tasks, hits, misses, staged, h2d_mb | `[Q4E AdaptiveWork]`, `AILA_Q4E_WORK_SUMMARY=1` | `src/models/Qwen4ExpBnb4Backend.cpp:2489-2505` |
| ple, hc, linear_attn, full_attn, router, experts, acquire, shared, head (device-inclusive, serialized after each stage) | `[Q4E Stages]`, `AILA_PROFILE_Q4E_STAGES=1` | print `Qwen4ExpBnb4Backend.cpp:2655-2660`; sync-per-stage `:2005-2011` |
| router_sync, hits/misses/staged, DMA MB | `[Q4E Adaptive Profile]` | `Qwen4ExpBnb4Backend.cpp:2613-2616` |
| chunk boundaries: start/count/checkpoint_after | `[Q4E PrefillSlice]`, `AILA_Q4E_ACTIVE_PRESET` non-empty | `src/engine/EngineGeneration.cpp:97-99` |
| linear-attn dispatch mix (fused_gemm/cached/on_demand/sync_waits/direct_io) | `[Q4E Prefill Dispatch]` | `Qwen4ExpBnb4Backend.cpp:2641-2646` |
| startup only: arena layer total (weight refs / copy+upload), explicit reads total, source prefetch total | `[Q4E LoadTiming]`, `AILA_Q4E_LOAD_TIMING=1` | `src/models/Qwen4ExpHostArena.cpp:383-384`, `:430-431`, `:439-443` |
| startup knobs: `AILA_Q4E_ARENA_READ_THREADS` (0=mmap), `AILA_Q4E_ARENA_FILL_PREFETCH`, `AILA_Q4E_HOST_ARENA_PINNED` | env | `Qwen4ExpHostArena.cpp:282-283`, `:392-393`, `:73-84` (af12ca9) |

Notes:
- `router_wait_ms` is host time waiting on the device router event
  (`wait_actual_route`, `Qwen4ExpBnb4Backend.cpp:2163-2168`), so it includes
  whatever queue work precedes that event; it is not a pure D2H cost (D/R
  re-attribution item, charter unsolved 2).
- In steady-state prefill, host fill is a RAM->staging memcpy from the arena
  (`src/models/Qwen4ExpAdaptiveCache.cpp:621-624`), NOT a file read. NVMe /
  mmap / `ARENA_READ_THREADS` only matter at startup (af12ca9 cold start
  116 -> 63.8 s); they do not move per-block TTFT.
- Chunking: `AILA_Q35_PREFILL_CHUNK` (code default 512, production 4096),
  loop `EngineGeneration.cpp:72-100`; protocol cap 8192
  (`src/ipc/InferenceProtocol.hpp:110`).
- Kernel entries for the two device stages: GDN/linear
  `src/models/Qwen4ExpPrefill.cpp:111-118`; QSA sparse-prefill gate
  `(start_pos+count)/4 > 2048/4` -> `Qwen4ExpPrefill.cpp:258-263`;
  full attention `attention_prefill` / `attention_prefill_cached`
  `Qwen4ExpPrefill.cpp:545-547`, tiled variant `:560-566`. Any QSA probe must
  use a prompt above the ~2048-token gate to stay generalizable.

### 1.2 The table (baseline arm = serial staging fill, reduced 14K runner)

Blocks of the measured 14K request (token counts 4096/4096/4096/1656; reduced
runner warms with a 6K request first). All counters from
`tmp/q4e-strata-c1/c1-d0-worker.log:32,34,36,38`; wall from
`tmp/q4e-strata-c1/c1-d0-run.log` (`context-14000` 70.174 s).

| Item | Block 1 | Block 2 | Block 3 | Block 4 (1656) | Source | Status |
|---|---:|---:|---:|---:|---|---|
| wall = submitted_ms | 19.242 s | 19.834 s | 20.070 s | 10.884 s | c1-d0-worker.log | measured |
| H2D staged bytes | 45.70 GiB | 46.21 GiB | 46.48 GiB | 41.14 GiB | same (h2d_mb) | measured |
| H2D time @9.2 GB/s | 5.33 s | 5.39 s | 5.42 s | 4.80 s | H2D rate `charter:18` (9.2-9.8 GB/s) | [estimate] on measured bytes |
| cold unique keys staged | 19,360 | 19,577 | 19,692 | 17,431 | same (staged) | measured |
| staging batches (16 slots) | 1,210 | 1,224 | 1,231 | 1,090 | 16-slot default, `Qwen4ExpBnb4Backend.cpp:743-746` | derived |
| host fill (arena->bounce) | 7.214 s | 7.287 s | 7.319 s | 6.540 s | same (fill_ms) | measured |
| bank_wait | 3.840 s | 3.787 s | 3.850 s | 0.312 s | same | measured |
| experts_host counter | 11.604 s | 11.630 s | 11.727 s | 7.298 s | same | measured |
| router_wait | 7.515 s | 8.083 s | 8.224 s | 3.432 s | same; semantics caveat S1.1 | measured |
| ple / hc host | 0.46 ms / 35 ms | 0.46 / 35 | 0.46 / 35 | 0.63 / 45 | same | measured |
| GDN (linear_attn) device | 4.07 s | 4.07 s | 4.07 s | n/a | C1 doc:32, charter:32; serialized-stage accounting | needs re-measure (no `[Q4E Stages]` log archived for current code; nearest raw `tmp/q4e-bottleneck-stages-worker.log:2813-2835` shows 4.01-4.02 s at count=4096 on the legacy path) |
| full attention device | 2.85 s | 2.85 s | 2.85 s | n/a | same provenance; legacy raw 2.92-3.14 s | same |
| experts device | 12.58 s | 12.58 s | 12.58 s | n/a | charter:32 = fill 7.28 + bank_wait 3.76 + ~1.5 rest | same accounting caveat; async counters give 11.6-11.7 s |

Consistency notes:
- wall(19.24-20.07 s) ~= router_wait + experts_host + small stages in the
  async counters; the serialized per-stage numbers (GDN 4.07 + full 2.85 +
  experts 12.58 = 19.5 s) are a different, non-additive accounting. Do not mix.
- Parallel-fill arm d2 (`c1-d2-worker.log:31-37`): fill 2.42-2.46 s, bank_wait
  8.56-8.63 s, wall 19.47/20.04/20.23 s -> fill+bank conserved at ~11.0 s.
- Direct-DMA arm e1 (`c1-e1-worker.log:33-39`): fill ~0.001 s, bank_wait
  ~0.00005 s, experts_host 4.79-4.91 s, router_wait 14.5 s, wall 19.47/20.30/
  20.49 + 11.33 s; TTFT 72.231 s (`c1-e1-run.log`) = +2.9%.

### 1.3 H2D volume and overlap conclusion (quoting C1/C2; not re-run)

- Total staged bytes for this single 14K request: 179.53 GiB = 192.74 GB ->
  20.95 s of pure H2D at 9.2 GB/s, i.e. ~30% of the 70.2 s TTFT is copy
  traffic. The full agent sequence (14K+PI) is quoted as 310 GB
  (`docs/plans/qwen38-optimization-roadmap-2026-10-10.md:105`); VTune total
  332.5 GB / 34.6 s (`roadmap:116`). Scopes differ; do not mix.
- H2D cannot be hidden behind this workload: ALU-load ping-pong overlaps
  (4.04-4.18 ms vs serial 7.07), memory-bound load does not (6.70 ms vs serial
  7.16, max 3.70). C1 doc:36-48; C2 doc:11-25. Copy-only queue (L0 ordinal 1)
  runs at the same speed as the default queue (9.81 vs 9.76 GB/s; charter
  1.2) -> C2 NO-GO, do not re-run.

---

## 2. "If X were free" upper bounds (14K TTFT from 70.2 s)

Rules: each row zeroes ONE item; rows must NOT be summed (overlapping accounting).
Where an oracle requires code that is not in the tree, the row says so.

| # | If this were free | Oracle / method | 14K wall bound | Evidence |
|---|---|---|---|---|
| 1 | Per-layer re-transfer (layer-major floor) | replay of workload export + observed chunk-major bytes | 55.3 s (-21%) [estimate] | S3.1-3.2; 128.1 GiB saved / 14.95 s @9.2 |
| 2 | Staging H2D entirely (all bytes) | C1 "oracle=1" skip-device-memcpy arm (temporary patch, not in tree) | 52.71 s (-24.9%), contaminated: routing changed (staged 47.3 -> 25.4 GiB/block) | C1 doc:27,56; `c1-b04-run.log` |
| 3 | Host fill | measured d2 (parallel) / e1 (direct DMA) | 0 net (-0.3% / +2.9%) | C1 doc:24-31, 56; d2/e1 logs |
| 4 | bank_wait | measured d2/e1 (conserved into fill or absorbed elsewhere) | 0 net | C1 doc:24-31; d2/e1 logs |
| 5 | router_wait | not a supply item; needs D/R re-attribution (queue depth vs real wait) | <=26 s (-37%) naive upper bound | S1.1 semantics; d0 log |
| 6 | GDN + full-attn device stages | K domain; serialized-stage data needs current-code re-measure | <=23.5 s (-33%) naive upper bound | C1 doc:32; charter:75 |
| 7 | Resident capacity as an oracle (all 2,792 slots aligned per chunk) | union math on the export (config-only) | <=2.6 s (-3.7%) [estimate], below gate | S3.3 arithmetic |

Supply-bench mode mapping (what each mode covers, so nobody re-runs the wrong
thing). Tool: `perf/qwen4exp/q4e_prefill_supply_bench.cpp` (declared
`build-refactor/AilaQwen4ExpPrefillSupplyBench.exe`), modes at `:70-105`,
dispatch `:123-135`:
- mode 0 = resident: payload already in the pool, kernel only -> the
  kernel-side "H2D free" floor (row 2 at kernel granularity, synthetic shape
  256 experts x 8 rows, H=2560/I=640, slot 2,534,400 B).
- modes 1/4/6 = unrelated background H2D (whole-slot / one contiguous copy per
  batch / 512 KiB slices): "can a copy hide at all" probe; no data dependency.
- modes 2/5/7 = own payload with direct dependency (same three granularities):
  exposed-copy baseline. mode 7 = 512 KiB sliced dependent copy.
- mode 3 = same DAG plus explicit in-order submit barrier.
- `--bench` covers 0-5 at batch 256/64/32; `--sliced` covers {0,1,2,6,7} at 32.
- Coverage boundary: the bench is synthetic and has no router/resident tiers,
  so it bounds kernel-side exposure only -- e2e bounds come from C1/C2 arms.
  C2 already ran the equivalent queue-overlap probe; re-running it is a denied
  item. Re-run only if K wants batch 32/64/256 quad numbers for a kernel audit.

---

## 3. Layer-major feasibility evaluation (charter unsolved 1)

### 3.1 Recheck of the "H2D 310 GB -> ~62 GB" claim

- 310 GB = full agent-sequence H2D at chunk 4096 (roadmap:105). The single 14K
  request actually observed 179.53 GiB = 192.74 GB (S1.3). The 62 GB figure =
  24,576 keys x 2,534,400 B = 62.29 GB (58.0 GiB) = "each cold key once" model
  (charter:64, roadmap-status:39-40). Both statements are scope-mixed: the
  fair single-request comparison is 192.7 GB -> ~55 GB, and the full-sequence
  comparison is 310 GB -> ~62 GB. The model itself checks out.
- The model matches the RAM complement: 21,776 cold keys = 51.46 GiB
  (`docs/plans/qwen38-a770-agent-performance-plan-2026-10-12.md:215`), i.e.
  24,576 keys minus the 2,792 GPU-resident (clamp
  `src/models/Qwen4ExpBnb4Backend.cpp:751-754`).

### 3.2 Floor recomputed from the real workload export

Export `tmp/q4e-strata-c4/heldout-14k.profile` (full workload, C4 doc:19-22);
analysis `tmp/q4e-sprint-r1-20261014/prefill/layer_major_floor.py` (re-runnable):

| Quantity | Value |
|---|---:|
| keys with prefill>0 / any>0 | 24,352 / 24,365 of 24,576 |
| all-key "once" floor | 61.72 GB = 57.48 GiB -> 6.71 s @9.2 GB/s |
| cold floor after resident credit | ~51-52 GiB (matches plan:215) -> ~6.0 s |
| per-layer union (min / median / max) | 491 / 510 / 512 |
| observed chunk-major staged (d0, 4 blocks) | 179.53 GiB = 192.74 GB |
| layer-major saving (cold floor) | 128.1 GiB = 137.6 GB = 14.95 s @9.2 GB/s |

Caveats: the export aggregates the whole agent sequence (6K warm + 14K + PI),
so the single-14K union is slightly smaller; and 9.2 GB/s assumes the copy
path stays additive (S1.3). Probe P-1 below replaces this with a single-run
measurement.

### 3.3 Feasible structures and VRAM arithmetic

- Per-layer union is effectively the full layer (median 510/512). A retention
  window that keeps an expert across the layer's row sub-batches must hold the
  union: 512 x 2.417 MiB = 1.21 GiB (median 510 -> 1.20 GiB).
- Cross-chunk cold window inside chunk-major: to reuse layer L's experts in
  the next chunk, all 48 layers' windows must stay live for a whole chunk:
  48 x W x 2.417 MiB. W=16 -> 1.81 GiB and captures <=16/403 = 4% of a
  block's uploads (~0.6 s); W=11 -> 1.25 GiB (roughly the reserve headroom,
  roadmap:114) saving ~0.4 s. NO-GO; this also explains why every previous
  "bigger chunk / deeper ring" attempt was VRAM-starved.
- Layer split into groups of G layers: window = G x 1.21 GiB; G=1 is the only
  form that fits 16 GiB (G=2 = 2.42 GiB > headroom). I.e. layer-major is not
  one option among many -- it is the only VRAM-feasible ordering.
- Alternative that avoids the window: layer-wide single-pass MoE per layer
  (14,336 rows x top-k = 143,360 activations, ~287 rows/expert): stage 16
  experts at a time, each consumed fully in that pass (floor = keys once).
  Cost: row-scaled scratch (x/y bf16 ~70+70 MB, act/quant buffers scaled ~3.5x
  from the current 4096-row sizing [estimate]); changes the `count` semantics
  of the MoE row bookkeeping.
- Resident-oracle sanity (row 7 of S2): even perfectly aligned extra residency
  can absorb at most 2,792/19,360 = 14.4% of a block's keys -> <=0.8 s/block
  -> <=2.6 s total (-3.7%) [estimate]; the deployed hotset profiles already
  capture part of that. Below the 5% gate.

### 3.4 Engineering breakdown (what the days go into)

1. Driver: reorder `EngineGeneration.cpp:85-100` chunk loop (or add a backend
   entry) so layer L runs for all chunks back-to-back; KV writes stay causal
   (chunk i's K/V for layer L exists before chunk i+1 uses it). Hidden carry:
   4 chunks x 4096 x 2560 bf16 = 80 MiB for x, plus y/hc intermediates.
2. Scheduler/state machine: `Request.prefill_remaining`, cancellation at
   layer granularity, fault propagation (charter S5), `SnapshotPolicy`.
3. Snapshot/checkpoint interaction: production prefix reuse depends on an
   interior checkpoint (cached=14072 = 3x4096+1784; live record
   `docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md:161,244`).
   Layer-major must keep sub-chunk checkpoint capability
   (process the interior slice as its own sub-chunk) or accept coarser prefix
   reuse (~1,784 tokens re-prefilled ~= +2-3 s on resume) [estimate].
4. Per-layer host work (PLE/HC/ple_host_ms) must run once per layer, not per
   sub-batch (8192 lesson b: ple_host 0.1 -> 8.0 s, roadmap:106).
5. Budget: window 1.21 GiB vs reserve headroom ~1.3 GiB (roadmap:114) leaves
   nothing for hidden carry -> either trade ~100-200 resident slots (455
   slots ~= decode -2%, plan:96-97) or use the layer-wide-MoE variant.

### 3.5 Minimal prototype (per plan:215) and env-only approximation verdict

- Environment-only approximation of layer-major ordering does NOT exist: no
  env reorders the forward loop, and the b04 skip-memcpy oracle was a
  temporary patch (C1 doc:27) that is not in the tree. The closest config-only
  tools are the export replay (probe P-1) and a resident-oracle profile
  (bounded <=-3.7%, skipped).
- Smallest honest test = the plan:215 two-chunk prototype: two 4096 blocks
  (prompt 8192), keep the per-chunk attention/GDN math verbatim, reuse one
  layer's cold supply across both chunks, carry the required HC hidden blocks;
  measure peak VRAM (monitor/DXGI) and full wall against the P-2 baseline.
  This must be written by an implementation agent (multi-day); P supplies the
  baseline pair and the falsify gate.

### 3.6 Falsification criteria (pre-registered)

On the two-chunk prototype vs the P-2 baseline pair (same binary pair, A/B/B/A
if wall moves): 

1. H2D per prompt must drop >=30% (baseline expectation ~93 GiB for 8192
   tokens [estimate]; floor ~56 GiB) -- else the structure does not deliver.
2. Wall must improve >=5% (noise floor +/-0.6%, roadmap:4-5). If H2D drops but
   wall is flat -> the 8192 lesson repeated -> NO-GO without further work.
3. Peak VRAM within budget (no `UR_OUT_OF_RESOURCES`, reserve 3584 respected,
   peak memory monitor green).
4. Outputs unchanged under the P4/alignment tolerance contract (bitwise where
   the math is preserved).
5. P1 prefix reuse regression still passes (cached=14072 path).
6. Decode not regressed: resident slots untouched, or the slot trade is
   quantified and stays inside noise.

### 3.7 Which 8192 lessons would repeat, and the mitigations

| 8192 failure mode (roadmap:103-110) | Why it could repeat | Mitigation built into the prototype gate |
|---|---|---|
| bank_wait 17.7 -> 34.0 s | wider windows stall the ping-pong | layer-major has long consumption windows; measure bank_wait explicitly in the A/B |
| router_wait +9.7 s | fewer, bigger router calls change queue depth | keep per-chunk router cadence (4 calls/layer pass), same as today |
| per-layer host gap (ple_host 0.1 -> 8.0 s) | per-layer host work duplicated per sub-batch | run PLE/HC once per layer, share across sub-batches |
| +4.6 GiB forward reserve for 8192 rows | layer-major must not grow forward rows | keep 4096-row sub-batches; carry only hidden (~80 MiB) |
| chunk/staging "bigger is better" intuition | monotonicity already falsified (2048: 83.89, 4096: 70.57, 8192: 89.22) | prototype re-uses 4096 per-chunk math unchanged; only the order and the cold-upload count change |

---

## 4. Candidate ranking and probe requests

### 4.1 Ranking (by upper bound x feasibility)

1. Layer-major prefill (P scope, only >=5% lever here): upper bound -21%
   [estimate], roadmap claim -13..-18%, multi-day. GO gate = S3.6.
2. router_wait re-attribution (cross-domain, D/R + R): naive upper <=26 s
   (-37%); semantics unknown (queue depth vs real wait) -- a pure
   re-attribution cannot be claimed as a supply win until measured.
3. GDN + full-attn device stages (K scope): naive upper <=23.5 s (-33%);
   requires a current-code serialized re-measure (`AILA_PROFILE_Q4E_STAGES=1`)
   before kernel work is scheduled.
Explicitly not candidates (measured zero or denied): host fill, bank_wait,
staging copy parallelism, copy-only queue, chunk 8192/staging 64/ring 4,
resident-oracle profile (<= -3.7%), and the chunk-major cross-chunk window
(S3.3 NO-GO).

### 4.2 Probe P-1 -- single-workload union and window sizing (config-only)

Purpose: replace the workload-level export numbers (S3.2) with a run-scoped
measurement and size the window/scratch for the prototype; also re-confirms the
4-block chunk-major volume in the same log.

```powershell
# GPU mutex first: Get-Process AilaWorker,AilaServer must be empty.
python tmp/q4e-strata-c1/run_ttft14k.py p-lm-sizing `
  AILA_Q4E_HOTSET_EXPORT=E:/RiderProjects/Aila/tmp/q4e-sprint-r1-20261014/prefill/ttft14k-export.profile
python tmp/q4e-sprint-r1-20261014/prefill/layer_major_floor.py `
  tmp/q4e-sprint-r1-20261014/prefill/ttft14k-export.profile
```

Fields to read: worker log (`p-lm-sizing-worker.log`) 4x `[Q4E AdaptiveWork]`
pos=0/4096/8192/12288 -> staged / h2d_mb / submitted_ms (chunk-major volume);
script output -> per-layer union min/median/max and floor GB. Expected:
union >= ~470/layer, floor ~50-58 GiB. Decision rule: union <= ~430/layer ->
retention window fits the headroom; union >= ~490/layer -> choose the
layer-wide single-pass MoE variant. Note: export path must be absolute
(C4 doc:19-22 pitfall); export is written at clean worker teardown.

### 4.3 Probe P-2 -- two-chunk baseline pair for the prototype gate

Purpose: produce the A arm the layer-major prototype will be compared against
(8192-token prompt = exactly two 4096 chunks; warm-up 6K inside the fixture).

```powershell
python tmp/q4e-sprint-r1-20261014/prefill/run_ttft_pair.py p-8192-baseline
```

Fields to read: `p-8192-baseline-run.log` -> the `context-8192`
`first_content_seconds` (expected ~39-40 s [estimate] = 2 x ~19.5 s + IPC);
worker log -> two `[Q4E AdaptiveWork]` rows (expected ~46.5 GiB H2D each
[estimate]). This pair becomes the baseline for the S3.6 gate (>=30% H2D
reduction and >=5% wall improvement required).

Both probes are single-owner GPU runs with poll-based completion (runner polls
stdout with a 900 s cap; no fixed sleeps). Neither touches src/, tests/, tools/.

---

## 5. Source index (key references)

- `docs/plans/qwen38-performance-research-charter-2026-10-14.md:27-32` (per-block
  device split, H2D rate), `:64` (310->62), `:75` (GDN/attn question).
- `docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md:20-33`
  (baseline arm table), `:36-58` (non-overlap, oracle, verdicts).
- `docs/plans/qwen38-strata-c2-copy-queue-screen-2026-10-13.md:11-35`.
- `docs/plans/qwen38-optimization-roadmap-2026-10-10.md:103-110` (8192 / deep
  staging closure), `:116` (VTune H2D).
- `docs/plans/qwen38-roadmap-status-2026-10-12.md:39-40` (layer-major claim).
- `docs/plans/qwen38-a770-agent-performance-plan-2026-10-12.md:215` (minimal
  prototype shape), `:96-97` (slot trade), `:114` (reserve breakdown).
- Code: `src/ipc/InferenceProtocol.hpp:110` (protocol cap),
  `src/models/Qwen4ExpBnb4Backend.cpp:751-754` (resident clamp), `:743-746`
  (staging default); live prefix-reuse record in
  `docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md:161,244`.
- `docs/plans/qwen38-strata-c4-profile-mix-screen-2026-10-13.md:19-22`
  (export provenance and absolute-path pitfall).
- Logs: `tmp/q4e-strata-c1/c1-d0-worker.log`, `c1-d2-worker.log`,
  `c1-e1-worker.log`, `c1-a04-run.log`, `c1-d0-run.log`, `c1-b04-run.log`.
- Export: `tmp/q4e-strata-c4/heldout-14k.profile`.
- Scripts: `tmp/q4e-sprint-r1-20261014/prefill/layer_major_floor.py`,
  `run_ttft_pair.py`.
