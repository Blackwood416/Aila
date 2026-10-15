# R1 decode-attention GQA K/V sharing (role K3) - 2026-10-15

Repo HEAD `9abb858`, pair `retained23`. Charter: `docs/plans/qwen38-performance-research-charter-2026-10-14.md`
(sec 2 negatives, sec 4 role K, sec 5 discipline). This round took zero GPU actions (no `.exe`, no
cmake/ninja/build, no git writes). Deliverables: this document plus a test-only bench arm in
`tests/ops/Qwen4ExpKernelTests.cpp` (`--decode-attn-bench`). No `src/ops` production kernel was changed.

Target shapes: 48 layers = 12 full attention + 36 GDN; full attention Hq=24 / Hkv=2 / D=256; bf16 KV in the
retained arm (`docs/plans/qwen38-r1-kernel-2026-10-14.md:5-6`).

## 0. Summary

1. The 12x K/V re-read is real and sits in the production `attention_decode` / `attention_decode_rows` tile
   kernels, not in a helper: one work-group per (head, tile) reads its own K tile and V tile, and the 12
   q-heads of a kv group are 12 separate groups. Code evidence in section 2.
2. The retained 0.792 ms/row (len 14336) is 7.85x above the 0.101 ms/row unique-byte floor and 1.92x above
   the 0.412 ms/row all-L2 model, i.e. today only *part* of the 12x amplification is absorbed on-chip
   (`docs/plans/qwen38-r1-kernel-2026-10-14.md:81,131-139`).
3. The proposed design (section 3) is a phase-1 kernel that owns one (kv_head, tile) per work-group and
   runs the 12 q-heads of that kv group *sequentially through the identical tile body*, so the K/V tile is
   read once per kv head per row and the 12 revisits are served by L1/L2 instead of DRAM. It needs no SLM
   staging (SLM < 3 KiB per WG), and it keeps the per-tile partials bitwise identical to today, so the
   existing rows/pair bit-equality contracts stay valid.
4. Roofline: floor 0.101 ms/row (29.36 MB/row at 291.11 GB/s); the all-L2 bound is 0.412 ms/row at
   854.6 GB/s and by itself *fails* the 2x gate (1.92x), which is why the design must co-locate the reuse
   inside one work-group rather than only reorder the grid. Expected band `[estimate]` 0.12-0.25 ms/row,
   center 0.15-0.20 ms/row (3.2x-6.6x vs 0.792).
5. Bench: `AilaQwen4ExpKernelTests.exe --decode-attn-bench` (len 4096/8192/14336, rows 1-2, Hq24/Hkv2/D256,
   bf16, 20-sample medians, parseable JSON, flushed stage breadcrumbs). Adoption gate: >= 2x on
   `median_us_per_row` at rows=2 / len 14336, then the 12-layer account (section 4.4).

## 1. Retained numbers and constants (all sourced)

| Quantity | Value | Source |
|---|---:|---|
| Measured decode attention, 2 rows bf16, len 14336 | 1.5833 ms (0.792 ms/row `[estimate]`) | `docs/plans/qwen38-int8kv-vec-2026-10-12.md:23` (20-sample median) |
| Unique K/V bytes per row | 29.36 MB | `qwen38-r1-kernel-2026-10-14.md:81` (2 kv heads x 14336 x 256 x 2 B x 2) |
| Unique-byte DRAM floor | 0.101 ms/row | same, at 291.11 GB/s |
| 12x-amplified bytes per row (no reuse) | 352.3 MB | `qwen38-r1-kernel-2026-10-14.md:137` |
| No-reuse DRAM model | 1.210 ms/row | same (352.3 MB at 291.11 GB/s) |
| All-L2 model | 0.412 ms/row | same (352.3 MB at 854.6 GB/s) |
| Dense decode curve (external anchor) | 0.5581 ms at 8192, 1 row | `qwen38-r1-kernel-2026-10-14.md:82` (`qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md` curve) |
| 12 full layers x 0.792 | 9.5 ms/token `[estimate]` | derived from the row above; 12 layers from `qwen38-r1-kernel-2026-10-14.md:5-6` |
| Decode wall clock | 64.4 ms/token (K3 brief, tmp L3 ledger); 66.01 ms/token in the L3 cost model | `tmp/q4e-sprint-r1-20261014/COORD-NOTE-2026-10-15.md` (15.50/15.62 tok/s at aa-p1/p2, retained20); `docs/plans/qwen38-r1-backlog-mining-2026-10-15.md:130` |
| DRAM contiguous read (256 MiB, 256 B msgs) | 291.11 GB/s | skill `references/hardware/dram-contiguous-roofline.md:21-23` |
| L2 reuse | 854.6 GB/s | same `:25` |
| SLM read | 4115.8 GB/s | same `:26` |
| L1 per Xe-core / SLM per Xe-core / max SLM per WG / L2 | 192 KB / 128 KB / 64 KB / 16 MB | skill `references/hardware/hardware.md:31-33` |

## 2. Entry points and the exact re-read points (code evidence)

### 2.1 Dispatch map (file `src/ops/AttentionOps.cpp`)

| Entry | Condition | Kernel | Lines |
|---|---|---|---|
| `attention_decode` (1 row) | D=256, len >= 1024, partials buffer valid | `attention_decode_exact_head256_partial_merge` | def 432/738; dispatch 1749-1751 |
| `attention_decode` (1 row) | D=128, len >= 128, partials buffer valid | `attention_decode_exact_head128_partial_merge` | def 774/930; dispatch 1730-1732 |
| `attention_decode` (1 row) | D=128, bf16 KV, contiguous, joint-matrix allowed | `attention_decode_joint_matrix` (+ tiled variants) | def 955/1112-1131; dispatch 1741-1743 |
| `attention_decode` (1 row) | otherwise | `attention_decode_baseline` | def 126/270; dispatch 1758 |
| `attention_decode_rows` (1..4 rows) | all rows >= threshold and a 4-D partials block | multi-row exact merge (D=128/256) | def 1402; phase-1 grid 1454; dispatch 1865/1914-1931 |
| `attention_decode_rows` | D=128, len < 128 | multi-row fused joint matrix | def 1252; mapping 1286-1291; gate 1883-1888 |
| `attention_decode_rows` | D=256, all rows < 1024 | multi-row baseline | def 1140; dispatch 1936-1945 |

Paths taken by the bench shape (D=256, bf16, len >= 4096): rows=1 -> per-row `attention_decode` ->
exact-head256 partial merge; rows=2 -> `attention_decode_rows_exact_merge_impl<false, 256>` (the
`[ATTN] path=multi-row-exact-merge` log at :1895-1903).

### 2.2 Where K/V are re-read

Exact-head256 phase 1 (`src/ops/AttentionOps.cpp:432-737`), one group per (head, tile):

- Grid `num_heads * num_tiles * tile_wg` with `tile_wg = 128`, `tile = 128` (`:482`, `:93-98`); group ->
  head/tile mapping and `kv_head = head / heads_per_kv` at `:486-488` with `:453`.
- Each group loads its own q head into SLM (`:505-507`) and its own K tile: per lane one token row over
  `head_dim` (`:521-530`, vec8 bf16 path). The tile is not shared with the other 11 heads of the kv group;
  each group's V pass re-reads the V tile (bf16 `:675-677`; int8 token-major `:609-635`; int8 accessor
  `:655-656`).
- Phase 2 merges per-tile partials (max, sum, f32 256-wide V partial in a 272-float stride) with one group
  per head (`:686-737`, grid `:690`). Partial traffic is 10% of the K/V traffic at tile 128 (section 3.5).

Multi-row path keeps the same bodies verbatim (comment `:1131-1136`, `:1395-1401`); rows are independent
sequences with private partials blocks, so rows do not share K/V either. The head_dim 128 fused
joint-matrix path is a *short-context* fusion only: it needs `cached_len < kDecodeExact128MinLen = 128`
(`:99`, `:1884-1888`), one WG per (sequence, head) (`:1286-1291`), joint-matrix QK^T into SLM scores
(`:1323-1343`), then an SLM softmax and a per-head vec8 V pass (`:1373-1390`). It removes the global
scores round-trip; it does not share K/V across heads, so it carries the same 12x per-head reads.

### 2.3 Why only part of the 12x is absorbed today

- Per row per layer the multiplied re-read is 24 heads x 14336 tokens x 512 B x 2 (K and V) = 352.3 MB vs
  29.36 MB unique, factor 12.0 (`qwen38-r1-kernel-2026-10-14.md:81,137`; group-by-q-head note in the same
  section and in `qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md` item 3).
- The phase-1 grid is head-major (`head = group / num_tiles` at `:486-488`), so the same K/V tile is
  revisited only after a full per-head sweep. One kv head's K+V footprint at len 14336 is
  2 x 14336 x 256 x 2 = 14.68 MB against a 16 MB L2 (skill `hardware.md:33`), and both kv heads plus
  partials and Q exceed it, so tile revisits are a mix of L2 hits and DRAM misses.
- Measured 0.792 ms/row lies between the all-L2 model 0.412 ms and the no-reuse model 1.210 ms
  (`qwen38-r1-kernel-2026-10-14.md:135-139`), i.e. the re-reads are partially on-chip today and the lever
  is making the reuse explicit and co-resident instead of relying on scheduler order.

## 3. Design (proposal; not implemented in this round)

### 3.1 D1 (primary): one work-group per (kv_head, tile), head-serial reuse

- Grid: one WG per (sequence, kv_head, tile) = 2 x 112 = 224 WGs per row at len 14336 (today: 2688). WG
  stays 128 lanes with the same tile body (`tile_wg = 128`); inside the WG a loop runs `head =
  kv_head * 12 + slot, slot = 0..11`, each slot executing exactly today's phase-1 tile body (lane per
  token, ascending-d dot, `reduce_over_group` over the same 128 lanes, same partial writes).
- Effect: the K tile (128 x 256 x 2 B = 64 KiB) and the V tile (64 KiB) are touched once per kv head per
  tile; the 11 revisits from sibling heads hit the 192 KB L1 of the Xe-core that owns the WG, or L2.
  DRAM traffic per row drops from 352.3 MB toward the 29.36 MB unique bytes.
- SLM budget (cap 64 KiB per WG, skill `hardware.md:32`): q_cache 256 x f32 = 1 KiB, logits 128 x f32 =
  0.5 KiB, stats 8 B -> under 2 KiB per WG. No new SLM pressure; occupancy is not SLM-limited.
- Occupancy: 224 WGs x 4 hardware threads (128 lanes, SG32) = 896 of 4096 hardware threads (21.9%).
  `[HYPOTHESIS]` that is enough for a bandwidth-bound stream; rows=2 doubles the grid, and the bench arm
  measures both. If it is not enough, split each (kv_head, tile) into two half-tiles of 64 tokens with two
  64-lane-phase-1 calls that keep the 128-lane reduction by pairing them (see 3.4) or fall back to D4.

### 3.2 D2 (fallback within D1): stage the K tile in SLM

- If L1 thrash is measured (several resident WGs per Xe-core sharing 192 KB), stage only the K tile in SLM:
  tile 96 -> 48 KiB, tile 64 -> 32 KiB (both fit the 64 KiB cap with q/logits). V still streams through
  L1/L2. Numerics stay identical by construction (same per-token dot and reduction); only the tile count
  changes, which changes the *number* of partials, not their values.
- Cost: smaller tiles raise partial traffic. At tile 128 a row emits 24 x 112 x 1088 B = 2.93 MB of
  partials (10.0% of 29.36 MB); at tile 96 it is 3.92 MB (13.3%); at tile 64, 5.85 MB (19.9%).

### 3.3 Rejected: stage K+V for all 12 heads in one WG's SLM

K+V for a tile of T tokens is 2 x T x 256 x 2 B. The 64 KiB per-WG cap forces T <= 64, which either idles
half the 128 lanes (64 tokens on 128 lanes) or changes the reduction geometry (64-lane reduce), which
breaks the bitwise partial contract; and tile 64 doubles partial traffic to 19.9% of unique bytes (3.2).
Not proposed.

### 3.4 D4 (minimal-change variant, ships only with its own measurement)

Keep the head-parallel grid but reorder it tile-major ((tile, kv_head, head)) so the 12 sibling groups of
the same tile are adjacent in dispatch order. This targets the all-L2 model, which is 0.412 ms/row =
1.92x - below the 2x gate. Use it as the cheap control arm when measuring D1, not as the primary design.

### 3.5 Roofline targets and expected band

| Model | Bytes | Bandwidth | Time |
|---|---:|---:|---:|
| Unique K/V (floor) | 29.36 MB/row | 291.11 GB/s | 0.101 ms/row |
| 12x amplified, all L2 | 352.3 MB/row | 854.6 GB/s | 0.412 ms/row |
| 12x amplified, all DRAM | 352.3 MB/row | 291.11 GB/s | 1.210 ms/row |
| D1 on-chip revisits | 352.3 MB/row | 4115.8 GB/s (SLM read class) | 0.086 ms/row `[estimate]` |
| D1 partial merge (tile 128) | 2.93 MB/row | 291.11 GB/s | 0.010 ms/row `[estimate]` |

With D1 the K/V bytes become DRAM-unique and the on-chip revisit cost drops below the DRAM floor, so the
binding bound returns to 0.101 ms/row and the partial merge adds ~10%. Applying the 1.49x attainment
factor the project already measured on a comparable pure-stream kernel (`qwen38-r1-kernel-2026-10-14.md:77`
row a: 1.3809 ms vs 0.929 ms bound) gives about 0.15 ms/row. Expected band `[estimate]`: 0.12-0.25 ms/row
(3.2x-6.6x vs 0.792), center 0.15-0.20 ms/row. Gate: >= 2x means <= 0.396 ms/row, which the L2-only model
misses and D1 should pass.

### 3.6 Numerical exactness

The compile-time contracts are: a multi-row row output is bitwise identical to the single-row call
(`src/ops/Ops.hpp:371-376`), and the fused pair kernel is bitwise identical to two B1 calls
(`tests/ops/Qwen4ExpAttentionPairTests.hpp`, "Attention pair must match two B1 calls bitwise"). D1 keeps
them because (a) each per-(head, token) dot stays a single-lane ascending-d f32 accumulation,
(b) the tile max/sum reductions keep `reduce_over_group` over the same 128-lane group with the same
sub-group size, and (c) the V partial keeps the ascending-t per-lane accumulation into the same 272-float
stride. Partials are therefore bit-identical; only the grid changes. Any variant that changes the
reduction geometry (D3/3.4 hybrids) must re-baseline the pair kernel and its test in the same change.
INT8 KV: the D256 partial kernel already carries a `KVCacheAccessor<Fmt>` int8 path with the token-major V
variant (`:609-656`, `AILA_KV_QUANT_VEC`); D1 must reuse it per head slot, not hardcode bf16.

## 4. Bench arm (`--decode-attn-bench`)

### 4.1 Usage

```
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
build-refactor/AilaQwen4ExpKernelTests.exe --decode-attn-bench
```

Target `AilaQwen4ExpKernelTests` (`CMakeLists.txt:707-718`). The arm is test-only and does not change any
existing arm. Fixed parameters: Hq24/Hkv2/D256, bf16 KV, len 4096/8192/14336 (all above the 1024 exact
merge threshold and at/above the 2052 QSA sparse length, so the lengths match the route-eligible production
regime), rows 1..2, 3 warmups, 20 timed samples, shared K/V cache image across rows (retained pair
semantics), row 1 an exact copy of row 0.

### 4.2 Output

- Breadcrumbs `[DECODE ATTN] stage=...` (host_fill, allocate, upload, warm, verify, measure) with
  `warm_done` reporting `warm_ms`, plus `[DECODE ATTN FAIL] stage=... length=... rows=...` on the failure
  path and a `TerminateGuard` line on unhandled termination.
- One JSON row per (length, rows):
  `[DECODE ATTN COST] {"length":14336,"rows":2,"reps":20,"hq":24,"hkv":2,"head_dim":256,"dtype":"bf16",
  "warm_ms":...,"median_ms_total":...,"median_us_per_row":...,"min_us_per_row":...,"max_us_per_row":...}`
  Parse with `json.loads(line.split("] ", 1)[1])`.
- Correctness guard `[DECODE ATTN CHECK]`: all outputs finite, rows 1/2 bitwise equal when rows=2, and a
  CPU f64 reference over heads 0 and 12 with `max_abs <= 2e-4`.

### 4.3 Decision rule (as handed down)

Baseline expectation from the retained arm: `median_ms_total` ~ 1.583 ms (2 rows) / ~0.792 ms per row at
len 14336. A change is a lever only at >= 2x, i.e. rows=2 len 14336 `median_ms_total <= 792 us`
(`median_us_per_row <= 396 us`). Secondary sanity: the len 8192 rows=1 series should land near the
external dense anchor 0.5581 ms (`qwen38-r1-kernel-2026-10-14.md:82`) on the pre-change binary.

### 4.4 12-layer account

- Today: 12 x 0.792 = 9.5 ms/token `[estimate]` = 14.4-14.8% of the 64.4-66.0 ms/token wall.
- At the 2x gate: 12 x 0.396 = 4.75 ms/token, saving 4.75 ms/token ~ 7.2-7.4% `[estimate]` (matches the
  backlog's C2 estimate of ~-7% `[estimate]`, `qwen38-r1-backlog-mining-2026-10-15.md:75`).
- At the design center (0.15-0.20 ms/row): saving 7.1-7.7 ms/token, ~11-12% `[estimate]`; at the floor
  (0.101) it would be 8.3 ms/token ~ 13% `[estimate]`. The step change is only worth scheduling if the
  arm clears 2x.

## 5. Risks

1. Bitwise contracts: partial values must stay bit-identical, or the rows contract (`Ops.hpp:371-376`) and
   the pair test (`Qwen4ExpAttentionPairTests.hpp`) break. Keep the tile body, reduction geometry and
   partial layout; re-baseline them together if a variant changes the geometry (3.6).
2. Cancellation: decode attention is one submit per call today. D1 keeps a single submit and the per-slot
   4-D partials block, so the "drain GPU work before freeing USM" rule (AGENTS.md sec 6) is unchanged; do
   not hold K/V or partial pointers across calls.
3. Dynamic rows: `row_count` is 1..4 (`Ops.hpp:365-369`, checked at `:1773`). D1's grid is
   (sequence, kv_head, tile), so adding rows only multiplies group count; rows=3/4 and the mixed
   window/sink rows (which correctly fall back to per-row, conditions `:1801-1808` and `:1857-1888`) must
   be covered by the follow-up A/B, not only the bench's 1..2.
4. INT8 KV: the multi-row entry rejects s8 caches (`AttentionOps.cpp:1790`) because scales are per session;
   the single-row D256 kernel has the int8 and int8-DPAS variants. D1 must be written against
   `KVCacheAccessor<Fmt>` and the `AILA_KV_QUANT_VEC` token-major V path, and must not regress the int8
   numbers (1.0532 ms at 14336 after vectorization, `qwen38-int8kv-vec-2026-10-12.md:23`; the DPAS variant
   stays opt-in and negative, `:52-59` - do not reopen it).
5. L1 contention and occupancy: several resident WGs share a 192 KB L1, so some revisits will be L2-served;
   the all-L2 bound is 0.412 ms/row, so a partially-hitting D1 lands in 0.25-0.41 ms/row - still >= 1.9x,
   marginal for the gate. This is the main reason to measure before scheduling; D2 (SLM K staging) is the
   mitigation.
6. Tile tails: the last tile of a row is short (`tile_len < 128`); the existing `-1e30`/zero-fill lane
   pattern must be preserved per head slot, and the merge must keep reading only real partials.

## 6. Next actions

1. Land the bench arm (done in this round) and record a pre-change baseline run on the current binary.
2. Implement D1 behind an env gate (e.g. `AILA_ATTN_KV_SHARE=1`) in `attention_decode_exact_head256_partial_merge_impl`
   phase 1 plus the multi-row variant, reusing the phase-2 merge unchanged; no default flip in the same change.
3. A/B on the same binary with one variable: rows=2/len 14336 first, then rows=1 series; require >= 2x and
   no regression on rows=1 len 4096, then run the pair/rows bitwise suites plus the int8 three-arm regression.
4. If the gate fails but the direction holds, measure D2 (SLM K tile) before closing the line; if both fail,
   record the negative and keep the arm as the decode-attention timing gap closer for the kernel audit
   table row e (`qwen38-r1-kernel-2026-10-14.md:81`), which had no arm at all.
