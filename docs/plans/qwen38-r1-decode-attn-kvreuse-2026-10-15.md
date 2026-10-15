# R1 (role B3): decode attention K/V re-read (GQA reuse) - audit, variants, bench plan

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, HEAD `9abb858`.
Scope: this document + `tmp/q4e-sprint-r1-20261014/decode-attn/` only.
Discipline: zero GPU actions were taken by this role (no exe, no test, no build,
no GPU command, no src/tests/tools edit, no git write; prof-h4 holds the GPU).
Every measured number is cited from an existing artifact; every derived number is
marked `[estimate]`. Text is ASCII only.

Target geometry (all numbers below unless stated): Qwen3.8-Flash-Next full
attention, Hq=24 / Hkv=2 / D=256, bf16 KV cache, T1 (one row) or the B=2 arm of
the `--kv-fp8-only` test. Source of geometry: config.json cited in
`docs/plans/qwen38-r1-kernel-2026-10-14.md:3`.

---

## 0. Conclusions

1. **Mechanism confirmed (STRONG).** Decode attention groups work by query head:
   `grid = num_heads * num_tiles`, one WG per (head, 128-token tile), in all three
   decode bodies - exact-merge D256 (`src/ops/AttentionOps.cpp:452-680`),
   exact-merge D128 (`:800-877`), baseline (`:1140-1236`, wg 512). With 24 q-heads
   over 2 KV heads every K row and every V row is therefore read 12x. The byte
   stream is 352.3 MB/row at len 14336 against 29.36 MB unique (12.0x).
2. **The 12x stream is a real byte cost, not free L2 credit.** Measured dense
   throughput on the *amplified* bytes rises with length 201 -> 270 -> 356 ->
   447 GB/s (len 2048/4096/8192/14336) against the 291.11 GB/s contiguous-DRAM
   and 854.6 GB/s L2 reference points (SKILL `dram-contiguous-roofline.md:21-25`).
   The long-length marginal segment is 4.64 us per 128-token tile = 678 GB/s on
   the amplified stream, i.e. duplicate reads are being served at L2-like rates.
3. **Gap decomposition at len 14336 (0.787 ms/row measured).** Unique K/V DRAM
   0.101 ms (13%, [K audit] `qwen38-r1-kernel-2026-10-14.md:133-134`); duplicate
   K/V stream 0.42-0.58 ms at observed effective rates (55-70% `[estimate]`);
   residual 0.10-0.25 ms of per-element work that does not shrink with L2 (V is
   read as 2-byte scalar loads: 88.1M lane loads per row `[estimate]`, source
   `AttentionOps.cpp:637-677`), softmax, merge and launch. Ratio to the pure
   min-byte floor is 7.85x at 14336 (K audit) and 17.4x at len 2048.
4. **Two variants are designed, both keep the partials layout (stride 272) and
   the per-head merge kernel untouched.** B = head-group inside phase 1 (G q-heads
   per WG; G in {2,4,12}); *bit-exact by construction* (per-head accumulation
   order verbatim), 2-4 dev days, 1.5-2.6x expected. A = KV-head-wide WG modeled
   on Strata `attn_chunk_kernel` (12 q-heads, q tile 12 KB + probs 6 KB SLM, one
   16B K load per lane per cell, one V element per thread); not bit-exact
   (score/softmax reduction order changes), 4-6 dev days, 2.2-3.5x expected.
5. **Decisive planning fact: the production 14K agent decode does NOT run dense
   attention.** With the QSA indexer active and visible prefix > 2048 the decode
   branch routes to ~512 active blocks = ~2052 tokens
   (`src/models/Qwen4ExpPrefill.cpp:219-250`, `:283-295`). Routed attention is
   ~0.27 ms/layer (dense-2052 analog, `qsa-cost-curve-A1.log`) + indexer
   select ~0.07 -> ~4.2 ms/token = 6.5% of the 64.4 ms/token wall
   (`qwen38-r1-decode-2026-10-14.md:42`). So at the production shape this lever's
   e2e ceiling is <=6.5% and a realistic -2..-3%; the -5% e2e gate is reachable
   only in a dense long-context configuration (indexer off / non-QSA model),
   where 12 layers cost 9.4 ms/token (14.6%) and a 50% kernel cut is -7.3%.
6. **Probe plan: 2 existing arms + 2 new.** R1 `--qsa-probe` (dense 4096/8192 T1,
   D256, gate shapes), R2 `--kv-fp8-only` (B=2, len 333/1834/14336 incl. the long
   arm + short-shape regression). R3 new `--decode-attn-reuse-bench` with
   `AILA_ATTN_KVREUSE=0|pair2|pair4|head12` (one binary, single variable, dense
   D256 4096/8192/14336 + D128 4096/16384). R4 e2e A/B on the 14K agent arm.
   Thresholds: kernel >=3% on both gate shapes (recommend requiring >=15% at 4096
   and >=25% at 8192 to be worth the routed-shape e2e exposure); no >2% regression
   at len 333/1834; e2e >=5% (expected to pass only in the dense profile - see 5).
7. **No charter section 2 negative is reopened.** The closest entries are
   "2 rows/subgroup" (decode GEMV rows, not attention q-heads) and "SLM input
   caching in GEMV" (vec8 bank conflicts); variants A/B do not stage K/V in SLM,
   they stage q (12 KB) and probabilities (6 KB) whose reads are broadcasts or
   contiguous per-lane slices. The 09-30 audit already listed this lever as
   unimplemented item 3 (`qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md:66`).

---

## 1. Gap decomposition

### 1.1 Source map (what actually runs)

| case | kernel | grid | source |
|---|---|---|---|
| len >= 1024, D=256 | exact partial merge, 2 launches | phase1 `num_heads*num_tiles*128`, merge `num_heads*128` | `AttentionOps.cpp:452-744` |
| len >= 128, D=128 | exact partial merge, 2 launches | same shape | `AttentionOps.cpp:800-934` |
| below those thresholds / window / sink / int8 without scales | baseline | `row_count*num_heads*512` | `AttentionOps.cpp:1140-1236` |
| multi-row (B>1) | per-row calls, or row copies of the merge kernels (`attention_decode_rows`) | - | `AttentionOps.cpp:1767-1947` |
| <128 tokens D=128 | fused joint-matrix | `row_count*num_heads*128` | `AttentionOps.cpp:1252-1395` |

Phase-1 body per (head, tile) WG, tile=128 tokens, WG=128 threads, D=256
(`AttentionOps.cpp:470-680`):

1. Stage q[256] as float into SLM (2 values per thread), barrier.
2. Score: thread `lid < tile_len` owns virtual token `t = tile_start+lid` and
   serial-accumulates all 256 dims in d-major order via 32 `vec8<bf16>` loads
   (+256 cvt + 256 FMA lane ops).
3. `reduce_over_group` tile max, exp in place, `reduce_over_group` tile sum,
   write `partial[0]=stats max`, `partial[1]=tile sum`.
4. V partials: for `d = lid; d < 256; d += 128` every thread accumulates
   `acc += logits[i] * v_acc.load(kv_head, cache_t, d)` over the tile
   (`:637-677`). This is a scalar 2-byte load per element with a 512-byte row
   stride - per tile WG: 256 loads/thread x 128 threads = 32768 lane loads.

Merge phase per head (`:685-744`): lid 0 folds num_tiles tile stats (max, exp
factors, norm), then 128 threads write 256 dims (+ optional sigmoid gate).

### 1.2 Byte model per row (bf16, D=256, Hq=24/Hkv=2)

Per token per KV head: K = 256x2 B = 512 B, V = 512 B.
Unique per row = `2 * L * 512 * 2`; amplified (x12) = `24576 * L` bytes.

| len | unique | amplified | floor @291.11 | floor @560 | L2 dup time @854.6 `[estimate]` |
|---:|---:|---:|---:|---:|---:|
| 2048 | 4.19 MB | 50.33 MB | 0.0144 ms | 0.0075 ms | 0.0540 ms |
| 4096 | 8.39 MB | 100.66 MB | 0.0288 ms | 0.0150 ms | 0.1080 ms |
| 8192 | 16.78 MB | 201.33 MB | 0.0576 ms | 0.0300 ms | 0.2160 ms |
| 14336 | 29.36 MB | 352.32 MB | 0.1009 ms | 0.0525 ms | 0.3779 ms |
| 16384 | 33.55 MB | 402.65 MB | 0.1153 ms | 0.0600 ms | 0.4319 ms |

D128 parity note: D=128 at len 16384 has exactly the same amplified byte count
as D=256 at len 8192 (201.33 MB) and the same unique 16.78 MB, so the measured
D256@8192 point is a byte-parity estimate for a D128@16384 arm `[estimate]`.

### 1.3 Measured points (existing artifacts, no new GPU work)

| arm | shape | measured | amplified-rate | source |
|---|---|---:|---:|---|
| dense T1 D256 len 2048 | `--qsa-probe` reps=15 median | 0.2503 / 0.2526 ms | 201 GB/s | `tmp/q4e-sprint-r1-20261014/kernel/qsa-cost-curve-A1.log:16,19` |
| dense T1 len 2051/2052 | dense control / first sparse | 0.266 / 0.266 ms | 189 GB/s | same `:30,44` |
| dense T1 len 4096 | gate shape | 0.3730 / 0.3736 ms | 270 GB/s | same `:58,61` |
| dense T1 len 8192 | gate shape | 0.5649 / 0.5638 ms | 356 GB/s | same `:72,75` |
| dense T1 len 8192 (09-30 curve) | cross-check | 0.5581 ms | 361 GB/s | `qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md` curve table |
| bf16 B=2 len 1834 | `--kv-fp8-only` | 0.4438/2 = 0.222 ms/row | - | `tmp/q4e-int8-dpas-20261012/kv-dpas-off-1.log`, `-2.log` |
| bf16 B=2 len 14336 | long arm (the K-audit point) | 1.57485/2 = 0.787 ms/row | 447 GB/s | same logs; `qwen38-r1-kernel-2026-10-14.md:81` |
| bf16 B=2 len 333 | short arm | 0.4659/2 = 0.233 ms/row | - | same log |
| routed (score+select+attention) len 8192 | production analog | 0.356 / 0.3591 ms | - | `qsa-cost-curve-A1.log:72,75` |
| int8 VEC=1 len 14336 | read-path A/B | 1.0532 ms (bf16 1.5833) | - | `qwen38-int8kv-vec-2026-10-12.md:15-19` |

Note on the two harnesses: `--qsa-probe` times one `attention_decode` plus a
synchronize per rep (T1); `--kv-fp8-only` times two row calls plus one
synchronize (B=2). Per-rep launch+drain overhead is therefore inside both
numbers and is *not* per-layer production cost (production is async and deep in
queue).

### 1.4 Decomposition at len 14336 (0.787 ms/row)

| bucket | time | share | basis |
|---|---:|---:|---|
| unique K/V DRAM (unavoidable) | 0.101 ms | 13% | K audit `:133-134`, @291.11 |
| duplicate K/V (11/12 of the stream) | 0.42-0.58 ms `[estimate]` | 55-70% | measured effective rates 270-678 GB/s on the duplicated bytes; between DRAM 291 and L2 854.6 |
| per-element work that survives L2 hits: V scalar loads (88.1M lane loads/row `[estimate]` from `:637-677`), bf16 cvt+FMA chains, tile reductions, merge, launches | 0.10-0.25 ms `[estimate]` | 13-32% | residual + int8 read-path arm (-43.2% at identical amplification, `qwen38-int8kv-vec:15-19`) |

Marginal-rate check (longest segment, most representative of large len):
(0.7874 - 0.5649) / (112-64) tiles = 4.64 us per 128-token tile; one tile
carries 3.146 MB of amplified K/V (128 tokens x 1 KB x 24 heads) -> 678 GB/s,
close to the 854.6 GB/s L2 reference and far above the 291.11 GB/s DRAM
reference. Interpretation: at long context the kernel is running at L2-like
rates on a stream that is 12x larger than necessary. Fitting `t = a + b*tiles`
through the 2048 and 14336 points gives a = 0.161 ms, b = 5.59 us, residuals
+0.033 (4096) and +0.046 (8192) `[estimate]`; the "fixed" term is at least
partly per-rep launch/sync overhead, so do not treat it as an intrinsic
per-layer cost.

### 1.5 Why both memory and issue must be attacked

- Byte side: the duplicate stream is served at 678 GB/s (L2-like) at the
  margin, so killing the duplicates removes a real L2-level cost, not a
  hypothetical one.
- Issue side: at the same 12x amplification the int8 arm (byte loads instead of
  bf16 scalar cvt loads, scale hoisted) runs 1.0532 ms vs bf16 1.5833 ms at
  14336 (-33.5%) and its own read-path-only change was -43.2%
  (`qwen38-int8kv-vec-2026-10-12.md:15-19`). A reuse variant that only fixes
  bytes but keeps the per-element bf16 scalar V path will therefore land above
  the pure byte model; the variants below also cut V loads and cvt work by the
  group factor.

---

## 2. Variant design

Both variants live inside phase 1 only; the partials layout (`stats[16] +
acc[256]`, stride 272 for D256 / 144 for D128) and the per-head merge kernel
(`AttentionOps.cpp:685-744`, `:879-934`) are unchanged, so the merge cost and
the gate fusion are untouched. Neither variant stages K or V in SLM; only q and
per-tile probabilities live in SLM.

### 2.1 Variant B - head group inside the existing phase-1 WG (recommended first)

Change: grid becomes `(num_heads/G) * num_tiles` instead of
`num_heads * num_tiles`; WG stays 128. Thread `lid < tile_len` still owns virtual
token `t`; for each `vec8<bf16>` K chunk it loads the 16 B once, converts once,
then computes G partial scores against `q_cache[g][*]` in SLM, accumulating each
head in the verbatim d-major order. Tile softmax then runs per head with the
existing 128-thread `reduce_over_group` order. V phase: thread keeps `acc[G][2]`
for columns `lid` and `lid+128`; each token loads `v_lo`,`v_hi` once and folds
them into all G heads (`acc[g][k] += logits[g][i] * v`), same token-major order
as today.

Bit-exactness: every score, tile max, tile sum and V partial is produced by the
same operation sequence on the same operands as today; only load reuse changes.
Expected bit-identical output vs the current kernel (residual risk: compiler
reassociation - keep the per-head expression textually identical and check the
`[KV ACC] bitwise_eq` field).

Group size trade-off: bytes and V-loads scale as 1/G; per-thread instruction
count grows (G score FMAs per K element, G x 2 V FMAs per V element) but WG
count drops as 1/G. Grid occupancy: G=2 -> 1344 WGs/row, G=4 -> 672, G=12 -> 224
(28.7k threads for the whole row at G=12 - the occupancy risk).

### 2.2 Variant A - KV-head-wide WG (higher ceiling, Strata model)

Model: Strata `attn_chunk_kernel` (`tmp/Strata/src/kernels/cuda/qsa_decode_attn.cu:120-276`)
uses G=12, CHUNK=64 cells, THREADS=256, grid (n_chunks, n_kv_heads); SLM holds
`sq[12][256]` float (12 KB) + `sp[12][64]`; the score loop has each lane hold 8
dimensions of one cell, loading the K row once per cell and computing all 12
heads' partial dots, then one warp reduction per head; the V loop has "thread t
owns dimension t for all 12 heads" with `float acc[G]` - one 2-byte V load per
cell per thread, unique by construction. Results go to per-chunk partials and a
separate merge kernel - the same two-phase structure Aila already has.

Aila adaptation: WG=256, tile 128 tokens (keep `tile_t`), grid
`2 kv_heads * num_tiles` (224 WGs at len 14336 - thin; if occupancy measures
bad, split the token dimension into sub-tiles with their own partials slots,
or widen the grid by processing a subset of heads per WG). SLM: `q_all[12][256]` float 12 KB +
`probs[12][128]` float 6 KB + stats (<=0.1 KB) = ~18.1 KB, under the 32 KB
per-WG caution line in the SKILL (`references/hardware/hardware.md:23-24,67`).
Score phase: SG16 lane l owns dims `[16l, 16l+16)`; per cell load 2 `vec8` K
(32 B), convert 16, compute 12 partials (16 FMA each) and 12 SG reductions; per
cell bytes read once per WG (16 SGs x 8 cells x 512 B = 64 KB = the tile).
Softmax phase: 12 SGs, one per head (4 idle), tokens strided by 16. V phase:
thread t owns dim t for all 12 heads, `acc[12]` registers, one V element per
cell, 12 FMA + broadcast reads of `probs[g][c]`.

Bit-exactness: NOT bit-exact. Score reduction changes from "1 thread serial
256-dim chain" to "16-lane partial + SG tree", and the tile sum reduction tree
changes from the 128-thread strided order. Guard: mirror the QSA probe's f64
reference check (max_abs <= 2e-4) plus a bf16-output comparison vs the current
kernel; keep the merge phase as-is.

### 2.3 Budgets

| variant | WG | grid (len 14336, T1) | SLM | registers/thread (acc) | K/V global bytes | bit-exact |
|---|---:|---:|---:|---:|---:|---|
| today | 128 | 2688 | 1 KB + 0.5 KB | ~1 | 352.3 MB | - |
| B G=2 | 128 | 1344 | 2 KB + 1 KB | 4 | 176.2 MB | yes |
| B G=4 | 128 | 672 | 4 KB + 2 KB | 8 | 88.1 MB | yes |
| B G=12 | 128 | 224 | 12 KB + 6 KB | 24 | 29.4 MB | yes (occupancy risk) |
| A | 256 | 224 | 12 KB + 6 KB | 12 | 29.4 MB | no |

### 2.4 Relation to charter section 2 (known negatives)

- "2 rows/subgroup" is a decode-GEMV item (two weight rows per subgroup); this
  is q-heads sharing K/V inside an attention WG. The 2026-09-30 audit already
  lists GQA K/V sharing as unimplemented item 3 and the K audit states it is
  not in the negative list (`qwen38-r1-kernel-2026-10-14.md:201`).
- "SLM input caching in GEMV" failed on 16 lanes x vec8 bank conflicts. A/B do
  not put K/V in SLM; q and probabilities are read as broadcasts (all lanes the
  same address) or contiguous per-lane slices. If a K/V-in-SLM variant (A-SLM)
  is ever tried, the conflict question reopens and tile budget would hit the
  64 KB per-WG cap immediately at D256 (128x256x2 B = 64 KB for K alone).
- "qmap-private" / "uint64 loads" / "blocked layout" / "fused residual": unrelated.
- Route lever (K2: routed vs dense +37% at 8192) stays orthogonal: variants
  apply to the attention kernel that the routed path also calls, but see the
  section 0.5 e2e ceiling.

### 2.5 Expected gains and fail criteria

Kernel-level (len 14336 baseline 0.787 ms/row measured; other lengths scaled by
the byte model and the observed marginal rate) `[estimate]`:

| variant | est ms/row @14336 | kernel delta | est @8192 | est @4096 | e2e dense, if fully exposed | e2e routed 14K |
|---|---:|---:|---:|---:|---:|---:|
| B G=2 | 0.42-0.50 | -36..-47% | ~0.30 | ~0.21 | -5.3..-6.8% | -1.8..-2.3% |
| B G=4 | 0.30-0.38 | -52..-62% | ~0.22 | ~0.16 | -7.6..-9.1% | -2.6..-3.1% |
| A | 0.22-0.34 | -57..-72% | ~0.18 | ~0.13 | -8.3..-10.6% | -2.9..-3.6% |

e2e columns: dense = 12 x (0.787 - new) / 64.4 ms and is an upper bound (it
assumes every saved device millisecond is exposed in the wall; the 64.4 ms wall
itself was measured with the QSA route on). Routed = 12 x 0.27 x kernel delta
fraction / 64.4 ms, i.e. only the attention part of the routed block (~0.27 of
the ~0.35 ms/layer) shrinks and the indexer select (~0.07) does not. The routed
column is the production 14K agent shape and the honest expectation there.

Fail criteria (pre-registered):
1. kernel delta < 3% on either gate shape (4096 or 8192, T1, D256) -> stop.
2. regression > 2% on the short arms (len 333, 1834) or on the routed
   2052/4096 arms -> stop (short-context users are in scope).
3. variant B: any bit difference vs the current kernel on the existing
   correctness arms (`--kv-fp8-only` equivalence + QSA sparse tests) -> fix or
   stop; variant A: max_abs vs f64 > 2e-4, or NaN/Inf, or USM leak / canary
   failure -> stop.
4. B=4 multi-row arm regression: if the variant is not implemented in
   `attention_decode_rows_exact_merge_impl` (a second copy of phase 1,
   `:1402-1660`) and the batch path silently keeps the old kernel, B=4 users see
   0 gain - acceptable only if the multi-row copy is a follow-up with its own
   arm.

---

## 3. Bench plan and thresholds

### 3.1 Arms that exist today (no code change, runnable as soon as the GPU is free)

R1 (gate shapes, T1 D256, 1 row, dense + routed):
`build-refactor\AilaQwen4ExpKernelTests.exe --qsa-probe`
emits `[QSA COST] {...dense_attention_ms...}` for 2048/2051/2052/4096/8192
(reps=15 median, warm 3). Use 4096 and 8192 as the two kernel gate shapes.

R2 (long arm + short arms, B=2, bf16 and quantized):
`build-refactor\AilaQwen4ExpKernelTests.exe --kv-fp8-only`
emits `[KV WALL] len=... arm=bf16 samples=20 median_ms=... p10=... p90=...` for
333/1834/14336 and the two quantized arms, plus `[KV ACC] ... bitwise_eq=...`.

### 3.2 New arm (design only - no file was changed)

Interface: `AilaQwen4ExpKernelTests.exe --decode-attn-reuse-bench [--json]` with
env `AILA_ATTN_KVREUSE=0|pair2|pair4|head12` (0 = current kernel; the others
select variant B G=2/G=4 or variant A). Alternative if the kernel change is env
branched in place: reuse `--qsa-probe`/`--kv-fp8-only` with the env set, but a
dedicated arm is preferred so that (a) shapes include D=128, (b) the A/A control
and the bit-exactness diff run in the same process.

Pseudocode (probe):

```
for shape in [(Hq24,Hkv2,D256,len 4096,B1),
              (Hq24,Hkv2,D256,len 8192,B1),
              (Hq24,Hkv2,D256,len 14336,B1),   # new T1 point
              (Hq24,Hkv2,D128,len 4096,B1),
              (Hq24,Hkv2,D128,len 16384,B1),
              (Hq24,Hkv2,D256,len 333,B1),     # regression guard
              (Hq24,Hkv2,D256,len 1834,B1)]:
    alloc q,k,v,out,scores,partials (stride 272 or 144), canary-fill
    for arm in [0, pair2, pair4, head12, 0]:      # A/A control last
        warm 3, then 15 reps: attention_decode(...); synchronize; collect ms
    print [ATTN REUSE] {"shape":...,"arm":...,"median_ms":...,"p10":...,"p90":...,
                        "max_abs_vs_arm0":...,"bitwise_eq":...}
    assert no USM delta, no canary damage, routed/tail arms still correct
```

The probe must exercise the same code path as production (exact-merge for
len >= 1024 / >= 128, active_blocks variant once, gate buffer on one arm) and
must print which kernel body the dispatcher selected (the existing `[ATTN]`
log line at `AttentionOps.cpp:1887-1903` already does this).

### 3.3 Thresholds and decision rule

1. Kernel gate (R3 or R1-derived): arm passes if median kernel time improves
   >= 3% on BOTH 4096 and 8192 dense T1, with the A/A arm within +-1%; given the
   e2e math above, require >= 15% at 4096 and >= 25% at 8192 before spending
   e2e GPU time.
2. Short-shape guard: no > 2% regression at 333/1834 and on routed 2052/4096.
3. e2e gate (R4): median ms/token over >= 256 decode tokens, same session,
   A/B/B/A with `AILA_ATTN_KVREUSE=0` vs the winning arm; pass at >= 5%.
   Expectation: pass only in the dense long-context profile; the routed 14K
   production arm is expected at -1.5..-3% and a "fail" there is a
   configuration fact, not a kernel failure - report both.
4. All numbers need p10/p90 (the `--kv-fp8-only` arm already prints them; the
   new arm should too) and the same-binary single-variable rule.

### 3.4 e2e arm notes

- The production wall reference is 64.4 ms/token (median, 14K agent shape,
  retained17-era; retained20 re-measure is pending, see the role-D document
  section 1). Any e2e percentage must name the wall it divides by.
- Attention device time is not instrumented per layer; the routed attention
  share used in section 2.5 comes from the QSA probe arms and is a probe-harness
  number. A per-layer `kStageFullAttn` style stage timing (role D's R3) would
  settle it; not required for the kernel gate.

---

## 4. Risks

1. Bit-exactness (B): the reuse keeps the arithmetic per head, but a compiler
   change (reassociation, fast-math, vectorization of the G-head loop) can still
   break bit equality. Mitigation: keep per-head expressions verbatim, run the
   `[KV ACC] bitwise_eq` and QSA sparse arms, treat any diff as a bug.
2. Tolerance (A): score partial-sum tree and tile-sum tree change; bf16 output
   can move by 1-2 ulp. Accept only with the f64 max_abs <= 2e-4 gate used by
   the QSA probe; if the merge factors move, downstream sampling differences are
   bounded but not zero.
3. Occupancy: G=12 (B) and A both drop to ~224 WGs per row per layer; DG2 hides
   memory latency with resident waves, so the byte win can be eaten by poor
   latency hiding. This is the single biggest risk and the reason to measure
   G=2/4 first and to consider tile splitting if G=12 underperforms.
4. L2 window assumptions: the "duplicates are L2-served" reading comes from
   effective rates (678 GB/s marginal), not from a cache counter. If the real
   mechanism is something else (e.g. DRAM sector over-fetch on the 16B vec8 K
   loads), variant gains will be lower than modelled.
5. Scattered active blocks (routed production path): the routed arm remaps each
   virtual token through the block list; variant A's cell resolver and variant
   B's tile addressing must handle active_blocks/active_indices and the
   block+tail split (`AttentionOps.cpp:503-536`). Recommendation: implement the
   dense/contiguous case first and keep the routed path on the current kernel;
   the routed shape has the smaller absolute win anyway.
6. Two copies of phase 1: the single-row and multi-row (`attention_decode_rows`)
   implementations must not diverge (see fail criterion 4); the row-copy
   comments at `:1402-1410` explicitly promise bitwise identity to the
   single-row merge.
7. Quantized KV: int8/fp8 caches multiply the same amplification but are
   opt-in/prototype; the variant must not change their guards (`:1786-1800`
   rejects int8 on the multi-row path).
8. Probe overhead: per-rep synchronize + launches inflate the "fixed" term in
   all existing arms; do not quote probe medians as per-layer production cost.

---

## 5. Evidence index

- `src/ops/AttentionOps.cpp:452-744` (D256 exact merge), `:800-934` (D128),
  `:1140-1236` (baseline), `:1402-1660` (multi-row merge), `:1767-1947`
  (dispatcher), `:1887-1903` (`[ATTN]` path log).
- `src/models/Qwen4ExpPrefill.cpp:150-260` (decode branch + QSA route),
  `:475-540` (row-wise verification path).
- `tmp/Strata/src/kernels/cuda/qsa_decode_attn.cu:120-276` (12 q-head sharing),
  `:526-560` (merge).
- `docs/plans/qwen38-r1-kernel-2026-10-14.md:45-50,81,133-140,199-203` (K audit).
- `docs/plans/qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md:66` (item 3).
- `docs/plans/qwen38-int8kv-vec-2026-10-12.md:11-19` (read-path A/B).
- `docs/plans/qwen38-r1-decode-2026-10-14.md:42` (64.4 ms/token wall).
- `tmp/q4e-sprint-r1-20261014/kernel/qsa-cost-curve-A1.log` (dense/routed curve).
- `tmp/q4e-int8-dpas-20261012/kv-dpas-off-1.log`, `-2.log` (B=2 lengths).
- SKILL `A770-Kernel-Optimization/references/hardware/dram-contiguous-roofline.md:21-25`
  (291.11 / 854.6 / 4115.8 GB/s), `hardware.md:23-24,67` (SLM 128 KB per Xe-core,
  64 KB per WG, >32 KB occupancy caution).
