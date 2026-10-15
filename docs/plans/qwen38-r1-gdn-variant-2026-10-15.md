# R1 Role B5 - GDN prefill SLM-warm-row variant + independent bench (kernel level)

Date: 2026-10-15. Author: role B5 (kernel, GDN prefill variant only). Repo `E:\RiderProjects\Aila`,
HEAD `40239cd`, pair `retained24`. Charter: `docs/plans/qwen38-performance-research-charter-2026-10-14.md`;
the GDN line was opened by K4 (`docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md`, numbers reused verbatim below).

Scope discipline: no build, no GPU run, no commit. Only three new files were written:
`src/ops/GdnPrefillVariant.{hpp,cpp}`, `perf/qwen4exp/q4e_gdn_bench.cpp`, plus two CMakeLists.txt
registrations (source list + new executable). Existing GDN/conv kernels and `Qwen4ExpPrefill.cpp` were
read-only reference. Everything below is either MEASURED (with artifact path, from K4/earlier rounds) or
tagged `[estimate]`.

---

## 1. Current gap and the structural catch that selects this variant

### 1.1 What K4 measured (reused, not re-derived)

| Fact | Value | Source |
|---|---|---|
| GDN prefill kernel, vh=48 n=4096, mode 0 | median 74.8172 ms, 380.54 ns/head/token | K4 M1 (`tmp/q4e-gdn-k1-20261010/sweep-off.log`) |
| Same kernel at vh=96 n=4096 | 119.031 ms (302.71 ns/head/token) | K4 M3 |
| Effective 2R+1W state bandwidth | 516.7 GB/s = 1.78x DRAM read (291.11), 0.60x L2 (854.6) | K4 D4 |
| Thread occupancy | 48 WGs x 128 threads = 6144 = 384/4096 HW threads (9.4%) | K4 section 1.2 (C6) |
| Per token per head | 5 WG barriers + 3 group reductions; SLM = q/k/out locals + head params = 1544 B (2.4% of 64 KiB) | K4 section 1.2 |
| State | fp32 [v_heads,128,128] in global USM, 64 KiB/head, 3.0 MiB/layer | `GatedDeltaNetOps.cpp:514`, K4 |
| Headroom to the SLM-traffic bound | 8.0x (74.82 / 9.39 ms) | K4 D6/D7 |
| e2e saving needed for the 5% TTFT gate | >= 29.3 ms/layer (i.e. kernel <= 45.5 ms/layer) | K4 R3 |

### 1.2 The catch in K4 section 6 direction (i) (new this round)

K4 proposed splitting the 128 v-columns across 2 work-groups per head ("order-preserving, doubles WGs to
96, halves per-WG state to 32 KiB"). That claim does not survive contact with the output normalization:
the per-head output RMS norm reduces `out_local` over **all 128 columns** of the head
(`GatedDeltaNetOps.cpp:730-740`: `out_local[dv] = out * scale`, then
`out_sq = reduce_over_group(group, out_local[lid]^2, plus)`). A column split moves half of that sum into
another work-group, so every token would need a cross-work-group exchange (or a raw-output scratch plus a
second kernel) before the norm can be applied. That is a per-token global coupling, not a free split.

Rows do not have this coupling: row j of column dv is read and written by lane dv only, and rows never
enter a reduction. A row split therefore changes only **load addresses**, never the arithmetic.

### 1.3 Strata reference (same model, different quantization/layout)

- `tmp/Strata/sycl/src/kernels/cuda/native_gdn.dp.cpp:2-3,59-100`: one 32-lane warp per (head, column
  block); each lane keeps **four state rows in registers** (`s_shard[4]`), reducing the per-token state
  traffic from 2R+1W to 1R+1W. That is read-once but uses cross-lane `warp_sum` reductions, i.e. it
  changes the summation order (their comment: compiled with `--use_fast_math`).
- `tmp/Strata/sycl/src/kernels/cuda/gdn.dp.cpp:16-34`: the packing lesson. The thread count of that
  recurrence is fixed at `S * h_v = 6144` no matter how work is packed; their kernel is latency-bound and
  the fix was **more, smaller blocks** (24 blocks of 32 threads -> 192 blocks; "a warp that issues one
  128-byte transaction and then waits ~600 cycles needs other warps to run meanwhile").
- Strata state layout is `(S, h_v, S)` (row-major j, then head, then column); Aila's is `(h_v, S, S)`.
  Their kernels are per-token steps with their own chunked prefill path, so no number is transferred -
  only the two mechanisms (register shard, block packing) are.

---

## 2. Variant design: SLM-warm rows (bit-exact by construction)

`src/ops/GdnPrefillVariant.{hpp,cpp}` adds `run_linear_delta_prefill_gpu_batched_slm(...)`: the
production batched prefill kernel (48 WGs x 128 threads, one WG per value head) with rows
`[0, slm_rows)` of each head's state staged in work-group SLM for the whole window.

### 2.1 SLM budget (A770 reference: 64 KiB per work-group, 128 KiB per Xe-core)

| Item | Bytes |
|---|---:|
| `slm_state` = slm_rows x 128 cols x 4 B, slm_rows = 120 -> 30 four-row groups | 61440 |
| q_local + k_local + out_local (128 fp32 each) | 1536 |
| head_params (beta, decay) | 8 |
| **Total** | **62984 <= 65536 (2552 B headroom)** |

- `kGdnSlmRowsMax = 120`; `slm_rows` is clamped to `[0, 120]` and rounded **down to a multiple of 4**
  (`clamp_slm_rows`): the state walk always visits four rows per iteration, so a partial group would
  straddle the SLM/global boundary. Rows 121-127 cannot be staged: 121 rows already overflow the cap.
- The accessor is runtime-sized (`range<1>(max(slm_rows,1) * 128)`), so the footprint tracks the knob
  (R=0 allocates 512 B). This makes co-residency effects measurable rather than baked in.

### 2.2 Traffic model

- Global per token per head: `3 * 65536 * (128 - R)/128` bytes (cold rows keep the 2R+1W pattern).
  R=120 -> 12,288 B vs 196,608 B baseline = **1/16 of the global stream**; R=56 -> 1/2.7.
- Stage + unstage per kernel call per head: `2 * 65536 * R/128` bytes. R=120: 122,880 B/head ->
  5.9 MB per layer per 4096-token chunk vs 38.65 GB nominal = **0.015% extra** - negligible.

### 2.3 Why this is expected to be bit-exact

1. Every lane's fp32 expression sequence is unchanged: the warm loop and the cold loop concatenate to
   exactly the production four-row sequence `j = 0..127 step 4`, with the same statements.
2. Only the load/store target changes (`slm_state[...]` vs `S[...]`); the values are the same because the
   stage pass copies the rows verbatim and every lane touches only its own column.
3. Barrier/reduction structure is untouched (including the stage barrier and the pre-unstage barrier),
   and the unstage writes exactly the rows the warm loops updated.

The bench still treats parity as a guard: bitwise first, and a tolerance fallback
(`state_nrmse <= 1e-3` and `out_max_abs <= 2e-2`) because a compiler is free to associate differently
across the two loop shapes. If the fallback fires, the receipt says `"parity":"tolerance"` and the
adoption discipline of charter 5.9 (bit-level regression review) applies.

### 2.4 What the variant deliberately does NOT change

- No change to WG count, lane mapping, barriers, reductions or instruction structure of the walk: the
  mechanism under test is purely "state loads/stores now hit SLM instead of global USM".
- No register-shard read-once transform (Strata native_gdn direction): it changes the reduction order
  and the register budget, and K4's section 1.4 already recorded the failed software-pipelining
  experiment (M8: +84%/+92%). That door stays closed this round.
- No chunkwise change: `GdnChunkOps` modes 1/2 already exist and measured -2.7% ttft e2e (K4 M7).

---

## 3. Bench: `AilaQwen4ExpGdnBench` (independent, kernel-level)

### 3.1 Command shape

```powershell
# GPU exclusivity (charter 5.4); must print nothing
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue
Get-ChildItem Env: | Where-Object { $_.Name -like 'AILA_*' -or $_.Name -like 'SYCL_*' -or $_.Name -like 'UR_L0_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
build-refactor/AilaQwen4ExpGdnBench.exe *> tmp/q4e-sprint-r1-20261014/gdn/bench-b5-A1.log
```

Options: `--vh 48` or `48,96` (comma list), `--rows 4096,8192`, `--arms baseline,slm`,
`--r 120` or `0,56,96,120`, `--samples 15`, `--warm 2`. Defaults: vh=48, rows=4096,8192,
arms=baseline+slm, r=120, samples=15, warm=2 (one process, no model file; ~1-2 min on an idle A770).

### 3.2 Guards (fail closed, before any timing)

1. CPU canary: f64 recurrence on a 64-token window, baseline and SLM arm must each match the
   `GdnChunkTests` canary (state/output NRMSE < 1.5e-2); `[GDN GUARD]` lines.
2. Parity per cell: baseline vs SLM snapshot of state (fp32), conv history (fp32) and output (bf16);
   `[GDN PARITY]` line with first-diff indices; bitwise expected, tolerance fallback allowed as in 2.3.
3. Output sentinel tail (out tensor has +16 elements) must stay -9; conv history must equal the CPU
   reference (convolution taps are not part of the variant).

### 3.3 Timing protocol and output

One warm-up per (cell, arm) plus N round-robin samples across cells; reset (state + conv H2D) happens
before the timed `ext_oneapi_submit_barrier`, so reset is excluded. Timing = wall between the two
barriers; device span from profiling events (K4 section 4.3 convention). Per (arm, vh, rows):

```json
[GDN BENCH] {"schema":"aila.gdn-slm-bench.v1","rows":4096,"median_ms":74.812,"gbps":516.7,
             "arm":"slm_r120","variant":1,"vh":48,"r_slm":120,"p10_ms":...,"p90_ms":...,
             "device_span_ms":...,"wall_over_span":1.00,"samples":15,
             "nominal_bytes":38654705664,"dram_gbps":...,"parity":"bitwise"}
```

`gbps` = nominal 2R+1W stream rate (comparable with K4 D4; the baseline arm at vh=48/rows=4096 must
reproduce ~516.7 GB/s at ~74.8 ms - `[GDN ANCHOR]` prints the +/-3% re-anchor check against M1).
`dram_gbps` uses the model in 2.2 (global bytes only), so the two rows of the same cell separate
"faster because fewer global bytes" from "faster because SLM is faster per byte".

### 3.4 Gates (frozen from K4 R3, applied by the bench)

| Gate | Rule | Bench output |
|---|---|---|
| R0 sanity | A/A spread <= 3% at (vh=48, rows=4096) | run the exe twice, compare medians (`[GDN BENCH]`) |
| Anchor | baseline within +/-3% of 74.8172 ms | `[GDN ANCHOR]` (re-anchor if outside, K4 section 4.2) |
| R3 kernel | `kernel_saving_pct >= 3` at both 4096 and 8192 | `[GDN GATE]`, `[GDN TWO-SHAPE]` |
| R3 e2e | `e2e_projected_pct >= 5` (36 layers, 3.4045 chunk-equivalents, 71.82 s TTFT) | `[GDN GATE]` rows=4096, `[GDN TWO-SHAPE] pass` |
| R2 (context) | a variant needing <= 45.5 ms/layer to clear the e2e gate | compare `median_ms` against the K4 R3 arithmetic (29.3 ms/layer) |

P2 shape (`--vh 96 --rows 4096`) is required for the two-shape rule; the default run only covers P1.

### 3.5 What each arm answers

| Arm | Question it answers |
|---|---|
| `slm_r0` | SLM allocation/co-residency cost alone (all loads still global) - if r0 ~= baseline, the footprint is free; if slower, the difference is the occupancy price of the cap |
| `slm_r56` | half-fraction sweep point (28 KiB/head staged) |
| `slm_r120` | the only arm that can clear the e2e gate (global traffic 1/16) |
| baseline | anchor + A/A + parity reference |

Expected shape `[estimate]`: monotone decreasing with R; R=0 within noise of baseline; R=120 bounded
below by the K4 D8 issue floor (~1.8-2.0 ms/layer) and the barrier structure, so 20-50 ms/layer is the
plausible band, i.e. it may or may not clear 45.5 ms - that is exactly what the gate decides.

---

## 4. Risks and non-claims

1. **SLM footprint vs co-residency** `[estimate]`: 62984 B/WG means at most 2 WGs per Xe-core under the
   128 KiB reference (driver reserves may reduce that to 1). 48 WGs over 32 Xe-cores need only ~1.5
   slots/core, but if the cap drops co-residency below the baseline's, the r0 arm exposes it. This is
   the single most likely way for the variant to lose - measure r0 first.
2. **Compiler re-association**: the bit-exactness argument is source-level; a parity FAIL is a real
   result (recorded, not hidden). The tolerance fallback keeps the bench useful but changes the
   adoption question (charter 5.9 bit-level review, validator arm).
3. **Rows=8192 cell memory**: fixture VRAM ~570 MB for vh=48 (in 270 MB + out 100 MB + state/weights);
   benches must not run next to a live worker on a 16 GB card.
4. **Variant is not wired to production**: `run_linear_delta_prefill_gpu_batched_slm` is opt-in and
   currently called only by the bench. Adoption needs (a) two-shape kernel >= 3%, (b) projected e2e
   >= 5% confirmed on the 14K fixture, which requires a separate `src/` switch (not this round).
5. **Non-claims**: no DRAM-vs-L2 attribution (needs VTune counters); the traffic model is a byte model;
   no claim about decode (the same state tensor feeds decode kernels, whose cost is ~1% of the token
   budget, K4 section 1.3); Strata numbers are not comparable (different quantization and layout).

## 5. Coordinator commands (after build)

```powershell
# P1 shape, A/A + sweep: run twice, compare medians (R0 gate)
build-refactor/AilaQwen4ExpGdnBench.exe --vh 48 --rows 4096,8192 --arms baseline,slm --r 0,56,120 *> tmp/q4e-sprint-r1-20261014/gdn/b5-p1-A1.log
build-refactor/AilaQwen4ExpGdnBench.exe --vh 48 --rows 4096,8192 --arms baseline,slm --r 0,56,120 *> tmp/q4e-sprint-r1-20261014/gdn/b5-p1-A2.log
# P2 shape (two-shape gate)
build-refactor/AilaQwen4ExpGdnBench.exe --vh 96 --rows 4096 --arms baseline,slm --r 120 *> tmp/q4e-sprint-r1-20261014/gdn/b5-p2-A1.log
```

Read order: `[GDN DEVICE]` -> `[GDN GUARD]` -> `[GDN PARITY]` -> `[GDN ANCHOR]` -> `[GDN BENCH]` ->
`[GDN GATE]`/`[GDN TWO-SHAPE]`. A guard failure exits non-zero with `[GDN FAIL]` and no timing is
reported (fail closed).

## 6. Sources

- Code (HEAD `40239cd`): `src/ops/GdnPrefillVariant.hpp`, `src/ops/GdnPrefillVariant.cpp`,
  `perf/qwen4exp/q4e_gdn_bench.cpp`, `CMakeLists.txt` (source list + `AilaQwen4ExpGdnBench`).
- Reference (read-only): `src/ops/GatedDeltaNetOps.cpp` (6-30, 45-63, 116-119, 477-490, 484-487, 514, 518, 531-575,
  646-663, 711-731, 730-740, 741-753, 772); `src/ops/GatedDeltaNetOps.hpp` (DeltaNetGeometry/Weights/
  State); `tests/ops/GdnChunkTests.cpp` (fixture, CPU canary, timing convention); `src/core/Context.hpp`
  (`Context(trace, enableProfiling)`); `src/core/Tensor.hpp`.
- K4: `docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md` (M1/M3/M6/M7/M8, D1-D10, sections 1.2/1.4/3c/4.2-4.4).
- Strata: `tmp/Strata/sycl/src/kernels/cuda/native_gdn.dp.cpp` (2-3, 59-100, 137-143);
  `tmp/Strata/sycl/src/kernels/cuda/gdn.dp.cpp` (16-34, 43-45);
  `tmp/Strata/sycl/src/kernels/cuda/fused_gdn.dp.cpp` (16-60).
- Charter: `docs/plans/qwen38-performance-research-charter-2026-10-14.md` (5.3/5.4/5.9).
- A770 reference constants: K4 C6 (64 KiB/WG SLM cap, 128 KiB/Xe-core).

## 7. Coordinator verification (2026-10-15 ~19:57-20:04, window B)

Two bench runs uild-refactor/AilaQwen4ExpGdnBench.exe --vh 48 --rows 4096,8192 --arms baseline,slm --r 0,56,120
(A1 and A2, GPU exclusive). Anchor check: 74.588/74.630 ms vs 74.817 (-0.31%/-0.25%, PASS).
A/A max cell delta 0.13% -> medians trustworthy.

| shape | baseline | slm_r0 | slm_r56 | slm_r120 |
|-------|----------|--------|---------|----------|
| vh48 rows4096 | 74.59/74.63 | 71.16/71.19 (-4.59%) | 120.08/120.16 (-61.0%) | 138.74/138.84 (-86.0%) |
| vh48 rows8192 | 149.12/149.14 | 142.21/142.32 (-4.64%) | 239.29/239.32 (-60.5%) | 276.95/276.99 (-85.7%) |

- Parity: bitwise=1 (state/conv/out first_diff=-1) on **all** arms and shapes.
- SLM-resident-row hypothesis is REFUTED by measurement: residency itself is the
  regression (r0 isolation arm is faster than baseline, so staging/occupancy cost
  scales with r_slm).
- Side finding: the variant kernel at r=0 is bit-exact AND +4.6% at both shapes
  (reproduced), but the bench's own e2e projection is 0.585% << 5% gate.
- Verdict: GDN prefill line CLOSED for adoption; keep bench+variant as opt-in
  evidence, no production wiring. P2 (vh=96 r=120) skipped: the two-shape question
  is already decided and r120 is uniformly negative. Receipt:
  	mp/q4e-sprint-r1-20261014/receipt-gdn-slm.json.
