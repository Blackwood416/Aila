# Qwen3.8-R1 / P9: prefill expert kernels - activation-in-SLM variant (AILA_Q4E_PREFILL_ACT_SLM)

Date: 2026-10-16. Base: `HEAD=0fe5121` (worktree was clean; this patch is uncommitted).
Scope: `src/ops/MoeExpertOps.cpp` only (+344/-16). No `perf/`, no `CMakeLists.txt`, no
`Qwen4ExpExpertExecutor.*`, no `Qwen4ExpMtp*`, no GPU run, no `build.ps1`.

## 0. TL;DR

1. The proposed "L2 first: one work-group handles M activations x N rows with the weight
   row loaded once" tile **already exists in this codebase and has been measured**:
   M=4 (quad) is the default production path, M=8 exists and was refuted, and the
   cross-subgroup weight-line sharing (`wgshare`) was screened and adopted at share=4.
   See section 1 for the receipts; do not re-implement any of them.
2. The one cell of that matrix that was never built is **M-tile x activation reuse**
   (the 16 sub-groups of a work-group all read the same activation rows, but each of
   them reads it from global/L1). That is what this patch adds, gated and default-off:
   `AILA_Q4E_PREFILL_ACT_SLM=1`.
3. **L1 verdict (RUN, measured): NO-GO.** The arm `slm-b1` shows
   `[Q4E PrefillActSlm] enabled=1 in_features=2560 intermediate_size=640 gate_up_quad_tile_bytes=20480`,
   byte-identical SSE, but TTFT 70.1 s vs 69.6/69.6 s and steady blocks 19.24/19.82/20.07 s
   vs 19.05/19.64/19.89 s (~+1%): the 16x intra-work-group activation re-read was already
   served on-chip. The flag stays in the tree, default 0 (section 9).
4. Excluding the activation side, the corrected budget for the expert kernels is
   **weights 25.17 GB/layer = 1.208 TB/block at M=4, i.e. a 97.1 GB/s effective weight
   stream that accounts for the whole 12.44 s `experts_dev`** against a 426 GB/s device
   ceiling - the cache/dependency chain, not DRAM bandwidth, sets the rate (section 10).
   The requested "M activations x 16 rows" tile is the *existing* quad bucket, and every
   M>4 shape is already measured (section 1); section 10 gives the zero-code curve that
   decides whether weight *bytes* are worth attacking at all.
4. Bit-exactness: no arithmetic change at all - same staged bytes, same consumer
   indices, same FMA chains and reduction trees (section 5, plus a CPU index-coverage
   replay `tmp/q4e-sprint-r1-20261014/actslm/coverage_check.py`).

## 1. Priority correction: the M-direction weight-reuse tile is done (receipts)

`src/models/Qwen4ExpBnb4Backend.cpp:1724-1740` selects the section layout per expert:
`AILA_Q4E_WEIGHT_REUSE` (default **true**) enables pair/quad buckets, `AILA_Q4E_4TOKEN_REUSE`
(default **true**) enables the quad bucket, `AILA_Q4E_8TOKEN_REUSE` (default **false**)
enables the eights bucket. Layout is eights -> quads -> pairs -> singles
(`Bnb4Backend.cpp:1741-1780`), and the section dispatchers submit one kernel per bucket
(`MoeExpertOps.cpp:5867-6059` gate/up, `:6063-6256` down).

Measured status of every M-tile shape:

| Tile | Kernel(s) | Default | Bit-exact evidence | Measured verdict |
|---|---|---|---|---|
| M=2 pair | `launch_..._2token` (`MoeExpertOps.cpp:1937`, `:2127`) | on (tail) | no dedicated direct-vs-reference case; covered through the section dispatcher in `tests/models/Qwen4ExpModuleTests.cpp:5166-5198` | used only for `n_e % 4` tails |
| M=4 quad | `launch_..._4token` (`:2162`, `:3669`) | **on** (all acts at chunk 4096) | `Qwen4ExpKernelTests.cpp:1517-1560` vs serial reference | the measured optimum ("window curve minimum at M=4") |
| M=8 octet | `launch_..._8token` (`:4033`, `:4397`) | off | `Qwen4ExpKernelTests.cpp:1772-1812` | **REFUTED**: +22.8..+23.1% block; VTune spill 3,904 B vs 320/576 B; kernel 1,373 us vs 476 us - `docs/plans/qwen38-token8-window-screen-2026-10-10.md` |
| weight line sharing | `..._4token_wgshare<ShareQuads>` (`:2416`, `:3849`) | off in source, `=4` in `tools/server/start-qwen38.ps1:32-33` | `Qwen4ExpKernelTests.cpp:1614-1621` (vs plain 4token) | **ADOPTED**: -3.6..-3.8% block, -2.94% TTFT 14K, bit-exact; share=8 fails (L1 capacity) - `docs/plans/qwen38-wgshare-screen-2026-10-10.md` |
| weight dequant in SLM | `..._wgshare_slm` (`:2918`) | diagnostic only | `Qwen4ExpKernelTests.cpp:4290-4300` | **NO-GO**: -11..-18%; oracle bounds `_nounpack` -54..-68%, `_nolut` -13.6..-33.1% - `docs/plans/qwen38-p4-slmshare-2026-10-13.md` |

Two consequences:

* "Load the weight row once and reuse it over M activations" is *already* implemented and
  measured at M=4/M=8; adding M>4 needs to beat a 3,904 B spill wall with 16 accumulators
  in registers, and the alternative (more sub-groups sharing in time) is the wgshare path,
  which saturates at share=4. Both are closed by evidence.
* The only remaining operand-traffic lever on this kernel is the **activation** side, which
  is exactly what the P4 collateral also warned about: an SLM relay only pays when the
  *source* traffic was the bottleneck, because the consumer load count does not change.

Also note which kernels the sprint arms actually run: `tmp/q4e-sprint-r1-20261014/run.py`
loads `docs/plans/qwen38-server-mtp-2026-10-07.json`, whose Q4E block does **not** set
`AILA_Q4E_4TOKEN_WGSHARE` -> the sprint's production path is the plain M=4 quad kernels.
(The `=4` default lives only in `start-qwen38.ps1`.)

## 2. What this patch adds

New env `AILA_Q4E_PREFILL_ACT_SLM` (default **0**, read once via
`aila::env::read_flag`), plus a new `bool ActSlm` template parameter on four kernels:

| Kernel | file:line (new) | SLM tile | rows staged |
|---|---|---|---|
| `launch_..._prefill_gate_up` (plain, whole-array path) | `MoeExpertOps.cpp:1618` | 5,120 B | 1 activation row (in_features=2560 bf16) |
| `launch_..._prefill_down` (plain) | `:1785` | 1,280 B | 1 h row (in_features=640 bf16) |
| `launch_..._prefill_gate_up_4token` (production) | `:2252` | 20,480 B | 4 activation rows of the quad |
| `launch_..._prefill_down_4token` (production) | `:3800` | 5,120 B | 4 h rows of the quad |

Mechanism (gate/up quad, staging `:2299-2323`): the work-group computes
`wg_base_row = group * 16`; the dispatcher only routes to this variant when
`intermediate_size % 16 == 0` (and `out_features % 16 == 0` for down), so every one of the
16 sub-groups satisfies `act_idx = global_row / intermediate_size = wg_base_row /
intermediate_size`. The group stages those rows with a cooperative `vec8` copy
(`e = lid*8; e += 256*8`), one `item.barrier(local)`, and each sub-group then reads
`x_slm_ptr + k*in_features + byte_offset*2` instead of `x_token{k} + byte_offset*2`
(`:2393` consumer reads). The plain kernels stage the single row of their activation the same way
(`:1665` gate/up, `:1832` down).

Wiring (all env-gated, default path untouched):
`moe_expert_major_prefill_gate_up_4token` (`:5410`), `moe_expert_major_prefill_down_4token`
(`:5455`), the gate/up section dispatcher (`:5914` flags, `:5961` quad, `:6023` single,
`:5888` reuse-off early return) and the down dispatcher (`:6111` flags, `:6158` quad,
`:6220` single, `:6084` early return). No logging lives in the ops TU (see section 7); the
one-shot route proof is emitted by the worker backend, which links the logger:
`src/models/Qwen4ExpBnb4Backend.cpp:1797-1812` prints
`[Q4E PrefillActSlm] enabled=... in_features=... intermediate_size=... gate_up_quad_tile_bytes=...`
once per process from the ALU-section gate, so an arm can prove the flag state and shapes
it ran with.

Not covered by this flag: the pair (`2token`) and octet (`8token`) buckets, and the
wgshare regroupings. At chunk=4096 the quad bucket carries ~98.8% of acts [est: per-expert
`n_e ~= 128`, so `sum(n_e % 4) ~= 256*1.5` of 32,768 acts], so for the 14K sprint arm this
is a ~99% coverage A/B; short prompts (<16 acts/expert) would dilute it, which is why the
A/B below uses the 14K prompt and the real-shape kernel bench.

## 3. Why the activation side is the right experiment (bandwidth account)

Per 4096-token layer, gate/up quad kernel: WGs = 8192 quads x 40 row-blocks = 327,680.
Down quad kernel: WGs = 8192 x 160 = 1,310,720.

| Traffic class | gate/up per layer | down per layer | both, per block (x48) |
|---|---:|---:|---:|
| weights (M=4, one read per quad-row-block) | 13.42 GB | 6.71 GB | 0.97 TB |
| activation rows, charged per sub-group read | 107.4 GB | 107.4 GB | 10.31 TB |
| activation rows, unique per work-group (= ActSlm staging bytes) | 6.71 GB | 6.71 GB | 0.64 TB |

* If all charged activation traffic actually hit DRAM, the measured 12.5 s device time
  implies 825 GB/s - above the 426 GB/s streaming measurement and the ~512 GB/s device
  ceiling, i.e. impossible. So a large part of the 16x intra-work-group re-read is served
  on-chip already.
* If all of it hit L1, the same 12.5 s implies only 129 GB/s of real traffic - far below
  the ceiling, i.e. the kernel would then be bound by something else (the per-nibble
  decode chain and L1/L2 pressure - exactly what `_nolut`/`_nounpack` isolate in P4).
* The coordinator's bench (`AilaQwen4ExpMoeBench.exe --layers 48 --experts 16`: gate 206.6
  us/layer/16 acts = 12.9 us/act vs 4.92 MB @426 GB/s = 11.5 us) matches the streaming
  model, but that is the *decode-shaped* kernel with one activation per expert (no M=4
  weight reuse, no quad bucket). It does not bound the production quad kernel.

So the honest statement is: the production kernel sits somewhere between "L1 absorbs the
16x re-read" and "DRAM-bound"; ActSlm removes the on-chip part *and* divides the DRAM part
by 16. Post-patch DRAM floor if the kernel keeps streaming at 426 GB/s: (0.64+0.97) TB /
426 GB/s ~= 3.8 s/block [est] vs 12.5 s measured; how much of that is realizable depends on
whether the pre-patch bottleneck was memory or the decode chain, which the A/B decides.

This is deliberately *not* the M-tile redesign: no new accumulators, no register growth
(the 8-accumulator quad kernel keeps its 320/576 B spill profile), no weight-side SLM
(the P4 negative), and no change to the reduction structure.

## 4. SLM and occupancy account

* Tile sizes: 20,480 B (gate/up quad) / 5,120 B (down quad) / 5,120 B / 1,280 B (plain),
  all <= the `kQ4eActSlmMaxBytes = 32 KB` guard. A770 per-Xe-core SLM is 128 KB with a
  64 KB per-work-group cap; the skill's sweep shows residency loss starts above ~32 KB/WG
  and a 64 KB tile fails to launch, so 32 KB is the guard and production tiles sit at or
  below 20.5 KB (qmap included).
* Resident work-groups per Xe-core: with 16 HW threads/WG and 128 KB SLM, the pre-patch
  kernel can hold up to 8 WGs (register-limited in practice, see the 320/576 B spill);
  20.5 KB caps it at 6. This is the main risk of the gate/up variant; the down variant
  (5.1 KB) is unaffected. If the A/B shows the gate/up arm regressing while the down arm
  improves, reduce the gate/up tile by staging 2 of the 4 rows (or half-rows) per pass -
  the same code path, just a smaller `acts_per_wg` in the guard.
* Access pattern: the cooperative copy writes consecutive 16 B chunks (lane i -> bytes
  [16i, 16i+16)), and the consumers read 16 B at `16*sg_lane` within each 256 B chunk.
  Both are the conflict-free consecutive form (unlike the AoS pattern recorded as a
  known negative). `AGENTS.md` section 7 lists "SLM input caching in GEMV" as a regression;
  that record is for the decode GEMV N=1 shape, this patch stages activations for a
  multi-row prefill kernel where the same line is consumed 16x per work-group - the
  A/B decides it for this shape, and the result must be recorded either way.

## 5. Bit-exactness argument

1. Staged bytes are copied verbatim from the same global addresses the current path reads:
   `x + token*in_features + e` for gate/up, `h + (token*top_k + rank)*in_features + e` for
   down, `e` a multiple of 8 with `in_features % 8 == 0` (guard), so every 16 B `vec8`
   covers exactly the same bf16 range and no partial element is ever staged or read.
2. Consumer indices are unchanged: gate/up reads `x_slm_ptr + k*in_features + byte_offset*2`,
   the old code read `x_token{k} + byte_offset*2`; down reads the same offsets of the same
   four h rows (`in_expert{k} + byte_offset*2`). `vec8` element order `in_v[i]` and every
   `sycl::fma` chain, scale broadcast, `reduce_over_group` tree and output index are
   bit-identical; only the address space of the load changed.
3. Coverage: `tmp/q4e-sprint-r1-20261014/actslm/coverage_check.py` replays both index
   programs for the production shapes and asserts, per row, that the staged range equals
   the consumer range equals `[0, in_features)` (2560 and 640) and that the tile stays
   <= 32 KB. All four kernels pass.
4. The work-group uniform guard is what makes step 1 legal: `wg_active` and the staged
   `act_idx`/`quad_idx` are functions of `item.get_group(0)` only, and the barrier sits
   outside every conditional, so the barrier is reached by all work-items of the group.
   The dispatcher guarantees `features_per_act % 16 == 0`; if a future shape violates it
   the route falls back to the historical global-read kernel instead of trusting the
   alignment.
5. `if constexpr (ActSlm)` means the default instantiation does not contain any staging
   code (only a 2-byte accessor object), so the off path is the pre-patch kernel plus an
   unused pointer.

Verification commands (no GPU needed for #1/#2, GPU for #3):

```powershell
# 1. offline index coverage (already run, PASS)
python tmp/q4e-sprint-r1-20261014/actslm/coverage_check.py
# 2. syntax check (already run: MOEEXPERT_SYNTAX_EXIT=0)
cmd /c tmp\q4e-sprint-r1-20261014\actslm\syntax-check.cmd
# 3. bit-exact + shape bench: same exe, only the env flips
$env:AILA_Q4E_PREFILL_ACT_SLM='0'; build-refactor\AilaQwen4ExpKernelTests.exe --prefill-bench-only
$env:AILA_Q4E_PREFILL_ACT_SLM='1'; build-refactor\AilaQwen4ExpKernelTests.exe --prefill-bench-only
#    (expect: all [PASS] both arms; the env-on arm must print
#     [Q4E PrefillActSlm] kernel=gate_up_quad in_features=2560 features_per_act=640)
```

## 6. A/B/B/A receipt (production arm)

GPU mutex first: `Get-Process AilaWorker,AilaServer` must be empty, and leave >= 15 s
between arms for the arena/page pool. B arms add exactly one variable:

```powershell
python tmp/q4e-sprint-r1-20261014/run.py p9-a1 --prime --diag --generate=8
python tmp/q4e-sprint-r1-20261014/run.py p9-b1 --prime --diag --generate=8 AILA_Q4E_PREFILL_ACT_SLM=1
python tmp/q4e-sprint-r1-20261014/run.py p9-b2 --prime --diag --generate=8 AILA_Q4E_PREFILL_ACT_SLM=1
python tmp/q4e-sprint-r1-20261014/run.py p9-a2 --prime --diag --generate=8
```

Read from `p9-*-worker.log` / `-run.log`:
`[Q4E PrefillActSlm]` (wiring proof, must appear in B arms only), the per-block wall and the
`experts_dev_ms` component from the device profile arm, plus SSE `content_sha256`.

Acceptance (mission gate): content sha identical across all four arms; A/A spread <= 1%;
B improves either `experts_dev_ms <= 7.0 s` or block wall by >= 20% (i.e. experts_dev
12.5 -> <= 10.0 s) for both B arms, otherwise record the negative.

Pre-registered interpretation (decide with the numbers, not after):

* B ~= A (|delta| <= 1%, sha identical): the 16x re-read was already L1-resident. The
  activation-traffic line closes here; the next lever is the decode chain / L1 pressure
  (P4 `_nolut` .. `_nounpack` span, i.e. weight unpack + LUT gather), or the L2-side
  ordering of the quad stream.
* A - B is 1-20%: partial DRAM/L2 residency; keep the flag as the new baseline candidate,
  then (a) re-run with `AILA_Q4E_4TOKEN_WGSHARE=4` to check composition, (b) extend the
  same staging to the pair bucket, (c) re-measure the spill profile before touching M.
* B slower by >= 3%: SLM residency (6 vs 8 WGs) or the added barrier dominates; fall back
  to a half-tile variant (2 rows staged) before closing the line.

## 7. Risks and rollback

* Build history: v1 of this patch put `AILA_LOG_INFO` inside `MoeExpertOps.cpp`. The
  `AilaQwen4ExpInferRefAdapter` target compiles that TU but does not link `aila::log`, so
  `build.ps1` failed with LNK2019 (unresolved external symbol `aila::log`). Fixed by deleting the
  `profile/Profiling.hpp` include and the one-shot logger from the ops TU and moving the
  route evidence into `Qwen4ExpBnb4Backend.cpp:1797-1812` (the backend already logs).
  `-fsyntax-only` cannot detect this class of failure - the link gate stays with the owner's
  build.
* Register/SLM residency: see section 4; the tile cap is enforced in code, and the flag is
  default-off, so the shipped path is unchanged.
* Silent fallback: any shape failing the guard keeps the old kernel; the
  `[Q4E PrefillActSlm]` line is the only proof of routing and the A/B must show it.
* Correctness: bit-exact by construction; the kernel-test arm above is the gate before any
  production run.
* Rollback: remove the `AILA_Q4E_PREFILL_ACT_SLM=1` env (no code change needed), or revert
  this one-file patch.

## 8. Evidence locations

* Patch: `src/ops/MoeExpertOps.cpp` (uncommitted, +344/-16).
* Syntax check: `tmp/q4e-sprint-r1-20261014/actslm/syntax-check.{cmd,log}` (exit 0).
* Index coverage replay: `tmp/q4e-sprint-r1-20261014/actslm/coverage_check.{py,log}`.
* Prior verdicts reused here: `docs/plans/qwen38-token8-window-screen-2026-10-10.md`,
  `docs/plans/qwen38-wgshare-screen-2026-10-10.md`,
  `docs/plans/qwen38-p4-slmshare-2026-10-13.md`, `docs/plans/qwen38-esimd-path-2026-10-12.md`,
  `tmp/q4e-sprint-r1-20261014/dpe-b1-worker.log`.

## 9. L1 (activation rows in SLM) - measured verdict: NO-GO, keep default 0

Arm: `tmp/q4e-sprint-r1-20261014/slm-b1` (`AILA_Q4E_PREFILL_ACT_SLM=1`, standard 14K arm),
wiring proven by `[Q4E PrefillActSlm] enabled=1 in_features=2560 intermediate_size=640
gate_up_quad_tile_bytes=20480` in the worker log, SSE byte-identical (bit-exactness as
designed).

| metric | A (baseline) | B (slm-b1) | delta |
|---|---:|---:|---:|
| TTFT 14000 | 69.6 / 69.6 s | 70.1 s | +0.5 s (+0.7%) |
| steady block | 19.05 / 19.64 / 19.89 s | 19.24 / 19.82 / 20.07 s | +1% |

Reading: the 16 sub-groups of a work-group read the same activation rows, and those reads
were already served by L1/L2 (the DRAM-impossibility arithmetic in section 10 says the same:
12.88 TB/block of charged activation traffic cannot be DRAM traffic). Staging them in SLM
therefore only adds the cooperative copy, one extra barrier and occupancy pressure (20.5 KB
per work-group). The implementation is kept for reproducibility, default 0; do not re-run
the x-side line - this arm plus the weight model closes it.

## 10. L2 (weight reuse across activations) - disposition and the deciding experiment

### 10.1 The requested tile already exists (and is the default)

The design "one work-group = M activations x 16 output rows, sub-group owns one row, the
weight row is loaded once and reused across the M activations in registers" *is* the
existing quad bucket:

* Layout: `Qwen4ExpBnb4Backend.cpp:1724-1780` buckets each expert's activations into
  eights -> quads -> pairs -> singles; all activations of one expert share that expert's
  slot, so a bucket of 4 consecutive acts is 4 activations of the same expert.
* Kernel: `launch_moe_expert_major_prefill_gate_up_4token` (`MoeExpertOps.cpp:2252`) and
  `..._down_4token` (`:3800`) load `g4`/`u4` once per (row, iteration) and FMA them into 4
  separate token accumulators (`gate_acc0..3`, `up_acc0..3`) - weight bytes divided by 4.
* Gate: `AILA_Q4E_WEIGHT_REUSE` (default **true**) + `AILA_Q4E_4TOKEN_REUSE` (default
  **true**), `Qwen4ExpBnb4Backend.cpp:1724-1726`.

So "M=4 weight reuse" is the production default; the M=1 bound is what produces the
"54-67 GB/layer" figure. The M>4 shapes are closed by measurement:

| shape | mechanism | verdict |
|---|---|---|
| M=8 | same tile, 16 accumulators | **refuted**: +22.8..23.1% block, spill 3,904 B vs 320/576 B, kernel 1,373 vs 476 us (`qwen38-token8-window-screen-2026-10-10.md`) |
| M=8 effective (weight lines shared by 4 sub-groups) | `AILA_Q4E_4TOKEN_WGSHARE=4` | adopted, but only -3.6..-3.8% (share=8 fails: L1 capacity) - sharing *lines* does not reduce the weight load/decode chain |
| weight relay through SLM | P4 `_wgshare_slm` | NO-GO, -11..-18% (`qwen38-p4-slmshare-2026-10-13.md`) |
| XMX/DPAS with M=8..128 | grouped XMX / ESIMD | NO-GO on real shapes (0.65-1.01x) |

### 10.2 Corrected weight/activation budget from the `dpe-b1` receipt

`dpe-b1-worker.log` block 1 (`AILA_Q4E_WORK_SUMMARY=1` + device events):
`hits=365813 misses=1600267` over 48 layers = **40,960 acts/layer** (= 4096 tokens x
top_k=10), `experts_dev_ms=12436`.

| quantity | value |
|---|---:|
| M=4 weight bytes (gate/up 16.78 + down 8.39 GB/layer) | 25.17 GB/layer = 1.208 TB/block |
| effective weight rate = bytes / `experts_dev` | **97.1 GB/s** (device streaming ceiling 426 GB/s) |
| M=1 weight bytes (reuse off) | 100.66 GB/layer = 4.832 TB/block (4.0x M=4) |
| charged activation traffic (M-independent) | 268.4 GB/layer = 12.88 TB/block |
| unique activation bytes per work-group (= ActSlm staging volume) | 16.78 GB/layer = 0.81 TB/block |
| `experts_dev` share of a 19.9 s block | 62% |

Two readings follow, and they agree with the L1 arm:

1. The charged activation traffic cannot be DRAM traffic (it would need 2.8x the measured
   kernel time), so the activation side is on-chip - which is exactly why the L1/SLM arm
   changed nothing.
2. The weight bytes *at the observed 97.1 GB/s effective rate* account for the entire
   `experts_dev` time. The kernel is therefore limited by the weight *load+decode chain*
   (`_nounpack` oracle: -54..-68% when the packed loads/decode are removed; `_nolut`:
   -13..-33% for the LUT gather alone), not by DRAM bandwidth (which would give 1.208 TB /
   426 GB/s = 2.8 s, not 12.44 s).

### 10.3 "If X were free" upper bounds for the expert kernels (charter section 6 form)

| operand | measured/bounded effect | gate check (>= 20% block, or `experts_dev` <= 7 s) |
|---|---|---|
| activation re-reads -> SLM | measured +1% (L1 arm) | fails; line closed |
| weight bytes -> 0 (M=inf, zero spill) | <= 12.44 s of a 19.9 s block -> block -62% ideal; the P4 oracle says ~-54..-68% of the kernel is weight-load+decode bound | passes in principle, but no bit-exact mechanism reaches it (M>4 closed above) |
| weight bytes -> /2 (M=8) | -31% block [est] *if* no spill | at the gate, but refuted as-is by the 3,904 B spill |
| weight bytes -> /4 (M=16) | -47% block [est] | needs 32 accumulators/lane; impossible in 128 GRF |
| fewer bytes per weight (int8/W4A8) | proportional to the byte ratio, same chain | the only remaining byte-reduction route; roadmap B2 exists, adoption gate >= +5% |

### 10.4 Deciding experiment (zero code, all bit-exact)

The three reuse points already exist as gated, bit-exact paths, so the marginal cost of the
weight stream can be measured directly without touching source:

| arm | flag | weight bytes/layer | expected block if the kernel is weight-chain bound |
|---|---|---:|---|
| M=4 | default | 25.2 GB | 19.9 s (measured) |
| M=2 | `AILA_Q4E_4TOKEN_REUSE=0` (pair buckets) | 50.3 GB | ~32 s block (1.6x) |
| M=1 | `AILA_Q4E_WEIGHT_REUSE=0` (plain kernel) | 100.7 GB | ~57 s block (2.9x) |

```powershell
# GPU mutex first; >= 15 s between arms
python tmp/q4e-sprint-r1-20261014/run.py p10-m4 --prime --diag --generate=8
python tmp/q4e-sprint-r1-20261014/run.py p10-m2 --prime --diag --generate=8 AILA_Q4E_4TOKEN_REUSE=0
python tmp/q4e-sprint-r1-20261014/run.py p10-m1 --prime --diag --generate=8 AILA_Q4E_WEIGHT_REUSE=0
```

Pre-registered reading (all arms must stay byte-identical on SSE):

* block(M=1) >= 1.8x block(M=4): weight bytes dominate -> the byte-reduction line is the top
  priority, and the concrete next step is the restructured M=8 tile below (or W4A8).
* block(M=1) <= 1.2x block(M=4): weight *bytes* are not the binding constraint (the chain is
  per-row, not per-byte) -> close the weight-reuse line entirely, including any further
  M-tiling, and spend the GPU budget on the per-(act,row) work instead.
* M=2 locates the curve between the two; report it either way.

### 10.5 Contingent design: the M=8 tile with short live ranges (only if 10.4 says "bytes dominate")

The existing M=8 kernel fails on register pressure because its source loads all 8
activation vectors before the FMA block - 8 x `vec8` (32 GRF) + 16 accumulators + 16 scaled
weight floats + addresses, which IGC keeps live; VTune measured 3,904 B of spill. The
variant that has a chance is the same tile with **per-token consumption**: load `g4`/`u4`
once per iteration, then for each token k load `in_v{k}`, apply the 8 gate and 8 up FMAs,
and release it - the arithmetic order per accumulator is unchanged (each token's own chain
is untouched, the scaled weight values are the same `qmap[nib]*scale`), so the tile stays
bit-exact, while the live activation set drops from 8 vectors to 1. Wiring would reuse the
existing eights bucket (`AILA_Q4E_8TOKEN_REUSE=1` layout) behind a new
`AILA_Q4E_PREFILL_EXPERT_M_TILE=8` flag, with the stock eights kernel as the same-run
control. Not implemented yet: the upstream prior (spill) plus the 10.4 curve should decide
whether it is worth a build + two arms.

### 10.6 Evidence for this section

* `tmp/q4e-sprint-r1-20261014/dpe-b1-worker.log` (acts/layer, `experts_dev_ms`).
* `tmp/q4e-sprint-r1-20261014/slm-b1-worker.log` + run log (L1 arm).
* `tmp/q4e-sprint-r1-20261014/actslm/weight_traffic.{py,log}` (this section's arithmetic).
* Prior verdicts: token8 screen, wgshare screen, P4 slmshare, ESIMD path (section 1 links).
