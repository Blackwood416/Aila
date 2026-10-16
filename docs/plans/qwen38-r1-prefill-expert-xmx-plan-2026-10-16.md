# R1 prefill expert phase: dequantize-once bf16 + oneDNN batched GEMM (XMX)

Date: 2026-10-16. Repo `E:\RiderProjects\Aila`, base HEAD `bb07130`.
Anchors are taken against the current working tree, which carries one
uncommitted fix unrelated to this plan: the `sycl::event{}` fence selection in
`src/ops/MoeExpertOps.cpp` (`single_submitted` etc., +40/-8, closing audit
section 5) and the decode `AILA_Q4E_JOIN_RELEASE` hook in
`src/models/Qwen4ExpBnb4Backend.cpp:1255-1262` (+15).

Scope: design only. No build, no GPU run, no source edit, no commit. Labels:
`[measured]` = number exists in a named log/doc; `[est]` = derived here.

Inputs: the audit `docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md`; the
probe `tmp/q4e-bottleneck-audit-20261016/{expert_shape_probe2.cpp,expert2-L*.log}`;
the block ledger `tmp/q4e-sprint-r1-20261014/dpe-b1-worker.log:19`; the routing
histogram `tmp/q4e-strata-c4/heldout-14k.profile`; the sibling quality recipe
`docs/plans/qwen38-r1-expert-xmx-quality-gate-recipe-2026-10-16.md`.

---

## 0. Decision

1. **Adopt the probe's measured design point verbatim**: per (expert, 64-weight
   block) one NF4->bf16 vectorized dequant pass, batched oneDNN bf16 GEMM for
   gate_up, an elementwise SwiGLU, a second batched GEMM for down, then a
   scatter into the existing rank layout. Probe at 40 960 acts/layer (one 4096
   block / 48 layers), layers 0/3/10/20/30/40/47: compute 3.39-4.01x faster,
   with the same 2-bank H2D chain 1.89-1.96x faster `[measured,
   tmp/q4e-bottleneck-audit-20261016/expert2-L*.log]`.
2. **The supply path is not touched.** NF4 arrives through the same resident
   pool / layer arena / staging banks and the same H2D events; only the
   consumer of those bytes changes. Block estimate 19.9 s -> ~13.5 s, TTFT
   66-71 s -> ~48 s `[est, audit 2.4]`.
3. **The old grouped-XMX verdict is not evidence against this**: it was an
   M8/N32/K32 joint_matrix prototype with 3 barriers per K step and a separate
   unpack launch (`src/ops/MoeGroupedXmx.cpp:28-126`), never priced at the real
   M distribution. Dense prefill already ships "NF4->bf16 dequant + oneDNN"
   (`src/ops/Bnb4BitLinear.cpp:2443-2491`, `:2518-2527`); the expert GEMMs were
   the only large family excluded from it.
4. **Chunking rule**: usage-descending greedy packing with `B <= 16` experts and
   `B * M <= 2048` padded rows, `M = round_up_16(max rows in chunk)`. On the real
   histogram this pads 1.16-1.20x (better than the probe's fixed-16 rule
   1.23-1.79x `[est]`) and bounds the activation scratch at 75 MiB.
5. **New device bytes 375 MiB** (300 MiB bf16 weight banks + 75 MiB activations),
   allocated only when the flag is on; it lands inside the existing 3584 MB
   reserve (Backend.cpp:737-755).
6. **Gated off by default** via `AILA_Q4E_PREFILL_EXPERT_XMX=0`; this path is
   not bit-exact, so adoption needs the section-4 teacher-forced KL/top-1 gate,
   not a hash diff.

---

## 1. Production call chain to modify (file:line anchors)

| Step | Anchor | What it does today |
|---|---|---|
| Layer loop / MoE branch | `src/models/Qwen4ExpBnb4Backend.cpp:2052` (`ForwardImpl`), decision at `:2808-2820` | `count > 4` -> `RunAdaptivePrefill`; else resident decode |
| Prefill expert driver | `:1600-1711` (`RunAdaptivePrefill`) | usage histogram `:1606-1610`; builds `prefill_unique_` (residents `:1612-1618`, arena epoch `:1626-1637`, banks `:1639-1643`); min-rows gate `:1644-1646`; three `RunGroupedExperts` phases `:1649-1704`; rank merge `:1706-1710` |
| Per-microbatch ops dispatcher | `:1713-1838` (`RunGroupedExperts`) | quad/pair/single bucketing `:1726-1744`, host token/rank/slot maps `:1745-1792`, ALU dispatch `:1796-1819`, old XMX tiles `:1820-1830` |
| Gate_up dispatch | `src/ops/MoeExpertOps.cpp:5867` (`moe_expert_major_prefill_gate_up`) | eights/quads/pairs/singles segments, one kernel each |
| Down dispatch | `:6074` (`dispatch_moe_expert_major_prefill_down`), public wrapper `:6282` | same segmentation, `Compact` variant |
| Old grouped-XMX tiles (kept, still default-off) | `src/ops/MoeGroupedXmx.cpp:24-141`; call sites `Qwen4ExpBnb4Backend.cpp:1821-1826`; flag `AILA_Q4E_GROUPED_XMX_MIN_ROWS` (`:414-421`) | M8/N32/K32 prototype; refuted, do not extend |
| Staging slots (NF4 H2D banks) | `src/models/Qwen4ExpAdaptiveCache.hpp:27-45` (16 slots/bank, 2 banks by default), `:189-194` (`GetStagingSlotDevicePtr`), `:201-209` (`GetResidentSlotDev`/`GetStagingSlotDev`), `:216-221` (fences), `:340-347` (storage); `.cpp:146-166` (alloc), `:605-659` (`StageColdExpertsAsync`, H2D task + reader fence `:651-654`) | 16 experts x 2.417 MiB = 38.7 MiB device + host pinned per bank |
| Layer arena (whole-layer cold set) | `src/models/Qwen4ExpLayerStage.hpp:103/118/125/138-139` | optional (`AILA_Q4E_LAYER_STAGE`), same slot descriptor shape |
| VRAM admission | `Qwen4ExpBnb4Backend.cpp:709-772` (reserve math `:737-755`), slot cap `:793-805`, runtime buffers `:1030-1048`, `:1146`, `:1165-1190` | reserve 3584 MB; `moe_down_acc` (f32 `<count*top_k, 2560>`, 400 MiB at 4096) already lives in phase scratch |
| Dense reference implementation of the same numerics | `src/ops/Bnb4BitLinear.cpp:2443-2491` (fused prefill XMX), `:2105-2139` (`dequantize_weight`), `:2973-3091` (`Bnb4GroupedMoE` scratch/`run_matmul`/`dequant_subset` incl. nested scale `:3074-3082`) | proves the numerics class and the prim-cache pattern |
| Nested NF4 scale decode to reuse verbatim | `src/ops/MoeGroupedXmx.cpp:13-22` (`scale<Nested>`), LUT `src/ops/MoeExpertOps.cpp:8-13` | `qmap[code]*absmax + offset` |
| SwiGLU epilogue to reuse | `src/ops/MoeExpertOps.cpp:5303-5330` (`moe_silu_gate_up_mul_flat`) | `silu(g)*u`, works on any row count; padding rows -> 0 |

New entry point: `RunGroupedExpertsXmx(...)`, called from `RunAdaptivePrefill`
only when `AILA_Q4E_PREFILL_EXPERT_XMX=1`, `grouped_xmx_supported_`
(`Qwen4ExpBnb4Backend.cpp:414-421`, A770) and the 2560/640/64 shape hold;
otherwise every call site is byte-identical to today.

---

## 2. Chosen design

### 2.1 Chunk plan (host, per layer)

1. Sort the used experts of `prefill_unique_` by descending
   `prefill_expert_uses_`, ties by expert id.
2. Greedily fill a chunk while `B <= 16` and `(B+1) * max(M_cur, n_next) <=
   AILA_Q4E_PREFILL_EXPERT_XMX_ROWS` (default 2048); else start a new chunk.
   A single expert with `n_e > 2048` gets its own chunk (B=1, M = round_up_16
   of its rows, <= 4096).
3. Per chunk record B, `M = round_up_16(max rows)`, per-member rows/base, and
   each source class (resident / arena epoch / bank + its ready event).
   Rank/token maps are per-expert contiguous (no quad/pair/single bucketing);
   `down_acc` rows are `(token * top_k + rank)`, written once each, so chunk
   order cannot change the rank-ordered merge `:1706-1710`.
4. Bank staging keeps `AILA_Q4E_STAGING_SLOTS=16` per bank; the bank pump
   (`:1670-1704`) stages the same bytes in the new sorted chunk order, so chunk
   == H2D batch and `batch.ready_event` gates exactly that chunk's dequant.

### 2.2 Per-chunk device pipeline (all on `ctx.queue()`, in-order)

| # | Step | Kernel/op | Notes |
|---|---|---|---|
| 1 | dequant NF4->bf16 | new, `1 work-item = 1 uint32 (8 weights) + vec8 bf16 store`; nested scale from `MoeNestedScaleDev` | write `wgu[B][2I][H]` and `wdn[B][H][I]` bf16 in the packed [N,K] linear order (no transpose; oneDNN takes the strided view, as the probe did); 0.354-0.378 us/act `[measured probe arm D]` |
| 2 | gather | new, `a1[B][M][H]`, zeros for padded rows | reads `x` rows via token map |
| 3 | gate_up GEMM | oneDNN batched bf16 matmul `[B,M,H] x [B,H,2I] -> c1[B,M,2I]` bf16, f32 accumulation | primitive cached by `(B,M)`, same pattern as `Bnb4BitLinear.cpp:2996-3025`; keep `dnnl::memory` handles alive until the queue drains (comment `Bnb4BitLinear.hpp:351-355`) |
| 4 | SwiGLU | reuse `moe_silu_gate_up_mul_flat` (`MoeExpertOps.cpp:5303`) on `B*M` rows | padded rows are zero |
| 5 | down GEMM | oneDNN batched bf16 matmul `[B,M,I] x [B,I,H] -> c2[B,M,H]` f32 | f32 dst matches `moe_down_acc` and the probe |
| 6 | scatter | new, row `b*M+m -> down_acc[(token*top_k+rank)*H]` | skips padding; f32->f32 |

Then the unchanged `moe_ordered_accumulate_down_rows` (`:1706-1710`) merges
ranks and applies router weights. Returns the last scatter event; that event is
what `RecordStagingConsumer`/`RecordResidentConsumer` must see, so the existing
fences stay valid.

### 2.3 M-tile justification (real histogram)

4096-token block = 40 960 routed acts/layer; `dpe-b1` logged 1 600 267 misses /
48 layers = 33 339 routed-miss acts/layer and 19 360 staged experts `[measured,
dpe-b1-worker.log:19]`. Per-expert usage, computed from
`tmp/q4e-strata-c4/heldout-14k.profile` (same file the probe consumes):

| Layer | used experts | avg acts/expert | max | experts >= 1400 | probe fixed-16 padding | plan (R=2048) padding | chunks | max B*M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 502 | 81.6 | 586 | 0 | 1.227 | 1.202 | 23 | 2048 |
| 20 | 506 | 80.9 | 1368 | 0 | 1.479 | 1.157 | 25 | 2048 |
| 40 | 499 | 82.1 | 1702 | 3 | 1.514 | 1.186 | 25 | 2048 |
| 47 | 485 | 84.5 | 2437 | 3 | 1.789 | 1.193 | 24 | 2432 * |

`[est, computed]`; `*` = single-expert chunk (B=1, allowed by 2.1 rule 2). The
probe reported used_experts 495/503/489/463, max_m 592/1376/1712/2448,
pad_ratio 1.225/1.478/1.510/1.787 for the same layers
`[measured, PROBE2 CONFIG lines]`. Conclusions:

* ~84 acts/expert average means most experts are a single band; padding at
  R=2048 is 1.16-1.20x, inside the probe's measured 1.2-1.8x band, and the
  chunk count (23-25/layer) is close to the probe's fixed 31-32.
* Hot experts (1400-2437 rows, 3 per layer at L40/L47) form B=1..2 chunks; that
  keeps `B*M` bounded without splitting any expert, so each expert's bf16
  weights are still read exactly once by the GEMMs. A split band would cost one
  extra 9.8 MB read (~0.1% of the layer's weight traffic) - acceptable fallback.
* Weight traffic per layer (dequant read 2.53 MB + write 9.83 MB + GEMM read
  9.83 MB per expert) ~= 11.1 GB `[est]`; the probe's X arm ran 19.3-21.5 ms
  for it, i.e. already at the A770 DRAM roofline. Further gains must come from
  hiding, not from tiling.

### 2.4 Probe numbers to reproduce (ms/layer, 40 960 acts)

| Layer | ALU K | H2D H | ALU+H2D KH | XMX DX | XMX+H2D DXH | K/DX | KH/DXH |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 159.555 | 126.552 | 289.131 | 39.696 | 152.622 | 4.014 | 1.886 |
| 20 | 138.902 | 128.656 | 270.037 | 38.664 | 141.208 | 3.593 | 1.912 |
| 40 (wgshare=4) | 148.182 | 124.978 | 305.238 | 41.827 | 155.819 | 3.543 | 1.959 |
| 47 | 137.731 | 118.372 | 282.374 | 40.692 | 148.566 | 3.385 | 1.901 |

`[measured, expert2-L{0,20,40,47}*.log; the audit's table adds L3/L10/L30]`.
Arm definitions: K = production ALU dispatchers; H = the production 2-bank H2D
chain; DX = dequant + batched oneDNN + SwiGLU + scatter (no H2D); DXH = the
same with the H2D chain. Additivity K+H vs KH: overlap -0.02..-0.26; DXH-H =
12-31 ms means 26-68% of the XMX compute hides under H2D.

---

## 3. Memory / VRAM budget

One NF4 slot is 3 x 2560 x 640 x 33/64 = **2 534 400 B (2.417 MiB)**; its bf16
expansion is (2IH + HI) x 2 = **9 830 400 B (9.375 MiB) = 3.88x**. At chunk 4096
the new all-flag-on device bytes are:

| Buffer | Bytes | Source |
|---|---:|---|
| bf16 weight banks, 2 x 16 experts x 9.375 MiB | 314 572 800 (300.0 MiB) | **new** (device); 2 banks so chunk k+1 dequant can overlap chunk k GEMMs |
| A1 gather `[4096, 2560]` bf16 | 20 971 520 (20.0 MiB) | **new** |
| C1 gate_up `[4096, 1280]` bf16 | 10 485 760 (10.0 MiB) | **new** |
| A2 SwiGLU `[4096, 640]` bf16 | 5 242 880 (5.0 MiB) | **new** |
| C2 down `[4096, 2560]` f32 | 41 943 040 (40.0 MiB) | **new** |
| row bases/counts per chunk (host pinned + 64 B dev) | < 1 KiB | **new** (trivial) |
| `moe_down_acc` `[40960, 2560]` f32 = 400 MiB | 0 | **existing** phase-scratch peak (`Backend.cpp:1165-1166`) |
| oneDNN scratchpad + prim handles | few MiB `[est]` | new, oneDNN-owned |
| **Total new** | **393 216 000 B = 375.0 MiB** | flag-on only; flag-off allocates nothing |

Notes:

* Buffers sized "4096 rows" cover any single band; the 2048 budget bounds
  multi-expert chunks. Above chunk 4096 the sizing follows `cap * top_k`.
* Experts in flight: 16 NF4 sources per bank (existing layout) and 16 bf16
  weight slots per bf16 bank (new); the bf16 bank is written by the dequant
  kernel and read by the GEMMs of that same chunk, gated by the bank's
  ping-pong event, mirroring `StageColdExpertsAsync`'s reader fence.
* The 3584 MB reserve (`AILA_Q4E_VRAM_RESERVE_MB`, `Backend.cpp:737-755`;
  the R1 arms pin it in `run.py:36`) carries the runtime buffers; +375 MiB is
  ~10.5% of it, leaving ~3.2 GB free against the documented ~1-2 GB WDDM
  paging threshold (`Backend.cpp:710-712`). No change to slot count (2800) or
  to the staging banks (2 x 38.7 MiB). No new pinned host bytes.
* Known negative to respect: chunk-8192 died from cap-scaling buffers (audit
  4.4); here only the activation row capacity follows `cap`, capped at 4096.

---

## 4. Correctness / quality gate (not bit-exact)

Existing tooling found by search (no new tool needed for the decision):

| Tool | What it yields | Where |
|---|---|---|
| `AilaQwen4ExpToleranceTests --grouped-kernel` | XMX vs independent double-precision NF4 oracle, NRMSE < 0.001, tail/guard checks | `tests/models/Qwen4ExpGroupedXmxAcceptance.hpp:13-124`; target `CMakeLists.txt:854` |
| X3 OPQUALITY in-model probe | ALU vs candidate on live expert slices: `h_nrmse`, `down_nrmse`, `p99_abs` | same file `:145-247` |
| `--decode-model` teacher-forced probe + `q4e_decode_summary.py` | per-row KL (nats), max NRMSE, top-1 agreement, 128 rows/prompt | `tests/models/Qwen4ExpDecodeProbe.hpp`, entry `Qwen4ExpToleranceTests.cpp:711-713`; `perf/qwen4exp/q4e_decode_summary.py:63-95` |
| Tolerance contract | nrmse <= 0.03, kl <= 0.01, top-1 diagnostic | `perf/qwen4exp/q4e_mtp_reference.py:169,185` |
| Top-1 gate precedent | 0.95 bar, 61/63 = 0.9683 PASS | `docs/plans/qwen38-r1-mtp-top1-2026-10-15.md` |
| Teacher-forced `CheckpointMetrics` (`[X2 QUALITY]`) | mean_kl / mean_nll_delta / max_logits_nrmse | `tests/models/Qwen4ExpCheckpointAcceptance.hpp:22-51` |
| e2e runner | TTFT (first_content), steady AdaptiveWork blocks, SSE content sha | `tmp/q4e-sprint-r1-20261014/run.py`, `scan_ttft.py` |

Smallest credible gate (order matters; full recipe in the sibling doc):

1. **Kernel pre-check (CPU-referenced, <1 min GPU)**: `--grouped-kernel` must
   pass unchanged, then the X3 OPQUALITY probe on the flag-on path: accept
   `h_nrmse <= 0.01` and `down_nrmse <= 0.01` (existing warning-line threshold),
   record `p99_abs`/`max_abs`.
2. **Deciding model gate**: `--decode-model` flag-off vs flag-on at prompt
   lengths 512 and 30 (each = 128 teacher-forced rows), compared by
   `q4e_decode_summary.py`: **mean KL <= 0.01 nats, max NRMSE <= 0.03, top-1
   >= 122/128 (95.3%)** per prompt, pooled 256 rows. `exact_greedy_parity` is
   reported, not gated: this change is explicitly non-bit-exact, so differing
   `step_hashes` are expected. Positive control: `[DECODE PASS]` cache
   counters must be nonzero.
3. **e2e divergence report**: the section-5 A/B/B/A arms record SSE content sha
   per arm; equality is neither required nor sufficient here (unlike pair2).
   A sha change without passing 2 is a NO-GO.

Flag: `AILA_Q4E_PREFILL_EXPERT_XMX`, default `0`, read once
(`aila::env::read_flag`), required to be `1` for any code path above. Gate
conditions: `grouped_xmx_supported_` (A770) + hidden 2560 / I 640 / bs 64; any
other shape logs once and stays on the ALU path.

Rollback: unset the flag (no persisted state, no checkpoint or slot-layout
change). Flag-off stays byte-identical to today's ALU dispatchers by
construction - the flag only selects `RunGroupedExpertsXmx` vs the three calls.

---

## 5. Measurement plan (evidence-first)

### 5.1 (a) Kernel-level arm

No project bench exists yet for this pipeline; two acceptable arms, in order:

```powershell
# A. zero-build: the audit probe at production shapes (7 layers x 40 960 acts)
./tmp/q4e-bottleneck-audit-20261016/build_probe.ps1 -Source tmp/q4e-bottleneck-audit-20261016/expert_shape_probe2.cpp
./tmp/q4e-bottleneck-audit-20261016/run_probe.ps1 -Exe ./tmp/q4e-bottleneck-audit-20261016/expert_shape_probe2.exe `
    -Log tmp/q4e-bottleneck-audit-20261016/expert2-L20-plan.log -ProbeArgs @('--reps','5','--layer','20')
# B. integrated: a --prefill-expert-xmx-bench arm on the plan's chunking,
#    median + CV per arm as X3 OPTIME does (perf/qwen4exp style)
```

Gates: DX <= 40 ms/layer at L20 (probe 38.7), K/DX >= 3.4, KH/DXH >= 1.85 at
L0/L20/L40/L47; additivity K+H vs KH reported.

### 5.2 (b) e2e A/B/B/A on the standard 14K arm

```powershell
python tmp/q4e-sprint-r1-20261014/run.py px-a1 --prime --diag --generate=32 AILA_Q4E_PREFILL_EXPERT_XMX=0
python tmp/q4e-sprint-r1-20261014/run.py px-b1 --prime --diag --generate=32 AILA_Q4E_PREFILL_EXPERT_XMX=1
python tmp/q4e-sprint-r1-20261014/run.py px-b2 --prime --diag --generate=32 AILA_Q4E_PREFILL_EXPERT_XMX=1
python tmp/q4e-sprint-r1-20261014/run.py px-a2 --prime --diag --generate=32 AILA_Q4E_PREFILL_EXPERT_XMX=0
python tmp/q4e-sprint-r1-20261014/scan_ttft.py tmp/q4e-sprint-r1-20261014/px-*-run.log
```

Per arm report: TTFT (`first_content_seconds`), steady-state block times from
`[Q4E AdaptiveWork] submitted_ms` deltas (`dpe-b1-worker.log:19` is the ledger
format), `experts_dev_ms`, and the required evidence line proving the arm
engaged: one `[Q4E PrefillExpertXmx] enabled=1 chunks=...` log per process
(absence means the B arm measured the ALU path).

### 5.3 (c) Pre-registered gates

1. **Block gate**: steady-state 4096-token block <= **13.5 s** (baseline 19.9 s
   `[measured, audit 2.4]`). Miss -> NO-GO for default-on; keep opt-in with the
   measured number recorded.
2. **Default-on gate**: 14K TTFT >= **+5%** over the A median, with A/A spread
   <= 1.5% and the section-4 gate passed. Between 0 and +5%: stays opt-in
   unless the owner explicitly overrides. Negative: NO-GO.
3. Quality gate (section 4) is a prerequisite for any default-on decision;
   it is not traded against the timing gates.

---

## 6. Risks and refuted variants

Already-refuted variants - do not repeat:

| Variant | Verdict | Evidence |
|---|---|---|
| M=8 joint_matrix accumulator (prefill token8 window) | NO-GO: spill 3904 B vs 320/576 B, +23.1% block | `docs/plans/qwen38-token8-window-screen-2026-10-10.md` (VTune r8 table) |
| wgshare=4 as the fix | ADOPTED but only -3.6..-3.8% block / -2.94% TTFT | `docs/plans/qwen38-wgshare-screen-2026-10-10.md:19-23,43` |
| P4 SLM dequant-relay (`_wgshare_slm`) | NO-GO: -11..-18%; oracle bounds -54..-68% | `docs/plans/qwen38-p4-slmshare-2026-10-13.md` |
| Old grouped-XMX tiles M8/N32/K32, separate unpack launch | Never priced at real M; do not extend | `src/ops/MoeGroupedXmx.cpp:24-126`; audit section 4.3 |

Live risks:

1. **oneDNN memory lifetime**: destroying `dnnl::memory` handles while a batch
   is queued can deadlock this driver (`Bnb4BitLinear.hpp:351-355`); the plan
   reuses that pattern (keep handles until the queue drains).
2. **Queue/batch release**: the production L0 v1 regular command list holds
   submitted kernels until a SYCL wait (audit 3.3). The XMX path adds ~150
   launches/layer (~7 200/block, ~45-50 ms at 5.9-7 us `[est]`); if the arm
   under-delivers, check this before blaming the GEMMs.
3. **Dequant correctness for nested codes**: scale must be
   `nested_quant_map[codes[blk]] * nested_absmax[blk/256] + offset` from
   `MoeNestedScaleDev` (copy `MoeGroupedXmx.cpp:13-22`); a wrong offset or
   blocksize silently butchers quality without failing the kernel pre-check.
4. **Padding cost**: 1.16-1.20x extra FLOPs `[est]`; acceptable because the
   GEMM arm is weight-bandwidth bound, but the kernel arm must confirm DX on
   the *plan* chunking, not only the probe's fixed-16 chunking.
5. **VRAM**: +375 MiB inside the 3584 MB reserve; if the A arm already sits
   near the WDDM paging threshold, check the A arm's free-VRAM monitor and
   unchanged `AILA_Q4E_VRAM_RESERVE_MB=3584` before blaming the flag.
6. **Numerics**: bf16 weights + f32 accumulation differ from the ALU f32 LUT
   chain; the quality gate in section 4 exists precisely because this is not
   bit-exact. Do not add a bit-exactness gate that the dense path never had.

---

## 7. Suggested parallel workstreams (disjoint ownership)

| WS | Owns (files) | Deliverable | Depends on |
|---|---|---|---|
| A. Dequant + glue kernels | `src/ops/MoeExpertXmx.hpp` + `.cpp` (new), `CMakeLists.txt` (ops source list, `:179-181` neighbourhood) | `moe_expert_xmx_dequant_chunk`, gather, scatter, plus a chunk driver (oneDNN) with prim cache; kernel unit test in `tests/ops/MoeExpertXmxTests.cpp` | nothing; frozen interface = section 2.2 signatures |
| B. Backend integration | `src/models/Qwen4ExpBnb4Backend.{cpp,hpp}` only | chunk planner in `RunAdaptivePrefill`, `RunGroupedExpertsXmx`, buffer alloc behind the flag, `[Q4E PrefillExpertXmx]` log, flag read | A's header; can start immediately with the header stub |
| C. Quality + measurement | `tests/models/Qwen4ExpGroupedXmxAcceptance.hpp` (extend), `perf/qwen4exp/` (bench script), `tmp/q4e-sprint-r1-20261014/px-*` receipts, doc updates | section-4 gate runnable, kernel bench, A/B/B/A receipts | B's flag; the `--decode-model` recipe exists now |

Coordination contract: WS-A is the only writer of the new ops file and of the
ops source list; WS-B consumes only its public header; WS-C never edits
`src/`. Merge order A -> B -> C, each independently buildable.

---

## 8. Open questions

1. **Row budget value**: 2048 is chosen to match the 1.16-1.20x padding `[est]`
   and 75 MiB scratch. Is a smaller budget (1024) worth +~4% padding for 41 MiB
   less scratch? Only relevant if the A arm shows paging pressure.
2. **gate_up dst dtype**: the probe used bf16 C1 + f32 C2. Keep bf16 C1 (dense
   prefill parity) or f32 C1 for a tighter KL? f32 C1 doubles C1/SwiGLU traffic
   (~+126 MB/layer `[est]`).
3. **Layer arena interaction**: with `AILA_Q4E_LAYER_STAGE=1` the arena serves a
   whole layer's cold set; should the planner merge arena+resident into one
   sorted order (fewer chunks) or keep the three source-class passes (simpler,
   same bytes)? Current plan: keep the three passes.
4. **Dequant parallelism** (1 work-item/uint32 vs 2 per vec8): probe's
   0.354 us/act leaves ~0.9 s/block visible after H2D hiding `[est]`; measure
   before optimizing. Should `..._XMX=2` mean "resident-only XMX, banks ALU"
   as a bisect aid? Not in scope for run 1.
