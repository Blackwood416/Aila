# R1 decode MoE (C1): 27.5 ms experts_dev decomposition, variants, and bench

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, HEAD as of writing (pair
`retained25`). Scope: new `src/ops/MoeDecodeVariant.{hpp,cpp}`,
`perf/qwen4exp/q4e_decode_moe_bench.cpp`, `CMakeLists.txt` registrations, this
doc. Zero GPU, zero build, zero commit by this agent. ASCII only. Every number
carries its source; estimates are marked `[estimate]`.

---

## 1. What actually runs, and where the 27.49 ms goes

### 1.1 Production path (evidence)

Decode MoE device work lives in `Qwen4ExpBnb4Backend::RunResidentDecode`
(the only call site is `src/models/Qwen4ExpBnb4Backend.cpp:2423`, 5-argument
form => the overload at `:1179`, not the lease overload at `:1369`, which has
no caller in the tree). Facts pinned from the source:

- descriptors are host `std::array<ops::MoeExpertSlotDev, 16>` copies
  (`:1310-1313` hot, `:1329-1333` staging), so the kernel uses the array-argument path
  (`MoeExpertOps.cpp:4912-4920`), not a device descriptor load;
- `nested_fixed256 = true` is already passed (`:1187`, calls `:1315-1327`).
  This specialization was screened and retained on 2026-10-09
  (`docs/plans/qwen38-nested-div256-screen-2026-10-09.md`: ~5% kernel-chain
  gain in the existing micro-bench, 14K e2e pair-median -2.40%/-2.60%, not a
  decode claim); do not re-open it. The runtime 64-bit `index / blocksize`
  division only survives in the (currently uncalled) `Qwen4ExpExpertExecutor`
  decode lambdas (`:1174-1177`, `:1700-1707`);
- per token the layer loop splits routes into hot hits (resident slots) and
  misses; misses go either to the CPU arena (`StartArenaCold`, `:1295`) or to
  staging H2D (`StageColdExpertsAsync`, `:1301`);
- per layer the GPU sees up to three launches: hot gu (`:1314`), hot down
  (`:1320`), staging gu/down (`:1335-1344`, dependent on the DMA event),
  then the rank merge (`moe_ordered_accumulate_down_rows` or the CPU-host-rows
  form, `:1356`).

### 1.2 Per-token counts from the instrumentation arm (evidence)

`tmp/q4e-sprint-r1-20261014/dec-instr-b2-worker.log` count=1 lines
(e.g. `:26-46`), fields added by the B6 arm:

| token (line) | hits | misses | cpu_tasks | staged | cpu_compute_ms | cpu_join_ms | experts_dev_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| :28 | 287 | 193 | 193 | 0 | 13.32 | 9.79 | 27.64 |
| :34 | 137 | 343 | 343 | 0 | 19.45 | 16.82 | 31.65 |
| :36 | 80 | 400 | 400 | 0 | 21.80 | 18.26 | 33.06 |
| :46 | 165 | 315 | 315 | 0 | 17.69 | 12.94 | 30.57 |

Across the ten sampled tokens (lines `:26-46`, even lines): hits
271/287/187/216/137/80/185/138/194/160 (mean 186 => E_g ~ 3.9 hot experts
per layer), misses == cpu_tasks (all misses are served by the CPU arena),
`staged=0` (the staging path is idle in this profile), `cpu_compute_ms`
13.3-21.8 (mean 17.8), `cpu_join_ms` 9.8-18.3 (mean 13.2),
`experts_dev_ms` 27.6-36.7 (mean 31.0; the n=128 median in the cost model is
27.49, `docs/plans/qwen38-r1-cost-model-2026-10-14.md` section 6.17).

### 1.3 Arithmetic: kernels are a small, mostly hidden share

Bytes per hot expert are fixed by the adaptive geometry
(`Qwen4ExpAdaptiveCache.cpp:100-119`, compact nested):

| component | bytes | source |
|---|---:|---|
| gu packed | 1,638,400 | `2*I*(H/2)`, H=2560, I=640 |
| gu nested codes | 51,200 | `2*I*(H/64)`, 1 B/code |
| dn packed | 819,200 | `H*(I/2)` |
| dn nested codes | 25,600 | `H*(I/64)` |
| total slot | 2,534,400 (2.417 MiB) | matches R1 kernel doc:33 |

At the observed E_g = 3.9: gu 6.59 MB + dn 3.29 MB = 9.88 MB/layer.

- DRAM floor at 291.11 GB/s (measured streaming read): 33.9 us/layer
  `[estimate]`, 1.63 ms/token.
- K1 attainment: `--decode-gu-bench` E=10 measured 0.12343 ms for 16.90 MB
  (`docs/plans/qwen38-r1-kernel-2026-10-14.md` section 3, probe K1) = 2.13x
  off the wall. Scaling that ratio to E=3.9 (bit-identical instantiation
  modulo the descriptor count, `[estimate]`): gu ~48 us, dn ~24-55 us
  (dn has no historical bench; the range covers "same ratio" to "same
  bytes/row efficiency"), merge 2560 f32 rows ~3 us. Per layer ~75-105 us
  `[estimate]`, i.e. **3.6-5.1 ms/token of kernel busy time**.
- Per-layer CPU cold work is `cpu_compute_ms/48` = 277-454 us, and the host
  join inside `RunResidentDecode` blocks the submission of the merge until
  the CPU arena finished (`JoinResidentCold`, `:1349`). The in-order queue
  therefore has gu/down hidden under the CPU workers and then goes idle for
  (cpu_compute - gpu_kernels) ~ 200-350 us/layer.

Conclusion (STRONG for the counts, `[estimate]` for the split): of the
27.49 ms experts_dev interval, only ~3.6-5.1 ms/token is GPU kernel execution;
the remainder is host-side join waiting, result publication and submission
gaps. A 5% faster GEMV inside the current schedule buys ~0.25 ms/token
(~0.35% wall) - far below the charter 5% e2e gate. Kernel-side MoE work is
only worth pursuing *after* D/R rebalances the CPU/GPU split (their C1
candidate, upper bound ~10.2 ms/token); until then the correct kernel-side
deliverable is a calibrated floor, not a rewrite. This bench provides it.

---

## 2. Variants in `src/ops/MoeDecodeVariant.{hpp,cpp}` (research only)

Both knobs are exact-math rewrites of
`launch_moe_decode_gate_up_swiglu_grouped_dev` (MoeExpertOps.cpp:853-999) and
`launch_moe_decode_down_to_rank_f32_dev` (:1211-1335); the FMA sequence,
nibble order, scale reconstruction (two separated `fma` steps) and the
sub-group-16 row mapping are copied verbatim, so output is expected
bit-identical and the bench asserts that.

| # | knob | mechanism | status |
|---|---|---|---|
| V1 | `fixed_scale_256` | `index >> 8` for the nested block and `byte_offset >> 5` for `blocksize/2` (both validated exact powers of two, caller contract MoeExpertOps.hpp:138-139, `make_slot` throws otherwise, AdaptiveCache.cpp:319) | **already shipped** (2026-10-09 screen, retained); kept only as the A/B control so the calibration arms run against the production instantiation, plus insurance for the executor call sites that omit the flag |
| V2 | `uniform_scale_loads` | every lane evaluates the nested absmax chain for its (uniform) block index; removes 4 `group_broadcast` + 2 selects per iteration (gu) / 2 broadcasts (dn) | new; the only remaining non-denied micro-lever |

Not attempted, on purpose: 2 rows/subgroup, uint64/vec2 weight loads,
half-unpack or byte-pair LUT, blocked layouts, qmap in registers, SLM input
caching, fused residual, atomics in the merge (all charter section 2 or
project negative list). No SLM is used by these kernels beyond the 16-float
`qmap` (64 B); register pressure is unchanged by both knobs.

---

## 3. Bench `AilaQwen4ExpDecodeMoeBench` - protocol and gates

Outputs one JSON line per arm:

    [DECODE MOE BENCH] {"arm":...,"experts":...,"rows":...,"median_ms":...,
                        "p10_ms":...,"p90_ms":...,"cv_pct":...,"mb":...,"gbps":...}
    [DECODE MOE PARITY] {"arm":...,"h_mismatch":0,"down_mismatch":0}
    [DECODE MOE GUARD]  {"sentinel_ok":1}
    [DECODE MOE CHAIN]  {"arm":...,"layers":48,"experts":5,"wall_ms":...,
                         "busy_ms":...,"gap_ms":...,"kernel_share_pct":...}
    [DECODE MOE GATE]   {...}   (per shape, gain vs the production instantiation)
    [DECODE MOE CANDIDATE] {...} (both-shape verdict for fixed_scale_256)

Arms per shape: `gu/dn-ref-nested-div` (production instantiations on device
descriptors), `gu/dn-ref-fp32` (K1-iso: fp32 absmax, non-nested),
`gu/dn-ref-arrayform` (E<=16: host descriptors == the production call form),
`gu/dn-var-uniform`, `gu/dn-var-fixed256`, `gu/dn-var-fixed256-uniform`.
Parity: every variant arm is compared bitwise against the production nested
instantiation on both `h` (E*640 bf16) and `down` (rows*top_k*2560 f32);
array-vs-device descriptor form is compared the same way. Sentinels after the
written regions fail the run on out-of-bounds writes; non-finite outputs throw.

Chain arm: 48 layers x (gu -> down -> merge) on a `layers x E` slot pool with
distinct weights per layer (608 MB device for E=5 `[estimate]`), ref vs
fixed256 vs fixed256+uniform, reporting wall = last end - first start,
busy = sum of all 144 command spans, gap = wall - busy. `per_layer_ms` is the
kernel-side floor to compare with `experts_dev_ms/48 = 0.573 ms` (27.49/48).

Gates (charter section 5.3): a variant is kernel-level GO only if
`gain >= 3%` at BOTH shapes E=10 and E=20 (rows=1; rows=2 is an optional
third point for MTP-interest), and e2e adoption needs >= +5% wall - which,
per section 1.3, a kernel-side change cannot produce on its own right now.
e2e projection for any kernel gain g, using `kernel_busy = chain.busy_ms` at
E=5 (the production hot-count proxy): in the current schedule the kernels are
hidden under the CPU arena work (section 1.3), so the wall delta is ~0; in a
hypothetical kernel-bound schedule it would be `g * kernel_busy`, i.e. with
the expected 4-7 ms floor even a 10% kernel gain is 0.5-0.9% of the 74 ms
token. Both readings sit below the 5% e2e gate.
The arms E=5 chain and `gu-ref-fp32` are diagnostic, not gated.

Predictions `[estimate]`: gu E=10 nested == fixed256 (bit-identical);
uniform gain 0-3%; E=5 per-byte efficiency within 15% of E=10; chain
per-layer 0.09-0.15 ms at E=5.

---

## 4. Risks

- Parity is asserted, not assumed: the variant must keep the exact
  `qmap[...] * ga` -> `fma` expression and the row->subgroup mapping; the
  bench fails closed on any bit mismatch, and a mismatch means "variant
  invalid", not "tolerance".
- The profiling queue changes per-command timing slightly; wall medians with
  warmups are the primary numbers, spans are diagnostic (same convention as
  the K1 arm).
- Small-shape (E=5) timing is launch/occupancy sensitive; A/A on two
  processes is required before reading any <3% difference.
- The chain arm cannot reproduce the CPU-join bubble or staging DMA; it is
  deliberately a *kernel-only* floor. Do not compare it to wall directly.
- Device memory: chain pool 608 MB + shape pools; the bench allocates its own
  Context and must run with no AilaWorker/AilaServer resident (GPU mutex).

---

## 5. Coordinator commands

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # must be empty
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor          # registers the new files
build-refactor/AilaQwen4ExpDecodeMoeBench.exe --experts 10,20 --rows 1 *> tmp/q4e-sprint-r1-20261014/decode-moe/decode-moe-A1.log
build-refactor/AilaQwen4ExpDecodeMoeBench.exe --experts 10,20 --rows 1 *> tmp/q4e-sprint-r1-20261014/decode-moe/decode-moe-A2.log   # A/A
```

Read order per log: `[DECODE MOE DEVICE]` -> `PARITY` (must be 0 mismatches on
every arm, and must not throw) -> `GUARD` -> `BENCH` medians -> `CHAIN`
(`busy_ms`, `kernel_share_pct`) -> `GATE`/`CANDIDATE`.

Expected decision (pre-registered): fixed256 shows ~0% (already shipped),
uniform shows 0-3% (below the kernel gate at two shapes), and the chain
`busy_ms` lands at 4-7 ms per token, i.e. **~15-25% of the 27.49 ms
experts_dev span**. In that case close the decode-MoE kernel line, keep the
bench as the calibrated floor, and route decode work to D/R (CPU/GPU split
rebalance; their C1 upper bound 10.22 ms/token).

---

## 6. Sources

- `src/models/Qwen4ExpBnb4Backend.cpp:1179-1365` (RunResidentDecode adaptive
  overload: array descriptors, fixed256=true, hot/staging/CPU split,
  merge), `:2423` (call site), `:1369` (uncalled lease overload).
- `src/models/Qwen4ExpAdaptiveCache.cpp:100-119` (slot geometry),
  `:312-332` (make_slot: nested blocksize 256 contract).
- `src/ops/MoeExpertOps.cpp:406-423` (scale reconstruction),
  `:853-999` (gu dev kernel), `:1211-1335` (down dev kernel),
  `:6027-6075` (merge), `:4912-4920` (array dispatch).
- `docs/plans/qwen38-r1-cost-model-2026-10-14.md` section 6.17
  (experts_dev 27.49 ms median, n=128; device-span method description).
- `docs/plans/qwen38-r1-kernel-2026-10-14.md` section 1.1/1.2/3
  (slot bytes, K1 bench shape and decision rule).
- `docs/plans/qwen38-r1-decode-2026-10-15.md` section 4 (cpu_join 10.22
  ms/token and the CPU/GPU split candidate C1).
- `tmp/q4e-sprint-r1-20261014/dec-instr-b2-worker.log:26-46` (per-token
  hits/misses/cpu_tasks/staged/cpu_compute/cpu_join/experts_dev).
- `docs/plans/qwen38-r1-decode-instrumentation-2026-10-15.md` section 2.2
  (marker semantics: experts_dev includes in-order bubbles).
- `docs/plans/qwen38-nested-div256-screen-2026-10-09.md` (fixed-256
  specialization already screened/retained; both existing bitwise-gated
  benches; ~5% kernel-chain micro gain; no decode claim).
