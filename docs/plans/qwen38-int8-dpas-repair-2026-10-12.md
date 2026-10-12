# INT8 DPAS attention repair (2026-10-12)

Reviewed conversation: `01a123d3-c443-72a1-9516-1bf5ae69ed29`, especially
the B1 Stage A experiment and its archived patch in
`tmp/q4e-int8kv-dpas-20261012/`. Starting checkout: `b835180`.

## Findings

1. The attention patch inherited the build's default subgroup size 16;
   every standalone probe explicitly requested SG16. The failed N8 INT32
   matrix probes never tried SG8. Their IGC divide-by-zero failures do not
   establish that the toolchain cannot implement INT8 joint_matrix.
   Existing `tests/models/Qwen4ExpDpasMoeProbe.hpp` already uses SG8 INT8
   matrices on this device.
2. In the archived patch, each subgroup computed two N8 chunks but stored
   both in the same `dpas_acc[sg * 8]` slice. Both chunks were read only
   after the second store, duplicating the second chunk's scores.
3. The old guard tested only `t0 < tile_len`. A joint_matrix load still
   reads eight entire cache rows when fewer than eight are live. Masking
   the output later does not bound those loads.
4. The archived bisect version disabled the DPAS body with `#if 0` while
   retaining `!kv_dpas` guards on scalar scoring. Enabling its switch would
   consume uninitialized logits. That artifact was never shipped.

## Repair and scope

- A separate SG8 phase-1 kernel computes QK using M1/N8/K32 INT8 matrices.
  K remains `[kv_head, token, 256]` and is loaded as col-major K-transpose.
- The 128-thread workgroup has 16 SG8 groups, each owning one eight-token
  chunk and distinct base/residual result slices. No second chunk overwrites
  a first chunk's result.
- Q uses base plus residual INT8 passes with independently reduced scales.
  Both passes reuse each K matrix load. Per-token K scaling follows the
  integer dot products; V retains the existing FP32 dequantization path.
- Incomplete N8 chunks compute scalar QK over valid rows only. Zero Q and
  zero residual scales produce zero codes without division by zero.
- Existing partial stride (272), merge and output gate are retained.
  No dynamic USM allocation is introduced in attention execution.
- `AILA_KV_QUANT_DPAS=1` enables only INT8 D256 exact decode with usable
  partial scratch and effective length >=1024. Sparse indices, sparse blocks
  and a nonzero sink use the original implementation. A contiguous window
  can retain its nonzero cache start. Short D256, D128, prefill and PV DPAS
  are outside this repair's scope.
- Default is off. Matrix capability is checked and cached per calling thread
  and device; only the s8/s8/s32 N8/K32 combination is accepted.

## Verification

Stack: Intel Arc A770; driver `32.0.101.8724`; IntelLLVM 2026.1;
oneDNN 3.11.2; Release, existing SPIR-V per-kernel JIT flags.

- SG8 probe: M8 global A/output, M8 SLM A/output and M1 SLM A/output;
  136/136 signed integer dot products exactly match a logical CPU reference.
- `AilaQwen4ExpInt8DpasAttention`: passed. Independent FP64 attention uses
  original BF16 Q, logical signed KV rows and independent token scales.
  Tests cover lengths 1024, 1025, 1031, 1032, 1152, 1834, 14336; cache
  capacity equals live length; additional sparse/gated case has length 1153.
  Tile-max error <=6.62e-5; BF16 output NRMSE <=3.27e-4 against this reference.
  Output canaries, hot allocation counts and final USM teardown pass.
- Existing BF16/FP8/INT8 three-arm checks pass with the DPAS switch on.
  At 14336 tokens, INT8-vs-original-BF16 NRMSE is 0.00510398 and cosine
  0.999987. The dual-Q path adds small rounding differences and is not bitwise
  equivalent to scalar dequantization.
- Full primary build succeeds with `build.ps1 -BuildDir build-refactor`.
- CPU/ABI CTest selection: 18 passed, one GPU C API test skipped as designed
  without a model/worker. Existing `AilaQwen4ExpDpasSmoke` also passes.
  Independent FP64 regression passes with the switch off as well.

Artifacts and repeatable guarded runners:
`tmp/q4e-int8-dpas-repair-20261012/`. This directory preserves the old failed
experiments separately from the new passing probes.

## Performance

Same binary and cache fixtures; Hq24/Hkv2/D256; two sequential decode calls
per sample, three warmups and 20 samples per arm, reported wall medians with
p10/p90 in the logs. After the build, the process sequence was off/on/off/on/
off/on; no other GPU pressure target ran concurrently. The initial run that
overlapped host linking is excluded from the repeated results below.

| Live length | Vector INT8: three medians (ms) | DPAS INT8: three medians (ms) | DPAS / vector |
|---|---|---|---|
| 333 | 0.46660 / 0.46800 / 0.48425 | 0.46685 / 0.46700 / 0.46685 | Same short-context fallback; noise only |
| 1834 | 0.34100 / 0.33970 / 0.34115 | 0.51605 / 0.51895 / 0.51795 | 1.519x |
| 14336 | 1.04300 / 1.04255 / 1.04300 | 1.92545 / 1.93055 / 1.92245 | 1.846x |

At 14336 tokens, the median of repeated medians is 1.92545 ms for DPAS vs
1.04300 ms for vector INT8: **84.6% slower**. The three DPAS medians span
less than 0.5%; individual measurements contain outliers recorded by p90.
This is wall time, not profiled device time; the bottleneck has not been
attributed by VTune. `performance.json` retains the medians and dispersion.

Correctness is restored; the current QK DPAS implementation is experimental
and stays off by default. No full-model performance test or speedup is claimed.

Reproduce the new regression:

```powershell
python tmp/q4e-int8-dpas-repair-20261012/run_guarded.py --label dpas-regression ctest --test-dir build-refactor -R '^AilaQwen4ExpInt8DpasAttention$' --output-on-failure
```

## External reference

Intel's [joint_matrix specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)
lists the DG2 INT8 N8/K32 combination. Actual correctness and timings in this
record come from this machine's tests, not from the capability table alone.
