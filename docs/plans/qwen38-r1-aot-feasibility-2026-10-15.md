# Q4E R1: AOT device-compile feasibility (Qwen4Exp / A770)

Date: 2026-10-15. Repo HEAD: `58f7874` (pair retained26). Round: E7 (source/build research +
one compile-only smoke). Owner: subagent E7.

Scope discipline actually honored:

- No GPU work of any kind (the d0-deferwait-b2 arm held the GPU for the whole round;
  `Get-Process` showed `AilaServer`/`AilaWorker` alive and untouched).
- No repo binary was executed. No `ninja`/`cmake` configure, no `build-refactor` access.
- `src/`, `tests/`, `CMakeLists.txt` are read-only in this round. No git writes, no commit.
- All smoke artifacts and logs are under `tmp/q4e-sprint-r1-20261014/aot/`.

## 0. Verdict

**GO for AOT device compilation at the compile phase** (measured, see section 2), with **one
untested step and one plumbing prerequisite**:

| Item | Status |
|---|---|
| Driver accepts `-fsycl-targets=intel_gpu_acm_g10` / `...,spir64` on this toolchain | PROVEN (section 2) |
| Kernel TUs incl. joint-matrix/XMX and the 42-launch attention TU compile clean under AOT | PROVEN (section 2) |
| Device payload is tagged per AOT target (`sycl-spir64_gen-unknown-unknown-acm_g10`) | PROVEN (verbose logs) |
| SPIR-V -> native (ocloc/IGC) translation during **link/device-link** | NOT TESTED (out of this round's boundary) |
| Opt-in build plumbing for the AOT flag | MISSING: global `CMAKE_CXX_FLAGS` does not work (proven error, section 1.3) |

Prerequisites before a full scratch build (do not start one before solving P1):

- **P1 (build plumbing)**: `-fsycl-targets` cannot be injected through `CMAKE_CXX_FLAGS`; the driver
  hard-errors when it is used without `-fsycl`, and three targets compile with `-fno-sycl`
  (`soxr`, `simdjson_proxy`, `AilaShared`). An opt-in CMake option is required (patch sketch in
  section 3.2; excluded from this round because `CMakeLists.txt` is read-only here).
- **P2 (link-time AOT)**: the actual hardware lowering happens in the link/device-link stage. A
  5-minute, GPU-free link probe on a trivial kernel should be run first (section 6, TODO-1).
- **P3 (runtime acceptance)**: cold-start A/B protocol in section 4 has to pass before this can be
  called a win.

## 1. Source and build research (read-only)

### 1.1 Current SYCL flags

`CMakeLists.txt:116-122` defines the only SYCL interface target:

```
add_library(AilaSyclOptions INTERFACE)
target_compile_options(AilaSyclOptions INTERFACE
    -fsycl
    -fsycl-device-code-split=per_kernel
    $<$<BOOL:${MSVC}>:-fsycl-default-sub-group-size=16>
)
target_link_options(AilaSyclOptions INTERFACE -fsycl)
```

- `-fsycl-targets` appears **nowhere** in the repo (only in plan docs, e.g.
  `docs/Qwen38FlashNext_A770_Handoff.md:425-426`, item 5 of "Recommended next steps", and the C14
  row of `docs/plans/qwen38-r1-backlog-mining-2026-10-15.md`).
- Device code split is `per_kernel` globally. This matters for AOT: during device link every kernel
  becomes its own device image, i.e. the AOT compiler is invoked per kernel (section 5, risk R2).
- `build.ps1` (lines 57-70) only selects the oneAPI stack and passes the compiler path; it adds no
  SYCL target flags. Stack `oneapi-2026.1` = compiler 2026.1.0 (build 20260617), dnnl 2026.0,
  tbb 2023.1, umf 1.1 (`perf/oneapi-stacks.json`).
- `ocloc` is a **separate** installed component at
  `C:\Program Files (x86)\Intel\oneAPI\ocloc\2026.1\bin\ocloc.exe` (IGC: `igc64.dll`,
  `iga64.dll`, `opencl-clang264.dll`). The compiler's own `env\vars.bat` puts
  `%CMPLR_ROOT%\lib\ocloc` on `PATH`, but that directory does not exist in this installation
  (verified), so no ocloc path is provided by the repo environment helper.

### 1.2 Kernel/device-code inventory (`rg` counts, HEAD 58f7874, src/ only)

| Metric | Value |
|---|---|
| TUs that touch SYCL (`.cpp`) | 52 |
| `parallel_for` launch sites | 337 |
| `joint_matrix` references | 133 |
| ESIMD (`esimd`, `simd_view`, `simd<`) | 0 (no ESIMD source today) |
| Biggest kernel TUs | `src/ops/AttentionOps.cpp` 5711 lines / 42 launches / 54 JM / 5 DPAS; `src/ops/Bnb4BitLinear.cpp` 4102 / 31 / 59; `src/ops/Qwen4ExpBnb4Backend.cpp` 3158 / 11; `src/ops/QsaIndexerOps.cpp` 2127 / 15 |
| Small JM TU (used as smoke proxy) | `src/ops/MoeGroupedXmx.cpp` 151 lines / 1 launch / 12 JM, M8/N32/K32 SG8 XMX hypothesis kernel |

Consequence: ESIMD is not a risk today (nothing to test). joint-matrix/DPAS paths are the ones that
must survive AOT, and they are exactly what the smoke covered.

### 1.3 What the driver actually does (from `-v` sub-command logs)

AOT compile-only (`-fsycl-targets=intel_gpu_acm_g10`, `tmp/.../run_07_verbose.log`):

1. Device pass: `clang-cl.exe -cc1 -triple spir64_gen-unknown-unknown ... -D__SYCL_TARGET_INTEL_GPU_ACM_G10__
   ... -emit-llvm-bc ... -o NormOps-acm_g10-<hash>.bc`
2. Host pass: `-fsycl-is-host ... -D__SYCL_TARGET_INTEL_GPU_ACM_G10__ ...`
3. `clang-offload-bundler -type=o -targets=sycl-spir64_gen-unknown-unknown-acm_g10,host-x86_64-pc-windows-msvc`

JIT control (`run_08_verbose_jit.log`): bundler target string is
`sycl-spir64-unknown-unknown` instead, and the `.bc` device file has no `acm_g10` suffix.

Three conclusions that change the plan:

1. With `-c` the device part is stored as **LLVM bitcode**, not SPIR-V and not native code, in both
   modes. The `intel_gpu_acm_g10` target is carried as a bundle target triple
   (`...-unknown-unknown-acm_g10`) plus the `__SYCL_TARGET_INTEL_GPU_ACM_G10__` macro.
2. Therefore the real AOT translation (bitcode -> SPIR-V -> ocloc/IGC native) happens at
   **link/device-link**, and `ocloc` is *not* needed (and not invoked) for `-c`. This is confirmed
   by run 03: AOT compile succeeded with ocloc absent from `PATH`.
3. Invalid target names are rejected by the driver itself:
   `icx-cl: error: SYCL target is invalid: 'intel_gpu_bogus_g99'` (exit 1).

Object-size delta between the two modes (same TU, same source, same flags otherwise):

| TU | AOT obj bytes | JIT obj bytes | delta |
|---|---|---|---|
| `MoeGroupedXmx.cpp` (JM) | 138,284 | 120,656 | +17,628 (+14.6%) |
| `AttentionOps.cpp` (JM/DPAS) | 2,166,116 | 1,910,550 | +255,566 (+13.4%) |
| `NormOps.cpp` (plain) | 220,924 | 220,724 | +200 (+0.09%) |

The delta tracks joint-matrix usage: the device module differs (extra `spir64_gen` target image and
JM/XMX-specific instantiation), while a plain norm TU is nearly identical. Exact payload encoding
was not decoded (no zlib/zstd/SPIR-V/ELF magic is visible in the raw objects; the bundle payload is
opaque to a plain byte scan). Not needed for the decision, but do not over-claim it: the objects
prove *target tagging*, not native ISA generation.

## 2. Smoke evidence (compile-only, CPU only)

Harness: `tmp/q4e-sprint-r1-20261014/aot/env_setup.bat` (vcvars64 + oneAPI component `vars.bat`,
optionally the separate ocloc component on PATH) + `run_common.bat` (the exact repo include/define
set from `CMakeLists.txt`) + one `.bat` per run. Each run is `icx-cl -c` only; nothing is linked.

| Run | Source | Targets | ocloc on PATH | Wall | Exit | Log |
|---|---|---|---|---|---|---|
| 01 | `src/ops/MoeGroupedXmx.cpp` (JM) | `intel_gpu_acm_g10,spir64` | yes | 16.9 s | 0 | `run_01_aot_moegroupedxmx.log` |
| 02 | same (JIT control) | `spir64` | yes | 10.5 s | 0 | `run_02_jit_moegroupedxmx.log` |
| 03 | `src/ops/NormOps.cpp` | `intel_gpu_acm_g10` | **no** | 10.1 s | 0 | `run_03_noocloc.log` |
| 04 | same | `intel_gpu_acm_g10` | yes | 10.3 s | 0 | `run_04_aot_normops.log` |
| 05 | `src/ops/AttentionOps.cpp` (42 launches, JM+DPAS) | `intel_gpu_acm_g10,spir64` | yes | 27.9 s | 0 | `run_05_aot_attentionops.log` |
| 06 | same (JIT control) | `spir64` | yes | 21.2 s | 0 | `run_06_jit_attentionops.log` |
| 07/08 | `NormOps.cpp` `-v` sub-command capture | AOT / JIT | yes | ~11 s each | 0 | `run_07_verbose.log`, `run_08_verbose_jit.log` |
| probe A | `NormOps.cpp` | `intel_gpu_bogus_g99` | no | <2 s | 1 | `probe_bogus_target.log` |
| probe B | trivial `.cpp`, `-fsycl-targets` without `-fsycl` | `intel_gpu_acm_g10,spir64` | no | <2 s | 1 | `probe_targets_without_fsycl.log` |

Exact command shape (run 05):

```
icx-cl.exe -c -O3 -ffast-math /Qstd=c++17 /EHsc /Qftz \
  -fsycl -fsycl-device-code-split=per_kernel -fsycl-default-sub-group-size=16 \
  -fsycl-targets=intel_gpu_acm_g10,spir64 \
  -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DAILA_STATIC_LIB -DAILA_BUILD_ID="aila-aot-smoke" \
  -I<repo include dirs from CMakeLists.txt> -I"<dnnl include>" \
  src\ops\AttentionOps.cpp -Fo tmp\q4e-sprint-r1-20261014\aot\attentionops_aot.obj
```

Probe B raw error (this is the reason P1 exists):

```
icx-cl: error: '-fsycl-targets' must be used in conjunction with '-fsycl' to enable offloading
```

No warnings emanated from any AOT device pass. No kernel-side compile error appeared for
joint-matrix, DPAS, subgroup-size-16 or SLM-heavy kernels.

## 3. Full scratch build: exact command draft

### 3.1 Shape

Separate build dir `build-aot-scratch/` (never `build-refactor/`), limited to the three artifacts
that must carry device images (avoids building the 54 CTest executables):

```powershell
# 1. Environment exactly like build.ps1 (vcvars64 + umf/tbb/dnnl/compiler vars.bat),
#    e.g. dot-source perf\PerfCommon.ps1 and call:
#      $cfg = Get-AilaOneApiStackConfig -RepoRoot <repo>
#      $stack = Get-AilaOneApiStack -Config $cfg -Name 'oneapi-2026.1'
#      Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

# 2. Configure (scratch dir only)
cmake -S E:\RiderProjects\Aila -B E:\RiderProjects\Aila\build-aot-scratch -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER="C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin\icx-cl.exe" `
  -Ddnnl_DIR="C:\Program Files (x86)\Intel\oneAPI\dnnl\2026.0\lib\cmake\dnnl" `
  -DTBB_DIR="C:\Program Files (x86)\Intel\oneAPI\tbb\2023.1\lib\cmake\tbb" `
  -DAILA_ALLOW_LEGACY_ONEAPI_BASELINE=OFF `
  -DAILA_SYCL_AOT_TARGETS="intel_gpu_acm_g10,spir64"     # option from P1; does not exist yet

# 3. Build only the artifacts that must carry AOT device images
cmake --build E:\RiderProjects\Aila\build-aot-scratch --target AilaWorker AilaInference Aila --parallel 36
```

Expected cost `[estimate]`, anchored on the section 2 measurements:

| Quantity | Value | Basis |
|---|---|---|
| SYCL TU compile wall, AOT vs JIT | +0 to +60% per TU | 16.9 vs 10.5 s (JM TU), 27.9 vs 21.2 s (attention TU), ~equal on plain norm TU |
| Compile phase for 52 SYCL TUs at `-j36` | 2-6 min `[estimate]` | per-TU times above; TU sizes vary widely |
| Link/device-link AOT (ocloc over ~337 split device images) | unknown, could dominate | not measured; `per_kernel` split multiplies ocloc calls (risk R2) |
| Binary size growth | +3-10 MB `[estimate]` | +13-15% on JM TUs; native images add on top of the retained `spir64` images |
| Disk for scratch dir | 3-6 GB `[estimate]` | mirrors `build-refactor` object/lib volume |

### 3.2 P1 patch sketch (for the next round; NOT applied here)

```cmake
set(AILA_SYCL_AOT_TARGETS "" CACHE STRING "AOT SYCL targets, e.g. intel_gpu_acm_g10,spir64")
add_library(AilaSyclOptions INTERFACE)
target_compile_options(AilaSyclOptions INTERFACE
    -fsycl
    -fsycl-device-code-split=per_kernel
    $<$<BOOL:${MSVC}>:-fsycl-default-sub-group-size=16>
    $<$<BOOL:${AILA_SYCL_AOT_TARGETS}>:-fsycl-targets=${AILA_SYCL_AOT_TARGETS}>
)
target_link_options(AilaSyclOptions INTERFACE
    -fsycl
    $<$<BOOL:${AILA_SYCL_AOT_TARGETS}>:-fsycl-targets=${AILA_SYCL_AOT_TARGETS}>
)
```

Notes: both compile *and* link options are needed (the device link is what materializes the AOT
image). Targets that compile with `-fno-sycl` do not link `AilaSyclOptions`, so they are unaffected.
`AilaInference.dll` and `AilaWorker.exe` must both go through AOT (a JIT-only DLL linked into an AOT
process would leave a JIT window in the product path).

## 4. Cold-start acceptance protocol (A/B: AOT binary vs JIT binary, same pair)

Target: first-request penalty < 10 s, versus the measured ~80 s first-use JIT today
(`docs/plans/qwen38-r1-cost-model-2026-10-14.md` lines 97-116 and L10 line 868: router 50.4 s +
full-attn 19.8 s + linear 7.3 s + hc 1.2 s).

Setup:

- A = `build-aot-scratch/AilaWorker.exe` + `AilaInference.dll` (AOT, targets
  `intel_gpu_acm_g10,spir64`); B = `build-refactor/AilaWorker.exe` (JIT). Same model dir, same
  pair (retained26), same request (19-token prompt), same `AILA_Q4E_LOAD_TIMING=1`.
- GPU mutex: take the cooperating GPU-run guard before launching and hold it until both binaries
  have exited (`tools/server/gpu_run_guard.py`, `ProcessFileLock(<state dir>/n)`); additionally
  assert with `Get-Process AilaWorker,AilaServer` that nothing else owns the GPU. This round's
  environment already had `AilaServer`/`AilaWorker` running - the protocol must refuse to start in
  that state rather than co-schedule.
- No-sleep polling: detect readiness by polling the worker's timing log and the server endpoint in
  a tight loop (>= 20 iterations/s, wall-clock budget 180 s); never use fixed `Start-Sleep`
  intervals, and never treat the readiness poll as part of the metric.
- Metric definition: `t0` = process start; issue the first request as soon as the server accepts
  connections; record (a) first-token latency of that first request, (b) the sum of the
  `AILA_Q4E_LOAD_TIMING` router/full-attn/linear/hc buckets, (c) process CPU time during the wait.
- Repeats: 3 fresh processes per arm, alternating order A,B,B,A,A,B (fresh process each time;
  restart the worker, never reuse a warm one).
- Negative control: arm B must still show the known ~80 s first-use penalty in the same session -
  otherwise the session itself changed and the A/B is void.
- "No JIT happened" evidence for arm A: bucket sum ~0, no new SYCL program-build log lines, and (if
  needed) `syclbin-dump.exe` on `AilaWorker.exe` showing native images instead of SPIR-V.
- Token parity: same greedy 32-token continuation on both arms must match token-for-token; if AOT
  changes numerics (DPAS lowering by IGC instead of runtime SPIR-V), that is a finding, not a
  silent pass.

Script stub (not executed this round): `tmp/q4e-sprint-r1-20261014/aot/coldstart_ab.ps1`.

## 5. Risk list

**R1 - Link/device-link AOT never exercised (highest).** The measurable win lives in the link
stage, which this round could not touch (no build, no link, no `.exe`). Mitigation: TODO-1 probe
first; if the trivial probe links and embeds an acm_g10 image, proceed to the scratch build.

**R2 - `per_kernel` split multiplies AOT work.** 337 launch sites across 52 TUs; with
`-fsycl-device-code-split=per_kernel` the device link will produce per-kernel images, each needing
an ocloc pass. Expect a slow link and a bigger binary. Mitigations to try in order:
`per_source`, `off`, or AOT only for the hot TUs. Do not silently change the JIT build's split
policy - interleaved device images change load/JIT behaviour for the control arm.

**R3 - Device-name lock-in / driver drift.** The AOT image is bound to `acm_g10`
(`...-unknown-unknown-acm_g10`) and produced by this IGC build. Unknown GPU, another driver
revision, or a future stepping can miss the AOT image. Keeping `spir64` in the target list is the
fallback (JIT, slow first use, still correct); dropping it removes the fallback entirely.

**R4 - Runtime kernel-name/registration mismatch.** AOT embeds kernel names (mangled, plus the
`uid...` unique prefix) that must match what the host launcher requests. Renaming kernels, changing
`-fsycl-unique-prefix`, or mixing AOT and JIT objects for the same TU in one binary can produce
"kernel not found at runtime" or duplicate-image warnings. Verify in the first acceptance run, not
by inspection.

**R5 - oneDNN side is untouched.** dnnl 2026.0 GEMM fallbacks use their own runtime codegen; AOT of
our SYCL TUs does not pre-compile them and does not remove their first-use cost. If the fallback is
on the product path, budget it separately (currently believed off the Q4E hot path).

**R6 - ESIMD: no risk today, keep the check.** `rg` finds zero ESIMD use in `src/`. If ESIMD is added
later, re-run this smoke: ESIMD kernels historically had the most AOT-specific issues (device
features/`-Xs` flags have to match the deployment device).

**R7 - issue #4 interaction (likely positive).** `SYCL_CACHE_PERSISTENT=1` crashes at the first JIT
build (reproduced L15). AOT removes the JIT storm from the product path, so it is also the strongest
available mitigation for that crash - but the crash root cause is still unknown, so treat "AOT also
fixes #4" as a hypothesis to be tested, not an established fact.

**R8 - AOT does not cover non-JIT startup costs.** Host arena fill/page hints, the PLE gather
(~20.5 s, L22), expert supply and mmap behaviour are unaffected. If the acceptance run still misses
the 10 s target, attribute the residual with the existing `AILA_Q4E_LOAD_TIMING` buckets before
blaming AOT.

**R9 - Rollback is cheap.** AOT lives only in a separate scratch dir with an opt-in flag; the JIT
build and `build-refactor/` stay untouched, and the default path stays JIT until the acceptance
numbers say otherwise.

## 6. TODO (ordered, all GPU-free until TODO-4)

1. **AOT link probe (5 min, no GPU, no repo binary run).** Link a trivial one-kernel TU with
   `-fsycl -fsycl-targets=intel_gpu_acm_g10` into a scratch `.exe` in `tmp/`, then inspect sections
   (`dumpbin /headers`, `syclbin-dump.exe`, `ocloc disasm`) to confirm a native image is embedded
   and that ocloc was invoked. If it fails without the ocloc component on `PATH`, add
   `C:\Program Files (x86)\Intel\oneAPI\ocloc\2026.1\bin` to `PATH` and retry: that is the exact
   fix for the scratch build's environment.
2. **P1 patch** (section 3.2) as an opt-in, default-off CMake option; verify the JIT build is
   byte-identical in behaviour afterwards.
3. **Scratch build** (section 3.1), limited to `AilaWorker`, `AilaInference`, `Aila`; record link
   wall time and binary sizes; try `per_source`/`off` split if the link dominates.
4. **Cold-start A/B** (section 4) under the GPU guard, 3 fresh runs per arm, criterion
   first-request penalty < 10 s plus token parity.
5. Only if 1-4 pass: decide on the product default (and revisit the opt-in warmup, which costs
   +111 s startup to hide +80 s of JIT).

## 7. Artifacts

`tmp/q4e-sprint-r1-20261014/aot/`

- `env_setup.bat`, `run_common.bat`, `run_01..run_08*.bat`, `run_probe_*.bat`: exact reproduction.
- `run_0*.log`, `probe_*.log`: raw outputs and wall times (the `-v` runs contain the full
  sub-command lines quoted in section 1.3).
- `*.obj`: AOT/JIT objects for the three TUs (proof of targets; not linked, never executed).
- `dump_obj.ps1`, `scan_obj.ps1`, `decode_payload.py`: object inspection helpers.
- `coldstart_ab.ps1`: protocol stub for section 4 (not run; needs GPU mutex + real pair).
