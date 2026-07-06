# BnB4 Prefill Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve or at least precisely benchmark Aila's BnB NF4 prefill GEMM path for Qwen3.5 foreground prefill without changing model output behavior.

**Architecture:** Keep the production path behind existing `AILA_BNB4_FUSED_PREFILL` behavior and add a dedicated microbench target that exercises `packed_nf4_gemm_bf16` with real foreground-like shapes. Use the existing correctness test as the guardrail, then try one layout/tile change at a time. Treat OpenVINO as guidance for weight compression and fused MLP direction, not as drop-in code because Aila uses bitsandbytes NF4 LUT + per-block absmax.

**Tech Stack:** C++17, SYCL, oneAPI, CMake/Ninja, existing `Context`/`Tensor` runtime.

---

### Task 1: Add A Reusable BnB4 Prefill Microbench

**Files:**
- Create: `tools/perf/Bnb4PrefillBench.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/ops/Bnb4BitLinearKernelTests.cpp`

- [ ] **Step 1: Add a benchmark target**

Create `tools/perf/Bnb4PrefillBench.cpp` that includes `src/ops/Bnb4BitLinear.cpp`, allocates deterministic packed NF4 weights, qmap, absmax, bf16 input, and output tensors, then runs `packed_nf4_gemm_bf16` for shapes:

```text
M=118, K=2560, N=10240
M=118, K=10240, N=2560
M=91, K=2560, N=10240
```

Report one CSV line per shape:

```text
shape,M,N,K,warmup_ms,best_ms,avg_ms,max_abs_diff,tokens_per_s
```

- [ ] **Step 2: Wire the target into CMake**

Add:

```cmake
add_executable(AilaBnb4PrefillBench tools/perf/Bnb4PrefillBench.cpp)
target_compile_definitions(AilaBnb4PrefillBench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(AilaBnb4PrefillBench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)
target_link_libraries(AilaBnb4PrefillBench PRIVATE DNNL::dnnl simdjson)
```

- [ ] **Step 3: Verify benchmark compiles and emits timings**

Run:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --config Release --target AilaBnb4PrefillBench AilaBnb4BitLinearKernelTests
.\build\AilaBnb4BitLinearKernelTests.exe
.\build\AilaBnb4PrefillBench.exe
```

Expected: correctness test passes and every benchmark row has `max_abs_diff <= 0.08`.

### Task 2: Try One Layout-Or-Tile Optimization

**Files:**
- Modify: `src/ops/Bnb4BitLinear.cpp`
- Test: `tests/ops/Bnb4BitLinearKernelTests.cpp`

- [ ] **Step 1: Record baseline benchmark output**

Save the baseline command output in the working notes or final response. Do not change code until a baseline exists.

- [ ] **Step 2: Implement exactly one kernel variant**

Prefer a minimal production-safe change in `packed_nf4_gemm_bf16`, selected from the baseline bottleneck:

```text
Candidate A: tune BM/BN/TM/TN for foreground shapes.
Candidate B: pre-expand qmap*absmax for the BN tile into local bf16/float fragments.
Candidate C: add an opt-in transposed packed-weight path only if it can be generated at init time and guarded by tests.
```

Start with Candidate A because it is reversible and does not change weight storage.

- [ ] **Step 3: Verify correctness and benchmark delta**

Run:

```powershell
.\build\AilaBnb4BitLinearKernelTests.exe
.\build\AilaBnb4PrefillBench.exe
git diff --check
```

Expected: correctness passes. Keep the change only if at least one foreground-like shape improves without a meaningful regression on the others.

### Task 3: Real Pipeline Validation

**Files:**
- No source changes expected.

- [ ] **Step 1: Build AliaEngine**

Run:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --target AliaEngine --config Release
```

- [ ] **Step 2: Run short Chinese stream smoke**

Run:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -AudioPath 'tmp\alia-real-smoke\bnb4_prefill_layout_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\bnb4_prefill_layout_short.wav' `
  -LogPath 'tmp\alia-real-smoke\bnb4_prefill_layout_short.log' `
  -RequestText '请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Expected: `ALIA_REAL_MODEL_SMOKE_PASS`, `foreground_lora_applied=true`, no output audio corruption. Compare `foreground_profile_prompt_prefill_ms` to the prior normal-smoke value around `329 ms`.

- [ ] **Step 3: Commit the useful round**

If the benchmark and smoke are acceptable:

```powershell
git add CMakeLists.txt tools/perf/Bnb4PrefillBench.cpp src/ops/Bnb4BitLinear.cpp tests/ops/Bnb4BitLinearKernelTests.cpp docs/alia-engine/2026-07-06-bnb4-prefill-layout-plan.md
git commit -m "perf: benchmark BnB4 prefill layout"
```

If the optimization does not hold up, commit only the benchmark and plan, and leave the rejected kernel result in the final response.
