# Aila Project Guide (Post-Refactor Architecture)

Aila is a high-performance SYCL-based LLM inference engine optimized for Intel Arc GPUs (DG2 / Alchemist, e.g. A770). It supports bitsandbytes NF4-quantized and BF16 models (Qwen3 dense, Qwen3.5 Hybrid) with shared-model multi-session execution, continuous batching, and isolated C++ HTTP/SSE serving.

---

## 1. Quick Start & Build

### Requirements
- **OS**: Windows 10/11 x64
- **Toolchains**:
  - Intel oneAPI Base Toolkit 2026.1 (DPC++ compiler `icx-cl`, oneDNN 3.11, oneTBB, SYCL runtime)
  - Microsoft Visual Studio 2022 / Build Tools (MSVC v143/v145, C++17)
  - CMake >= 3.24, Ninja >= 1.12, Python >= 3.10, PowerShell 7 (`pwsh`)

### Building the Primary Runtime (Worker, Lib, Tests)
```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
```
Artifacts produced in `build-refactor/`:
- `Aila.exe`: Standalone CLI inference binary.
- `AilaWorker.exe`: IPC v2 inference worker host.
- `AilaShared.dll`: Public C ABI export (`aila_api.h`) with legacy adapter.
- `AilaInference.dll`: Core async C ABI export (`aila_inference.h`).
- 54 CTest test executables.

### Building the HTTP/SSE Inference Server
```powershell
pwsh -NoProfile -File tools/server/build.ps1 -SkipDependencies
```
Artifacts produced in `build-refactor-server/`:
- `AilaServer.exe`: High-concurrency Drogon-based HTTP/SSE OpenAI-compatible server (pure MSVC C++17, zero SYCL/oneDNN linkage).

---

## 2. System Architecture

```
+-------------------------------------------------------------+
|                  Client / OpenAI API / CLI                  |
+-------------------------------------------------------------+
                              | HTTP / SSE (Port 8080)
+-------------------------------------------------------------+
|               AilaServer.exe (Pure MSVC C++17)              |
|  - Drogon HTTP / SSE Server        - ChatCodec / Jinja      |
|  - Tokenizer / Thinking Stripper   - host::InferenceClient  |
+-------------------------------------------------------------+
                              | Win32 Named Pipes (IPC v2)
+-------------------------------------------------------------+
|             AilaWorker.exe (Intel oneAPI DPC++/SYCL)        |
|  - worker::InferenceWorker         - inference::Scheduler   |
|  - DeviceRuntime (USM Ledger)      - LoadedModel (Weights)  |
|  - Sessions (KV Cache / DeltaNet)  - BatchDecodeExecutor    |
+-------------------------------------------------------------+
                              | SYCL In-Order Queue
+-------------------------------------------------------------+
|                 Intel Arc GPU (e.g. A770 16GB)              |
|  - Packed NF4 GEMV / GEMM         - Multi-row Attention     |
|  - XMX Joint Matrix Acceleration  - oneDNN MatMul Fallback  |
+-------------------------------------------------------------+
```

### Resource Ownership Hierarchy
| Component | Responsibility & Lifetime |
|---|---|
| `DeviceRuntime` | Manages SYCL queue/device, oneDNN engine, and explicit USM allocation ledger (`device_budget_`, `host_budget_`). |
| `LoadedModel` | Read-only model configuration, fused NF4/BF16 weights, and static layer plans. Loaded once; shared across all sessions. |
| `Session` | Per-conversation state: independent KV caches (Full Attention), DeltaNet recurrent matrix & conv history (Linear Attention), and checkpoint snapshots. |
| `Request` | Single generation lifecycle: input prompt token IDs, generation limits, sampling parameters (RNG seed, temp, top-k/p, penalties), thinking budget controller, and stream cancellation flags. |
| `Scheduler` | Round-robin token/chunk scheduler running on a dedicated worker thread. Handles dynamic continuous batching, admission limits, backpressure, and fault propagation. |

---

## 3. Kernel & Compute Architecture

### Quantization (BNB NF4)
- NormalFloat 4-bit weights packed into `uint8` (2 × 4-bit nibbles per byte) with per-block float absmax and a 16-element float dequantization LUT.
- 0.5 bytes/weight provides a fundamental 4× memory bandwidth advantage over BF16 for memory-bound decode GEMV.

### Continuous Decode Batching (P7)
- **Batch Bounds**: Configurable via `--decode-batch-size 1..4` (default 1).
- **Multi-Row Decode Projections**:
  - NF4: `run_decode_qkv_rows`, `run_decode_o_proj_rows`, `run_decode_ffn_rows` using subgroup-cooperative GEMV (1 output row per subgroup).
  - BF16: `run_decode_jm_matvec_rows` and `run_decode_ffn_gate_up_swiglu_rows` reducing launch overhead 4× and eliminating duplicate weight loads (+57% throughput at B=4).
- **Multi-Row Decode Attention (`attention_decode_rows`)**:
  - `head_dim == 256` below threshold: Multi-row baseline kernel.
  - `head_dim == 128 / 256` at or above threshold: Multi-row exact-partial-merge kernel with private `[4, head, tile, stride]` scratch blocks per sequence.
  - `head_dim == 128` below 128 tokens: Fused multi-row Joint-Matrix kernel (`attention_decode_rows_joint_matrix_impl`) executing QKᵀ, Softmax, and V-accumulation in SLM without global DRAM scores traffic.
  - Window/sink or mixed-threshold rows gracefully fall back to per-row dispatch.

### Model Backends
- `Qwen3Bnb4Backend`: Dense Qwen3 models (0.6B, 4B) with BNB NF4 weights and GQA full attention.
- `Qwen35HybridBnb4Backend`: Qwen3.5 Hybrid models (0.8B, 4B) with alternating GQA full attention (`head_dim=256`) and DeltaNet linear attention (`head_dim=128`).
- `Qwen35HybridTextBackend`: Qwen3.5 Hybrid with unquantized BF16 weights.
- `Qwen4ExpBnb4Backend`: Qwen3.8-Flash-Next (48-layer Hybrid: 36 Gated DeltaNet + 12 Query-Sparse Attention, 512 routed NF4 experts top-8 + 1 shared expert, 4-stream Hyper-Connection, Layer-1 PLE, 2-tier `Qwen4ExpAdaptiveCache` + `Qwen4ExpHostArena` residency). See `docs/Qwen38FlashNext_A770_Handoff.md`.
- `DiffusionGemmaBnb4Backend`: DiffusionGemma 26B A4B MoE Block Diffusion Model.
- `Gemma4VisionEncoder` / `YOLO26`: Vision tower and object detection modules.

---

## 4. Chat Template & Reasoning Engine

- **Jinja2 Templating**: Native high-speed C++ Jinja engine (`src/templates/jinja/`) formatting prompts according to tokenizer chat templates.
- **Thinking Block Stripping**: `aila::chat::strip_thinking_blocks` automatically strips historical `<think>...</think>` blocks from external multi-turn history (`mm_history_`), preventing reasoning pollution across turns.
- **Thinking Budget Control**: `ThinkingBudgetController` monitors reasoning token generation against `max_thinking_tokens` / `thinking_budget`. When exhausted, it forcefully injects `</think>\n` to transition the model to content generation.

---

## 5. Verification & Testing Guide

### 0. Unified 3-Tier Verification Entry Point (`tools/verify-q4e.ps1`)
All incremental target builds, operator tests, CTest suites, and real-model acceptance runs for `Qwen4Exp` go through `tools/verify-q4e.ps1` (which automatically configures MSVC + Intel oneAPI library paths and sanitizes SYCL/Level Zero runtime environment variables):

```powershell
# Incremental build of specific target(s) only:
pwsh -NoProfile -File tools/verify-q4e.ps1 -Tier build -Targets Aila,AilaQwen4ExpKernelTests

# Tier 1 (Fast Inner Loop, ~2 min): Operator & kernel unit suites (Bnb4BitLinear, GDN, MoeExpertXmx, HostArena, ArenaVerify, ExpertReader)
pwsh -NoProfile -File tools/verify-q4e.ps1 -Tier fast

# Tier 2 (Pre-Commit Gate, ~6 min): Tier 1 + Qwen4ExpKernelTests + CTest 'qwen4_exp' (11 suites) + CTest 'cpu|abi' (20 suites)
pwsh -NoProfile -File tools/verify-q4e.ps1 -Tier ctest

# Tier 3 (Real-Model & Preset Promotion Gate, ~5 min): Real-model E2E tolerance (--model), preset interior-checkpoint parity (--preset-model), and 8K prefill benchmark
pwsh -NoProfile -File tools/verify-q4e.ps1 -Tier real
```

### 1. CPU & ABI Unit Tests (Fast CI)
```powershell
ctest --test-dir build-refactor -L 'cpu|abi' --output-on-failure
```

### 2. Full CTest Suite
```powershell
ctest --test-dir build-refactor --output-on-failure
```

### 3. Server Socket & Concurrency Tests (CPU Fake Worker)
```powershell
python tests/server/socket_tests.py --server build-refactor-server/AilaServer.exe --worker build-refactor/AilaFakeInferenceWorker.exe --mode fake --output tmp/socket-fake
python tests/server/batching_socket_tests.py --server build-refactor-server/AilaServer.exe --worker build-refactor/AilaFakeInferenceWorker.exe --output tmp/batch-fake
```

### 4. GPU Real HTTP & Concurrency Acceptance
```powershell
python tests/server/socket_tests.py --server build-refactor-server/AilaServer.exe --worker build-refactor/AilaWorker.exe --model models/qwen3-0.6B-bnb-nf4-offline --mode gpu --case incremental_utf8_json_sse --output tmp/gpu-http
python tests/server/batching_gpu_tests.py --server build-refactor-server/AilaServer.exe --worker build-refactor/AilaWorker.exe --backend qwen3_nf4 --model models/qwen3-0.6B-bnb-nf4-offline --output tmp/gpu-batching
```

### 5. Service Benchmark & DXGI Memory Profiling
```powershell
python tools/server/service_benchmark.py --server build-refactor-server/AilaServer.exe --worker build-refactor/AilaWorker.exe --model models/qwen3-0.6B-bnb-nf4-offline --output tmp/bench-probe --mode probe
```

---

## 6. Coding Conventions & Invariants

- **Kernel Functions**: `snake_case` inside anonymous namespaces.
- **Class Methods & Public Types**: `PascalCase` with `camelCase` private members.
- **SYCL Types**: `using bf16 = sycl::ext::oneapi::bfloat16;`.
- **Sub-group Attribute**: Explicit `[[sycl::reqd_sub_group_size(16)]]` or `[[sycl::reqd_sub_group_size(8)]]`.
- **Environment Variables**: Cached read-once static locals via `read_env_int_local` or `aila::env::read_*`.
- **Zero Allocations on Hot Path**: Decode loops must never issue dynamic USM allocations or rebind oneDNN descriptors. Scratchpads and intermediate buffers must be pre-allocated per execution slot.
- **Memory Safety & Cancellation**: Canceled requests must drain submitted GPU work before freeing USM; never free device pointers while kernels may still reference them.

---

## 7. Known Negative Optimizations (Do Not Attempt)

Documented regressions to avoid:
- **SLM input caching in GEMV**: Bank conflicts (32 banks cannot service 16 lanes × vec8).
- **BF16 weight path for decode GEMV**: Bandwidth loss (4× slower than packed NF4).
- **Blocked weight layout for GEMV**: Causes +15ms regression on DG2 memory controller.
- **2 rows per subgroup in GEMV**: Degrades register pressure and occupancy.
- **uint64 weight loads**: Intel SIMD hardware is heavily optimized for uint32 vector loads.
- **QMAP in private registers instead of SLM**: Reduces throughput from 58 to 45 tok/s.
- **Fused residual add in GEMV**: -1.5% due to register bloat.

---

## 8. Development, Hygiene & Anti-Entropy Specification

To keep the codebase lean, deterministic, and free of dead experimental baggage, all contributors and coding agents must follow these six rules:

1. **Mandatory Verification Tiers (`tools/verify-q4e.ps1`)**:
   - Run `-Tier fast` during inner-loop operator/kernel iteration.
   - Run `-Tier ctest` (`qwen4_exp` + `cpu|abi` 100% green) before every code commit.
   - Run `-Tier real` (real-model E2E tolerance + `--preset-model` checkpoint parity + `tools/prefill-bench.ps1` back-to-back A/B) before changing any production default or `Qwen38A770Preset.hpp` entry.
2. **Zero Dead-Experiment Retention (Test-Then-Prune)**:
   - Develop candidate kernels or schedules behind a default-OFF flag or inside the subsystem's existing `--bench` harness (`AilaBnb4BitLinearKernelTests`, `AilaGdnChunkTests`, `AilaMoeExpertXmxTests`, `AilaQsaAttentionProbe`).
   - Measure paired back-to-back A/B in a single session. If a candidate wins and passes Tier 2 + Tier 3, promote it and **delete the superseded branch**. If it is neutral or negative, **revert/delete it immediately**—never leave losing or superseded kernel modes parked behind environment flags in production files.
3. **No Ad-Hoc Probe Headers or One-Off Campaign Scripts**:
   - Do not add new `tests/models/*Probe.hpp` or `*Acceptance.hpp` files included into `Qwen4ExpToleranceTests.cpp`. Add unit/operator tests to the canonical test binary for that subsystem.
   - Do not commit one-off `.py` or `.ps1` campaign wrappers into `perf/qwen4exp/`. Temporary scripts, CSVs, and logs belong exclusively in gitignored `tmp/`.
4. **Single-Source Documentation (`docs/`)**:
   - `docs/Qwen38FlashNext_A770_Handoff.md` is the **single authoritative architecture and performance reference** for `Qwen4Exp`.
   - `docs/Environment_Variables.md` is the **single authoritative registry** of active environment variables.
   - Do not create chronological session diaries, process plans, or JSON result dumps in `docs/plans/`. Update the authoritative reference in place when architecture or scoreboard numbers change.
5. **Environment Variable & Preset Precedence Discipline**:
   - Effective precedence is strictly: `Explicit Environment Variable > Qwen38A770Preset.hpp > Code Default`.
   - Before modifying a code default, check whether `src/utils/Qwen38A770Preset.hpp` pins that variable. Any flag promotion or retirement must update the code default, `Qwen38A770Preset.hpp`, and `docs/Environment_Variables.md` in the same commit.
6. **Hardware & Numeric Safety Invariants**:
   - **VRAM Headroom**: Maintain >= 2.5 GB VRAM headroom on 16 GB Arc A770; never pre-allocate full-context scratch buffers at model load (`AILA_Q4E_RUNTIME_MIN_CAP` causes severe WDDM eviction stalls).
   - **Two-Level SLM Reductions**: Any two-level reduction staged through SLM must include an explicit `item.barrier(sycl::access::fence_space::local_space)` between the per-subgroup write and the second-level read.
   - **Single-Row vs. Multi-Row Symmetry**: Single-row and batched (`_rows`) entry points for the same operator must dispatch to numerically symmetric kernel variants so verification and interior-checkpoint canaries stay aligned.


