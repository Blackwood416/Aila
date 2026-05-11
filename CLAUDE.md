# Aila Project Guide

Aila is a high-performance SYCL-based LLM inference engine optimized for Intel Arc GPUs. It supports bitsandbytes NF4-quantized models (Qwen3, Qwen3.5 Hybrid).

## Build

```bash
pwsh build.ps1   # Windows + Intel oneAPI DPC++
```

Requires: Intel oneAPI Base Toolkit (compiler + oneDNN), CMake ≥ 3.24.

## Verify

```bash
# Smoke test
echo -e 'What is 2+2?\n/quit\n' | ./build/Aila.exe -m '<model>' --max-tokens 32 --greedy --no-stream 2>/dev/null

# Benchmark
pwsh bench.ps1 -ModelDir '<model>' -PromptTokens 2048 -GenTokens 512 -BenchIters 3 -WarmupIters 1
```

## Architecture

### Quantization

NF4 (bitsandbytes 4-bit NormalFloat) with per-block absmax. Weights stored as packed uint8 (2 × 4-bit values per byte) with float absmax per block. Quant map is a 16-element float LUT for non-linear dequant.

The NF4 format is ~0.5 bytes/weight vs bf16's 2 bytes/weight — a 4× bandwidth advantage that is **fundamental** to Aila's performance. Any bf16 weight path is always slower for memory-bound GEMV.

### Kernel Architecture

- **Decode GEMV** (`packed_nf4_gemv_bf16`): Sub-group cooperative vec8+FMA with SG=16. Each SG processes one output row. 16 SGs per work-group (WG=256). Inner loop: uint32 load (4 packed bytes → 8 NF4 values) + vec8 bf16 input load + 8× inline dequant + FMA.

- **Prefill GEMM** (`packed_nf4_gemm_bf16`): Tiled GEMM with BM=128, BN=128, BK=128. SLM staging for A (input) and B (packed weights). Inline NF4 dequant within the inner compute loop.

- **Attention** (decode): Joint Matrix (XMX) QK^T for head_dim=128 linear attention. Phase 1: JM QK^T → Phase 2: softmax + vec8 V accumulation. For head_dim=256 full attention: exact partial_merge with per-tile softmax stats + cross-tile merge.

- **FFN**: Fused gate+up GEMV + SiLU in one kernel (`packed_nf4_gemv_gate_up_swiglu`). No SLM barriers; SiLU computed after sub-group reduction.

### Model Backends

- `Qwen35HybridBnb4Backend`: Qwen3.5 Hybrid with BNB NF4 quantization. Alternating full attention (head_dim=256, GQA) and linear attention (DeltaNet, head_dim=128) layers.
- `Qwen3Bnb4Backend`: Qwen3 dense with BNB NF4. Standard GQA attention.
- `Qwen35HybridTextBackend`: Qwen3.5 Hybrid with bf16 weights (no quantization). Used for vision encoder.

### Dependency: oneDNN

oneDNN provides XMX-accelerated matmul for prefill GEMM and the bf16 decode fallback. Integrated via `dnnl::sycl_interop` for zero-copy USM memory sharing.

### SYCL Runtime

In-order SYCL queue (`sycl::property::queue::in_order()`). Kernels execute in submission order. Profiling (`ctx.synchronize()`) serializes the async pipeline and should only be used for relative comparison.

## Coding Conventions

- Kernel functions: `snake_case` in anonymous namespace
- Class methods: `PascalCase` with `lowerCamelCase` members
- SYCL bf16 alias: `using bf16 = sycl::ext::oneapi::bfloat16;`
- SLM declarations: `sycl::local_accessor<T, 1> name(range, cgh);`
- Sub-group size: `[[sycl::reqd_sub_group_size(16)]]` on kernel lambdas
- Env vars: `static const` locals read once via `aila::env::read_*`
- Profiling: `AILA_LOG_INFO` with structured format, profile stages via `time_stage()`

## Known Regressions

Do not attempt these optimizations (documented in `.claude/skills/decode-perf-optimize.md`):
- SLM input caching in GEMV (bank conflicts: 32 banks can't serve 16 lanes × vec8)
- bf16 weight path for decode (4× bandwidth loss vs NF4)
- Blocked weight layout GEMV (+15ms regression)
- Cooperative GEMV with 2 rows/subgroup
- uint64 weight loads (Intel SIMD prefers uint32)
- QMAP via private registers instead of SLM (45 tok/s vs 58)
- Fused residual add in GEMV (-1.5%)

## Key Files

| File | Purpose |
|------|---------|
| `src/ops/Bnb4BitLinear.cpp` | NF4 GEMV/GEMM kernels |
| `src/ops/AttentionOps.cpp` | All attention variants (JM, baseline, partial_merge) |
| `src/ops/NormOps.cpp` | RMS norm, fused add+norm |
| `src/ops/ElementwiseOps.cpp` | SiLU, SwiGLU, RoPE, residual add |
| `src/models/Qwen35HybridBnb4Backend.cpp` | Qwen3.5 Hybrid BNB backend |
| `CMakeLists.txt` | Build config, compiler flags |
| `docs/Environment_Variables.md` | All AILA_* env vars |
| `docs/perf-tuning.md` | Performance tuning guide |
| `.claude/skills/decode-perf-optimize.md` | Optimization history and methodology |
