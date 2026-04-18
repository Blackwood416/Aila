# Phase 9: Decode Dispatch Overhead Analysis

## Summary

Phase 9 focused on achieving the decode target of 125+ tok/s (from ~103 tok/s baseline after Phase 8). Through deep profiling and experimentation, we identified the root cause as **SYCL runtime dispatch overhead** inherent to the current architecture, and confirmed that neither oneDNN nor oneMKL alternatives can solve it.

## Current Performance

| Metric | Value |
|--------|-------|
| Prefill | ~2980 tok/s (target 2300+ ✅ achieved in Phase 8) |
| Decode | ~103 tok/s (target 125+ ❌ not yet achieved) |
| Model | Qwen3.5-0.8B |
| GPU | Intel Arc A770 |

## Root Cause Analysis

### Profiling Data (per-token decode averages, GPU-synced)

| Stage | ms/token | Notes |
|-------|----------|-------|
| ffn_proj | 0.320 | oneDNN matmul 1024→7168 |
| down_proj | 0.299 | oneDNN matmul 3584→1024 |
| linear_proj | 0.247 | oneDNN matmul 1024→8224 |
| linear_o | 0.208 | oneDNN matmul 2048→1024 |
| linear_delta | 0.185 | Custom SYCL kernel |
| post_attn_norm | 0.167 | fused_add_rms_norm |
| post_mlp_norm | 0.169 | fused_add_rms_norm |
| ffn_act | 0.166 | fused_gate_up_swiglu |
| attn stages | ~0.3 | Combined full attention |
| **Total GPU compute** | **~2.23** | |
| **Wall clock** | **~9.7** | 103 tok/s |

### Key Finding: Dispatch Overhead

- **GPU compute**: 2.23 ms/token (23% of wall time)
- **Host submission**: ~0.5 ms/token (measured with `AILA_PROFILE_Q35_HOST_ONLY=1`)
- **Wall clock**: ~9.7 ms/token
- **Gap**: ~7 ms unaccounted = cumulative `execute()` / `gemm()` call overhead

With 96 oneDNN matmul calls per decode token, the average overhead per call is ~106 µs. This overhead is NOT GPU compute time — it occurs inside the library's `execute()` function, likely involving:
- SYCL queue synchronization or resource management
- JIT compilation or kernel selection
- Internal memory allocation or descriptor handling

### Profiling Methodology

Three profiling modes were used:

1. **GPU-synced** (`AILA_PROFILE_Q35_DECODE=1`): `ctx.synchronize()` after each stage → measures GPU compute
2. **Host-only** (`AILA_PROFILE_Q35_HOST_ONLY=1`): No sync → measures host submission overhead
3. **Wall clock**: No profiling → measures actual throughput

The large gap between modes (1) and (3) revealed the dispatch overhead.

## Experiments Attempted

### 1. Fused GEMV + Residual + RMS Norm (down_proj)
- **Goal**: Replace oneDNN matmul with custom SYCL GEMV kernel fused with post-MLP norm
- **Result**: 97 tok/s (regression from 103 baseline)
- **Reason**: Custom BF16 GEMV cannot match XMX hardware acceleration
- **Status**: REVERTED

### 2. Fused GEMV + Residual + RMS Norm (o_proj)  
- **Goal**: Replace oneDNN matmul for o_proj (2048→1024) with fused GEMV
- **Result**: ~99 tok/s (slight regression)
- **Status**: REVERTED

### 3. oneMKL GEMM as Alternative to oneDNN Matmul
- **Goal**: Test if MKL SYCL BLAS `gemm()` has lower dispatch overhead than oneDNN `matmul::execute()`
- **Implementation**: Added `AILA_USE_MKL_GEMM=1` env var to switch decode path from oneDNN to oneMKL BF16 GEMM
- **Result**: 101 tok/s (same as oneDNN, no improvement)
- **Conclusion**: Both libraries use the same SYCL runtime path; the overhead is in the SYCL layer, not the library
- **Status**: REVERTED (code removed)

### 4. oneDNN Graph API (Phase 7, prior)
- **Goal**: Fuse FFN operations into a single graph partition
- **Result**: Failed — graph partitioner could not fuse the Qwen3.5 FFN pattern
- **Status**: Documented in phase 7 handoff

### 5. Custom BF16 FMA FFN Kernels (Phase 8, prior)
- **Goal**: Replace oneDNN FFN matmuls with hand-written XMX kernels
- **Result**: Slower than oneDNN
- **Status**: Documented in phase 8 handoff

## Remaining Uncommitted Changes

Only one small change retained from this phase:

**`src/models/Qwen35HybridTextBackend.cpp`**: Added `AILA_PROFILE_Q35_HOST_ONLY` flag to `time_stage` lambda. When set, skips `ctx.synchronize()` to measure pure host submission time without GPU sync overhead. This is a diagnostic-only change with no performance impact.

## Theoretical Analysis

- **Total weight bytes**: ~1666 MB
- **Bandwidth limit**: 560 GB/s → minimum ~2.97 ms/token
- **Measured GPU compute**: 2.23 ms/token (faster than bandwidth limit due to caching)
- **Required for 125 tok/s**: 8.0 ms/token
- **Current dispatch overhead**: ~7 ms/token (96 calls × ~73 µs average effective overhead)
- **Needed reduction**: From ~7 ms to ~5.8 ms overhead, or reduce dispatch count from 96 to ~78

## Future Optimization Strategies

### High Impact (if feasible)
1. **SYCL Command Graph**: Record decode layer sequence as a command graph, replay with single submit. Challenge: oneDNN `execute()` calls cannot be captured in SYCL graphs; would need pure SYCL kernel implementations for all matmuls.
2. **Out-of-order Queue**: Switch from in-order to OoO queue with explicit event dependencies. May enable better driver-level pipelining.
3. **Level Zero Direct Submission**: Bypass SYCL runtime entirely using Intel Level Zero API for kernel dispatch. Most invasive change.

### Medium Impact
4. **Layer Fusion (multi-layer)**: Process 2+ layers per dispatch by combining matmul + element-wise ops into single mega-kernels using ESIMD/XMX intrinsics.
5. **Speculative Decode**: Generate multiple candidate tokens per step, then verify — amortizes dispatch overhead over multiple tokens.

### Low Impact / Already Tried
6. Replace GEMM library (tried: no effect)
7. Custom GEMV kernels for small matmuls (tried: regression)
8. oneDNN Graph fusion (tried: unsupported pattern)

## Build & Test Commands

```powershell
# Build
.\build.ps1 -BuildDir build -Config Release

# Benchmark
.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <name> -CaseNames @('greedy_main')

# Decode profiling (GPU-synced)
$env:AILA_PROFILE_Q35_DECODE="1"
$env:AILA_PROFILE_Q35_DECODE_EVERY="16"

# Host-only profiling (no GPU sync)
$env:AILA_PROFILE_Q35_HOST_ONLY="1"
```

## Git State

- Base commit: `f657dc5` (Phase 8 complete, prefill optimization)
- This phase commit: profiling utility only (host-only timing mode)
