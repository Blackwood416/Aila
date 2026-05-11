---
name: decode-perf-optimize
description: Systematic SYCL kernel performance optimization for Aila decode. Use when asked to improve decode throughput, close gap to llama.cpp Vulkan, or optimize SYCL/NF4 GEMV kernels.
---

# Decode Performance Optimization Skill

This skill captures the methodology, patterns, and lessons learned during the Qwen3.5-4B BNB NF4 decode optimization on Intel Arc A770.

## Workflow

### 1. Establish Baseline

```
# llama.cpp Vulkan (reference)
./llama.cpp/build-vulkan/bin/Release/llama-bench.exe \
  -m ./models/Qwen3.5-4B-UD-Q4_K_XL.gguf -p 2048 -n 1024 -ngl 99

# Aila (target to optimize)
pwsh bench.ps1 -ModelDir "./models/<model>" \
  -PromptTokens 2048 -GenTokens 1024 -BenchIters 5 -WarmupIters 1
```

### 2. Profile Hotspots

```
AILA_PROFILE_Q35_DECODE=1 AILA_PROFILE_Q35_DECODE_EVERY=4 \
./build/Aila.exe -m "<model>" --bench --bench-pp 2048 --bench-tg 8 \
  --bench-iters 1 --bench-warmup 0
```

Profile stages: `linear_proj` → `linear_delta` → `linear_o` (DeltaNet), `full_qkv` → `attn` → `full_o` (GQA), `ffn_proj` → `ffn_act` → `down` (FFN), `post_attn` → `post_mlp` (RMS norms), `lm_head`.

**Important**: Profile timings include `ctx.synchronize()` which serialises the pipeline. Real decode time (without profiling) is ~2× faster. Use the profile for RELATIVE comparison, not absolute tok/s.

### 3. Identify Targets

Rank stages by ms, divide by layer count for per-layer cost. Focus on the largest contributors.

### 4. Apply Optimizations

See "Proven Optimizations" below. Always build → smoke test → profile → benchmark. Revert if regressed; analyse why before giving up (some regressions reveal hardware constraints).

### 5. Verify

```bash
pwsh build.ps1
echo -e "What is 2+2?\n/quit\n" | ./build/Aila.exe -m "<model>" --max-tokens 32 --greedy --no-stream 2>/dev/null
pwsh bench.ps1 -ModelDir "<model>" -PromptTokens 2048 -GenTokens 1024 -BenchIters 3 -WarmupIters 1
```

## Proven Optimizations

### Pattern 1: vec8 + FMA for Memory-Bound GEMV Kernels (+6.8%)

**File**: `src/ops/Bnb4BitLinear.cpp` — `packed_nf4_gemv_bf16` SG16 path (`f94af7d`)

Replace 8 individual `input_ptr[...]` reads with one `vec<bf16,8>` load + `sycl::fma` chain. Pre-compute `qmap[nybble]*am` inline in the FMA to avoid 8 named float temporaries. GEMV ops 8-11% faster.

### Pattern 2: vec8 for Elementwise Kernels (+1.4%)

**File**: `src/ops/NormOps.cpp` — `fused_add_rms_norm` (`ea1e1f2`)

`vec<bf16,8>` loads for input, residual, weight tensors in RMS norm. For 1024-hidden path reduce WG from 256→128.

### Pattern 3: Kernel Fusion Without SLM Barriers (+1.6%)

**File**: `src/ops/Bnb4BitLinear.cpp` — `packed_nf4_gemv_gate_up_swiglu` (`3e2531e`)

Fused gate+up GEMV + SiLU. Removed SLM input caching (barrier overhead); SiLU computed AFTER subgroup reduction. Wired in `Qwen35HybridBnb4Backend.cpp` decode FFN path. Eliminates 32 kernel launches/token.

### Pattern 4: Compiler Flags (+3.7%, zero code changes)

**File**: `CMakeLists.txt` (`f842df6`)

Only ONE flag actually works on Windows icx-cl:
- `-fsycl-default-sub-group-size 16` — match DG2 native SIMD width (+3.7% decode)

All `-Xsycl-target-backend` flags (`-cl-fast-relaxed-math`, `-cl-mad-enable`, `-ze-opt-large-register-file`, `-force_stos_opt`) are IGNORED on icx-cl (produce "unused" warnings). AOT (`-fsycl-targets=spir64_gen`) causes runtime crash. `-ftarget-register-alloc-mode=large` also unused.

### Pattern 5: Barrier Reduction in JM Attention (minor, no regression)

**File**: `src/ops/AttentionOps.cpp` (`898f5df`)

Pre-load entire Q vector (256 bf16 = 512 bytes) into SLM once per JM tile instead of per-K-block. Eliminates 15 barriers per tile. attn: 2.60→2.53ms.

### Pattern 6: vec8 V Accumulation in Attention Phase 2 (minor)

**File**: `src/ops/AttentionOps.cpp` (`e9918ea`)

vec8 loads for V cache reads; probability shared across 8 output dims. No regression but limited impact (softmax reductions dominate Phase 2).

### Pattern 7: Inline dequant into FMA (code quality, flat benchmark)

**File**: `src/ops/Bnb4BitLinear.cpp` (`2b0b79f`)

Eliminate 8 intermediate float variables (d0-d7). Compiler already reuses registers; benchmark flat but cleaner code.

### Pattern 8: static env var caching (CPU overhead reduction)

**Files**: `src/models/Qwen35HybridBnb4Backend.cpp`, `Qwen35HybridTextBackend.cpp` (`997fbae`)

10 `env::read_*` calls per token (each does malloc+atoi+free on Windows). Convert to `static const` locals. GPU-dominated benchmark flat but eliminates unnecessary syscalls.

## Regressed Optimizations (Do Not Attempt)

| Optimization | Regression | Root Cause |
|---|---|---|
| SLM input caching in GEMV | +15ms | Barrier sync; 16 lanes × 4 SLM banks = 64 accesses onto 32 banks → 2× bank conflict |
| Blocked weight layout GEMV | +15ms | Transposed layout scatters cache lines |
| oneDNN cached dequant | +5ms | bf16 weights use 4× VRAM; oneDNN for batch>1 |
| JM bf16 FFN (bypass NF4) | +4ms | bf16 reads 4× more weight data |
| uint64 weight loads | +3ms | Intel SIMD prefers uint32 |
| SiLU inner-loop fusion | +2ms | exp/div in memory-bound inner loop |
| Cooperative GEMV (SLM input + 2 rows/sg) | +4.7ms | SLM bank conflicts; occupancy loss for small out_features |
| 2 rows/subgroup (runtime threshold) | -1.2 tok/s | Loop overhead from runtime rows_per_sg variable |
| Linear delta loop fusion | +0.5ms | Recomputing decay costs more than saved S writes |
| 2× GEMV unroll | +0.7ms | Register pressure spills |
| RoPE pre-compute + SLM freq | +3ms | USM malloc+memcpy overhead |
| vec_u32x2 wider GEMV loads | flat | Double register pressure cancels ILP gain |

## Key Architectural Insights

### NF4 Bandwidth Advantage Is Fundamental
NF4: 0.5 bytes/weight. bf16: 2 bytes/weight. Any bf16 path reads 4× more data → always slower for GEMV.

### Memory-Bound vs Compute-Bound
GEMV inner loop is ~85% memory-wait. Adding ANY compute (exp, div, extra loads) to inner loop = direct latency increase.

### SLM Bank Conflicts
A770 has 32 SLM banks (4-byte granularity). vec8 load = 16 bytes = 4 banks per lane. 16 lanes × 4 banks = 64 accesses onto 32 banks → 2× bank conflict. SLM caching of input vector is WORSE than L1-cached global reads.

### ~290 Kernel Launches per Token
Each `queue::submit()` ≈ 100μs CPU overhead. Fusion is the only way to reduce this, but most fusions add memory/compute overhead that outweighs launch savings.

### Profiling Overhead
Decode profile with `AILA_PROFILE_Q35_DECODE=1` adds `ctx.synchronize()` after each stage → serialises the async pipeline. Actual benchmark (without profiling) is ~2× faster. Use profile only for RELATIVE comparison.

### icx-cl Flag Support
On Windows, `-Xsycl-target-backend` flags are silently ignored. The ONLY GPU tuning flag that works is `-fsycl-default-sub-group-size 16`.

### Decode Path Structure
~290 kernel launches/token across 32 layers. Each launch is a `queue::submit()` → ~29ms CPU overhead (overlapped with GPU execution in in-order queue). Further fusion faces barrier/compute trade-offs.

## Final Performance (pp2048 tg1024, Arc A770)

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| Prefill | 1606 tok/s | 1637 tok/s | +1.9% |
| Decode | 50.45 tok/s | 57.9 tok/s | +14.8% |
| vs llama.cpp Vulkan | -16.4% | -4.0% | gap narrowed 12.4pp |

## Commit History

```
f94af7d perf: vec8 input loads + sycl::fma in packed_nf4_gemv_bf16 decode path
ea1e1f2 perf: vec8 loads in fused_add_rms_norm decode path
3e2531e perf: rewrite packed_nf4_gemv_gate_up_swiglu with vec8+FMA, no SLM
997fbae perf: cache env var reads as static locals in hybrid backend forward()
2b0b79f perf: inline NF4 dequant into FMA chain in packed_nf4_gemv_bf16
b874c82 perf: add GPU compiler flags (subgroup size, fast math, large GRF)
f842df6 perf: add default sub-group size 16 compiler flag (clean, no warnings)
898f5df perf: pre-load entire Q vector into SLM in JM attention decode
e9918ea perf: vec8 V accumulation in JM attention decode Phase 2
```

## Quick Verification

```bash
pwsh build.ps1
echo -e "What is 2+2?\n/quit\n" | ./build/Aila.exe -m "<model>" --max-tokens 32 --greedy --no-stream 2>/dev/null
pwsh bench.ps1 -ModelDir "<model>" -PromptTokens 2048 -GenTokens 1024 -BenchIters 3 -WarmupIters 1
```
