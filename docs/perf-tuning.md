# Performance Tuning Guide

## Benchmark Quick Start

```bash
pwsh build.ps1
pwsh bench.ps1 -ModelDir '<model>' -PromptTokens 2048 -GenTokens 512 -BenchIters 3 -WarmupIters 1
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **pp** (prefill) | Prompt processing throughput in tok/s |
| **tg** (decode) | Token generation throughput in tok/s |
| **ttft** | Time To First Token ≈ prefill time + one decode step |
| **stddev** | Standard deviation across benchmark iterations |

## Profiling vs Real Decode

The built-in profiler (`AILA_PROFILE_Q35_DECODE=1`) adds `ctx.synchronize()` after each pipeline stage, serializing the async GPU pipeline.

- **Profiled decode time is ~2.5× slower** than real decode
- **Profiled prefill time is ~1.3× slower**
- Use profiling ONLY for relative comparison between stages, not absolute numbers

### Decode Profile

```
AILA_PROFILE_Q35_DECODE=1 AILA_PROFILE_Q35_DECODE_EVERY=4 \
./build/Aila.exe -m "<model>" --bench --bench-pp 2048 --bench-tg 32 \
  --bench-iters 1 --bench-warmup 0
```

### Prefill Profile

```
AILA_PROFILE_Q35_PREFILL=1 AILA_PROFILE_Q35_PREFILL_EVERY=1 \
./build/Aila.exe -m "<model>" --bench --bench-pp 2048 --bench-tg 1 \
  --bench-iters 1 --bench-warmup 0
```

## Key Environment Variables

### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `AILA_BNB4_FUSED_PREFILL` | 1 | Enable fused NF4 dequant+GEMM for prefill (faster than oneDNN) |
| `AILA_BNB4_GEMV_WG` | 256 | Override SG16 GEMV work-group size (min 32) |
| `AILA_FUSE_RESIDUAL_ADD` | 0 | Fuse residual +=hidden into O-proj/down GEMV (experimental, -1.5%) |

### Attention

| Variable | Default | Description |
|----------|---------|-------------|
| `AILA_ATTN_JM` | 1 | Joint Matrix attention (0=off, 1=auto, 2=force) |
| `AILA_ATTN_JM_TILE` | auto | JM tile selection (0=1×8×16, 1=32×32×16, 2=8×8×16) |
| `AILA_ATTN_DECODE_WG` | 512 | Phase 2 softmax/V work-group size |
| `AILA_ATTN_DECODE_WINDOW` | 0 | Sliding window attention length (0=full context) |

### Pipelining

| Variable | Default | Description |
|----------|---------|-------------|
| `AILA_STREAM_OUTPUT` | auto | Stream tokens as they are generated |
| `AILA_LOG_LEVEL` | info | Log verbosity (verbose/debug/info/warning/error) |

## Known-Good Configurations

### Intel Arc A770 + Qwen3.5-4B BNB NF4

```
AILA_ATTN_JM=1          # JM XMX attention enabled
AILA_BNB4_FUSED_PREFILL=1  # NF4 fused GEMM prefill
```
- Prefill: ~1637 tok/s
- Decode: ~57.9 tok/s

### Intel Arc A770 + Qwen3.5-0.8B BNB NF4

```
AILA_ATTN_JM=1
AILA_BNB4_FUSED_PREFILL=1
```
- Higher throughput due to smaller model size.

### Intel Arc A770 + DiffusionGemma-26B-A4B-it BNB NF4

The A770 dispatch uses `BM=64`, `BN=64`, and `BK=64` for blocksize-64
prefill-shaped NF4 GEMM. The smaller K tile reduces per-work-group SLM from
about 20.5 KiB to 10.5 KiB. Other GPUs and larger NF4 block sizes retain their
existing dispatch.

The vectorized NF4 GEMM (`packed_nf4_gemm_bf16_tiled_v2`) stages the
dequantized bf16 B tile in SLM and uses `vec8` inner-loop loads. On A770 the
blocksize-64 dispatch uses `BM=32`, `BN=128`, `BK=64`, `TM=2`, `TN=8`, and a
16x16 work-group. The old `BM=64`/`BN=64` scalar geometry remains available as
the fallback (`AILA_BNB4_GEMM_V2=0`) and is bitwise identical on real
DiffusionGemma weights for every `BN>=32` geometry swept; `BN=16`/`TN=1`
geometries were rejected because the device compiler reassociates their k
accumulation under `-ffast-math`.

Validated on driver `32.0.101.8860`, oneAPI `2026.1`, and oneDNN `3.11.2`, with
seed 42, a 256-token canvas, and 25 denoising steps. Each sample was a separate
process and tensor dumping plus synchronizing profiling were disabled. Aila
does not require `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS`; that variable is only
for the separate Python/Transformers reconstruction and reference workflow.

| A770 K tile | 25-step wall samples (ms) | Median (ms) |
|-------------|---------------------------|-------------|
| BK128 baseline | 328312.37, 329871.85, 330980.95 | 329871.85 |
| BK64 | 260323.84, 257500.51, 258545.84 | 258545.84 |
| BK64 + v2 GEMM | 139772.34, 141609.89, 140662.14 | 140662.14 |

BK64 reduces median generation wall time by 21.6% (1.276x speedup); the v2
GEMM reduces it by a further 45.6% over BK64 (1.84x total vs the BK128
baseline). Both paths produced the same 256 committed token IDs; the
1024-byte int32 token stream had SHA256
`A4F674A0CB0C81E0682FC4F7B490251BEB9011DDF749475F216087B5761D6CD1`.
Model-load memory remained 13665.85 MiB current and 13799.81 MiB peak. After
allocating the decoder's runtime buffers, memory was 14673.44 MiB current and
14705.43 MiB peak; the v2 GEMM adds no global or persistent allocation.

20-sample operator medians for the v2 GEMM on the real decode shapes
(blocksize 64):

| Shape `M x N x K` | BK64 median (ms) | v2 median (ms) |
|-------------------|------------------|----------------|
| `256 x 2816 x 2112` | 20.60 | 5.95 |
| `256 x 2112 x 2816` | 27.50 | 7.74 |
| `256 x 1024 x 2816` | 27.68 | 7.74 |
| `256 x 4224 x 2816` | 45.22 | 11.71 |

The `AILA_BNB4_GEMM_V2=0` environment variable selects the original scalar
geometry for A/B comparison. `AILA_BNB4_GEOM_BENCH=1` in
`AilaBnb4BitLinearKernelTests` runs the geometry sweep, and
`AILA_BNB4_ROUND_PROBE=1` exercises device bf16 rounding/multiply edge cases.

### vLLM-style init warmup for DiffusionGemma

The engine now auto-runs a vLLM-style dummy pass at init for DiffusionGemma
(`AILA_INIT_WARMUP` defaults to auto). Like vLLM's `_dummy_run`, it runs the
full model once with the real canvas shape so SYCL kernel JIT, oneDNN
primitive creation, buffer allocation, and GPU clock ramp happen before the
first real request instead of inside `generate()`. The warmup consists of a
dummy 8-token prompt encode, one dummy 256-token `diffusion_decode_canvas`
pass with zero soft self-conditioning, and one soft-embedding call. Its
output is discarded and the backend is reset, so the committed token stream
is unchanged.

Three fresh-process 25-step runs on driver `32.0.101.8860`:

| Mode | Warmup (ms) | Generate (ms) |
|------|-------------|---------------|
| No warmup (`AILA_INIT_WARMUP=0`) | 0 | 143284, 149916, 149910, 151797 (median 149913) |
| Default warmup (1 pass) | 17903, 17788, 18009 (median 17903) | 129230, 124560, 132580 (median 129230) |

Warmup moves the one-time JIT/primitive cost out of `generate()`, cutting the
median generation wall time by 13.8% (149.91 s to 129.23 s) while the total
init+warmup+generate time stays neutral-to-better. Every run preserved the
committed 256-token stream (SHA256 `A4F674A0...`). `AILA_DG_WARMUP_PASSES`
controls the number of dummy decoder passes and `AILA_DG_WARMUP_CANVAS`
overrides the dummy canvas length; a smaller canvas only saves compute
because the JIT cost is shape-independent.

## Benchmark Methodology

1. **Always warm up**: First iteration is up to 10% slower due to GPU cold state and oneDNN JIT compilation.
2. **Use 3+ iterations**: Average across iterations to smooth thermal/driver variance.
3. **Close other GPU applications**: Browser GPU acceleration, video playback, etc. can steal GPU time.
4. **Let GPU cool between runs**: Sustained benchmark can cause thermal throttling on consumer GPUs.
5. **Profile for diagnosis, benchmark for numbers**: Never report profile timings as real performance.
