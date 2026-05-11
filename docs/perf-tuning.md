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

## Benchmark Methodology

1. **Always warm up**: First iteration is up to 10% slower due to GPU cold state and oneDNN JIT compilation.
2. **Use 3+ iterations**: Average across iterations to smooth thermal/driver variance.
3. **Close other GPU applications**: Browser GPU acceleration, video playback, etc. can steal GPU time.
4. **Let GPU cool between runs**: Sustained benchmark can cause thermal throttling on consumer GPUs.
5. **Profile for diagnosis, benchmark for numbers**: Never report profile timings as real performance.
