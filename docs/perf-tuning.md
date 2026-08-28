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

### Batched MoE transfers

The host-driven MoE path used one blocking h2d/d2h pair per expert group
(about 60 groups per layer), which measured ~51 ms/layer of transfer/sync
overhead on top of ~17 ms of dequant+matmul. The batched path
(`Bnb4GroupedMoE::forward_batched`, default on via `AILA_DG_MOE_BATCHED=1`)
keeps every per-expert oneDNN matmul byte-identical but gathers all groups
into one shared device buffer with a single flat gather kernel and copies
the combined per-expert outputs back with one d2h. Per-expert matmuls run on
offset views of the same buffer, so the committed arithmetic order and the
oneDNN kernels are unchanged; the MoE harness (`AILA_DG_MOE_HARNESS=1`)
confirmed zero bitwise mismatches versus the host path on every layer.

Fresh-process 25-step runs with warmup + batched MoE (seed 42):

| Mode | Warmup median (ms) | Generate median (ms) |
|------|--------------------|----------------------|
| Cold, host MoE | 0 | 149913 (4 runs) |
| Warmup, host MoE | 17903 (3 runs) | 129230 (3 runs) |
| Warmup + batched MoE | 16675 (3 runs) | 94071 (3 runs) |

Batched MoE cuts the median `generate()` time from 129.23 s to 94.07 s
(-27.2%) with a tighter spread (93.2-94.5 s vs 143-152 s cold). Runtime peak
memory after decoder buffers is 14733.99 MiB (+28.6 MiB vs the BK64 record,
inside the 256 MiB gate); all runs preserved SHA256 `A4F674A0...`.

The final build's three-run median with warmup + batched MoE is 94.98 s
(94.23/94.98/95.05 s). `AILA_DG_MOE_PACE` drains the queued oneDNN matmuls
every N expert groups (default 8). On this A770/Level-Zero stack, draining
less often than every group while `AILA_PROFILE_DG` adds per-phase syncs
deadlocks the GPU driver (all threads park in ntdll waits inside
`ze_intel_gpu64.dll`; the process exits `0xC0000005`), so the batched path
automatically drains every group under `AILA_PROFILE_DG`. Production is
unaffected and stable.

### 2026-08-27 incremental wins

- MoE dequant coalescing: consecutive work-items now map to consecutive
  expert rows so the bf16 stores are fully coalesced (the packed reads become
  strided, which is cheaper than striding 4x the bytes of stores).
- MoE split+SiLU fusion: the gate/up split and gelu kernels are one kernel
  reading the fused gate_up output directly (one launch and two intermediate
  buffers saved per expert group).
- GPU logits divide: `logits / temperature` now happens in place on the GPU
  (IEEE division, bitwise identical to the host loop), removing a 256 MB
  host->device round trip and a 67M-element host divide loop per step.

Three-run 25-step medians with warmup (seed 42): generate 88.45 s
(87.96/88.45/88.63 s), warmup 16.75 s, runtime peak 14733.99 MiB; all runs
preserve SHA256 `A4F674A0...`. `AILA_DG_STEP_TIMING=1` prints the unprofiled
per-step decode/transfer/host/soft breakdown.

Measured negative results (kept for reference):

- `packed_nf4_gemm_bf16_tiled_v3` (double-buffered A/B SLM, one barrier per K
  tile) is bitwise identical but slower than v2 on every geometry
  (32x128x64: 7.72 ms vs 5.99 ms; 32x128x32: 6.15 ms). The kernel is
  load/latency-bound, so the extra 40 KB SLM occupancy loss outweighs the
  prefetch overlap.
- The soft-embedding GEMM `256x262144x2816` runs ~300 ms in oneDNN
  `jit:gemm:any` on A770 regardless of weight tag (ab) or transposed ba
  orientation; only the K=2816 logits projection selects the fast systolic
  path. The K=262144 contraction has no oneDNN fast path on this stack.

### 2026-08-27 next-round A770 candidates

The DiffusionGemma decoder now has one persistent raw attention-scratch arena
(`q/k/v`, attention output, full K/V, mask, and scores). Each layer creates a
compact row-major view with its exact current geometry, so the full-attention
and sliding-attention layouts remain unchanged. `AILA_DG_REUSE_LAYER_SCRATCH`
defaults to `1`; setting it to `0` restores the per-layer allocation path.
After the reuse-only gate passed, `AILA_DG_REUSE_LAYER_SYNC=0` became the
production default and removes the layer-tail queue drain. Setting it to `1`
keeps the staged synchronization while diagnosing stale USM data. Both A/B
paths produced the same committed stream and no NaN/Inf values.

Fresh-process 25-step runs (seed 42, canvas 256, warmup enabled, no profile or
tensor dump) were interleaved as follows:

| Path | Samples (s) | Median (s) | Peak MiB |
|------|-------------|------------|----------|
| fallback: scratch alloc + old softmax | 77.27, 77.54, 77.50 | 77.50 | 14713.79 |
| default: persistent scratch + no layer sync + in-place softmax | 73.66, 73.67, 73.92 | 73.67 | 14715.93 |

The default candidate is 4.9% faster than this fresh fallback and 16.7% below
the handoff's 88.45 s baseline, staying well below the 84.03 s candidate line.
Every full run preserved the target committed-token SHA256
`A4F674A0CB0C81E0682FC4F7B490251BEB9011DDF749475F216087B5761D6CD1`.
The resident-memory delta was about 17 MiB in these runs (inside the
256 MiB gate); the exact amount depends on the allocator's prior per-layer
high-water mark.

The in-place softmax rounds logits to bf16 first, computes max and exp in a
parallel kernel, accumulates the denominator in the historical ascending
column order, and writes bf16 probabilities. It reuses the dead f32 logits
buffer, then leaves the oneDNN soft-embedding `ab` contraction unchanged.
`AILA_DG_SOFTMAX_INPLACE=0` restores the old row-serial implementation.
`AILA_DG_TOKEN_DUMP=<file>` records only the committed int32 stream, avoiding
the synchronization and tensor writes associated with `AILA_DG_DUMP_DIR`.
For direct row-0 probability/embedding A/B inspection, the debug-only
`AILA_DG_SOFTMAX_DUMP_DIR=<dir>` switch writes the original logits, bf16
probabilities, and soft embedding; it intentionally adds one synchronization
and does not enable decoder tensor dumps.

The B-only v2 SLM K-chunk-major swizzle is retained as an opt-in experiment
(`AILA_BNB4_GEMM_V2_SWIZZLE=1`) but is not enabled by default. The four-shape
20-sample sweep (three warmups, milliseconds shown as p10/median/p90) was:

| MxNxK | old v2 | B-swizzle v2 |
|-------|--------|--------------|
| 256x2816x2112 | 5.84/5.99/6.45 | 5.88/5.97/6.49 |
| 256x2112x2816 | 7.68/7.73/8.23 | 7.65/7.71/8.36 |
| 256x1024x2816 | 7.63/7.68/8.23 | 7.58/7.63/8.15 |
| 256x4224x2816 | 11.08/11.24/11.72 | 11.03/11.21/11.74 |

The fresh run shows only sub-percent changes, far below the required weighted
8% improvement; earlier isolated runs also showed shape-sensitive variance,
so the candidate remains opt-in. The real-weight bitwise test passes for both
layouts.

MoE routing now uses a fixed 128-expert stable counting layout shared by the
host, batched, and GPU variants; it preserves ascending expert order and the
original token/top-k insertion order without `std::map` allocations.
`AILA_DG_MOE_ROUTE_HIST=1` prints the observed M histogram. The real harness
showed many M=1--5 experts plus a small number of heavily loaded experts
(roughly M=120--255 for a 256-token canvas), and the batched path remained
bitwise identical to the host reference. A grouped NF4 small-M kernel was not
merged: no candidate yet meets the real-weight exactness gate.

`AILA_DG_STEP_TIMING=1` now reports input H2D, decoder submission and logits
D2H wait, argmax, entropy, gumbel, accept/renoise, host total, softmax and
soft-embedding submission windows, soft-embed D2H wait, and total. The
submission windows intentionally add no queue synchronization; the blocking
D2H buckets include the corresponding asynchronous device execution.

The synchronized profile path still reproduced the known A770 Level-Zero
`UR_RESULT_ERROR_DEVICE_LOST` during warmup, and decoder tensor dumping can
terminate with the existing `0xC0000005` diagnostic-path fault. Neither path is
used as a production performance gate; unprofiled production runs and the
InferRef NF4 GEMM test remain stable.
The new softmax-only dump uses the same explicit-sync diagnostic route; after
the driver entered that faulted state it produced no files, so rerun the direct
probability/embedding gate after an adapter reset.

## Benchmark Methodology

1. **Always warm up**: First iteration is up to 10% slower due to GPU cold state and oneDNN JIT compilation.
2. **Use 3+ iterations**: Average across iterations to smooth thermal/driver variance.
3. **Close other GPU applications**: Browser GPU acceleration, video playback, etc. can steal GPU time.
4. **Let GPU cool between runs**: Sustained benchmark can cause thermal throttling on consumer GPUs.
5. **Profile for diagnosis, benchmark for numbers**: Never report profile timings as real performance.
