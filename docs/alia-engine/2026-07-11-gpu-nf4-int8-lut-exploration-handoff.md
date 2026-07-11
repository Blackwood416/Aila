# 2026-07-11 Handoff: GPU NF4 INT8 LUT Exploration

## Mission

Explore whether the CPU backend's successful NF4-to-int8 codebook mapping can
improve GPU bitsandbytes NF4 throughput without unacceptable model-quality or
numerical regressions.

This is an isolated investigation. Do not change the default GPU NF4 path until
microbenchmarks and real-model validation both pass. Do not modify the native
CPU Q35 backend except to read its implementation as a reference.

## Workspace

- Worktree: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine`
- Branch: `codex/alia-custom-engine`
- CPU reference commit: `8b57264 perf: add int8 lut packed nf4 decode`
- Latest CPU batch-prefill commit at handoff: `a04e5e7 perf: batch cpu q35 mixer input projections`

The worktree may continue receiving CPU backend commits from another agent.
Create a separate worktree and branch for this investigation rather than
editing the shared CPU optimization worktree directly.

Suggested branch:

```text
codex/gpu-nf4-int8-lut
```

## Important Correction To The Initial Hypothesis

The GPU backend does not always dequantize NF4 weights into a persistent BF16
cache before multiplication.

There are currently three materially different paths:

1. Packed NF4 decode GEMV
   - `packed_nf4_gemv_bf16()` in `src/ops/Bnb4BitLinear.cpp`.
   - Reads packed NF4 codes, a FP32 16-entry quant map, and FP32 block absmax.
   - Performs inline FP32 codebook lookup and multiply before accumulating.

2. Packed NF4 prefill GEMM
   - `packed_nf4_gemm_bf16()` and `packed_nf4_gemm_bf16_tiled()`.
   - Reads packed NF4 directly and stages packed tiles in local memory.
   - Uses a local FP32 quant-map cache and pre-dequantizes values inside the
     compute loop.

3. BF16 cached/fallback path
   - `Bnb4BitLinear::dequantize_weight()` creates `cached_weight_bf16_`.
   - Used when policy or shape prevents the packed fastpath, and by the oneDNN
     path where applicable.

The first task is therefore profiling and path classification. Do not assume
that reducing `cached_weight_bf16_` addresses the active decode or prefill hot
path.

## CPU Evidence

The native CPU Q35 backend now supports:

```text
AILA_CPU_Q35_WEIGHT_CACHE=packed_nf4_i8
```

Its representation keeps:

- packed 4-bit NF4 codes;
- one FP32 final absmax per 64-weight block;
- one int8 16-entry codebook per weight reference;
- one FP32 codebook dequantization scale;
- FP32 activation and FP32 accumulation.

For each original NF4 codebook value:

```text
lut_scale = max(abs(nf4_codebook)) / 127
lut_i8[code] = round(nf4_codebook[code] / lut_scale)
weight ~= lut_i8[code] * lut_scale * block_absmax
```

On the target Haswell CPU, replacing FP32 LUT lookup with `pshufb` int8 lookup
changed matched real-model results as follows:

```text
FP16 cache, 4-token background:          24560 ms
packed NF4 exact FP32 LUT:               31114 ms
packed NF4 int8 LUT:                     22239 ms

FP16 cache, formal 384-token background: 85047 ms
packed NF4 int8 LUT:                     77957 ms
```

The int8 LUT path also reduced the CPU cache from 1434 MiB to 751 MiB. Output
remained schema valid. This is evidence that the extra 16-value codebook
quantization can be numerically viable for the 0.8B background extractor; it is
not proof that foreground, ASR, or TTS GPU models tolerate the same error.

Reference files:

- `src/models/cpu/CpuBnb4.hpp`
- `src/models/cpu/CpuBnb4.cpp`
- `tests/models/CpuBnb4Tests.cpp`

## Investigation Scope

Start with generic `Bnb4BitLinear` kernels only. Do not begin with the model
wrappers.

Primary files:

- `src/ops/Bnb4BitLinear.cpp`
- `src/ops/Bnb4BitLinear.hpp`
- `src/models/Bnb4BitModelLoader.cpp` and related weight-reference loading code
- `tools/perf/Bnb4PrefillBench.cpp`
- any existing BNB4 decode benchmark found under `tools/perf/`

Model integration references:

- `src/models/Qwen35HybridBnb4Backend.cpp`
- `src/models/Qwen3ASRBnb4Backend.cpp` or the current ASR BNB4 backend located
  with `rg`
- `src/models/Qwen3TTSBackend.cpp` and nested BNB4 modules if applicable

## Proposed GPU Representation

Keep packed NF4 codes and FP32 block absmax unchanged. Add an optional device
side codebook representation:

```cpp
int8_t nf4_lut_i8[16];
float nf4_lut_scale;
```

The kernel reconstructs:

```text
dequant_scale = nf4_lut_scale * block_absmax
weight = float(nf4_lut_i8[code]) * dequant_scale
```

Activation and accumulation should remain in their current types for the first
experiment. Do not add activation INT8 quantization in the same change; that is
a separate numerical and scheduling question.

Because the GPU kernels use SYCL, the CPU `pshufb` implementation is only a
conceptual reference. The GPU experiment should test whether an int8 local LUT
and integer-to-float conversion are cheaper than FP32 local-memory lookup on the
target Intel GPU.

## Isolation Controls

Add one narrow opt-in environment variable:

```text
AILA_BNB4_GPU_NF4_INT8_LUT=1
```

Requirements:

- Default is off.
- It affects only packed GPU NF4 kernels.
- It does not change CPU BNB4 behavior.
- It does not silently force packed kernels for unsupported shapes.
- Logs state whether decode and prefill selected the int8 LUT path.

If decode and prefill need separate diagnosis, use scoped controls:

```text
AILA_BNB4_GPU_DECODE_INT8_LUT=1
AILA_BNB4_GPU_PREFILL_INT8_LUT=1
```

Prefer the scoped variables during A/B work. A combined alias can be added only
after both paths are understood.

## Phase 1: Numerical Feasibility

Before optimizing a kernel, quantify the codebook approximation itself.

1. Read the real 16-entry quant maps from each fixed model.
2. Compute the int8 LUT and reconstruction error.
3. Report per model:
   - maximum absolute codebook error;
   - maximum relative error excluding values near zero;
   - exact zero preservation;
   - sign preservation;
   - reconstructed minimum and maximum.
4. Compare one packed NF4 projection against the current FP32-LUT result using
   identical activations.

At minimum test quant maps from:

- `models/Qwen3-ASR-1.7B-BNB-NF4`
- `models/qwen3.5-4B-bnb-nf4-offline-visiondense`
- `models/qwen3.5-0.8B-bnb-nf4-offline`
- BNB4 weights inside `models/Qwen3-TTS-12Hz-0.6B-Base`, if present

Do not assume every tensor has an identical quant map merely because the quant
type is NF4. Verify and log this.

## Phase 2: Decode Microbenchmark

Start with `packed_nf4_gemv_bf16()` because decode is simple and high-frequency.

Implement a second kernel or a clearly isolated template specialization. Avoid
placing an environment branch inside the innermost weight loop.

Compare:

```text
current: code -> FP32 local LUT -> multiply absmax -> FMA
int8:    code -> int8 local LUT -> convert -> multiply combined scale -> FMA
```

Measure representative Qwen3.5 decode shapes, including:

- hidden projections around K=1024 and K=2560;
- MLP projections with large N;
- fused rows used by QKV and gate/up paths;
- single-token `M=1` only in this phase.

Report median and best time over enough iterations to amortize JIT and queue
startup. Include the current exact packed kernel and BF16 cached path when both
are valid for the shape.

Do not proceed on an isolated win smaller than normal measurement variance.

## Phase 3: Prefill Microbenchmark

Only after decode is understood, add the int8 LUT option to:

- `packed_nf4_gemm_bf16_tiled()`;
- the M80 and default tile variants;
- fused gate/up SwiGLU prefill only if the generic result is positive.

Use existing `tools/perf/Bnb4PrefillBench.cpp` shapes plus the real ASR and
Qwen3.5 shapes already documented in optimization handoffs. Important shapes
include:

```text
ASR gate/up-like: M~=72, N=12288, K=2048
Qwen3.5 prompt prefill: M around 100-300 with model-specific N/K
short append shapes: M=24 and M=32
```

The int8 LUT may reduce local-memory bandwidth but add conversion instructions.
Measure instead of assuming which side wins.

## Phase 4: Real-Model Validation

Use the fixed model set and foreground LoRA:

- ASR: `models/Qwen3-ASR-1.7B-BNB-NF4`
- Foreground: `models/qwen3.5-4B-bnb-nf4-offline-visiondense`
- Foreground LoRA: `F:/unsloth/qwen35_4b_alia_identity_r16_lr5e4/checkpoint-350`
- Background: `models/qwen3.5-0.8B-bnb-nf4-offline`
- TTS: `models/Qwen3-TTS-12Hz-0.6B-Base`

Validate paths separately before enabling them together:

1. Foreground Qwen3.5 decode only.
2. Foreground Qwen3.5 prefill only.
3. ASR BNB4 decode/prefill if the generic kernel is shared.
4. TTS BNB4 modules if they use the shared kernel.
5. Full voice-text-voice smoke.

Real smoke command shape:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\gpu_nf4_i8_short.wav' `
  -LogPath 'tmp\alia-real-smoke\gpu_nf4_i8_short.log' `
  -RequestText '艾莉亚，请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Run the five-scenario Chinese voice matrix only after a short smoke shows a
repeatable latency benefit and acceptable text/audio behavior.

## Correctness Gates

Kernel-level:

- Compare current FP32-LUT and int8-LUT outputs on deterministic fixtures.
- Report max absolute and relative error, not only pass/fail.
- Cover nested absmax, fused rows, odd tails, and all 16 NF4 codes.
- Confirm no out-of-bounds local-memory lookup for signed int8 values.

Model-level:

- No NaN or Inf in outputs.
- Foreground text remains coherent across repeated Chinese prompts.
- ASR transcript does not materially regress.
- TTS audio remains non-corrupt and output-ASR remains understandable.
- Background JSON remains schema valid when the background GPU path is tested.
- LoRA remains applied and produces valid output.

Pipeline-level:

- `ALIA_REAL_MODEL_SMOKE_PASS`.
- No new playback buffer gaps.
- No TTFA regression outside normal run variance.
- Full matrix passes before any default change.

The CPU background extractor tolerated the int8 codebook approximation, but the
foreground 4B model and TTS are more quality-sensitive. A performance win is
not sufficient if text or audio quality changes materially.

## Performance Gates

Keep the path opt-in unless all relevant gates pass:

- Decode kernel improves by at least 5% on multiple real shapes.
- Prefill kernel does not regress important short and ASR shapes.
- Real short-smoke model time improves repeatably.
- Voice matrix average TTFA improves or stays neutral.
- No meaningful increase in JIT time, local-memory pressure, register spills,
  or GPU memory use.

If decode wins but prefill loses, retain separate scoped toggles rather than
forcing one global policy.

## Likely Risks

1. GPU integer-to-float conversion may cost more than FP32 local LUT lookup.
2. An int8 LUT may increase register pressure or reduce occupancy.
3. Fused-row kernels may amplify codebook approximation differently from a
   single projection.
4. Foreground sampling can diverge after tiny logit changes, making latency and
   text comparisons noisy.
5. TTS is especially sensitive to token-shape changes; inspect audio and output
   ASR, not only kernel timing.
6. oneDNN/BF16 fallback paths may remain dominant for some shapes, making a
   packed-kernel optimization invisible in end-to-end metrics.

## Do Not Repeat

- Do not replace all BF16 caches before proving which path is hot.
- Do not combine activation INT8 with NF4 LUT INT8 in the first experiment.
- Do not use API-only tests as real-model evidence.
- Do not enable a global default based on one microbenchmark shape.
- Do not alter CPU `packed_nf4_i8`; use it only as a numerical and algorithmic
  reference.

## Expected Deliverable

The exploration agent should return:

1. A path map showing which real workloads use packed decode, packed prefill,
   or BF16 cached fallback.
2. Quant-map error statistics for the fixed models.
3. Decode and prefill microbenchmark tables for exact versus int8 LUT.
4. One opt-in implementation commit if results justify it.
5. Short real-model smoke results, and a full matrix only if the short result is
   positive.
6. A clear recommendation: reject, keep diagnostic-only, enable for decode
   only, enable for prefill only, or proceed toward a default.

## Copy-Paste Prompt

```text
You are exploring a GPU NF4 INT8-LUT path for the Alia custom engine.
Work in a separate worktree and branch derived from
E:/RiderProjects/Aila/.worktrees/alia-custom-engine at commit a04e5e7 or later.
Read docs/alia-engine/2026-07-11-gpu-nf4-int8-lut-exploration-handoff.md first
and follow it.

Goal: determine whether mapping the 16-entry NF4 codebook to int8 improves the
existing packed GPU NF4 decode and/or prefill kernels while keeping activation
and accumulation types unchanged. Keep all new paths opt-in. Do not modify the
native CPU Q35 backend. Validate numerical error, kernel microbenchmarks, and
real fixed-model smoke; do not rely on API-only tests.
```
