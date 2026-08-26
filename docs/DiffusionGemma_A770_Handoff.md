# DiffusionGemma A770 Handoff

## Scope and current state

DiffusionGemma text generation is working on Aila `main` with the corrected NF4
checkpoint:

```text
E:\RiderProjects\Aila\models\diffusiongemma-26B-A4B-it-bnb-nf4-true14g
```

Use this checkpoint. Do not return to the older
`DiffusionGemma-26B-A4B-it-BNB-NF4` export: its quantization state and inference
path were not trustworthy.

The trusted MI300X export/reference notebook is:

```text
D:\Downloads\DiffusionGemma_26B_True14GB_BNB_NF4_Complete_v3.ipynb
```

Do not execute that notebook's global model reconstruction on the A770. It can
require about 65 GiB of VRAM. Use it as a semantic reference or run it only on
suitable remote hardware.

Important environment distinction:

- Aila does **not** need `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`.
- Set that variable only for the Python/Transformers reconstruction or
  reference workflow when its individual allocations exceed 4 GiB.
- Aila's commands below intentionally omit it.

## Implemented commits

The working implementation is the following sequence on `main`:

```text
a85e439 feat: align DiffusionGemma NF4 inference with transformers
b45d556 perf: accelerate DiffusionGemma NF4 generation on A770
ebe8fd4 perf: optimize DiffusionGemma NF4 GEMM on A770
```

The handoff documentation and corrected environment guidance are committed
after `ebe8fd4`; use `git log -4 --oneline` to obtain that documentation commit.

## Validated platform

- GPU: Intel Arc A770 16 GiB
- Driver: `32.0.101.8860`
- oneAPI: `2026.1`
- oneDNN: `3.11.2`
- Quantization: bitsandbytes NF4, blocksize 64, double quantization, bf16
  compute
- Canvas: 256 tokens
- Denoising: 25 steps
- Correctness seed: 42

## Build and run

Use the dedicated build directory. The repository's older `build` directory may
have been configured with a different compiler stack.

```powershell
pwsh build.ps1 -BuildDir build-diffusiongemma -Jobs 36
```

Minimal 25-step Aila run, without the Python-only relaxed-allocation variable.
The engine runs a vLLM-style warmup at init by default (one dummy 256-token
canvas pass) so a fresh process pays the one-time SYCL JIT/oneDNN primitive
cost before `generate()` starts; `AILA_INIT_WARMUP=0` below selects the
cold-start measurement used for the benchmark records in this document.

```powershell
$env:AILA_INIT_WARMUP = "0"
$env:AILA_DIFFUSION_STEPS = "25"

$request = '{"messages":[{"role":"user","content":"你好！请用条理清晰的中文，列出 3 个离散扩散语言模型在文本生成上的潜在技术优势。"}]}'
$request | .\build-diffusiongemma\Aila.exe `
  -m 'E:\RiderProjects\Aila\models\diffusiongemma-26B-A4B-it-bnb-nf4-true14g' `
  --max-tokens 256 --messages-json - --seed 42 --no-stream `
  --log-level info
```

For Python/Transformers reference runs only:

```powershell
$env:UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS = "1"
```

## Performance result

The A770 blocksize-64 NF4 GEMM dispatch now uses `BM=64`, `BN=64`, `BK=64`.
BK64 reduces work-group SLM from about 20.5 KiB to 10.5 KiB. Other GPUs and
larger NF4 block sizes keep their previous dispatch.

All timings below are `generate()` wall time, including prompt encoding, 25
denoising passes, host sampling, and soft self-conditioning. Each timing is a
fresh process with tensor dumping and synchronizing profiling disabled.

| Dispatch | 25-step samples (ms) | Median (ms) |
|----------|----------------------|-------------|
| BK128 baseline | 328312.37, 329871.85, 330980.95 | 329871.85 |
| BK64 production | 260323.84, 257500.51, 258545.84 | 258545.84 |

Median latency improved by 21.6%, equivalent to a 1.276x speedup.

Runtime memory after allocating decoder buffers was 14673.44 MiB current and
14705.43 MiB peak. BK64 changes local SLM only and adds no global or persistent
allocation over BK128.

Representative 20-sample BK64 operator medians:

| Shape `M x N x K` | Median (ms) | p10 / p90 (ms) |
|-------------------|-------------|----------------|
| `256 x 2816 x 2112` | 20.60 | 20.46 / 21.57 |
| `256 x 2112 x 2816` | 27.35 | 27.14 / 28.14 |
| `256 x 1024 x 2816` | 27.46 | 27.13 / 28.18 |
| `256 x 4224 x 2816` | 44.68 | 44.26 / 46.24 |

## Correctness evidence

BK128 and BK64 produced exactly the same 256 committed token IDs for the fixed
prompt and seed. Both 1024-byte int32 streams had SHA256:

```text
A4F674A0CB0C81E0682FC4F7B490251BEB9011DDF749475F216087B5761D6CD1
```

The output was normal Chinese text and every full 25-step process exited with
code 0. No OOM, TDR, or device-lost event occurred.

The final regression set passed:

```powershell
ctest --test-dir build-diffusiongemma `
  -R "^(AilaBnb4BitLinearKernelTests|AilaDiffusionGemmaModelSpecTests|AilaChatTests)$" `
  --output-on-failure
```

InferRef fresh-run also passed. Its synthetic NF4 GEMM boundary compared 4096
bf16 values with exact match, zero NaNs, and zero Infs:

```powershell
& 'C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe' `
  tools\diffusion_gemma_inferref\make_nf4_gemm_testcase.py `
  build-diffusiongemma\inferref-dg-nf4

inferref agent run build-diffusiongemma\inferref-dg-nf4 `
  --adapter tests\inferref\diffusion_gemma_nf4.adapter.json `
  --runs-dir build-diffusiongemma\inferref-runs `
  --atol 0.02 --rtol 0.005 --json
```

The repository `.venv` contains the DiffusionGemma Transformers reference but
does not contain the InferRef Python package. The global Python 3.13 environment
owns the installed `inferref` command and package.

## Profiling conclusions

Before BK64, one profiled decoder pass was approximately:

- decoder plus logits: 11.77 s
- attention total: 5.61 s
- dense MLP: 3.35 s
- MoE: 2.37 s

The detailed profile showed the masked-attention core itself at only about
114-119 ms. Q/K/V and O NF4 projections accounted for nearly all of the
attention category. Therefore the approved plan's 10% threshold was not met and
the attention implementation was not rewritten.

With BK64, the corresponding decoder plus logits pass was about 8.99 s:

- attention total: 3.90 s
- Q/K/V projections: 2.36 s
- attention core: 0.12 s
- O projection: 1.42 s
- dense MLP: 2.20 s
- MoE: 2.43 s

Use `AILA_PROFILE_DG=1` only for relative phase diagnosis. It calls
`ctx.synchronize()` at phase boundaries and is not a valid absolute performance
measurement.

## Rejected experiments

Do not restore these candidates directly:

1. A first full-model GPU MoE gather/scatter implementation exited with code 1
   shortly after decoder start. It was completely rolled back. Retry only after
   building a small isolated MoE region harness that proves bf16 rounding and
   sorted-expert accumulation order.
2. The first SG16/WG16 joint-matrix NF4-to-XMX candidate terminated under the
   15-second isolated watchdog with Windows status `0xC0000409`. Per the A770
   stability rule, all XMX geometry exploration stopped and the code was
   removed.
3. Do not rewrite masked attention in the next pass unless a fresh profile
   shows its core above 10% of decoder time. The projection GEMMs, not SDPA,
   were the bottleneck in the measured model.

No failed XMX path, MoE experiment switch, or temporary BK128 rollback variable
exists in production code.

## Recommended next work

The next performance session should start from `main` and preserve the exact
token gate. Suggested order:

1. Build a small InferRef-compatible MoE harness using real expert weights and
   the observed small-M routing distribution. Validate sorted expert order,
   bf16 multiplication, and accumulation before any full-model change.
2. Continue tuning the common scalar NF4 GEMM around BK64 using real q_proj,
   O-proj, dense MLP, and MoE shapes. Every new A770 geometry must run in an
   isolated process with a 15-second watchdog.
3. Investigate host sampling and soft self-conditioning only after refreshing
   the unsynchronized wall breakdown; do not infer absolute costs from
   `AILA_PROFILE_DG`.
4. Keep the merge gates: all 256 tokens exact, three-run 25-step median at least
   5% faster, runtime peak no more than 256 MiB above baseline, and no
   OOM/TDR/device-lost.

## 2026-08-26 session: vectorized NF4 GEMM (merged) and MoE GPU path (WIP)

### Merged: v2 NF4 GEMM (`packed_nf4_gemm_bf16_tiled_v2`)

The blocksize-64 A770 dispatch is now `BM=32, BN=128, BK=64, TM=2, TN=8`,
16x16 work-group. The B tile is dequantized to bf16 SLM once per K block and
the inner loop uses `vec8` A/B loads. 20-sample operator medians on the real
decode shapes improved 3.5-3.9x over the BK64 scalar kernel.

Three-run 25-step medians (fresh process, seed 42, 256-token canvas, 25
denoising steps, dumping/profiling disabled):

| Dispatch | 25-step samples (ms) | Median (ms) |
|----------|----------------------|-------------|
| BK64 baseline | 328312/329872/330981 (BK128), 260324/257501/258546 (BK64) | 258545.84 |
| BK64 + v2 | 139772.34, 141609.89, 140662.14 | 140662.14 |

Median latency improved 45.6% over the BK64 baseline (1.84x total vs BK128).
All 256 committed token IDs were exact in every run
(SHA256 `A4F674A0CB0C81E0682FC4F7B490251BEB9011DDF749475F216087B5761D6CD1`).
Runtime peak after decoder buffers: 14714.51 MiB (+9 MiB vs the BK64 record),
no OOM/TDR/device-lost. ctest regression set and InferRef fresh-run pass
(InferRef exact match, zero NaNs/Infs).

Numerical gate that made this safe: the test suite now runs a bitwise A/B on
the real scgate/qproj weight fixtures for every supported GEMM geometry
(`run_real_v1_v2_bitwise_test`). On this stack, only geometries with
`BN >= 32` are bitwise identical to the committed scalar kernel; `BN=16`/
`TN=1` geometries are 2-5x faster in the microbenchmark but the device
compiler reassociates their k accumulation under `-ffast-math` and changes
last-ulp results (visible only on real weights, not synthetic fixtures).
`-ffp-model=strict`, `/fp:strict`, and `#pragma clang fp` do not affect the
DPC++ device codegen for these kernels; the geometry gate is the only
reliable protection.

`AILA_BNB4_GEMM_V2=0` selects the original scalar geometry for A/B.
`AILA_BNB4_GEOM_BENCH=1` sweeps geometries with bitwise validation, and
`AILA_BNB4_ROUND_PROBE=1` probes device bf16 conversion/multiply edge cases
(the probe documents that a fused bf16-load `*` float multiply on this device
rounds one float ulp below the IEEE host multiply; widen bf16 through
`bf16_to_f32_raw` before multiplying when bitwise host parity is required).

### Not merged: GPU-resident MoE path (validated but slower)

`Bnb4GroupedMoE::forward_gpu` moves the per-expert gather and the final bf16
accumulation to the GPU while keeping the dequant and oneDNN matmul steps
unchanged. `AILA_DG_MOE_HARNESS=1` runs the committed host path and the GPU
path for every layer and compares bitwise: with the `bf16_to_f32_raw` widen
fix, all layers/passes matched with zero mismatches. This satisfies the
handoff's "isolated MoE region harness" prerequisite.

It is NOT the production default (`AILA_DG_MOE_GPU` defaults to 0) because:

1. Per-group wall time dropped from ~50-90 ms to ~42 ms per layer, but the
   extra gather/accumulate submissions cost more than the removed
   h2d/d2h transfers; the 25-step median was ~151 s vs ~141 s for the host
   path with the v2 GEMM.
2. `AILA_PROFILE_DG=1` together with the GPU path crashes with
   `0xC0000005` (RIP=0) after the first MoE layer, while production mode
   completes with the exact token hash. Treat the GPU path as WIP; a batched
   (single-kernel) gather/accumulate or fused dequant+matmul is required
   before it can win.

The host MoE path is unchanged and remains the committed behavior.

### Merged: vLLM-style init warmup

Following vLLM's `_dummy_run` methodology (run the full model once with the
real max shape before serving so JIT/primitive costs never hit a request),
the engine now auto-warms DiffusionGemma at init: a dummy 8-token prompt
encode, one dummy 256-token `diffusion_decode_canvas` pass with zero soft
self-conditioning, and one soft-embedding call. The output is discarded and
the backend is reset; the committed token stream is unchanged.

Fresh-process 25-step medians (seed 42, canvas 256, 25 steps):

| Mode | Warmup median (ms) | Generate median (ms) |
|------|--------------------|----------------------|
| Cold (`AILA_INIT_WARMUP=0`) | 0 | 149913 (4 runs) |
| Default warmup (1 pass) | 17903 (3 runs) | 129230 (3 runs) |

Warmup cuts median `generate()` wall time by 13.8% (149.91 s -> 129.23 s) and
keeps total init+warmup+generate neutral-to-better; all runs preserved
SHA256 `A4F674A0...`. `AILA_DG_WARMUP_PASSES` and `AILA_DG_WARMUP_CANVAS`
override the dummy pass count/canvas length (a smaller canvas only saves
compute; JIT cost is shape-independent).

### Merged: batched MoE transfers

The host MoE path's ~60 blocking h2d/d2h pairs per layer measured ~51 ms/layer
of transfer/sync overhead. `Bnb4GroupedMoE::forward_batched` (default on,
`AILA_DG_MOE_BATCHED=0` restores the per-group path) gathers every group into
one shared device buffer with a single flat gather kernel and copies all
per-expert outputs back with one d2h. Each expert's oneDNN matmul runs on an
offset view of the same buffer, so kernels, values, and accumulation order are
unchanged; the harness verified zero bitwise mismatches on every layer.

Fresh-process 25-step medians (seed 42, canvas 256, 25 steps):

| Mode | Warmup median (ms) | Generate median (ms) |
|------|--------------------|----------------------|
| Cold, host MoE | 0 | 149913 (4 runs) |
| Warmup, host MoE | 17903 (3 runs) | 129230 (3 runs) |
| Warmup + batched MoE | 16675 (3 runs) | 94071 (3 runs) |

Batched MoE cuts median `generate()` to 94.07 s (-27.2% vs warmup-only) with
a 93.2-94.5 s spread; runtime peak 14733.99 MiB (+28.6 MiB vs the BK64
record); all runs preserved SHA256 `A4F674A0...`.

Note: the GPU-resident MoE WIP (`AILA_DG_MOE_GPU=1`) became flaky after the
shared transfer buffers were added (bitwise mismatches in the harness even in
isolation, plus the earlier `AILA_PROFILE_DG` access violation). It remains
disabled and should be re-validated from scratch before any further work.

Driver-hang root cause (2026-08-27): the `AILA_PROFILE_DG` crash was a
Level-Zero driver deadlock, not a host bug. With the batched MoE path, all
threads park in ntdll waits inside `ze_intel_gpu64.dll` when a large burst of
oneDNN matmul submissions is followed by the profile mode's per-phase syncs
(first manifests at layer 5's linear-attention sync in a cold run). Draining
the queue after every expert group avoids it entirely. Production (no
profile) is stable with the default drain every 8 groups; the batched path
auto-drains every group under `AILA_PROFILE_DG`, and `AILA_DG_MOE_PACE`
overrides the interval. The oneDNN memory handles from `run_matmul` are also
retained until the queue drains (`pending_memories_`) as defensive hygiene.

Final three-run 25-step medians with warmup + batched MoE (seed 42):
warmup 16.70 s, generate 94.98 s (94.23/94.98/95.05 s), runtime peak
14733.99 MiB, SHA256 `A4F674A0...` on all runs. Profile mode now completes.

## Important files

- `src/ops/Bnb4BitLinear.cpp`: A770 NF4 GEMM dispatch and kernels
- `src/models/DiffusionGemmaBnb4Backend.cpp`: decoder, profile phases, MoE path
- `src/generation/DiffusionGemmaGenerator.cpp`: 25-step loop, token dump, wall
  and memory summary
- `tests/ops/Bnb4BitLinearKernelTests.cpp`: real and synthetic NF4 shapes plus
  the opt-in 20-sample microbenchmark
- `tests/inferref/DiffusionGemmaNf4InferRefAdapter.cpp`: InferRef engine adapter
- `tools/diffusion_gemma_inferref/`: numerical comparison and testcase tools
- `docs/perf-tuning.md`: committed benchmark record

Preserve the existing untracked `AGENTS.md`, release notes, model files, and
build directories. They were intentionally not included in the implementation
commits.
