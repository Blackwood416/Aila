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

Minimal 25-step Aila run, without the Python-only relaxed-allocation variable:

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
