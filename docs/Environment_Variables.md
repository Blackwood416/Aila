# Environment Variables

Aila supports the following environment variables for runtime configuration. Boolean flags accept `0`/`false` (off) and `1`/`true` (on). Integer values are parsed from decimal strings.

---

## Windows Runtime Isolation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_RUNTIME_DLL_DIR` | directory path | directory containing `AilaShared.dll` | Directory containing `AilaWorker.exe` and the private oneAPI runtime. Relative values resolve relative to AilaShared.dll; absolute values are accepted. The selected path is normalized to an absolute path internally. An unset or empty value enables the legacy flat layout. |
| `AILA_LEGACY_TEXT_V2` | integer flag | `0` | Set `1` to route supported pure-text legacy C ABI calls through shared Model/Session/Request and IPC v2. Requires matching `AilaInference.dll` beside `AilaShared.dll`; Worker remains in `AILA_RUNTIME_DLL_DIR`. At most 4 Sessions share each owner. The generated-token ceiling follows context capacity, up to 32768, while prompt + generation must fit the shared 32768-token reservation budget. Media/quantized-KV/legacy-linear models retain v1. See [legacy text adapter](legacy-text-adapter.md). |
| `AILA_LEGACY_TEXT_MAX_SESSIONS` | unsigned decimal, 1–4 | `4` | Lower the number of legacy text Sessions sharing a Model/Worker; a full owner rejects another Engine without starting an extra Worker. |
| `AILA_LEGACY_TEXT_MAX_SESSION_CACHE_BYTES` | uint64 decimal bytes | `0` | Total reserved device cache/state payload for Sessions sharing one legacy text Model; `0` uses the Session-count bound alone. |
| `AILA_LEGACY_TEXT_MAX_DEVICE_BYTES` | uint64 decimal bytes | `0` | Maximum explicit device USM owned by this shared legacy Worker, including weights, Session state and execution scratch. `0` disables this additional byte ceiling. |
| `AILA_LEGACY_TEXT_MAX_PINNED_HOST_BYTES` | uint64 decimal bytes | `0` | Maximum explicit pinned host USM in this shared legacy Worker, including sampling scratch. `0` disables this additional byte ceiling. |
| `AILA_LEGACY_TEXT_MAX_SESSION_HOST_BYTES` | uint64 decimal bytes | `0` | Total reserved pageable host Session payload, including Hybrid mirrors; `0` uses the Session-count bound alone. |
| `AILA_WORKER_ENV_PASSTHROUGH` | `;`-separated variable names | `""` | Names of scrubbed GPU-runtime variables (see below) that the host explicitly wants forwarded to the worker anyway. Matched case-insensitively; surrounding spaces are ignored. Example: `AILA_WORKER_ENV_PASSTHROUGH=ZES_ENABLE_SYSMAN;UR_L0_USE_COPY_ENGINE`. |
| `AILA_KEEP_SYCL_CACHE_PERSISTENT` | flag | `0` | `Aila.exe` and `AilaWorker.exe` force `SYCL_CACHE_PERSISTENT=0` at startup because an enabled persistent cache crashes the bundled DPC++ runtime (null dereference while hashing oneDNN interop programs at the first JIT build). Set `1` to keep the inherited value, e.g. to test a runtime that fixes the bug. For the worker, the variable must also be forwarded via `AILA_WORKER_ENV_PASSTHROUGH=SYCL_CACHE_PERSISTENT`. |

The unset/empty fallback exists only for compatibility with legacy deployments.
When the proxy and runtime DLLs share one directory, exposing the proxy directory
to the host DLL loader also exposes the private runtime, so the flat layout does
not provide host DLL-search isolation. Python, ComfyUI, and other embedding hosts
must use the split layout and explicitly set `AILA_RUNTIME_DLL_DIR`.

The recommended release/integration layout keeps the host API DLLs in the
integration root, with `AilaWorker.exe`, `Aila.exe`, and the oneAPI runtime DLLs
under `aila_runtime/`:

```text
integration_root/
|-- AilaShared.dll
|-- AilaInference.dll        (async API / shared legacy text)
|-- <host C++ runtime DLLs>  (when required by AilaInference.dll)
`-- aila_runtime/
    |-- AilaWorker.exe
    |-- Aila.exe
    `-- <oneAPI and other runtime DLLs>
```

For this layout, set `AILA_RUNTIME_DLL_DIR=aila_runtime` before the host loads
`AilaShared.dll`. The default v1 path uses one Worker per Engine. With
`AILA_LEGACY_TEXT_V2=1`, compatible text Engines share a Worker when their
model/runtime/capacity/backend/environment match. The worker
starts in the selected directory with a child-only `PATH` containing that
directory and Windows system directories. The host's DLL search path remains
unchanged.

The five legacy text admission settings are read once for each owner from the
same Win32 environment snapshot used to select that owner. They apply only to
the `AILA_LEGACY_TEXT_V2=1` path. Values must contain decimal digits only;
signs, whitespace, units and uint64 overflow are rejected as invalid arguments
before a Worker starts. Budgets describe Aila-owned payloads and explicit USM,
not total process RSS or WDDM residency. Different model/runtime/capacity or
environment groups have independent owners and budgets. Defaults leave the
existing optional adapter behavior unchanged.

The worker also does not inherit host GPU-runtime instrumentation variables.
GPU hosts such as PyTorch XPU set tracing and instrumentation variables in
their own process (`UR_ENABLE_LAYERS`, `XPTI_TRACE_ENABLE`,
`XPTI_FRAMEWORK_DISPATCHER`, `XPTI_SUBSCRIBERS`, `ZES_ENABLE_SYSMAN`, ...).
Inherited by the worker, these reconfigure the private oneAPI runtime and can
load the host's profiling DLLs into the isolated process, which has crashed the
worker under ComfyUI and other torch-based hosts. `SYCL_CACHE_PERSISTENT=1`
(commonly exported for llama.cpp-SYCL or torch XPU startup speedups) crashes
the bundled DPC++ runtime outright (issue #4). The proxy therefore removes
variables matching these prefixes from the worker environment: `XPTI_`, `UR_`,
`ZES_`, `ZET_`, `ZE_ENABLE_`, `OCL_ICD_`, `SYCL_CACHE_`, and
`__KMP_REGISTERED_LIB_`. All other variables — including `AILA_*`,
`ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, and `ZE_FLAT_DEVICE_HIERARCHY` —
are inherited unchanged. To forward a scrubbed variable deliberately, list its
name in `AILA_WORKER_ENV_PASSTHROUGH`.

ASR and TTS stream handles keep shared ownership of their proxy engine. Destroy
ASR streams, and wait for or cancel then destroy TTS streams, before destroying
`AilaEngine`. Otherwise, worker shutdown can be deferred until the final stream
is destroyed.

Python and other embedding hosts must add only the directory containing the
proxy to their DLL search path. Do not call
`os.add_dll_directory("aila_runtime")`, and do not append `aila_runtime` to the
host `PATH`; doing so mixes Aila's private oneAPI DLLs back into the host runtime.

If the directory or `AilaWorker.exe` is missing, or if the worker does not match
the proxy's protocol, ABI, or build identity, engine initialization fails with a
diagnostic available through the C API. Worker failures are not automatically
retried. Deploy `AilaShared.dll` and `AilaWorker.exe` from the same release.

This variable controls the Windows proxy/worker architecture. Non-Windows builds
continue to run inference in-process.

---

## General

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_MODEL_DIR` | string | `""` | Default model directory (fallback for `-m` CLI argument) |
| `AILA_LORA_DIR` | string | `""` | Default LoRA adapter directory (fallback for `--lora`) |
| `AILA_MAX_SEQ_LEN` | int | `4096` | Maximum context window length |
| `AILA_DECODE_CHUNK_SIZE` | int | `12` | Non-streaming greedy decode chunk size (tokens per host sync) |
| `AILA_STREAM_CHUNK_SIZE` | int | `4` | Streaming greedy decode chunk size |
| `AILA_THINKING_BUDGET` | int | `-1` | Thinking budget: `-1` = disabled, `0` = no-think prompt mode, `>0` = max generated tokens inside `<think>` |
| `AILA_STREAM_OUTPUT` | int | auto | Force streaming (`1`) or non-streaming (`0`). Auto-detected from terminal type when not set |
| `AILA_LOG_LEVEL` | string | `info` | Minimum log level: `verbose` (-1), `debug` (0), `info` (1), `warning` (2), `error` (3). Accepts names (case-insensitive) or numeric values |
| `AILA_INIT_WARMUP` | int | `-1` (auto) | Init warmup: `0` = skip, `1` = force, `-1` = auto. The legacy engine auto-enables except for Qwen3-TTS and unsupported Qwen3.5 specs; the Qwen4Exp worker path (`LoadedModel::Load`) runs only when set to `1`. On that path the warmup executes one production-shaped chunk plus one decode step in a scratch session at load. Measured 2026-10-15 on A770 (pair `retained21`): warmup ~96 s, first large request 110 s -> 51 s, startup +111 s, decode unchanged |
| `AILA_INIT_WARMUP_ROWS` | int | `4096` | Row count of the Qwen4Exp load-time warmup chunk (clamped to the max sequence length). Only read when `AILA_INIT_WARMUP=1` on the Qwen4Exp path; the warmup is skipped when `AILA_Q4E_HOTSET_EXPORT` is set so synthetic routes cannot pollute an exported profile |
| `AILA_DG_WARMUP_PASSES` | int | `1` | DiffusionGemma warmup: number of dummy full-canvas decoder passes run at init (vLLM-style JIT/primitive amortization) |
| `AILA_DG_WARMUP_CANVAS` | int | `0` | DiffusionGemma warmup canvas override: `0` = real canvas length, `>0` = smaller dummy canvas (JIT cost is shape-independent; only compute scales) |
| `AILA_DG_MOE_BATCHED` | int | `1` | DiffusionGemma MoE batched-transfer path: single flat gather + one d2h for all expert groups instead of one blocking h2d/d2h per group. Bitwise identical to the per-group host path (`0` restores it) |
| `AILA_DG_MOE_GPU` | bool | `false` | Opt-in exact GPU-resident MoE output path. Keeps the same per-expert dequantization and oneDNN matmuls as the batched path, then performs the stable expert-order bf16 accumulation on GPU to avoid the output D2H, host accumulation, and result H2D |
| `AILA_DG_MOE_GROUPED_XMX` | bool | `false` | Experimental grouped NF4/XMX MoE GEMM used only with `AILA_DG_MOE_GPU=1`. Double-quant absmax decoding is supported, but the XMX accumulation order is not bitwise identical to oneDNN; do not enable for exact-token production runs |
| `AILA_DG_MOE_HARNESS` | bool | `false` | Run the host MoE reference and compare enabled batched/GPU candidates byte-for-byte for every layer. Adds blocking copies and duplicate computation; validation only |
| `AILA_DG_MOE_PACE` | int | auto | DiffusionGemma batched MoE drain interval in expert groups: `0`/unset = 8 in production, 1 under `AILA_PROFILE_DG` (larger bursts plus per-phase syncs hang the Level-Zero driver on A770) |
| `AILA_DG_STEP_TIMING` | int | `0` | Print unprofiled per-step buckets: input H2D, decoder/logits total, decoder submission, logits D2H wait, argmax, entropy, gumbel, accept/renoise, host total, softmax submission, soft-embedding GEMM submission, soft-embed D2H wait, and total; no extra profiling synchronization is added |

---

## Chat Formatting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_CHAT_TEMPLATE` | string | unset | Raw Jinja chat template override. Takes precedence over request/model/built-in template selection when the request does not provide an explicit template override. |
| `AILA_CHAT_TEMPLATE_PATH` | string | unset | Path to a Jinja chat template override file. Used when `AILA_CHAT_TEMPLATE` is unset and the request does not provide an explicit template override. |

Qwen3.5 Hybrid models use Aila's built-in fixed Qwen3.5 template when no explicit override is provided. Request-level `chat_template_kwargs` can set `enable_thinking`, `preserve_thinking`, `auto_disable_thinking_with_tools`, `max_tool_arg_chars`, and `max_tool_response_chars`. Request JSON may also set `reasoning_budget`, `thinking_budget`, or `thinking_budget_tokens`.

---

## KV Cache Quantization

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_KV_QUANT` | bool | `false` | Global KV cache quantization fallback for all model backends |
| `AILA_ASR_KV_QUANT` | bool | inherits global | ASR-specific override; can enable or disable quantization independently |
| `AILA_TTS_KV_QUANT` | bool | inherits global | TTS-specific override; applies to talker and predictor caches |
| `AILA_VLM_KV_QUANT` | bool | inherits global | Qwen3.5 VLM-specific override for full and linear attention caches |

---

## Bitsandbytes 4-bit (AILA_BNB4_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_BNB4_CACHE_DEQUANT` | bool | `false` | Cache dequantized bf16 weights in GPU memory. Saves dequant cost but increases VRAM ~2 GB for 4B models. Recommended off for large prefill |
| `AILA_BNB4_GEMV_WG` | int | `256` | Override GEMV SG16 work-group size (min 32). Controls rows-per-WG: rows = WG / 16 |
| `AILA_FUSE_RESIDUAL_ADD` | bool | `false` | Fuse residual +=hidden into O-proj and down GEMV outputs (experimental, -1.5% decode regression on tested config) |
| `AILA_BNB4_FUSED_PREFILL` | int | `-1` (auto) | Prefill GEMM mode: `1` forces fused NF4 GEMM, `0` forces oneDNN JIT GEMM, `-1` (auto) uses fused XMX for 9B+ with XMX enabled and oneDNN for 4B/0.8B |
| `AILA_Q35_PREFILL_XMX` | bool | auto | Qwen3.5 prefill dense-NF4 XMX Joint Matrix GEMM on A770. Defaults to `true` for 9B (`hidden_size >= 4096`) and `false` for 4B/0.8B |
| `AILA_BNB4_GEMM_XMX` | bool | unset | Process-wide override for the A770 dense-NF4 XMX GEMM: `1` forces it on, `0` forces it off. Unset follows the per-backend default (on for Qwen3.5 prefill, off elsewhere) |
| `AILA_BNB4_GEMM_XMX_BK` | int | `16` | A770 XMX GEMM K-block width (`16`/`32`/`64`). Measured negative: BK=32/64 stage larger SLM tiles and lose occupancy; keep `16` for production |
| `AILA_BNB4_GEMM_XMX_TILE` | int | `0` | A770 XMX GEMM BM/BN tile for the M>=64/N>=64 prefill path: `0`=64x64, `1`=64x32, `2`=32x64, `3`=32x128, `4`=128x32, `5`=64x128, `6`=128x64. Swept on Qwen3.5-9B: 64x64 wins at M=128 and M=512; keep `0` for production |
| `AILA_BNB4_GEMM_V2_SWIZZLE` | bool | `false` | DiffusionGemma/A770 v2 NF4 GEMM B-tile K-chunk-major SLM layout candidate; `0` keeps the validated legacy v2 layout |

---

## Decode Attention (AILA_ATTN_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_ATTN_JM` | int | `1` | Decode attention joint-matrix (XMX) mode: `0` = off, `1` = auto (use if available), `2` = force |
| `AILA_ATTN_JM_TILE` | int | `-1` (auto) | Force joint-matrix tile ID: `0` = 16×32×16, `1` = 32×32×16, `2` = 8×8×16. Auto-selects the largest tile when `-1` |
| `AILA_ATTN_DECODE_ROWS` | int | `1` | Batched decode: `1` merges eligible decode rows into one attention launch (baseline for head_dim 256 below threshold, exact partial merge for head_dim 128/256 at or above threshold, and fused Joint-Matrix for head_dim 128 below 128 tokens), `0` restores per-row decode attention calls. Only window/sink and mixed-threshold rows keep per-row dispatch |
| `AILA_ATTN_DECODE_WG` | int | `512` | Decode attention work-group size |
| `AILA_ATTN_DECODE_WINDOW` | int | `0` | Decode attention sliding window size (`0` = full context). Positive values restrict each decode step to recent tokens for significant throughput gains |
| `AILA_ATTN_DECODE_WINDOW_START` | int | `-1` | Enable decode window only after context length exceeds this threshold. Auto = `max(512, window)` when `-1` |
| `AILA_ATTN_DECODE_SINK` | int | `-1` | Number of prefix sink tokens kept with the recent window. Auto = `0` when `-1` |

### Attention Window Tuning

For multi-turn chat on Arc A770, a practical starting point:
```
AILA_ATTN_DECODE_WINDOW=128
AILA_ATTN_DECODE_WINDOW_START=512
AILA_ATTN_DECODE_SINK=0
```

Window mode is a quality/speed trade-off. Keep `WINDOW=0` for strict quality parity.

---

## Qwen3.5 Hybrid (AILA_Q35_*)

### Linear Attention

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q35_LINEAR_DELTA` | bool | `true` | Qwen3.5 linear attention mode: `1` = GPU DeltaNet recurrent path, `0` = legacy attention fallback (requires `AILA_Q35_ALLOW_UNSUPPORTED_LEGACY_LINEAR`) |
| `AILA_Q35_ALLOW_UNSUPPORTED_LEGACY_LINEAR` | bool | `false` | Allow legacy linear attention path (not recommended) |
| `AILA_Q35_EXPERIMENTAL_GQA_FASTPATH` | bool | `false` | Enable experimental GQA grouped fast path for linear attention |
| `AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU` | bool | `false` | Run grouped linear attention on GPU (experimental) |
| `AILA_Q35_FORCE_HOST_GROUPED_LINEAR` | bool | `false` | Force host-side grouped linear attention fallback |

### LM Head

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q35_DIRECT_TIED_LM_HEAD` | bool | `false` | Force direct tied LM head (use embed weight directly instead of transposed copy) |

### Vision

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q35_VISION_THREADS` | int | `0` (auto) | CPU threads for vision image preprocessing. Auto = `std::thread::hardware_concurrency()` |
| `AILA_Q35_VISION_SPLIT_QKV_BIAS_FUSED` | int | `0` | Vision split-QKV bias fusion mode (experimental) |
| `AILA_Q35_VISION_LINEAR_BIAS_FUSED` | int | `0` | Vision linear bias fusion mode (experimental) |
| `AILA_Q35_VISION_FC1_GELU_FUSED` | int | `0` | Vision FC1+GELU fusion mode (experimental) |
| `AILA_Q35_VISION_MIN_TOKENS` | int | from config | Minimum vision tokens (override) |
| `AILA_Q35_VISION_MAX_TOKENS` | int | from config | Maximum vision tokens (override) |
| `AILA_Q35_VISION_MIN_PIXELS` | int | from config | Minimum pixels (override) |
| `AILA_Q35_VISION_MAX_PIXELS` | int | from config | Maximum pixels (override) |
| `AILA_Q35_VISION_PATCH` | int | from config | Vision patch size (override) |
| `AILA_Q35_VISION_MERGE` | int | from config | Vision merge size (override) |

### Debug / Prefill

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q35_PREFILL_CHUNK` | int | `512` | Qwen3.5 prompt prefill chunk size. Caps temporary attention/runtime scratch memory during long prompt or incremental prefill. Set to `0` to disable chunking |
| `AILA_Q35_PREFILL_STEP` | int | `64` | Qwen3.5 DeltaNet recurrent-state checkpoint interval used for context rollback; must be positive. Overridden by `--q35-prefill-step` |
| `AILA_Q35_PREFILL_TOKENWISE` | bool | `false` | Tokenwise prefill mode (debug only — feeds tokens one at a time) |

---

## DiffusionGemma

The Aila DiffusionGemma backend does **not** require
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS`. It partitions runtime storage so that
no individual Aila allocation exceeds the Level Zero limit.

That Unified Runtime variable is only needed by the separate Python/Transformers
checkpoint reconstruction or reference workflow when it creates allocations
larger than 4 GiB. Do not treat it as an Aila runtime requirement.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_DIFFUSION_STEPS` | int | `min(model config, 25)` | Override the number of canvas denoising steps. The 26B checkpoint has been validated to converge at 25 steps; larger values remain available for quality experiments |
| `AILA_DG_CONFIDENCE_THRESHOLD` | float | `0.02` | Entropy threshold for canvas early stopping (`0.0` disables early stopping) |
| `AILA_DG_STABILITY_THRESHOLD` | int | `1` | Consecutive stable steps required to trigger canvas early stopping |
| `AILA_DG_PARALLEL_CANVASES` | bool | `true` | Process multiple diffusion canvases in parallel |
| `AILA_DG_GPU_SAMPLING` | bool | `true` | GPU-resident Gumbel noise generation and canvas token sampling |
| `AILA_DG_GPU_ROUTER` | bool | `true` | GPU-resident MoE router Top-K expert selection and softmax weights calculation |
| `AILA_DG_FUSED_ATTN` | bool | `true` | Enable XMX/SLM fused masked attention kernel for DiffusionGemma |
| `AILA_DG_FUSED_DECODER` | bool | `true` | Enable fused decoder post-attention and post-FFN norm & residual accumulation |
| `AILA_DG_FUSED_GATE_UP` | bool | `true` | Enable fused gate+up projection with activation |
| `AILA_DG_FUSED_POST_LOGITS` | bool | `true` | Enable fused post-logits soft-capping, temperature scaling, stats, and argmax |
| `AILA_DG_SOFT_EMBED_XMX` | bool | `true` | Enable XMX Joint Matrix accelerated Soft-Embedding matmul |
| `AILA_DG_RELEASE_VISION` | bool | `false` | Free vision tower weights from GPU memory after image encoding to save VRAM for diffusion generation |
| `AILA_DG_MOE_BATCHED` | int | `1` | DiffusionGemma MoE batched-transfer path: single flat gather + one d2h for all expert groups instead of one blocking h2d/d2h per group. Bitwise identical to the per-group host path (`0` restores it) |
| `AILA_DG_MOE_GPU` | bool | `false` | Opt-in exact GPU-resident MoE output path. Keeps the same per-expert dequantization and oneDNN matmuls as the batched path, then performs the stable expert-order bf16 accumulation on GPU to avoid the output D2H, host accumulation, and result H2D |
| `AILA_DG_MOE_GROUPED_XMX` | bool | `false` | Experimental grouped NF4/XMX MoE GEMM used only with `AILA_DG_MOE_GPU=1`. Double-quant absmax decoding is supported, but the XMX accumulation order is not bitwise identical to oneDNN; do not enable for exact-token production runs |
| `AILA_DG_MOE_HARNESS` | bool | `false` | Run the host MoE reference and compare enabled batched/GPU candidates byte-for-byte for every layer. Adds blocking copies and duplicate computation; validation only |
| `AILA_DG_MOE_DIAG` | bool | `false` | Enable diagnostic logging for MoE weights and routing |
| `AILA_DG_MOE_TRACE` | bool | `false` | Detailed trace logging for MoE layer execution |
| `AILA_DG_DEBUG_PROBES` | bool | `false` | Enable expensive host-side finite-value probes in the DiffusionGemma decoder |
| `AILA_DG_DUMP_DIR` | path | empty | Dump DiffusionGemma intermediate tensors and `committed_tokens.bin` for numerical comparison |
| `AILA_DG_TOKEN_DUMP` | path | empty | Dump only the committed int32 token stream; unlike `AILA_DG_DUMP_DIR`, this does not enable intermediate tensor dumps or their synchronization overhead |
| `AILA_DG_DUMP_LAYER` | int | `0` | Layer selected by `AILA_DG_DUMP_DIR` |
| `AILA_DG_DUMP_DECODER` | bool | `false` | Dump decoder/self-conditioning tensors instead of the causal encoder layer |
| `AILA_DG_REUSE_LAYER_SCRATCH` | bool | `true` | Reuse compact raw attention scratch buffers across DiffusionGemma layers; `0` restores per-layer allocations for A/B or stale-data diagnosis |
| `AILA_DG_REUSE_LAYER_SYNC` | bool | `false` | When layer scratch reuse is enabled, omit the production layer-tail queue drain; set `1` for the staged synchronization A/B or stale-data diagnosis |
| `AILA_DG_SOFTMAX_INPLACE` | bool | `true` | Reuse dead f32 logits as the exact-sum softmax workspace before soft-embedding; `0` restores the original row-serial kernel; decoder dump mode preserves row 0 before the overwrite for bitwise A/B comparison |
| `AILA_DG_SOFTMAX_DUMP_DIR` | path | empty | Debug-only row-0 dump of soft logits, bf16 probabilities, and soft embedding; does not enable layer/decoder dumps, but adds an explicit synchronization |
| `AILA_DG_MOE_ROUTE_HIST` | bool | `false` | Print the stable expert-major MoE route histogram (`M` rows per expert, active experts, and maximum M) used by the host/GPU paths |
| `AILA_PROFILE_DG` | bool | `false` | Synchronize DiffusionGemma decoder sub-phase boundaries and report projection, attention, dense MLP, router, MoE, and post-processing wall time |

---

## Qwen3.8-Flash-Next (AILA_Q4E_*)

Routed experts are partitioned across two cooperating resident tiers: `Qwen4ExpAdaptiveCache` (up to 2,800 GPU-resident expert slots, ~8.6 GB on a 16 GB A770) and `Qwen4ExpHostArena` (51.44 GiB of Windows large-page host RAM holding the 21,776 cold experts). Cold experts are streamed directly from the host arena via a dedicated in-order `copy_q_`. Each expert slot holds compact packed NF4 `gate_up` + `down` matrices plus nested `u8` absmax codes (2,534,400 bytes/expert).

### Expert Residency & Staging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q4E_EXPERT_CACHE_MB` | int | `0` (auto) | GPU expert slot pool budget. Auto = device memory minus static weights, session KV/recurrent state for `max_seq_len`, and `AILA_Q4E_VRAM_RESERVE_MB` (floored at 1024 MB). |
| `AILA_Q4E_VRAM_RESERVE_MB` | int | `1400` | Headroom kept free by the auto cache size for runtime buffers, oneDNN scratchpad, and display (~1.2–1.4 GB free VRAM at peak 8K prefill). |
| `AILA_Q4E_MAX_GPU_SLOTS` | int | `3456` (non-MTP) / `3072` (MTP) | Maximum number of resident expert slots allocated in GPU VRAM (`72 × 48 = 3,456` slots when MTP is disabled and `AILA_Q4E_LM_HEAD_Q8=1`, or `64 × 48 = 3,072` slots under `qwen38-a770` when `AILA_Q4E_MTP_DRAFT=1` / `AILA_Q4E_SERVER_MTP=1` to reserve `705 MiB` for the draft model + 8K prefill scratch headroom). |
| `AILA_Q4E_LM_HEAD_Q8` | bool | `true` (preset) | Quantize the `[248320, 2560]` LM head to block-64 INT8 at load, saving 568.4 MiB of static VRAM and halving decode LM-head memory traffic. |
| `AILA_Q4E_STAGING_SLOTS` | int | `16` | Number of expert slots per GPU staging bank (2 banks by default). |
| `AILA_Q4E_HOT_RAM_MB` | int | `32768` (preset) | Host RAM budget for `Qwen4ExpHostArena` large-page complement. |
| `AILA_Q4E_EXPERT_READER` | string | `readfile` (preset) | Reader backend used during initial model load (`readfile` or `mmap`). |
| `AILA_Q4E_DIRECT_STAGE_NO_HOST_WAIT` | bool | `true` | Skip host-side `staging_copy_.wait_and_throw()` when staging directly from `Qwen4ExpHostArena` onto the in-order `copy_q_`. |
| `AILA_Q4E_STAGE_DEQUANT_FENCE` | bool | `true` | Release each prefill staging bank as soon as `submit_dequant` completes instead of waiting for `submit_scatter`. |

### Prefill & Decode Kernels / State

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q35_PREFILL_CHUNK` | int | `8192` (preset) | Maximum prompt slice length per prefill forward. |
| `AILA_Q4E_PREFILL_XMX` | bool | `true` (preset) | Enable XMX joint-matrix / oneDNN paths for dense NF4 prefill projections. |
| `AILA_BNB4_ONEDNN_MIN_M` | int | `128` | Minimum sequence length `M` at which dense NF4 projections (`N = 2560` etc.) route to vectorized `dequantize_weight` + oneDNN `ab` MatMul. |
| `AILA_Q4E_DIRECT_PREFILL_IO` | bool | `true` (preset) | Read/write prefill projection inputs and outputs directly without intermediate scratch copies. |
| `AILA_Q4E_PHASE_SCRATCH` | bool | `true` (preset) | Share one phase-multiplexed device buffer across non-overlapping attention, GDN (`raw`), and MoE (`moe_down_acc`) prefill tensors. |
| `AILA_Q4E_PREFILL_EXPERT_XMX` | bool | `true` | Use batched oneDNN XMX GEMM for prefill MoE chunks (`count >= AILA_Q4E_PREFILL_EXPERT_XMX_GATE_ROWS`, default `1024`; adaptive band `MIN_M = 64` for `count < 4096` and `192` for `count >= 4096`). |
| `AILA_Q4E_CPU_OVERLAP` | bool | `true` | Enable CPU/GPU heterogeneous decode MoE execution: GPU computes resident hot experts while the physical-core-pinned CPU pool computes cold experts from `Qwen4ExpHostArena`. |
| `AILA_Q4E_CPU_POOL_PIN` | bool | `true` | Pin host decode thread to Core 0, PLE worker to Core 1, `CpuWorker` to Core 2, and `CpuTilePool` workers to Cores 3..N-1. |
| `AILA_Q4E_CPU_POOL_SPIN_US` | int | `20000` | Spin duration in microseconds before `CpuTilePool` workers sleep on `workCv_`. |
| `AILA_Q4E_CPU_RESULT_HOST_READ` | bool | `true` | Read CPU cold-expert down outputs directly from pinned host USM inside `moe_ordered_accumulate_cpu_host_rows` with by-value route weights. |
| `AILA_Q4E_MOE_COPY_VIA_COPYQ` | bool | `true` | Route CPU-complement host copies through the dedicated in-order copy queue. |
| `AILA_Q4E_CPU_THREADS` | int | `clamp(cores - 3, 1, 15)` | Number of `CpuTilePool` worker threads (`15` on an 18-core host, leaving 3 cores for the main submit thread, PLE worker, and OS). `PinCurrentThreadToNonWorkerCores()` automatically tracks this count. |
| `AILA_Q4E_CPU_POOL_CLAIM` | int | `1` | Number of tiles claimed per atomic fetch in `CpuTilePool::Consume`. |
| `AILA_Q4E_DECODE_DEFER_CPU_JOIN` | bool | `true` | Defer `JoinResidentCold` / `FinishResidentCpuRead` from layer `L` to layer `L+1` after `router_softmax_topk_device(L+1)` submission (requires `UR_L0_BATCH_SIZE >= 32`, preset to `64`). |
| `AILA_Q4E_DECODE_EXACT256_MIN_LEN` | int | `256` | Minimum attended length to dispatch `attention_decode_exact_head256_partial_merge` for `head_dim=256`. |
| `AILA_Q4E_ADAPTIVE_SWAP_FREQ` | int | `8` | Decode token interval for online hot-expert promotion into `Qwen4ExpAdaptiveCache`. |
| `AILA_Q4E_MAX_SWAPS` | int | `4` | Maximum expert swaps executed per swap interval. |
| `AILA_Q4E_MAX_SWAPS_PER_LAYER` | int | `2` | Maximum expert swaps allowed in a single layer per swap interval. |
| `AILA_Q4E_ADAPTIVE_SWAP_MIN_POS` | int | `48` | Minimum sequence position before online decode expert swaps begin. |
| `AILA_Q4E_ADAPTIVE_SWAP_DECAY` | bool | `true` | Multiply `Qwen4ExpAdaptiveCache` routing usage counters by `0.50` at end-of-prefill and `0.85` (`0.92` if `AILA_Q4E_ADAPTIVE_SWAP_FREQ <= 8`) after each swap interval. |
| `AILA_Q4E_VERIFY_LINEAR_REUSE` | int | `3` | 2-row NF4 weight-reuse level during multi-row verification decode (`0` = off, `1` = `2560x16480`, `2` = +`6144x2560`, `3` = +`2560x13312`, `2560x7168`, `2560x1280`, `2560x10240`, `2560x2560`, `640x2560`). |
| `AILA_Q4E_MTP_HIDDEN_REUSE` | bool | `true` | Enable 2-pair NF4 weight reuse for 4-row `2560x2560` MTP draft hidden projections. |
| `AILA_Q4E_SERVER_MTP` | bool | `false` | Enable speculative 2-token MTP draft/verify runtime on `LoadedModel::LoadQwen4ExpNf4`. |
| `AILA_Q4E_MTP_MAX_CONTEXT` | int | `2048` | Maximum sequence context length tracked by `Qwen4ExpMtpRuntime` before gracefully falling back to standard T1 decode. |
| `AILA_Q4E_CPU_COMPLETION_SINGLE_PASS` | bool | `true` | Fuse CPU cold-expert `gate_up`, inline `h` quantization, and completion-ordered `down` tiles into a single `CpuTilePool::ParallelFor` pass per layer. |
| `AILA_Q4E_DECODE_STRIDE_SNAPSHOT` | bool | `false` | Capture periodic `AILA_Q4E_PREFILL_STEP` recurrent snapshots during unforced single-token decode steps (off by default to avoid 64-token D2H stalls and preserve prompt checkpoints). |
| `AILA_BNB4_GEMV_INPUT_PREFETCH_OFF` | bool | `true` | Skip redundant per-subgroup LSC input prefetches in BF16 NF4 decode GEMV kernels. |
| `AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M` | int | `192` (preset) | Minimum padded band height `M` for prefill expert XMX batched GEMMs. |
| `AILA_Q4E_PREFILL_EXPERT_XMX_MAX_M` | int | `256` | Maximum band height for `count >= 1024` prefills: hot experts are split into `<= MAX_M` bands and the resulting `M` is rounded up to a 64-row multiple, so every full chunk hits one of the pre-warmed `(B=16, M in {64,128,192,256})` oneDNN primitives instead of compiling mid-prefill. Requires the bucketed route (`AILA_Q4E_XMX_BUCKET_ROUTE`). |
| `AILA_Q4E_VERIFY_CPU_COLD` | bool | `true` | Serve a 2..4-row verification window's cold experts through the same arena CPU batch path as `count == 1` decode, one row at a time and capped by `AILA_Q4E_CPU_MAX_EXPERTS`. |
| `AILA_Q4E_VERIFY_CPU_BATCH` | bool | `true` | Run the whole 2..4-row verification window's cold experts as ONE `CpuWorker` job (one join per window) via `StartArenaColdRows` / `JoinArenaColdRowsRow`. |
| `AILA_Q4E_VERIFY_CPU_MERGE` | bool | `true` | Group identical `(row, expert)` CPU tasks of a batched verification window into one tile-pool group instead of recomputing per row. |
| `AILA_Q4E_VERIFY_DECODE_KERNELS` | bool | `true` | Route a 2..4-row verification window's dense, shared-expert and PLE projections through `ForwardDecodeRows` and the two-launch fused hyper-connection, while the LM head keeps its single-launch multi-row kernel (`AILA_Q4E_VERIFY_SMALL_ROWS` still takes precedence and also switches the LM head and QSA indexer to per-row GEMVs). |
| `AILA_Q4E_VERIFY_DECODE_ATTN` | bool | `true` | Multi-row verification windows use the single-row decode attention kernels per query row (or the fused 2-row pair kernel) instead of the multi-row prefill attention. |
| `AILA_Q4E_VERIFY_ATTN_PAIR` | bool | `true` | With `AILA_Q4E_VERIFY_DECODE_ATTN`, a 2-row window at `start_pos + 1 >= 1024` runs the fused `q4e_attention_decode_pair` kernel. |
| `AILA_Q4E_VERIFY_DECODE_PREPARE` | bool | `true` | Replace the 6-launch `split_qgkv + head_rms_norm` sequence on 2..4-row verification windows with two fused `ops::decode_prepare_qgkv_packed_partial` calls. |
| `AILA_Q4E_PLE_PREFETCH_MIN` | int | `1` | Minimum token count at which `Qwen4ExpPle::prefetch_gather_pages` issues batched Win32 `PrefetchVirtualMemory` calls over per-head PLE NF4 weight/absmax ranges. |
| `AILA_Q4E_XMX_BUCKET_ROUTE` | bool | `true` | Bucket `(token, rank)` by expert once per layer in `O(count * top_k)` instead of rescanning routes per chunk. |
| `AILA_Q4E_XMX_SINGLE_PLAN_COPY` | bool | `true` | Pack each XMX chunk's `bases + tokens + ranks + slots` into a disjoint slice of pinned host USM and upload in a single `memcpy`. |
| `AILA_Q4E_GDN_CHUNK_MODE` | int | `1` (preset) | GDN prefill mode: `0` = recurrent, `1` = fused FP32 `chunk_scan` (`T = 16`), `2` = BF16 XMX products. |
| `AILA_QSA_HEADS_PER_WG` | int | `25` | QSA prefill attention kernel mode (`25` = 4 Q heads in registers + `vec4` KV loads + 1 barrier per 4-token block; `2` = 2-head fallback). |
| `AILA_QSA_FAST_BLOCK_SCORES` | bool | `true` | Use 8-block sub-group shuffle kernel for QSA compressed-K block scoring. |
| `AILA_QSA_SKIP_DENSE_SCORES` | bool | `true` | Skip QSA block scoring on wholly dense query tiles (`tile_max_blocks <= block_topk`). |
| `AILA_QSA_QUERY_TILE` | int | unset (budget-derived, 16..1024) | Query rows per QSA prefill tile. When unset the tile grows to whatever the `AILA_QSA_LOGITS_BUDGET_BYTES` score buffer allows at the window's block count (clamped to 16..1024), cutting the three launches per tile at long context; an explicit value is honored as-is. |
| `AILA_Q4E_HC_FUSED_SIGMOID` | bool | `true` | Fuse mix-up sigmoid directly into `ops::hc_mix_reduce`. |
| `AILA_Q4E_MOE_SHARED_FUSED` | bool | `true` | Fuse gated shared-expert addition into `ops::moe_ordered_accumulate_down_rows`. |
| `AILA_Q4E_INLINE_CHECKPOINT` | bool | `true` (preset) | Enable inline recurrent/KV checkpoint capture at `checkpoint_after`. |
| `AILA_Q4E_LONG_INLINE_CHECKPOINT` | bool | `true` (preset) | Allow inline checkpoint capture inside long untiled prefill slices (`count > 64`), collapsing `7760 + 4` splits into a single forward. |
| `AILA_Q4E_SKIP_REDUNDANT_END_SNAPSHOT` | bool | `true` | Skip the redundant end-of-slice `CaptureSnapshot` when an inline interior checkpoint was already captured. |
| `AILA_Q4E_HOST_EMBED` | bool | `true` | Gather input embedding rows from large-page host memory instead of keeping the 1.3 GB table in VRAM. |
| `AILA_Q4E_DENSE_GEMV` | bool | `true` | Single-row bf16 projections use native GEMV instead of oneDNN. |
| `AILA_Q4E_HC_FUSED` | bool | `true` | Decode hyper-connection runs as two fused kernels (down + inject, up + mix). |
| `AILA_Q4E_TOKEN_GRAPH` | bool | `false` | Experimental whole-token SYCL command-graph capture and replay for single-token decode. |
| `AILA_Q4E_GRAPH_GDN_STATE` | bool | `false` | Use device-owned GDN conv-ring state for captured single-token decode. |

### Profiling / Debug

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_PROFILE_Q4E_DECODE` | bool | `false` | Per-forward `[Q4E Adaptive Profile]` line: total time, router sync, cache hits/misses, staged experts, and DMA MiB. |
| `AILA_PROFILE_Q4E_STAGES` | bool | `false` | With `AILA_PROFILE_Q4E_DECODE`, synchronize after every stage and log per-stage time (relative cost only). |
| `AILA_Q4E_WORK_SUMMARY` | bool | `false` | Print per-forward `[Q4E AdaptiveWork]` summary counters. |
| `AILA_DEBUG_Q4E_DUMP_DIR` | path | empty | Dump per-layer hidden states, final mixer output, and logits for numerical comparison. |

---

## TTS (AILA_TTS_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_BF16_GEMV` | int | `1` | Enable native bf16 GEMV kernel for TTS decode (SG=16, vec8+FMA). Set to `0` to fall back to oneDNN matmul. Provides ~9x TTS throughput improvement |
| `AILA_TTS_STREAM_BATCH` | int | `4` | Number of audio frames per streaming TTS callback chunk. Higher values reduce callback overhead at the cost of increased latency |
| `AILA_TTS_MIMI_TRANSPOSE_CONV_VEC8` | bool | `true` | Reorder Mimi transpose-conv weights for contiguous vec8 input/weight loads |
| `AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL` | bool | `true` | Fuse Mimi decoder kernel-1 conv2 output with residual accumulation |

## TTS Reference Embedding Cache (AILA_REF_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_REF_CACHE_DIR` | string | `""` | Directory for persistent speaker embedding cache. When set, cache files are stored as `<dir>/<basename>.ref.bin`. When empty (default), cache files are stored alongside the reference audio (`<audio_path>.ref.bin`). |

The speaker embedding cache avoids re-extracting the same reference audio on every TTS run.
Embeddings are cached in memory (session-lifetime) and persisted to disk.
Set `AILA_REF_CACHE_DIR` to a shared directory to reuse embeddings across different reference audio locations.

---

## Profiling and Debugging

### Performance Profiling (AILA_PROFILE_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_PROFILE_Q35_PREFILL` | bool | `false` | Profile Qwen3.5 prefill (logs timing per operation) |
| `AILA_PROFILE_Q35_PREFILL_EVERY` | int | `1` | Profile every Nth prefill |
| `AILA_PROFILE_Q35_DECODE` | bool | `false` | Profile Qwen3.5 decode (logs timing per step) |
| `AILA_PROFILE_Q35_DECODE_EVERY` | int | `32` | Profile every Nth decode step |
| `AILA_PROFILE_Q35_HOST_ONLY` | bool | `false` | Measure host submission time only (skip GPU sync). Diagnostic only |
| `AILA_PROFILE_Q3_PREFILL` | bool | `false` | Profile Qwen3/Qwen3-ASR prefill operations |
| `AILA_PROFILE_Q3_PREFILL_EVERY` | int | `1` | Profile every Nth Qwen3/Qwen3-ASR prefill |
| `AILA_PROFILE_Q3_DECODE` | bool | `false` | Profile Qwen3/Qwen3-ASR decode operations |
| `AILA_PROFILE_Q3_DECODE_EVERY` | int | `32` | Profile every Nth Qwen3/Qwen3-ASR decode step |
| `AILA_PROFILE_Q3_HOST_ONLY` | bool | `false` | Measure Qwen3/Qwen3-ASR host submission time only. Diagnostic only |
| `AILA_Q35_VISION_PROFILE` | bool | `false` | Profile vision encoder |
| `AILA_Q35_VISION_PROFILE_BLOCKS` | bool | `false` | Profile individual vision transformer blocks |
| `AILA_Q35_VISION_PROFILE_PREP` | bool | `false` | Profile vision image preprocessing |
| `AILA_TTS_PROFILE` | int | `0` | Enable detailed TTS profiling (talker, mimi, text projection stage timing). Set to `1` to enable |

### Debug Output (AILA_DEBUG_*)

Shared inference also supports `AILA_EXECUTION_TRACE_DIR` (directory, unset by
default), read once before its DeviceRuntime is created. It writes exclusive,
per-domain JSONL files with durable host execution/wait/free boundaries; no
extra GPU synchronization is added. It changes host timing and is diagnostic
only. Trace failures stop new shared inference work without skipping cleanup.
This does not instrument legacy Context users or individual kernels. The HTTP
runner enables it only via `--mode probe --execution-trace`, otherwise removes
an inherited path. See [execution-tracing.md](execution-tracing.md).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_DEBUG_PROMPT_TEXT` | bool | `false` | Log the fully rendered prompt text before prefill, including chat-template special markers such as `<|im_start|>` and `<|im_end|>` |
| `AILA_DEBUG_CHAT_TEMPLATE` | bool | `false` | Log the selected chat template source/version without dumping the full prompt |
| `AILA_SERVER_TOOL_DIAGNOSTICS` | int | `0` | HTTP Server only: a nonzero value logs invalid tool blocks at WARN with the request ID, parse reason and at most 4096 bytes of JSON-escaped block text. Read once on the first tool parse failure; set before starting the Server. May include generated argument content; diagnostic only |
| `AILA_DEBUG_PROMPT_TEXT_MAX_CHARS` | int | `20000` | Maximum rendered prompt characters to log when `AILA_DEBUG_PROMPT_TEXT=1`. Set to `0` for unlimited output |
| `AILA_DUMP_LOGITS` | bool | `false` | Dump intermediate Qwen3-ASR audio-encoder tensors as `debug_cpp_*.bin` files in the process/worker working directory. Diagnostic only |
| `AILA_DEBUG_TOKEN_IDS` | bool | `false` | Log token IDs during generation |
| `AILA_DEBUG_Q35_LOGITS` | bool | `false` | Log top-N logits after prefill |
| `AILA_DEBUG_Q35_LAYER_STATS` | bool | `false` | Log per-layer statistics |
| `AILA_DEBUG_Q35_LAYER_DETAIL` | int | `-1` | Log detailed info for a specific layer index |
| `AILA_DEBUG_Q35_LINEAR_COMPARE` | bool | `false` | Compare linear layer outputs (GPU vs host reference) |
| `AILA_DEBUG_Q35_LINEAR_COMPARE_LAYER` | int | `-1` | Target layer index for linear comparison |
| `AILA_MIMI_DEBUG` | bool | `false` | Enable detailed step-by-step debug logging for Mimi Vocoder |

---

## Other

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_DEVICE_SAMPLING` | bool | `true` | Device-side sampling: `0` = force host fallback, `1` = allow GPU sampling |
| `AILA_PRINT_MATRIX_COMBOS` | bool | `false` | Print all supported joint-matrix (XMX) tile combinations at startup |
| `AILA_NO_MROPE` | bool | `false` | Disable multimodal rotary position IDs for embedded media inputs. Debug/compatibility use only |
| `AILA_YOLO_DUMP_DIR` | directory path | empty | Directory to dump YOLO detector debug visualizations and intermediate predictions |

---

## Vision Attention Tuning (AILA_ATTN_VISION_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_ATTN_VISION_BIDI_EXACT64_TILE` | int | `192` | Tile size for vision bidirectional attention (head_dim=64 exact path) |
| `AILA_ATTN_VISION_BIDI_EXACT64_SG` | int | `16` | Sub-group size for vision bidirectional attention |
| `AILA_ATTN_VISION_BIDI_EXACT64_VEC` | bool | `true` | Enable vectorization for vision bidirectional attention |
| `AILA_ATTN_VISION_BIDI_EXACT64_VUNROLL` | bool | `true` | Enable V-unroll for vision bidirectional attention |

