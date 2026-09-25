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
| `AILA_INIT_WARMUP` | int | `-1` (auto) | Init warmup: `0` = skip, `1` = force, `-1` = auto (enabled except for Qwen3-TTS and unsupported Qwen3.5 specs). Moves first-use kernel JIT out of the first request (~15 s for Qwen3.8-Flash-Next on A770) |
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

Routed experts stay in the memory-mapped checkpoint and are swapped into a GPU LRU
slot pool on demand. Each slot holds one expert's packed NF4 gate_up + down and either
compact u8 absmax codes (default for compatible nested checkpoints) or expanded f32 absmax.

### Expert Cache

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q4E_EXPERT_CACHE_MB` | int | `0` (auto) | GPU expert slot pool size. Auto = device memory (or the context allocation limit) minus the static weights, the full-attention KV for `max_seq_len`, and `AILA_Q4E_VRAM_RESERVE_MB`; at least 1024 MB. Prefill acquires a layer's experts in one batch only when the pool holds `num_experts + top_k` slots (~1.4 GB) |
| `AILA_Q4E_VRAM_RESERVE_MB` | int | `2560` | Headroom kept free by the auto cache size for runtime buffers, oneDNN scratch and the desktop. On a 16 GB A770 driving a display, ~1 GB free VRAM after warmup slowed all kernels ~2x |
| `AILA_Q4E_EXPERT_CHUNK_MB` | int | `1024` | Size of each device allocation backing the slot pool (Level Zero caps single allocations near 4 GB) |
| `AILA_Q4E_STAGING_MB` | int | `128` | Pinned host staging buffer for miss uploads; split in two halves so one fills while the other DMAs |
| `AILA_Q4E_FILL_THREADS` | int | `min(8, cores)` | Host threads (including the caller) that copy packed weights and absmax codes into staging, or expand absmax to f32 when compact mode is disabled |
| `AILA_Q4E_FILL_PARTS` | int | `8` | Parts each expert is split into across fill threads |
| `AILA_Q4E_ABSMAX_CODES` | bool | `true` | Keep nested u8 absmax codes in each GPU expert slot and reconstruct scales in the MoE kernels. Applies only when every expert layer has aligned nested NF4 metadata; otherwise falls back to f32 slots. `0` restores host f32 expansion. On this A770, slots shrink from 2,764,800 to 2,534,400 bytes (3463 to 3778 slots in the 9131 MB pool) |
| `AILA_Q4E_PREFETCH` | bool | `true` | Decode: route layer L's MoE input through layer L+1's router and stage the predicted experts on the fill threads while the GPU runs layer L+1's attention |
| `AILA_Q4E_PREFETCH_TOPN` | int | `10` | Number of most-probable predicted experts to prefetch |
| `AILA_Q4E_PREFILL_PREFETCH_MAX` | int | `48` | Maximum unique predicted experts to asynchronously prefetch per layer during prefill into the Hot Host RAM Pool (hides host fill latency behind GPU compute while preventing OS page-cache thrashing) |
| `AILA_Q4E_PREFILL_DEMOTE` | bool | `true` | Batched prefill keeps each layer's `slots / layers` most-used experts hot and queues the rest for eviction first, so decode after a long prompt still finds every layer's hottest experts (plain LRU would keep only the last layers) |
| `AILA_Q4E_PREDICT_STATS` | bool | `false` | With `AILA_PROFILE_Q4E_DECODE`, log how often the next-layer prediction matched the real routing |

### Kernels / State

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_Q4E_HOST_EMBED` | bool | `true` | Gather input embedding rows from the host mmap instead of keeping the 1.3 GB table in VRAM (the auto expert cache grows by ~460 slots). `0` uploads the table |
| `AILA_Q4E_DENSE_GEMV` | bool | `true` | Single-row bf16 projections (hyper-connection, router, shared-expert gate) use the native GEMV instead of oneDNN |
| `AILA_Q4E_HC_FUSED` | bool | `true` | Decode hyper-connection runs as two fused kernels (down + inject, up + mix) |
| `AILA_Q4E_MOE_SHARED_FUSED` | bool | `false` | Experimental gated shared-expert add inside MoE down. Changes a few BF16 values in layer 0 and caused a decode-route regression on the fixed benchmark; keep disabled by default |
| `AILA_Q4E_PREFILL_STEP` | int | `64` | Decode positions between recurrent-state snapshots used for context rollback |

### Profiling / Debug

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_PROFILE_Q4E_DECODE` | bool | `false` | Per-forward `[Q4E Profile]` line: total time, router sync, acquire (host fill / DMA wait / prefetch wait), cache hits and bytes uploaded |
| `AILA_PROFILE_Q4E_STAGES` | bool | `false` | With `AILA_PROFILE_Q4E_DECODE`, synchronize after every stage and log per-stage time (relative cost only) |
| `AILA_DEBUG_Q4E_DUMP_DIR` | path | empty | Dump per-layer hidden states, the final mixer output and logits for numerical comparison |
| `AILA_DEBUG_Q4E_ROUTE_TRACE` | path | empty | Write every expert-cache acquire as int32 `{layer, count, ids...}` for offline replacement-policy simulation with `perf/qwen4exp/q4e_cache_sim.py` |
| `AILA_DEBUG_Q4E_FAKE_HITS` | bool | `false` | Treat every expert as resident (no fill or DMA; outputs are wrong). Measures the decode floor at a 100% hit rate; use with `--bench`, since the garbage logits hit EOS at once |

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
