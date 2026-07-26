# Environment Variables

Aila supports the following environment variables for runtime configuration. Boolean flags accept `0`/`false` (off) and `1`/`true` (on). Integer values are parsed from decimal strings.

---

## Windows Runtime Isolation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_RUNTIME_DLL_DIR` | directory path | directory containing `AilaShared.dll` | Directory containing `AilaWorker.exe` and the private oneAPI runtime. Relative values resolve relative to AilaShared.dll; absolute values are accepted. The selected path is normalized to an absolute path internally. An unset or empty value enables the legacy flat layout. |
| `AILA_WORKER_ENV_PASSTHROUGH` | `;`-separated variable names | `""` | Names of scrubbed GPU-runtime variables (see below) that the host explicitly wants forwarded to the worker anyway. Matched case-insensitively; surrounding spaces are ignored. Example: `AILA_WORKER_ENV_PASSTHROUGH=ZES_ENABLE_SYSMAN;UR_L0_USE_COPY_ENGINE`. |
| `AILA_KEEP_SYCL_CACHE_PERSISTENT` | flag | `0` | `Aila.exe` and `AilaWorker.exe` force `SYCL_CACHE_PERSISTENT=0` at startup because an enabled persistent cache crashes the bundled DPC++ runtime (null dereference while hashing oneDNN interop programs at the first JIT build). Set `1` to keep the inherited value, e.g. to test a runtime that fixes the bug. For the worker, the variable must also be forwarded via `AILA_WORKER_ENV_PASSTHROUGH=SYCL_CACHE_PERSISTENT`. |

The unset/empty fallback exists only for compatibility with legacy deployments.
When the proxy and runtime DLLs share one directory, exposing the proxy directory
to the host DLL loader also exposes the private runtime, so the flat layout does
not provide host DLL-search isolation. Python, ComfyUI, and other embedding hosts
must use the split layout and explicitly set `AILA_RUNTIME_DLL_DIR`.

The recommended release/integration layout keeps only `AilaShared.dll` in the
integration root, with `AilaWorker.exe`, `Aila.exe`, and the oneAPI runtime DLLs
under `aila_runtime/`:

```text
integration_root/
|-- AilaShared.dll
`-- aila_runtime/
    |-- AilaWorker.exe
    |-- Aila.exe
    `-- <oneAPI and other runtime DLLs>
```

For this layout, set `AILA_RUNTIME_DLL_DIR=aila_runtime` before the host loads
`AilaShared.dll`. Each initialized engine uses its own worker process. The worker
starts in the selected directory with a child-only `PATH` containing that
directory and Windows system directories. The host's DLL search path remains
unchanged.

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
| `AILA_INIT_WARMUP` | int | `-1` (auto) | Init warmup: `0` = skip, `1` = force, `-1` = auto (skip for unsupported specs) |

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
| `AILA_BNB4_FUSED_PREFILL` | bool | `true` | Enable fused NF4 dequant + matmul kernel for prefill. Disable only for debugging |

---

## Decode Attention (AILA_ATTN_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_ATTN_JM` | int | `1` | Decode attention joint-matrix (XMX) mode: `0` = off, `1` = auto (use if available), `2` = force |
| `AILA_ATTN_JM_TILE` | int | `-1` (auto) | Force joint-matrix tile ID: `0` = 16×32×16, `1` = 32×32×16, `2` = 8×8×16. Auto-selects the largest tile when `-1` |
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

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_DEBUG_PROMPT_TEXT` | bool | `false` | Log the fully rendered prompt text before prefill, including chat-template special markers such as `<|im_start|>` and `<|im_end|>` |
| `AILA_DEBUG_CHAT_TEMPLATE` | bool | `false` | Log the selected chat template source/version without dumping the full prompt |
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

---

## Vision Attention Tuning (AILA_ATTN_VISION_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_ATTN_VISION_BIDI_EXACT64_TILE` | int | `192` | Tile size for vision bidirectional attention (head_dim=64 exact path) |
| `AILA_ATTN_VISION_BIDI_EXACT64_SG` | int | `16` | Sub-group size for vision bidirectional attention |
| `AILA_ATTN_VISION_BIDI_EXACT64_VEC` | bool | `true` | Enable vectorization for vision bidirectional attention |
| `AILA_ATTN_VISION_BIDI_EXACT64_VUNROLL` | bool | `true` | Enable V-unroll for vision bidirectional attention |
