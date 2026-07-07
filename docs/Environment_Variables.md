# Environment Variables

Aila supports the following environment variables for runtime configuration. Boolean flags accept `0`/`false` (off) and `1`/`true` (on). Integer values are parsed from decimal strings.

---

## General

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_MODEL_DIR` | string | `""` | Default model directory (fallback for `-m` CLI argument) |
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

## Bitsandbytes 4-bit (AILA_BNB4_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_BNB4_CACHE_DEQUANT` | bool | `false` | Cache dequantized bf16 weights in GPU memory. Saves dequant cost but increases VRAM ~2 GB for 4B models. Recommended off for large prefill |
| `AILA_BNB4_GEMV_WG` | int | `256` | Override GEMV SG16 work-group size (min 32). Controls rows-per-WG: rows = WG / 16 |
| `AILA_FUSE_RESIDUAL_ADD` | bool | `false` | Fuse residual +=hidden into O-proj and down GEMV outputs (experimental, -1.5% decode regression on tested config) |
| `AILA_BNB4_FUSED_PREFILL` | bool | `true` | Enable fused NF4 dequant + matmul kernel for prefill. Disable only for debugging |
| `AILA_Q35_INTERNAL_PREFILL_STATE_SNAPSHOTS` | bool | `true` | Export intermediate Qwen3.5 linear recurrent-state snapshots inside one batched prefill. Final fresh/full foreground prefill temporarily suppresses this work, while cached-prefix and ASR prefill paths keep the default restore granularity. |

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
| `AILA_Q35_PREFILL_TOKENWISE` | bool | `false` | Tokenwise prefill mode (debug only — feeds tokens one at a time) |

---

## TTS (AILA_TTS_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_BF16_GEMV` | int | `1` | Enable native bf16 GEMV kernel for TTS decode (SG=16, vec8+FMA). Set to `0` to fall back to oneDNN matmul. Provides ~9x TTS throughput improvement |
| `AILA_TTS_STREAM_BATCH` | int | `4` | Number of audio frames per streaming TTS callback chunk. Higher values reduce callback overhead at the cost of increased latency |

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

---

## Vision Attention Tuning (AILA_ATTN_VISION_*)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AILA_ATTN_VISION_BIDI_EXACT64_TILE` | int | `192` | Tile size for vision bidirectional attention (head_dim=64 exact path) |
| `AILA_ATTN_VISION_BIDI_EXACT64_SG` | int | `16` | Sub-group size for vision bidirectional attention |
| `AILA_ATTN_VISION_BIDI_EXACT64_VEC` | bool | `true` | Enable vectorization for vision bidirectional attention |
| `AILA_ATTN_VISION_BIDI_EXACT64_VUNROLL` | bool | `true` | Enable V-unroll for vision bidirectional attention |
