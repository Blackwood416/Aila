# Aila

<p align="center">
  <b>An inference engine leveraging Arc graphics.</b><br>
  <a href="README_zh.md">中文</a>
</p>

---

> [!NOTE]
> This project is under active development and does not yet fully support all listed models. Some models performance may not be optimal.

A high-performance LLM inference engine for **Intel Arc GPUs**, built with **SYCL + oneDNN**. Features hand-optimized kernels for bitsandbytes 4-bit (NF4) quantized models, fused dequant+matmul, and GPU-accelerated DeltaNet recurrence for Qwen3.5 hybrid architectures.

## ✨ Features

- **⚡ Bitsandbytes 4-bit (NF4) inference** — run quantized models directly on Intel Arc with fused dequant+matmul kernels, hand-written GEMV decode, and fused gate+up+SiLU projection
- **🔢 Bfloat16 inference** — dense (unquantized) models via oneDNN matmul primitives
- **🏗️ Qwen3.5 Hybrid architecture** — full GPU acceleration for the dual attention (GQA + DeltaNet linear attention) architecture
- **📐 Qwen3 Dense architecture** — standard Transformer with GQA, QK-norm, and SwiGLU FFN
- 👁️ Vision (Qwen3.5) — image understanding with CPU preprocessing and GPU vision transformer
- 🎙️ Audio (Qwen3-ASR) — speech-to-text transcription with audio preprocessing and GPU-accelerated audio encoder. Supports both offline wav transcription and real-time streaming input ASR
- 🎯 **Forced Alignment** — word-level timestamp alignment from audio + text, with CJK character-level tokenization and LIS-based timestamp correction
- 🔊 Audio (Qwen3-TTS) — text-to-speech synthesis with Mimi Vocoder and native zero-shot voice cloning. Supports direct raw audio WAV generation, voice cloning via reference audio, and offline Mimi decoding
- 🎤 **CustomVoice / VoiceDesign** — named speaker presets (vivian, ryan, etc.) for instant voice selection and text-described voice styles via VoiceDesign for creative TTS control
- 🔉 **Streaming TTS** — real-time raw 24kHz mono f32 PCM audio output to stdout for low-latency streaming speech synthesis
- ⚡ **Native bf16 GEMV kernel** — hand-optimized SG=16 vec8+FMA bf16 GEMV for TTS decode, delivering 9x faster TTS (RTF 8.08 to 0.89 on 0.6B)
- 🔄 Streaming output — token-level streaming callback with abort support
- **💬 Interactive CLI** — multi-turn conversation with runtime commands (`/clear`, `/greedy`, `/sample`, etc.)
- **📊 Benchmark mode** — measure prefill and decode throughput separately
- **🔌 C API** — stable C FFI interface (Python, C#, Rust, Go, Java) — see [docs/C_API.md](docs/C_API.md)
- **💭 Chat formatting** — llama.cpp-style Jinja rendering, fixed Qwen3.5 template, structured reasoning/tool-call parsing, and JSONL chat stream events

## 📦 Supported Models

| Model | Architecture | Quantization | Vision | Audio (ASR / TTS) |
|-------|-------------|-------------|--------|-------------------|
| [Qwen3.5-0.8B](https://huggingface.co/Blackwood416/Qwen3.5-0.8B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ | ❌ |
| [Qwen3.5-4B](https://huggingface.co/Blackwood416/Qwen3.5-4B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ | ❌ |
| [Qwen3-0.6B](https://huggingface.co/Blackwood416/Qwen3-0.6B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ | ❌ |
| [Qwen3-4B](https://huggingface.co/Blackwood416/Qwen3-4B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ | ❌ |
| [Qwen3-ASR-1.7B](https://huggingface.co/Blackwood416/Qwen3-ASR-0.6B-BNB-NF4) | Dense + Audio Encoder | BNB NF4, dense | ❌ | ✅ (ASR) |
| [Qwen3-ASR-1.7B](https://huggingface.co/Blackwood416/Qwen3-ASR-1.7B-BNB-NF4) | Dense + Audio Encoder | BNB NF4, dense | ❌ | ✅ (ASR) |
| [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Blackwood416/Qwen3-ForceAligner-0.6B-BNB-NF4) | Dense + Audio Encoder | BNB NF4, dense | ❌ | ✅ (Alignment) |
| [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS (Voice Cloning) |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS (Voice Cloning) |
| [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS (Pre-trained Voices) |
| [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS (Pre-trained Voices) |
| [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | Talker + Mimi Vocoder | BF16 | ❌ | ✅ TTS (Instruct-based) |

Other Qwen3 / Qwen3.5 model sizes may work if they match the supported architecture pattern.

## 🔧 Requirements

### 🖥️ Hardware
- **Intel Arc A770** (16 GB) — primary development and test target
- Other Intel Arc discrete GPUs (A750, A580, A380, B580) with ≥8 GB VRAM should work
- Integrated GPUs (Xe-LP, Xe-LPG) may work for small models but are not tested

### 💿 Operating System
- **Windows 10 22H2** or later / **Windows 11**

### 💻 Software
- [Intel Arc Graphics Driver](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html)
- The `Aila-vX.Y.Z-win64.zip` release bundle includes all required runtime DLLs

## 📥 Installation

1. Install the latest **Intel Arc Graphics Driver**.
2. Download `Aila-vX.Y.Z-win64.zip` from the [Releases](https://github.com/Blackwood416/Aila/releases) page.
3. Extract to a directory of your choice.
4. Place your model files in a directory (e.g. `./models/qwen3.5-0.8B-bnb-nf4-offline/`).

## 📊 Benchmark

Benchmark on Intel Arc A770 16 GB, Qwen3.5-4B, pp=2048 tg=1024:

| Engine | Backend | Model | Prefill | Decode |
|--------|---------|-------|---------|--------|
| **Aila 0.1.3** | SYCL + oneDNN | Qwen3.5-4B BNB NF4 | **1649 tok/s** | 58 tok/s |
| llama.cpp b8996 | SYCL | Qwen3.5-4B Q4_K_XL | 1290 tok/s | 28 tok/s |
| llama.cpp b8996 | Vulkan | Qwen3.5-4B Q4_K_XL | 700 tok/s | **60 tok/s** |

Aila delivers the highest prefill throughput and competitive decode performance against Vulkan while using a more accurate NF4 4-bit quantization that retains vision capabilities.

TTS benchmark on Intel Arc A770 16 GB, Qwen3-TTS-12Hz-0.6B-Base:

| Engine | Model | RTF | Notes |
|--------|-------|-----|-------|
| **Aila** | Qwen3-TTS-12Hz-0.6B-Base | **0.89** | Native bf16 GEMV kernel |
| Aila (pre-optimization) | Qwen3-TTS-12Hz-0.6B-Base | 8.08 | oneDNN matmul only |

Real-time factor (RTF) < 1 means the engine synthesizes speech faster than real-time. The native bf16 GEMV kernel delivers a 9x TTS speedup.

## 🚀 Usage

### ⌨️ CLI Quick Start

```powershell
# Interactive conversation
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision

# Offline audio transcription (ASR)
Aila.exe -m ./models/Qwen3-ASR-1.7B --transcribe input.wav

# Forced alignment (word-level timestamps)
Aila.exe -m ./models/Qwen3-ForcedAligner-0.6B-BNB-NF4 --align-text "你好世界" --align-audio input.wav --align-lang Chinese

# Text-to-speech synthesis (TTS)
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base --synthesize "Hello world!" --output-wav output.wav

# TTS with voice cloning (zero-shot)
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base --synthesize "今天天气真好。" --ref ./reference_speaker.wav --output-wav cloned.wav

# Single prompt from JSON file
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision --messages-json prompt.json

# Single prompt from stdin
echo '{"messages":[{"role":"user","content":"hello"}]}' | Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --messages-json -

# Benchmark (greedy)
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision --bench --bench-pp 512 --bench-tg 128

# Benchmark (sampling)
Aila.exe -m ./models/Qwen3.5-4B-BNB-NF4-with-vision --bench --sample
```

### ⚙️ CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model <path>` | Model directory | `AILA_MODEL_DIR` env |
| `-s, --max-seq <N>` | Max sequence length | 4096 |
| `-t, --temperature <F>` | Sampling temperature | 0.7 |
| `-k, --top-k <N>` | Top-K sampling | 15 |
| `-p, --top-p <F>` | Top-P (nucleus) sampling | 0.95 |
| `--seed <N>` | Sampling RNG seed | (none) |
| `--greedy` / `--sample` | Decoding mode | sample |
| `--stream` / `--no-stream` | Force streaming on/off | auto |
| `--max-tokens <N>` | Max new tokens | 1024 |
| `--thinking-budget <N>` | Thinking budget: -1 off, 0 no-think, >0 cap | -1 |
| `--decode-chunk <N>` | Decode chunk size | 12 |
| `--stream-chunk <N>` | Stream chunk size | 4 |
| `--rep-penalty <F>` | Repetition penalty | 1.0 |
| `--pres-penalty <F>` | Presence penalty | 0.0 |
| `--freq-penalty <F>` | Frequency penalty | 0.0 |
| `--bench` | Benchmark mode | off |
| `--bench-pp <N>` | Benchmark prompt length | 512 |
| `--bench-tg <N>` | Benchmark generation length | 128 |
| `--bench-iters <N>` | Benchmark iterations | 5 |
| `--bench-warmup <N>` | Benchmark warmup iterations | 1 |
| `--bench-sample` / `--bench-greedy` | Benchmark decode mode | greedy |
| `--log-level <level>` | Minimum log level (verbose/debug/info/warning/error) | info |
| `--messages-json <path>` | JSON prompt file (`-` = stdin) | (none) |
| `--chat-output-json` | With `--messages-json`, print structured assistant JSON instead of raw text | off |
| `--chat-stream-jsonl` | With `--messages-json`, print structured stream events as JSONL | off |
| `--transcribe <path>` | Transcription mode for audio WAV files | (none) |
| `--synthesize <text>` | TTS text-to-speech synthesis | (none) |
| `--output-wav <path>` | TTS output WAV file path | `output.wav` |
| `--ref, --speaker <name_or_path>` | TTS voice: reference audio path or named voice (e.g., vivian, ryan) | (none) |
| `--instruct <text>` | VoiceDesign style description for TTS (e.g., "deep warm voice") | (none) |
| `--language <lang>` | TTS language: chinese, english, japanese, korean, auto | auto |
| `--ref-cache-dir <dir>` | Speaker embedding cache directory | `AILA_REF_CACHE_DIR` env |
| `--stream-tts` | Stream raw 24kHz mono f32 PCM to stdout | off |
| `--stream-batch <N>` | Frames per streaming TTS chunk | 4 |
| `--align-text <text>` | Forced alignment: transcript text to align | (none) |
| `--align-audio <path>` | Forced alignment: audio file path | (none) |
| `--align-lang <lang>` | Forced alignment: language (default: Chinese) | Chinese |
| `--forced-lang <lang>` | Force ASR language (e.g. Chinese, English) | (none) |
| `--asr-system <prompt>` | ASR system prompt text bias | (none) |
| `--asr-segment <sec>` | ASR segment split duration in seconds | 0.0 (disabled) |
| `--asr-past` / `--no-asr-past` | Toggle past-text conditioning for ASR segments | no-asr-past |
| `-h, --help` | Show help | — |
| `-v, --version` | Show version | — |

### 🎮 Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit`, `/exit` | Exit |
| `/transcribe <path>` | Transcribe audio file (ASR) |
| `/align text="..." audio=<path> [language="..."]` | Forced alignment (word-level timestamps) |
| `/tts <text> [--ref <path>]` | Synthesize speech (TTS) with optional voice cloning |
| `/synthesize <text> [--ref <path>]` | Alias for `/tts` |
| `/voice <name>` | Set TTS voice (vivian, ryan, etc.) |
| `/clear` | Clear conversation history |
| `/context` | Show context usage |
| `/greedy` | Switch to greedy decoding |
| `/sample` | Switch to sampling |
| `/seed <N>` | Set sampling seed |
| `/stream_on` / `/stream_off` | Toggle streaming |
| `/decode_chunk <N>` | Set decode chunk size |
| `/stream_chunk <N>` | Set stream chunk size |
| `/thinking_budget <N|off>` | Set thinking budget |
| `/log_level <level>` | Set log level (verbose/debug/info/warning/error) |
| `/config` | Show current configuration |

### 💭 Chat Formatting and Tool Calls

`--messages-json` accepts OpenAI-style chat requests with `messages`, `tools`,
`tool_choice`, `chat_template_kwargs`, and generation parameters. Qwen3.5 Hybrid
models use Aila's built-in fixed Qwen3.5 Jinja template unless a request or
environment override is provided.

By default `--messages-json` prints raw assistant text. Add `--chat-output-json`
to print structured assistant JSON with `content`, `reasoning_content`,
`tool_calls`, `raw_text`, `finish_reason`, `warnings`, and `metadata`.
Use `--chat-stream-jsonl` instead to print structured stream events as
newline-delimited JSON.

Aila formats and parses tool calls but does not execute tools. Callers should
execute returned `tool_calls` externally and send tool results back as `tool`
messages. `tool_policy` can be `"warn"` or `"strict"`; strict policy marks
violations with `finish_reason: "tool_policy"`. Reasoning can be bounded with
`reasoning_budget` / `--thinking-budget`. C API callers can use
`aila_generate_chat_json_ex` and `aila_generate_chat_json_stream_ex` with
`AilaGenConfigV2` for ABI-safe chat options.

### 🎤 TTS Voice Cloning

Aila supports **zero-shot voice cloning** for Qwen3-TTS models. The speaker embedding is extracted via a native C++ ECAPA-TDNN encoder (no Python dependency).

**Reference audio requirements:**

| Requirement | Value |
|-------------|-------|
| **Format** | WAV, MP3, or FLAC |
| **Sample rate** | Any (auto-resampled to 24kHz) |
| **Channels** | Mono recommended (multi-channel is averaged to mono) |
| **Duration** | ≥ 3 seconds of clear speech recommended |
| **Content** | Clean speech from the target speaker with minimal background noise |
| **Encoding** | 16-bit PCM or 32-bit float |

```powershell
# Clone a voice from reference audio
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base `
    --synthesize "你好世界" `
    --ref ./reference_speaker.wav `
    --output-wav cloned_output.wav

# CustomVoice — use a named speaker preset
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice `
    --speaker vivian `
    --synthesize "Hello world!" `
    --output-wav vivian_output.wav

# VoiceDesign — describe the voice style in natural language
Aila.exe -m ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign `
    --instruct "A deep, warm voice with a slow pace" `
    --synthesize "Hello world!" `
    --output-wav styled_output.wav

# CustomVoice + language + instruct override
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice `
    --speaker ryan `
    --language english `
    --instruct "whispering softly" `
    --synthesize "Hello" `
    --output-wav ryan_english.wav

# Streaming TTS — real-time PCM audio output
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base `
    --stream-tts --synthesize "Hello!" 2>/dev/null | pcm_play

# Streaming TTS with custom batch size
Aila.exe -m ./models/Qwen3-TTS-12Hz-0.6B-Base `
    --stream-tts --stream-batch 8 --synthesize "Hello!" 2>/dev/null | pcm_play
```

The `--rep-penalty` flag controls repetition penalty (auto-set to 1.1 for TTS). Increase it (e.g. `--rep-penalty 1.3`) if the output has repetitive artifacts, or decrease it (e.g. `--rep-penalty 1.0`) for more variation.

Speaker embeddings are **automatically cached**: in-memory for the session lifetime, and persisted to disk (as `<audio_path>.ref.bin`) to avoid re-extraction on subsequent runs. Use `--ref-cache-dir <dir>` or `AILA_REF_CACHE_DIR` to store cache files in a centralized directory.

### 📄 Messages JSON Format

Supports standard OpenAI compatible JSON objects with `"messages"` as the top-level key, which also allows passing sampling/generation parameters directly (e.g. `temperature`, `max_tokens`/`max_new_tokens`, `seed`/`do_sample`):

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": [{"type": "text", "text": "Introduce yourself"}]}
  ],
  "temperature": 0.7,
  "max_tokens": 128
}
```

It also backward-compatibly supports a raw JSON array as the top-level key: `[{"role": "user", "content": "..."}]`.

Supports `text`, `image`, `audio`, and `video` content types. Image parts accept `image`, `image_url` (with base64 Data URI or file path), or `{"image_url":{"url":"..."}}`. Audio parts accept `audio`, `audio_url` (with base64 Data URI or file path), or `{"input_audio": {"data": "base64", "format": "wav"}}`.

### 🔌 C API

See **[docs/C_API.md](docs/C_API.md)** for the full C API reference (Python ctypes, C# P/Invoke, Rust FFI, etc.).

### 🌐 Environment Variables

See **[docs/Environment_Variables.md](docs/Environment_Variables.md)** for the complete list of configuration variables.

## 📦 Model Export

Use `export_bnb_nf4.py` to quantize a Hugging Face model to BNB NF4 format:

```powershell
# Text-only model
python export_bnb_nf4.py \
    --source Qwen/Qwen3.5-0.8B \
    --output ./models/Qwen3.5-0.8B-BNB-NF4

# Model with vision
python export_bnb_nf4.py \
    --source Qwen/Qwen3.5-4B \
    --output ./models/Qwen3.5-4B-BNB-NF4-with-vision \
    --vision

# ASR model
python export_bnb_nf4.py \
    --source Qwen/Qwen3-ASR-1.7B \
    --output ./models/Qwen3-ASR-1.7B-BNB-NF4

# ForceAligner model (auto-detected, classify_head kept dense)
python export_bnb_nf4.py \
    --source Qwen/Qwen3-ForcedAligner-0.6B \
    --output ./models/Qwen3-ForcedAligner-0.6B-BNB-NF4

# From local directory, overwriting existing export
python export_bnb_nf4.py \
    --source ./Qwen3-0.6B \
    --output ./models/Qwen3-0.6B-BNB-NF4 \
    --overwrite
```

Requirements: `torch`, `transformers`, `bitsandbytes` with Intel XPU backend.

## 🛠️ Build from Source

```powershell
# Requires: Intel oneAPI Base Toolkit, CMake 3.24+, Ninja
.\build.ps1

# Clean build
.\build.ps1 -Clean

# Debug build
.\build.ps1 -Config Debug
```

Outputs:
| File | Description |
|------|-------------|
| `build/Aila.exe` | CLI executable |
| `build/AilaShared.dll` | Shared library (C API) |
| `build/AilaLib.lib` | Static library |

## 📁 Project Structure

```
Aila/
├── include/
│   ├── aila_api.h              # Public C API header
│   └── engine/Engine.hpp       # InferenceEngine class
├── src/
│   ├── main.cpp                 # CLI entry point
│   ├── api/aila_api.cpp         # C API implementation
│   ├── cli/                     # CLI argument parsing & interactive loop
│   ├── core/                    # SYCL context & tensor management
│   ├── memory/                  # KV cache
│   ├── models/                  # Model backends (Qwen3, Qwen3.5, BNB4)
│   ├── ops/                     # SYCL kernels (attention, RMSNorm, Bnb4BitLinear, etc.)
│   ├── profile/                 # Logging, profiling & device info
│   ├── audio/                   # Audio preprocessing (ASR) + speaker encoder (TTS voice cloning)
│   ├── utils/                   # Tokenizer, SafeTensors, memory-mapped I/O
│   └── vision/                  # Vision encoder (Qwen3.5)
├── docs/
│   ├── C_API.md                 # C API documentation
│   └── Environment_Variables.md # Environment variable reference
├── third_party/simdjson/        # JSON parsing
├── build.ps1                    # Build script
├── bench.ps1                    # Benchmark script
├── smoke.ps1                    # Smoke test script
└── CMakeLists.txt
```

---

## 🙏 Credits

- **[oneDNN](https://github.com/oneapi-src/oneDNN)** — Intel's deep neural network library providing the matmul primitives used for bf16 inference
- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** — NF4 quantization format and dequantization reference
- **[simdjson](https://github.com/simdjson/simdjson)** — Fast JSON parser used for model config and tokenizer metadata
- **[dr_libs](https://github.com/mackron/dr_libs)** — Single-header audio decoding libraries (`dr_wav`, `dr_mp3`, `dr_flac`) used for ASR audio preprocessing
- **[llama.cpp jinja module](https://github.com/ggml-org/llama.cpp/tree/master/common/jinja)** — jinja chat template parser module, used for parsing and building chat template

## 📄 License

See [LICENSE](LICENSE) for details.
