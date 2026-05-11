# Aila

<p align="center">
  <b>An inference engine leveraging Arc graphics.</b><br>
  <a href="README_zh.md">中文</a>
</p>

---

> [!NOTE]
> This project is under active development and does not yet fully support all listed models. The current focus is on the **Qwen3.5** family; Qwen3 model performance may not be optimal.

A high-performance LLM inference engine for **Intel Arc GPUs**, built with **SYCL + oneDNN**. Features hand-optimized kernels for bitsandbytes 4-bit (NF4) quantized models, fused dequant+matmul, and GPU-accelerated DeltaNet recurrence for Qwen3.5 hybrid architectures.

## ✨ Features

- **⚡ Bitsandbytes 4-bit (NF4) inference** — run quantized models directly on Intel Arc with fused dequant+matmul kernels, hand-written GEMV decode, and fused gate+up+SiLU projection
- **🔢 Bfloat16 inference** — dense (unquantized) models via oneDNN matmul primitives
- **🏗️ Qwen3.5 Hybrid architecture** — full GPU acceleration for the dual attention (GQA + DeltaNet linear attention) architecture
- **📐 Qwen3 Dense architecture** — standard Transformer with GQA, QK-norm, and SwiGLU FFN
- **👁️ Vision (Qwen3.5)** — image understanding with CPU preprocessing and GPU vision transformer
- **🔄 Streaming output** — token-level streaming callback with abort support
- **💬 Interactive CLI** — multi-turn conversation with runtime commands (`/clear`, `/greedy`, `/sample`, etc.)
- **📊 Benchmark mode** — measure prefill and decode throughput separately
- **🔌 C API** — stable C FFI interface (Python, C#, Rust, Go, Java) — see [docs/C_API.md](docs/C_API.md)
- **💭 Chat template** — ChatML format with `<think>` block generation and `/no_think` suppression

## 📦 Supported Models

| Model | Architecture | Quantization | Vision |
|-------|-------------|-------------|--------|
| [Qwen3.5-0.8B](https://huggingface.co/Blackwood416/Qwen3.5-0.8B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ |
| [Qwen3.5-4B](https://huggingface.co/Blackwood416/Qwen3.5-4B-BNB-NF4-with-vision) | Hybrid (GQA + DeltaNet) | BNB NF4, dense | ✅ |
| [Qwen3-0.6B](https://huggingface.co/Blackwood416/Qwen3-0.6B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ |
| [Qwen3-4B](https://huggingface.co/Blackwood416/Qwen3-4B-BNB-NF4) | Dense (GQA) | BNB NF4, dense | ❌ |

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
| **Aila 0.1.2** | SYCL + oneDNN | Qwen3.5-4B BNB NF4 | **1649 tok/s** | 58 tok/s |
| llama.cpp b8996 | SYCL | Qwen3.5-4B Q4_K_XL | 1290 tok/s | 28 tok/s |
| llama.cpp b8996 | Vulkan | Qwen3.5-4B Q4_K_XL | 700 tok/s | **60 tok/s** |

Aila delivers the highest prefill throughput and competitive decode performance against Vulkan while using a more accurate NF4 4-bit quantization that retains vision capabilities.

## 🚀 Usage

### ⌨️ CLI Quick Start

```powershell
# Interactive conversation
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline

# Single prompt from JSON file
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --messages-json prompt.json

# Single prompt from stdin
echo '{"messages":[{"role":"user","content":"hello"}]}' | Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --messages-json -

# Benchmark (greedy)
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --bench --bench-pp 512 --bench-tg 128

# Benchmark (sampling)
Aila.exe -m ./models/qwen3.5-0.8B-bnb-nf4-offline --bench --sample
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
| `-h, --help` | Show help | — |
| `-v, --version` | Show version | — |

### 🎮 Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/quit`, `/exit` | Exit |
| `/clear` | Clear conversation history |
| `/context` | Show context usage |
| `/greedy` | Switch to greedy decoding |
| `/sample` | Switch to sampling |
| `/seed <N>` | Set sampling seed |
| `/stream_on` / `/stream_off` | Toggle streaming |
| `/decode_chunk <N>` | Set decode chunk size |
| `/stream_chunk <N>` | Set stream chunk size |
| `/log_level <level>` | Set log level (verbose/debug/info/warning/error) |
| `/config` | Show current configuration |

### 🤫 `/no_think` and `/think` Suffixes

Append `/no_think` to suppress the model's `<think>` block, or `/think` to force thinking (useful on Qwen3.5-0.8B which defaults to non-thinking):

```
User: What is 1+1? /no_think
Aila: 1+1 equals 2.

User: Explain quantum computing. /think
Aila: <think>
Let me break this down step by step...
</think>
Quantum computing is...
```

Both work in interactive mode and `--messages-json`.

### 📄 Messages JSON Format

```json
[
  {"role": "system", "content": "You are a concise assistant."},
  {"role": "user",   "content": [{"type": "text", "text": "介绍你自己"}]}
]
```

Supports `text`, `image`, and `video` content types. Image parts accept `image`, `image_url`, or `{"image_url":{"url":"..."}}`.

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
    --output ./models/qwen3.5-0.8B-bnb-nf4-offline

# Model with vision
python export_bnb_nf4.py \
    --source Qwen/Qwen3.5-4B \
    --output ./models/qwen3.5-4B-bnb-nf4-vision-offline \
    --vision

# From local directory, overwriting existing export
python export_bnb_nf4.py \
    --source ./Qwen3-0.6B \
    --output ./models/qwen3-0.6B-bnb-nf4-offline \
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

## 📄 License

See [LICENSE](LICENSE) for details.
