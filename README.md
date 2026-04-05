# Aila

A high-performance LLM inference engine built with **SYCL + oneDNN**, designed to run on Intel GPUs. Aila can be used as a standalone CLI tool or integrated into other applications as a library through its stable C API.

## Features

- **GPU-Accelerated Inference** — Leveraging Intel oneAPI SYCL and oneDNN for optimized matrix operations on Intel Arc GPUs
- **Library Integration** — Compiles as both static (`.lib`) and shared (`.dll`/`.so`) libraries with a pure C API for cross-language FFI (Python, C#, Go, Rust, etc.)
- **Streaming Generation** — Token-by-token streaming with configurable chunk sizes for latency/throughput trade-off
- **Unified Logging** — Centralized logging with custom callback support, allowing library consumers to redirect or suppress output
- **Flexible CLI** — Argparse-style command-line interface with interactive commands and environment variable fallbacks

## Model Support

Currently supports **Qwen3-0.6B** (BF16).

## Verified Devices

| Device | VRAM | Status |
|--------|------|--------|
| Intel Arc A770 | 16 GB | ✅ Verified |

## Build

### Prerequisites

- [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html) (SYCL compiler + oneDNN)
- CMake ≥ 3.24
- Ninja (recommended) or Visual Studio

### Build Commands

Use `build.ps1` to build on Windows — it automatically initializes all required oneAPI environment variables.

```powershell
# Quick build (Release mode)
./build.ps1

# Or manually:
# 1. Initialize oneAPI environment
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && set' | ...
# 2. Configure and build
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Build outputs (in `build/`):

| File | Description |
|------|-------------|
| `Aila.exe` | CLI executable |
| `AilaLib.lib` | Static library |
| `AilaShared.dll` | Shared library (+ dnnl.dll copied alongside) |
| `AilaShared.lib` | Import library for DLL linking |

> **Note:** `Release` build type is critical for performance; `Debug` builds are significantly slower.

## Usage

### CLI

```powershell
# Run with model directory
./run.ps1
# Or directly:
Aila.exe -m ./Qwen3-0.6B

# Full options
Aila.exe [options]

Options:
  -m, --model <path>       Model directory (required, or set AILA_MODEL_DIR)
  -s, --max-seq <N>        Maximum sequence length (default: 4096)
  -t, --temperature <F>    Sampling temperature (default: 0.7)
  -k, --top-k <N>          Top-K sampling (default: 15)
  --greedy                 Use greedy decoding (default)
  --sample                 Use sampling
  --stream / --no-stream   Force streaming output on/off
  --max-tokens <N>         Maximum new tokens (default: 1024)
  --decode-chunk <N>       Decode chunk size (default: 12)
  --stream-chunk <N>       Stream chunk size (default: 4)
  -h, --help               Show help
  -v, --version            Show version
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/config` | Show current configuration |
| `/greedy` | Switch to greedy decoding |
| `/sample` | Switch to sampling |
| `/stream_on` | Enable streaming output |
| `/stream_off` | Disable streaming output |
| `/decode_chunk <N>` | Set decode chunk size |
| `/stream_chunk <N>` | Set stream chunk size |
| `/quit`, `/exit` | Exit the program |

### Library Integration (C API)

Aila exposes a stable C ABI through `aila_api.h`, suitable for any language with C FFI support:

```c
#include "aila_api.h"

// Create and initialize engine
AilaEngine* engine = aila_engine_create();
aila_engine_init(engine, "./Qwen3-0.6B", 4096);

// Blocking generation
AilaGenConfig config = aila_default_gen_config();
char* response = aila_generate(engine, "Hello, who are you?", &config);
printf("%s\n", response);
aila_free_string(response);

// Streaming generation
aila_generate_stream(engine, "Tell me a story", &config,
    [](const char* token, void* ud) -> int {
        printf("%s", token);
        return 0;  // return non-zero to abort
    }, NULL);

// Cleanup
aila_engine_destroy(engine);
```

**Python example** (via ctypes):

```python
import ctypes

lib = ctypes.CDLL("./AilaShared.dll")

# ... setup function signatures ...

engine = lib.aila_engine_create()
lib.aila_engine_init(engine, b"./Qwen3-0.6B", 4096)
result = lib.aila_generate(engine, b"Hello!", None)
print(ctypes.string_at(result).decode())
lib.aila_free_string(result)
lib.aila_engine_destroy(engine)
```

## Runtime Tuning

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AILA_MODEL_DIR` | — | Default model directory (fallback for `-m`) |
| `AILA_MAX_SEQ_LEN` | `4096` | Maximum context window length |
| `AILA_STREAM_OUTPUT` | auto | Force streaming (`1`) or non-streaming (`0`) |
| `AILA_DECODE_CHUNK_SIZE` | `12` | Non-streaming greedy decode chunk size |
| `AILA_STREAM_CHUNK_SIZE` | `4` | Streaming greedy decode chunk size |

### Streaming Output

- Interactive mode enables streaming by default.
- Piped/script mode disables streaming by default.
- Override with `AILA_STREAM_OUTPUT=0|1` or `--stream`/`--no-stream`.

### Token Chunk Tuning

- Smaller `stream_chunk_size` → lower latency per visible token, but usually lower tok/s.
- Larger `stream_chunk_size` → higher throughput, but less real-time display.
- On Intel Arc A770, `stream_chunk_size=4` is a good default; `8` often gives higher peak throughput.

### Context Window / Memory

- KV cache scales linearly with `max_seq_len`.
- Activation and prefill score buffers are allocated lazily and grow on demand, so startup VRAM is much lower than preallocating full buffers.

## Project Structure

```
Aila/
├── include/
│   ├── aila_api.h           # Public C API header
│   └── engine/
│       ├── Engine.hpp        # InferenceEngine class
│       └── Types.hpp         # GenerationConfig, etc.
├── src/
│   ├── main.cpp              # CLI entry point
│   ├── api/aila_api.cpp      # C API implementation
│   ├── cli/                  # CLI argument parsing & interactive loop
│   ├── core/                 # SYCL context & tensor management
│   ├── memory/               # KV cache
│   ├── models/               # Qwen3 model implementation
│   ├── ops/                  # SYCL kernels (Linear, Attention, RMSNorm, etc.)
│   ├── profile/              # Logging, profiling & device info
│   └── utils/                # Tokenizer, SafeTensors, memory-mapped I/O
├── third_party/simdjson/     # JSON parsing
├── build.ps1                 # Build script (Windows)
├── run.ps1                   # Run script (Windows)
└── CMakeLists.txt
```

## License

See [LICENSE](LICENSE) for details.
