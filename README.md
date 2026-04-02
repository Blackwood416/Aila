# Aila, a pure LLM inference engine built with SYCL + oneDNN
## Model Support
Currently it just support Qwen3-0.6B BF16
## Verified Devices
GPU:
>Intel A770 16G
## Build
Use `build.ps1` to build on windows, it automatically initialize all required environment variables for OneAPI(cmake needs those headers to build the project).
`build.ps1` will force `CMAKE_BUILD_TYPE=Release` before building. This is important for performance; `Debug` builds are much slower.
## Run
Use `run.ps1` to run on windows, it automatically initialize all required environment variables for OneAPI(the exectuable needs those OneAPI dlls to run). And the run log is located at `E:\RiderProjects\Aila\run_log.txt`

## Runtime Tuning
### Streaming Output
- Interactive mode enables streaming by default.
- Piped/script mode (for benchmark) disables streaming by default.

You can force behavior with:
- `AILA_STREAM_OUTPUT=1` to force streaming
- `AILA_STREAM_OUTPUT=0` to force non-streaming

Interactive commands:
- `/stream_on`
- `/stream_off`

### Token Chunk Tuning
Two chunk controls are available:
- `decode_chunk_size`: non-streaming greedy decode chunk size (default `12`)
- `stream_chunk_size`: streaming greedy decode chunk size (default `4`)

Environment variables:
- `AILA_DECODE_CHUNK_SIZE=<N>`
- `AILA_STREAM_CHUNK_SIZE=<N>`

Interactive commands:
- `/decode_chunk <N>`
- `/stream_chunk <N>`

Notes:
- Smaller `stream_chunk_size` gives lower latency per visible token, but usually lower tok/s.
- Larger `stream_chunk_size` improves throughput, but token display becomes less real-time.
- On Intel Arc A770, `stream_chunk_size=4` is a good default balance; `8` often gives higher peak throughput.
