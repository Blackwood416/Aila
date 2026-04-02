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
