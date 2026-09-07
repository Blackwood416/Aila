# Windows HTTP/SSE dependency probe

Date: 2026-09-07. Scope: local dependency inspection for a future C++ server. No model was loaded, no HTTP service was started, and no dependencies were downloaded.

## Result

The machine has a usable C++ toolchain and the Visual Studio distribution of vcpkg, but no discoverable Drogon/Trantor installation or existing HTTP server library in this repository. A Drogon loopback SSE probe was therefore **not built or run**. This inspection establishes the missing dependency prerequisite; it does not establish whether a selected Drogon release will compile or meet the streaming requirements on this machine.

The existing Windows proxy provides a suitable boundary for an ordinary C++ host: the inspected `build/AilaShared.dll` imports only `KERNEL32.dll` and `USER32.dll`. GPU runtime dependencies remain in the worker process.

## Observed environment

| Item | Local evidence |
| --- | --- |
| CMake | `4.1.2-msvc8`, available on `PATH` |
| Ninja | Available on `PATH` under the Python 3.13 Scripts directory |
| MSVC | `cl.exe` is present under VS 18 Community, MSVC toolset `14.50.35717` |
| Project compiler | oneAPI `2026.1`, `icx-cl.exe`; the extracted worker adapter and WorkerMain passed C++17 syntax checks using the project environment importer |
| vcpkg | Not on `PATH`; found at `C:/Program Files/Microsoft Visual Studio/18/Community/VC/vcpkg/vcpkg.exe` |
| vcpkg version | `2025-12-16-44bb3ce006467fc13ba37ca099f64077b8bbf84d` |
| vcpkg distribution | `deployment: VisualStudio`, `readonly: true`, `usegitregistry: true`; embedded registry SHA `544a4c5c297e60e4ac4a5a1810df66748d908869` |
| Package state | No `installed` directory in the discovered vcpkg root; no repository `vcpkg.json` or `vcpkg_installed` directory |
| `vcpkg list` | Exits with “Could not locate a manifest” and “This vcpkg distribution does not have a classic mode instance” |
| Conan | Not on `PATH`; no `.conan` or `.conan2` directory in the inspected user profile |
| Package discovery | No configured `VCPKG_ROOT`, `VCPKG_INSTALLED_DIR`, `CMAKE_PREFIX_PATH`, `BOOST_ROOT`, or `OPENSSL_ROOT_DIR` in the task environment |
| Drogon/Trantor | No matching header, library, CMake package, or tool found in the inspected project roots; no corresponding CMake user package registry entry |
| Existing network library | No Drogon, Trantor, Asio, cpp-httplib, Boost HTTP, libuv, or equivalent server integration found in repository source/CMake searches |

Searches covered the repository, `E:/RiderProjects`, `E:/VisualStudioProjects`, conventional vcpkg locations, the discovered VS vcpkg root, and relevant user package/configuration locations. They were not an exhaustive scan of every file on every disk.

The inspected proxy artifact was `build/AilaShared.dll`, 938,496 bytes, last modified 2026-09-02 15:30:08 local time. `dumpbin /dependents` reported only the two Windows DLLs above. `dumpbin /exports` confirmed `aila_engine_create`, `aila_engine_init`, `aila_generate_chat_json_stream_ex`, and `aila_version`. This was an existing artifact, not a newly linked result of the current refactor.

Useful read-only reproduction commands:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\vcpkg\vcpkg.exe' --disable-metrics version
& 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\vcpkg\vcpkg.exe' --disable-metrics list
& 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx86\x64\dumpbin.exe' /dependents build/AilaShared.dll
```

## Dependency cost and next build step

The available vcpkg is a manifest distribution. A probe needs its own manifest and pinned registry baseline/version selection; the installed vcpkg executable is not itself an installed Drogon package. Keep that manifest, installed tree, and standalone CMake project under `build-refactor-http-probe` until the framework passes the probe.

Plan for the Drogon/Trantor networking framework and its JSON dependency, including the possible duplication of JSON libraries alongside Aila's simdjson. The exact selected port must be inspected before installation: TLS, compression, DNS, command-line tools, and database features can add dependencies such as OpenSSL, zlib, Brotli, c-ares, or database clients. For the first loopback HTTP probe, use the smallest feature set supported by that port and omit database, ORM, tool, and TLS features where optional. Do not assume that disabling default features removes a port's unconditional dependencies.

No package build, download size, installed size, or build time was measured. A numerical estimate would currently be speculative. First inspect the resolved installation plan, then build only the selected framework and probe. If dependency resolution or its Windows build fails, preserve that concrete failure before evaluating another established HTTP library.

Use a standalone ordinary C++17 CMake project with `find_package(Drogon CONFIG REQUIRED)` and `Drogon::Drogon`, subject to confirmation against the chosen release. The repository's top-level CMake currently requires IntelLLVM, so it cannot directly be configured with MSVC merely to build a server. The production build must either keep the ordinary C++ server as a separate subproject/imported-proxy consumer or adjust that compiler requirement as a deliberate build-system change. No such change is made by this probe.

## Minimal loopback SSE acceptance probe

Once Drogon is installed, the probe should use an entirely synthetic producer and bind only `127.0.0.1` on a test port. It should exercise the chosen release's asynchronous streaming API rather than accumulating the whole response before sending it.

| Scenario | Acceptance condition |
| --- | --- |
| Incremental delivery | UTF-8 `text/event-stream` records arrive before the producer finishes, each with a terminating blank line; response headers are sent before the first record |
| Two concurrent clients | Two synthetic request IDs produce independently ordered streams; one stream's completion does not terminate the other |
| Disconnect | A client closes after its first event; disconnect reaches that request's cancellation state and its timer/producer is released |
| Slow reader | Pausing one client's reads keeps its queued byte count bounded while a second client and a health route remain responsive |
| Completion race | Disconnect and normal completion racing produce one terminal state; no write uses a destroyed connection or stream object |
| Shutdown | Stop accepting requests, cancel active synthetic producers, close streams, and join all server threads within a bounded timeout |

The client harness should read actual socket bytes with a deadline, explicitly close the first client socket for the disconnect case, and record producer cancellation as well as client observations. A buffering command-line client alone is insufficient to demonstrate incremental flushing or cleanup.

Keep a process handle and PID for any launched probe. Run it with a hidden window, terminate it through the framework's shutdown path, and use a bounded cleanup in a `finally` block for failures. Record the selected port, test results, and process exit status. The present inspection launched no persistent processes and allocated no listening port.

## Ordinary C++ server to proxy contract

1. Link the host to `AilaShared` through `include/aila_api.h`; do not link the host to `AilaLib` or `AilaSyclOptions`. The current Windows proxy target explicitly uses `-fno-sycl` and a static MSVC runtime. A separate server can use its framework's CRT choice because the C ABI owns its allocations; all returned strings/samples must be released through the matching `aila_free_*` functions, never a host `free` or `delete`.
2. Deploy a matching proxy and worker build. `AILA_RUNTIME_DLL_DIR=aila_runtime` selects the separate runtime directory relative to `AilaShared.dll`; the unset fallback is the proxy directory itself. Preserve the documented split layout and worker-only GPU runtime search path. See `docs/Environment_Variables.md` and `src/runtime/RuntimeDirectory.cpp`.
3. Current text generation calls block until generation completes. Run them outside the HTTP event loop. Callback strings and structured-event pointers are valid only during the callback; copy the required values into an owning, bounded response queue before returning.
4. Current `AilaEngine` instances are not a multi-session service API. Proxy operations are serialized, streaming operations reject competing engine calls, and each initialized engine starts a separate worker. Multiple HTTP threads calling one handle do not provide shared-model concurrent generation. Integrate the future Session/Request scheduler behind a service interface before advertising that capability.
5. The current text cancellation mechanism is the callback's nonzero return; there is no exported general text Request cancellation handle. A callback-level disconnect flag cannot guarantee prompt cancellation during a long prefill with no callbacks. The new Request interface needs cancellation checks at execution boundaries and request-scoped errors, rather than a mutable shared `last_error` slot.
6. Preserve the existing caller-thread semantics for legacy text callbacks and the existing ASR/TTS handle lifetime behavior. The new asynchronous server service should specify callback/executor ownership separately. It must not run HTTP socket writes or user callbacks while holding the shared model, transport, or scheduler lock.

Relevant existing CPU suites for later integration are `AilaIpcProtocolTests`, `AilaWorkerDispatcherTests`, `AilaRuntimeIsolationTests`, and `AilaProxyAbiTests`. Their existing coverage includes stream ordering, callback abort, stale handles, worker failure, and bounded shutdown. They do not substitute for the socket-level SSE/disconnect probe above.
