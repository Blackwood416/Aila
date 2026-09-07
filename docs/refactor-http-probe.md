# Windows HTTP/SSE dependency probe

Date: 2026-09-07. Scope: an ordinary Windows C++17 build and actual loopback HTTP/SSE acceptance probe for a future C++ server. All dependency sources, binaries, package caches, and run artifacts are isolated in `build-refactor-http-probe`. No model, Aila inference API, or GPU was used.

## Result

**The six socket acceptance scenarios passed with Drogon 1.9.11 plus the local Windows listener-callback patch, Trantor 1.5.24, JsonCpp 1.9.6, and zlib 1.3.1.** This is a synthetic dependency/protocol probe, not an integrated inference server or a claim of general production backpressure.

Unmodified Drogon 1.9.11 built successfully but did **not** meet the complete Windows acceptance criteria. Its non-Linux listener branch did not forward the after-accept socket-option callback or the connection callback. The original run reported 5/6 scenarios; the socket buffer readback remained zero because the callback never ran. That failed report is preserved alongside the passing run.

The local patch only forwards the same three callbacks already forwarded in the Linux branch: before-listen socket options, after-accept socket options, and connection state. It changes 14 lines in `lib/src/ListenerManager.cc`. It has not been sent upstream.

The existing Windows proxy provides a suitable boundary for an ordinary C++ host: the inspected `build/AilaShared.dll` imports only `KERNEL32.dll` and `USER32.dll`. GPU runtime dependencies remain in the worker process.

## Initial environment inspection

These observations preceded the isolated dependency installation below.

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

## Fixed dependencies and reproducible build

The actual vcpkg resolve plan was inspected with baseline `544a4c5c297e60e4ac4a5a1810df66748d908869`, triplet `x64-windows-static-md`, and all Drogon features disabled. That port still requires Brotli 1.2.0, c-ares 1.34.6, OpenSSL 3.6.0#3, JsonCpp 1.9.6, and zlib 1.3.1. TLS, DNS, and compression are unconditional port dependencies, so disabling default features is insufficient. Database, ORM, YAML, and `drogon_ctl` features were not selected.

To build the minimal plaintext probe, vcpkg installs only JsonCpp and zlib at that same baseline. Drogon and Trantor are built from their exact upstream release archives. The SHA512 values are copied from the pinned vcpkg ports and checked before extraction. The build uses MSVC, C++17, static framework libraries, and the dynamic MSVC CRT.

| Dependency | Fixed source | Selected configuration |
| --- | --- | --- |
| Drogon | `an-tao/drogon`, `v1.9.11`, archive SHA512 in `build.ps1` | No ORM/databases, tools, examples, YAML, Brotli, coroutines, or TLS; local Windows callback patch |
| Trantor | `an-tao/trantor`, `v1.5.24`, archive SHA512 in `build.ps1` | No TLS provider, c-ares, tests, examples, or shared library |
| JsonCpp | vcpkg baseline above, 1.9.6 | Static library, dynamic CRT |
| zlib | vcpkg baseline above, 1.3.1 | Static library, dynamic CRT |

Trantor configuration reports `Trantor using SSL library: None`. Both OpenSSL and Botan discovery are explicitly disabled. `BUILD_C-ARES`, `BUILD_ORM`, `BUILD_CTL`, and `BUILD_BROTLI` are off in the recorded CMake caches. The executable's PE imports contain Windows system and MSVC/UCRT libraries, with no OpenSSL or GPU runtime DLLs.

Reproduce from the repository root:

```powershell
pwsh -NoProfile -File tools/http-probe/build.ps1 -Parallelism 4
pwsh -NoProfile -File tools/http-probe/run.ps1
```

`build.ps1` locates ordinary MSVC with `vswhere`, records the full and minimal vcpkg plans, verifies source archives and the patch, installs to the isolated prefix, and builds the standalone probe. `-DependenciesOnly` stops after the framework install. After a successful full build, `-ProbeOnly` rebuilds just the probe. Dependency source trees and caches stay under the chosen `-BuildRoot`; no top-level CMake changes are required.

The normalized-LF SHA256 of `tools/http-probe/drogon-windows-listener-callbacks.patch` is:

```text
f267e88c29eaa99bb8b138fae3f8224d4843a0ef9dc393406dbd7c43498cb179
```

The script verifies that exact patch, uses a separate Git root inside the extracted dependency to prevent a parent-repository path prefix from silently skipping it, and checks that the applied patch can be reversed. `logs/source-lock.json`, `logs/applied-patch.json`, and `logs/verify-windows-listener-patch.txt` preserve the source and patch evidence. `logs/probe-dependents.txt` and `logs/artifact-inventory.json` record PE imports and artifact hashes for the measured run. Its executable was 1,301,504 bytes; the static library sizes are recorded separately. A clean full-build wall-clock time was not measured separately from investigation and incremental rebuilds.

The standalone `tools/http-probe/CMakeLists.txt` successfully uses `find_package(Drogon CONFIG REQUIRED)` and `Drogon::Drogon`. The repository's top-level CMake requires IntelLLVM, so it cannot directly be configured with MSVC merely to build a server. The production build must either keep the ordinary C++ server as a separate subproject/imported-proxy consumer or adjust that compiler requirement deliberately. This probe does neither. The repository's broad `*.txt` ignore rule also covers the new standalone `CMakeLists.txt`; include it explicitly when committing the probe.

## Actual loopback acceptance results

The server uses `HttpResponse::newAsyncStreamResponse` with synthetic timers, binds only `127.0.0.1`, and runs two I/O loops. Python's standard `http.client` handles HTTP/chunk framing; the harness reads the decoded body one byte at a time and assembles SSE events with strict UTF-8 decoding. It does not substitute an in-memory fake, a fully buffered HTTP client, or a custom HTTP parser.

| Scenario | Observed passing result |
| --- | --- |
| Incremental delivery | Headers at 0.76 ms, first UTF-8 event at 103 ms, final event at 488 ms; all eight ordered token events contain `汉字🙂` and terminating blank lines |
| Two concurrent clients | Seven-token and eleven-token streams overlap, retain their own IDs/order, and each deliver exactly one `done` event |
| Disconnect | Closing the victim's socket after one event reaches its actual connection callback and releases its timer/producer; cancellation observed within 23.5 ms. A second, simultaneously active stream completes all six tokens |
| Slow reader | Client receive buffer 4,096 B, server send buffer readback 16,384 B; accounted transport pending-byte upper bound peaks at 52,220 B under the 65,536 B gate. Producer pauses 41 timer ticks, resumes after reading, and cancels on disconnect. Eight health requests finish within 20.1 ms; an independent five-token stream completes |
| Completion race | Eight disconnect/completion races exercise both outcomes; each has one state transition to a terminal status and no more than one `done` event |
| Shutdown | Two active streams, including a slow reader, are cancelled. All 17 request records have released timers and destroyed producers; active count is zero. Final server report appears 138 ms after shutdown request, and the process exits with code 0 within the runner's bounded wait, without forced cleanup |

These timings describe one local acceptance run, not load-test latency or inference throughput. The passing run recorded 53 after-accept callbacks and 53 connection-close callbacks. Per-request `disconnect_callback_seen` proves that the victim cancellation reached its own producer, and the survivor test proves it did not terminate the other request.

| Evidence | Isolated path / process |
| --- | --- |
| Unmodified framework failure | `runs/20260907-171304-712bea34/{client,server,process}.json`; PID 22936, port 60960, exit 0, no forced cleanup |
| Patched framework passing run | `runs/20260907-172618-47843b20/{client,server,process}.json`; PID 18416, port 55218, exit 0, no forced cleanup |
| Latest run pointer | `build-refactor-http-probe/latest-run.json` |

Run paths in the table are relative to `build-refactor-http-probe`. Each run also retains `launch.json`, stdout, and stderr. `run.ps1` starts the known server PID with `-WindowStyle Hidden`, waits for its framework shutdown, and in `finally` retries graceful shutdown before a bounded, PID-specific forced cleanup if necessary. Neither recorded process remains running.

## Backpressure boundary and remaining limitations

`ResponseStream::send()` only reports whether the stream is open. Trantor 1.5.24's `sendAsyncDataInLoop()` does not invoke the ordinary-send high-water callback. Merely limiting a producer queue and then unconditionally calling `send()` would still permit the framework's stream buffer to grow.

The probe therefore admits each chunk on its connection's I/O loop only when this upper bound, plus the next chunk and closing marker, fits 64 KiB:

```text
4096 bytes reserved for headers
+ cumulative HTTP chunk bytes submitted by this stream
- connection bytesSent() since the request began
```

The subtraction is clamped at zero. Chunk framing and the five-byte final marker are included. `bytesSent()` counts bytes accepted by the socket, so this bounds pending bytes in Trantor's async stream node for the tested connection; it is not merely the number of tokens waiting in a producer queue. The kernel send buffer is separately configured and read back as 16 KiB. One prepared event is capped at roughly 16 KiB. These are payload/accounting bounds, not a measurement of total process heap, allocator overhead, kernel memory, TCP acknowledgements, or remote application consumption.

**The proof and acceptance are limited to fresh, single-request, plaintext HTTP/1.1 SSE connections with no other writer.** The Python harness opens a new connection for every request. The probe router's `active.count(connection)` check is not a complete prohibition of connection reuse or HTTP pipelining; preceding or additional responses on the same connection can invalidate the byte-accounting assumptions. The strategy must not be promoted directly to generic production backpressure. Production integration needs enforced exclusive connection ownership with reuse/pipelining rejected, or a framework interface reporting its actual pending bytes. TLS, HTTP/2, proxies, and cross-thread output queues need separate verification. A scheduler-to-I/O handoff queue must itself be bounded; this probe submits on the I/O thread and does not test an unbounded `queueInLoop` producer.

During deliberate abrupt-disconnect races, Trantor emitted three `FATAL`-labelled socket diagnostics (`0` / Windows error `10053`) in the passing run's stdout. They did not abort the process, strand a producer, or prevent shutdown; the logs are retained. This remains Windows error-reporting/handling behavior to review before production use, not something the local callback patch fixes.

No authentication, TLS termination, admission policy for arbitrary HTTP traffic, inference scheduler binding, proxy integration, or production load test is included. The existing C API and worker integration still need the separate work below.

## Ordinary C++ server to proxy contract

1. Link the host to `AilaShared` through `include/aila_api.h`; do not link the host to `AilaLib` or `AilaSyclOptions`. The current Windows proxy target explicitly uses `-fno-sycl` and a static MSVC runtime. A separate server can use its framework's CRT choice because the C ABI owns its allocations; all returned strings/samples must be released through the matching `aila_free_*` functions, never a host `free` or `delete`.
2. Deploy a matching proxy and worker build. `AILA_RUNTIME_DLL_DIR=aila_runtime` selects the separate runtime directory relative to `AilaShared.dll`; the unset fallback is the proxy directory itself. Preserve the documented split layout and worker-only GPU runtime search path. See `docs/Environment_Variables.md` and `src/runtime/RuntimeDirectory.cpp`.
3. Current text generation calls block until generation completes. Run them outside the HTTP event loop. Callback strings and structured-event pointers are valid only during the callback; copy the required values into an owning, bounded response queue before returning.
4. Current `AilaEngine` instances are not a multi-session service API. Proxy operations are serialized, streaming operations reject competing engine calls, and each initialized engine starts a separate worker. Multiple HTTP threads calling one handle do not provide shared-model concurrent generation. Integrate the future Session/Request scheduler behind a service interface before advertising that capability.
5. The current text cancellation mechanism is the callback's nonzero return; there is no exported general text Request cancellation handle. A callback-level disconnect flag cannot guarantee prompt cancellation during a long prefill with no callbacks. The new Request interface needs cancellation checks at execution boundaries and request-scoped errors, rather than a mutable shared `last_error` slot.
6. Preserve the existing caller-thread semantics for legacy text callbacks and the existing ASR/TTS handle lifetime behavior. The new asynchronous server service should specify callback/executor ownership separately. It must not run HTTP socket writes or user callbacks while holding the shared model, transport, or scheduler lock.

Relevant existing CPU suites for later integration are `AilaIpcProtocolTests`, `AilaWorkerDispatcherTests`, `AilaRuntimeIsolationTests`, and `AilaProxyAbiTests`. Their existing coverage includes stream ordering, callback abort, stale handles, worker failure, and bounded shutdown. They do not substitute for the socket-level SSE/disconnect probe above.
