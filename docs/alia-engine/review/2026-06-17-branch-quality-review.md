# Alia Custom Engine — Branch Quality Review

- Date: 2026-06-17
- Branch: `codex/alia-custom-engine`
- Baseline: `main`
- Scope: 33 files, +8198 lines (core new code under `src/alia/`, ~4000 lines)
- Reviewer: Claude (Opus 4.8), code-review pass over `git diff main...HEAD`
- Method: full read of all new `src/alia/` sources + backend integration diffs
  (`Qwen35HybridBnb4Backend`, `Qwen3TTSBackend`, `LoraLoader`,
  `AssistantOutputParser`), ABI header, and `CMakeLists.txt`; findings verified
  against the code directly.

> Status note: this is a correctness/stability review, not a fix. Each finding
> cites `file:line`. Severities: 🔴 high, 🟠 medium, 🟡 low-medium, ⚪ minor.

---

## 1. Engine goal (from the PRD)

`E:\RiderProjects\Alia\docs\design\aila_engine_prd.md` defines the target: turn the
generic single-model Aila engine into an **Alia-specific multi-model real-time
voice runtime**, shipped as `AilaShared.dll` with a pure C ABI consumed by the
C# host over P/Invoke. One `AliaContext` owns four pipelines over a shared SYCL
device/context:

- **ASR** (Qwen3-ASR-1.7B): streaming 16 kHz f32 PCM in → stable/partial text.
- **Foreground VLM** (Qwen3.5-4B NF4 + identity LoRA): low-latency turn handling,
  tool-call interception, incremental prefill.
- **TTS** (Qwen3-TTS + Mimi): streams VLM tokens → 24 kHz PCM via audio callback.
- **Background VLM** (0.8B): async memory extraction, schema-constrained JSON.

Key non-functional targets: TTFT ≤ 400 ms, decode ≥ 35 tok/s, abort ≤ 100 ms,
foreground/background queue isolation, zero-marshalling internal text flow, and
**no exception may escape the DLL**.

## 2. Progress (from commit history + branch-status)

The ABI skeleton and four-pipeline orchestration are **implemented and pass a
real-model full-pipeline smoke** (`RunAliaTargetPipeline.ps1` →
`ALIA_REAL_MODEL_SMOKE_PASS`; the 5-scenario voice matrix passes). Implemented:
multi-slot load/validate, foreground prefill+decode with overlapped streaming
TTS, tool-call pause/resume, identity-LoRA application (PEFT prefix normalization
+ full-attention q/k/v/o adapters), background JSON schema validation/retry/repair,
cooperative abort (per-layer in VLM, per-batch in TTS), ASR-partial prefill reuse.

The most recent ~15 commits are almost all TTS/ASR perf micro-tuning. **Core gaps
(acknowledged by the branch's own docs):** first content still ~3.7 s and first
audio ~4.9–5.4 s — far from the 400 ms TTFT target; 4B selective rollback crashes
and is not production-ready; Computer Use / YOLO / SAM / WGC are phase 2; hard
abort latency is not yet verified against real assets with timing probes.

Overall engineering quality is high (RAII wrappers, clear state machines, stable
error codes, thorough docs), but several **clear defects** must be addressed.

---

## 3. Findings (ranked by severity)

### 🔴 1. Foreground `Context` used concurrently by the VLM decode thread and the TTS worker thread — data race / heap corruption

`configure_model_slots()` binds the **ASR, foreground VLM, and TTS** slots all to
the same `runtime->foreground()` Context — `src/alia/AliaContext.cpp:14-20`.
`run_turn` starts the **TTS worker thread** via `start_async_turn`, then runs the
VLM decode loop on *this* thread; inside the loop `flush_tts` hands sentences to
the TTS thread for concurrent synthesis —
`src/alia/AliaForegroundPipeline.cpp:782-826` and `:1094-1096`. Both threads call
`Tensor::allocate → alloc_device/free_device` on the **same** Context.

That allocator is **unsynchronized**: `src/core/Context.hpp:31-57` mutates
`std::unordered_map alloc_bytes_` and `current_allocated_bytes_` with no mutex.
Concurrent insert/erase on one `unordered_map` from two threads is undefined
behavior → likely heap corruption / crash. The single `dnnl::stream` executed
concurrently from two threads is likewise not thread-safe.

- **Trigger:** any normal voice turn (overlap of VLM decode and TTS synthesis is
  by design). This matches the branch-status "`task_memory` access violation after
  foreground/TTS" — attributed there to a tool_call fragment, but this race is an
  equally plausible (and more insidious) root cause; non-deterministic
  reproduction is the signature of a data race. If the host also calls
  `alia_asr_get_text` mid-turn (ASR runs GPU work synchronously on the foreground
  Context — `src/alia/AliaAsrPipeline.cpp:337-343`), it becomes a three-thread race.
- **Fix direction:** lock the Context allocator/stream, or give TTS its own lane
  (separate queue+stream+allocator, still sharing the SYCL context to satisfy
  R-201 zero-copy), or serialize VLM/TTS Context access. The current single
  in-order foreground queue already degrades the overlap to serial on the GPU, so
  the overlap buys little while carrying high risk.

### 🔴 2. Alia ABI returns `malloc`'d strings but exports no free function

`alia_asr_get_text` returns stable/partial copied with `std::malloc` —
`src/alia/AliaApi.cpp:13-20`, `:137-172`. The new Alia ABI header
`include/alia_api.h` (10 exports) has **no** free function; the old
`aila_free_string` was removed together with `src/api/aila_api.cpp` from the build
and export surface (`CMakeLists.txt`; branch-status `dumpbin` confirms only
`alia_*` are exported).

- **Consequence:** the C# host receives pointers on the DLL's CRT heap with no
  matching deallocator → **memory leak**, or freeing with the host's allocator
  (`Marshal.FreeHGlobal`, etc.) → **cross-heap free → heap corruption**. The smoke
  uses `std::free` internally (same CRT), masking the host-side defect. Violates
  PRD R-501 static-marshalling intent.
- **Fix direction:** export `alia_free_string(char*)`, or switch to a
  host-preallocated `out_buf/max_len` convention (as the tool callback already uses).

### 🟠 3. `alia_vlm_rollback_kv_cache` / `alia_vlm_prefill_asr_text` are not exception-guarded — C++ exceptions escape the C ABI

Both entry points run GPU work synchronously (`DeviceAllocation → alloc_device`
throws `std::runtime_error` on OOM; `backend->forward` may throw
`ModelBackendCancelled` or runtime errors) but have **no** try/catch —
`src/alia/AliaApi.cpp:82-108`. Contrast `alia_context_init` and
`alia_asr_feed_audio` in the same file, which do wrap try/catch.

- **Consequence:** a C++ exception crosses the `extern "C"` boundary into .NET
  P/Invoke → process crash. Directly violates architecture doc §11 ("No exception
  should escape from AilaShared.dll") and PRD §7.2. `alia_asr_get_text`,
  `alia_start_conversation_turn`, and `alia_trigger_background_processing` also lack
  a top-level guard (string copy can throw `bad_alloc`; `std::thread` construction
  can throw `system_error`).
- **Fix direction:** wrap every `alia_*` entry point body in try/catch returning a
  stable error code (`ALIA_ERR_RUNTIME` / `ALIA_ERR_ABORTED`).

### 🟠 4. Rollback replay desyncs from real KV after a tool-resume turn

The resume pass `generate_with_loaded_vlm(record_generation_anchor=false)` forwards
the injected `<tool_result>` continuation tokens into the KV but only appends the
**sampled** tokens to `generation_token_ids_` —
`src/alia/AliaForegroundPipeline.cpp:1133-1138`. `rollback_kv_cache`'s
`replay_to_target` replays only `anchor_prompt_ids_ + generation_token_ids_` —
`:579-609`.

- **Consequence:** when the Hybrid backend cannot truncate exactly and falls back
  to replay, the rebuilt sequence is **missing the continuation tokens**; the
  length check `get_current_context_len()==replay_target_len` can still pass, so
  rollback "succeeds" with DeltaNet recurrent state corresponding to a different
  token sequence → corrupted subsequent generation. This is the same fragile area
  as the documented 4B rollback crash (branch-status lists rollback as a separate
  open investigation item; the observed crash may additionally involve recurrent
  state restore).
- **Fix direction:** include *all* forwarded tokens (incl. the continuation prompt)
  in the replay sequence, or disable rollback after a tool-resume turn until the
  path is fixed.

### 🟡 5. ASR holds its mutex across the full GPU transcription, blocking real-time audio feed

`process_pending` holds `mutex_` for the entire transcription while-loop —
`src/alia/AliaAsrPipeline.cpp:187-233` — while `feed_audio` needs the same mutex to
append audio — `:148-156`. `get_text` triggers `process_pending`, so calling
`alia_asr_get_text` runs hundreds of ms of ASR inference **on the caller's thread**
and blocks real-time audio ingestion meanwhile. Hurts 16 kHz streaming real-time
behavior (counter to R-301).

- **Fix direction:** move transcription out of the lock (snapshot the needed audio
  under lock, transcribe unlocked, commit results under lock), or run ASR on a
  dedicated worker.

### 🟡 6. Tool-call parser change lost unit-test coverage

`src/chat/AssistantOutputParser.cpp` gained malformed-parameter recovery
(`<parameter=id=42</parameter>`), a correctness-sensitive change — yet the same
branch removed `AilaChatTests` (incl. `AssistantOutputParserTests`) from the build
(`CMakeLists.txt`). The recovery path is now exercised only by the heavyweight
real-model smoke; the regression net is gone. branch-status itself notes LoRA tool
output is "less protocol-clean."

- **Fix direction:** keep the lightweight chat tests building, or add a focused
  parser test for the new recovery case.

### ⚪ 7. Dead code / minor

- `AliaContext::load_model_metadata()` is never called (init uses
  `configure_model_slots` + `load_model_slots`) — `src/alia/AliaContext.cpp:23-38`.
- `run_turn`'s local `decode_mode` is assigned once and never changes (always
  `LoadedVlm`) — `src/alia/AliaForegroundPipeline.cpp:798`.
- LoRA key normalization `prev_dot = rfind('.', last_dot-1)` relies on a trailing
  npos check; slightly fragile — `src/lora/LoraLoader.cpp`.

> Verified-OK: destruction order is **safe** — `AliaContext` member declaration
> order destroys pipelines (which join their worker threads) before the slots /
> runtime they reference; the background VLM uses an isolated `runtime->background()`
> Context, so it is correctly separated from the foreground race.

---

## 4. Conclusion

Goal is clear, skeleton completeness is high, and the real four-model pipeline runs
end-to-end — the "first usable milestone" (replace the mock DLL at the ABI level)
is met, with good engineering discipline. But it is **not production-ready**, and
two classes of clear defects should be fixed first:

1. **Correctness / stability:** #1 foreground-Context data race is top priority (it
   threatens the PRD "DLL must not crash" guarantee and is a likely root cause of
   the observed random AV); #2/#3 are deterministic host-facing ABI contract
   defects (no free export, exceptions escaping); #4 makes rollback unreliable in
   tool scenarios.
2. **Performance target:** core TTFT (~3.7 s first content / ~4.9 s first audio) is
   an order of magnitude off the 400 ms goal — already the branch's own #1 TODO and
   can proceed in parallel with the stability fixes.

Suggested fix order: **#1 → #2 → #3 → #4** (all stability/contract, high-risk but
local), then return to TTFT. #5/#6 can be folded in.
