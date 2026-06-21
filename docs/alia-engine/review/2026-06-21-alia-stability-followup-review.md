# Alia Stability Follow-up Review

- Date: 2026-06-21
- Branch: `codex/alia-custom-engine`
- Review range: `56c21be..HEAD` (`096e0b6 Fix Alia TTS callback lane lock`)
- Reviewer: Codex
- Request: `docs/alia-engine/review-request/2026-06-21-alia-stability-followup-review.md`
- Prior review context: `docs/alia-engine/review/2026-06-17-alia-stability-regression-fix-review.md`

## Scope

This review checked the follow-up fixes for:

- `Context::ExecutionLock::scoped_unlock()`.
- `AliaTtsPipeline::synthesize_text()` releasing the foreground lane around host
  audio callbacks.
- ABI string ownership documentation for `alia_asr_get_text()` and
  `alia_free_string()`.
- Branch-status export surface consistency.
- Absence of per-primitive/per-layer lock regression in `Bnb4BitLinear.cpp` and
  `Linear.cpp`.

No files under `E:\RiderProjects\Alia` were read or modified.

## Findings

No blocking findings.

## Verified OK

- Host `AliaAudioCallback` invocation now happens outside the foreground
  execution lock. `AliaTtsPipeline::synthesize_text()` acquires the lane at
  `src/alia/AliaTtsPipeline.cpp:202`, creates a scoped unlock at
  `src/alia/AliaTtsPipeline.cpp:217`, and invokes `audio_cb` at
  `src/alia/AliaTtsPipeline.cpp:218`.
- `Context::ExecutionLock::ScopedUnlock` reacquires the lane in its destructor
  (`src/core/Context.hpp:20`, `src/core/Context.hpp:30`), so normal returns and
  exception unwinding do not leave the execution lane permanently unlocked.
- The temporary unlock is structurally narrow. The Qwen3-TTS backend invokes the
  callback only after `decode_mimi_incremental()` has returned a host audio
  vector (`src/models/Qwen3TTSBackend.cpp:1251` to
  `src/models/Qwen3TTSBackend.cpp:1267`), and `decode_mimi_incremental()` has
  already produced host samples and updated its local stream state before
  returning (`src/models/Qwen3TTSBackend.cpp:2276` to
  `src/models/Qwen3TTSBackend.cpp:2297`).
- The unlock does not expose concurrent TTS backend mutation through public Alia
  ABI paths. Foreground VLM prefill/rollback/start paths are rejected while the
  foreground pipeline is busy, ASR may safely submit work through the same
  foreground `Context` while TTS is paused, and allocator bookkeeping remains
  protected by `alloc_mutex_`.
- `include/alia_api.h` now documents that strings returned by
  `alia_asr_get_text()` must be released with `alia_free_string()` and not host
  allocators (`include/alia_api.h:92`, `include/alia_api.h:95`).
- `docs/alia-engine/2026-06-12-alia-engine-branch-status.md` now consistently
  lists `alia_free_string` in the export list (`docs/alia-engine/2026-06-12-alia-engine-branch-status.md:298`).
- The reviewed range does not modify `src/ops/Bnb4BitLinear.cpp` or
  `src/ops/Linear.cpp`, and no per-primitive/per-layer lock was reintroduced
  there.

## Open Questions

- I assume the host does not destroy `AliaContext` from inside an audio callback.
  That reentrancy remains unsafe in the broader ABI lifecycle, but it is outside
  the foreground lane-lock fix reviewed here.
- The provided real-model verification was treated as existing evidence; this
  review did not rerun the smoke matrix.

## Final Assessment

Ready to merge.
