# Alia Stability Regression Fix Review

- Date: 2026-06-17
- Branch: `codex/alia-custom-engine`
- Review range: `git diff 4585303..codex/alia-custom-engine`
- Reviewer: Codex
- Request: `docs/alia-engine/review-request/2026-06-17-alia-stability-regression-review.md`
- Prior review context: `docs/alia-engine/review/2026-06-17-branch-quality-review.md`

## Scope

This review checked the stability and correctness fixes for the Alia custom
engine branch, with focus on:

- Shared foreground `Context` thread safety after the lane-lock regression fix.
- Performance risk in `src/ops/Bnb4BitLinear.cpp` and `src/ops/Linear.cpp`.
- Alia C ABI ownership and exception mapping.
- Rollback replay correctness after tool-result resume.
- Accuracy of `docs/alia-engine/2026-06-12-alia-engine-branch-status.md`.

No files under `E:\RiderProjects\Alia` were read or modified.

## Findings

### P1 - Host audio callback runs while holding the foreground execution lock

- File: `src/alia/AliaTtsPipeline.cpp:202`
- Related: `src/alia/AliaTtsPipeline.cpp:209`, `src/core/Context.hpp:18`,
  `src/alia/AliaContext.cpp:13`, `src/alia/AliaAsrPipeline.cpp:337`,
  `src/alia/AliaAsrPipeline.cpp:427`

`AliaTtsPipeline::synthesize_text()` holds the foreground
`Context::ExecutionLock` while invoking the host `audio_cb`. Because the
execution lock is backed by a non-recursive `std::mutex`, and ASR, foreground
VLM, and TTS share the same foreground `Context`, an audio callback that
re-enters an Alia ABI call such as `alia_asr_get_text()` can self-deadlock once
ASR transcription tries to acquire the same lane lock. It can also deadlock with
another thread holding the ASR mutex while waiting for the lane.

Recommended fix: do not invoke host callbacks while holding the foreground lane
lock. Buffer or copy the emitted samples, release the lane lock, then call the
host callback. If callbacks are intentionally non-reentrant, document that as an
ABI contract, but avoiding callbacks under internal locks remains safer.

### P2 - `alia_free_string` ownership is implemented but not documented in the ABI header

- File: `include/alia_api.h:89`
- Related: `src/alia/AliaApi.cpp:17`, `src/alia/AliaApi.cpp:164`

The implementation correctly allocates ASR strings with `std::malloc` and adds
`alia_free_string()` implemented with `std::free`, but the public ABI header does
not state that callers must release strings returned by `alia_asr_get_text()`
with `alia_free_string()` rather than host-side allocators.

Recommended fix: add a short ownership comment next to `alia_asr_get_text()` and
`alia_free_string()`.

### P3 - Branch status export list is stale

- File: `docs/alia-engine/2026-06-12-alia-engine-branch-status.md:298`
- Related: `docs/alia-engine/2026-06-12-alia-engine-branch-status.md:535`

The earlier dumpbin export list omits `alia_free_string`, while the later
2026-06-17 section says it is present. The status document should be internally
consistent.

Recommended fix: update the earlier export list to include `alia_free_string`.

## Verified OK

- `Context` allocator bookkeeping is now protected by `alloc_mutex_`.
- ASR transcription, foreground VLM prefill/decode/rollback, and TTS synthesis
  take the foreground lane lock around the reviewed GPU submission windows.
- The fix did not reintroduce per-layer or per-primitive locks in
  `src/ops/Bnb4BitLinear.cpp` or `src/ops/Linear.cpp`; `Linear.cpp` is unchanged
  in the reviewed range and `Bnb4BitLinear.cpp` has no substantive hot-path
  change.
- The `alia_*` entry points visible in `src/alia/AliaApi.cpp` are guarded so C++
  exceptions do not escape the C ABI. `ModelBackendCancelled` maps to
  `ALIA_ERR_ABORTED`; standard and unknown failures map to `ALIA_ERR_RUNTIME`.
- Rollback replay now records tool-result continuation tokens in the replayable
  sequence and refuses success unless the current backend context length matches
  the recorded sequence length.

## Open Questions

- I assumed host callbacks may call back into the Alia ABI unless the ABI
  explicitly forbids reentrancy.
- The verification commands listed in the review request were treated as
  existing evidence; this review did not rerun the real-model smoke matrix.

## Final Assessment

Ready with fixes. The core foreground lane-lock approach appears substantially
correct and avoids the prior per-primitive lock performance regression, but the
callback-under-lane-lock deadlock should be fixed before merge.
