# 2026-07-08 Tiny First Pause Hold

## Goal

Continue direction 3 from the checkpoint-350 optimization pass: reduce playback
gaps caused by tiny first phrases such as `啊` or `我` followed by ellipsis
pause segments and slower continuation text.

The issue observed after ellipsis-to-silence segmentation was not that the
first tiny phrase was expensive. It was that the first enqueue could contain
only `啊……`, then foreground generation waited for first TTS audio priority
before enough continuation text had been queued. The TTS worker then had too
little buffered audio while the next real text chunk was synthesized.

## Change

- `take_ready_tts_text_chunks()` now holds a low-latency first chunk when the
  hard-boundary candidate is only:
  - a text segment shorter than the first hard minimum, followed by
  - one or more synthetic ellipsis pause segments, and
  - no following text segment.
- `split_spoken_text_for_tts()` also merges that tiny pause-prefix chunk with
  the following chunk once continuation text arrives, so enqueue receives the
  continuation in the same call.
- Forced flush is unchanged, so final text still drains.

This keeps the first audible TTS text as the tiny interjection when the model
actually starts with `啊……`, but it makes the continuation and synthetic silence
available to the TTS queue before the foreground decode priority wait stalls
more token generation.

## Verification

Focused TDD:

```text
AilaAliaTtsTextChunkerTests: 49 passed, 0 failed
```

Release build:

```text
cmake --build build --config Release --target AliaEngine AilaAliaTtsTextChunkerTests
```

Short smoke:

```text
tmp/alia-real-smoke/ckpt350_tiny_pause_hold_short.log
ALIA_REAL_MODEL_SMOKE_PASS
foreground_lora_applied=true
foreground_profile_first_tts_chunk_pending_chars_at_enqueue=12
simulated_vad_to_first_audio_ms=986
tts_playback_total_gap_ms=518
tts_playback_buffer_total_gap_ms=114
```

The short smoke was not the target shape because the generated first chunk was
already 12 bytes, not a one-character pause prefix.

Real matrix:

```text
tmp/alia-real-smoke/ckpt350_tiny_pause_hold_matrix/summary.csv
ALIA_VOICE_SCENARIO_MATRIX_PASS
```

Compared with `tmp/alia-real-smoke/ckpt350_ellipsis_pause_matrix/summary.csv`:

```text
average TTFA:        942.0ms -> 937.0ms
average total gap:   317.6ms -> 341.8ms
average buffer gap:   69.2ms ->  40.8ms
average TTS backend: 4677.6ms -> 4337.7ms
```

Target scenario, `task_memory`:

```text
first enqueue pending:       9 -> 15 bytes
TTFA:                     944 -> 948 ms
playback total gap:       511 -> 385 ms
playback buffer gap:      119 ->   0 ms
```

## Conclusion

This is a targeted playback-continuity win for the one-character first phrase
case. It does not materially improve TTFA, and total playback gap still has
run-to-run variance in non-target rows, but it removes the buffer underrun in
the problematic `啊……` task-memory shape.

The next likely scheduler direction is deeper backend/enqueue overlap: when a
first text item is tiny and a following text item is already queued after a
synthetic pause, consider prioritizing or preparing the following text backend
work earlier instead of relying only on queue timing.
