# 2026-07-09 TTS Silence Lookahead Prefetch

This pass tested direction 2 after the queue-aware first-audio wait change:
prefetching the TTS text item after a synthetic ellipsis silence. The result is
negative for default behavior. The implementation is kept default-off as an
instrumented probe only.

## Change

- Added `AILA_TTS_SILENCE_LOOKAHEAD_PREFETCH=0` default-off.
- Added `AILA_TTS_SILENCE_LOOKAHEAD_PREFETCH_MIN_TEXT_BYTES=12`.
- The async TTS worker may prefetch only when:
  - the current item is a synthetic silence segment;
  - first audio has already been emitted;
  - the next queue item is text;
  - the next text item is at least `min_text_bytes`.
- Prefetched text audio is buffered locally. The worker emits the synthetic
  silence first, then the buffered text callbacks, preserving audio order.
- Added smoke/matrix metrics:
  - `tts_silence_lookahead_prefetch_enabled`
  - `tts_silence_lookahead_prefetch_min_text_bytes`
  - `tts_silence_lookahead_prefetch_attempts`
  - `tts_silence_lookahead_prefetch_hits`
  - `tts_silence_lookahead_prefetch_tiny_skips`
  - `tts_silence_lookahead_prefetch_text_bytes`
  - `tts_silence_lookahead_prefetch_audio_callbacks`
  - `tts_silence_lookahead_prefetch_audio_samples`
  - `tts_silence_lookahead_prefetch_backend_ms`

## Real Model Evidence

Aggressive target smoke before the `min_text_bytes` guard showed the main
failure mode:

```text
tmp/alia-real-smoke/ckpt350_silence_lookahead_prefetch_task_smoke.log
foreground_assistant_text="ah... na, that..."
simulated_vad_to_first_audio_ms=986
tts_silence_lookahead_prefetch_hits=1
tts_silence_lookahead_prefetch_text_bytes=3
tts_silence_lookahead_prefetch_backend_ms=853.827
tts_playback_total_gap_ms=1145
tts_playback_buffer_total_gap_ms=477
```

The guarded target smoke skipped the tiny next text item:

```text
tmp/alia-real-smoke/ckpt350_silence_lookahead_prefetch_min12_task_smoke.log
simulated_vad_to_first_audio_ms=988
tts_silence_lookahead_prefetch_attempts=1
tts_silence_lookahead_prefetch_hits=0
tts_silence_lookahead_prefetch_tiny_skips=1
tts_playback_total_gap_ms=363
tts_playback_buffer_total_gap_ms=0
```

Guarded full matrix:

```text
tmp/alia-real-smoke/ckpt350_silence_lookahead_prefetch_min12_matrix/summary.csv
```

Compared with the previous queue-aware baseline:

```text
baseline avg simulated_vad_to_first_audio_ms: 936.8
new avg simulated_vad_to_first_audio_ms:      905.0
baseline avg tts_playback_total_gap_ms:       348.8
new avg tts_playback_total_gap_ms:            606.8
baseline avg tts_playback_buffer_total_gap_ms: 37.0
new avg tts_playback_buffer_total_gap_ms:     166.6
```

Matrix hit details:

```text
short_hello: hits=1 text_bytes=69 backend_ms=1909.74 total_gap=2271 buffer_gap=833
task_memory: attempts=2 hits=0 tiny_skips=1 total_gap=0 buffer_gap=0
```

## Takeaway

Do not enable this by default. The existing worker already emits synthetic
silence quickly, then starts the next text backend work while that silence is
available to playback. The lookahead variant moves backend work before the
silence callback. That can reduce first-audio timing in some rows, but it moves
the audible gap earlier and makes playback continuity worse when a substantive
next text item is prefetched.

Further work on this direction would need a real concurrent TTS context or a
playback-buffer-aware scheduler. The single-worker buffered lookahead tested
here is not a good latency/gap trade-off.
