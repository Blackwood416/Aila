# 2026-07-09 Queue-Aware First-Audio Wait

This pass continued checkpoint-350 optimization for short Chinese voice turns.
It focused on tiny first phrases such as "ah"/"wo" before synthetic ellipsis
silence. KV quant is left off for speed-sensitive paths; ASR/TTS/VLM scoped KV
quant remains available from the previous pass but is not enabled by default.

## Change

- Added queue-aware tiny-first-text handling to foreground TTS first-audio
  priority wait.
- When the first enqueued TTS text starts with a tiny text segment before an
  ellipsis silence, foreground generation may continue briefly before waiting
  for first audio.
- Default policy:
  - `AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_QUEUE_AWARE_TINY=1`
  - `AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TINY_FOLLOWING_MIN_BYTES=16`
  - `AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TINY_MAX_DEFERRED_STEPS=2`
- Added smoke/matrix metrics:
  - `foreground_profile_tts_first_audio_priority_wait_deferred_count`
  - `foreground_profile_tts_first_audio_priority_following_text_bytes`

The deferred step cap matters. An uncapped prototype behaved like disabling
first-audio priority for tiny first phrases and could worsen gaps. A one-step
cap was too small on the target case.

## Real Model Evidence

Baseline:

```text
tmp/alia-real-smoke/ckpt350_tiny_pause_hold_matrix/summary.csv
```

Queue-aware capped matrix:

```text
tmp/alia-real-smoke/ckpt350_queue_aware_wait_capped_matrix/summary.csv
```

Five-scenario averages:

```text
simulated_vad_to_first_audio_ms: 937.0 -> 936.8 ms
tts_playback_total_gap_ms:       341.8 -> 348.8 ms
tts_playback_buffer_total_gap_ms: 40.8 -> 37.0 ms
first-audio priority wait:       218.0 -> 171.6 ms
```

Target `task_memory`:

```text
simulated_vad_to_first_audio_ms: 948 -> 978 ms
tts_playback_total_gap_ms:       385 -> 0 ms
tts_playback_buffer_total_gap_ms: 0 -> 0 ms
deferred_count:                  2
following_text_bytes:            18
```

Additional target smoke using the baseline request audio:

```text
tmp/alia-real-smoke/ckpt350_queue_aware_wait_capped_task.log
simulated_vad_to_first_audio_ms=988
tts_playback_total_gap_ms=0
tts_playback_buffer_total_gap_ms=0
foreground_profile_tts_first_audio_priority_wait_deferred_count=2
foreground_profile_tts_first_audio_priority_following_text_bytes=21
```

Rejected tuning probe:

```text
tmp/alia-real-smoke/ckpt350_queue_aware_wait_step1_task.log
AILA_FOREGROUND_TTS_FIRST_AUDIO_PRIORITY_TINY_MAX_DEFERRED_STEPS=1
simulated_vad_to_first_audio_ms=982
tts_playback_total_gap_ms=499
```

## Takeaway

Keep the capped queue-aware wait default-on for now. It is narrowly targeted:
non-tiny first chunks report `deferred_count=0`, and the matrix average TTFA is
flat. The target short Chinese tiny-prefix case trades roughly 30 ms TTFA for
removing the playback gap.

Direction 2, TTS worker prefetch after synthetic silence, was tested next and
is documented in `docs/alia-engine/2026-07-09-tts-silence-lookahead-prefetch.md`.
Keep that probe default-off; real matrix evidence showed worse playback gaps
when the worker withheld silence to prefetch the following text.
