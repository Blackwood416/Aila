# 2026-07-08 Ellipsis Pause Segment Pass

This pass tried the first user-suggested optimization direction: replace VLM
ellipsis text with fixed silence before TTS backend synthesis, instead of
letting Qwen3-TTS generate long pause audio from `…` text.

## Implementation

- Added `split_tts_text_pause_segments()` for `…`, `……`, and `...`.
- `AliaTtsPipeline::enqueue_text()` now expands a text chunk into text and
  silence queue items.
- Silence queue items emit 24 kHz PCM zero samples through the same audio
  callback path and do not enter the TTS backend.
- Added smoke/matrix metrics:
  - `tts_pause_silence_segments`
  - `tts_pause_silence_ms`

Default knobs:

```text
AILA_TTS_ELLIPSIS_PAUSE_SEGMENTS=1
AILA_TTS_ELLIPSIS_PAUSE_MS=160
AILA_TTS_ELLIPSIS_PAUSE_MAX_MS=240
```

## Focused Short Smoke

Log:

```text
tmp/alia-real-smoke/ckpt350_fresh_short_ellipsis_pause.log
```

Key metrics:

```text
simulated_vad_to_first_audio_ms=1013
tts_pause_silence_segments=1
tts_pause_silence_ms=160
tts_playback_total_gap_ms=436
tts_playback_buffer_total_gap_ms=0
```

The single smoke is not directly comparable to the prior short smoke because
the foreground answer was longer, but it confirmed that the ellipsis path was
hit and removed the previous small buffer gap.

## Matrix Result

Matrix:

```text
tmp/alia-real-smoke/ckpt350_ellipsis_pause_matrix/summary.csv
```

All five rows passed with output ASR verification.

```text
scenario             vad_to_first_audio  total_gap  buffer_gap  pause_segments  pause_ms
short_hello          915 ms              656 ms     227 ms      1               160
persona_chat         957 ms              108 ms     0 ms        0               0
preference_memory    960 ms              115 ms     0 ms        0               0
task_memory          944 ms              511 ms     119 ms      1               240
multi_turn_followup  934 ms              198 ms     0 ms        1               160
average              942.0 ms            317.6 ms   69.2 ms
```

Compared with the previous checkpoint-350 default matrix:

```text
vad_to_first_audio average:      932.4 ms -> 942.0 ms
tts_playback_total_gap average:  475.6 ms -> 317.6 ms
buffer_gap average:              104.2 ms -> 69.2 ms
tts_backend_total_ms average:   5913.5 ms -> 4677.6 ms
```

The tradeoff is worthwhile for playback continuity and backend load, but it is
not a TTFA win by itself. The remaining gap in hit scenarios comes from very
small first phrases such as `啊` followed by slower next-text synthesis.

## Rejected Probe

One task-memory probe tried:

```text
AILA_TTS_ELLIPSIS_PAUSE_MS=320
AILA_TTS_ELLIPSIS_PAUSE_MAX_MS=320
```

Log:

```text
tmp/alia-real-smoke/ckpt350_task_pause320.log
```

It was worse for the sampled output:

```text
tts_pause_silence_segments=3
tts_pause_silence_ms=960
tts_playback_buffer_total_gap_ms=222
```

Do not raise the default pause duration based on this pass.

## Next Lead

The next likely optimization is scheduler/chunker handling for tiny first
phrases before a longer continuation. Ellipsis replacement removes useless TTS
work, but it does not fully solve the gap after a very short first phrase.
