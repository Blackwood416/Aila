# 2026-07-08 Checkpoint-350 TTS Default Pass

This pass resumed Alia TTFA optimization with the active foreground LoRA:

```text
F:/unsloth/qwen35_4b_alia_identity_r16_lr5e4/checkpoint-350
```

Scope stayed limited to real voice-text-voice latency and playback gaps. Computer
Use, visual input, and tool-call work were not touched.

## Default Changes

- `AILA_TTS_FIRST_CHUNK_EARLY_TOKEN_DELAY`: `2 -> 0`
- `AILA_TTS_FIRST_CHUNK_EARLY_MS`: `80 -> 0`
- `AILA_TTS_PLAYBACK_AWARE_STEADY_BATCH`: `true -> false`
- `RunAliaTargetPipeline.ps1`, `RunAliaVoiceScenarioMatrix.ps1`, and the smoke
  executable fallback now default to checkpoint-350.
- `RunAliaVoiceScenarioMatrix.ps1` accepts and forwards `-ForegroundLora`.

## Fresh Short Smoke

Command used fresh request audio, not the stale June matrix fixture:

```text
tmp/alia-real-smoke/ckpt350_fresh_short_request.wav
```

Result log:

```text
tmp/alia-real-smoke/ckpt350_fresh_short_defaults_patch.log
```

Key metrics:

```text
simulated_vad_to_first_audio_ms=969
foreground_first_content_delta_ms=326
foreground_first_tts_enqueue_ms=361
tts_first_audio_ms=577
first_tts_enqueue_to_first_audio_ms=216
tts_backend_playback_aware_steady_batch=0
tts_playback_gap_count=3
tts_playback_max_gap_ms=295
tts_playback_total_gap_ms=443
tts_playback_buffer_gap_count=1
tts_playback_buffer_max_gap_ms=22
tts_playback_buffer_total_gap_ms=22
```

Compared with the prior fresh checkpoint-350 baseline (`1007ms` TTFA), this
reduced short-prompt TTFA by `38ms`. It also beat the previous env-only
fast-first/no-playback-aware candidate (`981ms`) while reducing its buffer gap
from `44ms` to `22ms`.

## Real Matrix

Matrix output:

```text
tmp/alia-real-smoke/ckpt350_defaults_patch_matrix/summary.csv
```

All five scenarios passed with output ASR verification:

```text
scenario             vad_to_first_audio  first_content  first_tts_enqueue  tts_first_audio
short_hello          902 ms              323 ms         359 ms             567 ms
persona_chat         954 ms              325 ms         361 ms             572 ms
preference_memory    945 ms              327 ms         362 ms             574 ms
task_memory          929 ms              323 ms         358 ms             573 ms
multi_turn_followup  932 ms              325 ms         360 ms             571 ms
average              932.4 ms            324.6 ms       360.0 ms           571.4 ms
```

Playback gap averages:

```text
tts_playback_total_gap_ms average=475.6
tts_playback_buffer_total_gap_ms average=104.2
```

The main tradeoff remains checkpoint-350's occasional longer answer shape. The
`task_memory` row had the largest buffer gap (`320ms`) after a longer first
phrase, while `persona_chat` and `multi_turn_followup` had no buffer gap.

## Follow-Up Notes

Stream ASR prefill still did not help these rows: the foreground prefill
candidate was rejected with `candidate suffix exceeds fast threshold`. The next
useful pass should either improve that conservative cached-prefix path with real
matrix evidence, or target TTS chunk scheduling specifically for long first
phrases without increasing short-prompt TTFA.
