# 2026-06-30 TTS Mimi decoder conv2 residual fusion

## Goal

Reduce recurring Mimi decoder block cost after stream shape warmup removed the
first-use ConvNeXt upsample spike.

The current runtime hotspot is no longer ConvNeXt upsample. For 28-frame conv
windows, `mimi_conv_stages` now spends most of its time in decoder residual
blocks:

```text
default shape-warmup profile, 28-frame window:
Decoder blocks: 133-135 ms
res_conv1:      62-63 ms
res_conv2:      35-36 ms
res_add:         1.7-2.0 ms
```

`conv1` is the largest remaining piece, but `conv2` is kernel-size 1 and already
had a default-off fused kernel available.

## Implementation

`AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL` is now default-on:

```text
AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL=1   default, fuse k1 conv2 + residual add
AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL=0   fallback, separate conv2 and add
```

The fused kernel writes directly into the residual buffer:

```text
residual = residual + bf16(conv2(input))
```

This preserves the old bf16 rounding point of the intermediate `conv2_out`
tensor while removing one temporary tensor and one residual-add kernel launch
per residual block.

## Reprobe

Command:

```powershell
$env:AILA_TTS_PROFILE = "1"
$env:AILA_TTS_MIMI_DECODER_BLOCK_PROFILE = "1"
$env:AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL = "1"
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\tts_fused_conv2_reprobe\short.wav' `
  -LogPath 'tmp\alia-real-smoke\tts_fused_conv2_reprobe\short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500
```

Representative profile comparison:

```text
window frames   default decoder   fused decoder   default conv2   fused conv2
4               27-31 ms          22-24 ms        6.5-7.4 ms     3.9-4.1 ms
8               45-49 ms          36-40 ms        11.6-11.7 ms   6.7-6.8 ms
12              64-65 ms          53-56 ms        16.1-17.3 ms   9.7-10.4 ms
16              80-83 ms          68-72 ms        21.5-22.6 ms   12.6-13.3 ms
24              117-118 ms        101-105 ms      31.7-32.3 ms   19.1-20.1 ms
28              133-135 ms        115-120 ms      35.0-36.6 ms   21.5-22.2 ms
```

The fused path also reports `res_add=0.0 ms`, as expected.

Output ASR on the generated wav:

```text
Kurasu，父亲大人说过要礼貌回应，虽然有点害羞，但我会努力做的。
```

The transcript matches the foreground text:

```text
Kurashu...父亲大人说过要礼貌回应。虽然有点害羞，但我会努力做的。
```

## Real matrix

The first full matrix attempt produced one non-TTS failure: `task_memory`
sampled an empty foreground response, so no TTS output was generated. A targeted
retry of the same task-memory audio passed with output ASR:

```text
tts_first_backend_audio_ms=219.033
output_asr_text=那个……父亲大人说过要诚实回答，但我不确定该不该说。艾莉亚会尽力，但可能说错。
```

A second full matrix passed 5/5:

```powershell
if (-not $env:MIMO_API_KEY) {
  $env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable('MIMO_API_KEY', 'Machine')
}
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\voice_matrix_fused_conv2_default_retry'
```

Summary: `tmp\alia-real-smoke\voice_matrix_fused_conv2_default_retry\summary.csv`

```text
scenario           vad_to_audio  first_audio  backend_audio  buffer_gap_total
short_hello        994 ms        637 ms       209.737 ms     56 ms
persona_chat       989 ms        566 ms       213.278 ms     0 ms
preference_memory  1256 ms       829 ms       224.907 ms     725 ms
task_memory        978 ms        564 ms       214.067 ms     0 ms
long_answer        1264 ms       708 ms       228.669 ms     148 ms
```

Averages:

```text
simulated_vad_to_first_audio_ms   1096.2
tts_first_audio_ms                 660.8
tts_first_backend_audio_ms         218.1
tts_playback_buffer_total_gap_ms   185.8
```

All five scenarios passed, and output ASR recognized the generated wavs. The
backend first-audio metric improved slightly from the stream-shape warmup
matrix average (`219.8 ms` to `218.1 ms`) while the profile shows the larger
benefit in recurring decoder-block windows. Playback buffer gaps are still
variable because several retry scenarios sampled long or fragmented foreground
responses; that points to chunk scheduling and foreground/TTS coordination
rather than the fused conv2 kernel itself.

## Conclusion

Promote the conv2 residual fusion to the default path. It does not change TTS
chunk sizing or sampling, and the short reprobe shows a consistent decoder-block
reduction across the recurring stream windows. The next larger target remains
decoder residual `conv1`, which dominates the post-fusion decoder block cost.
