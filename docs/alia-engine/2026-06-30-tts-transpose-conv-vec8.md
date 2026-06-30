# 2026-06-30 TTS Mimi transpose-conv vec8

## Goal

Probe a larger TTS kernel after command submission microbenchmarks showed that
small copy/fill kernels are dominated by host submission overhead.

The selected target is Mimi decoder `causal_conv_transpose1d`. In the default
layout, ConvTranspose1d weights are `[in_ch, out_ch, kernel]`, so each output
element walks `in_ch` with a large weight stride. The new path rewrites Mimi
transpose-conv weights to `[out_ch, kernel, in_ch]` during vocoder load and lets
`ops::causal_conv_transpose1d` use contiguous vec8 loads for both input and
weight.

The runtime switch is:

```powershell
AILA_TTS_MIMI_TRANSPOSE_CONV_VEC8=1   # default, optimized layout
AILA_TTS_MIMI_TRANSPOSE_CONV_VEC8=0   # fallback, original layout
```

## Implementation

- `src/ops/ConvOps.cpp`
  - `causal_conv_transpose1d` now auto-detects `[out_ch, kernel, in_ch]`.
  - The optimized path keeps the same accumulation order while loading weight
    vectors contiguously.
- `src/models/Qwen3TTSBackend.cpp`
  - Mimi loader rewrites only transpose-conv weights:
    - `decoder.upsample.{0,1}.0.conv.weight`
    - `decoder.decoder.{1..4}.block.1.conv.weight`
  - Default is on for the custom Alia engine branch, with env fallback.

## Profile result

Short real pipeline smoke with:

```powershell
$env:AILA_TTS_PROFILE = "1"
$env:AILA_TTS_MIMI_DECODER_BLOCK_PROFILE = "1"
$env:AILA_TTS_MIMI_TRANSPOSE_CONV_VEC8 = "1"
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\transpose_conv_vec8_profile\short.wav' `
  -LogPath 'tmp\alia-real-smoke\transpose_conv_vec8_profile\short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500
```

Representative decoder transpose timings:

```text
window frames   before       after
4               ~14-16 ms    ~3 ms
8               ~25-28 ms    ~5-6 ms
12              ~37-39 ms    ~8-9 ms
28              ~83-85 ms    ~18-19 ms
32              ~95-97 ms    ~21-22 ms
```

The first backend audio callback is still mostly bounded by codec generation,
Mimi pre-transformer, and scheduling. The main win is lower steady TTS RTF and
fewer playback-buffer gaps for larger streaming windows.

## Real matrix

Command:

```powershell
$env:AILA_TTS_MIMI_TRANSPOSE_CONV_VEC8 = "1"
if (-not $env:MIMO_API_KEY) {
  $env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable('MIMO_API_KEY', 'Machine')
}
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\voice_matrix_transpose_conv_vec8'
```

Summary: `tmp\alia-real-smoke\voice_matrix_transpose_conv_vec8\summary.csv`

```text
scenario           vad_to_audio  first_audio  backend_audio  buffer_gap_total
short_hello        1248 ms       892 ms       234.482 ms     164 ms
persona_chat       1042 ms       600 ms       227.734 ms     280 ms
preference_memory  1014 ms       585 ms       227.078 ms     0 ms
task_memory        1013 ms       586 ms       220.935 ms     60 ms
long_answer        1356 ms       803 ms       222.578 ms     159 ms
```

Averages:

```text
simulated_vad_to_first_audio_ms   1134.6
tts_first_audio_ms                 693.2
tts_first_backend_audio_ms         226.6
tts_playback_buffer_total_gap_ms   132.6
```

All five scenarios passed. Output ASR recognized the spoken content for all
generated wavs.

## Conclusion

Keep the optimized transpose-conv layout enabled by default on this branch. It
does not materially move the first backend callback by itself, but it removes a
large steady-state Mimi decoder cost and gives the pipeline more room to avoid
playback gaps.

Next likely TTS backend targets:

- residual `conv1`/`conv2` cost in Mimi decoder blocks;
- ConvNeXt upsample variance spikes;
- more direct scheduling overlap between codec frame generation and Mimi
  callback execution.
