# 2026-06-30 TTS Mimi stream shape warmup

## Goal

Remove runtime latency spikes from the Mimi vocoder streaming path without
changing audio chunk sizing or codec sampling behavior.

After the transpose-conv vec8 change, a profile run still showed occasional
large `ConvNeXt upsample` spikes for newly observed streaming window lengths.
The spike behaved like oneDNN primitive/JIT setup for a new shape rather than
true steady compute:

```text
tmp\alia-real-smoke\bigger_kernel_baseline_profile\short.log
ConvNeXt upsample (2 layers): 151.7 ms

tmp\alia-real-smoke\transpose_conv_vec8_profile\short.log
ConvNeXt upsample (2 layers): 144.4 ms
```

## Implementation

`Qwen3TTSBackend::load_mimi_vocoder` now warms the common streaming conv-stage
window sizes after the regular 4-frame vocoder warmup:

```text
AILA_TTS_MIMI_STREAM_SHAPE_WARMUP=1   default, warm frames 17, 28, 32
AILA_TTS_MIMI_STREAM_SHAPE_WARMUP=0   fallback, only the original warmup
```

The warmup feeds zero `pre_tfm_out` tensors into `mimi_conv_stages`; it does not
run the text/code generator and does not alter runtime chunk policy. The goal is
only to make the backend pay fixed conv-stage shape setup during model load.

## Profile result

Profile command:

```powershell
$env:AILA_TTS_PROFILE = "1"
$env:AILA_TTS_MIMI_DECODER_BLOCK_PROFILE = "1"
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\tts_shape_warmup_profile\short.wav' `
  -LogPath 'tmp\alia-real-smoke\tts_shape_warmup_profile\short.log' `
  -RequestText 'Alia, please say hello in one short sentence.' `
  -MaxTokens 48 -TimeoutSec 1500
```

The profile shows the first 17-frame ConvNeXt spike moving into warmup:

```text
warmup:
ConvNeXt upsample (2 layers): 137.3 ms
[TTS] Mimi stream shape warmup complete (frames=17,28,32)

runtime request:
ConvNeXt upsample (2 layers): 1.3 ms
ConvNeXt upsample (2 layers): 2.6 ms
ConvNeXt upsample (2 layers): 3.4 ms
...
```

Runtime request ConvNeXt upsample calls in that smoke stayed in the
approximately `0.8-3.4 ms` range instead of hitting another 100 ms class spike.

## Real matrix

Command:

```powershell
if (-not $env:MIMO_API_KEY) {
  $env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable('MIMO_API_KEY', 'Machine')
}
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -VerifyOutputAsr -OutputDir 'tmp\alia-real-smoke\voice_matrix_shape_warmup'
```

Summary: `tmp\alia-real-smoke\voice_matrix_shape_warmup\summary.csv`

```text
scenario           vad_to_audio  first_audio  backend_audio  buffer_gap_total
short_hello        968 ms        611 ms       212.825 ms     47 ms
persona_chat       1247 ms       826 ms       223.818 ms     184 ms
preference_memory  1250 ms       827 ms       222.487 ms     0 ms
task_memory        1185 ms       774 ms       222.681 ms     179 ms
long_answer        1321 ms       768 ms       216.958 ms     0 ms
```

Averages:

```text
simulated_vad_to_first_audio_ms   1194.2
tts_first_audio_ms                 761.2
tts_first_backend_audio_ms         219.8
tts_playback_buffer_total_gap_ms    82.0
```

All five scenarios passed. Output ASR recognized the generated wavs for all
scenarios.

For comparison, the previous transpose-conv vec8 matrix averaged
`226.6 ms` first backend audio and `132.6 ms` playback buffer gap total. The
end-to-end first-audio metric remains dominated by ASR/VLM and scheduling
variance, but the Mimi backend path is less exposed to first-use conv-shape
spikes.

## Conclusion

Keep stream shape warmup enabled by default on this branch. It is a one-time
model-load cost that reduces runtime tail latency in the TTS playback path.

Next likely TTS backend targets:

- residual codec-frame generation and pre-transformer scheduling cost;
- more direct overlap between codec generation and Mimi streaming decode;
- MTP/dynamic chunk experiments only with output ASR and playback-gap checks.
