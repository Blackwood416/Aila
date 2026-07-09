# 2026-07-09 TTS Speaker Encoder VRAM Cleanup

## Change

`Qwen3TTSBackend::load()` now erases `speaker_encoder.*` tensors from the TTS
`ModelWeights` after talker/code-predictor setup and before runtime buffer and
warmup allocation.

The Alia reference voice path still uses the CPU `SpeakerEncoder` in
`AliaTtsPipeline::ensure_reference_voice_loaded()` to extract one cached
speaker embedding. That CPU extractor is local to the load call and is not kept
alive after extraction.

## Reason

`LoadSafetensors()` uploads every tensor in
`Qwen3-TTS-12Hz-0.6B-Base/model.safetensors`, including `speaker_encoder.*`.
The TTS backend inference path does not use those GPU tensors; reference
embedding extraction reads the same safetensors through the CPU
`SpeakerEncoder`.

Actual Base model header count:

```text
total_tensors=478
total_mb=1744.54
speaker_encoder_tensors=76
speaker_encoder_mb=16.89
speaker_encoder_percent=0.97
```

The cleanup is intentionally before TTS warmup, so warmup temporary tensors and
oneDNN user scratchpads do not overlap with the unused speaker encoder weights.
Existing warmup primitive/scratch caches are preserved because they are used to
avoid first-synthesis JIT and allocation spikes.

## Verification

Focused test:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --config Release --target AilaSafeTensorsTests
.\build\AilaSafeTensorsTests.exe
```

Result:

```text
AilaSafeTensorsTests: 11 passed, 0 failed
```

Build:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --config Release --target AliaEngine AilaSafeTensorsTests
```

Result: build completed.

Real short smoke with checkpoint-350:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio `
  -SkipToolProbe -StreamAsrPrefill -StreamPrefillIntervalMs 1000 `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\tts_speaker_encoder_vram_short.wav' `
  -LogPath 'tmp\alia-real-smoke\tts_speaker_encoder_vram_short.log' `
  -RequestText '艾莉亚，请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Key output:

```text
[TTS] Erased unused speaker_encoder GPU weights: 76 tensors, 16.89 MB freed
foreground_lora_applied=true
tts_reference_audio_enabled=1
tts_reference_embedding_dim=1024
tts_first_audio_ms=642
simulated_vad_to_first_audio_ms=959
ALIA_REAL_MODEL_SMOKE_PASS
```
