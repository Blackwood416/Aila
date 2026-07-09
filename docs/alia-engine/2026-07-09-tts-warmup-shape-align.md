# 2026-07-09 TTS Warmup Shape Alignment

## Finding

The TTS backend had warmup-only shapes that did not match the Alia Base
reference-voice hot path:

- text projection warmup used `batch=4`;
- codec decode warmup used a truncated 5-token ChatML sequence;
- codec decode warmup passed an empty speaker embedding, so Base prefill warmed
  the no-speaker layout.

For Alia's real Base TTS path, text is formatted as:

```text
<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
```

A one-token text therefore has 9 formatted tokens. The Base reference-voice path
also includes one speaker slot, so the talker prefill shape differs from the old
empty-speaker warmup.

## Change

Added `Qwen3TTSWarmup.hpp` helpers for:

- the minimal formatted 9-token warmup text;
- a dummy zero speaker embedding for Base model warmup.

`Qwen3TTSBackend::load()` now uses those helpers for both text-projection and
codec-decode warmup. This avoids creating stale warmup-only `batch=4` and
no-speaker codec prefill cache entries, and warms the shape used by short Alia
reference-voice chunks.

## Verification

Focused test:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --config Release --target AilaQwen3TTSWarmupTests
.\build\AilaQwen3TTSWarmupTests.exe
```

Result:

```text
AilaQwen3TTSWarmupTests: 15 passed, 0 failed
```

Build:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --config Release --target AliaEngine AilaQwen3TTSWarmupTests AilaSafeTensorsTests
```

Result: build completed.

Real short smoke with checkpoint-350:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio `
  -SkipToolProbe -StreamAsrPrefill -StreamPrefillIntervalMs 1000 `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\tts_warmup_shape_align_short.wav' `
  -LogPath 'tmp\alia-real-smoke\tts_warmup_shape_align_short.log' `
  -RequestText '艾莉亚，请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Key output:

```text
[TTS] Text projection warmup complete (batch=9)
[TTS] Codec decode warmup complete (frames=2 text_tokens=9 speaker_dim=1024)
foreground_lora_applied=true
tts_reference_audio_enabled=1
tts_reference_embedding_dim=1024
tts_first_text_tokens=9
tts_first_backend_codes_ms=171.331
tts_first_backend_audio_ms=212.913
simulated_vad_to_first_audio_ms=966
ALIA_REAL_MODEL_SMOKE_PASS
```
