# 2026-07-08 KV Quant Isolation

## Goal

Make FP8 KV cache quantization debuggable per backend instead of only through
the legacy global `AILA_KV_QUANT` switch.

The global switch was unsafe for the Alia real pipeline because it also enabled
Qwen3 ASR and Qwen3 TTS KV quantization, not only Qwen3.5 foreground/background
VLM cache quantization.

## Change

Added scoped KV quant environment handling:

```text
AILA_KV_QUANT      legacy global default, still supported
AILA_ASR_KV_QUANT  ASR override, inherits AILA_KV_QUANT when unset
AILA_TTS_KV_QUANT  TTS override, inherits AILA_KV_QUANT when unset
AILA_VLM_KV_QUANT  Qwen3.5 foreground/background override, inherits AILA_KV_QUANT when unset
```

Explicit scoped `0` suppresses global `1`, including the CLI global
`--kv-quant` override path. This allows experiments such as:

```powershell
$env:AILA_VLM_KV_QUANT = '1'
$env:AILA_ASR_KV_QUANT = '0'
$env:AILA_TTS_KV_QUANT = '0'
```

## Validation

Focused tests:

```text
AilaEnvUtilsTests: 5 passed, 0 failed
AilaAliaTtsTextChunkerTests: 49 passed, 0 failed
```

Build:

```text
cmake --build build --config Release --target AliaEngine AilaEnvUtilsTests AilaAliaTtsTextChunkerTests
```

### Global KV Quant Probe

Log:

```text
tmp/alia-real-smoke/ckpt350_global_kv_quant_short.log
```

Global `AILA_KV_QUANT=1` made ASR, Qwen3.5, and TTS all allocate quantized KV
caches. The run technically passed, but ASR was broken:

```text
asr_ms=3221
asr_profile_generated_tokens=256
asr_partial_text="艾!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!..."
simulated_vad_to_first_audio_ms=4244
```

Conclusion: do not use global KV quant for the real voice-text-voice pipeline.

### VLM-Only KV Quant Smoke

Log:

```text
tmp/alia-real-smoke/ckpt350_vlm_kv_quant_short.log
```

The scoped run confirmed isolation:

```text
ASR KVCache quantized=false
Qwen3.5 foreground quantized=true
Qwen3.5 background quantized=true
TTS KVCache quantized=false
asr_ms=396
simulated_vad_to_first_audio_ms=979
```

### VLM-Only KV Quant Matrix

Summary:

```text
tmp/alia-real-smoke/ckpt350_vlm_kv_quant_matrix/summary.csv
ALIA_VOICE_SCENARIO_MATRIX_PASS
```

Compared with `tmp/alia-real-smoke/ckpt350_tiny_pause_hold_matrix/summary.csv`:

```text
average TTFA:             937.0ms -> 939.8ms
average prompt prefill:   313.6ms -> 314.0ms
average foreground decode:1358.6ms -> 1971.0ms
average foreground model: 1679.2ms -> 2292.2ms
average playback gap:     341.8ms -> 411.2ms
average buffer gap:        40.8ms ->  38.8ms
average TTS backend:     4337.7ms -> 5049.7ms
```

Per-scenario notes:

```text
short_hello: TTFA 912 -> 899ms, but decode 1789 -> 3735ms and gap 640 -> 754ms
task_memory: TTFA 948 -> 965ms, decode 1739 -> 2789ms and gap 385 -> 573ms
shorter rows were mostly flat
```

## Conclusion

Scoped KV quant isolation is useful and should stay as a debug/experiment
surface. It prevents ASR/TTS corruption while allowing Qwen3.5-only A/B tests.

Do not enable `AILA_VLM_KV_QUANT` by default based on current evidence. It does
not improve prompt prefill or TTFA, and it slows foreground decode on the longer
rows enough to increase downstream TTS work and playback gaps.
