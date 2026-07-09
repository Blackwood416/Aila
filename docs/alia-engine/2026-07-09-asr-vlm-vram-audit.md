# 2026-07-09 ASR/VLM VRAM Audit

Scope: inspect ASR and VLM backend residency for unused GPU weights or warmup
artifacts. The VLM vision tower remains loaded for future visual input.

## Header Summary

Fixed model safetensors metadata:

```text
ASR Qwen3-ASR-1.7B-BNB-NF4
  thinker.audio_tower BF16 397 tensors 605.54 MB
  thinker.model       BF16 114 tensors 593.74 MB
  thinker.model       F32  588 tensors   0.53 MB
  thinker.model       U8   588 tensors 693.03 MB

Foreground VLM qwen3.5-4B-bnb-nf4-offline-visiondense
  model.language_model BF16 178 tensors 1214.33 MB
  model.language_model F32  744 tensors    1.09 MB
  model.language_model U8   744 tensors 1755.10 MB
  model.visual         BF16 297 tensors  636.13 MB

Background VLM qwen3.5-0.8B-bnb-nf4-offline
  model.language_model BF16 134 tensors 485.95 MB
  model.language_model F32  558 tensors   0.31 MB
  model.language_model U8   558 tensors 244.73 MB
```

## Findings

- ASR `thinker.audio_tower.*` is actually bound by
  `Qwen3ASRAudioEncoder`: conv frontend, 24 encoder layers, `ln_post`, and
  `proj1/proj2`. No speaker-encoder-like unused subtree was found.
- Foreground VLM `model.visual.*` is loaded and should stay loaded. The vision
  tower weights are all BF16 in the fixed model, so there is no broad f32 to
  bf16 duplicate.
- The foreground VLM patch projection is temporal-2:
  `model.visual.patch_embed.proj.weight = 1024x3x2x16x16` (3.00 MB). The
  backend fuses this into a 2D patch projection copy and no longer needs the
  original temporal source tensor. This source is now erased after the fuse.
- `warmup_loaded_vlm()` only warms the language prompt/prefix path. It does not
  call the vision encoder warmup, so no image runtime buffers are retained
  during the current voice-only smoke path.
- ASR and background VLM tied `lm_head` preprocessing are larger possible VRAM
  knobs, but they are performance caches rather than unused weights. Direct
  tied lm_head mode needs separate ASR/background A/B before changing defaults.

## Validation

Build and tests:

```text
cmake --build build --config Release --target AliaEngine AilaQwen35VisionWeightsTests AilaSafeTensorsTests AilaQwen3TTSWarmupTests
AilaQwen35VisionWeightsTests: 5 passed, 0 failed
AilaSafeTensorsTests: 11 passed, 0 failed
AilaQwen3TTSWarmupTests: 15 passed, 0 failed
```

Real short Chinese smoke with checkpoint-350 LoRA:

```text
[Vision] Erased temporal patch source GPU weight after 2D fuse: 3.00 MB freed
foreground_lora_applied=true
foreground_lora_pair_count=32
tts_reference_audio_enabled=1
simulated_vad_to_first_audio_ms=906
tts_playback_gap_count=0
ALIA_REAL_MODEL_SMOKE_PASS
```
