# 2026-07-07 Handoff: Alia Current Status

This is the current handoff entry point after the July 7 TTFA optimization pass.
Performance work is intentionally paused for now; the next agent should use this
document for state, defaults, and "do not repeat" conclusions.

## Workspace

- Engine worktree: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine`
- Branch: `codex/alia-custom-engine`
- Latest performance commit before this docs handoff: `70f3f5f perf: scope Q35 snapshot suppression to final prefill`
- Main C API reference: `docs/API/Alia_C_API.md`
- TTFA optimization spec: `docs/superpowers/spec/2026-07-06-alia-ttfa-performance-optimization-spec.md`

## Current Model Contract

Use these exact models unless the user says otherwise:

- ASR: `models/Qwen3-ASR-1.7B-BNB-NF4`
- Foreground VLM: `models/qwen3.5-4B-bnb-nf4-offline-visiondense`
- Foreground LoRA: `F:/unsloth/qwen35_4b_alia_identity_r16_lr1e5/checkpoint-750`
- Background VLM: `models/qwen3.5-0.8B-bnb-nf4-offline`
- TTS: `models/Qwen3-TTS-12Hz-0.6B-Base`
- TTS reference audio: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine/alia_ref.wav`
- Default max sequence length for real smoke/host integration: `2048`

Smoke and matrix prompts should be Chinese. Mild "父亲大人" repetition is a
known corpus/model style issue, not a performance blocker by itself.

## Current Defaults To Preserve

- `tools/alia/RunAliaTargetPipeline.ps1` defaults foreground LoRA to
  `checkpoint-750`.
- First TTS chunk early flush is default-on through
  `AILA_TTS_FIRST_CHUNK_EARLY_FLUSH=true`.
- TTS first-audio priority is default-on with the adaptive active extra window.
- Foreground action-tag guard is bypassed because the current LoRA was trained
  without action tags; `out_action_tags_json` remains available for ABI
  compatibility and future avatar routing.
- `AILA_Q35_INTERNAL_PREFILL_STATE_SNAPSHOTS` defaults to `true`, but final
  foreground fresh/full prompt prefill temporarily suppresses internal recurrent
  snapshot exports. ASR prefill and cached-prefix paths keep the default
  checkpoint granularity for truncate/restore.
- Cached final prefixes remain shape-aware and conservative. The default policy
  rejects cached prefixes when the remaining final suffix is above the fast
  threshold.

## Landed Optimization Outcomes

### TTS First Chunk And First Audio

The TTFA spec's first two tracks were implemented and validated:

- `29a37e0 perf: add TTS first chunk early flush probe`
- `22d8af4 perf: update Alia TTFA defaults`

Useful matrix summaries:

- Baseline: `tmp/alia-real-smoke/tts_first_chunk_baseline_matrix/summary.csv`
- Early flush: `tmp/alia-real-smoke/tts_first_chunk_early_matrix/summary.csv`
- Current default/adaptive: `tmp/alia-real-smoke/adaptive_default_lora750_matrix/summary.csv`

Five-pass averages:

```text
baseline:     first_enqueue 459.4ms, first_audio 676.0ms, vad_to_audio 1035.4ms
early flush:  first_enqueue 398.8ms, first_audio 614.6ms, vad_to_audio  972.0ms
adaptive 750: first_enqueue 396.4ms, first_audio 616.4ms, vad_to_audio  972.4ms
```

Keep this path unless a future matrix shows playback gaps getting worse.

### Action Tags And LoRA

The active LoRA is checkpoint-750. Action-tag guard work was removed from the
hot path:

- `a9d5167 perf: bypass Alia stream action tag guard`

The latest Chinese smoke with checkpoint-750 reported:

```text
foreground_lora_applied=true
foreground_action_tag_count=0
```

Do not reintroduce an action-tag wait/guard in the low-latency path unless the
model contract changes back to producing action tags.

### Cached Prefix And Prefix Metrics

Prefix shape observability landed in:

- `7e2f0a3 perf: add Alia prefix shape metrics`

The A/B that forced a larger cached-prefix suffix threshold was worse:

```text
prefix_metrics_cn_matrix: vad_to_first_audio  978.6ms, prefilled 0.0, suffix 120.6
prefix_ab_max32_cn_matrix: vad_to_first_audio 1093.2ms, prefilled 90.2, suffix 30.4
```

Keep cached-prefix reuse conservative. Do not raise
`AILA_TURN_SCHEDULER_MAX_CACHED_FINAL_SUFFIX_TOKENS` or
`AILA_FOREGROUND_DECODE_SUFFIX_TOKENS` globally without new matrix evidence.

### Qwen3.5 Foreground Prefill Snapshot Work

The first attempt globally disabled Q35 internal prefill recurrent snapshots,
which risked making ASR/cached-prefix truncate restore coarser. The final change
kept the backend default enabled and scoped suppression only to final
foreground fresh/full prefill:

- `70f3f5f perf: scope Q35 snapshot suppression to final prefill`

Evidence:

```text
q35_default_cn_matrix:         prompt_prefill 328.0ms, vad_to_audio 976.6ms
q35_snapshot_limit1_cn_matrix: prompt_prefill 313.0ms, vad_to_audio 961.6ms
latest scoped short smoke:     prompt_prefill 316ms, first_content 328ms,
                               first_enqueue 390ms, action_tags 0
```

The scoped approach preserves interrupt/restore correctness: `abort` only sets
abort flags, while explicit rollback anchors after prompt prefill. ASR prefill
and cached-prefix paths still keep internal snapshots for common-prefix
truncate/restore.

### ASR Kernel And Prefix Reuse Work

These were tried and should stay secondary/default-off for now:

- ASR prefix reuse warm24 matrix was effectively flat:
  `1042.4ms` with prefix reuse vs `1041.8ms` without.
- M80 ASR prefill matrix was effectively neutral:
  `1052.4ms` with M80 vs `1051.4ms` off.
- Fused ASR SwiGLU reduced a local stage but made total ASR prefill worse in the
  short run.
- ASR attention/joint_matrix work was not pursued as the first target because
  current short ASR profiles are dominated by NF4 projection/FFN work.

## Latest Short Smoke

Command shape:

```powershell
$env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable('MIMO_API_KEY', 'Machine')
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\no_action_guard_cn_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\q35_scoped_internal_snapshot_suppress_cn_short.wav' `
  -LogPath 'tmp\alia-real-smoke\q35_scoped_internal_snapshot_suppress_cn_short.log' `
  -RequestText '艾莉亚，请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Key output:

```text
ALIA_REAL_MODEL_SMOKE_PASS
foreground_lora_applied=true
foreground_profile_final_prefix_path="fresh_full"
foreground_profile_prompt_tokens=121
foreground_profile_prefilled_prompt_tokens=0
foreground_profile_prompt_suffix_tokens=121
foreground_profile_prompt_prefill_ms=316
foreground_profile_first_content_delta_ms=328
foreground_profile_first_tts_enqueue_ms=390
first_tts_enqueue_to_first_audio_ms=225
simulated_vad_to_first_audio_ms=1008
foreground_action_tag_count=0
```

## Stop Point

Do not continue performance optimization unless the user explicitly reopens it.
The next workstream is documentation completion and host integration readiness.
If optimization resumes later, start from the current-state metrics above and
prefer a fresh matrix over isolated short-smoke wins.

## Verification Commands For Future Code Changes

Build and focused tests:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --config Release --target AliaEngine AilaAliaTurnSchedulerTests AilaAliaTtsTextChunkerTests
.\build\AilaAliaTurnSchedulerTests.exe
.\build\AilaAliaTtsTextChunkerTests.exe
```

Short Chinese smoke:

```powershell
$env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable('MIMO_API_KEY', 'Machine')
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\<run-name>_short.wav' `
  -LogPath 'tmp\alia-real-smoke\<run-name>_short.log' `
  -RequestText '艾莉亚，请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Matrix only when behavior or scheduling changes:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -VerifyOutputAsr `
  -OutputDir 'tmp\alia-real-smoke\<run-name>'
```
