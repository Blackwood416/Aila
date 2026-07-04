# 2026-07-04 LoRA-500 and Chinese Smoke Prompts

Current foreground LoRA default:

```text
F:/unsloth/qwen35_4b_alia_identity_r16_lr1e5/checkpoint-500
```

This checkpoint was trained on cleaned data without action tags. Prefer it over
`checkpoint-1400` for Alia voice-pipeline validation. If real smoke or matrix
runs show regressions, report the issue and include the generated foreground
text, `foreground_action_tag_count`, `foreground_profile_first_spoken_delay_ms`,
and output ASR transcript.

The default smoke prompt and voice matrix scenarios now use Chinese input
because the current training data is Chinese-heavy:

```text
short_hello: 艾莉亚，请用一句话打个招呼。
persona_chat: 艾莉亚，我今天有点累，请温柔地简短安慰我。
preference_memory: 艾莉亚，请记住我晚上喜欢简短的中文回复。
task_memory: 艾莉亚，下班后提醒我伸展肩膀。
multi_turn_followup: 刚才我说今晚想早点休息。艾莉亚，请用一句短句接着提醒我。
```

Historical docs may still mention `checkpoint-1400` and English request text as
past baselines. Do not treat those as the current default.

## Initial Real-Model Validation

Commands:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -SkipToolProbe `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -AudioPath 'tmp\alia-real-smoke\lora500_chinese_default_utf8\short_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\lora500_chinese_default_utf8\short_output.wav' `
  -LogPath 'tmp\alia-real-smoke\lora500_chinese_default_utf8\short.log' `
  -MaxTokens 48 -TimeoutSec 1500

.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -VerifyOutputAsr `
  -OutputDir 'tmp\alia-real-smoke\voice_matrix_lora500_chinese'
```

The first smoke initially failed because Windows narrow `argv` corrupted the
Chinese `--request-text` passed from PowerShell. `AliaRealModelSmoke` now reads
the Windows command line as UTF-16 and converts arguments to UTF-8 before
parsing. The rerun confirmed:

```text
foreground_lora="F:\unsloth\qwen35_4b_alia_identity_r16_lr1e5\checkpoint-500"
request_text="艾莉亚，请用一句话打个招呼。"
asr_partial_text="艾莉亚，请用一句话打个招呼。"
ALIA_REAL_MODEL_SMOKE_PASS
```

Matrix summary:

```text
path: tmp/alia-real-smoke/voice_matrix_lora500_chinese/summary.csv
pass: 4/5

scenario           TTFA  first_spoken_delay  action_tags  fg_tokens  tts_jobs  tts_backend_ms  buffer_gap_ms
short_hello        989   0                   0            12         3         3151.7          0
persona_chat       1065  0                   0            20         3         4247.59         0
preference_memory  1017  0                   0            17         3         4372.11         72
task_memory        1191  0                   0            53         4         12794.1         163
long_answer        1060  0                   0            96         63        169777          128
```

Good signal:

- The cleaned LoRA removed action tags on this matrix:
  `foreground_action_tag_count=0` and
  `foreground_profile_first_spoken_delay_ms=0` for all five scenarios.
- Short Chinese prompts now keep foreground prompt prefill in the expected
  330-345 ms range and TTFA around 1.0-1.2 s.

Issues to report for `checkpoint-500`:

- Role/self-description drift remains. Some replies still start with
  `父亲大人` and mention being a virtual AI.
- `task_memory` ignored the reminder request and drifted into repeated
  companionship/game text:
  `艾莉亚会陪主人玩游戏的...艾莉亚只是虚拟的AI...`
- `long_answer` failed the matrix by repeating
  `艾利亚……` until `MaxTokens=96`, creating 63 TTS synthesis jobs and about
  170 s of backend TTS work.

The long-answer scenario is now retired from the default voice matrix. Typical
Alia voice turns should be short, so the matrix now uses `multi_turn_followup`
instead: a short Chinese follow-up request that embeds prior context in the
current turn. This keeps the matrix focused on TTFA, short-answer quality, and
context continuity without letting long-text repetition dominate TTS runtime.
Scenario `MaxTokens` values are also capped for short voice turns: 32 for the
greeting scenario and 48 for the remaining matrix scenarios.

For future LoRA checks, keep the short Chinese matrix because it separates the
solved action-tag problem from the remaining repetition/role-drift problem.

## Short Multi-Turn Matrix Update

`long_answer` was replaced with `multi_turn_followup` on 2026-07-04 because
long-form answers are not representative for the current Alia voice assistant
loop. Validation commands used the same real-model matrix with output ASR:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -VerifyOutputAsr `
  -OutputDir 'tmp\alia-real-smoke\voice_matrix_lora500_chinese_multiturn'
```

Result from the first replacement run, before the matrix `MaxTokens` caps were
tightened: failed, 2/5 pass. `persona_chat` and `preference_memory` completed.
`short_hello` generated a repeated role/self-description response, hit the
48-token cap, produced about 23.2 s of TTS backend work, then exited with
`-1073741819`. `task_memory` and `multi_turn_followup` exited with the same
code during model/warmup initialization before useful pipeline metrics were
emitted. After tightening the scenario token caps, a diagnostic single-run of
`multi_turn_followup` still failed in the same initialization region; repeating
the diagnostic with `AILA_TTS_REF_AUDIO=0` also failed there, so the crash is
not explained by `alia_ref.wav` reference embedding alone.

Keep the scenario replacement, but treat the latest validation as exposing two
separate follow-up issues:

- `checkpoint-500` can still drift or repeat on short Chinese prompts, even
  though action tags are gone.
- Repeated real-model smoke processes can hit an access violation around the
  post-Mimi-warmup foreground buffer initialization path. Investigate this
  before relying on full matrix results after a runaway TTS turn.
