# 2026-07-05 Handoff: Alia Engine Optimization Agent

This handoff is for an agent continuing performance work on the Alia custom
engine branch. Keep the branch specialized for the real voice-text-voice
pipeline. Do not drift back into the generic Aila mainline path.

## Workspace

- Engine worktree: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine`
- Branch: `codex/alia-custom-engine`
- Latest known commit at handoff: `6cef4c1 feat: expose Alia host integration API`
- Main API reference: `docs/API/Alia_C_API.md`
- Host design docs: `E:/RiderProjects/Alia/docs/design/`

The root repository path `E:/RiderProjects/Aila` may also exist, but work on the
custom-engine worktree above unless the user explicitly redirects you.

## Current model contract

Use these exact models unless the user says otherwise:

- ASR: `models/Qwen3-ASR-1.7B-BNB-NF4`
- Foreground VLM: `models/qwen3.5-4B-bnb-nf4-offline-visiondense`
- Foreground LoRA: `F:/unsloth/qwen35_4b_alia_identity_r16_lr1e5/checkpoint-500`
- Background VLM: `models/qwen3.5-0.8B-bnb-nf4-offline`
- TTS: `models/Qwen3-TTS-12Hz-0.6B-Base`
- TTS reference audio: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine/alia_ref.wav`

Smoke and matrix prompts should be Chinese. The user can continue tuning LoRA
quality, so do not block performance work on style drift unless the output is
clearly broken or the audio is corrupted.

## Non-goals and guardrails

- Do not pursue Computer Use, visual input, or tool-call features in this pass.
  Tool-call work should remain a TODO unless the user explicitly changes scope.
- Do not write API unit tests as a substitute for real model verification.
- Use real smoke/matrix runs to validate meaningful changes.
- Use `apply_patch` for manual file edits.
- Do not revert unrelated changes in the worktree.
- Commit each useful round so the user can roll back.
- Keep docs under `docs/alia-engine/` or `docs/API/`.

Important previous conclusions:

- TTS first-audio size has no hard fixed requirement anymore, but streaming
  chunks must not create later gaps. Avoid strategies where a tiny first chunk
  is followed by a larger/slower second chunk that playback cannot cover.
- Device-side sampling was tried and rejected because it changed sampling text
  and chunk shape enough to destabilize matrix TTFA. Revisit only with a strict
  deterministic comparison or a TTS-specific sampling contract.
- Qwen3-TTS has nested transformer structure; some generation variance is
  expected. Focus on whether the module/pipeline is near the practical limit.
- Qwen3-ASR text decoder profiling showed decode is distributed across
  attention, qkv/o/ffn/down, and norms; attention/joint_matrix alone was not a
  clear first target on the short smoke. ASR prefix reuse previously regressed
  because the 24-token append prefill shape was not warmed. Backend warmup now
  includes 24 and 32 tokens. In the 2026-07-06 short smoke,
  `AILA_ASR_PREFIX_REUSE=1` hit reuse of 48 tokens and appended 24 tokens; the
  final 24-token prefill profile had `o_proj=17.932ms` instead of the earlier
  100ms+ shape-init spike. Non-profile tail was 317 ms with prefix reuse versus
  335 ms without prefix reuse on the same build. Keep prefix reuse opt-in until
  a broader ASR matrix confirms quality and latency across lengths.

## Latest evidence snapshot

After `6cef4c1`, a short real smoke passed:

```text
ALIA_REAL_MODEL_SMOKE_PASS
foreground_lora_applied=true
foreground_state=Completed
foreground_first_content_delta_ms=338
foreground_first_tts_enqueue_ms=451
tts_first_audio_ms=666
simulated_vad_to_first_audio_ms=1023
tts_reference_audio_path="E:\RiderProjects\Aila\.worktrees\alia-custom-engine\alia_ref.wav"
background_state=Completed
```

The most recent full matrix with stream ASR prefill and output ASR verification
passed 5/5 at:

```text
tmp/alia-real-smoke/voice_matrix_foreground_token_lifetime_fix/summary.csv
```

Key matrix values:

```text
scenario              vad_to_first_audio  first_content  first_tts_enqueue  tts_first_audio
short_hello           973 ms              704 ms         766 ms             609 ms
persona_chat          1061 ms             748 ms         858 ms             651 ms
preference_memory     1049 ms             733 ms         845 ms             652 ms
task_memory           1157 ms             714 ms         951 ms             784 ms
multi_turn_followup   1118 ms             725 ms         912 ms             730 ms
```

The target remains aggressive: push short-prompt TTFA below 1s and keep moving
toward lower latency through pipeline overlap and scheduling.

## Current default behavior to preserve

- TTS reference voice loads from the worktree-local `alia_ref.wav`.
- Foreground LoRA checkpoint-500 is the preferred adapter.
- Matrix scenarios are Chinese and include `multi_turn_followup` instead of
  `long_answer`.
- Foreground action tags in parentheses are filtered out of spoken text and
  exposed separately.
- The default TTS stream shape currently emits 4-frame audio callbacks with a
  playback-aware steady batch path. If you change chunking, check playback gaps,
  not just first-audio latency.

## Most promising optimization directions

1. Profile the current build before cutting code.

   Start by confirming whether the biggest current cost is final ASR tail,
   foreground prompt prefill, VLM-to-TTS enqueue timing, or first TTS audio.
   The latest matrix still shows final `vad_to_first_audio` around 1.0-1.15s,
   with `foreground_profile_prompt_prefill_ms` around 320 ms and first TTS audio
   roughly 600-780 ms after foreground start.

2. Improve ASR partial/prefill usefulness during UserListening.

   The simulated stream currently uses 1000 ms chunks/intervals in the matrix.
   Investigate whether smaller ASR chunks or smarter throttling can increase
   useful prefill before final commit without spending so much GPU time that it
   steals from foreground/TTS. Track:

   - `asr_stream_get_text_total_ms`
   - `asr_stream_vlm_prefill_total_ms`
   - `foreground_asr_prefill_tokens`
   - `foreground_asr_prefill_reused_tokens`
   - `foreground_profile_final_cached_prefix_reject_reason`

   Recent matrix rows often show cached prefix rejected because the cached suffix
   exceeds the fast threshold. That is a high-value place to inspect.

3. Revisit final ASR tail and prefix reuse.

   Prior work added ASR prefix/mel/audio encoder reuse. Before changing it,
   profile real `alia_asr_get_text` and `alia_asr_get_partial_text` timing.
   CPU/GPU mel/STFT work was previously identified as useful for both ASR and
   TTS; treat it as a larger follow-up unless current profiling points directly
   there.

4. Tighten pipeline scheduling instead of only backend kernels.

   The TTS backend first-audio inner loop is much better than earlier passes.
   If backend RTF is close to practical limit, shift focus to:

   - overlap ASR partial decode, VLM prefill, and host VAD silence window;
   - avoid foreground decode holding the lane while TTS first audio is due;
   - tune scheduler decisions around final commit, cached prefill, and TTS
     first-audio priority;
   - reduce lock hold/wait spikes shown in smoke metrics.

5. Keep speculative silence window as a TODO until real host VAD integration.

   The user has a design where partial text flush starts a timer shorter than
   the VAD silence window, commits ASR early, and rolls back KV if audio resumes.
   Do not implement it in engine-only smoke yet unless the host VAD contract is
   available. Document it as a future host-integrated scheduling feature.

## Useful files

- `src/alia/AliaTurnScheduler.*`
- `src/alia/AliaAsrPipeline.*`
- `src/alia/AliaForegroundPipeline.*`
- `src/alia/AliaTtsPipeline.*`
- `src/models/Qwen3TTSBackend.*`
- `src/models/Qwen3ASRBackend.*` or ASR backend equivalents found by `rg`
- `tools/alia/AliaRealModelSmoke.cpp`
- `tools/alia/RunAliaTargetPipeline.ps1`
- `tools/alia/RunAliaVoiceScenarioMatrix.ps1`

## Standard verification commands

Build:

```powershell
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --target AliaEngine --config Release
```

Short real smoke:

```powershell
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\<run-name>_short.wav' `
  -LogPath 'tmp\alia-real-smoke\<run-name>_short.log' `
  -RequestText '请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Full matrix:

```powershell
.\tools\alia\RunAliaVoiceScenarioMatrix.ps1 -SkipBuild -TimeoutSec 1500 `
  -StreamAsrPrefill -StreamChunkMs 1000 -StreamPrefillIntervalMs 1000 `
  -VerifyOutputAsr `
  -OutputDir 'tmp\alia-real-smoke\<run-name>'
```

Output ASR verification script:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\RiderProjects\Mimo-ASR\mimo-asr.ps1" -AudioFile "audio.wav"
```

If `MIMO_API_KEY` is missing in the shell, load it from Machine environment:

```powershell
$env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable("MIMO_API_KEY", "Machine")
```

Always run:

```powershell
git diff --check
```

## Suggested first task for the next optimization agent

Start with a profiling-only pass on the current build:

1. Run one short smoke with stream ASR prefill enabled and one without.
2. Compare ASR partial/final costs, foreground prefill reuse/reject reason, first
   TTS enqueue, and first audio.
3. Pick the largest remaining serialized segment. If the cached suffix reject is
   dominant, inspect the foreground prefill reuse threshold/fast path before
   touching TTS kernels.
4. Make one scoped change, build, run a short real smoke, then run the matrix if
   the change affects scheduling or generation behavior.

## Copy-paste prompt for a new optimization agent

```text
You are Codex continuing Alia custom-engine optimization in
E:/RiderProjects/Aila/.worktrees/alia-custom-engine on branch
codex/alia-custom-engine. Read
docs/alia-engine/2026-07-05-handoff-optimization.md first and follow it.

Goal: reduce real voice-text-voice TTFA, especially short Chinese prompts,
without pursuing Computer Use, visual input, or tool-call features. Use the
fixed model set and checkpoint-500 LoRA. Validate with real model smoke/matrix,
not API-only tests. Start by profiling the current build and identify the
largest serialized latency segment before editing code. Commit each useful
round.
```
