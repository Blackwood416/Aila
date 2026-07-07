# 2026-07-05 Handoff: Alia Host Integration Agent

> Current status note, 2026-07-07: use
> `docs/alia-engine/2026-07-07-handoff-current-status.md` for current runtime
> defaults. The active foreground LoRA is checkpoint-750. This host-integration
> handoff still describes the intended FSM/API binding flow.

This handoff is for an agent connecting the custom Alia engine to the C# host
program. The engine ABI is now ready for a first real host integration pass, but
the host skeleton still reflects an older flow.

## Repositories and status

Engine worktree:

- Path: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine`
- Branch: `codex/alia-custom-engine`
- Latest known engine commit at original handoff: `6cef4c1 feat: expose Alia host integration API`
- Latest known performance commit before the July 7 docs update: `70f3f5f perf: scope Q35 snapshot suppression to final prefill`
- API reference: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine/docs/API/Alia_C_API.md`

Host repository:

- Path: `E:/RiderProjects/Alia`
- Design docs: `E:/RiderProjects/Alia/docs/design/`
- Important files:
  - `E:/RiderProjects/Alia/src/Alia.Host/Core/Native/AliaNativeMethods.cs`
  - `E:/RiderProjects/Alia/src/Alia.Host/Core/Native/AliaContextSafeHandle.cs`
  - `E:/RiderProjects/Alia/src/Alia.Host/Program.cs`

At the time of this handoff, the host repository had unrelated untracked files:

```text
?? .claude/
?? docs/design/speculative_ttfa_pipeline.md
```

Do not delete, overwrite, or stage these unless the user explicitly asks.

## Engine model/config contract

Host initialization should use `alia_context_init_ex`, not legacy
`alia_context_init`.

Use:

- ASR: `models/Qwen3-ASR-1.7B-BNB-NF4`
- Foreground VLM: `models/qwen3.5-4B-bnb-nf4-offline-visiondense`
- Foreground LoRA: `F:/unsloth/qwen35_4b_alia_identity_r16_lr1e5/checkpoint-750`
- Background VLM: `models/qwen3.5-0.8B-bnb-nf4-offline`
- TTS: `models/Qwen3-TTS-12Hz-0.6B-Base`
- TTS reference audio: `E:/RiderProjects/Aila/.worktrees/alia-custom-engine/alia_ref.wav`
- `max_seq_len`: start with `2048`

The engine short smoke after the API change passed with:

```text
ALIA_REAL_MODEL_SMOKE_PASS
foreground_lora_applied=true
foreground_state=Completed
simulated_vad_to_first_audio_ms=1023
tts_reference_audio_path="E:\RiderProjects\Aila\.worktrees\alia-custom-engine\alia_ref.wav"
background_state=Completed
```

## New C ABI surface to bind

The engine header is:

```text
E:/RiderProjects/Aila/.worktrees/alia-custom-engine/include/alia_api.h
```

Bind at least these in C#:

- `AliaErrorCode`
- `AliaPipelineMask`
- `AliaAsyncState`
- `AliaGenConfig`
- `AliaContextConfig`
- `alia_context_init_ex`
- `alia_context_destroy`
- `alia_free_string`
- `alia_abort_inference`
- `alia_get_last_error`
- `alia_asr_feed_audio`
- `alia_asr_reset`
- `alia_asr_get_text`
- `alia_asr_get_partial_text`
- `alia_vlm_prefill_asr_text`
- `alia_start_conversation_turn`
- `alia_start_speculative_conversation_turn`
- `alia_commit_speculative_conversation_turn`
- `alia_foreground_get_state`
- `alia_foreground_wait`
- `alia_foreground_get_last_result`
- `alia_register_background_callback_ex`
- `alia_trigger_background_processing`
- `alia_background_get_state`
- `alia_background_wait`
- `alia_background_get_last_result`

Keep callback delegates strongly referenced for the full context lifetime:

- `AliaAudioCallback`
- `AliaBackgroundResultCallback`
- `AliaToolCallCallback` can remain unbound or null for the first integration
  unless the user explicitly starts tool-call work.

Native strings returned through `char**` must be freed with `alia_free_string`.
Do not free them with C# allocators. For NativeAOT, prefer `IntPtr` returns and
manual UTF-8 conversion/freeing over fragile automatic marshalling for returned
owned strings.

## Host flow that must replace the old skeleton

The current `Program.cs` starts a conversation turn on `VAD_Rise` and jumps
directly to `AiSpeaking`. That is wrong for the current engine.

Use this flow:

1. `Idle`

   Host VAD listens locally. Do not feed noise into ASR. Background memory may
   run after previous turns.

2. `VAD_Rise` from `Idle` -> `UserListening`

   Reset or prepare ASR for a new user turn if needed. Start feeding 16 kHz mono
   float PCM chunks to `alia_asr_feed_audio`.

3. `UserListening`

   On a timer, not inside the hot audio callback:

   - call `alia_asr_get_partial_text`;
   - display stable/partial text if useful;
   - call `alia_vlm_prefill_asr_text(stable, partial)`;
   - free returned strings.

   Keep this polling cadence conservative at first. A 100-250 ms host timer is a
   reasonable integration starting point; tune later with real latency metrics.

4. `VAD_Fall` -> `AiThinking`

   - call `alia_asr_get_text` to force final ASR text;
   - optionally call `alia_vlm_prefill_asr_text` one last time;
   - start foreground with `alia_start_conversation_turn`, or use
     `alia_commit_speculative_conversation_turn` only if a speculative path was
     explicitly started;
   - free ASR strings.

5. `AiThinking` / `AiSpeaking`

   - audio callback receives 24 kHz mono float samples and enqueues copies into
     the host playback ring buffer;
   - transition to `AiSpeaking` on the first audio callback, not merely when the
     foreground worker starts;
   - poll `alia_foreground_wait(ctx, 20-50, &state)` from a control task, not
     from the audio callback.

6. Foreground completed

   - call `alia_foreground_get_last_result`;
   - use `out_user_text` and `out_assistant_text` to build
     `User: ...\nAssistant: ...`;
   - send `out_action_tags_json` to avatar/action routing if needed;
   - trigger background extraction with `alia_trigger_background_processing`;
   - free all returned strings;
   - return to `Idle` when playback has drained or the foreground turn is done,
     depending on host UX choice.

7. Interruption

   On `VAD_Rise` during `AiThinking` or `AiSpeaking`:

   - call `alia_abort_inference(ctx, ALIA_PIPELINE_VLM_FOREGROUND | ALIA_PIPELINE_TTS)`;
   - stop playback and clear audio buffers;
   - discard pending avatar/audio events;
   - transition to `UserListening`, not `Idle`;
   - start feeding new speech audio to ASR.

   `ALIA_PIPELINE_ASR` is recorded in the abort mask but currently has no
   dedicated engine abort path, so host-side ASR reset/turn management still
   matters.

## Minimal C# binding shape

Use this as a shape guide, not as final code:

```csharp
[StructLayout(LayoutKind.Sequential)]
internal struct AliaContextConfig
{
    public IntPtr AsrModelDir;
    public IntPtr Vlm4BModelDir;
    public IntPtr Vlm4BLoraDir;
    public IntPtr Vlm08BModelDir;
    public IntPtr TtsModelDir;
    public IntPtr TtsReferenceAudioPath;
    public int MaxSeqLen;
}

[LibraryImport(DllName, EntryPoint = "alia_context_init_ex")]
internal static partial int alia_context_init_ex(
    out IntPtr ctx,
    in AliaContextConfig config);

[LibraryImport(DllName, EntryPoint = "alia_foreground_wait")]
internal static partial int alia_foreground_wait(
    AliaContextSafeHandle ctx,
    int timeoutMs,
    out int state);

[LibraryImport(DllName, EntryPoint = "alia_foreground_get_last_result")]
internal static partial int alia_foreground_get_last_result(
    AliaContextSafeHandle ctx,
    out IntPtr userText,
    out IntPtr assistantText,
    out IntPtr actionTagsJson);
```

Because `AliaContextConfig` contains pointers, the wrapper should allocate UTF-8
native strings, call init, then free the temporary path strings after init
returns. Alternatively use source-generated custom marshalling if already
present in the host codebase, but keep ownership explicit.

## Memory and threading rules

- Do not block the audio capture callback with ASR decode or VLM prefill. Feed
  audio quickly and schedule text polling on a control loop.
- Do not call foreground start/prefill/rollback concurrently from multiple host
  threads without a host-side lock.
- Audio callback sample pointers are valid only during the callback. Copy the
  samples before returning.
- Background callback JSON pointer is valid only during the callback. Copy it.
- Callback delegates and any `GCHandle` used as `user_data` must remain alive
  until the relevant worker is done or the context is destroyed.
- On shutdown, abort all pipelines, stop audio, then dispose the SafeHandle.

## First integration milestone

Aim for a boring, observable vertical slice:

1. Host starts and initializes engine with `alia_context_init_ex`.
2. VAD_Rise moves to `UserListening`.
3. Host feeds microphone audio to ASR.
4. Host polls partial/stable text and calls foreground prefill.
5. VAD_Fall starts foreground turn.
6. TTS audio callback plays through the host ring buffer.
7. Foreground completion returns user/assistant text.
8. Background memory extraction callback receives JSON.
9. VAD_Rise during thinking/speaking aborts and returns to `UserListening`.

Avoid polishing avatar actions, tools, or visual input until this path is stable.

## Verification plan

Engine-side sanity before host work:

```powershell
cd E:\RiderProjects\Aila\.worktrees\alia-custom-engine
. .\perf\PerfCommon.ps1
Initialize-AilaOneApiEnvironment
cmake --build build --target AliaEngine --config Release
.\tools\alia\RunAliaTargetPipeline.ps1 -SkipBuild -NoGenerateAudio -SkipToolProbe `
  -AudioPath 'tmp\alia-real-smoke\voice_matrix\short_hello_request.wav' `
  -OutputWav 'tmp\alia-real-smoke\host_integration_guard_short.wav' `
  -LogPath 'tmp\alia-real-smoke\host_integration_guard_short.log' `
  -RequestText '请用一句很短的中文和我打个招呼。' `
  -MaxTokens 32 -TimeoutSec 1500
```

Host-side checks:

```powershell
cd E:\RiderProjects\Alia
dotnet build
```

Then run the host manually with the engine DLL discoverable. If the host copies
native binaries from the engine build, document exactly where `AilaShared.dll`
and dependent runtime DLLs are loaded from. If it relies on `PATH`, keep that in
the run script rather than requiring permanent user environment edits.

For generated audio quality checks, use:

```powershell
powershell -ExecutionPolicy Bypass -File "E:\RiderProjects\Mimo-ASR\mimo-asr.ps1" -AudioFile "audio.wav"
```

If `MIMO_API_KEY` is absent:

```powershell
$env:MIMO_API_KEY = [Environment]::GetEnvironmentVariable("MIMO_API_KEY", "Machine")
```

## Known host/API gaps to defer

- Computer Use and visual input are intentionally not in scope.
- Tool calls can remain null/TODO for first voice integration.
- Speculative silence window should wait until real VAD timing is wired.
- Engine-side `ALIA_PIPELINE_ASR` abort is not a full ASR cancellation path yet.
- The host memory schema may not match the current background JSON exactly;
  adapt conservatively after seeing real callback payloads.

## Copy-paste prompt for a new host integration agent

```text
You are Codex integrating the Alia custom engine into the C# host. Read
E:/RiderProjects/Aila/.worktrees/alia-custom-engine/docs/alia-engine/2026-07-05-handoff-host-integration.md
and
E:/RiderProjects/Aila/.worktrees/alia-custom-engine/docs/API/Alia_C_API.md
first.

Engine worktree: E:/RiderProjects/Aila/.worktrees/alia-custom-engine,
branch codex/alia-custom-engine, latest engine commit 6cef4c1.
Host repo: E:/RiderProjects/Alia.

Goal: update the host P/Invoke bindings and FSM so VAD_Rise enters
UserListening, microphone audio feeds ASR, partial/stable ASR text prefills the
foreground VLM, VAD_Fall starts/commits the foreground turn, TTS audio plays via
the host ring buffer, foreground completion triggers background memory
extraction, and interruptions abort foreground/TTS then return to UserListening.
Do not pursue Computer Use, visual input, or tool-call implementation. Preserve
unrelated/untracked host files. Validate with build plus a real engine smoke
where relevant, and commit each useful round.
```
