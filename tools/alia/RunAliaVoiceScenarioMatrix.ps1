param(
    [string]$BuildDir = "build",
    [string]$OutputDir = "tmp\alia-real-smoke\voice_matrix",
    [string]$ModelRoot = "",
    [string]$ForegroundLora = "F:\unsloth\qwen35_4b_alia_identity_r16_lr5e4\checkpoint-350",
    [int]$TimeoutSec = 1500,
    [int]$StreamChunkMs = 1000,
    [int]$StreamPrefillIntervalMs = 0,
    [switch]$SkipBuild,
    [switch]$StreamAsrPrefill,
    [switch]$IncludeToolProbe,
    [switch]$VerifyOutputAsr,
    [string]$OutputAsrScriptPath = "E:\RiderProjects\Mimo-ASR\mimo-asr.ps1"
)

$ErrorActionPreference = "Stop"

function Convert-KeyValueLog {
    param([string]$LogPath)

    $values = @{}
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $values
    }

    foreach ($line in Get-Content -LiteralPath $LogPath) {
        if ($line -match '^([A-Za-z0-9_]+)=(.*)$') {
            $key = $matches[1]
            $value = $matches[2].Trim()
            if ($value.Length -ge 2 -and $value[0] -eq '"' -and $value[$value.Length - 1] -eq '"') {
                $value = $value.Substring(1, $value.Length - 2)
                $value = $value.Replace('\"', '"').Replace('\\', '\').Replace('\n', "`n").Replace('\r', "`r").Replace('\t', "`t")
            }
            $values[$key] = $value
        }
    }
    return $values
}

function Get-ValueOrEmpty {
    param(
        [hashtable]$Values,
        [string]$Key
    )

    if ($Values.ContainsKey($Key)) {
        return $Values[$Key]
    }
    return ""
}

function Invoke-OutputAsr {
    param(
        [string]$ScriptPath,
        [string]$AudioPath
    )

    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "Output ASR script not found: $ScriptPath"
    }
    if (-not (Test-Path -LiteralPath $AudioPath)) {
        throw "Output audio not found: $AudioPath"
    }

    if (-not $env:MIMO_API_KEY) {
        $machineApiKey = [Environment]::GetEnvironmentVariable("MIMO_API_KEY", "Machine")
        if ($machineApiKey) {
            $env:MIMO_API_KEY = $machineApiKey
        }
    }

    $output = & powershell -ExecutionPolicy Bypass -File $ScriptPath -AudioFile $AudioPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw (($output | Out-String).Trim())
    }
    return (($output | Out-String).Trim())
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$outputRoot = Join-Path $repoRoot $OutputDir
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

Push-Location $repoRoot
try {
    . .\perf\PerfCommon.ps1
    Initialize-AilaOneApiEnvironment

    if (-not $SkipBuild) {
        cmake -S . -B $BuildDir
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configure failed with exit code $LASTEXITCODE"
        }
        cmake --build $BuildDir --target AliaEngine --config Release
        if ($LASTEXITCODE -ne 0) {
            throw "AliaEngine build failed with exit code $LASTEXITCODE"
        }
    }

    $scenarios = @(
        [pscustomobject]@{
            Name = "short_hello"
            RequestText = "艾莉亚，请用一句话打个招呼。"
            MaxTokens = 32
        },
        [pscustomobject]@{
            Name = "persona_chat"
            RequestText = "艾莉亚，我今天有点累，请温柔地简短安慰我。"
            MaxTokens = 48
        },
        [pscustomobject]@{
            Name = "preference_memory"
            RequestText = "艾莉亚，请记住我晚上喜欢简短的中文回复。"
            MaxTokens = 48
        },
        [pscustomobject]@{
            Name = "task_memory"
            RequestText = "艾莉亚，下班后提醒我伸展肩膀。"
            MaxTokens = 48
        },
        [pscustomobject]@{
            Name = "multi_turn_followup"
            RequestText = "刚才我说今晚想早点休息。艾莉亚，请用一句短句接着提醒我。"
            MaxTokens = 48
        }
    )

    $rows = @()
    foreach ($scenario in $scenarios) {
        $audioPath = Join-Path $OutputDir ($scenario.Name + "_request.wav")
        $wavPath = Join-Path $OutputDir ($scenario.Name + "_output.wav")
        $logPath = Join-Path $OutputDir ($scenario.Name + ".log")
        $absoluteAudioPath = Join-Path $repoRoot $audioPath
        if (Test-Path -LiteralPath $absoluteAudioPath) {
            Remove-Item -LiteralPath $absoluteAudioPath -Force
        }

        $passed = $false
        $errorText = ""
        try {
            $targetArgs = @{
                BuildDir = $BuildDir
                SkipBuild = $true
                AudioPath = $audioPath
                OutputWav = $wavPath
                LogPath = $logPath
                ModelRoot = $ModelRoot
                ForegroundLora = $ForegroundLora
                RequestText = $scenario.RequestText
                MaxTokens = $scenario.MaxTokens
                TimeoutSec = $TimeoutSec
                StreamChunkMs = $StreamChunkMs
                StreamPrefillIntervalMs = $StreamPrefillIntervalMs
            }
            if ($StreamAsrPrefill) {
                $targetArgs.StreamAsrPrefill = $true
            }
            if (-not $IncludeToolProbe) {
                $targetArgs.SkipToolProbe = $true
            }
            .\tools\alia\RunAliaTargetPipeline.ps1 @targetArgs
            $passed = $true
        } catch {
            $errorText = $_.Exception.Message
        }

        $absoluteLogPath = Join-Path $repoRoot $logPath
        $absoluteWavPath = Join-Path $repoRoot $wavPath
        $values = Convert-KeyValueLog -LogPath $absoluteLogPath
        $logText = ""
        if (Test-Path -LiteralPath $absoluteLogPath) {
            $logText = Get-Content -Raw -LiteralPath $absoluteLogPath
        }
        $passed = $passed -and $logText.Contains("ALIA_REAL_MODEL_SMOKE_PASS")

        $outputAsrText = ""
        $outputAsrError = ""
        if ($VerifyOutputAsr) {
            try {
                $outputAsrText = Invoke-OutputAsr -ScriptPath $OutputAsrScriptPath -AudioPath $absoluteWavPath
                Write-Host ("output_asr_text[{0}]={1}" -f $scenario.Name, $outputAsrText)
            } catch {
                $outputAsrError = $_.Exception.Message
                $passed = $false
            }
            if (-not $outputAsrText) {
                $outputAsrError = "Output ASR returned empty transcript."
                $passed = $false
            }
        }

        $rows += [pscustomobject]@{
            scenario = $scenario.Name
            pass = $passed
            error = $errorText
            request_text = $scenario.RequestText
            model_load_ms = Get-ValueOrEmpty $values "model_load_ms"
            asr_ms = Get-ValueOrEmpty $values "asr_ms"
            asr_stream_chunk_ms = Get-ValueOrEmpty $values "asr_stream_chunk_ms"
            asr_stream_prefill_interval_ms = Get-ValueOrEmpty $values "asr_stream_prefill_interval_ms"
            asr_stream_text_calls = Get-ValueOrEmpty $values "asr_stream_text_calls"
            asr_stream_prefill_calls = Get-ValueOrEmpty $values "asr_stream_prefill_calls"
            asr_stream_prefill_skipped_unchanged = Get-ValueOrEmpty $values "asr_stream_prefill_skipped_unchanged"
            asr_stream_get_text_total_ms = Get-ValueOrEmpty $values "asr_stream_get_text_total_ms"
            asr_stream_get_text_max_ms = Get-ValueOrEmpty $values "asr_stream_get_text_max_ms"
            asr_stream_vlm_prefill_total_ms = Get-ValueOrEmpty $values "asr_stream_vlm_prefill_total_ms"
            asr_stream_vlm_prefill_max_ms = Get-ValueOrEmpty $values "asr_stream_vlm_prefill_max_ms"
            asr_stream_tick_total_ms = Get-ValueOrEmpty $values "asr_stream_tick_total_ms"
            asr_stream_tick_max_ms = Get-ValueOrEmpty $values "asr_stream_tick_max_ms"
            scheduler_asr_decode_allowed = Get-ValueOrEmpty $values "scheduler_asr_decode_allowed"
            scheduler_asr_decode_skipped = Get-ValueOrEmpty $values "scheduler_asr_decode_skipped"
            scheduler_asr_decode_hidden_budget_skipped = Get-ValueOrEmpty $values "scheduler_asr_decode_hidden_budget_skipped"
            scheduler_last_asr_decode_phase = Get-ValueOrEmpty $values "scheduler_last_asr_decode_phase"
            scheduler_last_asr_decode_action = Get-ValueOrEmpty $values "scheduler_last_asr_decode_action"
            scheduler_last_asr_decode_lane = Get-ValueOrEmpty $values "scheduler_last_asr_decode_lane"
            scheduler_last_asr_decode_reason = Get-ValueOrEmpty $values "scheduler_last_asr_decode_reason"
            scheduler_prefill_allowed = Get-ValueOrEmpty $values "scheduler_prefill_allowed"
            scheduler_prefill_skipped = Get-ValueOrEmpty $values "scheduler_prefill_skipped"
            scheduler_hidden_budget_skipped = Get-ValueOrEmpty $values "scheduler_hidden_budget_skipped"
            scheduler_speculative_allowed = Get-ValueOrEmpty $values "scheduler_speculative_allowed"
            scheduler_last_phase = Get-ValueOrEmpty $values "scheduler_last_phase"
            scheduler_last_action = Get-ValueOrEmpty $values "scheduler_last_action"
            scheduler_last_lane = Get-ValueOrEmpty $values "scheduler_last_lane"
            scheduler_last_reason = Get-ValueOrEmpty $values "scheduler_last_reason"
            asr_partial_full_decode_count = Get-ValueOrEmpty $values "asr_partial_full_decode_count"
            asr_partial_tail_decode_count = Get-ValueOrEmpty $values "asr_partial_tail_decode_count"
            asr_partial_throttled_count = Get-ValueOrEmpty $values "asr_partial_throttled_count"
            asr_profile_transcribe_calls = Get-ValueOrEmpty $values "asr_profile_transcribe_calls"
            asr_profile_generated_tokens = Get-ValueOrEmpty $values "asr_profile_generated_tokens"
            asr_profile_mel_cache_hits = Get-ValueOrEmpty $values "asr_profile_mel_cache_hits"
            asr_profile_mel_cache_reused_frames = Get-ValueOrEmpty $values "asr_profile_mel_cache_reused_frames"
            asr_profile_mel_cache_computed_frames = Get-ValueOrEmpty $values "asr_profile_mel_cache_computed_frames"
            asr_profile_mel_cache_max_abs_diff = Get-ValueOrEmpty $values "asr_profile_mel_cache_max_abs_diff"
            asr_profile_input_audio_ms = Get-ValueOrEmpty $values "asr_profile_input_audio_ms"
            asr_profile_mel_ms = Get-ValueOrEmpty $values "asr_profile_mel_ms"
            asr_profile_mel_stft_ms = Get-ValueOrEmpty $values "asr_profile_mel_stft_ms"
            asr_profile_mel_norm_ms = Get-ValueOrEmpty $values "asr_profile_mel_norm_ms"
            asr_profile_upload_ms = Get-ValueOrEmpty $values "asr_profile_upload_ms"
            asr_profile_encoder_ms = Get-ValueOrEmpty $values "asr_profile_encoder_ms"
            asr_profile_encoder_conv_ms = Get-ValueOrEmpty $values "asr_profile_encoder_conv_ms"
            asr_profile_encoder_transformer_ms = Get-ValueOrEmpty $values "asr_profile_encoder_transformer_ms"
            asr_profile_encoder_proj_ms = Get-ValueOrEmpty $values "asr_profile_encoder_proj_ms"
            asr_profile_readback_ms = Get-ValueOrEmpty $values "asr_profile_readback_ms"
            asr_profile_prompt_ms = Get-ValueOrEmpty $values "asr_profile_prompt_ms"
            asr_profile_prefill_ms = Get-ValueOrEmpty $values "asr_profile_prefill_ms"
            asr_profile_decode_ms = Get-ValueOrEmpty $values "asr_profile_decode_ms"
            asr_profile_total_ms = Get-ValueOrEmpty $values "asr_profile_total_ms"
            foreground_ms = Get-ValueOrEmpty $values "foreground_ms"
            foreground_prompt_tokens = Get-ValueOrEmpty $values "foreground_prompt_tokens"
            foreground_generated_tokens = Get-ValueOrEmpty $values "foreground_generated_tokens"
            foreground_asr_prefill_tokens = Get-ValueOrEmpty $values "foreground_asr_prefill_tokens"
            foreground_asr_prefill_reused_tokens = Get-ValueOrEmpty $values "foreground_asr_prefill_reused_tokens"
            foreground_asr_prefill_suffix_tokens = Get-ValueOrEmpty $values "foreground_asr_prefill_suffix_tokens"
            foreground_asr_prefill_skipped_small_suffix = Get-ValueOrEmpty $values "foreground_asr_prefill_skipped_small_suffix"
            foreground_asr_prefill_candidate_tokens = Get-ValueOrEmpty $values "foreground_asr_prefill_candidate_tokens"
            foreground_asr_prefill_candidate_suffix_tokens = Get-ValueOrEmpty $values "foreground_asr_prefill_candidate_suffix_tokens"
            foreground_asr_prefill_skip_reason = Get-ValueOrEmpty $values "foreground_asr_prefill_skip_reason"
            foreground_asr_prefill_ms = Get-ValueOrEmpty $values "foreground_asr_prefill_ms"
            foreground_first_content_delta_ms = Get-ValueOrEmpty $values "foreground_first_content_delta_ms"
            foreground_first_tts_enqueue_ms = Get-ValueOrEmpty $values "foreground_first_tts_enqueue_ms"
            foreground_profile_prompt_tokens = Get-ValueOrEmpty $values "foreground_profile_prompt_tokens"
            foreground_profile_prefilled_prompt_tokens = Get-ValueOrEmpty $values "foreground_profile_prefilled_prompt_tokens"
            foreground_profile_prompt_suffix_tokens = Get-ValueOrEmpty $values "foreground_profile_prompt_suffix_tokens"
            foreground_profile_final_cached_prefix_rejected = Get-ValueOrEmpty $values "foreground_profile_final_cached_prefix_rejected"
            foreground_profile_final_cached_prefix_reject_reason = Get-ValueOrEmpty $values "foreground_profile_final_cached_prefix_reject_reason"
            foreground_final_prefix_path = Get-ValueOrEmpty $values "foreground_final_prefix_path"
            foreground_profile_final_prefix_path = Get-ValueOrEmpty $values "foreground_profile_final_prefix_path"
            foreground_profile_generated_tokens = Get-ValueOrEmpty $values "foreground_profile_generated_tokens"
            foreground_profile_prompt_build_ms = Get-ValueOrEmpty $values "foreground_profile_prompt_build_ms"
            foreground_profile_prompt_prefill_ms = Get-ValueOrEmpty $values "foreground_profile_prompt_prefill_ms"
            foreground_profile_first_token_delta_ms = Get-ValueOrEmpty $values "foreground_profile_first_token_delta_ms"
            foreground_profile_first_content_delta_ms = Get-ValueOrEmpty $values "foreground_profile_first_content_delta_ms"
            foreground_profile_first_spoken_delay_ms = Get-ValueOrEmpty $values "foreground_profile_first_spoken_delay_ms"
            foreground_profile_first_tts_enqueue_ms = Get-ValueOrEmpty $values "foreground_profile_first_tts_enqueue_ms"
            foreground_profile_tts_first_audio_priority_wait_ms = Get-ValueOrEmpty $values "foreground_profile_tts_first_audio_priority_wait_ms"
            foreground_profile_tts_first_audio_priority_wait_deferred_count = Get-ValueOrEmpty $values "foreground_profile_tts_first_audio_priority_wait_deferred_count"
            foreground_profile_tts_first_audio_priority_following_text_bytes = Get-ValueOrEmpty $values "foreground_profile_tts_first_audio_priority_following_text_bytes"
            foreground_profile_first_tts_chunk_reason = Get-ValueOrEmpty $values "foreground_profile_first_tts_chunk_reason"
            foreground_profile_first_tts_chunk_pending_chars_at_first_content = Get-ValueOrEmpty $values "foreground_profile_first_tts_chunk_pending_chars_at_first_content"
            foreground_profile_first_tts_chunk_pending_chars_at_enqueue = Get-ValueOrEmpty $values "foreground_profile_first_tts_chunk_pending_chars_at_enqueue"
            foreground_profile_first_tts_chunk_wait_tokens = Get-ValueOrEmpty $values "foreground_profile_first_tts_chunk_wait_tokens"
            foreground_profile_first_tts_chunk_wait_ms = Get-ValueOrEmpty $values "foreground_profile_first_tts_chunk_wait_ms"
            foreground_profile_decode_ms = Get-ValueOrEmpty $values "foreground_profile_decode_ms"
            foreground_profile_model_ms = Get-ValueOrEmpty $values "foreground_profile_model_ms"
            simulated_vad_asr_tail_ms = Get-ValueOrEmpty $values "simulated_vad_asr_tail_ms"
            simulated_vad_to_first_content_ms = Get-ValueOrEmpty $values "simulated_vad_to_first_content_ms"
            simulated_vad_to_first_tts_enqueue_ms = Get-ValueOrEmpty $values "simulated_vad_to_first_tts_enqueue_ms"
            tts_first_audio_ms = Get-ValueOrEmpty $values "tts_first_audio_ms"
            first_tts_enqueue_to_first_audio_ms = Get-ValueOrEmpty $values "first_tts_enqueue_to_first_audio_ms"
            simulated_vad_to_first_audio_ms = Get-ValueOrEmpty $values "simulated_vad_to_first_audio_ms"
            tts_callback_count = Get-ValueOrEmpty $values "tts_callback_count"
            tts_chunks_synthesized = Get-ValueOrEmpty $values "tts_chunks_synthesized"
            tts_reference_audio_enabled = Get-ValueOrEmpty $values "tts_reference_audio_enabled"
            tts_reference_embedding_dim = Get-ValueOrEmpty $values "tts_reference_embedding_dim"
            tts_reference_embedding_ms = Get-ValueOrEmpty $values "tts_reference_embedding_ms"
            tts_reference_audio_path = Get-ValueOrEmpty $values "tts_reference_audio_path"
            tts_reference_audio_error = Get-ValueOrEmpty $values "tts_reference_audio_error"
            tts_first_text_chars = Get-ValueOrEmpty $values "tts_first_text_chars"
            tts_first_text_tokens = Get-ValueOrEmpty $values "tts_first_text_tokens"
            tts_first_backend_frames = Get-ValueOrEmpty $values "tts_first_backend_frames"
            tts_first_backend_callbacks = Get-ValueOrEmpty $values "tts_first_backend_callbacks"
            tts_first_backend_audio_samples = Get-ValueOrEmpty $values "tts_first_backend_audio_samples"
            tts_backend_stream_batch_frames = Get-ValueOrEmpty $values "tts_backend_stream_batch_frames"
            tts_backend_initial_stream_batch_frames = Get-ValueOrEmpty $values "tts_backend_initial_stream_batch_frames"
            tts_backend_steady_stream_batch_frames = Get-ValueOrEmpty $values "tts_backend_steady_stream_batch_frames"
            tts_backend_steady_batch_callback_count = Get-ValueOrEmpty $values "tts_backend_steady_batch_callback_count"
            tts_backend_playback_aware_steady_batch = Get-ValueOrEmpty $values "tts_backend_playback_aware_steady_batch"
            tts_audio_callback_max_frames = Get-ValueOrEmpty $values "tts_audio_callback_max_frames"
            tts_pause_silence_segments = Get-ValueOrEmpty $values "tts_pause_silence_segments"
            tts_pause_silence_ms = Get-ValueOrEmpty $values "tts_pause_silence_ms"
            tts_first_backend_codes_ms = Get-ValueOrEmpty $values "tts_first_backend_codes_ms"
            tts_first_backend_mimi_init_ms = Get-ValueOrEmpty $values "tts_first_backend_mimi_init_ms"
            tts_first_backend_audio_ms = Get-ValueOrEmpty $values "tts_first_backend_audio_ms"
            tts_first_backend_total_ms = Get-ValueOrEmpty $values "tts_first_backend_total_ms"
            tts_backend_total_ms = Get-ValueOrEmpty $values "tts_backend_total_ms"
            tts_total_samples = Get-ValueOrEmpty $values "tts_total_samples"
            tts_playback_gap_count = Get-ValueOrEmpty $values "tts_playback_gap_count"
            tts_playback_max_gap_ms = Get-ValueOrEmpty $values "tts_playback_max_gap_ms"
            tts_playback_total_gap_ms = Get-ValueOrEmpty $values "tts_playback_total_gap_ms"
            tts_playback_gap_ms = Get-ValueOrEmpty $values "tts_playback_gap_ms"
            tts_playback_buffer_gap_count = Get-ValueOrEmpty $values "tts_playback_buffer_gap_count"
            tts_playback_buffer_max_gap_ms = Get-ValueOrEmpty $values "tts_playback_buffer_max_gap_ms"
            tts_playback_buffer_total_gap_ms = Get-ValueOrEmpty $values "tts_playback_buffer_total_gap_ms"
            tts_playback_buffer_gap_ms = Get-ValueOrEmpty $values "tts_playback_buffer_gap_ms"
            tts_callback_times_ms = Get-ValueOrEmpty $values "tts_callback_times_ms"
            tts_callback_intervals_ms = Get-ValueOrEmpty $values "tts_callback_intervals_ms"
            tts_chunk_audio_ms = Get-ValueOrEmpty $values "tts_chunk_audio_ms"
            background_ms = Get-ValueOrEmpty $values "background_ms"
            background_schema_valid = Get-ValueOrEmpty $values "background_schema_valid"
            tool_probe_enabled = Get-ValueOrEmpty $values "tool_probe"
            tool_probe_ms = Get-ValueOrEmpty $values "tool_probe_ms"
            foreground_lora_applied = Get-ValueOrEmpty $values "foreground_lora_applied"
            foreground_lora_pair_count = Get-ValueOrEmpty $values "foreground_lora_pair_count"
            asr_partial_text = Get-ValueOrEmpty $values "asr_partial_text"
            foreground_assistant_text = Get-ValueOrEmpty $values "foreground_assistant_text"
            foreground_action_tag_count = Get-ValueOrEmpty $values "foreground_action_tag_count"
            foreground_action_tags = Get-ValueOrEmpty $values "foreground_action_tags"
            output_asr_text = $outputAsrText
            output_asr_error = $outputAsrError
            background_result_json = Get-ValueOrEmpty $values "background_result_json"
            log_path = $logPath
            output_wav = $wavPath
        }
    }

    $summaryPath = Join-Path $outputRoot "summary.csv"
    $rows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
    $rows | Format-Table scenario, pass, asr_ms, asr_stream_text_calls, asr_partial_tail_decode_count, simulated_vad_asr_tail_ms, simulated_vad_to_first_content_ms, simulated_vad_to_first_audio_ms, foreground_first_content_delta_ms, tts_first_audio_ms, foreground_ms, background_ms -AutoSize

    $failed = @($rows | Where-Object { -not $_.pass })
    if ($failed.Count -gt 0) {
        throw "$($failed.Count) voice scenario(s) failed. See $summaryPath"
    }

    Write-Host "ALIA_VOICE_SCENARIO_MATRIX_PASS summary=$summaryPath"
} finally {
    Pop-Location
}
