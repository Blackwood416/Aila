param(
    [string]$BuildDir = "build",
    [string]$OutputDir = "tmp\alia-real-smoke\voice_matrix",
    [int]$TimeoutSec = 1500,
    [switch]$SkipBuild,
    [switch]$IncludeToolProbe
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
            RequestText = "Alia, please say hello in one short sentence."
            MaxTokens = 48
        },
        [pscustomobject]@{
            Name = "persona_chat"
            RequestText = "Alia, I feel tired today. Say something gentle and brief."
            MaxTokens = 64
        },
        [pscustomobject]@{
            Name = "preference_memory"
            RequestText = "Alia, please remember that I prefer short Chinese replies at night."
            MaxTokens = 64
        },
        [pscustomobject]@{
            Name = "task_memory"
            RequestText = "Alia, remind me later to stretch my shoulders after work."
            MaxTokens = 64
        },
        [pscustomobject]@{
            Name = "long_answer"
            RequestText = "Alia, explain in three short sentences how you will help me focus tonight."
            MaxTokens = 96
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
                RequestText = $scenario.RequestText
                MaxTokens = $scenario.MaxTokens
                TimeoutSec = $TimeoutSec
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
        $values = Convert-KeyValueLog -LogPath $absoluteLogPath
        $logText = ""
        if (Test-Path -LiteralPath $absoluteLogPath) {
            $logText = Get-Content -Raw -LiteralPath $absoluteLogPath
        }
        $passed = $passed -and $logText.Contains("ALIA_REAL_MODEL_SMOKE_PASS")

        $rows += [pscustomobject]@{
            scenario = $scenario.Name
            pass = $passed
            error = $errorText
            request_text = $scenario.RequestText
            model_load_ms = Get-ValueOrEmpty $values "model_load_ms"
            asr_ms = Get-ValueOrEmpty $values "asr_ms"
            foreground_ms = Get-ValueOrEmpty $values "foreground_ms"
            foreground_prompt_tokens = Get-ValueOrEmpty $values "foreground_prompt_tokens"
            foreground_generated_tokens = Get-ValueOrEmpty $values "foreground_generated_tokens"
            foreground_asr_prefill_tokens = Get-ValueOrEmpty $values "foreground_asr_prefill_tokens"
            foreground_asr_prefill_ms = Get-ValueOrEmpty $values "foreground_asr_prefill_ms"
            foreground_first_content_delta_ms = Get-ValueOrEmpty $values "foreground_first_content_delta_ms"
            foreground_first_tts_enqueue_ms = Get-ValueOrEmpty $values "foreground_first_tts_enqueue_ms"
            tts_first_audio_ms = Get-ValueOrEmpty $values "tts_first_audio_ms"
            tts_callback_count = Get-ValueOrEmpty $values "tts_callback_count"
            tts_chunks_synthesized = Get-ValueOrEmpty $values "tts_chunks_synthesized"
            tts_first_text_chars = Get-ValueOrEmpty $values "tts_first_text_chars"
            tts_first_text_tokens = Get-ValueOrEmpty $values "tts_first_text_tokens"
            tts_first_backend_frames = Get-ValueOrEmpty $values "tts_first_backend_frames"
            tts_first_backend_callbacks = Get-ValueOrEmpty $values "tts_first_backend_callbacks"
            tts_first_backend_audio_samples = Get-ValueOrEmpty $values "tts_first_backend_audio_samples"
            tts_first_backend_codes_ms = Get-ValueOrEmpty $values "tts_first_backend_codes_ms"
            tts_first_backend_mimi_init_ms = Get-ValueOrEmpty $values "tts_first_backend_mimi_init_ms"
            tts_first_backend_audio_ms = Get-ValueOrEmpty $values "tts_first_backend_audio_ms"
            tts_first_backend_total_ms = Get-ValueOrEmpty $values "tts_first_backend_total_ms"
            tts_backend_total_ms = Get-ValueOrEmpty $values "tts_backend_total_ms"
            tts_total_samples = Get-ValueOrEmpty $values "tts_total_samples"
            background_ms = Get-ValueOrEmpty $values "background_ms"
            background_schema_valid = Get-ValueOrEmpty $values "background_schema_valid"
            tool_probe_enabled = Get-ValueOrEmpty $values "tool_probe"
            tool_probe_ms = Get-ValueOrEmpty $values "tool_probe_ms"
            foreground_lora_applied = Get-ValueOrEmpty $values "foreground_lora_applied"
            foreground_lora_pair_count = Get-ValueOrEmpty $values "foreground_lora_pair_count"
            asr_partial_text = Get-ValueOrEmpty $values "asr_partial_text"
            foreground_assistant_text = Get-ValueOrEmpty $values "foreground_assistant_text"
            background_result_json = Get-ValueOrEmpty $values "background_result_json"
            log_path = $logPath
            output_wav = $wavPath
        }
    }

    $summaryPath = Join-Path $outputRoot "summary.csv"
    $rows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
    $rows | Format-Table scenario, pass, asr_ms, foreground_first_content_delta_ms, foreground_first_tts_enqueue_ms, tts_first_audio_ms, foreground_ms, background_ms -AutoSize

    $failed = @($rows | Where-Object { -not $_.pass })
    if ($failed.Count -gt 0) {
        throw "$($failed.Count) voice scenario(s) failed. See $summaryPath"
    }

    Write-Host "ALIA_VOICE_SCENARIO_MATRIX_PASS summary=$summaryPath"
} finally {
    Pop-Location
}
