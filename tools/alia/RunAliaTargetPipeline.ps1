param(
    [string]$BuildDir = "build",
    [string]$AudioPath = "tmp\alia-real-smoke\alia_request.wav",
    [string]$OutputWav = "tmp\alia-real-smoke\alia_full_pipeline_target_models.wav",
    [string]$LogPath = "tmp\alia-real-smoke\alia_full_pipeline_target_models.log",
    [string]$ModelRoot = "",
    [string]$RequestText = "Alia, please say hello in one short sentence.",
    [string]$ForegroundLora = "F:\unsloth\qwen35_4b_alia_identity_r16_lr1e5\checkpoint-1400",
    [string]$ToolProbeText = "Call the host tool inspect_window with parameter id equal to 42 now. Return only the tool call.",
    [int]$MaxSeq = 2048,
    [int]$MaxTokens = 48,
    [int]$TimeoutSec = 1500,
    [int]$StreamChunkMs = 1000,
    [int]$StreamPrefillIntervalMs = 0,
    [switch]$SkipBuild,
    [switch]$NoGenerateAudio,
    [switch]$StreamAsrPrefill,
    [switch]$SkipToolProbe
)

$ErrorActionPreference = "Stop"

function Ensure-ParentDirectory {
    param([string]$Path)

    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

function Resolve-ModelRoot {
    param(
        [string]$RepoRoot,
        [string]$RequestedModelRoot
    )

    if ($RequestedModelRoot) {
        return (Resolve-Path -LiteralPath $RequestedModelRoot).Path
    }

    $localModels = Join-Path $RepoRoot "models"
    if (Test-Path -LiteralPath $localModels) {
        return (Resolve-Path -LiteralPath $localModels).Path
    }

    $mainWorktreeModels = Join-Path $RepoRoot "..\..\models"
    if (Test-Path -LiteralPath $mainWorktreeModels) {
        return (Resolve-Path -LiteralPath $mainWorktreeModels).Path
    }

    return $localModels
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$modelRootPath = Resolve-ModelRoot -RepoRoot $repoRoot -RequestedModelRoot $ModelRoot

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

    $smokeExe = Join-Path $BuildDir "AilaAliaRealSmoke.exe"
    if (-not (Test-Path -LiteralPath $smokeExe)) {
        throw "Smoke executable not found: $smokeExe"
    }
    if (-not (Test-Path -LiteralPath $ForegroundLora)) {
        throw "Foreground LoRA not found: $ForegroundLora"
    }

    Ensure-ParentDirectory $OutputWav
    Ensure-ParentDirectory $LogPath

    $env:AILA_INIT_WARMUP = "0"
    if (-not $env:AILA_LOG_LEVEL) {
        $env:AILA_LOG_LEVEL = "info"
    }
    if (-not $env:AILA_TTS_REF_AUDIO) {
        $refAudio = Join-Path $repoRoot "alia_ref.wav"
        if (Test-Path -LiteralPath $refAudio) {
            $env:AILA_TTS_REF_AUDIO = (Resolve-Path -LiteralPath $refAudio).Path
        }
    }

    $smokeArgs = @(
        "--asr-model", (Join-Path $modelRootPath "Qwen3-ASR-1.7B-BNB-NF4"),
        "--foreground-model", (Join-Path $modelRootPath "qwen3.5-4B-bnb-nf4-offline-visiondense"),
        "--background-model", (Join-Path $modelRootPath "qwen3.5-0.8B-bnb-nf4-offline"),
        "--tts-model", (Join-Path $modelRootPath "Qwen3-TTS-12Hz-0.6B-Base"),
        "--audio", $AudioPath,
        "--output-wav", $OutputWav,
        "--request-text", $RequestText,
        "--max-seq", "$MaxSeq",
        "--max-tokens", "$MaxTokens",
        "--timeout-sec", "$TimeoutSec",
        "--stream-chunk-ms", "$StreamChunkMs",
        "--stream-prefill-interval-ms", "$StreamPrefillIntervalMs",
        "--foreground-lora", $ForegroundLora,
        "--tool-probe-text", $ToolProbeText
    )
    if ($NoGenerateAudio) {
        $smokeArgs += "--no-generate-audio"
    }
    if ($StreamAsrPrefill) {
        $smokeArgs += "--stream-asr-prefill"
    }
    if ($SkipToolProbe) {
        $smokeArgs += "--skip-tool-probe"
    }

    & $smokeExe @smokeArgs 2>&1 | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Alia target full-pipeline smoke failed with exit code $exitCode. See $LogPath"
    }
} finally {
    Pop-Location
}
