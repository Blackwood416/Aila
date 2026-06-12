param(
    [string]$BuildDir = "build",
    [string]$AudioPath = "tmp\alia-real-smoke\alia_request.wav",
    [string]$OutputWav = "tmp\alia-real-smoke\alia_full_pipeline_target_models.wav",
    [string]$LogPath = "tmp\alia-real-smoke\alia_full_pipeline_target_models.log",
    [string]$RequestText = "Alia, please say hello in one short sentence.",
    [int]$MaxSeq = 2048,
    [int]$MaxTokens = 48,
    [int]$TimeoutSec = 1500,
    [int]$RollbackTokens = 0,
    [switch]$SkipBuild,
    [switch]$NoGenerateAudio
)

$ErrorActionPreference = "Stop"

function Ensure-ParentDirectory {
    param([string]$Path)

    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")

Push-Location $repoRoot
try {
    . .\perf\PerfCommon.ps1
    Initialize-AilaOneApiEnvironment

    if (-not $SkipBuild) {
        cmake -S . -B $BuildDir `
            -DAILA_BUILD_GENERIC_API=OFF `
            -DAILA_BUILD_GENERIC_CLI=OFF `
            -DAILA_BUILD_GENERIC_CHAT_TESTS=OFF `
            -DAILA_BUILD_ALIA_API_TESTS=OFF `
            -DAILA_BUILD_ALIA_REAL_SMOKE=ON
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

    Ensure-ParentDirectory $OutputWav
    Ensure-ParentDirectory $LogPath

    $env:AILA_INIT_WARMUP = "0"
    if (-not $env:AILA_LOG_LEVEL) {
        $env:AILA_LOG_LEVEL = "info"
    }

    $smokeArgs = @(
        "--asr-model", ".\models\Qwen3-ASR-1.7B-BNB-NF4",
        "--foreground-model", ".\models\qwen3.5-4B-bnb-nf4-offline-visiondense",
        "--background-model", ".\models\qwen3.5-0.8B-bnb-nf4-offline",
        "--tts-model", ".\models\Qwen3-TTS-12Hz-0.6B-Base",
        "--audio", $AudioPath,
        "--output-wav", $OutputWav,
        "--request-text", $RequestText,
        "--max-seq", "$MaxSeq",
        "--max-tokens", "$MaxTokens",
        "--timeout-sec", "$TimeoutSec",
        "--rollback-tokens", "$RollbackTokens"
    )
    if ($NoGenerateAudio) {
        $smokeArgs += "--no-generate-audio"
    }

    & $smokeExe @smokeArgs 2>&1 | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Alia target full-pipeline smoke failed with exit code $exitCode. See $LogPath"
    }
} finally {
    Pop-Location
}
