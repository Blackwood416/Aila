param(
    [string]$BuildDir = "build",
    [string]$TtsModel = "E:\RiderProjects\Aila\models\Qwen3-TTS-12Hz-0.6B-Base",
    [string]$RefAudio = "E:\RiderProjects\Aila\.worktrees\alia-custom-engine\alia_ref.wav",
    [string]$Text = "艾莉亚，请用一句话打个招呼。今天天气很好，我们一起出去走走吧。",
    [string]$OutputDir = "tmp\alia-real-smoke\tts_stream_probe",
    [int]$MaxTokens = 512,
    [int]$StreamBatch = 4,
    [int]$ChunkDelayMs = 120,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
Push-Location $repoRoot
try {
    . .\perf\PerfCommon.ps1
    Initialize-AilaOneApiEnvironment

    if (-not $SkipBuild) {
        cmake -S . -B $BuildDir
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configure failed with exit code $LASTEXITCODE"
        }
        cmake --build $BuildDir --target AilaAliaTtsStreamProbe --config Release
        if ($LASTEXITCODE -ne 0) {
            throw "AilaAliaTtsStreamProbe build failed with exit code $LASTEXITCODE"
        }
    }

    $probeExe = Join-Path $BuildDir "AilaAliaTtsStreamProbe.exe"
    if (-not (Test-Path -LiteralPath $probeExe)) {
        throw "Probe executable not found: $probeExe"
    }

    & $probeExe `
        --tts-model $TtsModel `
        --ref $RefAudio `
        --text $Text `
        --output-dir $OutputDir `
        --max-tokens $MaxTokens `
        --stream-batch $StreamBatch `
        --chunk-delay-ms $ChunkDelayMs
    if ($LASTEXITCODE -ne 0) {
        throw "TTS stream probe failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
