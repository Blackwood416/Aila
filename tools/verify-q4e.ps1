param(
    [ValidateSet("build", "fast", "ctest", "real", "decode", "all")]
    [string]$Tier = "ctest",
    [string[]]$Targets = @(),
    [string]$BuildDir = "build-refactor",
    [string]$Model = "D:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

# ---------------------------------------------------------------------------
# 1. Standardized SYCL / Level Zero Runtime Environment
# ---------------------------------------------------------------------------
if ($env:AILA_KEEP_SYCL_CACHE_PERSISTENT -ne "1") {
    $env:SYCL_CACHE_PERSISTENT = "0"
}
$env:UR_L0_USE_COPY_ENGINE = "1"
$env:SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE = "1"
$env:UR_L0_USE_IMMEDIATE_COMMANDLISTS = "0"
$env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = "0"
$env:SYCL_UR_USE_LEVEL_ZERO_V2 = "0"
$env:UR_L0_BATCH_SIZE = "64"
$env:SYCL_PI_LEVEL_ZERO_BATCH_SIZE = "64"

function Invoke-IncrementalBuild {
    param([string[]]$BuildTargets)
    $vcvarsCandidates = @(
        "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    $vcvars = $vcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $vcvars) {
        throw "Cannot locate vcvars64.bat in Visual Studio installation."
    }
    $oneApiLibCandidates = @(
        "C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\lib",
        "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
    )
    $oneApiLib = $oneApiLibCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $oneApiLib) {
        throw "Cannot locate Intel oneAPI compiler lib directory."
    }

    $BuildTargets = @($BuildTargets | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
    $targetArg = if ($BuildTargets.Count -gt 0) { "--target " + ($BuildTargets -join " ") } else { "" }
    $cmdLine = "call `"$vcvars`" >nul 2>&1 && set `"LIB=$oneApiLib;!LIB!`" && cmake --build `"$BuildDir`" $targetArg"
    Write-Host "[Verify] Building targets: $(if ($BuildTargets.Count -gt 0) { $BuildTargets -join ', ' } else { '<ALL>' })" -ForegroundColor Cyan
    cmd /v:on /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "Incremental build failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host "`n======================================================================" -ForegroundColor DarkGray
    Write-Host "  [STEP] $Name" -ForegroundColor Yellow
    Write-Host "======================================================================" -ForegroundColor DarkGray
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Action | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
    $sw.Stop()
    Write-Host ("  [PASS] {0} ({1:N1} s)" -f $Name, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    return [pscustomobject]@{
        Step    = $Name
        Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        Status  = "PASS"
    }
}

$fastTargets = @(
    "AilaBnb4BitLinearKernelTests",
    "AilaGdnChunkTests",
    "AilaMoeExpertXmxTests",
    "AilaQwen4ExpHostArenaTests",
    "AilaQwen4ExpArenaVerifyTests",
    "AilaQwen4ExpExpertReaderTests"
)

$ctestTargets = $fastTargets + @(
    "Aila",
    "AilaQwen4ExpKernelTests",
    "AilaQwen4ExpModuleTests",
    "AilaQwen4ExpGenerationTests",
    "AilaQwen4ExpToleranceTests",
    "AilaQwen4ExpAdaptiveCacheTests"
)

if (-not $SkipBuild) {
    if ($Targets.Count -gt 0) {
        Invoke-IncrementalBuild -BuildTargets $Targets
    } elseif ($Tier -eq "build") {
        Invoke-IncrementalBuild -BuildTargets @()
    } elseif ($Tier -eq "fast") {
        Invoke-IncrementalBuild -BuildTargets $fastTargets
    } elseif ($Tier -eq "decode") {
        Invoke-IncrementalBuild -BuildTargets @("Aila")
    } else {
        Invoke-IncrementalBuild -BuildTargets $ctestTargets
    }
}

if ($Tier -eq "build") {
    exit 0
}

$results = @()

if ($Tier -in @("fast", "ctest", "all")) {
    $results += Invoke-Step "Bnb4BitLinear Kernel Tests" {
        & ".\$BuildDir\AilaBnb4BitLinearKernelTests.exe"
    }
    $results += Invoke-Step "GDN Chunk Unit & Scan Parity Tests" {
        & ".\$BuildDir\AilaGdnChunkTests.exe"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & ".\$BuildDir\AilaGdnChunkTests.exe" --scan-parity
    }
    $results += Invoke-Step "MoE Expert XMX Pipeline Tests" {
        & ".\$BuildDir\AilaMoeExpertXmxTests.exe"
    }
    $results += Invoke-Step "HostArena & ArenaVerify & ExpertReader Tests" {
        & ".\$BuildDir\AilaQwen4ExpHostArenaTests.exe"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & ".\$BuildDir\AilaQwen4ExpArenaVerifyTests.exe"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & ".\$BuildDir\AilaQwen4ExpExpertReaderTests.exe"
    }
}

if ($Tier -in @("ctest", "all")) {
    $results += Invoke-Step "Qwen4Exp Operator Kernel Suite" {
        & ".\$BuildDir\AilaQwen4ExpKernelTests.exe"
    }
    $results += Invoke-Step "CTest qwen4_exp Suite (11 suites)" {
        ctest --test-dir $BuildDir -L qwen4_exp --output-on-failure
    }
    $results += Invoke-Step "CTest cpu|abi Fast CI Suite (20 suites)" {
        ctest --test-dir $BuildDir -L "cpu|abi" --output-on-failure
    }
}

if ($Tier -in @("real", "all")) {
    if (-not (Test-Path $Model)) {
        throw "Real-model tier requires model directory at '$Model'"
    }
    $results += Invoke-Step "Real-Model Tolerance & E2E Generation (--model)" {
        $prevWarmup = $env:AILA_INIT_WARMUP
        try {
            $env:AILA_INIT_WARMUP = "0"
            & ".\$BuildDir\AilaQwen4ExpToleranceTests.exe" --model $Model
        } finally {
            if ($null -eq $prevWarmup) {
                Remove-Item Env:\AILA_INIT_WARMUP -ErrorAction SilentlyContinue
            } else {
                $env:AILA_INIT_WARMUP = $prevWarmup
            }
        }
    }
    $results += Invoke-Step "Real-Model Preset & Interior Checkpoint (--preset-model)" {
        $prevWarmup = $env:AILA_INIT_WARMUP
        try {
            $env:AILA_INIT_WARMUP = "0"
            & ".\$BuildDir\AilaQwen4ExpToleranceTests.exe" --preset-model $Model
        } finally {
            if ($null -eq $prevWarmup) {
                Remove-Item Env:\AILA_INIT_WARMUP -ErrorAction SilentlyContinue
            } else {
                $env:AILA_INIT_WARMUP = $prevWarmup
            }
        }
    }
    $results += Invoke-Step "Real-Model Prefill Benchmark (7764 tokens, chunk 8192)" {
        pwsh -NoProfile -File "$RepoRoot\tools\prefill-bench.ps1" -Tokens @(7878) -Chunk 8192 -Model $Model -Exe "$BuildDir\Aila.exe"
    }
}

if ($Tier -in @("decode", "all")) {
    if (-not (Test-Path $Model)) {
        throw "Decode tier requires model directory at '$Model'"
    }
    $results += Invoke-Step "Real-Model Decode Benchmark (256 / 512 tokens, greedy)" {
        pwsh -NoProfile -File "$RepoRoot\tools\decode-bench.ps1" -Model $Model -Exe "$BuildDir\Aila.exe" -MaxTokens @(256, 512)
    }
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "  Verification Summary (Tier: $Tier)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
$results | Format-Table -AutoSize
