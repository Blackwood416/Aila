param(
    [string]$BuildDir = "build",
    [string]$OutputDir = "tmp\command-submission-bench",
    [string]$LevelZeroSdkRoot = "C:\level-zero-win-sdk-1.28.2",
    [int]$Warmup = 200,
    [int]$Iters = 2000,
    [string[]]$Bytes = @("4", "4096", "65536"),
    [int]$OpsPerGraph = 16,
    [string[]]$ImmediateModes = @("__unset__", "0", "1", "2"),
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$runName = Get-Date -Format "yyyyMMdd-HHmmss"
$outputRoot = Join-Path $repoRoot $OutputDir
$rawRoot = Join-Path $outputRoot $runName
New-Item -ItemType Directory -Force -Path $rawRoot | Out-Null

$byteValues = foreach ($entry in $Bytes) {
    foreach ($part in ($entry -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            [int]$trimmed
        }
    }
}

Push-Location $repoRoot
try {
    . .\perf\PerfCommon.ps1
    Initialize-AilaOneApiEnvironment

    if (-not $SkipBuild) {
        cmake -S . -B $BuildDir "-DAILA_LEVEL_ZERO_SDK_ROOT=$LevelZeroSdkRoot"
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configure failed with exit code $LASTEXITCODE"
        }
        cmake --build $BuildDir --target AilaCommandSubmissionBench --config Release
        if ($LASTEXITCODE -ne 0) {
            throw "AilaCommandSubmissionBench build failed with exit code $LASTEXITCODE"
        }
    }

    $benchExe = Join-Path $repoRoot (Join-Path $BuildDir "AilaCommandSubmissionBench.exe")
    if (-not (Test-Path -LiteralPath $benchExe)) {
        $benchExe = Join-Path $repoRoot (Join-Path $BuildDir "Release\AilaCommandSubmissionBench.exe")
    }
    if (-not (Test-Path -LiteralPath $benchExe)) {
        throw "AilaCommandSubmissionBench.exe not found under $BuildDir"
    }

    $csvPath = Join-Path $rawRoot "summary.csv"
    $latestCsvPath = Join-Path $outputRoot "latest-summary.csv"
    $headerWritten = $false
    $previousImmediateMode = [Environment]::GetEnvironmentVariable("UR_L0_USE_IMMEDIATE_COMMANDLISTS", "Process")

    foreach ($bytesValue in $byteValues) {
        foreach ($mode in $ImmediateModes) {
            if ($mode -eq "__unset__") {
                Remove-Item Env:UR_L0_USE_IMMEDIATE_COMMANDLISTS -ErrorAction SilentlyContinue
                $modeLabel = "unset"
            } else {
                $env:UR_L0_USE_IMMEDIATE_COMMANDLISTS = $mode
                $modeLabel = $mode
            }

            $rawPath = Join-Path $rawRoot "bytes_${bytesValue}_ur_${modeLabel}.log"
            Write-Host "Running bytes=$bytesValue UR_L0_USE_IMMEDIATE_COMMANDLISTS=$modeLabel" -ForegroundColor Cyan
            $output = & $benchExe --warmup $Warmup --iters $Iters --bytes $bytesValue --ops-per-graph $OpsPerGraph 2>&1
            $exitCode = $LASTEXITCODE
            $output | Set-Content -LiteralPath $rawPath
            if ($exitCode -ne 0) {
                throw "Benchmark failed for bytes=$bytesValue mode=$modeLabel with exit code $exitCode. See $rawPath"
            }

            foreach ($line in $output) {
                if ($line -like "backend,mode,*") {
                    if (-not $headerWritten) {
                        "ur_l0_use_immediate_commandlists,$line" | Set-Content -LiteralPath $csvPath
                        $headerWritten = $true
                    }
                } elseif ($line -match "^(sycl|level_zero),") {
                    "$modeLabel,$line" | Add-Content -LiteralPath $csvPath
                }
            }
        }
    }

    Copy-Item -LiteralPath $csvPath -Destination $latestCsvPath -Force
} finally {
    if ($null -eq $previousImmediateMode) {
        Remove-Item Env:UR_L0_USE_IMMEDIATE_COMMANDLISTS -ErrorAction SilentlyContinue
    } else {
        $env:UR_L0_USE_IMMEDIATE_COMMANDLISTS = $previousImmediateMode
    }
    Pop-Location
}

Write-Host "Summary: $csvPath" -ForegroundColor Green
Write-Host "Latest summary: $latestCsvPath" -ForegroundColor Green
Write-Host "Raw logs: $rawRoot" -ForegroundColor Green
