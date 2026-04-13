param(
    [string]$ModelDir = "..\Qwen3.5-0.8B",
    [int]$PromptTokens = 512,
    [int]$GenTokens = 512,
    [int]$BenchIters = 5,
    [int]$WarmupIters = 1,
    [switch]$Sample,
    [double]$Temperature = 0.7,
    [int]$TopK = 15,
    [double]$TopP = 0.95,
    [UInt64]$Seed = 42,
    [int]$WaitTimeoutSec = 3600
)

$ErrorActionPreference = "Stop"

$mutexName = 'Global\AilaBenchSingleton'
$mutex = [System.Threading.Mutex]::new($false, $mutexName)
$hasMutex = $false

function Acquire-BenchMutex {
    param(
        [System.Threading.Mutex]$Mutex,
        [int]$TimeoutSec
    )

    try {
        if ($Mutex.WaitOne(0)) {
            return $true
        }
    }
    catch [System.Threading.AbandonedMutexException] {
        Write-Host ':: benchmark mutex was abandoned, recovering lock ::' -ForegroundColor Yellow
        return $true
    }

    if ($TimeoutSec -lt 0) {
        Write-Host ':: another benchmark is running, waiting for lock ::' -ForegroundColor Yellow
        try {
            $Mutex.WaitOne() | Out-Null
            return $true
        }
        catch [System.Threading.AbandonedMutexException] {
            Write-Host ':: benchmark mutex was abandoned while waiting, recovering lock ::' -ForegroundColor Yellow
            return $true
        }
    }

    Write-Host ":: another benchmark is running, waiting up to $TimeoutSec s for lock ::" -ForegroundColor Yellow
    try {
        return $Mutex.WaitOne([TimeSpan]::FromSeconds($TimeoutSec))
    }
    catch [System.Threading.AbandonedMutexException] {
        Write-Host ':: benchmark mutex was abandoned while waiting, recovering lock ::' -ForegroundColor Yellow
        return $true
    }
}

try {
    $hasMutex = Acquire-BenchMutex -Mutex $mutex -TimeoutSec $WaitTimeoutSec
    if (-not $hasMutex) {
        throw "Timed out waiting for benchmark lock after $WaitTimeoutSec seconds."
    }

    cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && set' |
        ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
            }
        }
    Write-Host ':: oneAPI environment initialized ::' -ForegroundColor Green

    $root = "e:\RiderProjects\Aila"
    $buildDir = Join-Path $root "build"
    $logPath = Join-Path $root "bench_log.txt"
    $mode = if ($Sample.IsPresent) { "sample" } else { "greedy" }
    $cachePath = Join-Path $buildDir "CMakeCache.txt"

    if (Test-Path $cachePath) {
        $buildTypeLine = Select-String -Path $cachePath -Pattern '^CMAKE_BUILD_TYPE:STRING=' | Select-Object -First 1
        if ($buildTypeLine) {
            $buildType = ($buildTypeLine.Line -split '=', 2)[1]
            if ($buildType -and $buildType -ne 'Release') {
                Write-Host ":: warning: build dir is configured as $buildType; benchmark numbers may be much slower than Release ::" -ForegroundColor Yellow
            }
        }
    }

    $header = "=== bench $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') mode=$mode model=$ModelDir pp=$PromptTokens tg=$GenTokens iters=$BenchIters warmup=$WarmupIters seed=$Seed temp=$Temperature topk=$TopK topp=$TopP ==="
    $header | Tee-Object -FilePath $logPath -Append | Out-Null

    Push-Location $buildDir
    try {
        $args = @(
            "-m", $ModelDir,
            "--bench",
            "--bench-pp", "$PromptTokens",
            "--bench-tg", "$GenTokens",
            "--bench-iters", "$BenchIters",
            "--bench-warmup", "$WarmupIters",
            "--seed", "$Seed",
            "-t", "$Temperature",
            "-k", "$TopK",
            "-p", "$TopP"
        )

        if ($Sample.IsPresent) {
            $args += "--bench-sample"
        } else {
            $args += "--bench-greedy"
        }

        & .\Aila.exe @args | Tee-Object -FilePath $logPath -Append
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($hasMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
