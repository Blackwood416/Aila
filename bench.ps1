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
    [UInt64]$Seed = 42
)

$ErrorActionPreference = "Stop"

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

$header = "=== bench $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') mode=$mode model=$ModelDir pp=$PromptTokens tg=$GenTokens iters=$BenchIters warmup=$WarmupIters seed=$Seed temp=$Temperature topk=$TopK topp=$TopP ==="
$header | Tee-Object -FilePath $logPath -Append | Out-Null

Set-Location $buildDir

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

Set-Location $root
