param(
    [string]$Model = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [int]$Capacity = 2060,
    [string]$Log = '',
    [string]$BuildDir = 'build-refactor'
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $repo 'perf/PerfCommon.ps1')
$config = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

$exe = Join-Path $PSScriptRoot 'probe_qsa_layer.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "Missing $exe; run build_qsa_layer.ps1 first"
}

$mutex = [Threading.Mutex]::new($false, 'Global\AilaGpuSingleton')
$ownsMutex = $false
try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        throw 'GPU mutex Global\AilaGpuSingleton is busy'
    }

    & wsl.exe --shutdown
    if ($LASTEXITCODE -ne 0) {
        throw "wsl --shutdown failed: $LASTEXITCODE"
    }

    $busy = Get-Process Aila,AilaWorker,AilaServer,probe_token_graph,probe_qsa_layer `
        -ErrorAction SilentlyContinue
    if ($busy) {
        throw "GPU process already running: $($busy | ForEach-Object { $_.Name + ':' + $_.Id })"
    }

    $os = Get-CimInstance Win32_OperatingSystem
    $freeGiB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    if ($freeGiB -lt 54) {
        throw "Only $freeGiB GiB physical memory free; require >=54 GiB"
    }

    $prod = @{
        AILA_INIT_WARMUP = '0'
        UR_L0_USE_COPY_ENGINE = '1'
        SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE = '1'
        UR_L0_USE_IMMEDIATE_COMMANDLISTS = '0'
        SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = '0'
        SYCL_UR_USE_LEVEL_ZERO_V2 = '0'
        UR_L0_BATCH_SIZE = '0'
        SYCL_PI_LEVEL_ZERO_BATCH_SIZE = '0'
        SYCL_CACHE_PERSISTENT = '0'
    }
    foreach ($key in $prod.Keys) {
        [Environment]::SetEnvironmentVariable($key, $prod[$key], 'Process')
    }

    $argsList = @('--model', $Model, '--capacity', "$Capacity")
    $header = "[RUN] $(Get-Date -Format o) exe=$exe args=$($argsList -join ' ') free_gib=$freeGiB"
    if ($Log) {
        $header | Tee-Object -FilePath $Log
    } else {
        $header
    }

    $watch = [Diagnostics.Stopwatch]::StartNew()
    & $exe @argsList 2>&1 | ForEach-Object {
        if ($Log) {
            $_ | Tee-Object -FilePath $Log -Append
        } else {
            $_
        }
    }
    $code = $LASTEXITCODE
    $footer = "[RUN] exit=$code wall_s=$([math]::Round($watch.Elapsed.TotalSeconds, 1))"
    if ($Log) {
        $footer | Tee-Object -FilePath $Log -Append
    } else {
        $footer
    }
    exit $code
}
finally {
    if ($ownsMutex) {
        [void]$mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
