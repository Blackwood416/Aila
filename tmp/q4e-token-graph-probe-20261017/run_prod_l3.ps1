param(
    [string]$Model = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [int]$Capacity = 4096,
    [string]$PromptFile =
        'tmp/q4e-token-graph-probe-20261017/prefix-2060.txt',
    [string]$OutputDir =
        'tmp/q4e-token-graph-probe-20261017/prod-l3-4096-20261020',
    [ValidateSet('both', 'direct', 'graph')]
    [string]$Lanes = 'both'
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$out = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $out | Out-Null
$prompt = if ([IO.Path]::IsPathRooted($PromptFile)) {
    $PromptFile
} else {
    Join-Path $repo $PromptFile
}
if (-not (Test-Path -LiteralPath $prompt)) {
    throw "Missing prompt $prompt"
}
if ($Capacity -ne 4096 -and $Capacity -ne 8192) {
    throw 'Capacity must be 4096 or 8192'
}

. (Join-Path $repo 'perf/PerfCommon.ps1')
$config = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

$profilePath = Join-Path $repo 'docs/plans/qwen38-server-mtp-2026-10-07.json'
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
foreach ($entry in $profile.worker_environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable(
        $entry.Name, [string]$entry.Value, 'Process')
}
$envs = @{
    AILA_INIT_WARMUP = '0'
    AILA_Q4E_SERVER_MTP = '0'
    AILA_Q4E_VRAM_RESERVE_MB = '3584'
    AILA_Q4E_HOTSET_PROFILE = (
        Join-Path $repo 'perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile')
    AILA_Q4E_QSA_LAYER_GRAPH_TRACE = '1'
}
foreach ($key in $envs.Keys) {
    [Environment]::SetEnvironmentVariable($key, $envs[$key], 'Process')
}

$exe = Join-Path $PSScriptRoot 'probe_prod_l3.exe'
$lib = Join-Path $repo 'build-refactor/AilaLib.lib'
foreach ($path in @($exe, $lib)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing $path"
    }
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
    $busy = Get-Process Aila,AilaWorker,AilaServer,probe_* `
        -ErrorAction SilentlyContinue
    if ($busy) {
        throw "GPU process already running: $($busy | ForEach-Object {
            $_.Name + ':' + $_.Id
        })"
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGiB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    if ($freeGiB -lt 54) {
        throw "Only $freeGiB GiB physical memory free; require >=54 GiB"
    }

    $direct = Join-Path $out 'direct.bin'
    $graph = Join-Path $out 'graph.bin'
    foreach ($path in @(
        $direct, $graph,
        "$direct.txt", "$graph.txt",
        (Join-Path $out 'direct.stdout.log'),
        (Join-Path $out 'direct.stderr.log'),
        (Join-Path $out 'graph.stdout.log'),
        (Join-Path $out 'graph.stderr.log'),
        (Join-Path $out 'compare.log')
    )) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }

    $baseArgs = @(
        '--model', $Model,
        '--capacity', "$Capacity",
        '--prompt-file', $prompt
    )
    $receipt = [ordered]@{
        started_at = (Get-Date -Format o)
        commit = (& git -C $repo rev-parse HEAD)
        dirty = @(& git -C $repo status --short)
        capacity = $Capacity
        free_gib = $freeGiB
        lib_sha256 = (Get-FileHash -LiteralPath $lib -Algorithm SHA256).Hash
        exe_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
        prompt_sha256 = (
            Get-FileHash -LiteralPath $prompt -Algorithm SHA256).Hash
        scope = 'selected L3 K/V/indexer/compressed cache rows only'
        runs = @()
    }

    $lanesToRun = if ($Lanes -eq 'both') {
        @('direct', 'graph')
    } else {
        @($Lanes)
    }
    foreach ($lane in $lanesToRun) {
        $output = if ($lane -eq 'direct') { $direct } else { $graph }
        $stdout = Join-Path $out "$lane.stdout.log"
        $stderr = Join-Path $out "$lane.stderr.log"
        $argsList = @('--lane', $lane) + $baseArgs +
            @('--output', $output)
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath $exe -ArgumentList $argsList `
            -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $watch.Stop()
        $run = [ordered]@{
            lane = $lane
            exit_code = $process.ExitCode
            wall_s = [math]::Round($watch.Elapsed.TotalSeconds, 3)
            output_exists = [bool](Test-Path -LiteralPath $output)
            output_sha256 = if (Test-Path -LiteralPath $output) {
                (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
            } else {
                $null
            }
            stdout_sha256 = (
                Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash
            stderr_sha256 = (
                Get-FileHash -LiteralPath $stderr -Algorithm SHA256).Hash
        }
        $receipt.runs += $run
        if ($process.ExitCode -ne 0) {
            $receipt | ConvertTo-Json -Depth 8 |
                Set-Content -LiteralPath (Join-Path $out 'receipt.json')
            throw "$lane lane failed with exit $($process.ExitCode)"
        }
    }

    if ($Lanes -eq 'both') {
        $compareLog = Join-Path $out 'compare.log'
        $compare = Start-Process -FilePath $exe `
            -ArgumentList @('--compare', $direct, $graph) `
            -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait `
            -RedirectStandardOutput $compareLog `
            -RedirectStandardError (Join-Path $out 'compare.stderr.log')
        $receipt.compare_exit = $compare.ExitCode
        $receipt.compare_log = Get-Content -LiteralPath $compareLog -Raw
    }
    $receipt.completed_at = (Get-Date -Format o)
    $receipt | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $out 'receipt.json')
    if ($Lanes -eq 'both') {
        Write-Output $receipt.compare_log
    }
    if ($Lanes -eq 'both' -and $compare.ExitCode -ne 0) {
        exit $compare.ExitCode
    }
}
finally {
    if ($ownsMutex) {
        [void]$mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
