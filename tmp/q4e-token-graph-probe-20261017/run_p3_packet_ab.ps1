param(
    [string]$Model = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [uint32]$Replays = 4,
    [string]$OutputDir = 'tmp/q4e-token-graph-probe-20261017/p3-packet-ab-20261020'
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$out = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $out | Out-Null
$runner = Join-Path $PSScriptRoot 'run.ps1'
$pwsh = (Get-Command pwsh).Source

[Environment]::SetEnvironmentVariable('AILA_Q4E_COLD_MODE', '2', 'Process')
[Environment]::SetEnvironmentVariable(
    'AILA_Q4E_CPU_MAX_EXPERTS', '16', 'Process')

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

    $receipt = [ordered]@{
        started_at = (Get-Date -Format o)
        commit = (& git -C $repo rev-parse HEAD)
        free_gib = $freeGiB
        replays = $Replays
        runs = @()
    }
    foreach ($arm in @(
        @{
            name = 'scalar'
            mode = 'scalar'
            log = Join-Path $out 'gpu-p3-packet-scalar.log'
        },
        @{
            name = 'wg_stride'
            mode = 'wg_stride'
            log = Join-Path $out 'gpu-p3-packet-wg-stride.log'
        }
    )) {
        $argsList = @(
            '-NoProfile', '-File', $runner,
            '-TwoLayer',
            '-Model', $Model,
            '-Replays', "$Replays",
            '-CpuUs', '0',
            '-PacketCopyMode', $arm.mode,
            '-PacketWg', '256',
            '-Log', $arm.log
        )
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath $pwsh -ArgumentList $argsList `
            -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait
        $watch.Stop()
        $run = [ordered]@{
            arm = $arm.name
            exit_code = $process.ExitCode
            wall_s = [math]::Round($watch.Elapsed.TotalSeconds, 3)
            log_sha256 = if (Test-Path -LiteralPath $arm.log) {
                (Get-FileHash -LiteralPath $arm.log -Algorithm SHA256).Hash
            } else {
                $null
            }
        }
        $receipt.runs += $run
        if ($process.ExitCode -ne 0) {
            $receipt | ConvertTo-Json -Depth 8 |
                Set-Content -LiteralPath (Join-Path $out 'receipt.json')
            throw "$($arm.name) P3 run failed with exit $($process.ExitCode)"
        }
    }
    $receipt.completed_at = (Get-Date -Format o)
    $receipt | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $out 'receipt.json')
    $receipt | ConvertTo-Json -Depth 8
}
finally {
    if ($ownsMutex) {
        [void]$mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
