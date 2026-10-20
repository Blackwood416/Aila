param(
    [string]$OutputDir = 'tmp/q4e-token-graph-probe-20261017/visibility-20261020',
    [switch]$SkipHost
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$out = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $out | Out-Null

. (Join-Path $repo 'perf/PerfCommon.ps1')
$config = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

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
        runs = @()
    }
    $hostExe = Join-Path $PSScriptRoot 'probe_host_ring.exe'
    $copyExe = Join-Path $PSScriptRoot 'probe_packet_copy.exe'
    $cases = @(
        @{
            name = 'host-ring'
            exe = $hostExe
            args = @()
        },
        @{
            name = 'copy-pinned-r10'
            exe = $copyExe
            args = @('--top-k', '10', '--rows', '10',
                     '--warmup', '20', '--replays', '300',
                     '--payload-alloc', 'pinned')
        },
        @{
            name = 'copy-uncached-r10'
            exe = $copyExe
            args = @('--top-k', '10', '--rows', '10',
                     '--warmup', '20', '--replays', '300',
                     '--payload-alloc', 'uncached')
        },
        @{
            name = 'copy-pinned-r3'
            exe = $copyExe
            args = @('--top-k', '10', '--rows', '3',
                     '--warmup', '20', '--replays', '300',
                     '--payload-alloc', 'pinned')
        }
    )
    if ($SkipHost) {
        $cases = $cases | Where-Object { $_.name -ne 'host-ring' }
    }

    foreach ($case in $cases) {
        $stdout = Join-Path $out ("$($case.name).log")
        $stderr = Join-Path $out ("$($case.name).err.log")
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath $case.exe `
            -ArgumentList $case.args -WorkingDirectory $repo `
            -WindowStyle Hidden -PassThru -Wait `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $watch.Stop()
        $run = [ordered]@{
            name = $case.name
            exit_code = $process.ExitCode
            wall_s = [math]::Round($watch.Elapsed.TotalSeconds, 3)
            stdout_sha256 = (
                Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash
            stderr_sha256 = (
                Get-FileHash -LiteralPath $stderr -Algorithm SHA256).Hash
        }
        $receipt.runs += $run
        if ($process.ExitCode -ne 0) {
            $receipt | ConvertTo-Json -Depth 8 |
                Set-Content -LiteralPath (Join-Path $out 'receipt.json')
            throw "$($case.name) failed with exit $($process.ExitCode)"
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
