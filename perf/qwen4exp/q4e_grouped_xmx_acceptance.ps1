param(
    [string]$ModelDir = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [string]$BuildDir = 'build-refactor',
    [string]$Tag = 'x3-grouped',
    [ValidateSet('kernel', 'model')][string]$Mode = 'kernel',
    [int]$Microbatch = 0,
    [switch]$Diagnostic,
    [switch]$ProbeOnly,
    [int]$TimeoutSeconds = 900
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../PerfCommon.ps1')
$repo = Get-AilaRepoRoot
$cfg = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $cfg -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)
Get-ChildItem Env: | Where-Object { $_.Name -match '^(AILA_|ONEDNN_|DNNL_)' } | Remove-Item
$overrides = @{
    AILA_Q4E_CPU_OVERLAP = '0'; AILA_Q4E_CPU_EXPERT_DISPATCH = '0'; AILA_Q4E_STATIC_PINNING = '0'
    AILA_Q4E_PREFILL_XMX = '1'; AILA_Q4E_DIRECT_PREFILL_IO = '1'
    AILA_Q4E_INLINE_CHECKPOINT = '0'; AILA_Q4E_GROUPED_XMX_MIN_ROWS = '0'
    AILA_Q4E_DEMAND_MICROBATCH = "$Microbatch"
    AILA_TEST_X3_DIAGNOSTIC = $(if ($Diagnostic) { '1' } else { '0' })
    AILA_TEST_X3_PROBE_ONLY = $(if ($ProbeOnly) { '1' } else { '0' })
}
foreach ($key in $overrides.Keys) { Set-Item "env:$key" $overrides[$key] }
$exe = Join-Path (Join-Path $repo $BuildDir) 'AilaQwen4ExpToleranceTests.exe'
$prefix = Join-Path $repo "tmp/$Tag"
$metadata = [ordered]@{
    timestamp = (Get-Date -Format o); commit = (& git -C $repo rev-parse HEAD)
    dirty = @(& git -C $repo status --short)
    exe_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    harness_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tests/models/Qwen4ExpGroupedXmxAcceptance.hpp')).Hash
    kernel_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'src/ops/MoeGroupedXmx.cpp')).Hash
    model = $ModelDir; stack = (Get-AilaOneApiStackMetadata -Stack $stack)
    mode = $Mode; gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
    env_overrides = $overrides; max_seq = 512; prompt_tokens = @(30, 256)
    timing = 'Host wall: 3 interleaved pairs after warmup; real operator probes 3 x 20 pairs after 5 warmups'
}
$metadata | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
$arguments = @('--grouped-kernel')
if ($Mode -eq 'model') { $arguments = @('--grouped-model', ('"' + $ModelDir + '"')) }
$process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$prefix.log" -RedirectStandardError "$prefix.err.log"
$samples = [System.Collections.Generic.List[object]]::new()
$memoryPressure = $false
$timer = [System.Diagnostics.Stopwatch]::StartNew()
while (!$process.WaitForExit(500) -and $timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
    $process.Refresh()
    $freeMB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024
    $samples.Add([pscustomobject]@{ ms = $timer.ElapsedMilliseconds; working_set_bytes = $process.WorkingSet64
        private_commit_bytes = $process.PrivateMemorySize64; system_free_mb = $freeMB })
    if ($freeMB -lt 8192) { $memoryPressure = $true; break }
}
$samples | ConvertTo-Json -Depth 3 | Set-Content "$prefix.memory.json"
$metadata.peak_working_set_bytes = ($samples | Measure-Object working_set_bytes -Maximum).Maximum
$metadata.peak_private_commit_bytes = ($samples | Measure-Object private_commit_bytes -Maximum).Maximum
$metadata.minimum_system_free_mb = ($samples | Measure-Object system_free_mb -Minimum).Minimum
if (!$process.HasExited) {
    $process.Kill($true); $process.WaitForExit()
    $metadata.memory_pressure_stop = $memoryPressure
    $metadata | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
    throw "X3 watchdog stopped the process (memory_pressure=$memoryPressure); see $prefix.log"
}
$process.WaitForExit()
$metadata.exit_code = $process.ExitCode
$metadata | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
Get-Content "$prefix.log" | Select-String '\[X3|\[FAIL'
Get-Content "$prefix.err.log" | Select-Object -Last 6
exit $process.ExitCode
