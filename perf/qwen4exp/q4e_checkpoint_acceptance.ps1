param(
    [string]$ModelDir = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [string]$BuildDir = 'build-refactor',
    [string]$Tag = 'x2-checkpoint',
    [ValidateSet('checkpoint', 'memory')][string]$Mode = 'checkpoint',
    [ValidateSet('default', 'mmap', 'readfile')][string]$Reader = 'default',
    [int]$TimeoutSeconds = 900
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../PerfCommon.ps1')
$repo = Get-AilaRepoRoot
$stackConfig = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $stackConfig -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)
# This runner is a dedicated child process: remove inherited experiment overrides.
Get-ChildItem Env: | Where-Object { $_.Name -match '^(AILA_|ONEDNN_|DNNL_)' } | Remove-Item
$overrides = @{
    AILA_Q4E_CPU_OVERLAP = '0'; AILA_Q4E_STATIC_PINNING = '0'
    AILA_Q4E_PREFILL_XMX = '1'; AILA_Q4E_DIRECT_PREFILL_IO = '1'
    AILA_Q4E_INLINE_CHECKPOINT = '0'
}
if ($Reader -ne 'default') { $overrides.AILA_Q4E_EXPERT_READER = $Reader }
foreach ($key in $overrides.Keys) { Set-Item "env:$key" $overrides[$key] }
$exe = Join-Path (Join-Path $repo $BuildDir) 'AilaQwen4ExpToleranceTests.exe'
$prefix = Join-Path $repo "tmp/$Tag"
$metadata = [ordered]@{
    timestamp = (Get-Date -Format o); commit = (& git -C $repo rev-parse HEAD)
    dirty = @(& git -C $repo status --short)
    exe_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    harness_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tests/models/Qwen4ExpCheckpointAcceptance.hpp')).Hash
    model = $ModelDir; stack = (Get-AilaOneApiStackMetadata -Stack $stack)
    mode = $Mode; reader_override = $Reader
    gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
    env_overrides = $overrides; inherited_experiment_overrides_cleared = $true
    prompt_tokens = $(if ($Mode -eq 'checkpoint') { @(30, 256) } else { @(256, 256) })
    teacher_forced_targets = $(if ($Mode -eq 'checkpoint') { 48 } else { 0 })
    max_seq = 512
    cache = 'Init warmup; GPU cache capacity limited. Checkpoint mode also warms both paths; memory mode retains cold first long shape.'
    timing = $(if ($Mode -eq 'checkpoint') {
        '3 alternating pairs per XMX setting; prefill+47 fixed-token forwards and separate Engine generation; timing excludes logits download'
    } else { 'Two 256-token forwards; includes logits download/hash; diagnostic timing, no speedup claim' })
}
$metadata | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
$process = Start-Process -FilePath $exe -ArgumentList @("--$Mode-model", ('"' + $ModelDir + '"')) `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$prefix.log" -RedirectStandardError "$prefix.err.log"
$samples = [System.Collections.Generic.List[object]]::new()
$memoryPressure = $false
$timer = [System.Diagnostics.Stopwatch]::StartNew()
while (!$process.WaitForExit(500) -and $timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
    $process.Refresh()
    $freeMB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024
    $samples.Add([pscustomobject]@{ ms = $timer.ElapsedMilliseconds; working_set_bytes = $process.WorkingSet64
        private_commit_bytes = $process.PrivateMemorySize64; peak_working_set_bytes = $process.PeakWorkingSet64; system_free_mb = $freeMB })
    if ($freeMB -lt 8192) { $memoryPressure = $true; break }
}
$samples | ConvertTo-Json -Depth 3 | Set-Content "$prefix.memory.json"
$metadata.peak_working_set_bytes = ($samples | Measure-Object working_set_bytes -Maximum).Maximum
$metadata.peak_private_commit_bytes = ($samples | Measure-Object private_commit_bytes -Maximum).Maximum
$metadata.minimum_system_free_mb = ($samples | Measure-Object system_free_mb -Minimum).Minimum
if (!$process.HasExited) {
    $process.Kill($true)
    $process.WaitForExit()
    $metadata.memory_pressure_stop = $memoryPressure
    $metadata | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
    throw "X2 watchdog stopped the process (memory_pressure=$memoryPressure, timeout=$TimeoutSeconds s); see $prefix.log"
}
$process.WaitForExit()
$metadata.exit_code = $process.ExitCode
$metadata | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
Get-Content "$prefix.log" | Select-String '\[X2|\[FAIL'
Get-Content "$prefix.err.log" | Select-Object -Last 10
exit $process.ExitCode
