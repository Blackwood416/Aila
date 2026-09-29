param(
    [ValidateSet('kernel','baseline','bench','model','profile')][string]$Mode='kernel',
    [string]$Tag='x4-gdn',
    [switch]$Candidate,
    [string]$ModelDir='C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [int]$TimeoutSeconds=600
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../PerfCommon.ps1')
$repo=Get-AilaRepoRoot
$cfg=Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack=Get-AilaOneApiStack -Config $cfg -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)
Get-ChildItem Env: | Where-Object { $_.Name -match '^(AILA_|ONEDNN_|DNNL_)' } | Remove-Item
$overrides=@{
    AILA_Q4E_CPU_EXPERT_DISPATCH='0'; AILA_Q4E_CPU_OVERLAP='0'; AILA_Q4E_GROUPED_XMX_MIN_ROWS='0'
    AILA_Q4E_PREFILL_XMX='1'; AILA_Q4E_DIRECT_PREFILL_IO='1'; AILA_Q4E_INLINE_CHECKPOINT='0'
    AILA_Q4E_GDN_CHUNK_MODE='0'
}
foreach($key in $overrides.Keys) { Set-Item "env:$key" $overrides[$key] }
$exe=Join-Path $repo 'build-refactor/AilaGdnChunkTests.exe'
$arguments=@('--check')
if($Mode -eq 'baseline' -or $Mode -eq 'profile') {$arguments=@('--baseline-bench')}
if($Mode -eq 'profile' -and $Candidate) {$arguments=@('--profile-chunk')}
if($Mode -eq 'bench') {$arguments=@('--bench')}
if($Mode -eq 'model') {
    $exe=Join-Path $repo 'build-refactor/AilaQwen4ExpToleranceTests.exe'
    $arguments=@('--gdn-chunk-model',('"'+$ModelDir+'"'))
}
$prefix=Join-Path $repo "tmp/$Tag"
$meta=[ordered]@{
    timestamp=(Get-Date -Format o); commit=(& git -C $repo rev-parse HEAD); dirty=@(& git -C $repo status --short)
    mode=$Mode; model=$ModelDir; exe_sha256=(Get-FileHash -LiteralPath $exe).Hash
    stack=(Get-AilaOneApiStackMetadata -Stack $stack)
    gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
    env_overrides=$overrides; kernel_test_sha256=(Get-FileHash (Join-Path $repo 'tests/ops/GdnChunkTests.cpp')).Hash
    timer='Host wall; optional GPU event span includes queue gaps; reset/copies excluded; model timing excludes logits download'
}
if(Test-Path (Join-Path $repo 'src/ops/GdnChunkOps.cpp')) {$meta.kernel_sha256=(Get-FileHash (Join-Path $repo 'src/ops/GdnChunkOps.cpp')).Hash}
$meta | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
$launcher=$exe
if($Mode -eq 'profile') {
    $launcher='C:/Program Files (x86)/Intel/oneAPI/vtune/latest/bin64/vtune.exe'
    $arguments=@('-collect','gpu-hotspots','-knob','characterization-mode=instruction-count','-result-dir',('"'+$prefix+'-vtune"'),'--',('"'+$exe+'"'))+$arguments
}
$process=Start-Process -FilePath $launcher -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$prefix.log" -RedirectStandardError "$prefix.err.log"
$samples=[System.Collections.Generic.List[object]]::new()
$timer=[System.Diagnostics.Stopwatch]::StartNew()
$pressure=$false
while(!$process.WaitForExit(500) -and $timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
    $process.Refresh(); $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024
    $samples.Add([pscustomobject]@{ms=$timer.ElapsedMilliseconds; working_set_bytes=$process.WorkingSet64; private_commit_bytes=$process.PrivateMemorySize64; system_free_mb=$free})
    if($free -lt 8192) {$pressure=$true; break}
}
$samples | ConvertTo-Json -Depth 3 | Set-Content "$prefix.memory.json"
$meta.peak_working_set_bytes=($samples | Measure-Object working_set_bytes -Maximum).Maximum
$meta.peak_private_commit_bytes=($samples | Measure-Object private_commit_bytes -Maximum).Maximum
$meta.minimum_system_free_mb=($samples | Measure-Object system_free_mb -Minimum).Minimum
if(!$process.HasExited) {
    $process.Kill($true); $process.WaitForExit(); $meta.watchdog_stop=$true; $meta.memory_pressure=$pressure
    $meta | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
    throw "X4 watchdog stopped test; memory_pressure=$pressure"
}
$process.WaitForExit(); $meta.exit_code=$process.ExitCode
$meta | ConvertTo-Json -Depth 6 | Set-Content "$prefix.meta.json"
Get-Content "$prefix.log" | Select-String -SimpleMatch '[X4'
Get-Content "$prefix.err.log" | Select-Object -Last 10
exit $process.ExitCode
