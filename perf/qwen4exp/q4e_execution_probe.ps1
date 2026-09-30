param(
    [ValidateSet("story","memory","supply","front_graph","decode","projection","hc","mtp")][string]$Workload="story",
    [ValidateSet(30,512)][int]$DecodePromptTokens=30,
    [ValidateSet(1,2,4)][int]$VerifyRows=1,
    [ValidateRange(2,8)][int]$DecodeRepeats=4,
    [ValidateRange(1,60)][int]$MtpAnchors=60,
    [ValidateSet(64,512)][int]$MtpSlots=512,
    [ValidateRange(1,4)][int]$MtpRepeats=3,
    [ValidateRange(0,4)][int]$MtpVerifyPasses=3,
    [switch]$NoPrefetch,
    [switch]$Profile,
    [switch]$Stages,
    [switch]$SmallRowsDecode,
    [switch]$SmallRowsHc,
    [switch]$Smoke,
    [switch]$Sliced,
    [string]$Tag="p0-story",
    [int]$MaxTokens=48,
    [ValidateRange(2,8)][int]$MemoryRepeats=2,
    [ValidateSet(256,512,1024)][int]$MemoryTokens=256,
    [ValidateRange(0,64)][int]$FillParts=0,
    [ValidateRange(0,512)][int]$DemandMicrobatch=0,
    [switch]$Trace,
    [switch]$Vtune,
    [int]$TimeoutSeconds=240,
    [string]$ModelDir="C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2"
)
$ErrorActionPreference="Stop"
. (Join-Path $PSScriptRoot "../PerfCommon.ps1")
$repo=Get-AilaRepoRoot
$cfg=Get-AilaOneApiStackConfig -RepoRoot $repo -Path "perf/oneapi-stacks.json"
$stack=Get-AilaOneApiStack -Config $cfg -Name "oneapi-2026.1"
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)
Get-ChildItem Env: | Where-Object { $_.Name -match "^(AILA_|ONEDNN_|DNNL_)" } | Remove-Item
$prefix=Join-Path $repo "tmp/$Tag"
$overrides=@{
    AILA_Q4E_CPU_EXPERT_DISPATCH="0"; AILA_Q4E_PREFILL_XMX="1"
    AILA_Q4E_EXPERT_READER="readfile"; AILA_PROFILE_Q4E_DECODE="1"
    AILA_Q4E_STATIC_PINNING="0"; AILA_Q4E_GDN_CHUNK_MODE="0"
    AILA_Q4E_GROUPED_XMX_MIN_ROWS="0"
}
if($Trace) {$overrides.AILA_Q4E_CACHE_TRACE="$prefix.cache.jsonl"}
if($Vtune) {
    $overrides.AILA_Q4E_CACHE_TRACE="$prefix.cache.jsonl"
    $overrides.AILA_Q4E_TRACE_MARKERS="1"
}
if($FillParts) {$overrides.AILA_Q4E_FILL_PARTS=[string]$FillParts}
if($DemandMicrobatch) {$overrides.AILA_Q4E_DEMAND_MICROBATCH=[string]$DemandMicrobatch}
if($Workload -eq "memory") {$overrides.AILA_Q4E_MEMORY_CANARY_REPEATS=[string]$MemoryRepeats}
if($Workload -eq "memory") {$overrides.AILA_Q4E_MEMORY_CANARY_TOKENS=[string]$MemoryTokens}
foreach($key in $overrides.Keys) {Set-Item "env:$key" $overrides[$key]}
$exe=Join-Path $repo "build-refactor/Aila.exe"
$prompt=Join-Path $PSScriptRoot "prompts/story.json"
$arguments=@("-m",$ModelDir,"--max-seq","512","--greedy","--thinking-budget","0","--max-tokens",[string]$MaxTokens,"--messages-json",$prompt)
$inputIdentity=@{kind="messages_json";sha256=(Get-FileHash $prompt).Hash}
if($Workload -eq "memory") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--memory-model",$ModelDir)
    $inputIdentity=@{kind="generated_lighthouse";tokens=$MemoryTokens;max_seq=[Math]::Max(512,$MemoryTokens);repeats=$MemoryRepeats;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpCheckpointAcceptance.hpp")).Hash}
}
if($Workload -eq "supply") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpPrefillSupplyBench.exe"
    $arguments=@($(if($Smoke){"--smoke"}elseif($Sliced){"--sliced"}else{"--bench"}))
    $inputIdentity=@{kind="synthetic_nf4_supply_256_experts_8_rows";source_sha256=(Get-FileHash (Join-Path $repo "perf/qwen4exp/q4e_prefill_supply_bench.cpp")).Hash}
}
if($Workload -eq "front_graph") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--front-graph-model",$ModelDir)
    $inputIdentity=@{kind="real_layer0_decode_front";source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpFrontGraphProbe.hpp")).Hash}
}
if($Workload -eq "mtp") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--mtp-model",$ModelDir)
    $overrides.AILA_Q4E_MTP_PROMPT=[string]$DecodePromptTokens
    $overrides.AILA_Q4E_MTP_ANCHORS=[string]$MtpAnchors
    $overrides.AILA_Q4E_MTP_SLOTS=[string]$MtpSlots
    $overrides.AILA_Q4E_MTP_REPEATS=[string]$MtpRepeats
    $overrides.AILA_Q4E_MTP_VERIFY_PASSES=[string]$MtpVerifyPasses
    $overrides.AILA_Q4E_MTP_OUTPUT=$prefix
    $overrides.AILA_Q4E_VRAM_RESERVE_MB=$(if($MtpSlots -eq 64){"3008"}else{"4096"})
    $overrides.AILA_Q4E_SMALL_ROWS_DECODE="1"
    $overrides.AILA_Q4E_SMALL_ROWS_HC="1"
    $overrides.AILA_PROFILE_Q4E_DECODE="0"
    $inputIdentity=@{kind="mtp_greedy_conditional_three_draft_probe";prompt_tokens=$DecodePromptTokens;anchors=$MtpAnchors;repeats=$MtpRepeats;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpMtpAcceptance.hpp")).Hash}
}
if($Workload -eq "hc") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpKernelTests.exe"
    $arguments=@("--hc-only")
    $overrides.ONEDNN_VERBOSE="profile,dispatch"
    $inputIdentity=@{kind="bf16_hc_rows_cpu_b1_onednn_guards";source_sha256=(Get-FileHash (Join-Path $repo "tests/ops/Qwen4ExpHcRowsTests.hpp")).Hash}
}
if($Workload -eq "projection") {
    $exe=Join-Path $repo "build-refactor/AilaDecodeRowProjectionTests.exe"
    $arguments=@("--run")
    $inputIdentity=@{kind="nf4_decode_row_projection_cpu_and_b1_oracle";source_sha256=(Get-FileHash (Join-Path $repo "tests/inference/DecodeRowProjectionTests.cpp")).Hash}
}
if($Workload -eq "decode") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--decode-model",$ModelDir)
    $overrides.AILA_Q4E_DECODE_PROBE_PROMPT=[string]$DecodePromptTokens
    $overrides.AILA_Q4E_DECODE_PROBE_ROWS=[string]$VerifyRows
    $overrides.AILA_Q4E_DECODE_PROBE_REPEATS=[string]$DecodeRepeats
    $overrides.AILA_Q4E_DECODE_PROBE_LOGITS="$prefix.logits.bf16"
    $overrides.AILA_PROFILE_Q4E_DECODE=$(if($Profile){"1"}else{"0"})
    $inputIdentity=@{kind="fixed_teacher_forcing";prompt_tokens=$DecodePromptTokens;continuation_tokens=128;verify_rows=$VerifyRows;repeats=$DecodeRepeats;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpDecodeProbe.hpp")).Hash}
}
if($NoPrefetch) {$overrides.AILA_Q4E_PREFETCH="0"}
if($SmallRowsDecode) {$overrides.AILA_Q4E_SMALL_ROWS_DECODE="1"}
if($SmallRowsHc) {$overrides.AILA_Q4E_SMALL_ROWS_HC="1"}
if($Stages) {$overrides.AILA_PROFILE_Q4E_DECODE="1"; $overrides.AILA_PROFILE_Q4E_STAGES="1"}
foreach($key in $overrides.Keys) {Set-Item "env:$key" $overrides[$key]}
$meta=[ordered]@{
    timestamp=(Get-Date -Format o);commit=(& git -C $repo rev-parse HEAD);dirty=@(& git -C $repo status --short)
    workload=$Workload;input=$inputIdentity;arguments=$arguments;env_overrides=$overrides
    exe_sha256=(Get-FileHash $exe).Hash;model_config_sha256=(Get-FileHash "$ModelDir/config.json").Hash
    stack=(Get-AilaOneApiStackMetadata -Stack $stack)
    gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
    diagnostic_only=[bool]($Trace -or $Vtune -or $Profile -or $Stages -or $Workload -in @("supply","front_graph","decode","projection","hc","mtp"));watchdog_stop=$false
}
$launcher=$exe
if($Vtune) {
    $launcher="C:/Program Files (x86)/Intel/oneAPI/vtune/latest/bin64/vtune.exe"
    $arguments=@("-collect","xpu-offload","-return-app-exitcode","-result-dir",($prefix+"-vtune"),"--",$exe)+$arguments
}
$meta | ConvertTo-Json -Depth 8 | Set-Content "$prefix.meta.json"
# All arguments are paths, numeric values or fixed switches; quote paths with spaces.
$quoted=@($arguments | ForEach-Object { '"' + ([string]$_).Replace('"','\"') + '"' })
$proc=Start-Process -FilePath $launcher -ArgumentList $quoted -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$prefix.log" -RedirectStandardError "$prefix.err.log"
$samples=[System.Collections.Generic.List[object]]::new()
$timer=[System.Diagnostics.Stopwatch]::StartNew()
try {
    while(!$proc.WaitForExit(500) -and $timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $proc.Refresh();$free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024
        # Immediate children only; under VTune this is NOT the whole process tree.
        $children=@(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" | ForEach-Object {Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue})
        $childWs=0;foreach($child in $children){$childWs+=$child.WorkingSet64}
        $samples.Add([pscustomobject]@{ms=$timer.ElapsedMilliseconds;working_set_bytes=$proc.WorkingSet64;child_working_set_bytes=$childWs;system_free_mb=$free})
        if($free -lt 8192) {$meta.memory_pressure=$true;break}
    }
} finally {
    if(!$proc.HasExited) {$proc.Kill($true);$meta.watchdog_stop=$true}
    $proc.WaitForExit()
}
$meta.exit_code=$proc.ExitCode
if($samples.Count) {
    $meta.peak_working_set_bytes=($samples | Measure-Object working_set_bytes -Maximum).Maximum
    $meta.peak_child_working_set_bytes=($samples | Measure-Object child_working_set_bytes -Maximum).Maximum
    $meta.minimum_system_free_mb=($samples | Measure-Object system_free_mb -Minimum).Minimum
}
$meta.log_sha256=(Get-FileHash "$prefix.log").Hash
$meta.err_log_sha256=(Get-FileHash "$prefix.err.log").Hash
if(($Trace -or $Vtune) -and (Test-Path "$prefix.cache.jsonl")) {$meta.trace_sha256=(Get-FileHash "$prefix.cache.jsonl").Hash}
$meta | ConvertTo-Json -Depth 8 | Set-Content "$prefix.meta.json"
$samples | ConvertTo-Json -Depth 3 | Set-Content "$prefix.memory.json"
Get-Content "$prefix.log" | Select-Object -Last 6
Get-Content "$prefix.err.log" | Select-Object -Last 4
if($meta.watchdog_stop) {throw "Execution probe watchdog stopped the process"}
exit $proc.ExitCode
