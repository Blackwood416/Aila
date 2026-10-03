param(
    [ValidateSet("story","generation","resident","preset","window","mature","integer_quality","integer_state","dpas","integer","compact","tile","memory","supply","front_graph","decode","projection","hc","qsa","qsa_tests","qsa_batches","qsa_profile","mtp")][string]$Workload="story",
    [ValidateSet("","qwen38-a770")][string]$PerformancePreset="",
    [switch]$RuntimeDefaults,
    [switch]$IntegerDeviceProfile,
    [ValidateSet(0,1,2)][int]$ResidentDecode=0,
    [ValidateSet(0,1,2)][int]$ColdMode=0,
    [string]$HotsetProfile="",
    [string]$HotsetExport="",
    [ValidateRange(0,1024)][int]$HotsetInterval=4,
    [ValidateRange(0,512)][int]$HotsetSwaps=96,
    [int]$HotsetSlots=0,
    [switch]$HotsetBorrow,
    [switch]$HotsetPrompt,
    [switch]$HotsetPromptPrefill,
    [switch]$HotsetAdoptReady,
    [ValidateRange(0,4096)][int]$HotsetAdoptLimit=96,
    [ValidateRange(0,4096)][int]$HotsetPromptSwaps=1024,
    [ValidateSet(-1,0,1)][int]$HotsetRing=-1,
    [ValidateSet(-1,0,1,2)][int]$HotsetColdQueue=-1,
    [ValidateSet(0,1,2,3)][int]$ColdPolicy=0,
    [ValidateRange(0,256)][int]$ColdPcieNum=141,
    [switch]$ResidentPrefill,
    [switch]$ColdBench,
    [switch]$ColdQuality,
    [switch]$ColdDiagnostic,
    [switch]$ColdPipelineDiagnostic,
    [switch]$ColdLowRank,
    [switch]$ColdDeviceProfile,
    [switch]$ReverseCold,
    [ValidateSet(-1,0,1)][int]$ColdQualityH3=-1,
    [ValidateRange(1,12)][int]$ColdThreads=4,
    [ValidateRange(1,10)][int]$ColdCredits=2,
    [ValidateRange(100,10000)][int]$ColdBudgetUs=1000,
    [string]$Executable="",
    [ValidateSet("causal","checkpoint")][string]$IntegerStatePart="causal",
    [ValidateSet(0,256,512,1024)][int]$CompactMoe=0,
    [ValidateSet(0,128,256,512)][int]$AttentionTile=0,
    [switch]$WorkSummary,
    [ValidateSet(-1,0,1)][int]$InlineCheckpoint=-1,
    [ValidateSet(-1,0,1,2,3,4)][int]$QsaSelector=-1,
    [ValidateSet(-1,0,1)][int]$ImmediateLists=-1,
    [ValidateSet(30,512)][int]$DecodePromptTokens=30,
    [ValidateSet(1,2,4)][int]$VerifyRows=1,
    [ValidateRange(2,8)][int]$DecodeRepeats=4,
    [ValidateRange(1,60)][int]$MtpAnchors=60,
    [ValidateSet(64,512)][int]$MtpSlots=64,
    [ValidateRange(1,4)][int]$MtpRepeats=3,
    [ValidateRange(0,4)][int]$MtpVerifyPasses=3,
    [switch]$MtpCompareKv,
    [switch]$MtpWindowSweep,
    [switch]$MtpCompareResident,
    [string]$MtpPromptFile="",
    [switch]$NoPrefetch,
    [switch]$Profile,
    [switch]$Stages,
    [switch]$SmallRowsDecode,
    [switch]$SmallRowsHc,
    [switch]$Smoke,
    [switch]$Sliced,
    [string]$Tag="p0-story",
    [int]$MaxTokens=48,
    [ValidateRange(1,8)][int]$GenerationRepeats=4,
    [string]$GenerationPromptFile="",
    [string[]]$GenerationPromptFiles=@(),
    [ValidateRange(512,4096)][int]$GenerationMaxSeq=512,
    [ValidateRange(2,8)][int]$MemoryRepeats=2,
    [ValidateSet(256,512,1024,2052)][int]$MemoryTokens=256,
    [ValidateSet(0,8)][int]$MemoryDecodeTail=0,
    [ValidateSet(0,256,512,1024,2048)][int]$PrefillChunk=0,
    [switch]$PhaseScratch,
    [ValidateSet(16384,24576,32768)][int]$HostRamMB=32768,
    [ValidateRange(0,64)][int]$FillParts=0,
    [ValidateRange(0,512)][int]$DemandMicrobatch=0,
    [switch]$Trace,
    [switch]$Vtune,
    [switch]$VtuneInstructions,
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
if($ColdDeviceProfile) {$Trace=$true}
$overrides=@{
    AILA_Q4E_CPU_EXPERT_DISPATCH="0"; AILA_Q4E_PREFILL_XMX="1"
    AILA_Q4E_EXPERT_READER="readfile"; AILA_PROFILE_Q4E_DECODE=$(if($Profile){"1"}else{"0"})
    AILA_Q4E_HOT_RAM_MB=[string]$HostRamMB
    AILA_Q4E_STATIC_PINNING="0"; AILA_Q4E_GDN_CHUNK_MODE="0"
    AILA_Q4E_GROUPED_XMX_MIN_ROWS="0"
}
if($RuntimeDefaults) {
    $overrides=@{AILA_Q4E_CPU_EXPERT_DISPATCH="0"}
    if($PSBoundParameters.ContainsKey('HostRamMB')) {$overrides.AILA_Q4E_HOT_RAM_MB=[string]$HostRamMB}
}
if($InlineCheckpoint -ge 0) {$overrides.AILA_Q4E_INLINE_CHECKPOINT=[string]$InlineCheckpoint}
if($WorkSummary) {$overrides.AILA_Q4E_WORK_SUMMARY="1"}
if($PSBoundParameters.ContainsKey('ResidentDecode')) {$overrides.AILA_Q4E_RESIDENT_DECODE=[string]$ResidentDecode}
if($ColdMode -or $ColdBench -or $ColdQuality) {
    $overrides.AILA_Q4E_COLD_MODE=[string]$(if($ColdMode){$ColdMode}else{1})
    $overrides.AILA_Q4E_CPU_THREADS=[string]$ColdThreads
    $overrides.AILA_Q4E_CPU_MAX_EXPERTS=[string]$ColdCredits
    $overrides.AILA_Q4E_COLD_BUDGET_US=[string]$ColdBudgetUs
}
if($HotsetProfile) {$overrides.AILA_Q4E_HOTSET_PROFILE=(Resolve-Path -LiteralPath $HotsetProfile).Path}
if($HotsetExport) {$overrides.AILA_Q4E_HOTSET_EXPORT=$HotsetExport}
if($HotsetProfile) {
    $overrides.AILA_Q4E_HOTSET_INTERVAL=[string]$HotsetInterval
    $overrides.AILA_Q4E_HOTSET_SWAPS=[string]$HotsetSwaps
    $overrides.AILA_Q4E_HOTSET_BORROW=[string][int][bool]$HotsetBorrow
    $overrides.AILA_Q4E_HOTSET_PROMPT=[string][int][bool]$HotsetPrompt
    $overrides.AILA_Q4E_HOTSET_PROMPT_PREFILL=[string][int][bool]$HotsetPromptPrefill
    $overrides.AILA_Q4E_HOTSET_ADOPT_READY=[string][int][bool]$HotsetAdoptReady
    $overrides.AILA_Q4E_HOTSET_ADOPT_LIMIT=[string]$HotsetAdoptLimit
    $overrides.AILA_Q4E_HOTSET_PROMPT_SWAPS=[string]$HotsetPromptSwaps
    if($HotsetRing -ge 0) {$overrides.AILA_Q4E_HOTSET_RING=[string]$HotsetRing}
    if($HotsetColdQueue -ge 0) {$overrides.AILA_Q4E_HOTSET_COLD_QUEUE=[string]$HotsetColdQueue}
    if($HotsetSlots) {$overrides.AILA_Q4E_HOTSET_SLOTS=[string]$HotsetSlots}
}
$overrides.AILA_Q4E_COLD_POLICY=[string]$ColdPolicy
$overrides.AILA_Q4E_COLD_PCIE_NUM=[string]$ColdPcieNum
if($ResidentPrefill) {$overrides.AILA_Q4E_RESIDENT_PREFILL="1"}
if($ColdLowRank) {$overrides.AILA_Q4E_COLD_LOW_RANK="1"}
if($ColdDiagnostic) {$overrides.AILA_Q4E_COLD_DIAGNOSTIC="1"}
if($ColdPipelineDiagnostic) {$overrides.AILA_Q4E_COLD_PIPELINE_DIAGNOSTIC="1"}
if($ColdDeviceProfile) {$overrides.AILA_PROFILE_Q4E_COLD_DEVICE="1"}
if($AttentionTile) {$overrides.AILA_Q4E_PREFILL_ATTN_TILE=[string]$AttentionTile}
if($CompactMoe) {$overrides.AILA_Q4E_MOE_COMPACT_ACTS=[string]$CompactMoe}
if($Trace) {$overrides.AILA_Q4E_CACHE_TRACE="$prefix.cache.jsonl"}
if($QsaSelector -ge 0) {$overrides.AILA_QSA_SELECTOR=[string]$QsaSelector}
if($ImmediateLists -ge 0) {
    $overrides.UR_L0_USE_IMMEDIATE_COMMANDLISTS=[string]$ImmediateLists
    $overrides.SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=[string]$ImmediateLists
}
if($Vtune -or $VtuneInstructions) {
    $overrides.AILA_Q4E_CACHE_TRACE="$prefix.cache.jsonl"
    $overrides.AILA_Q4E_TRACE_MARKERS="1"
}
if($FillParts) {$overrides.AILA_Q4E_FILL_PARTS=[string]$FillParts}
if($DemandMicrobatch) {$overrides.AILA_Q4E_DEMAND_MICROBATCH=[string]$DemandMicrobatch}
if($PrefillChunk) {$overrides.AILA_Q35_PREFILL_CHUNK=[string]$PrefillChunk}
if($PhaseScratch) {$overrides.AILA_Q4E_PHASE_SCRATCH="1"}
if($Workload -eq "memory") {$overrides.AILA_Q4E_MEMORY_CANARY_REPEATS=[string]$MemoryRepeats}
if($Workload -eq "memory") {$overrides.AILA_Q4E_MEMORY_CANARY_TOKENS=[string]$MemoryTokens}
if($Workload -eq "memory") {$overrides.AILA_Q4E_MEMORY_CANARY_TAIL=[string]$MemoryDecodeTail}
foreach($key in $overrides.Keys) {Set-Item "env:$key" $overrides[$key]}
$exe=Join-Path $repo "build-refactor/Aila.exe"
$prompt=Join-Path $PSScriptRoot "prompts/story.json"
$arguments=@("-m",$ModelDir,"--max-seq","512","--greedy","--thinking-budget","0","--max-tokens",[string]$MaxTokens,"--messages-json",$prompt)
$inputIdentity=@{kind="messages_json";sha256=(Get-FileHash $prompt).Hash}
$stdinPath=$null
if($Workload -eq "generation") {
    if($GenerationPromptFile) {$prompt=(Resolve-Path -LiteralPath $GenerationPromptFile).Path}
    $messages=@(Get-Content -Raw -LiteralPath $prompt | ConvertFrom-Json)
    if($messages.Count -ne 1 -or $messages[0].role -ne "user") {throw "Generation probe requires exactly one user message"}
    $text=$messages[0].content
    # Interactive CLI accepts one message per line; preserve word boundaries.
    $text=$text -replace '[\r\n]+',' '
    $stdinPath="$prefix.stdin.txt"
    $commands=for($i=0;$i -lt $GenerationRepeats;$i++) {
        "/clear"
        $text
    }
    if($GenerationPromptFiles.Count) {
        $commands=foreach($source in $GenerationPromptFiles) {
            $trainingMessages=@(Get-Content -Raw -LiteralPath $source | ConvertFrom-Json)
            if($trainingMessages.Count -ne 1 -or $trainingMessages[0].role -ne 'user') {throw 'Calibration requires single user messages'}
            for($i=0;$i -lt $GenerationRepeats;$i++) {
                '/clear'
                ($trainingMessages[0].content -replace '[\r\n]+',' ')
            }
        }
    }
    @($commands)+@("/quit") | Set-Content -LiteralPath $stdinPath -Encoding utf8NoBOM
    $arguments=@("-m",$ModelDir,"--max-seq",[string]$GenerationMaxSeq,"--greedy","--thinking-budget","0","--max-tokens",[string]$MaxTokens,"--stream","--stream-chunk","1","--decode-chunk","1")
    $inputIdentity=@{kind="cli_stream_generation";sha256=(Get-FileHash $prompt).Hash;stdin_sha256=(Get-FileHash $stdinPath).Hash;repeats=$GenerationRepeats;max_tokens=$MaxTokens;source_sha256=(Get-FileHash (Join-Path $repo "src/cli/CLI.cpp")).Hash}
    if($GenerationMaxSeq -ne 512) {$inputIdentity.max_seq=$GenerationMaxSeq}
    if($GenerationPromptFiles.Count) {
        $inputIdentity.Remove('sha256')
        $inputIdentity.kind='cli_stream_calibration_corpus'
        $inputIdentity.requests=$GenerationRepeats*$GenerationPromptFiles.Count
        $inputIdentity.corpus=@($GenerationPromptFiles | ForEach-Object {
        @{path=(Resolve-Path -LiteralPath $_).Path;sha256=(Get-FileHash -LiteralPath $_).Hash}
    })}
}
if($Workload -eq "preset") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--preset-model",$ModelDir)
    $inputIdentity=@{kind="cli41_checkpoint_teacher_forcing";source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpPresetAcceptance.hpp")).Hash}
}
if($Workload -eq "resident") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--resident-model",$ModelDir)
    $residentPrompt=if($GenerationPromptFile){(Resolve-Path -LiteralPath $GenerationPromptFile).Path}else{Join-Path $PSScriptRoot "prompts/story.json"}
    $overrides.AILA_Q4E_RESIDENT_PROMPT_FILE=$residentPrompt
    $overrides.AILA_Q4E_RESIDENT_DECODE="2"
    $inputIdentity=@{kind="resident_nf4_state_quality";prompt_sha256=(Get-FileHash $residentPrompt).Hash;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpResidentAcceptance.hpp")).Hash}
}
if($ColdBench -or $ColdQuality) {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@($(if($ColdBench){"--cold-bench"}else{"--cold-model"}),$ModelDir)
    $coldPrompt=if($GenerationPromptFile){(Resolve-Path -LiteralPath $GenerationPromptFile).Path}else{Join-Path $PSScriptRoot "prompts/story.json"}
    $overrides.AILA_Q4E_RESIDENT_PROMPT_FILE=$coldPrompt
    $overrides.AILA_Q4E_COLD_BENCH_H3=[string][int][bool]$ResidentPrefill
    $overrides.AILA_Q4E_COLD_BENCH_REVERSE=[string][int][bool]$ReverseCold
    $overrides.AILA_Q4E_COLD_TEST_H3=[string]$ColdQualityH3
    $inputIdentity=@{kind=$(if($ColdBench){"frozen_cold_generation"}else{"frozen_cold_quality"});prompt_sha256=(Get-FileHash $coldPrompt).Hash;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpColdAcceptance.hpp")).Hash}
}
if($Workload -eq "window") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--window-model",$ModelDir)
    $windowPrompt=if($GenerationPromptFile){(Resolve-Path -LiteralPath $GenerationPromptFile).Path}else{Join-Path $PSScriptRoot "prompts/archive-natural.json"}
    $overrides.AILA_Q4E_WINDOW_PROMPT_FILE=$windowPrompt
    $inputIdentity=@{kind="natural_window_teacher_forcing";prompt_sha256=(Get-FileHash $windowPrompt).Hash;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpPresetAcceptance.hpp")).Hash}
}
if($Workload -eq "tile") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--tile-model",$ModelDir)
    $overrides.AILA_Q4E_WINDOW_PROMPT_FILE=Join-Path $PSScriptRoot "prompts/archive-natural.json"
    $inputIdentity=@{kind="attention_tile_state_and_quality";source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpAttentionTileAcceptance.hpp")).Hash}
}
if($Workload -in @("integer_quality","integer_state")) {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@($(if($Workload -eq "integer_state"){"--integer-state"}else{"--integer-quality"}),$ModelDir)
    $qualityPrompt=if($GenerationPromptFile){(Resolve-Path -LiteralPath $GenerationPromptFile).Path}else{Join-Path $PSScriptRoot "prompts/archive-natural.json"}
    $overrides.AILA_Q4E_WINDOW_PROMPT_FILE=$qualityPrompt
    $overrides.AILA_Q4E_INTEGER_QUALITY_PREFIX=$prefix
    $inputIdentity=@{kind="real_hidden_integer_quality";prompt_sha256=(Get-FileHash $qualityPrompt).Hash;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpIntegerQualityAcceptance.hpp")).Hash}
    if($Workload -eq "integer_state") {
        $overrides.AILA_Q4E_INTEGER_STATE_PART=$IntegerStatePart
        $inputIdentity.kind="integer_prefill_state"
        $inputIdentity.state_part=$IntegerStatePart
    }
}
if($Workload -eq "dpas") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=if($Smoke){@("--dpas-smoke")}else{@("--dpas-model",$ModelDir)}
    $overrides.AILA_Q4E_HOT_RAM_MB="0"
    $overrides.AILA_Q4E_MATURE_PROFILE=[string][int][bool]$VtuneInstructions
    $overrides.AILA_Q4E_INTEGER_DEVICE_PROFILE=[string][int][bool]$IntegerDeviceProfile
    $inputIdentity=@{kind="nf4_q8_block64_dpas";smoke=[bool]$Smoke;host_pool_disabled=$true;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpDpasMoeProbe.hpp")).Hash;fixture_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpIntegerMoeProbe.hpp")).Hash}
}
if($Workload -eq "integer") {
    $overrides.AILA_Q4E_INTEGER_DEVICE_PROFILE=[string][int][bool]$IntegerDeviceProfile
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--integer-model",$ModelDir)
    $overrides.AILA_Q4E_HOT_RAM_MB="0"
    $overrides.AILA_Q4E_MATURE_PROFILE=[string][int][bool]$VtuneInstructions
    $inputIdentity=@{kind="real_nf4_integer_lut_q8_block64";host_pool_disabled=$true;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpIntegerMoeProbe.hpp")).Hash}
}
if($Workload -eq "compact") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--compact-model",$ModelDir)
    $overrides.AILA_Q4E_HOT_RAM_MB="0"
    $overrides.AILA_Q4E_MATURE_PROFILE=[string][int][bool]$VtuneInstructions
    $inputIdentity=@{kind="real_nf4_compact_down_q36";source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpCompactMoeProbe.hpp")).Hash}
}
if($Workload -eq "mature") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--mature-moe-model",$ModelDir)
    $overrides.AILA_Q4E_HOT_RAM_MB="0"
    $overrides.AILA_Q4E_MATURE_PROFILE=[string][int][bool]$VtuneInstructions
    $inputIdentity=@{kind="two_real_experts_m32_bounded_bf16_staging";host_pool_disabled=$true;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpMatureGemmProbe.hpp")).Hash}
}
if($Workload -eq "memory") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--memory-model",$ModelDir)
    $inputIdentity=@{kind="generated_lighthouse";tokens=$MemoryTokens;max_seq=[Math]::Max(512,$MemoryTokens+$MemoryDecodeTail);tail=$MemoryDecodeTail;repeats=$MemoryRepeats;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpCheckpointAcceptance.hpp")).Hash}
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
    $overrides.AILA_Q4E_MTP_COMPARE_KV=[string][int][bool]$MtpCompareKv
    $overrides.AILA_Q4E_MTP_WINDOW_SWEEP=[string][int][bool]$MtpWindowSweep
    $overrides.AILA_Q4E_MTP_COMPARE_RESIDENT=[string][int][bool]$MtpCompareResident
    if($MtpPromptFile) {$overrides.AILA_Q4E_MTP_PROMPT_FILE=(Resolve-Path -LiteralPath $MtpPromptFile).Path}
    $overrides.AILA_Q4E_MTP_OUTPUT=$prefix
    $overrides.AILA_Q4E_VRAM_RESERVE_MB=$(if($MtpSlots -eq 64){"3008"}else{"4096"})
    $overrides.AILA_Q4E_SMALL_ROWS_DECODE="1"
    $overrides.AILA_Q4E_SMALL_ROWS_HC="1"
    $overrides.AILA_PROFILE_Q4E_DECODE="0"
    $inputIdentity=@{kind="mtp_greedy_conditional_three_draft_probe";prompt_tokens=$DecodePromptTokens;anchors=$MtpAnchors;repeats=$MtpRepeats;source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpMtpAcceptance.hpp")).Hash}
    if($MtpPromptFile) {$inputIdentity.prompt_file=$overrides.AILA_Q4E_MTP_PROMPT_FILE;$inputIdentity.prompt_sha256=(Get-FileHash $MtpPromptFile).Hash}
}
if($Workload -eq "hc") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpKernelTests.exe"
    $arguments=@("--hc-only")
    $overrides.ONEDNN_VERBOSE="profile,dispatch"
    $inputIdentity=@{kind="bf16_hc_rows_cpu_b1_onednn_guards";source_sha256=(Get-FileHash (Join-Path $repo "tests/ops/Qwen4ExpHcRowsTests.hpp")).Hash}
}
if($Workload -in @("qsa","qsa_tests","qsa_batches","qsa_profile")) {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpKernelTests.exe"
    $arguments=@("--qsa-probe")
    if($Workload -eq "qsa_tests") {$arguments=@("--qsa-tests")}
    if($Workload -eq "qsa_batches") {$arguments=@("--qsa-batches")}
    if($Workload -eq "qsa_profile") {$arguments=@("--qsa-profile")}
    $inputIdentity=@{kind="fixed_bf16_qsa_decode_curve";source_sha256=(Get-FileHash (Join-Path $repo "tests/ops/Qwen4ExpQsaProbe.hpp")).Hash}
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
if($PerformancePreset -and !$ColdBench -and !$ColdQuality) {
    if($Workload -notin @("story","generation")) {throw "CLI preset requires story or generation"}
    $arguments+=@("--performance-preset",$PerformancePreset)
}
if($Executable) {$exe=(Resolve-Path -LiteralPath $Executable).Path}
$meta=[ordered]@{
    performance_preset=$PerformancePreset;runtime_defaults=[bool]$RuntimeDefaults
    timestamp=(Get-Date -Format o);commit=(& git -C $repo rev-parse HEAD);dirty=@(& git -C $repo status --short)
    workload=$Workload;input=$inputIdentity;arguments=$arguments;env_overrides=$overrides
    exe_sha256=(Get-FileHash $exe).Hash;model_config_sha256=(Get-FileHash "$ModelDir/config.json").Hash
    stack=(Get-AilaOneApiStackMetadata -Stack $stack)
    gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
    diagnostic_only=[bool]($Trace -or $Vtune -or $VtuneInstructions -or $Profile -or $Stages -or $Workload -in @("resident","preset","window","mature","integer_quality","integer_state","dpas","integer","compact","tile","supply","front_graph","decode","projection","hc","qsa","qsa_tests","qsa_batches","qsa_profile","mtp"));watchdog_stop=$false
}
$launcher=$exe
if($HotsetProfile) {$meta.hotset_profile=@{path=(Resolve-Path -LiteralPath $HotsetProfile).Path;sha256=(Get-FileHash -LiteralPath $HotsetProfile).Hash}}
if($ColdQuality) {$meta.diagnostic_only=$true;$meta.workload="cold_quality"}
if($ColdBench) {$meta.workload="cold_bench"}
if($Vtune -or $VtuneInstructions) {
    $launcher="C:/Program Files (x86)/Intel/oneAPI/vtune/latest/bin64/vtune.exe"
    $collection=if($VtuneInstructions){@("-collect","gpu-hotspots","-knob","characterization-mode=instruction-count")}else{@("-collect","xpu-offload")}
    $arguments=$collection+@("-return-app-exitcode","-result-dir",($prefix+"-vtune"),"--",$exe)+$arguments
}
$meta | ConvertTo-Json -Depth 8 | Set-Content "$prefix.meta.json"
# All arguments are paths, numeric values or fixed switches; quote paths with spaces.
$quoted=@($arguments | ForEach-Object { '"' + ([string]$_).Replace('"','\"') + '"' })
$launchOptions=@{}
if($stdinPath) {$launchOptions.RedirectStandardInput=$stdinPath}
$proc=Start-Process -FilePath $launcher -ArgumentList $quoted -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$prefix.log" -RedirectStandardError "$prefix.err.log" @launchOptions
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
if($HotsetExport -and (Test-Path -LiteralPath $HotsetExport)) {$meta.hotset_export=@{path=$HotsetExport;sha256=(Get-FileHash -LiteralPath $HotsetExport).Hash}}
if($samples.Count) {
    $meta.peak_working_set_bytes=($samples | Measure-Object working_set_bytes -Maximum).Maximum
    $meta.peak_child_working_set_bytes=($samples | Measure-Object child_working_set_bytes -Maximum).Maximum
    $meta.minimum_system_free_mb=($samples | Measure-Object system_free_mb -Minimum).Minimum
}
# Save process outcome before hashing: a profiler can still hold a log handle.
$meta | ConvertTo-Json -Depth 8 | Set-Content "$prefix.meta.json"
$meta.log_sha256=(Get-FileHash "$prefix.log").Hash
$meta.err_log_sha256=(Get-FileHash "$prefix.err.log").Hash
if(($Trace -or $Vtune -or $VtuneInstructions) -and (Test-Path "$prefix.cache.jsonl")) {$meta.trace_sha256=(Get-FileHash "$prefix.cache.jsonl").Hash}
if(($Trace -or $Vtune -or $VtuneInstructions) -and (Test-Path "$prefix.cache.jsonl.mtp")) {$meta.mtp_trace_sha256=(Get-FileHash "$prefix.cache.jsonl.mtp").Hash}
$meta | ConvertTo-Json -Depth 8 | Set-Content "$prefix.meta.json"
$samples | ConvertTo-Json -Depth 3 | Set-Content "$prefix.memory.json"
Get-Content "$prefix.log" | Select-Object -Last 6
Get-Content "$prefix.err.log" | Select-Object -Last 4
if($meta.Contains('memory_pressure') -and $meta.memory_pressure) {throw "Execution probe stopped the process below the 8 GiB available RAM floor"}
if($meta.watchdog_stop) {throw "Execution probe watchdog stopped the process"}
exit $proc.ExitCode
