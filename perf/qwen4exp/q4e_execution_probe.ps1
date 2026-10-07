param(
    [ValidateSet("story","generation","resident","preset","window","mature","integer_quality","integer_state","dpas","integer","compact","tile","memory","supply","front_graph","head","decode","projection","hc","qsa","qsa_tests","qsa_batches","qsa_profile","mtp")][string]$Workload="story",
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
    [ValidateRange(1,18)][int]$ColdThreads=4,
    [ValidateRange(1,10)][int]$ColdCredits=2,
    [ValidateRange(100,10000)][int]$ColdBudgetUs=1000,
    [ValidateSet(-1,0,1)][int]$CpuNf4Q8=-1,
    [ValidateSet(-1,0,1)][int]$CpuNf4Q8Row4=-1,
    [ValidateSet(-1,0,1)][int]$CpuNf4Q8Block64=-1,
    [ValidateRange(-1,8)][int]$CpuQ8Chunks=-1,
    [switch]$CpuDiagnostic,
    [switch]$CpuVerify,
    [switch]$CpuA64Verify,
    [ValidateSet(0,2,4)][int]$VerifyCpuRows=0,
    [ValidateSet(0,2,4)][int]$VerifyGpuReuse=0,
    [ValidateSet(0,1,2,4)][int]$MtpLiveWidth=0,
    [ValidateSet(0,1)][int]$MtpStateOnlyReplay=0,
    [ValidateSet(640,2048)][int]$MtpMaxContext=640,
    [switch]$MtpLiveCompareWidths,
    [ValidateSet(0,1)][int]$MtpSmallRowsDecode=1,
    [switch]$MtpLiveProfile,
    [ValidateSet(0,1)][int]$VerifyDecodeAttention=0,
    [ValidateSet(0,1)][int]$VerifyAttentionPair=0,
    [ValidateSet(0,1)][int]$VerifyAsyncRoute=0,
    [switch]$ProfileComponents,
    [ValidateSet(0,1,2,3)][int]$VerifyLinearReuse=0,
    [ValidateSet(0,1)][int]$VerifyDecodePrepare=0,
    [ValidateSet(0,1)][int]$VerifyGdnOrder=0,
    [int]$MtpShadowPosition=-1,
    [switch]$MtpDraftMargins,
    [switch]$MtpSelectProfile,
    [ValidateSet(0,1)][int]$MtpGpuGreedy=0,
    [ValidateSet(0,1)][int]$MtpGreedyRows=0,
    [ValidateSet(0,1)][int]$MtpSkipDraftTail=0,
    [ValidateSet(0,1)][int]$MtpHiddenReuse=0,
    [switch]$GpuMemoryDiagnostic,
    [ValidateSet(-1,0,1)][int]$RouterResidentFused=-1,
    [ValidateSet(-1,0,1)][int]$DecodePendingMlpResidual=-1,
    [ValidateSet(-1,64,128,256)][int]$HcDownWorkGroup=-1,
    [ValidateSet(-1,0,1)][int]$HcUpParallelStreams=-1,
    [switch]$HcReferenceCanary,
    [switch]$GpuQ8ReferenceCanary,
    [ValidateSet(-1,0,1)][int]$GpuNf4Q8=-1,
    [ValidateSet(-1,0,1)][int]$GpuQ8FastRte=-1,
    [ValidateSet(-1,0,1)][int]$MoeFrontFused=-1,
    [ValidateSet(-1,0,1)][int]$AttentionNf4Q8=-1,
    [ValidateSet(-1,0,1)][int]$GdnHistoryFused=-1,
    [ValidateRange(-1,5000)][int]$CpuHandoffSpinUs=-1,
    [ValidateSet(-1,0,1)][int]$CpuResultHostRead=-1,
    [ValidateSet(-1,0,1)][int]$LmHeadQ8=-1,
    [ValidateSet(-1,0,1)][int]$BnbGemvInputPrefetchOff=-1,
    [ValidateRange(-1,4096)][int]$VramReserveMB=-1,
    [ValidateSet(-1,0,1)][int]$CpuPerExpertDag=-1,
    [ValidateSet(-1,0,1)][int]$CpuTilePool=-1,
    [ValidateSet(-1,0,1)][int]$CpuLightRoute=-1,
    [ValidateSet(-1,0,1)][int]$CpuScaleRegister=-1,
    [ValidateSet(-1,0,1)][int]$CpuDeferArenaWait=-1,
    [ValidateSet(-1,0,1)][int]$CpuByteShuffle=-1,
    [string]$ColdReferenceExport="",
    [string]$ColdReferenceImport="",
    [ValidateRange(-1,8192)][int]$CpuPackedCacheMB=-1,
    [ValidateSet(-1,0,1)][int]$CpuPackedCachePrompt=-1,
    [ValidateSet(-1,0,1)][int]$CpuMmapCold=-1,
    [ValidateSet(-1,0,1)][int]$CpuMmapPrefetch=-1,
    [ValidateSet(-1,0,1)][int]$CpuMmapTouch=-1,
    [ValidateRange(-1,32)][int]$FillShortParts=-1,
    [ValidateSet(-1,0,1)][int]$HostComplement=-1,
    [ValidateRange(0,4096)][int]$HostComplementScan=0,
    [string]$Executable="",
    [ValidateSet("causal","checkpoint")][string]$IntegerStatePart="causal",
    [ValidateSet(0,256,512,1024)][int]$CompactMoe=0,
    [ValidateSet(0,128,256,512)][int]$AttentionTile=0,
    [switch]$WorkSummary,
    [ValidateSet(-1,0,1)][int]$InlineCheckpoint=-1,
    [ValidateSet(-1,0,1,2,3,4)][int]$QsaSelector=-1,
    [ValidateSet(-1,0,1)][int]$ImmediateLists=-1,
    [ValidateSet(-1,0,1)][int]$LevelZeroV2=-1,
    [ValidateSet(-1,0,1)][int]$CpuEarlyStart=-1,
    [ValidateSet(-1,0,1)][int]$CpuResultPacket=-1,
    [ValidateSet(-1,0,1)][int]$CpuAsyncRoute=-1,
    # HC reuse failed the real-model exact reference and has been removed.
    [ValidateSet(-1,0)][int]$HcReuseActivation=-1,
    [ValidateSet(-1,0,1)][int]$CpuScaleShift=-1,
    [ValidateSet(-1,0,1)][int]$CpuQ8Activation64=-1,
    [ValidateSet(-1,0,1)][int]$GpuMmapDma=-1,
    [ValidateRange(-1,2048)][int]$CpuScaleCacheMB=-1,
    [ValidateSet(-1,0,1)][int]$CpuA64Specialize=-1,
    [ValidateSet(-1,0,1)][int]$CpuA64Row2=-1,
    [ValidateRange(-1,4096)][int]$DecodeTrimCap=-1,
    [ValidateSet(-1,128,256,512)][int]$BnbGemvWorkGroup=-1,
    [ValidateSet(-1,0,1,2)][int]$LevelZeroDeviceEvents=-1,
    [ValidateRange(-1,256)][int]$LevelZeroBatchSize=-1,
    [ValidateSet(30,512)][int]$DecodePromptTokens=30,
    [ValidateSet(1,2,4)][int]$VerifyRows=1,
    [ValidateRange(2,8)][int]$DecodeRepeats=4,
    [ValidateRange(1,124)][int]$MtpAnchors=60,
    [ValidateSet(64,512)][int]$MtpSlots=64,
    [ValidateRange(1,4)][int]$MtpRepeats=3,
    [ValidateRange(0,4)][int]$MtpVerifyPasses=3,
    [switch]$MtpCompareKv,
    [switch]$MtpWindowSweep,
    [switch]$MtpCompareResident,
    [string]$MtpPromptFile="",
    [switch]$NoPrefetch,
    [ValidateSet(-1,0,1)][int]$DecodePrefetch=-1,
    [ValidateSet(-1,0,1)][int]$PrefillPrefetch=-1,
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
    [ValidateSet(0,8192,16384,24576,32768)][int]$HostRamMB=32768,
    [ValidateRange(0,16384)][int]$ExpertCacheMB=0,
    [ValidateRange(0,64)][int]$FillParts=0,
    [ValidateRange(0,32)][int]$FillThreads=0,
    [ValidateRange(0,512)][int]$DemandMicrobatch=0,
    [switch]$Trace,
    [switch]$Vtune,
    [switch]$VtuneInstructions,
    [int]$TimeoutSeconds=240,
    [string]$ModelDir="C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2"
)
$ErrorActionPreference="Stop"
if(($ColdReferenceImport -or $ColdReferenceExport) -and !$ColdQuality){throw 'Reference import/export requires ColdQuality.'}
if($ColdReferenceImport -and $ColdReferenceExport){throw 'Choose reference import or export.'}
if ($GpuMemoryDiagnostic -and ($Vtune -or $VtuneInstructions)) {
    throw 'GpuMemoryDiagnostic requires a direct Aila launch; VTune launches have a different process ID.'
}
if($HcReferenceCanary -and !$ColdQuality){throw 'HcReferenceCanary requires ColdQuality; it changes the test reference arm.'}
if($GpuQ8ReferenceCanary -and (!$ColdQuality -or ($GpuNf4Q8 -ne 1 -and $AttentionNf4Q8 -ne 1))){throw 'GpuQ8ReferenceCanary requires ColdQuality and a GPU Q8 candidate enabled.'}
if($CpuPackedCachePrompt -eq 1 -and $CpuPackedCacheMB -le 0){throw 'CpuPackedCachePrompt requires a positive CpuPackedCacheMB.'}
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
if($ExpertCacheMB -gt 0) {$overrides.AILA_Q4E_EXPERT_CACHE_MB=[string]$ExpertCacheMB}
if($VramReserveMB -ge 0) {$overrides.AILA_Q4E_VRAM_RESERVE_MB=[string]$VramReserveMB}
if($BnbGemvInputPrefetchOff -ge 0) {$overrides.AILA_BNB4_GEMV_INPUT_PREFETCH_OFF=[string]$BnbGemvInputPrefetchOff}
if($CpuPerExpertDag -ge 0) {$overrides.AILA_Q4E_CPU_PER_EXPERT_DAG=[string]$CpuPerExpertDag}
if($CpuTilePool -ge 0) {$overrides.AILA_Q4E_CPU_TILE_POOL=[string]$CpuTilePool}
if($CpuLightRoute -ge 0) {$overrides.AILA_Q4E_CPU_LIGHT_ROUTE=[string]$CpuLightRoute}
if($CpuScaleRegister -ge 0) {$overrides.AILA_Q4E_CPU_SCALE_REGISTER=[string]$CpuScaleRegister}
if($CpuDeferArenaWait -ge 0) {$overrides.AILA_Q4E_CPU_DEFER_ARENA_WAIT=[string]$CpuDeferArenaWait}
if($CpuByteShuffle -ge 0) {$overrides.AILA_Q4E_CPU_BYTE_SHUFFLE=[string]$CpuByteShuffle}
if($InlineCheckpoint -ge 0) {$overrides.AILA_Q4E_INLINE_CHECKPOINT=[string]$InlineCheckpoint}
if($WorkSummary) {$overrides.AILA_Q4E_WORK_SUMMARY="1"}
if($DecodePrefetch -ge 0) {$overrides.AILA_Q4E_PREFETCH_DECODE=[string]$DecodePrefetch}
if($PrefillPrefetch -ge 0) {$overrides.AILA_Q4E_PREFETCH_PREFILL=[string]$PrefillPrefetch}
if($CpuNf4Q8 -ge 0) {$overrides.AILA_Q4E_CPU_NF4_Q8=[string]$CpuNf4Q8}
if($CpuNf4Q8Block64 -ge 0) {$overrides.AILA_Q4E_CPU_NF4_Q8_BLOCK64=[string]$CpuNf4Q8Block64}
if($CpuNf4Q8Row4 -ge 0) {$overrides.AILA_Q4E_CPU_NF4_Q8_ROW4=[string]$CpuNf4Q8Row4}
if($CpuQ8Chunks -ge 0) {$overrides.AILA_Q4E_CPU_Q8_CHUNKS=[string]$CpuQ8Chunks}
if($CpuDiagnostic) {$overrides.AILA_Q4E_CPU_DIAGNOSTIC='1'}
if($CpuVerify) {$overrides.AILA_Q4E_CPU_Q8_VERIFY='1'}
if($CpuA64Verify) {$overrides.AILA_Q4E_CPU_A64_VERIFY='1'}
$overrides.AILA_Q4E_VERIFY_CPU_ROWS=[string]$VerifyCpuRows
$overrides.AILA_Q4E_VERIFY_GPU_REUSE=[string]$VerifyGpuReuse
$overrides.AILA_Q4E_MTP_LIVE_WIDTH=[string]$MtpLiveWidth
$overrides.AILA_Q4E_MTP_STATE_ONLY_REPLAY=[string]$MtpStateOnlyReplay
$overrides.AILA_Q4E_MTP_MAX_CONTEXT=[string]$MtpMaxContext
$overrides.AILA_Q4E_MTP_LIVE_COMPARE_WIDTHS=[string][int][bool]$MtpLiveCompareWidths
$overrides.AILA_Q4E_MTP_LIVE_PROFILE=[string][int][bool]$MtpLiveProfile
$overrides.AILA_Q4E_VERIFY_DECODE_ATTN=[string]$VerifyDecodeAttention
$overrides.AILA_Q4E_VERIFY_ASYNC_ROUTE=[string]$VerifyAsyncRoute
$overrides.AILA_Q4E_PROFILE_COMPONENTS=[string][int][bool]$ProfileComponents
$overrides.AILA_Q4E_VERIFY_LINEAR_REUSE=[string]$VerifyLinearReuse
$overrides.AILA_Q4E_VERIFY_DECODE_PREPARE=[string]$VerifyDecodePrepare
$overrides.AILA_Q4E_VERIFY_GDN_ORDER=[string]$VerifyGdnOrder
$overrides.AILA_Q4E_MTP_SHADOW_POSITION=[string]$MtpShadowPosition
$overrides.AILA_Q4E_MTP_DRAFT_MARGINS=[string][int][bool]$MtpDraftMargins
$overrides.AILA_Q4E_MTP_SELECT_PROFILE=[string][int][bool]$MtpSelectProfile
$overrides.AILA_Q4E_MTP_GPU_GREEDY=[string]$MtpGpuGreedy
$overrides.AILA_Q4E_MTP_GREEDY_ROWS=[string]$MtpGreedyRows
$overrides.AILA_Q4E_MTP_SKIP_DRAFT_TAIL=[string]$MtpSkipDraftTail
$overrides.AILA_Q4E_VERIFY_ATTN_PAIR=[string]$VerifyAttentionPair
$overrides.AILA_Q4E_MTP_HIDDEN_REUSE=[string]$MtpHiddenReuse
if($RouterResidentFused -ge 0) {$overrides.AILA_Q4E_ROUTER_RESIDENT_FUSED=[string]$RouterResidentFused}
if($DecodePendingMlpResidual -ge 0) {$overrides.AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=[string]$DecodePendingMlpResidual}
if($HcDownWorkGroup -ge 0) {$overrides.AILA_Q4E_HC_DECODE_DOWN_WG=[string]$HcDownWorkGroup}
if($HcUpParallelStreams -ge 0) {$overrides.AILA_Q4E_HC_DECODE_UP_STREAMS=[string]$HcUpParallelStreams}
if($HcReferenceCanary) {$overrides.AILA_Q4E_COLD_TEST_HC_REFERENCE='1'}
if($GpuQ8ReferenceCanary) {$overrides.AILA_Q4E_COLD_TEST_GPU_Q8_REFERENCE='1'}
if($GpuNf4Q8 -ge 0) {$overrides.AILA_Q4E_GPU_NF4_Q8=[string]$GpuNf4Q8}
if($GpuQ8FastRte -ge 0) {$overrides.AILA_Q4E_GPU_Q8_FAST_RTE=[string]$GpuQ8FastRte}
if($MoeFrontFused -ge 0) {$overrides.AILA_Q4E_MOE_FRONT_FUSED=[string]$MoeFrontFused}
if($AttentionNf4Q8 -ge 0) {$overrides.AILA_Q4E_ATTN_NF4_Q8=[string]$AttentionNf4Q8}
if($GdnHistoryFused -ge 0) {$overrides.AILA_Q4E_GDN_HISTORY_FUSED=[string]$GdnHistoryFused}
if($CpuHandoffSpinUs -ge 0) {$overrides.AILA_Q4E_CPU_HANDOFF_SPIN_US=[string]$CpuHandoffSpinUs}
if($CpuResultHostRead -ge 0) {$overrides.AILA_Q4E_CPU_RESULT_HOST_READ=[string]$CpuResultHostRead}
if($LmHeadQ8 -ge 0) {$overrides.AILA_Q4E_LM_HEAD_Q8=[string]$LmHeadQ8}
if($ColdReferenceExport) {$overrides.AILA_Q4E_COLD_REFERENCE_EXPORT=$ColdReferenceExport}
if($ColdReferenceImport) {$overrides.AILA_Q4E_COLD_REFERENCE_IMPORT=(Resolve-Path -LiteralPath $ColdReferenceImport).Path}
if($CpuPackedCacheMB -ge 0) {$overrides.AILA_Q4E_CPU_PACKED_CACHE_MB=[string]$CpuPackedCacheMB}
if($CpuPackedCachePrompt -ge 0) {$overrides.AILA_Q4E_CPU_PACKED_CACHE_PROMPT=[string]$CpuPackedCachePrompt}
if($CpuMmapCold -ge 0) {$overrides.AILA_Q4E_CPU_MMAP_COLD=[string]$CpuMmapCold}
if($CpuMmapPrefetch -ge 0) {$overrides.AILA_Q4E_CPU_MMAP_PREFETCH=[string]$CpuMmapPrefetch}
if($CpuMmapTouch -ge 0) {$overrides.AILA_Q4E_CPU_MMAP_TOUCH=[string]$CpuMmapTouch}
if($FillShortParts -ge 0) {$overrides.AILA_Q4E_FILL_SHORT_PARTS=[string]$FillShortParts}
if($HostComplement -ge 0) {$overrides.AILA_Q4E_HOST_COMPLEMENT=[string]$HostComplement}
if($HostComplementScan) {$overrides.AILA_Q4E_HOST_COMPLEMENT_SCAN=[string]$HostComplementScan}
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
if($LevelZeroV2 -ge 0) {$overrides.SYCL_UR_USE_LEVEL_ZERO_V2=[string]$LevelZeroV2}
if($CpuEarlyStart -ge 0) {$overrides.AILA_Q4E_CPU_EARLY_START=[string]$CpuEarlyStart}
if($CpuResultPacket -ge 0) {$overrides.AILA_Q4E_CPU_RESULT_PACKET=[string]$CpuResultPacket}
if($CpuAsyncRoute -ge 0) {$overrides.AILA_Q4E_CPU_ASYNC_ROUTE=[string]$CpuAsyncRoute}
if($HcReuseActivation -ge 0) {$overrides.AILA_Q4E_HC_REUSE_ACTIVATION=[string]$HcReuseActivation}
if($CpuScaleShift -ge 0) {$overrides.AILA_Q4E_CPU_SCALE_SHIFT=[string]$CpuScaleShift}
if($CpuQ8Activation64 -ge 0) {$overrides.AILA_Q4E_CPU_Q8_ACTIVATION64=[string]$CpuQ8Activation64}
if($GpuMmapDma -ge 0) {$overrides.AILA_Q4E_GPU_MMAP_DMA=[string]$GpuMmapDma}
if($CpuScaleCacheMB -ge 0) {$overrides.AILA_Q4E_CPU_SCALE_CACHE_MB=[string]$CpuScaleCacheMB}
if($CpuA64Specialize -ge 0) {$overrides.AILA_Q4E_CPU_A64_SPECIALIZE=[string]$CpuA64Specialize}
if($CpuA64Row2 -ge 0) {$overrides.AILA_Q4E_CPU_A64_ROW2=[string]$CpuA64Row2}
if($DecodeTrimCap -ge 0) {$overrides.AILA_Q4E_DECODE_TRIM_CAP=[string]$DecodeTrimCap}
if($BnbGemvWorkGroup -ge 0) {$overrides.AILA_BNB4_GEMV_WG=[string]$BnbGemvWorkGroup}
if($LevelZeroDeviceEvents -ge 0) {
    $overrides.SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=[string]$LevelZeroDeviceEvents
    $overrides.UR_L0_DEVICE_SCOPE_EVENTS=[string]$LevelZeroDeviceEvents
}
if($LevelZeroBatchSize -ge 0) {
    $overrides.SYCL_PI_LEVEL_ZERO_BATCH_SIZE=[string]$LevelZeroBatchSize
    $overrides.UR_L0_BATCH_SIZE=[string]$LevelZeroBatchSize
}
if($Vtune -or $VtuneInstructions) {
    $overrides.AILA_Q4E_CACHE_TRACE="$prefix.cache.jsonl"
    $overrides.AILA_Q4E_TRACE_MARKERS="1"
}
if($FillParts) {$overrides.AILA_Q4E_FILL_PARTS=[string]$FillParts}
if($FillThreads) {$overrides.AILA_Q4E_FILL_THREADS=[string]$FillThreads}
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
if($Workload -eq "head") {
    $exe=Join-Path $repo "build-refactor/AilaQwen4ExpToleranceTests.exe"
    $arguments=@("--head-kernel-model",$ModelDir)
    $overrides.AILA_Q4E_HOT_RAM_MB="0"
    $inputIdentity=@{kind="real_bf16_lm_head_only";source_sha256=(Get-FileHash (Join-Path $repo "tests/models/Qwen4ExpHeadProbe.hpp")).Hash}
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
    $overrides.AILA_Q4E_SMALL_ROWS_DECODE=[string]$MtpSmallRowsDecode
    $overrides.AILA_Q4E_SMALL_ROWS_HC="1"
    $overrides.AILA_PROFILE_Q4E_DECODE=$(if($Profile){"1"}else{"0"})
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
    diagnostic_only=[bool]($GpuMemoryDiagnostic -or $CpuVerify -or $CpuA64Verify -or $CpuDiagnostic -or $Trace -or $Vtune -or $VtuneInstructions -or $Profile -or $Stages -or $Workload -in @("resident","preset","window","mature","integer_quality","integer_state","dpas","integer","compact","tile","supply","front_graph","head","decode","projection","hc","qsa","qsa_tests","qsa_batches","qsa_profile","mtp"));watchdog_stop=$false
}
$launcher=$exe
if($HotsetProfile) {$meta.hotset_profile=@{path=(Resolve-Path -LiteralPath $HotsetProfile).Path;sha256=(Get-FileHash -LiteralPath $HotsetProfile).Hash}}
if($ColdQuality) {$meta.diagnostic_only=$true;$meta.workload="cold_quality"}
if($ColdReferenceImport) {
    $referenceInfo=Get-Content -LiteralPath ($overrides.AILA_Q4E_COLD_REFERENCE_IMPORT+'.meta.json') -Raw | ConvertFrom-Json
    $referenceHash=(Get-FileHash -LiteralPath $overrides.AILA_Q4E_COLD_REFERENCE_IMPORT).Hash
    if($referenceInfo.reference_sha256 -ne $referenceHash -or $referenceInfo.model_config_sha256 -ne $meta.model_config_sha256 -or !$referenceInfo.original_bf16_teacher){throw 'Original reference integrity/model mismatch.'}
    $meta.original_reference=@{path=$overrides.AILA_Q4E_COLD_REFERENCE_IMPORT;sha256=$referenceHash;source_run=$referenceInfo.source_run;source_exe_sha256=$referenceInfo.source_exe_sha256}
}
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
$meta.process_id=$proc.Id
$gpuSamples=[System.Collections.Generic.List[object]]::new()
$gpuCounterPaths=@('\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage',
    '\GPU Process Memory(*)\Total Committed','\GPU Process Memory(*)\Local Usage',
    '\GPU Process Memory(*)\Non Local Usage','\GPU Adapter Memory(*)\Dedicated Usage',
    '\GPU Adapter Memory(*)\Shared Usage','\GPU Adapter Memory(*)\Total Committed')
$samples=[System.Collections.Generic.List[object]]::new()
$timer=[System.Diagnostics.Stopwatch]::StartNew()
try {
    while(!$proc.WaitForExit(500) -and $timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $proc.Refresh();$free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024
        # Immediate children only; under VTune this is NOT the whole process tree.
        $children=@(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" | ForEach-Object {Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue})
        $childWs=0;foreach($child in $children){$childWs+=$child.WorkingSet64}
        $samples.Add([pscustomobject]@{ms=$timer.ElapsedMilliseconds;working_set_bytes=$proc.WorkingSet64;child_working_set_bytes=$childWs;system_free_mb=$free})
        if($GpuMemoryDiagnostic) {
            try {
                $counter=Get-Counter -Counter $gpuCounterPaths -MaxSamples 1 -ErrorAction Stop
                $gpuRows=@($counter.CounterSamples | Where-Object {
                    $_.Path -match '\\gpu adapter memory\(' -or $_.InstanceName -match ('^pid_'+$proc.Id+'_')
                } | ForEach-Object {
                    [pscustomobject]@{timestamp=$_.Timestamp;path=$_.Path;instance=$_.InstanceName;
                        # PDH VALID_DATA=0 and NEW_DATA=1 both carry valid values.
                        status=$_.Status;bytes=$(if($_.Status -in @(0,1)){[uint64]$_.CookedValue}else{$null})}
                })
                $gpuSamples.Add([pscustomobject]@{ms=$timer.ElapsedMilliseconds;pid=$proc.Id;rows=$gpuRows;
                    process_available=[bool](@($gpuRows | Where-Object {$_.instance -match ('^pid_'+$proc.Id+'_') -and $_.status -in @(0,1)}).Count)})
            } catch {
                $gpuSamples.Add([pscustomobject]@{ms=$timer.ElapsedMilliseconds;pid=$proc.Id;
                    process_available=$false;unavailable_reason=$_.Exception.Message})
            }
        }
        if($free -lt 8192) {$meta.memory_pressure=$true;break}
    }
} finally {
    if(!$proc.HasExited) {$proc.Kill($true);$meta.watchdog_stop=$true}
    $proc.WaitForExit()
}
$meta.exit_code=$proc.ExitCode
if($ColdReferenceExport -and $proc.ExitCode -eq 0 -and !$meta.watchdog_stop) {
    $referenceExportHash=(Get-FileHash -LiteralPath $ColdReferenceExport).Hash
    @{original_bf16_teacher=$true;reference_sha256=$referenceExportHash;model_config_sha256=$meta.model_config_sha256;source_run=$prefix;source_exe_sha256=$meta.exe_sha256} |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath ($ColdReferenceExport+'.meta.json') -Encoding utf8
    $meta.exported_original_reference=@{path=$ColdReferenceExport;sha256=$referenceExportHash}
}
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
if($GpuMemoryDiagnostic) {
    $gpuSamples | ConvertTo-Json -Depth 6 | Set-Content "$prefix.gpu-memory.json"
    $meta.gpu_memory_sha256=(Get-FileHash "$prefix.gpu-memory.json").Hash
    $meta | ConvertTo-Json -Depth 8 | Set-Content "$prefix.meta.json"
}
Get-Content "$prefix.log" | Select-Object -Last 6
Get-Content "$prefix.err.log" | Select-Object -Last 4
if($meta.Contains('memory_pressure') -and $meta.memory_pressure) {throw "Execution probe stopped the process below the 8 GiB available RAM floor"}
if($meta.watchdog_stop) {throw "Execution probe watchdog stopped the process"}
exit $proc.ExitCode
