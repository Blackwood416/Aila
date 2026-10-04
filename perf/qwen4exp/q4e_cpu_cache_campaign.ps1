param(
 [ValidateSet('baseline','cpu-cache1024','cpu-cache2048','cpu-cache4096','cpu-cache1024-prompt','cpu-cache2048-prompt','cpu-cache4096-prompt')]
 [string[]]$Cases=@('baseline','cpu-cache1024','cpu-cache2048','cpu-cache4096'),
 [ValidateRange(1,3)][int]$Rounds=1,[switch]$ShortScreen,
 [string]$Tag='h10-cpu-cache',[string]$Executable=''
)
$ErrorActionPreference='Stop'
if(!$Cases.Count -or @($Cases | Select-Object -Unique).Count -ne $Cases.Count){throw 'CPU cache cases must be nonempty and unique'}
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $repo
try {
 for($round=1;$round -le $Rounds;$round++) {
  $order=@($Cases);if($round%2 -eq 0){[array]::Reverse($order)}
  foreach($case in $order) {
   $settings=@{
    Workload='generation';PerformancePreset='qwen38-a770';GenerationMaxSeq=2560;PrefillChunk=2048;MaxTokens=48;GenerationRepeats=1
    HotsetProfile='perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile';HotsetBorrow=$true;HotsetColdQueue=2;ResidentPrefill=$true
    ColdMode=2;ColdPolicy=1;ColdPcieNum=141;ColdCredits=10;ColdThreads=12;ColdBudgetUs=1000;WorkSummary=$true
    HotsetInterval=4;HotsetPrompt=$true;HotsetPromptSwaps=0;HotsetSwaps=0;HotsetAdoptReady=$true;HotsetAdoptLimit=4096
    DecodePrefetch=0;PrefillPrefetch=0;CpuNf4Q8=1;CpuNf4Q8Row4=0;CpuQ8Chunks=0;CpuNf4Q8Block64=0;RouterResidentFused=0
    DecodePendingMlpResidual=1;HcDownWorkGroup=256;HcUpParallelStreams=0
    CpuMmapCold=1;CpuMmapPrefetch=0;CpuMmapTouch=0;HostComplement=0;HostComplementScan=512
    FillParts=8;FillShortParts=1;FillThreads=8;HostRamMB=0;PhaseScratch=$true;DemandMicrobatch=32;InlineCheckpoint=1
    TimeoutSeconds=600;Tag="$Tag-$round-$case"
    CpuPackedCacheMB=if($case -eq 'baseline'){0}else{[int]($case.Substring(9).Split('-')[0])}
    GenerationPromptFiles=@('perf/qwen4exp/prompts/story.json','perf/qwen4exp/prompts/archive-natural.json',
       'perf/qwen4exp/prompts/code.json','perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/story.json')
   }
   if($case -like '*-prompt'){$settings.CpuPackedCachePrompt=1}
   if($Executable){$settings.Executable=$Executable}
   if($ShortScreen){$settings.MaxTokens=24;$settings.GenerationPromptFiles=$settings.GenerationPromptFiles[0..2]}
   Write-Host "Running $($settings.Tag)"
   $probe=(Join-Path $PSScriptRoot 'q4e_execution_probe.ps1').Replace("'","''")
   $json=($settings | ConvertTo-Json -Depth 4 -Compress).Replace("'","''")
   $program="`$settings=ConvertFrom-Json -AsHashtable -InputObject '$json'`n& '$probe' @settings"
   & pwsh -NoProfile -EncodedCommand ([Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($program)))
   if($LASTEXITCODE -ne 0){throw "CPU cache probe failed: $($settings.Tag)"}
  }
 }
} finally {Pop-Location}
