param(
    [ValidateSet('baseline','gpu8192','gpu7168','gpu6144')]
    [string[]]$Cases=@('baseline','gpu8192','gpu7168','gpu6144'),
    [ValidateRange(1,3)][int]$Rounds=3,
    [ValidateRange(1,3)][int]$StartRound=1,
    [switch]$ShortScreen,
    [string]$Executable='',
    [string]$Tag='h9-gpu-budget'
)
$ErrorActionPreference='Stop'
if($StartRound -gt $Rounds){throw 'StartRound exceeds total rounds'}
if(!$Cases.Count -or @($Cases | Select-Object -Unique).Count -ne $Cases.Count){throw 'Cases must be nonempty and unique'}
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $repo
try {
    for($round=$StartRound;$round -le $Rounds;$round++) {
        $order=@($Cases)
        if($round%2 -eq 0){[array]::Reverse($order)}
        foreach($case in $order) {
            # The H8 q8-mapped-ram0-short1 setup is fixed. Only expert GPU
            # capacity varies; every probe gets a fresh process.
            $settings=@{
                Workload='generation';PerformancePreset='qwen38-a770';GenerationMaxSeq=2560
                PrefillChunk=2048;MaxTokens=48;GenerationRepeats=1
                HotsetProfile='perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile'
                HotsetBorrow=$true;HotsetColdQueue=2;ResidentPrefill=$true
                ColdMode=2;ColdPolicy=1;ColdPcieNum=141;ColdCredits=10;ColdThreads=12;ColdBudgetUs=1000;WorkSummary=$true
                HotsetInterval=4;HotsetPrompt=$true;HotsetPromptSwaps=0;HotsetSwaps=0
                HotsetAdoptReady=$true;HotsetAdoptLimit=4096
                DecodePrefetch=0;PrefillPrefetch=0
                CpuNf4Q8=1;CpuNf4Q8Row4=0;CpuQ8Chunks=0;RouterResidentFused=0;CpuNf4Q8Block64=0;ExpertCacheMB=0
                CpuMmapCold=1;CpuMmapPrefetch=0;CpuMmapTouch=0;HostComplement=0;HostComplementScan=512
                FillParts=8;FillShortParts=1;FillThreads=8;HostRamMB=0;PhaseScratch=$true;DemandMicrobatch=32;InlineCheckpoint=1
                TimeoutSeconds=600;Tag="$Tag-$round-$case"
                GenerationPromptFiles=@('perf/qwen4exp/prompts/story.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/code.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/story.json')
            }
            if($Executable){$settings.Executable=$Executable}
            if($ShortScreen) {
                $settings.MaxTokens=24
                $settings.GenerationPromptFiles=@('perf/qwen4exp/prompts/story.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/code.json')
            }
            if($case -ne 'baseline'){$settings.ExpertCacheMB=[int]$case.Substring(3)}
            Write-Host "Running $($settings.Tag)"
            $probe=(Join-Path $PSScriptRoot 'q4e_execution_probe.ps1').Replace("'","''")
            $json=($settings | ConvertTo-Json -Depth 4 -Compress).Replace("'","''")
            $program="`$settings=ConvertFrom-Json -AsHashtable -InputObject '$json'`n& '$probe' @settings"
            $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($program))
            & pwsh -NoProfile -EncodedCommand $encoded
            if($LASTEXITCODE -ne 0){throw "Probe $($settings.Tag) failed: $LASTEXITCODE"}
        }
    }
} finally {Pop-Location}
