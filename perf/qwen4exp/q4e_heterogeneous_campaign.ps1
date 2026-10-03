param(
    [ValidateSet('baseline','baseline-ram16','baseline-ram16-short1','baseline-ram8-short1','baseline-ram0-short1','complement','q8','q8-ram16','q8-ram16-short1','q8-complement','q8-complement-fill1','q8-complement-split64','q8-mapped','q8-mapped-fill1','q8-mapped-split64','q8-mapped-ram16','q8-mapped-short1','q8-mapped-ram16-short1','q8-mapped-ram16-short1-prefetch','q8-mapped-ram16-short1-touch','q8-mapped-ram16-short1-prefetch-touch','q8-mapped-ram8-short1-touch','q8-mapped-ram8-short1-prefetch-touch','q8-mapped-ram8-short1-split64','q8-mapped-ram16-short1-split64','q8-mapped-ram0-short1')]
    [string[]]$Cases=@('baseline','baseline-ram0-short1','q8-mapped-ram0-short1'),
    [ValidateRange(1,3)][int]$Rounds=1,
    [ValidateRange(1,3)][int]$StartRound=1,
    [switch]$ShortScreen,
    [string]$Tag='h8-heterogeneous'
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
            $settings=@{
                Workload='generation';PerformancePreset='qwen38-a770';GenerationMaxSeq=2560
                PrefillChunk=2048;MaxTokens=48;GenerationRepeats=1
                HotsetProfile='perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile'
                HotsetBorrow=$true;HotsetColdQueue=2;ResidentPrefill=$true
                ColdMode=2;ColdPolicy=1;ColdCredits=10;ColdThreads=12;WorkSummary=$true
                HotsetPrompt=$true;HotsetPromptSwaps=0;HotsetSwaps=0
                HotsetAdoptReady=$true;HotsetAdoptLimit=4096
                DecodePrefetch=0;PrefillPrefetch=0
                CpuNf4Q8=0;CpuMmapCold=0;CpuMmapPrefetch=0;CpuMmapTouch=0;HostComplement=0;HostComplementScan=512
                FillParts=8;FillShortParts=0;FillThreads=8;HostRamMB=32768;PhaseScratch=$true;DemandMicrobatch=32;InlineCheckpoint=1
                TimeoutSeconds=600;Tag="$Tag-$round-$case"
                GenerationPromptFiles=@('perf/qwen4exp/prompts/story.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/code.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/story.json')
            }
            if($ShortScreen) {
                $settings.MaxTokens=24
                $settings.GenerationPromptFiles=@('perf/qwen4exp/prompts/story.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/code.json')
            }
            if($case -eq 'complement' -or $case.StartsWith('q8-complement')){$settings.HostComplement=1}
            if($case.StartsWith('q8')){$settings.CpuNf4Q8=1}
            if($case.StartsWith('q8-mapped')){$settings.CpuMmapCold=1}
            if($case.Contains('-ram16')){$settings.HostRamMB=16384}
            if($case.Contains('-ram8')){$settings.HostRamMB=8192}
            if($case.Contains('-ram0')){$settings.HostRamMB=0}
            if($case.Contains('-short1')){$settings.FillShortParts=1}
            if($case.Contains('-prefetch')){$settings.CpuMmapPrefetch=1}
            if($case.Contains('-touch')){$settings.CpuMmapTouch=1}
            if($case.EndsWith('fill1')){$settings.FillParts=1}
            if($case.EndsWith('split64')){$settings.ColdPolicy=3;$settings.ColdPcieNum=64}
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
