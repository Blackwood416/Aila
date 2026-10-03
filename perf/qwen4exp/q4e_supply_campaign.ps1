param(
    [ValidateRange(1,3)][int]$Pairs=3,
    [ValidateRange(1,3)][int]$StartPair=1,
    [ValidateSet('all','prefill','off','off-static')][string[]]$Modes=@('all','prefill'),
    [ValidateSet(512,1024,2048)][int]$PrefillChunk=512,
    [switch]$WindowComparison,
    [string]$Tag='h7-supply'
)
$ErrorActionPreference='Stop'
if($StartPair -gt $Pairs){throw 'StartPair must not exceed the total pair limit'}
if($Modes.Count -eq 0){throw 'At least one supply mode is required'}
if(@($Modes | Select-Object -Unique).Count -ne $Modes.Count){throw 'Supply modes must be unique'}
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $repo
try {
    # Repeat the same five-request history in isolated processes. Reverse the
    # process order in pair two; preserve every startup/adaptation observation.
    for($pair=$StartPair;$pair -le $Pairs;$pair++) {
        $arms=if($WindowComparison) {
            @(@{Mode='all';Chunk=512},@{Mode='off-static';Chunk=512},@{Mode='off-static';Chunk=2048})
        } else {
            @($Modes | ForEach-Object {@{Mode=$_;Chunk=$PrefillChunk}})
        }
        if($pair%2 -eq 0){[array]::Reverse($arms)}
        foreach($case in $arms) {
            $arm=$case.Mode
            $runTag=if($WindowComparison){"$Tag-$pair-$arm-$($case.Chunk)"}else{"$Tag-$pair-$arm"}
            $options=@{
                Workload='generation';PerformancePreset='qwen38-a770';GenerationMaxSeq=2560
                PrefillChunk=$case.Chunk;MaxTokens=48;GenerationRepeats=1
                HotsetProfile='perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile'
                HotsetBorrow=$true;HotsetColdQueue=2;ResidentPrefill=$true
                ColdMode=2;ColdPolicy=1;ColdCredits=10;ColdThreads=12;WorkSummary=$true
                HotsetPrompt=$true;HotsetPromptSwaps=0;HotsetSwaps=512
                HotsetAdoptReady=$true;HotsetAdoptLimit=4096
                DecodePrefetch=1;PrefillPrefetch=1
                TimeoutSeconds=600;Tag=$runTag
                GenerationPromptFiles=@('perf/qwen4exp/prompts/story.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/code.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/story.json')
            }
            switch($arm) {
                'prefill' {$options.DecodePrefetch=0}
                'off' {$options.DecodePrefetch=0;$options.PrefillPrefetch=0}
                'off-static' {$options.DecodePrefetch=0;$options.PrefillPrefetch=0;$options.HotsetSwaps=0}
            }
            Write-Host "Running $($options.Tag)"
            # The probe resets process environment and exits independently.
            # Encode a structured hashtable rather than passing it through
            # native command-line binding, which cannot preserve array values.
            $probe=(Join-Path $PSScriptRoot 'q4e_execution_probe.ps1').Replace("'","''")
            $json=($options | ConvertTo-Json -Depth 4 -Compress).Replace("'","''")
            $program="`$settings=ConvertFrom-Json -AsHashtable -InputObject '$json'`n& '$probe' @settings"
            $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($program))
            & pwsh -NoProfile -EncodedCommand $encoded
            if($LASTEXITCODE -ne 0){throw "Probe $($options.Tag) failed: $LASTEXITCODE"}
        }
    }
} finally {Pop-Location}
