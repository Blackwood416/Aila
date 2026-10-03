param(
    [ValidateRange(1,3)][int]$Pairs=3,
    [ValidateRange(0,4096)][int]$AdoptLimit=4096,
    [string]$Tag='h6-ready',
    [switch]$NoPrefetch
)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $repo
try {
    # Match H5's causal seed control; isolate physical Ready adoption only.
    # Alternating process order retains startup/adaptation samples in both arms.
    for($pair=1;$pair -le $Pairs;$pair++) {
        $arms=if($pair%2){@('control','adopt')}else{@('adopt','control')}
        foreach($arm in $arms) {
            $options=@{
                Workload='generation';PerformancePreset='qwen38-a770';GenerationMaxSeq=2560
                PrefillChunk=512;MaxTokens=48;GenerationRepeats=1;HotsetProfile='perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile'
                HotsetBorrow=$true;HotsetColdQueue=2;ResidentPrefill=$true
                ColdMode=2;ColdPolicy=1;ColdCredits=10;ColdThreads=12;WorkSummary=$true
                HotsetPrompt=$true;HotsetPromptSwaps=0;HotsetSwaps=512
                HotsetAdoptReady=($arm -eq 'adopt');HotsetAdoptLimit=$AdoptLimit
                NoPrefetch=[bool]$NoPrefetch;TimeoutSeconds=600;Tag="$Tag-$pair-$arm"
                GenerationPromptFiles=@('perf/qwen4exp/prompts/story.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/code.json',
                    'perf/qwen4exp/prompts/archive-natural.json','perf/qwen4exp/prompts/story.json')
            }
            Write-Host "Running $($options.Tag)"
            # Isolate the probe's environment and exit from the campaign driver.
            $probe=(Join-Path $PSScriptRoot 'q4e_execution_probe.ps1').Replace("'","''")
            $json=($options | ConvertTo-Json -Depth 4 -Compress).Replace("'","''")
            $program="`$settings=ConvertFrom-Json -AsHashtable -InputObject '$json'`n& '$probe' @settings"
            $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($program))
            & pwsh -NoProfile -EncodedCommand $encoded
            if($LASTEXITCODE -ne 0){throw "Probe $($options.Tag) failed: $LASTEXITCODE"}
        }
    }
} finally {Pop-Location}
