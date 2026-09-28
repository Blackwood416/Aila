param(
    [string]$Tag = 'dataflow_before',
    [string]$Metadata = 'tmp/q4e_prof_prepared12_pair3_fast_small40.meta.json'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
Push-Location $repo
try {
    $meta = Get-Content $Metadata -Raw | ConvertFrom-Json
    foreach ($property in $meta.env_vars.PSObject.Properties) {
        Set-Item -Path "env:$($property.Name)" -Value $property.Value
    }
    try {
        & 'C:/Program Files (x86)/Intel/oneAPI/vtune/latest/bin64/vtune.exe' -collect hotspots -knob sampling-mode=sw -result-dir "tmp/q4e_vtune_$Tag" -- ./build-refactor/Aila.exe -m 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2' --max-seq 512 --greedy --thinking-budget 0 --max-tokens 48 --messages-json perf/qwen4exp/prompts/story.json *> "tmp/q4e_vtune_$Tag.log"
        if ($LASTEXITCODE -ne 0) { throw "VTune collection failed: tmp/q4e_vtune_$Tag.log" }
    } finally {
        foreach ($property in $meta.env_vars.PSObject.Properties) {
            Remove-Item -Path "env:$($property.Name)" -ErrorAction SilentlyContinue
        }
    }
} finally { Pop-Location }
