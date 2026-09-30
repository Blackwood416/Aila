param([Parameter(Mandatory=$true)][string]$Prefix)
$ErrorActionPreference="Stop"
# Prefix is the same absolute or repo-relative prefix used by execution_probe.
# A fine resolution is necessary: default VTune interval exports merge events.
$vtune="C:/Program Files (x86)/Intel/oneAPI/vtune/latest/bin64/vtune.exe"
& $vtune -report timeline -r "$Prefix-vtune" `
    -report-knob column-by=GPUComputeTask -report-knob query-type=interval `
    -report-knob object-names=true -report-knob bin-count=100000000 `
    -report-knob time-precision=9 -format csv -report-output "$Prefix.gpu.tsv"
if($LASTEXITCODE) {throw "VTune export failed: $LASTEXITCODE"}
& python (Join-Path $PSScriptRoot "q4e_timeline_summary.py") "$Prefix.gpu.tsv" "$Prefix.cache.jsonl" --output "$Prefix.timeline.json"
if($LASTEXITCODE) {throw "Timeline validation failed: $LASTEXITCODE"}
