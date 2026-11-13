<#
.SYNOPSIS
Measures Qwen4Exp decode throughput and the per-token latency distribution.

.DESCRIPTION
Decode quality is a distribution, not a single number: the first tokens after
prefill are slower, and the tail is dominated by router sync and expert staging.
The A/B runner cannot report that, because its GenerateMessages wall time folds
prefill, sampling, and process overhead into one scalar. This script drives
Aila.exe directly with AILA_Q4E_WORK_SUMMARY=1 and reads the per-forward
[Q4E AdaptiveWork] line the Qwen4Exp backend prints, so every decode step
contributes one latency sample instead of only the end-to-end wall time.

Within a forward, count=1 marks a decode step and count>1 a prefill chunk.
Forwards logged before the "Engine ready!" banner are ignored, so load-time
warmup and preset calibration cannot pollute the samples.

Per target --max-tokens the script reports

    PromptTokens     exact prompt length from [GenerateMessages]
    Generated        tokens actually generated (stops early on EOS)
    WallMs           [GenerateMessages] wall time: prefill + decode + sampling
    PrefillMs        sum of submitted_ms over the prefill chunk forwards
    DecodeSumMs      sum of submitted_ms over the decode forwards
    MeanMsPerTok     DecodeSumMs / decode steps
    TokPerSec        decode steps / (DecodeSumMs / 1000)
    P50Ms/P90Ms/MinMs/MaxMs   per-decode-step submitted_ms distribution
    MeanRouterWaitMs mean router-readback sync per decode step
    MeanCpuComputeMs mean CPU expert task time per decode step
    MeanCpuJoinMs    mean CPU pool join wait per decode step
    HitRatePct       hits / (hits + misses) over the decode steps
    Tok_1_64         decode interval throughput for steps 1-64
    Tok_65_256       decode interval throughput for steps 65-256
    Tok_257_512      decode interval throughput for steps 257-512

submitted_ms is the coordinator wall time of one forward (submit path plus the
router readback that drains the previous step), which is what a token costs.
Interval throughput is what separates "fast steady state" from "fast average":
a target that only reaches tok/s by skipping the slow head or tail is visible.

.PARAMETER MaxTokens
Generation limits to sweep. Each becomes one row. Comma-separated strings and
arrays are both accepted.

.PARAMETER MaxSeq
Context window. It sizes the KV cache and therefore the admitted resident
expert pool, so it is part of the measurement identity and must stay fixed
across arms.

.PARAMETER SkipWarmup
Sets AILA_INIT_WARMUP=0 so the load-time warmup does not run. Warmup forwards
are already excluded from the samples; this only shortens the run.

.EXAMPLE
pwsh -NoProfile -File tools/decode-bench.ps1 -MaxTokens @(256, 512)
#>
param(
    [Alias('Binary')]
    [string]$Exe = 'build-refactor/Aila.exe',
    [string]$Model = 'D:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [string]$Prompt = 'perf/qwen4exp/prompts/essay.json',
    [Alias('Tokens')]
    [object[]]$MaxTokens = @(256, 512),
    [int]$MaxSeq = 4096,
    [string]$OutDir = 'tmp/decode-bench',
    [switch]$SkipWarmup
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $repo $Path)
}

$Exe = Resolve-RepoPath $Exe
$Prompt = Resolve-RepoPath $Prompt
$work = Resolve-RepoPath $OutDir
New-Item -ItemType Directory -Force -Path $work | Out-Null

# Baseline environment the decode work already established as neutral: no
# immediate command lists, copy engine on, persistent SYCL cache.
$env:UR_L0_USE_COPY_ENGINE = '1'
$env:SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE = '1'
$env:UR_L0_USE_IMMEDIATE_COMMANDLISTS = '0'
$env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = '0'
$env:SYCL_UR_USE_LEVEL_ZERO_V2 = '0'
$env:UR_L0_BATCH_SIZE = '64'
$env:SYCL_PI_LEVEL_ZERO_BATCH_SIZE = '64'
$env:SYCL_CACHE_PERSISTENT = '0'

# Both flags are read once per process, so they are set before the first run
# rather than per iteration.
$env:AILA_Q4E_WORK_SUMMARY = '1'
if ($SkipWarmup) { $env:AILA_INIT_WARMUP = '0' }

function Get-ForwardField {
    param([string]$Line, [string]$Name)
    $m = [regex]::Match($Line, '(?:^|\s)' + [regex]::Escape($Name) + '=(-?[0-9.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

function Get-Rounded {
    param($Value, [int]$Digits = 3)
    if ($null -eq $Value) { return $null }
    return [Math]::Round([double]$Value, $Digits)
}

# A segment or cache counter can have no samples at all; print that as '-' so a
# blank field is never mistaken for a measured zero.
function Format-Metric {
    param($Value, [int]$Digits = 1)
    if ($null -eq $Value) { return '-' }
    return ([double]$Value).ToString("F$Digits")
}

function Get-Percentile {
    param([double[]]$Sorted, [double]$Percent)
    if ($Sorted.Count -eq 0) { return $null }
    if ($Sorted.Count -eq 1) { return $Sorted[0] }
    $rank = ($Percent / 100.0) * ($Sorted.Count - 1)
    $lo = [int][Math]::Floor($rank)
    $hi = [int][Math]::Ceiling($rank)
    if ($lo -eq $hi) { return $Sorted[$lo] }
    return $Sorted[$lo] + ($Sorted[$hi] - $Sorted[$lo]) * ($rank - $lo)
}

function Get-SegmentRate {
    param([double[]]$Latencies, [int]$First, [int]$Last)
    $ms = 0.0
    $n = 0
    for ($i = 0; $i -lt $Latencies.Count; ++$i) {
        $step = $i + 1
        if ($step -lt $First -or $step -gt $Last) { continue }
        $ms += $Latencies[$i]
        ++$n
    }
    if ($n -eq 0 -or $ms -le 0) { return $null }
    return $n / ($ms / 1000.0)
}

$MaxTokens = @($MaxTokens | ForEach-Object { "$_" -split ',' } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$rows = @()
foreach ($target in $MaxTokens) {
    $logPath = Join-Path $work "engine-$target.log"
    & $Exe -m $Model --performance-preset qwen38-a770 --max-seq $MaxSeq --greedy `
        --thinking-budget 0 --max-tokens $target --messages-json $Prompt `
        --no-stream --chat-output-json --log-level info > $logPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Aila.exe exited with code $LASTEXITCODE for --max-tokens $target (log: $logPath)"
    }

    $lines = @(Get-Content -Path $logPath)
    $readyAt = -1
    for ($i = 0; $i -lt $lines.Count; ++$i) {
        if ($lines[$i] -like '*Engine ready!*') { $readyAt = $i; break }
    }
    if ($readyAt -lt 0) {
        Write-Warning "no 'Engine ready!' banner in $logPath; every forward is treated as a sample"
    }

    $decodeMs = New-Object System.Collections.Generic.List[double]
    $prefillMs = 0.0
    $routerWaitMs = 0.0
    $cpuComputeMs = 0.0
    $cpuJoinMs = 0.0
    $hits = 0.0
    $misses = 0.0
    $lastVerifyLine = $null
    $lastVerifyPos = -1
    $lastVerifySubmitted = 0.0
    for ($i = $readyAt + 1; $i -lt $lines.Count; ++$i) {
        $line = $lines[$i]
        if ($line -match '\[Q4E MtpWindow\] pos=(\d+) count=(\d+) emitted=(\d+) accepted=(\d+) window_ms=([0-9.]+)') {
            $winPos = [int]$Matches[1]
            $emitted = [Math]::Max(1, [int]$Matches[3])
            $winMs = [double]$Matches[5]
            if ($null -ne $lastVerifyLine -and $lastVerifyPos -eq $winPos) {
                $prefillMs = [Math]::Max(0.0, $prefillMs - $lastVerifySubmitted)
                $value = Get-ForwardField $lastVerifyLine 'router_wait_ms'
                if ($null -ne $value) { $routerWaitMs += $value }
                $value = Get-ForwardField $lastVerifyLine 'cpu_compute_ms'
                if ($null -ne $value) { $cpuComputeMs += $value }
                $value = Get-ForwardField $lastVerifyLine 'cpu_join_ms'
                if ($null -ne $value) { $cpuJoinMs += $value }
                $value = Get-ForwardField $lastVerifyLine 'hits'
                if ($null -ne $value) { $hits += $value }
                $value = Get-ForwardField $lastVerifyLine 'misses'
                if ($null -ne $value) { $misses += $value }
                $lastVerifyLine = $null
            }
            $perTokMs = $winMs / $emitted
            for ($k = 0; $k -lt $emitted; ++$k) {
                $decodeMs.Add($perTokMs)
            }
            continue
        }
        if ($line -notmatch '\[Q4E AdaptiveWork\] pos=(\d+) count=(\d+)') { continue }
        $pos = [int]$Matches[1]
        $count = [int]$Matches[2]
        $submitted = Get-ForwardField $line 'submitted_ms'
        if ($null -eq $submitted) { continue }
        if ($count -ne 1) {
            $prefillMs += $submitted
            $lastVerifyLine = $line
            $lastVerifyPos = $pos
            $lastVerifySubmitted = $submitted
            continue
        }

        $decodeMs.Add($submitted)
        $value = Get-ForwardField $line 'router_wait_ms'
        if ($null -ne $value) { $routerWaitMs += $value }
        $value = Get-ForwardField $line 'cpu_compute_ms'
        if ($null -ne $value) { $cpuComputeMs += $value }
        $value = Get-ForwardField $line 'cpu_join_ms'
        if ($null -ne $value) { $cpuJoinMs += $value }
        $value = Get-ForwardField $line 'hits'
        if ($null -ne $value) { $hits += $value }
        $value = Get-ForwardField $line 'misses'
        if ($null -ne $value) { $misses += $value }
    }

    $promptTokens = 0
    $generated = 0
    $wallMs = 0.0
    foreach ($line in $lines) {
        if ($line -match '\[GenerateMessages\] Prompt=(\d+) Generated=(\d+) in ([0-9.]+) ms') {
            $promptTokens = [int]$Matches[1]
            $generated = [int]$Matches[2]
            $wallMs = [double]$Matches[3]
        }
    }

    if ($decodeMs.Count -eq 0) {
        Write-Warning "no decode forwards for --max-tokens $target (log: $logPath)"
        continue
    }

    $sorted = $decodeMs.ToArray()
    [Array]::Sort($sorted)
    $decodeSumMs = ($decodeMs | Measure-Object -Sum).Sum
    $meanMs = $decodeSumMs / $decodeMs.Count
    $tokPerSec = $decodeMs.Count / ($decodeSumMs / 1000.0)
    $accesses = $hits + $misses
    $hitRate = if ($accesses -gt 0) { 100.0 * $hits / $accesses } else { $null }

    $row = [pscustomobject]@{
        TargetTokens     = $target
        Generated        = $generated
        PromptTokens     = $promptTokens
        WallMs           = Get-Rounded $wallMs 1
        PrefillMs        = Get-Rounded $prefillMs 1
        DecodeSumMs      = Get-Rounded $decodeSumMs 1
        MeanMsPerTok     = Get-Rounded $meanMs 3
        TokPerSec        = Get-Rounded $tokPerSec 1
        P50Ms            = Get-Rounded (Get-Percentile $sorted 50) 3
        P90Ms            = Get-Rounded (Get-Percentile $sorted 90) 3
        MinMs            = Get-Rounded $sorted[0] 3
        MaxMs            = Get-Rounded $sorted[-1] 3
        MeanRouterWaitMs = Get-Rounded ($routerWaitMs / $decodeMs.Count) 3
        MeanCpuComputeMs = Get-Rounded ($cpuComputeMs / $decodeMs.Count) 3
        MeanCpuJoinMs    = Get-Rounded ($cpuJoinMs / $decodeMs.Count) 3
        HitRatePct       = Get-Rounded $hitRate 1
        Tok_1_64         = Get-Rounded (Get-SegmentRate $decodeMs 1 64) 1
        Tok_65_256       = Get-Rounded (Get-SegmentRate $decodeMs 65 256) 1
        Tok_257_512      = Get-Rounded (Get-SegmentRate $decodeMs 257 512) 1
    }
    $rows += $row
    Write-Host ("{0,6} tokens  gen {1,4}  {2,7:N1} tok/s  mean {3,6:N3} ms  p50 {4,6:N3}  p90 {5,6:N3}  prefill {6,8:N1} ms" -f `
        $row.TargetTokens, $row.Generated, $row.TokPerSec, $row.MeanMsPerTok,
        $row.P50Ms, $row.P90Ms, $row.PrefillMs)
    Write-Host ("        intervals: 1-64 {0} tok/s   65-256 {1}   257-512 {2}   cache hit {3}%" -f `
        (Format-Metric $row.Tok_1_64), (Format-Metric $row.Tok_65_256), (Format-Metric $row.Tok_257_512),
        (Format-Metric $row.HitRatePct))
}

$rows | Format-Table -AutoSize | Out-String | Write-Host
$csv = Join-Path $work 'decode.csv'
$rows | Export-Csv -Path $csv -NoTypeInformation
Write-Host "wrote $csv"
