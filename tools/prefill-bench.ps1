<#
.SYNOPSIS
Measures prefill throughput (prompt tokens per second) with the plain CLI.

.DESCRIPTION
The A/B runner used for decode work is not a trustworthy instrument for prefill:
it forces its own hotset profile, enables several trace flags, and its
GenerateMessages wall time picks up one-time costs that are not prefill. This
script drives Aila.exe directly, parses the two lines the engine prints

    [GenerateMessages] Full prefill: N tokens
    [GenerateMessages] Prompt=N Generated=1 in T ms

and reports N / T. Generation is limited to one token so the number is prefill.

.PARAMETER Tokens
Approximate prompt size. The prompt is built from repeated sentences, so the
engine reports the exact token count and that is what the throughput uses.

.PARAMETER Chunk
Value for AILA_Q35_PREFILL_CHUNK. The qwen38-a770 preset sets 512; 2048 keeps a
2K-token prompt in a single chunk, which avoids paying expert staging twice.
#>
param(
    [int[]]$Tokens = @(1024, 2048, 4096),
    [int]$Chunk = 2048,
    [string]$Model = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [string]$Exe = 'build-refactor/Aila.exe'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$work = Join-Path $repo 'tmp/prefill-bench'
New-Item -ItemType Directory -Force -Path $work | Out-Null

# Baseline environment the decode work already established as neutral: no
# immediate command lists, copy engine on, persistent SYCL cache.
$env:UR_L0_USE_COPY_ENGINE = '1'
$env:SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE = '1'
$env:UR_L0_USE_IMMEDIATE_COMMANDLISTS = '0'
$env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = '0'
$env:SYCL_UR_USE_LEVEL_ZERO_V2 = '0'
$env:SYCL_CACHE_PERSISTENT = '1'
$env:AILA_Q35_PREFILL_CHUNK = [string]$Chunk

$sentence = 'The quick brown fox jumps over the lazy dog and then writes a long report about inference engines. '
$rows = @()
foreach ($target in $Tokens) {
    $reps = [Math]::Max(1, [int][Math]::Round($target / 19.3))
    $prompt = Join-Path $work "prompt-$target.json"
    $body = @{ messages = @(@{ role = 'user'; content = ($sentence * $reps) }); max_tokens = 1 } |
        ConvertTo-Json -Depth 5 -Compress
    Set-Content -Path $prompt -Value $body -Encoding UTF8

    $out = & $Exe -m $Model --performance-preset qwen38-a770 --max-seq 4096 --greedy `
        --thinking-budget 0 --max-tokens 1 --messages-json $prompt `
        --no-stream --chat-output-json --log-level info 2>&1 | Out-String

    $n = 0; $ms = 0.0
    if ($out -match 'Full prefill: (\d+) tokens') { $n = [int]$Matches[1] }
    if ($out -match 'GenerateMessages\] Prompt=\d+ Generated=\d+ in ([\d.]+) ms') { $ms = [double]$Matches[1] }
    if ($n -le 0 -or $ms -le 0) {
        Write-Warning "could not parse a measurement for target $target"
        continue
    }
    $rows += [pscustomobject]@{
        Tokens  = $n
        WallMs  = [Math]::Round($ms, 1)
        MsPerTok = [Math]::Round($ms / $n, 3)
        TokPerSec = [Math]::Round($n / ($ms / 1000.0), 1)
    }
    Write-Host ("{0,6} tokens  {1,10:N1} ms  {2,7:N3} ms/token  {3,7:N1} tok/s" -f `
        $n, $ms, ($ms / $n), ($n / ($ms / 1000.0)))
}

$rows | Format-Table -AutoSize | Out-String | Write-Host
$csv = Join-Path $work 'prefill.csv'
$rows | Export-Csv -Path $csv -NoTypeInformation
Write-Host "wrote $csv"

# Marginal cost per token and the fixed per-run cost, from a linear fit over the
# measurements. The goal is expressed as marginal throughput: 400 tok/s means
# 2.5 ms per prompt token.
if ($rows.Count -ge 2) {
    $a = $rows[0]; $b = $rows[-1]
    $slopeMs = ($b.WallMs - $a.WallMs) / ($b.Tokens - $a.Tokens)
    $fixedMs = $a.WallMs - $slopeMs * $a.Tokens
    Write-Host ("marginal {0:N2} ms/token  ->  {1:N0} tok/s   fixed per run {2:N0} ms" -f `
        $slopeMs, (1000.0 / $slopeMs), $fixedMs)
    Write-Host ("target 400 tok/s needs marginal 2.50 ms/token; now {0:N2}" -f $slopeMs)
}
