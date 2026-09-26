param(
    [string]$Tag = 'run',
    [int]$MaxTokens = 48,
    [string]$Messages = "$PSScriptRoot/prompts/story.json",
    [hashtable]$EnvVars = @{},
    [string]$ModelDir = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [string]$BuildDir = 'build-refactor',
    [int]$MaxSeq = 512,
    # Fixed-length decode that ignores EOS (--bench); summarizes the measured decode only.
    [switch]$Bench,
    [int]$BenchPP = 32
)
# Runs one profiled Qwen3.8-Flash-Next generation and prints a one-line summary of the
# [Q4E Profile] decode rows (per-token averages). The raw log goes to tmp/q4e_prof_<Tag>.log.
#
#   pwsh perf/qwen4exp/q4e_prof.ps1 -Tag base
#   pwsh perf/qwen4exp/q4e_prof.ps1 -Tag long -MaxTokens 24 -Messages perf/qwen4exp/prompts/lighthouse_274.json
#   pwsh perf/qwen4exp/q4e_prof.ps1 -Tag floor -Bench -MaxTokens 64 -EnvVars @{AILA_DEBUG_Q4E_FAKE_HITS='1'}
#
# The first run after a rebuild is slower (the build evicts model pages from the OS page
# cache), so compare second runs.
$repo = Resolve-Path "$PSScriptRoot/../.."
$exe = Join-Path $repo "$BuildDir/Aila.exe"
$logDir = Join-Path $repo 'tmp'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "q4e_prof_$Tag.log"

$env:AILA_PROFILE_Q4E_DECODE = '1'
foreach ($k in $EnvVars.Keys) { Set-Item -Path "env:$k" -Value $EnvVars[$k] }
try {
    if ($Bench) {
        & $exe -m $ModelDir --max-seq $MaxSeq --bench --bench-pp $BenchPP --bench-tg $MaxTokens `
            --bench-iters 1 --bench-warmup 1 --bench-greedy *> $log
    } else {
        & $exe -m $ModelDir --max-seq $MaxSeq --greedy --thinking-budget 0 `
            --max-tokens $MaxTokens --messages-json $Messages *> $log
    }
} finally {
    foreach ($k in $EnvVars.Keys) { Remove-Item -Path "env:$k" -ErrorAction SilentlyContinue }
    Remove-Item env:AILA_PROFILE_Q4E_DECODE -ErrorAction SilentlyContinue
}

$rx = 'count=1 total=([\d.]+)ms router_sync=([\d.]+)ms acquire=([\d.]+)ms \(fill=([\d.]+)ms dma_wait=([\d.]+)ms pf_wait=([\d.]+)ms pf=(\d+)/(\d+)\) hit=(\d+)/(\d+) xfer=([\d.]+)MB'
$rows = foreach ($line in (Get-Content $log)) {
    if ($line -match $rx) {
        [pscustomobject]@{ total=[double]$Matches[1]; sync=[double]$Matches[2]; acq=[double]$Matches[3];
            fill=[double]$Matches[4]; dma=[double]$Matches[5]; pfw=[double]$Matches[6]; pfu=[int]$Matches[7];
            pfi=[int]$Matches[8]; hit=[int]$Matches[9]; n=[int]$Matches[10]; mb=[double]$Matches[11] }
    }
}
# Bench: keep the measured decode; generation: drop the first steps (cache still warming).
$rows = if ($Bench) { @($rows | Select-Object -Last ([Math]::Max(1, $MaxTokens - 4))) } else { @($rows | Select-Object -Skip 4) }
if ($rows.Count -eq 0) { Write-Output "[$Tag] no decode rows"; Get-Content $log | Select-Object -Last 10; return }
$avg = { param($p) ($rows | Measure-Object -Property $p -Average).Average }
$prefill = Select-String -Path $log -Pattern 'pos=0 count=(\d{2,}) total=([\d.]+)ms.*fill=([\d.]+)ms' | Select-Object -Last 1
$gen = (Select-String -Path $log -Pattern 'Generated=\d+ in [\d.]+ ms' | Select-Object -First 1).Matches.Value
$missMs = (& $avg 'fill') / [math]::Max(1, ((& $avg 'n') - (& $avg 'hit')))

# Write sidecar JSON metadata
$commit = & git -C $repo rev-parse HEAD 2>$null
$exeHash = if (Test-Path $exe) { (Get-FileHash -Path $exe -Algorithm SHA256).Hash } else { "" }
$modeLine = Select-String -Path $log -Pattern '\[Qwen4Exp\] Prefill MoE mode: ([^,]+), Router pred: (\S+)' | Select-Object -First 1
$poolLine = Select-String -Path $log -Pattern '\[Q4E ExpertCache\] slots=(\d+) slot_bytes=(\d+) pool=([\d.]+)MB' | Select-Object -First 1
$sidecar = [ordered]@{
    tag = $Tag
    timestamp = (Get-Date -Format 'o')
    commit = "$commit".Trim()
    exe_sha256 = $exeHash
    max_seq = $MaxSeq
    max_tokens = $MaxTokens
    bench = [bool]$Bench
    env_vars = $EnvVars
    prefill_moe_mode = if ($modeLine) { $modeLine.Matches[0].Groups[1].Value.Trim() } else { "unknown" }
    router_pred_mode = if ($modeLine) { $modeLine.Matches[0].Groups[2].Value.Trim() } else { "unknown" }
    expert_slots = if ($poolLine) { [int]$poolLine.Matches[0].Groups[1].Value } else { 0 }
    dev_pool_mb = if ($poolLine) { [double]$poolLine.Matches[0].Groups[3].Value } else { 0.0 }
    decode_avg_ms = [math]::Round((& $avg 'total'), 2)
    decode_tok_per_s = [math]::Round((1000.0 / (& $avg 'total')), 2)
}
$metaPath = Join-Path $logDir "q4e_prof_$Tag.meta.json"
$sidecar | ConvertTo-Json -Depth 4 | Set-Content -Path $metaPath -Encoding UTF8

'[{0}] steps={1} total={2:N1}ms sync={3:N1} acq={4:N1} fill={5:N1} dma={6:N1} pf_wait={14:N1} pf_used/issued={15:N1}/{16:N1} hit={7:N0}/{8:N0} xfer={9:N0}MB fill/miss={10:N3}ms tok/s={11:N2} | prefill: {12} | {13}' -f `
    $Tag, $rows.Count, (& $avg 'total'), (& $avg 'sync'), (& $avg 'acq'), (& $avg 'fill'), (& $avg 'dma'),
    (& $avg 'hit'), (& $avg 'n'), (& $avg 'mb'), $missMs, (1000.0 / (& $avg 'total')),
    ($(if ($prefill) { "count=$($prefill.Matches[0].Groups[1].Value) total=$($prefill.Matches[0].Groups[2].Value)ms fill=$($prefill.Matches[0].Groups[3].Value)ms" } else { 'n/a' })),
    $gen, (& $avg 'pfw'), (& $avg 'pfu'), (& $avg 'pfi')
