[CmdletBinding()]
param(
    [string]$Model = 'D:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [int]$Port = 8080,
    [int]$Context = 32768,
    [int]$PrefillChunk = 8192,
    [int]$MaxNewTokens = 2048,
    [switch]$EnableMtp,
    [switch]$DisableReuse
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
# The worker applies the same preset when it sees this marker. The preset only
# fills flags that are unset, so anything below still wins.
$env:AILA_Q4E_ACTIVE_PRESET = 'qwen38-a770'
# MTP stays opt-in: the draft head shares the target's QSA/head arithmetic but
# has not passed the long-context numerical acceptance gate.
$env:AILA_Q4E_SERVER_MTP = if ($EnableMtp) { '1' } else { '0' }
# Measured on A770: leave room for the lazy PLE workspace to grow during
# long prefill instead of assigning every released byte to expert residency.
$env:AILA_Q4E_VRAM_RESERVE_MB = '3584'
$hotsetProfile = Join-Path $repoRoot 'perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile'
if (Test-Path -LiteralPath $hotsetProfile) {
    $env:AILA_Q4E_HOTSET_PROFILE = $hotsetProfile
}
# Past 64K tokens the bf16 KV cache no longer fits comfortably in 16 GB VRAM
# next to the expert hotset, so 262K context needs int8 KV unless the caller
# pinned a format explicitly.
if ($Context -gt 65536 -and -not (Test-Path Env:AILA_KV_QUANT)) {
    $env:AILA_KV_QUANT = '1'
    $env:AILA_KV_QUANT_FORMAT = 'int8'
}
# A770/Haswell paired screening: faster short continuations with the same
# two staging banks. Preserve an explicit caller override (including '0').
if (-not (Test-Path Env:AILA_Q4E_STAGE_COPY_PARALLEL)) {
    $env:AILA_Q4E_STAGE_COPY_PARALLEL = '1'
}
# A770 paired screening: 4-token quads in a workgroup share weight rows for
# different quads, cutting 14K first-token latency ~2.9% with bit-identical
# outputs. Preserve an explicit caller override (including '0').
if (-not (Test-Path Env:AILA_Q4E_4TOKEN_WGSHARE)) {
    $env:AILA_Q4E_4TOKEN_WGSHARE = '4'
}
$serverPath = Join-Path $repoRoot 'build-refactor-server/AilaServer.exe'
$workerPath = Join-Path $repoRoot 'build-refactor/AilaWorker.exe'
& $serverPath --worker $workerPath --model $Model --backend qwen4exp_nf4 --model-id qwen4exp `
    --port $Port --context $Context --sessions 1 --decode-batch-size 1 `
    --prefill-chunk-size $PrefillChunk --max-new-tokens $MaxNewTokens --max-body-bytes 4194304 `
    --reuse-sessions ([int](-not $DisableReuse)) --request-timeout-ms 600000
exit $LASTEXITCODE
