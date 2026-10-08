[CmdletBinding()]
param(
    [string]$Model = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [int]$Port = 8080,
    [int]$Context = 32768,
    [int]$PrefillChunk = 4096,
    [int]$MaxNewTokens = 2048,
    [switch]$DisableReuse
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$profilePath = Join-Path $repoRoot 'docs/plans/qwen38-server-mtp-2026-10-07.json'
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
foreach ($entry in $profile.worker_environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($entry.Name, [string]$entry.Value, 'Process')
}
# Long-context agent serving uses T1 until MTP's numerical alignment and sparse
# draft attention are validated. Keep the existing compute optimizations.
$env:AILA_Q4E_SERVER_MTP = '0'
# Measured on A770: leave room for the lazy PLE workspace to grow during
# long prefill instead of assigning every released byte to expert residency.
$env:AILA_Q4E_VRAM_RESERVE_MB = '3584'
$env:AILA_Q4E_HOTSET_PROFILE = Join-Path $repoRoot 'perf/qwen4exp/profiles/qwen38-nf4-h4-v1.profile'
$serverPath = Join-Path $repoRoot 'build-refactor-server/AilaServer.exe'
$workerPath = Join-Path $repoRoot 'build-refactor/AilaWorker.exe'
& $serverPath --worker $workerPath --model $Model --backend qwen4exp_nf4 --model-id qwen4exp `
    --port $Port --context $Context --sessions 1 --decode-batch-size 1 `
    --prefill-chunk-size $PrefillChunk --max-new-tokens $MaxNewTokens --max-body-bytes 4194304 `
    --reuse-sessions ([int](-not $DisableReuse)) --request-timeout-ms 600000
exit $LASTEXITCODE
