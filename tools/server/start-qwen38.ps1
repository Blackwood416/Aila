[CmdletBinding()]
param(
    [string]$Model = 'D:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [int]$Port = 8080,
    [int]$Context = 32768,
    [int]$PrefillChunk = 8192,
    [int]$MaxNewTokens = 2048,
    [ValidateSet('auto', 'bf16', 'int8')]
    [string]$KvType = 'auto',
    [int]$VramReserveMb = 1400,
    [int]$MtpMaxContext = 32768,
    [string]$HotsetProfile = '',
    [string]$ShutdownFile = '',
    [string]$DefaultTemperature = 'auto',
    [switch]$EnableMtp,
    [switch]$Warmup,
    [switch]$DisableReuse
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$env:AILA_Q4E_VRAM_RESERVE_MB = [string]$VramReserveMb
if ($Warmup -and -not (Test-Path Env:AILA_INIT_WARMUP)) {
    $env:AILA_INIT_WARMUP = '1'
}
if (-not [string]::IsNullOrWhiteSpace($HotsetProfile)) {
    $resolvedProfile = if ([IO.Path]::IsPathRooted($HotsetProfile)) {
        $HotsetProfile
    } else {
        Join-Path $repoRoot $HotsetProfile
    }
    if (-not (Test-Path -LiteralPath $resolvedProfile)) {
        throw "Hotset profile not found: $resolvedProfile"
    }
    $env:AILA_Q4E_HOTSET_PROFILE = $resolvedProfile
}

$resolvedDefaultTemp = if ($DefaultTemperature -ne 'auto') {
    $DefaultTemperature
} elseif ($EnableMtp) {
    '0'
} else {
    '0.6'
}

$serverPath = Join-Path $repoRoot 'build-refactor-server/AilaServer.exe'
$workerPath = Join-Path $repoRoot 'build-refactor/AilaWorker.exe'
$serverArgs = @(
    '--worker', $workerPath,
    '--model', $Model,
    '--backend', 'qwen4exp_nf4',
    '--kv-type', $KvType,
    '--model-id', 'qwen4exp',
    '--port', [string]$Port,
    '--context', [string]$Context,
    '--sessions', '1',
    '--decode-batch-size', '1',
    '--prefill-chunk-size', [string]$PrefillChunk,
    '--max-new-tokens', [string]$MaxNewTokens,
    '--default-temperature', $resolvedDefaultTemp,
    '--max-body-bytes', '4194304',
    '--reuse-sessions', ([string][int](-not $DisableReuse)),
    '--request-timeout-ms', '600000'
)
if ($EnableMtp) {
    $serverArgs += @('--mtp', '--mtp-max-context', [string]([Math]::Min($Context, $MtpMaxContext)))
} else {
    $serverArgs += @('--no-mtp')
}
if (-not [string]::IsNullOrWhiteSpace($ShutdownFile)) {
    $serverArgs += @('--shutdown-file', $ShutdownFile)
}
& $serverPath @serverArgs
exit $LASTEXITCODE
