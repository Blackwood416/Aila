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
# The worker applies the same preset when it sees this marker. The preset only
# fills flags that are unset, so anything below still wins.
$env:AILA_Q4E_ACTIVE_PRESET = 'qwen38-a770'
$env:AILA_Q4E_SERVER_MTP = if ($EnableMtp) { '1' } else { '0' }
$env:AILA_Q4E_VRAM_RESERVE_MB = [string]$VramReserveMb

if ($EnableMtp -and -not (Test-Path Env:AILA_Q4E_MTP_MAX_CONTEXT)) {
    $env:AILA_Q4E_MTP_MAX_CONTEXT = [string]([Math]::Min($Context, $MtpMaxContext))
}
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

# Past 64K tokens the bf16 KV cache no longer fits comfortably in 16 GB VRAM
# next to the expert hotset, so 262K context needs int8 KV unless the caller
# pinned a format explicitly. Both worker env and server --kv-type must agree.
$resolvedKvType = if ($KvType -ne 'auto') {
    $KvType
} elseif ($Context -gt 65536 -or ($env:AILA_KV_QUANT -eq '1' -and $env:AILA_KV_QUANT_FORMAT -eq 'int8')) {
    'int8'
} else {
    'bf16'
}
if ($resolvedKvType -eq 'int8') {
    $env:AILA_KV_QUANT = '1'
    $env:AILA_KV_QUANT_FORMAT = 'int8'
} else {
    $env:AILA_KV_QUANT = '0'
}

# Standard Level Zero / SYCL runtime environment aligned with CLI benchmarks.
if ($env:AILA_KEEP_SYCL_CACHE_PERSISTENT -ne '1') {
    $env:SYCL_CACHE_PERSISTENT = '0'
}
if (-not (Test-Path Env:UR_L0_USE_COPY_ENGINE)) { $env:UR_L0_USE_COPY_ENGINE = '1' }
if (-not (Test-Path Env:SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE)) { $env:SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE = '1' }
if (-not (Test-Path Env:UR_L0_USE_IMMEDIATE_COMMANDLISTS)) { $env:UR_L0_USE_IMMEDIATE_COMMANDLISTS = '0' }
if (-not (Test-Path Env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS)) { $env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = '0' }
if (-not (Test-Path Env:SYCL_UR_USE_LEVEL_ZERO_V2)) { $env:SYCL_UR_USE_LEVEL_ZERO_V2 = '0' }
if (-not (Test-Path Env:UR_L0_BATCH_SIZE)) { $env:UR_L0_BATCH_SIZE = '64' }
if (-not (Test-Path Env:SYCL_PI_LEVEL_ZERO_BATCH_SIZE)) { $env:SYCL_PI_LEVEL_ZERO_BATCH_SIZE = '64' }
$requiredPassthrough = @('UR_L0_USE_COPY_ENGINE', 'UR_L0_USE_IMMEDIATE_COMMANDLISTS', 'UR_L0_BATCH_SIZE')
$existingPassthrough = @()
if (-not [string]::IsNullOrWhiteSpace($env:AILA_WORKER_ENV_PASSTHROUGH)) {
    $existingPassthrough = @($env:AILA_WORKER_ENV_PASSTHROUGH -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
foreach ($name in $requiredPassthrough) {
    if ($existingPassthrough -notcontains $name) { $existingPassthrough += $name }
}
$env:AILA_WORKER_ENV_PASSTHROUGH = ($existingPassthrough -join ';')

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
    '--kv-type', $resolvedKvType,
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
if (-not [string]::IsNullOrWhiteSpace($ShutdownFile)) {
    $serverArgs += @('--shutdown-file', $ShutdownFile)
}
& $serverPath @serverArgs
exit $LASTEXITCODE
