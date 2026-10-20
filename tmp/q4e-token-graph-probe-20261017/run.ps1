# Runs the P0/P3 device-doorbell probe under Aila's production Level Zero knobs.
# This is opt-in and is not invoked by the build script.
param(
    [uint32]$Replays = 1000,
    [string]$CpuUs = '0,50,200,500',
    [uint32]$SpinMax = 2000000,
    [int]$HostTimeoutMs = 2000,
    [int]$Warmup = 10,
    [string]$Model = '',
    [switch]$TwoLayer,
    [ValidateSet('scalar', 'wg_stride')]
    [string]$PacketCopyMode = 'scalar',
    [ValidateRange(32, 1024)]
    [int]$PacketWg = 256,
    [string]$Log = ''
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$busy = Get-Process AilaWorker,AilaServer,Aila,probe_token_graph `
    -ErrorAction SilentlyContinue
if ($busy) {
    throw "GPU busy: $($busy | ForEach-Object { $_.Name + ':' + $_.Id })"
}

$os = Get-CimInstance Win32_OperatingSystem
$freeGiB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
if ($freeGiB -lt 45) {
    throw "Only $freeGiB GiB physical memory free; require >=45 GiB"
}

. (Join-Path $repo 'perf/PerfCommon.ps1')
$config = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

$prod = @{
    AILA_INIT_WARMUP = '0'
    UR_L0_USE_COPY_ENGINE = '1'
    SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE = '1'
    UR_L0_USE_IMMEDIATE_COMMANDLISTS = '0'
    SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = '0'
    SYCL_UR_USE_LEVEL_ZERO_V2 = '0'
    UR_L0_BATCH_SIZE = '0'
    SYCL_PI_LEVEL_ZERO_BATCH_SIZE = '0'
    SYCL_CACHE_PERSISTENT = '0'
}
foreach ($k in $prod.Keys) {
    [Environment]::SetEnvironmentVariable($k, $prod[$k], 'Process')
}

$exe = Join-Path $PSScriptRoot 'probe_token_graph.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "Missing $exe; run build.ps1 first"
}
$argsList = @(
    '--replays', "$Replays",
    '--warmup', "$Warmup",
    '--cpu-us', $CpuUs,
    '--spin-max', "$SpinMax",
    '--host-timeout-ms', "$HostTimeoutMs"
)
if ($Model) {
    $argsList += @('--real-front-model', $Model)
}
if ($TwoLayer) {
    $argsList += '--two-layer'
}
if ($PacketCopyMode -ne 'scalar' -and -not $TwoLayer) {
    throw "-PacketCopyMode wg_stride requires -TwoLayer"
}
if ($PacketCopyMode -ne 'scalar') {
    $argsList += @('--packet-copy-mode', $PacketCopyMode,
                   '--packet-wg', "$PacketWg")
}
$envLine = ($prod.Keys | Sort-Object | ForEach-Object {
    "$_=" + [Environment]::GetEnvironmentVariable($_, 'Process')
}) -join ' '
$header = "[RUN] $(Get-Date -Format o) exe=$exe args=$($argsList -join ' ') " +
          "free_gib=$freeGiB env: $envLine"

if ($Log) {
    $header | Tee-Object -FilePath $Log
} else {
    $header
}
$sw = [Diagnostics.Stopwatch]::StartNew()
& $exe @argsList 2>&1 | ForEach-Object {
    if ($Log) { $_ | Tee-Object -FilePath $Log -Append } else { $_ }
}
$code = $LASTEXITCODE
$footer = "[RUN] exit=$code wall_s=$([math]::Round($sw.Elapsed.TotalSeconds, 1))"
if ($Log) {
    $footer | Tee-Object -FilePath $Log -Append
} else {
    $footer
}
exit $code
