[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ModelDir,
    [string]$BuildDir = 'build-refactor',
    [string]$OutputDir = 'tmp/refactor/q3-scheduler-bench',
    [string]$OneApiStack = 'oneapi-2026.1',
    [hashtable]$EnvOverrides = @{},
    [ValidateRange(1, 3600)][int]$TimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'PerfCommon.ps1')

$repoRoot = Get-AilaRepoRoot
$buildPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BuildDir
$modelPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $ModelDir
$outputRoot = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputDir
$stackConfig = Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $stackConfig -Name $OneApiStack
Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $buildPath -Stack $stack
Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildPath -Stack $stack -RequireValues
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

$executable = Join-Path $buildPath 'AilaQwen3SchedulerTests.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw "Build AilaQwen3SchedulerTests first: $executable" }
if (-not (Test-Path -LiteralPath $modelPath -PathType Container)) { throw "Model directory does not exist: $modelPath" }
$runName = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runPath = Join-Path $outputRoot $runName
Ensure-AilaDirectory -Path $runPath
$metricsPath = Join-Path $runPath 'benchmark.json'
$reportPath = Join-Path $runPath 'run.json'
$logPath = Join-Path $runPath 'benchmark.log'
$sourcePatchPath = Join-Path $runPath 'tracked-source.patch'

$runEnvironment = @{
    AILA_KV_QUANT = '0'
    AILA_LOG_LEVEL = 'error'
    AILA_PROFILE_Q3_DECODE = '0'
    AILA_PROFILE_Q3_PREFILL = '0'
    AILA_PROFILE_Q3_HOST_ONLY = '0'
}
foreach ($key in $EnvOverrides.Keys) { $runEnvironment[$key] = [string]$EnvOverrides[$key] }
$effectiveEnvironment = [ordered]@{}
Get-ChildItem Env: | Where-Object { $_.Name -match '^(AILA_|SYCL_|ONEAPI_|DNNL_|ONEDNN_|UR_|ZE_)' } |
    Sort-Object Name | ForEach-Object { $effectiveEnvironment[$_.Name] = $_.Value }
foreach ($key in $runEnvironment.Keys) { $effectiveEnvironment[$key] = $runEnvironment[$key] }

# Keep an exact source/binary identity even when this benchmark is run before
# committing a refactor. New inference files are included by explicit hashes.
& git -C $repoRoot diff --binary HEAD --output $sourcePatchPath
if ($LASTEXITCODE -ne 0) { throw 'Unable to record the tracked source diff' }
$sourcePaths = @(
    'CMakeLists.txt',
    'src/models/Qwen3Bnb4Backend.hpp', 'src/models/Qwen3Bnb4Backend.cpp',
    'tests/inference/Qwen3SchedulerTests.cpp', 'perf/shared-session-bench.ps1'
)
$sourcePaths += @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'src/inference') -File |
    ForEach-Object { [IO.Path]::GetRelativePath($repoRoot, $_.FullName) })
$sourceFiles = @($sourcePaths | Sort-Object -Unique | ForEach-Object {
    $path = Join-Path $repoRoot $_
    [ordered]@{ path = $_; sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
})
$modelFiles = @(Get-ChildItem -LiteralPath $modelPath -File | Where-Object {
    $_.Extension -eq '.safetensors' -or $_.Name -in @(
        'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'generation_config.json', 'chat_template.jinja', 'model.safetensors.index.json')
} | Sort-Object Name | ForEach-Object {
    [ordered]@{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
})
$driver = @()
if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
    $driver = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate)
}
$arguments = @('--model', $modelPath, '--benchmark', $metricsPath)
$binaryHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
$report = [ordered]@{
    schemaVersion = 1
    startedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    passed = $false
    git = Get-AilaGitInfo -RepoRoot $repoRoot
    workingTreeStatus = @(& git -C $repoRoot status --short)
    trackedSourcePatch = $sourcePatchPath
    sourceFiles = $sourceFiles
    buildInfo = Read-AilaJsonFile -Path (Join-Path $buildPath 'build_info.json')
    executable = $executable
    executableSha256 = $binaryHash
    modelDirectory = $modelPath
    modelFiles = $modelFiles
    driver = $driver
    environment = $effectiveEnvironment
    arguments = $arguments
    metricsPath = $metricsPath
    logPath = $logPath
    execution = $null
}

# Coordinate with the legacy benchmark runner. CTest does not take this
# mutex: the integration owner must still serialize GPU correctness tests.
$benchmarkMutex = [Threading.Mutex]::new($false, 'Global\AilaBenchSingleton')
$ownsMutex = $false
try {
    try { $ownsMutex = $benchmarkMutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) { throw 'Another Aila benchmark is running; retry after it completes.' }
    $run = Invoke-AilaProcess -Executable $executable -ArgumentList $arguments -WorkingDirectory $buildPath `
        -EnvOverrides $runEnvironment -LogPath $logPath -TimeoutSeconds $TimeoutSeconds
    $report.execution = [ordered]@{
        exitCode = $run.exitCode; timedOut = $run.timedOut; durationMs = $run.durationMs
    }
    if ($run.timedOut -or $run.exitCode -ne 0) { throw "Scheduler benchmark failed; see $logPath" }
    $metrics = Read-AilaJsonFile -Path $metricsPath
    if ($metrics.schema -ne 'aila.qwen3.scheduler.benchmark.v1' -or $metrics.results.Count -ne 3) {
        throw 'Unexpected scheduler benchmark result schema'
    }
    if ((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash -ne $binaryHash) {
        throw 'The benchmark executable changed during execution'
    }
    $report.passed = $true
} finally {
    if ($ownsMutex) { $benchmarkMutex.ReleaseMutex() }
    $benchmarkMutex.Dispose()
    Write-AilaJsonFile -Path $reportPath -Data $report
    Write-AilaJsonFile -Path (Join-Path $outputRoot 'latest-run.json') -Data ([ordered]@{
        passed = $report.passed; reportPath = $reportPath; metricsPath = $metricsPath; logPath = $logPath
    })
}
Write-Host "Scheduler benchmark completed; report: $reportPath"
