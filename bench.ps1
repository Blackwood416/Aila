param(
    [string]$BuildDir = 'build',
    [string]$PresetsFile = 'perf\presets.json',
    [string]$Preset = '',
    [string]$ModelAlias = '',
    [string]$ModelDir = '..\Qwen3.5-0.8B',
    [string]$OutputDir = '',
    [string]$Phase = 'manual',
    [string[]]$CaseNames = @(),
    [int]$PromptTokens = 512,
    [int]$GenTokens = 512,
    [int]$BenchIters = 5,
    [int]$WarmupIters = 1,
    [switch]$Sample,
    [double]$Temperature = 0.7,
    [int]$TopK = 15,
    [double]$TopP = 0.95,
    [UInt64]$Seed = 42,
    [hashtable]$EnvOverrides = @{},
    [int]$WaitTimeoutSec = 3600
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

$mutexName = 'Global\AilaBenchSingleton'
$mutex = [System.Threading.Mutex]::new($false, $mutexName)
$hasMutex = $false

function Acquire-BenchMutex {
    param(
        [System.Threading.Mutex]$Mutex,
        [int]$TimeoutSec
    )

    try {
        if ($Mutex.WaitOne(0)) {
            return $true
        }
    }
    catch [System.Threading.AbandonedMutexException] {
        Write-Host ':: benchmark mutex was abandoned, recovering lock ::' -ForegroundColor Yellow
        return $true
    }

    if ($TimeoutSec -lt 0) {
        Write-Host ':: another benchmark is running, waiting for lock ::' -ForegroundColor Yellow
        try {
            $Mutex.WaitOne() | Out-Null
            return $true
        }
        catch [System.Threading.AbandonedMutexException] {
            Write-Host ':: benchmark mutex was abandoned while waiting, recovering lock ::' -ForegroundColor Yellow
            return $true
        }
    }

    Write-Host ":: another benchmark is running, waiting up to $TimeoutSec s for lock ::" -ForegroundColor Yellow
    try {
        return $Mutex.WaitOne([TimeSpan]::FromSeconds($TimeoutSec))
    }
    catch [System.Threading.AbandonedMutexException] {
        Write-Host ':: benchmark mutex was abandoned while waiting, recovering lock ::' -ForegroundColor Yellow
        return $true
    }
}

function Resolve-ModelPathForBench {
    param(
        [string]$RepoRoot,
        [string]$BuildDirPath,
        [string]$InputPath
    )

    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return [System.IO.Path]::GetFullPath($InputPath)
    }

    $repoCandidate = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $InputPath))
    if (Test-Path -LiteralPath $repoCandidate) {
        return $repoCandidate
    }

    $buildCandidate = [System.IO.Path]::GetFullPath((Join-Path $BuildDirPath $InputPath))
    if (Test-Path -LiteralPath $buildCandidate) {
        return $buildCandidate
    }

    return $repoCandidate
}

$repoRoot = Get-AilaRepoRoot
$gitInfo = Get-AilaGitInfo -RepoRoot $repoRoot
$buildDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BuildDir
$buildMeta = Get-AilaBuildMetadata -BuildDir $buildDirPath

if ($buildMeta.buildType -and $buildMeta.buildType -ne 'Release') {
    Write-Host ":: warning: build dir is configured as $($buildMeta.buildType); benchmark numbers may be much slower than Release ::" -ForegroundColor Yellow
}

$cases = @()
$resolvedModel = $null
$config = $null

if (-not [string]::IsNullOrWhiteSpace($Preset)) {
    $config = Get-AilaPerfConfig -RepoRoot $repoRoot -PresetsFile $PresetsFile
    $presetConfig = Get-AilaPreset -Config $config -PresetName $Preset

    if ([string]::IsNullOrWhiteSpace($ModelAlias)) {
        $ModelAlias = $presetConfig.anchorModel
    }
    $resolvedModel = Get-AilaModelInfo -Config $config -Alias $ModelAlias -RepoRoot $repoRoot

    foreach ($case in $presetConfig.benchmarks) {
        if ($CaseNames.Count -gt 0 -and ($CaseNames -notcontains $case.name)) {
            continue
        }
        $cases += [pscustomobject]@{
            name        = $case.name
            promptTokens = [int]$case.promptTokens
            genTokens   = [int]$case.genTokens
            benchIters  = [int]$case.benchIters
            warmupIters = [int]$case.warmupIters
            mode        = $case.mode
            temperature = [double]$case.temperature
            topK        = [int]$case.topK
            topP        = [double]$case.topP
            seed        = [UInt64]$case.seed
        }
    }
}
else {
    $resolvedModel = [pscustomobject]@{
        alias      = if ([string]::IsNullOrWhiteSpace($ModelAlias)) { 'manual' } else { $ModelAlias }
        path       = Resolve-ModelPathForBench -RepoRoot $repoRoot -BuildDirPath $buildDirPath -InputPath $ModelDir
        maxSeqLen  = $null
        description = 'Manual benchmark model'
    }

    $cases += [pscustomobject]@{
        name        = if ($Sample.IsPresent) { 'manual_sample' } else { 'manual_greedy' }
        promptTokens = $PromptTokens
        genTokens   = $GenTokens
        benchIters  = $BenchIters
        warmupIters = $WarmupIters
        mode        = if ($Sample.IsPresent) { 'sample' } else { 'greedy' }
        temperature = $Temperature
        topK        = $TopK
        topP        = $TopP
        seed        = $Seed
    }
}

if ($cases.Count -eq 0) {
    throw 'No benchmark cases selected.'
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $baseDir = New-AilaOutputDir -RepoRoot $repoRoot -OutputRoot 'tmp\perf' -Phase $Phase -ShortCommit $gitInfo.shortCommit
    $OutputDir = $baseDir
}
else {
    $OutputDir = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputDir
}
Ensure-AilaDirectory -Path $OutputDir
Ensure-AilaDirectory -Path (Join-Path $OutputDir 'bench_logs')

$globalLogPath = Join-Path $repoRoot 'bench_log.txt'
Initialize-AilaOneApiEnvironment

$hasMutex = Acquire-BenchMutex -Mutex $mutex -TimeoutSec $WaitTimeoutSec
if (-not $hasMutex) {
    throw "Timed out waiting for benchmark lock after $WaitTimeoutSec seconds."
}

try {
    $caseResults = @()
    foreach ($case in $cases) {
        $header = "=== bench $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') case=$($case.name) preset=$Preset model=$($resolvedModel.alias) path=$($resolvedModel.path) pp=$($case.promptTokens) tg=$($case.genTokens) iters=$($case.benchIters) warmup=$($case.warmupIters) mode=$($case.mode) seed=$($case.seed) temp=$($case.temperature) topk=$($case.topK) topp=$($case.topP) ==="
        Add-Content -LiteralPath $globalLogPath -Value $header -Encoding UTF8

        $logPath = Join-Path (Join-Path $OutputDir 'bench_logs') ("{0}.log" -f $case.name)
        $args = @(
            '-m', $resolvedModel.path,
            '--bench',
            '--bench-pp', ([string]$case.promptTokens),
            '--bench-tg', ([string]$case.genTokens),
            '--bench-iters', ([string]$case.benchIters),
            '--bench-warmup', ([string]$case.warmupIters),
            '--seed', ([string]$case.seed),
            '-t', ([string]$case.temperature),
            '-k', ([string]$case.topK),
            '-p', ([string]$case.topP)
        )
        if ($case.mode -eq 'sample') {
            $args += '--bench-sample'
        }
        else {
            $args += '--bench-greedy'
        }

        $run = Invoke-AilaProcess -Executable '.\Aila.exe' -ArgumentList $args -WorkingDirectory $buildDirPath -LogPath $logPath -EnvOverrides $EnvOverrides
        if ($run.exitCode -ne 0) {
            throw "Benchmark case '$($case.name)' failed with exit code $($run.exitCode). See $logPath"
        }

        Add-Content -LiteralPath $globalLogPath -Value $run.outputText -Encoding UTF8
        $parsed = Parse-AilaBenchmarkOutput -OutputText $run.outputText

        $caseResults += [ordered]@{
            name         = $case.name
            mode         = $case.mode
            promptTokens = $case.promptTokens
            genTokens    = $case.genTokens
            benchIters   = $case.benchIters
            warmupIters  = $case.warmupIters
            temperature  = $case.temperature
            topK         = $case.topK
            topP         = $case.topP
            seed         = $case.seed
            commandLine  = $run.commandLine
            envOverrides = $EnvOverrides
            prefill      = $parsed.prefill
            decode       = $parsed.decode
            rawLogPath   = $logPath
        }
    }

    $benchJsonPath = Join-Path $OutputDir 'bench.json'
    Write-AilaJsonFile -Path $benchJsonPath -Data ([ordered]@{
        schemaVersion = 1
        generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
        phase         = $Phase
        preset        = if ([string]::IsNullOrWhiteSpace($Preset)) { 'manual' } else { $Preset }
        git           = [ordered]@{
            shortCommit = $gitInfo.shortCommit
            fullCommit  = $gitInfo.fullCommit
            branch      = $gitInfo.branch
        }
        build         = [ordered]@{
            buildDir   = $buildMeta.buildDir
            buildType  = $buildMeta.buildType
            compiler   = $buildMeta.compiler
            generator  = $buildMeta.generator
        }
        model         = [ordered]@{
            alias      = $resolvedModel.alias
            path       = $resolvedModel.path
            description = $resolvedModel.description
        }
        cases         = $caseResults
    })

    Write-Host (":: benchmark results written to {0} ::" -f $benchJsonPath) -ForegroundColor Green
}
finally {
    if ($hasMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
