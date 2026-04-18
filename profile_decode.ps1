param(
    [string]$BuildDir = 'build',
    [string]$PresetsFile = 'perf\presets.json',
    [string]$Preset = 'phase_gate_q35_text',
    [string]$ModelAlias = '',
    [string]$OutputDir = '',
    [string]$Phase = 'manual',
    [Nullable[int]]$ProfileEvery = $null,
    [switch]$EnableDnnlVerbose,
    [hashtable]$EnvOverrides = @{}
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

$repoRoot = Get-AilaRepoRoot
$gitInfo = Get-AilaGitInfo -RepoRoot $repoRoot
$buildDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BuildDir
$buildMeta = Get-AilaBuildMetadata -BuildDir $buildDirPath
$config = Get-AilaPerfConfig -RepoRoot $repoRoot -PresetsFile $PresetsFile
$presetConfig = Get-AilaPreset -Config $config -PresetName $Preset

if ([string]::IsNullOrWhiteSpace($ModelAlias)) {
    $ModelAlias = $presetConfig.anchorModel
}
$resolvedModel = Get-AilaModelInfo -Config $config -Alias $ModelAlias -RepoRoot $repoRoot

$profileCase = $presetConfig.decodeProfile
if ($null -eq $profileCase) {
    throw "Preset '$Preset' does not define decodeProfile."
}

$profileEveryValue = if ($PSBoundParameters.ContainsKey('ProfileEvery') -and $null -ne $ProfileEvery) {
    [int]$ProfileEvery
}
else {
    [int]$profileCase.profileEvery
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = New-AilaOutputDir -RepoRoot $repoRoot -OutputRoot 'tmp\perf' -Phase $Phase -ShortCommit $gitInfo.shortCommit
}
else {
    $OutputDir = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputDir
}

Ensure-AilaDirectory -Path $OutputDir
Initialize-AilaOneApiEnvironment

$effectiveEnv = @{}
foreach ($key in $EnvOverrides.Keys) {
    $effectiveEnv[$key] = [string]$EnvOverrides[$key]
}
$effectiveEnv['AILA_PROFILE_Q35_DECODE'] = '1'
$effectiveEnv['AILA_PROFILE_Q35_DECODE_EVERY'] = [string]$profileEveryValue
if ($EnableDnnlVerbose.IsPresent) {
    $effectiveEnv['DNNL_VERBOSE'] = '1'
}

$args = @(
    '-m', $resolvedModel.path,
    '--bench',
    '--bench-pp', ([string][int]$profileCase.promptTokens),
    '--bench-tg', ([string][int]$profileCase.genTokens),
    '--bench-iters', ([string][int]$profileCase.benchIters),
    '--bench-warmup', ([string][int]$profileCase.warmupIters),
    '--seed', ([string][UInt64]$profileCase.seed),
    '-t', ([string][double]$profileCase.temperature),
    '-k', ([string][int]$profileCase.topK),
    '-p', ([string][double]$profileCase.topP)
)
if ($profileCase.mode -eq 'sample') {
    $args += '--bench-sample'
}
else {
    $args += '--bench-greedy'
}

$logPath = Join-Path $OutputDir 'decode_profile.log'
$run = Invoke-AilaProcess -Executable '.\Aila.exe' -ArgumentList $args -WorkingDirectory $buildDirPath -LogPath $logPath -EnvOverrides $effectiveEnv
if ($run.exitCode -ne 0) {
    throw "Decode profile run failed with exit code $($run.exitCode). See $logPath"
}

$samples = Parse-AilaDecodeProfiles -OutputText $run.outputText
$summary = Get-AilaDecodeProfileSummary -Samples $samples
$summaryPath = Join-Path $OutputDir 'decode_profile_summary.json'

Write-AilaJsonFile -Path $summaryPath -Data ([ordered]@{
    schemaVersion = 1
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    phase         = $Phase
    preset        = $Preset
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
        alias       = $resolvedModel.alias
        path        = $resolvedModel.path
        description = $resolvedModel.description
    }
    case          = [ordered]@{
        name         = $profileCase.name
        promptTokens = [int]$profileCase.promptTokens
        genTokens    = [int]$profileCase.genTokens
        benchIters   = [int]$profileCase.benchIters
        warmupIters  = [int]$profileCase.warmupIters
        mode         = $profileCase.mode
        temperature  = [double]$profileCase.temperature
        topK         = [int]$profileCase.topK
        topP         = [double]$profileCase.topP
        seed         = [UInt64]$profileCase.seed
        profileEvery = $profileEveryValue
    }
    commandLine   = $run.commandLine
    rawLogPath    = $logPath
    envOverrides  = $effectiveEnv
    sampleCount   = $summary.sampleCount
    average       = $summary.average
    hotspots      = $summary.hotspots
    lastSample    = $summary.lastSample
    samples       = $samples
})

$topHotspots = @($summary.hotspots | Select-Object -First 5)
Write-Host (":: decode profile samples: {0} ::" -f $summary.sampleCount) -ForegroundColor Cyan
foreach ($hotspot in $topHotspots) {
    Write-Host ("   {0,-14} {1,8:N3} ms" -f $hotspot.stage, [double]$hotspot.avgMs)
}
Write-Host (":: decode profile summary written to {0} ::" -f $summaryPath) -ForegroundColor Green
