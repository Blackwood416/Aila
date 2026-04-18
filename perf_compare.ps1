param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$BaselineDir,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$CurrentDir,

    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

function Resolve-AilaPerfArtifactPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StageDir,

        [Parameter(Mandatory = $true)]
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        $fullPath = Join-Path $StageDir $candidate
        if (Test-Path -LiteralPath $fullPath) {
            return $fullPath
        }
    }

    $joined = $Candidates -join ', '
    throw "None of the expected artifact paths exist under '$StageDir': $joined"
}

function Get-AilaBenchCaseMap {
    param(
        [Parameter(Mandatory = $true)]
        $BenchJson
    )

    $map = @{}
    foreach ($case in $BenchJson.cases) {
        $map[[string]$case.name] = $case
    }
    return $map
}

function Get-AilaPercentDelta {
    param(
        [double]$Baseline,
        [double]$Current
    )

    if ([math]::Abs($Baseline) -lt 1e-9) {
        return $null
    }

    return [math]::Round((($Current - $Baseline) / $Baseline) * 100.0, 4)
}

$repoRoot = Get-AilaRepoRoot
$baselineDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BaselineDir
$currentDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $CurrentDir

$baselineBenchPath = Resolve-AilaPerfArtifactPath -StageDir $baselineDirPath -Candidates @('bench.json', 'bench\bench.json')
$currentBenchPath = Resolve-AilaPerfArtifactPath -StageDir $currentDirPath -Candidates @('bench.json', 'bench\bench.json')
$baselineProfilePath = Resolve-AilaPerfArtifactPath -StageDir $baselineDirPath -Candidates @('decode_profile_summary.json', 'profile\decode_profile_summary.json')
$currentProfilePath = Resolve-AilaPerfArtifactPath -StageDir $currentDirPath -Candidates @('decode_profile_summary.json', 'profile\decode_profile_summary.json')

$baselineBench = Read-AilaJsonFile -Path $baselineBenchPath
$currentBench = Read-AilaJsonFile -Path $currentBenchPath
$baselineProfile = Read-AilaJsonFile -Path $baselineProfilePath
$currentProfile = Read-AilaJsonFile -Path $currentProfilePath

$baselineCases = Get-AilaBenchCaseMap -BenchJson $baselineBench
$currentCases = Get-AilaBenchCaseMap -BenchJson $currentBench
$commonCaseNames = @($baselineCases.Keys | Where-Object { $currentCases.ContainsKey($_) } | Sort-Object)

$benchComparisons = @()
foreach ($caseName in $commonCaseNames) {
    $baseCase = $baselineCases[$caseName]
    $currCase = $currentCases[$caseName]
    $benchComparisons += [ordered]@{
        name              = $caseName
        mode              = [string]$currCase.mode
        baselinePrefill   = [double]$baseCase.prefill.tokPerSec
        currentPrefill    = [double]$currCase.prefill.tokPerSec
        prefillDeltaPct   = Get-AilaPercentDelta -Baseline ([double]$baseCase.prefill.tokPerSec) -Current ([double]$currCase.prefill.tokPerSec)
        baselineDecode    = [double]$baseCase.decode.tokPerSec
        currentDecode     = [double]$currCase.decode.tokPerSec
        decodeDeltaPct    = Get-AilaPercentDelta -Baseline ([double]$baseCase.decode.tokPerSec) -Current ([double]$currCase.decode.tokPerSec)
    }
}

$stageNames = @(
    'total', 'embed', 'linear_proj', 'linear_delta', 'linear_o',
    'full_qkv', 'full_split', 'qk_rope', 'kv_copy', 'attn', 'attn_gate',
    'full_o', 'post_attn', 'ffn_proj', 'ffn_act', 'down', 'post_mlp', 'lm_head'
)

$profileComparisons = @()
foreach ($stage in $stageNames) {
    $baselineValue = [double]$baselineProfile.average.$stage
    $currentValue = [double]$currentProfile.average.$stage
    $profileComparisons += [ordered]@{
        stage          = $stage
        baselineAvgMs  = $baselineValue
        currentAvgMs   = $currentValue
        deltaMs        = [math]::Round(($currentValue - $baselineValue), 6)
        deltaPct       = Get-AilaPercentDelta -Baseline $baselineValue -Current $currentValue
    }
}

$largestStageChanges = @($profileComparisons | Sort-Object -Property @{ Expression = { [math]::Abs([double]$_.deltaMs) } } -Descending | Select-Object -First 8)

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $baselineLeaf = Split-Path -Leaf $baselineDirPath
    $OutputPath = Join-Path $currentDirPath ("compare_vs_{0}.json" -f $baselineLeaf)
}
else {
    $OutputPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputPath
}

$compareData = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    baselineDir    = $baselineDirPath
    currentDir     = $currentDirPath
    baseline       = [ordered]@{
        benchPath   = $baselineBenchPath
        profilePath = $baselineProfilePath
        git         = $baselineBench.git
    }
    current        = [ordered]@{
        benchPath   = $currentBenchPath
        profilePath = $currentProfilePath
        git         = $currentBench.git
    }
    benchCases     = $benchComparisons
    profileStages  = $profileComparisons
    largestStageChanges = $largestStageChanges
}

Write-AilaJsonFile -Path $OutputPath -Data $compareData

Write-Host ':: benchmark comparison ::' -ForegroundColor Cyan
foreach ($item in $benchComparisons) {
    Write-Host ("   {0,-14} pp {1,8:N2} -> {2,8:N2} tok/s ({3,8}%), tg {4,8:N2} -> {5,8:N2} tok/s ({6,8}%)" -f `
        $item.name,
        $item.baselinePrefill,
        $item.currentPrefill,
        $item.prefillDeltaPct,
        $item.baselineDecode,
        $item.currentDecode,
        $item.decodeDeltaPct)
}

Write-Host ':: decode stage deltas ::' -ForegroundColor Cyan
foreach ($stage in $largestStageChanges) {
    Write-Host ("   {0,-14} {1,8:N3} -> {2,8:N3} ms ({3,8}%)" -f `
        $stage.stage,
        $stage.baselineAvgMs,
        $stage.currentAvgMs,
        $stage.deltaPct)
}

Write-Host (":: comparison written to {0} ::" -f $OutputPath) -ForegroundColor Green
