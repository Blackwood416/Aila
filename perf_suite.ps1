param(
    [string]$BuildDir = '',
    [string]$PresetsFile = 'perf\presets.json',
    [string]$Preset = 'phase_gate_q35_text',
    [string]$ModelAlias = '',
    [string]$OutputDir = '',
    [string]$Phase = '',
    [switch]$CleanBuild,
    [switch]$SkipBuild,
    [switch]$RunAutotune,
    [switch]$EnableDnnlVerbose,
    [hashtable]$EnvOverrides = @{}
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

function Get-AilaObjectPropertyValue {
    param(
        [Parameter(Mandatory = $true)]
        $Object,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        $Default = $null
    )

    if ($null -eq $Object) {
        return $Default
    }

    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) {
        return $Default
    }

    return $prop.Value
}

function Merge-AilaEnvOverrides {
    param(
        [hashtable]$Base = @{},
        [hashtable]$Extra = @{}
    )

    $merged = @{}
    if ($null -ne $Base) {
        foreach ($key in $Base.Keys) {
            $merged[$key] = [string]$Base[$key]
        }
    }
    if ($null -ne $Extra) {
        foreach ($key in $Extra.Keys) {
            $merged[$key] = [string]$Extra[$key]
        }
    }

    return $merged
}

function Test-AilaSmokeExpectation {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ResponseText,

        [Parameter(Mandatory = $true)]
        $Case
    )

    $passed = $true
    $details = New-Object System.Collections.Generic.List[object]
    $matchedValue = $null

    $expectNonEmpty = [bool](Get-AilaObjectPropertyValue -Object $Case -Name 'expectNonEmpty' -Default $false)
    if ($expectNonEmpty) {
        $ok = -not [string]::IsNullOrWhiteSpace($ResponseText)
        $details.Add([pscustomobject]@{
            type   = 'non_empty'
            passed = $ok
        })
        if (-not $ok) {
            $passed = $false
        }
    }

    $expectContainsAny = @(Get-AilaObjectPropertyValue -Object $Case -Name 'expectContainsAny' -Default @())
    if ($expectContainsAny.Count -gt 0) {
        foreach ($candidate in $expectContainsAny) {
            if ($ResponseText.IndexOf([string]$candidate, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $matchedValue = [string]$candidate
                break
            }
        }
        $ok = -not [string]::IsNullOrWhiteSpace($matchedValue)
        $details.Add([pscustomobject]@{
            type       = 'contains_any'
            passed     = $ok
            matched    = $matchedValue
            candidates = $expectContainsAny
        })
        if (-not $ok) {
            $passed = $false
        }
    }

    $expectRegex = [string](Get-AilaObjectPropertyValue -Object $Case -Name 'expectRegex' -Default '')
    if (-not [string]::IsNullOrWhiteSpace($expectRegex)) {
        $ok = $ResponseText -match $expectRegex
        $details.Add([pscustomobject]@{
            type    = 'regex'
            passed  = $ok
            pattern = $expectRegex
        })
        if (-not $ok) {
            $passed = $false
        }
    }

    return [pscustomobject]@{
        passed  = $passed
        matched = $matchedValue
        details = $details.ToArray()
    }
}

function Invoke-AilaSmokeCase {
    param(
        [Parameter(Mandatory = $true)]
        $Case,

        [Parameter(Mandatory = $true)]
        $Config,

        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$BuildDirPath,

        [Parameter(Mandatory = $true)]
        [string]$OutputDir,

        [hashtable]$EnvOverrides = @{}
    )

    $modelAlias = [string](Get-AilaObjectPropertyValue -Object $Case -Name 'modelAlias' -Default '')
    if ([string]::IsNullOrWhiteSpace($modelAlias)) {
        throw "Smoke case '$($Case.name)' is missing modelAlias."
    }

    $modelInfo = Get-AilaModelInfo -Config $Config -Alias $modelAlias -RepoRoot $RepoRoot
    $messagesJsonPath = Resolve-AilaPath -RepoRoot $RepoRoot -Path ([string]$Case.messagesJson)
    $logPath = Join-Path (Join-Path $OutputDir 'smoke_logs') ("{0}.log" -f [string]$Case.name)

    $mode = [string](Get-AilaObjectPropertyValue -Object $Case -Name 'mode' -Default 'greedy')
    $temperature = [double](Get-AilaObjectPropertyValue -Object $Case -Name 'temperature' -Default 0.7)
    $topK = [int](Get-AilaObjectPropertyValue -Object $Case -Name 'topK' -Default 15)
    $topP = [double](Get-AilaObjectPropertyValue -Object $Case -Name 'topP' -Default 0.95)
    $seed = Get-AilaObjectPropertyValue -Object $Case -Name 'seed' -Default $null
    $maxTokens = [int](Get-AilaObjectPropertyValue -Object $Case -Name 'maxTokens' -Default 16)

    $args = @(
        '-m', $modelInfo.path,
        '--messages-json', $messagesJsonPath,
        '--max-tokens', ([string]$maxTokens),
        '-t', ([string]$temperature),
        '-k', ([string]$topK),
        '-p', ([string]$topP),
        '--no-stream'
    )
    if ($mode -eq 'sample') {
        $args += '--sample'
    }
    else {
        $args += '--greedy'
    }
    if ($null -ne $seed -and -not [string]::IsNullOrWhiteSpace([string]$seed)) {
        $args += '--seed'
        $args += ([string][UInt64]$seed)
    }

    $run = Invoke-AilaProcess -Executable '.\Aila.exe' -ArgumentList $args -WorkingDirectory $BuildDirPath -LogPath $logPath -EnvOverrides $EnvOverrides
    $responseText = Get-AilaResponseText -OutputLines $run.outputLines
    $expectation = Test-AilaSmokeExpectation -ResponseText $responseText -Case $Case
    $success = ($run.exitCode -eq 0) -and $expectation.passed

    return [ordered]@{
        name               = [string]$Case.name
        modelAlias         = $modelInfo.alias
        modelPath          = $modelInfo.path
        messagesJsonPath   = $messagesJsonPath
        mode               = $mode
        maxTokens          = $maxTokens
        commandLine        = $run.commandLine
        envOverrides       = $EnvOverrides
        exitCode           = $run.exitCode
        durationMs         = $run.durationMs
        success            = $success
        responseText       = $responseText
        expectationMatched = $expectation.matched
        expectationDetails = $expectation.details
        rawLogPath         = $logPath
    }
}

function Get-AilaEnvMatrixCombinations {
    param(
        [Parameter(Mandatory = $true)]
        $EnvMatrix
    )

    $entries = @()
    foreach ($prop in $EnvMatrix.PSObject.Properties) {
        $values = @($prop.Value)
        if ($values.Count -eq 0) {
            continue
        }
        $entries += [pscustomobject]@{
            Name   = $prop.Name
            Values = $values
        }
    }

    if ($entries.Count -eq 0) {
        return @(@{})
    }

    $results = New-Object System.Collections.Generic.List[hashtable]

    function Expand-AilaEnvCombination {
        param(
            [int]$Index,
            [hashtable]$Current
        )

        if ($Index -ge $entries.Count) {
            $copy = @{}
            foreach ($key in $Current.Keys) {
                $copy[$key] = [string]$Current[$key]
            }
            $results.Add($copy)
            return
        }

        $entry = $entries[$Index]
        foreach ($value in $entry.Values) {
            $Current[$entry.Name] = [string]$value
            Expand-AilaEnvCombination -Index ($Index + 1) -Current $Current
        }
        [void]$Current.Remove($entry.Name)
    }

    Expand-AilaEnvCombination -Index 0 -Current @{}
    return $results.ToArray()
}

function Get-AilaAutotuneObjectiveValue {
    param(
        [Parameter(Mandatory = $true)]
        $CaseResult,

        [Parameter(Mandatory = $true)]
        [string]$Objective
    )

    switch ($Objective) {
        'decodeTokPerSec' { return [double]$CaseResult.decode.tokPerSec }
        'prefillTokPerSec' { return [double]$CaseResult.prefill.tokPerSec }
        default { throw "Unsupported autotune objective '$Objective'." }
    }
}

$repoRoot = Get-AilaRepoRoot
$gitInfo = Get-AilaGitInfo -RepoRoot $repoRoot
$config = Get-AilaPerfConfig -RepoRoot $repoRoot -PresetsFile $PresetsFile
$presetConfig = Get-AilaPreset -Config $config -PresetName $Preset

$phaseName = if ([string]::IsNullOrWhiteSpace($Phase)) { $Preset } else { $Phase }
$buildSettings = $presetConfig.build
$buildDirArg = if ([string]::IsNullOrWhiteSpace($BuildDir)) { [string]$buildSettings.buildDir } else { $BuildDir }
$buildConfig = [string](Get-AilaObjectPropertyValue -Object $buildSettings -Name 'config' -Default 'Release')
$buildJobs = [int](Get-AilaObjectPropertyValue -Object $buildSettings -Name 'jobs' -Default 36)
$doCleanBuild = $CleanBuild.IsPresent -or [bool](Get-AilaObjectPropertyValue -Object $buildSettings -Name 'clean' -Default $false)
$anchorModelAlias = if ([string]::IsNullOrWhiteSpace($ModelAlias)) { [string]$presetConfig.anchorModel } else { $ModelAlias }
$buildDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $buildDirArg

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = New-AilaOutputDir -RepoRoot $repoRoot -OutputRoot 'tmp\perf' -Phase $phaseName -ShortCommit $gitInfo.shortCommit
}
else {
    $OutputDir = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputDir
}

Ensure-AilaDirectory -Path $OutputDir
Ensure-AilaDirectory -Path (Join-Path $OutputDir 'smoke_logs')

if (-not $SkipBuild.IsPresent) {
    & (Join-Path $repoRoot 'build.ps1') -BuildDir $buildDirArg -Config $buildConfig -Clean:$doCleanBuild -Jobs $buildJobs
}

$buildInfoPath = Join-Path $buildDirPath 'build_info.json'
$buildInfo = if (Test-Path -LiteralPath $buildInfoPath) { Read-AilaJsonFile -Path $buildInfoPath } else { $null }

& (Join-Path $repoRoot 'bench.ps1') `
    -BuildDir $buildDirArg `
    -PresetsFile $PresetsFile `
    -Preset $Preset `
    -ModelAlias $anchorModelAlias `
    -OutputDir $OutputDir `
    -Phase $phaseName `
    -EnvOverrides $EnvOverrides

$benchPath = Join-Path $OutputDir 'bench.json'
$benchData = Read-AilaJsonFile -Path $benchPath

& (Join-Path $repoRoot 'profile_decode.ps1') `
    -BuildDir $buildDirArg `
    -PresetsFile $PresetsFile `
    -Preset $Preset `
    -ModelAlias $anchorModelAlias `
    -OutputDir $OutputDir `
    -Phase $phaseName `
    -EnableDnnlVerbose:$EnableDnnlVerbose `
    -EnvOverrides $EnvOverrides

$profilePath = Join-Path $OutputDir 'decode_profile_summary.json'
$profileData = Read-AilaJsonFile -Path $profilePath

Initialize-AilaOneApiEnvironment

$smokeResults = @()
foreach ($smokeCase in $presetConfig.smokes) {
    $result = Invoke-AilaSmokeCase -Case $smokeCase -Config $config -RepoRoot $repoRoot `
        -BuildDirPath $buildDirPath -OutputDir $OutputDir -EnvOverrides $EnvOverrides
    $smokeResults += $result

    $status = if ($result.success) { 'PASS' } else { 'FAIL' }
    $preview = $result.responseText
    if ($preview.Length -gt 80) {
        $preview = $preview.Substring(0, 80) + '...'
    }
    Write-Host (":: smoke {0}: {1} -> {2}" -f $status, $result.name, $preview)
}

$smokesPath = Join-Path $OutputDir 'smokes.json'
Write-AilaJsonFile -Path $smokesPath -Data ([ordered]@{
    schemaVersion = 1
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    phase         = $phaseName
    preset        = $Preset
    git           = [ordered]@{
        shortCommit = $gitInfo.shortCommit
        fullCommit  = $gitInfo.fullCommit
        branch      = $gitInfo.branch
    }
    build         = if ($null -ne $buildInfo) { $buildInfo.build } else { $null }
    cases         = $smokeResults
})

$autotunePath = $null
$autotuneData = $null
$autotuneConfig = Get-AilaObjectPropertyValue -Object $presetConfig -Name 'autotune' -Default $null
if ($RunAutotune.IsPresent) {
    if ($null -eq $autotuneConfig -or -not [bool](Get-AilaObjectPropertyValue -Object $autotuneConfig -Name 'enabled' -Default $false)) {
        throw "Preset '$Preset' does not have autotune enabled."
    }

    $autotuneCaseName = [string](Get-AilaObjectPropertyValue -Object $autotuneConfig -Name 'benchmarkCase' -Default '')
    $objective = [string](Get-AilaObjectPropertyValue -Object $autotuneConfig -Name 'objective' -Default 'decodeTokPerSec')
    $combos = Get-AilaEnvMatrixCombinations -EnvMatrix $autotuneConfig.envMatrix
    $autotuneResults = @()
    $autotuneRoot = Join-Path $OutputDir 'autotune_runs'
    Ensure-AilaDirectory -Path $autotuneRoot

    $runIndex = 0
    foreach ($combo in $combos) {
        $runIndex += 1
        $mergedEnv = Merge-AilaEnvOverrides -Base $EnvOverrides -Extra $combo
        $runDir = Join-Path $autotuneRoot ("run_{0:D2}" -f $runIndex)

        & (Join-Path $repoRoot 'bench.ps1') `
            -BuildDir $buildDirArg `
            -PresetsFile $PresetsFile `
            -Preset $Preset `
            -ModelAlias $anchorModelAlias `
            -OutputDir $runDir `
            -Phase $phaseName `
            -CaseNames @($autotuneCaseName) `
            -EnvOverrides $mergedEnv

        $runBench = Read-AilaJsonFile -Path (Join-Path $runDir 'bench.json')
        $caseResult = @($runBench.cases | Select-Object -First 1)[0]
        $score = Get-AilaAutotuneObjectiveValue -CaseResult $caseResult -Objective $objective

        $autotuneResults += [pscustomobject][ordered]@{
            rank        = 0
            runIndex    = $runIndex
            caseName    = $autotuneCaseName
            objective   = $objective
            score       = $score
            envOverrides = $mergedEnv
            prefillTokPerSec = [double]$caseResult.prefill.tokPerSec
            decodeTokPerSec  = [double]$caseResult.decode.tokPerSec
            benchPath    = (Join-Path $runDir 'bench.json')
        }
    }

    $sorted = @($autotuneResults | Sort-Object -Property @{ Expression = { [double]$_.score } } -Descending)
    for ($i = 0; $i -lt $sorted.Count; ++$i) {
        $sorted[$i].rank = $i + 1
    }

    $best = if ($sorted.Count -gt 0) { $sorted[0] } else { $null }
    $autotunePath = Join-Path $OutputDir 'autotune.json'
    $autotuneData = [ordered]@{
        schemaVersion = 1
        generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
        phase         = $phaseName
        preset        = $Preset
        objective     = $objective
        benchmarkCase = $autotuneCaseName
        resultCount   = $sorted.Count
        recommended   = $best
        results       = $sorted
    }
    Write-AilaJsonFile -Path $autotunePath -Data $autotuneData
}

$failedSmokes = @($smokeResults | Where-Object { -not $_.success })
$summaryPath = Join-Path $OutputDir 'summary.json'
$summaryData = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    phase         = $phaseName
    preset        = $Preset
    git           = [ordered]@{
        shortCommit = $gitInfo.shortCommit
        fullCommit  = $gitInfo.fullCommit
        branch      = $gitInfo.branch
    }
    build         = [ordered]@{
        buildDir   = $buildDirPath
        buildType  = if ($null -ne $buildInfo) { $buildInfo.build.buildType } else { $buildConfig }
        compiler   = if ($null -ne $buildInfo) { $buildInfo.build.compiler } else { $null }
        generator  = if ($null -ne $buildInfo) { $buildInfo.build.generator } else { $null }
    }
    artifacts     = [ordered]@{
        benchPath          = $benchPath
        decodeProfilePath  = $profilePath
        smokesPath         = $smokesPath
        autotunePath       = $autotunePath
    }
    bench         = @($benchData.cases | ForEach-Object {
        [ordered]@{
            name           = $_.name
            mode           = $_.mode
            prefillTokPerSec = [double]$_.prefill.tokPerSec
            decodeTokPerSec  = [double]$_.decode.tokPerSec
        }
    })
    decodeProfile = [ordered]@{
        sampleCount = [int]$profileData.sampleCount
        hotspotsTop = @($profileData.hotspots | Select-Object -First 5)
    }
    smokes        = [ordered]@{
        total      = $smokeResults.Count
        passed     = ($smokeResults.Count - $failedSmokes.Count)
        failed     = @($failedSmokes | ForEach-Object { $_.name })
    }
    autotune      = if ($null -ne $autotuneData) {
        [ordered]@{
            enabled     = $true
            recommended = $autotuneData.recommended
        }
    } else {
        [ordered]@{
            enabled = $false
        }
    }
}

Write-AilaJsonFile -Path $summaryPath -Data $summaryData
Write-Host (":: perf suite summary written to {0} ::" -f $summaryPath) -ForegroundColor Green

if ($failedSmokes.Count -gt 0) {
    $failedNames = ($failedSmokes | ForEach-Object { $_.name }) -join ', '
    throw "Smoke regressions failed: $failedNames"
}
