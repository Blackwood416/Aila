param(
    [string]$BuildDir = 'build',
    [string]$PresetsFile = 'perf\presets.json',
    [string]$Preset = 'phase_gate_q35_text',
    [string]$ModelDir = '',
    [string]$ModelLabel = '',
    [string]$OutputDir = '',
    [string]$Phase = 'smoke',
    [string[]]$CaseNames = @(),
    [int]$MaxSeqLen = 0,
    [hashtable]$EnvOverrides = @{},
    [string]$OneApiStack = '',
    [string]$OneApiStacksFile = 'perf\oneapi-stacks.json'
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

function Test-AilaSmokeExpectation {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
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
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$BuildDirPath,

        [Parameter(Mandatory = $true)]
        [string]$OutputDir,

        [Parameter(Mandatory = $true)]
        [string]$ResolvedModelPath,

        [Parameter(Mandatory = $true)]
        [string]$ResolvedModelLabel,

        [int]$MaxSeqLen = 0,

        [hashtable]$EnvOverrides = @{}
    )

    $messagesJsonPath = Resolve-AilaPath -RepoRoot $RepoRoot -Path ([string]$Case.messagesJson)
    $logPath = Join-Path (Join-Path $OutputDir 'smoke_logs') ("{0}.log" -f [string]$Case.name)

    $mode = [string](Get-AilaObjectPropertyValue -Object $Case -Name 'mode' -Default 'greedy')
    $temperature = [double](Get-AilaObjectPropertyValue -Object $Case -Name 'temperature' -Default 0.7)
    $topK = [int](Get-AilaObjectPropertyValue -Object $Case -Name 'topK' -Default 15)
    $topP = [double](Get-AilaObjectPropertyValue -Object $Case -Name 'topP' -Default 0.95)
    $seed = Get-AilaObjectPropertyValue -Object $Case -Name 'seed' -Default $null
    $maxTokens = [int](Get-AilaObjectPropertyValue -Object $Case -Name 'maxTokens' -Default 128)

    $args = @(
        '-m', $ResolvedModelPath,
        '--messages-json', $messagesJsonPath,
        '--max-tokens', ([string]$maxTokens),
        '-t', ([string]$temperature),
        '-k', ([string]$topK),
        '-p', ([string]$topP),
        '--no-stream'
    )
    if ($MaxSeqLen -gt 0) {
        $args += @('--max-seq', ([string]$MaxSeqLen))
    }
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
    $responseText = [string](Get-AilaResponseText -OutputLines $run.outputLines)
    $expectation = Test-AilaSmokeExpectation -ResponseText $responseText -Case $Case
    $success = ($run.exitCode -eq 0) -and $expectation.passed

    return [ordered]@{
        name               = [string]$Case.name
        modelLabel         = $ResolvedModelLabel
        modelPath          = $ResolvedModelPath
        messagesJsonPath   = $messagesJsonPath
        mode               = $mode
        maxTokens          = $maxTokens
        maxSeqLen          = $MaxSeqLen
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

$repoRoot = Get-AilaRepoRoot
$gitInfo = Get-AilaGitInfo -RepoRoot $repoRoot
$buildDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BuildDir
$buildInfoPath = Join-Path $buildDirPath 'build_info.json'
$buildInfo = if (Test-Path -LiteralPath $buildInfoPath) { Read-AilaJsonFile -Path $buildInfoPath } else { $null }

if (-not [string]::IsNullOrWhiteSpace($OneApiStack)) {
    $schemaProperty = if ($null -eq $buildInfo) { $null } else { $buildInfo.PSObject.Properties['schemaVersion'] }
    if ($null -eq $schemaProperty) {
        throw "Unsupported build info schema in '$buildInfoPath': <missing>"
    }
    if ($schemaProperty.Value -ne 2) {
        throw "Unsupported build info schema in '$buildInfoPath': $($schemaProperty.Value)"
    }

    $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path $OneApiStacksFile
    $stack = Get-AilaOneApiStack -Config $config -Name $OneApiStack
    $oneApiProperty = $buildInfo.PSObject.Properties['oneApi']
    if ($null -eq $oneApiProperty -or $null -eq $oneApiProperty.Value) {
        throw "Build info '$buildInfoPath' is missing oneApi metadata."
    }
    $nameProperty = $oneApiProperty.Value.PSObject.Properties['name']
    if ($null -eq $nameProperty -or [string]::IsNullOrWhiteSpace([string]$nameProperty.Value)) {
        throw "Build info '$buildInfoPath' is missing oneApi name metadata."
    }
    if ([string]$nameProperty.Value -ne $OneApiStack) {
        throw "Build '$BuildDir' belongs to '$($nameProperty.Value)', not '$OneApiStack'."
    }

    Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack
    Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack -RequireValues
    Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)
}
else {
    Initialize-AilaOneApiEnvironment
}

if ($null -eq $buildInfo) {
    $buildInfo = [pscustomobject]@{
        build  = $null
        oneApi = $null
    }
}
else {
    if ($null -eq $buildInfo.PSObject.Properties['build']) {
        $buildInfo | Add-Member -NotePropertyName build -NotePropertyValue $null
    }
    if ($null -eq $buildInfo.PSObject.Properties['oneApi']) {
        $buildInfo | Add-Member -NotePropertyName oneApi -NotePropertyValue $null
    }
}

$config = Get-AilaPerfConfig -RepoRoot $repoRoot -PresetsFile $PresetsFile
$presetConfig = Get-AilaPreset -Config $config -PresetName $Preset
$smokeCases = @($presetConfig.smokes)

if ($CaseNames.Count -gt 0) {
    $smokeCases = @($smokeCases | Where-Object { $CaseNames -contains [string]$_.name })
}

if ($smokeCases.Count -eq 0) {
    throw "No smoke cases selected."
}

$resolvedModelPath = ''
$resolvedModelLabel = ''
if (-not [string]::IsNullOrWhiteSpace($ModelDir)) {
    $resolvedModelPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $ModelDir
    $resolvedModelLabel = if ([string]::IsNullOrWhiteSpace($ModelLabel)) { [System.IO.Path]::GetFileName($resolvedModelPath.TrimEnd('\')) } else { $ModelLabel }
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = New-AilaOutputDir -RepoRoot $repoRoot -OutputRoot 'tmp\perf' -Phase $Phase -ShortCommit $gitInfo.shortCommit
}
else {
    $OutputDir = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputDir
}

Ensure-AilaDirectory -Path $OutputDir
Ensure-AilaDirectory -Path (Join-Path $OutputDir 'smoke_logs')

$results = @()
foreach ($smokeCase in $smokeCases) {
    $caseModelPath = $resolvedModelPath
    $caseModelLabel = $resolvedModelLabel
    if ([string]::IsNullOrWhiteSpace($caseModelPath)) {
        $modelAlias = [string](Get-AilaObjectPropertyValue -Object $smokeCase -Name 'modelAlias' -Default '')
        if ([string]::IsNullOrWhiteSpace($modelAlias)) {
            throw "Smoke case '$($smokeCase.name)' is missing modelAlias and no -ModelDir override was provided."
        }
        $modelInfo = Get-AilaModelInfo -Config $config -Alias $modelAlias -RepoRoot $repoRoot
        $caseModelPath = $modelInfo.path
        $caseModelLabel = $modelInfo.alias
    }

    $result = Invoke-AilaSmokeCase -Case $smokeCase -RepoRoot $repoRoot -BuildDirPath $buildDirPath `
        -OutputDir $OutputDir -ResolvedModelPath $caseModelPath -ResolvedModelLabel $caseModelLabel `
        -MaxSeqLen $MaxSeqLen -EnvOverrides $EnvOverrides
    $results += $result

    $status = if ($result.success) { 'PASS' } else { 'FAIL' }
    $preview = [string]$result.responseText
    if ($preview.Length -gt 80) {
        $preview = $preview.Substring(0, 80) + '...'
    }
    Write-Host (":: smoke {0}: {1} -> {2}" -f $status, $result.name, $preview)
}

$smokesPath = Join-Path $OutputDir 'smokes.json'
$failed = @($results | Where-Object { -not $_.success })

Write-AilaJsonFile -Path $smokesPath -Data ([ordered]@{
    schemaVersion = 1
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    phase         = $Phase
    preset        = $Preset
    git           = [ordered]@{
        shortCommit = $gitInfo.shortCommit
        fullCommit  = $gitInfo.fullCommit
        branch      = $gitInfo.branch
    }
    build         = $buildInfo.build
    oneApi        = $buildInfo.oneApi
    model         = [ordered]@{
        label = if ([string]::IsNullOrWhiteSpace($resolvedModelLabel)) { $null } else { $resolvedModelLabel }
        path  = if ([string]::IsNullOrWhiteSpace($resolvedModelPath)) { $null } else { $resolvedModelPath }
    }
    cases         = $results
    summary       = [ordered]@{
        total  = $results.Count
        passed = ($results.Count - $failed.Count)
        failed = @($failed | ForEach-Object { $_.name })
    }
})

Write-Host (":: smoke results written to {0} ::" -f $smokesPath) -ForegroundColor Green

if ($failed.Count -gt 0) {
    $failedNames = ($failed | ForEach-Object { $_.name }) -join ', '
    throw "Smoke failures: $failedNames"
}
