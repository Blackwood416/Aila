[CmdletBinding()]
param(
    [string]$BaselineBuildDir = 'build-oneapi-2025.3',
    [string]$CandidateBuildDir = 'build-oneapi-2026.1',
    [string]$BaselineStack = 'oneapi-2025.3',
    [string]$CandidateStack = 'oneapi-2026.1',
    [string[]]$AccuracyCases = @(
        'q35_08_dense_text',
        'q35_vision_ocr',
        'qwen3_asr_zh_nf4',
        'qwen3_aligner_zh_nf4',
        'qwen3_tts_06_base'
    ),
    [string]$PerformancePreset = 'phase_gate_q35_text',
    [string[]]$PerformanceCases = @('greedy_short'),
    [string]$ReportRoot = 'tmp\perf\oneapi-compare',
    [switch]$SkipBuild,
    [switch]$SkipAccuracy,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

$script:RepoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$script:ReportSchemaVersion = 1

function Invoke-CheckedPwsh {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & pwsh -NoProfile @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Child PowerShell failed: pwsh $($Arguments -join ' ')"
    }
}

function Test-AilaGate {
    param(
        [Parameter(Mandatory = $true)]$Comparison,
        [Parameter(Mandatory = $true)][double]$MinimumDeltaPercent
    )
    return [double]$Comparison.deltaPercent -ge $MinimumDeltaPercent
}

function Test-AilaExpansionRequired {
    param([Parameter(Mandatory = $true)]$Comparison)
    $delta = [double]$Comparison.deltaPercent
    return $delta -ge -5.0 -and $delta -le -3.0
}

function Get-AilaInitClassification {
    param(
        [AllowNull()]$BaselineMs,
        [AllowNull()]$CandidateMs
    )

    if ($null -eq $BaselineMs -or $null -eq $CandidateMs -or [double]$BaselineMs -le 0.0) {
        return [pscustomobject]@{
            status = 'not-isolated'
            deltaPercent = $null
            absoluteDeltaMs = $null
            blocking = $false
        }
    }

    $baselineValue = [double]$BaselineMs
    $candidateValue = [double]$CandidateMs
    $deltaPercent = (($candidateValue - $baselineValue) / $baselineValue) * 100.0
    $absoluteDeltaMs = $candidateValue - $baselineValue
    $status = if ($deltaPercent -le 20.0) {
        'observe'
    }
    elseif ($deltaPercent -le 50.0) {
        'repeat-and-profile'
    }
    elseif ($absoluteDeltaMs -gt 1000.0) {
        'investigate'
    }
    else {
        'observe'
    }
    if ($candidateValue -ge (2.0 * $baselineValue)) {
        $status = 'blocking'
    }

    return [pscustomobject]@{
        status = $status
        deltaPercent = [math]::Round($deltaPercent, 4)
        absoluteDeltaMs = [math]::Round($absoluteDeltaMs, 4)
        blocking = $status -eq 'blocking'
    }
}

function Get-AilaTask10Status {
    param(
        [Parameter(Mandatory = $true)][bool]$AutomaticPassed,
        [Parameter(Mandatory = $true)][bool]$ManualReviewRequired,
        [AllowNull()]$PerformancePassed
    )

    if (-not $AutomaticPassed -or $null -eq $PerformancePassed -or -not [bool]$PerformancePassed) {
        return 'failed'
    }
    if ($ManualReviewRequired) {
        return 'manual-review-required'
    }
    return 'pass'
}

function Get-AilaOwnedReportRoot {
    param([Parameter(Mandatory = $true)][string]$Requested)

    $resolved = Resolve-AilaPath -RepoRoot $script:RepoRoot -Path $Requested
    $tmpRoot = Join-Path $script:RepoRoot 'tmp'
    if (-not (Test-AilaPathWithinRoot -Path $resolved -Root $tmpRoot)) {
        throw "ReportRoot must be inside repository tmp: $resolved"
    }
    return $resolved
}

function Write-AilaAtomicText {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text
    )

    Ensure-AilaDirectory -Path (Split-Path -Parent $Path)
    $temporaryPath = "$Path.tmp-$([guid]::NewGuid().ToString('N'))"
    try {
        [System.IO.File]::WriteAllText($temporaryPath, $Text, [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
    }
    finally {
        Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
}

function Reset-AilaOwnedReports {
    param([Parameter(Mandatory = $true)][string]$Root)
    Remove-Item -LiteralPath (Join-Path $Root 'report.json'), (Join-Path $Root 'report.md') -Force -ErrorAction SilentlyContinue
}

function Get-AilaBenchmarkRawSamples {
    param(
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][string]$Stack,
        [Parameter(Mandatory = $true)][string]$OrderTag
    )

    $text = Get-Content -LiteralPath $LogPath -Raw -Encoding UTF8
    $pattern = '(?m)^\s*(?<metric>pp|tg) iter (?<iteration>\d+):\s*(?<elapsed>[\d.]+) ms \((?<tokps>[\d.]+) tok/s\)\s*$'
    $samples = @(
        foreach ($match in [regex]::Matches($text, $pattern)) {
            [pscustomobject]@{
                stack = $Stack
                orderTag = $OrderTag
                metric = $match.Groups['metric'].Value
                iteration = [int]$match.Groups['iteration'].Value
                tokPerSec = [double]$match.Groups['tokps'].Value
                elapsedMs = [double]$match.Groups['elapsed'].Value
                logPath = $LogPath
            }
        }
    )
    if (@($samples | Where-Object metric -eq 'pp').Count -eq 0 -or @($samples | Where-Object metric -eq 'tg').Count -eq 0) {
        throw "No raw pp/tg iteration samples found in benchmark log: $LogPath"
    }
    return $samples
}

function Get-AilaWarmupSamples {
    param([Parameter(Mandatory = $true)][string[]]$LogPaths)

    return @(
        foreach ($logPath in $LogPaths) {
            $text = Get-Content -LiteralPath $logPath -Raw -Encoding UTF8
            $match = [regex]::Match($text, '(?m)^\[Warmup\] Completed in (?<ms>[\d.]+) ms\s*$')
            if ($match.Success) {
                [double]$match.Groups['ms'].Value
            }
        }
    )
}

function Get-AilaTtsObservation {
    param(
        [Parameter(Mandatory = $true)][string]$AccuracyDir,
        [Parameter(Mandatory = $true)][string]$Stack
    )

    $resultPath = Join-Path $AccuracyDir 'case_results\qwen3_tts_06_base.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        return [pscustomobject]@{ stack = $Stack; samples = @(); sampleCount = 0; source = 'unavailable'; resultPath = $resultPath }
    }
    $result = Read-AilaJsonFile -Path $resultPath
    $stderrPath = [string]$result.logs.stderr.path
    $rtf = $null
    $source = 'unavailable'
    if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
        $stderr = Get-Content -LiteralPath $stderrPath -Raw -Encoding UTF8
        $match = [regex]::Match($stderr, '\[TTS\] Synthesis complete:.*?RTF=(?<rtf>[\d.]+)')
        if ($match.Success) {
            $rtf = [double]$match.Groups['rtf'].Value
            $source = 'reported-tts-rtf'
        }
    }
    if ($null -eq $rtf -and $null -ne $result.wav -and [double]$result.wav.durationSeconds -gt 0.0) {
        $rtf = ([double]$result.process.durationMs / 1000.0) / [double]$result.wav.durationSeconds
        $source = 'process-elapsed-over-wav-duration-approximation'
    }
    return [pscustomobject]@{
        stack = $Stack
        samples = if ($null -eq $rtf) { ,@() } else { ,@([math]::Round($rtf, 6)) }
        sampleCount = if ($null -eq $rtf) { 0 } else { 1 }
        source = $source
        resultPath = $resultPath
        processDurationMs = [double]$result.process.durationMs
        audioDurationSeconds = [double]$result.wav.durationSeconds
    }
}

function ConvertTo-AilaTask10Markdown {
    param([Parameter(Mandatory = $true)]$Report)

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# oneAPI 2026 migration comparison')
    $lines.Add('')
    $lines.Add("- Status: **$($Report.status)**")
    $lines.Add("- Generated: $($Report.completedAtUtc)")
    $lines.Add("- Git: $($Report.git.fullCommit) ($($Report.git.branch))")
    $lines.Add("- Accuracy: automaticPassed=$($Report.accuracy.automaticPassed), manualReviewRequired=$($Report.accuracy.manualReviewRequired)")
    $lines.Add("- Accuracy report: $($Report.accuracy.comparisonPath)")
    $lines.Add('')
    $lines.Add('## Performance')
    $lines.Add('')
    if ($Report.performance.skipped) {
        $lines.Add("Performance skipped: $($Report.performance.reason)")
    }
    else {
        foreach ($metric in @('pp', 'tg')) {
            $entry = $Report.performance.metrics.$metric
            $lines.Add("- $metric tok/s: baseline median $($entry.baselineMedian), candidate median $($entry.candidateMedian), delta $($entry.deltaPercent)%, gate $($entry.gatePassed)")
        }
        $lines.Add('')
        $lines.Add('Raw ABBA samples:')
        foreach ($sample in $Report.performance.rawSamples) {
            $lines.Add("- $($sample.orderTag) $($sample.stack) $($sample.metric) iter $($sample.iteration): $($sample.tokPerSec) tok/s")
        }
    }
    $lines.Add('')
    $lines.Add('## TTS RTF')
    $lines.Add('')
    $lines.Add("- Baseline samples ($($Report.ttsRtf.baseline.sampleCount)): $([string]::Join(', ', @($Report.ttsRtf.baseline.samples))) ($($Report.ttsRtf.baseline.source))")
    $lines.Add("- Candidate samples ($($Report.ttsRtf.candidate.sampleCount)): $([string]::Join(', ', @($Report.ttsRtf.candidate.samples))) ($($Report.ttsRtf.candidate.source))")
    if ($null -eq $Report.ttsRtf.comparison) {
        $lines.Add('- Delta: unavailable')
    }
    else {
        $lines.Add("- Delta: $($Report.ttsRtf.comparison.deltaPercent)% ; gate: $($Report.ttsRtf.gatePassed)")
    }
    $lines.Add('')
    $lines.Add('## Initialization')
    $lines.Add('')
    $lines.Add("- Metric: $($Report.initialization.metric) ($($Report.initialization.reliability))")
    $lines.Add("- Classification: $($Report.initialization.classification.status)")
    if ($Report.manualLinks.Count -gt 0) {
        $lines.Add('')
        $lines.Add('## Manual review')
        $lines.Add('')
        foreach ($link in $Report.manualLinks) { $lines.Add("- $link") }
    }
    if ($Report.unresolvedIssues.Count -gt 0) {
        $lines.Add('')
        $lines.Add('## Unresolved issues')
        $lines.Add('')
        foreach ($issue in $Report.unresolvedIssues) { $lines.Add("- $issue") }
    }
    return [string]::Join([Environment]::NewLine, $lines) + [Environment]::NewLine
}

function Write-AilaTask10Report {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)]$Report
    )

    $jsonPath = Join-Path $Root 'report.json'
    $markdownPath = Join-Path $Root 'report.md'
    Write-AilaAtomicText -Path $jsonPath -Text ($Report | ConvertTo-Json -Depth 24)
    Write-AilaAtomicText -Path $markdownPath -Text (ConvertTo-AilaTask10Markdown -Report $Report)
    return [pscustomobject]@{ jsonPath = $jsonPath; markdownPath = $markdownPath }
}

function Invoke-AilaBenchmarkRun {
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)][string]$Stack,
        [Parameter(Mandatory = $true)]$BenchmarkCase,
        [Parameter(Mandatory = $true)]$Model,
        [Parameter(Mandatory = $true)][string]$OutputDir,
        [Parameter(Mandatory = $true)][string]$OrderTag,
        [Parameter(Mandatory = $true)][int]$Iterations
    )

    $arguments = @(
        '-File', (Join-Path $script:RepoRoot 'bench.ps1'),
        '-BuildDir', $BuildDir,
        '-ModelAlias', [string]$Model.alias,
        '-ModelDir', [string]$Model.path,
        '-OutputDir', $OutputDir,
        '-Phase', 'oneapi-compare',
        '-PromptTokens', [string]$BenchmarkCase.promptTokens,
        '-GenTokens', [string]$BenchmarkCase.genTokens,
        '-BenchIters', [string]$Iterations,
        '-WarmupIters', '1',
        '-Temperature', [string]$BenchmarkCase.temperature,
        '-TopK', [string]$BenchmarkCase.topK,
        '-TopP', [string]$BenchmarkCase.topP,
        '-Seed', [string]$BenchmarkCase.seed,
        '-OneApiStack', $Stack
    )
    if ([string]$BenchmarkCase.mode -eq 'sample') { $arguments += '-Sample' }
    Invoke-CheckedPwsh -Arguments $arguments
    $logPath = @(Get-ChildItem -LiteralPath (Join-Path $OutputDir 'bench_logs') -Filter '*.log' -File)[0].FullName
    return [pscustomobject]@{
        orderTag = $OrderTag
        stack = $Stack
        outputDir = $OutputDir
        benchJsonPath = Join-Path $OutputDir 'bench.json'
        logPath = $logPath
        samples = @(Get-AilaBenchmarkRawSamples -LogPath $logPath -Stack $Stack -OrderTag $OrderTag)
    }
}

function Invoke-AilaFullComparison {
    $startedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    $root = Get-AilaOwnedReportRoot -Requested $ReportRoot
    Ensure-AilaDirectory -Path $root
    Reset-AilaOwnedReports -Root $root

    if ($PerformanceCases.Count -ne 1) {
        throw 'This representative migration comparison currently requires exactly one performance case.'
    }

    $baselineBuildPath = Resolve-AilaPath -RepoRoot $script:RepoRoot -Path $BaselineBuildDir
    $candidateBuildPath = Resolve-AilaPath -RepoRoot $script:RepoRoot -Path $CandidateBuildDir
    if (-not $SkipBuild) {
        Invoke-CheckedPwsh -Arguments @('-File', (Join-Path $script:RepoRoot 'build.ps1'), '-BuildDir', $BaselineBuildDir, '-OneApiStack', $BaselineStack, '-Clean')
    }
    Invoke-CheckedPwsh -Arguments @('-File', (Join-Path $script:RepoRoot 'verify_oneapi_build.ps1'), '-BuildDir', $BaselineBuildDir, '-OneApiStack', $BaselineStack)
    if (-not $SkipBuild) {
        Invoke-CheckedPwsh -Arguments @('-File', (Join-Path $script:RepoRoot 'build.ps1'), '-BuildDir', $CandidateBuildDir, '-OneApiStack', $CandidateStack, '-Clean')
    }
    Invoke-CheckedPwsh -Arguments @('-File', (Join-Path $script:RepoRoot 'verify_oneapi_build.ps1'), '-BuildDir', $CandidateBuildDir, '-OneApiStack', $CandidateStack)

    foreach ($build in @($baselineBuildPath, $candidateBuildPath)) {
        $escapedBuild = $build.Replace("'", "''")
        Invoke-CheckedPwsh -Arguments @('-Command', "& ctest --test-dir '$escapedBuild' --output-on-failure; if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }")
    }

    $baselineAccuracyDir = Join-Path $root "accuracy\$BaselineStack"
    $candidateAccuracyDir = Join-Path $root "accuracy\$CandidateStack"
    $accuracyComparisonPath = Join-Path $root 'accuracy_comparison.json'
    if (-not $SkipAccuracy) {
        foreach ($run in @(
            [pscustomobject]@{ build = $BaselineBuildDir; stack = $BaselineStack; output = $baselineAccuracyDir },
            [pscustomobject]@{ build = $CandidateBuildDir; stack = $CandidateStack; output = $candidateAccuracyDir }
        )) {
            $caseLiteral = '@(' + (($AccuracyCases | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ',') + ')'
            $scriptLiteral = (Join-Path $script:RepoRoot 'regress.ps1').Replace("'", "''")
            $buildLiteral = $run.build.Replace("'", "''")
            $stackLiteral = $run.stack.Replace("'", "''")
            $outputLiteral = $run.output.Replace("'", "''")
            Invoke-CheckedPwsh -Arguments @('-Command', "& '$scriptLiteral' -BuildDir '$buildLiteral' -OneApiStack '$stackLiteral' -CaseNames $caseLiteral -OutputDir '$outputLiteral'")
        }
        Invoke-CheckedPwsh -Arguments @(
            '-File', (Join-Path $script:RepoRoot 'compare_accuracy.ps1'),
            '-BaselineDir', $baselineAccuracyDir,
            '-CandidateDir', $candidateAccuracyDir,
            '-OutputPath', $accuracyComparisonPath
        )
    }
    if (-not (Test-Path -LiteralPath $accuracyComparisonPath -PathType Leaf)) {
        throw "Accuracy comparison not found: $accuracyComparisonPath"
    }
    $accuracy = Read-AilaJsonFile -Path $accuracyComparisonPath
    $manualLinks = @($accuracy.cases | Where-Object manualReviewRequired | ForEach-Object {
        if ($null -ne $_.candidate.output.path) { [string]$_.candidate.output.path } else { "$($_.name): $($_.reasons -join '; ')" }
    })
    $unresolved = [System.Collections.Generic.List[string]]::new()
    if ([bool]$accuracy.manualReviewRequired) { $unresolved.Add('Accuracy comparison requires manual review.') }
    if (-not [bool]$accuracy.automaticPassed) { $unresolved.Add('Accuracy automatic gate failed; performance was skipped.') }

    $performance = [ordered]@{ skipped = $true; reason = 'accuracy automatic gate failed'; order = @(); rawSamples = @(); metrics = $null }
    $performancePassed = $null
    $init = [pscustomobject]@{
        metric = 'engine-warmup'
        reliability = 'not-isolated'
        baselineSamplesMs = @()
        candidateSamplesMs = @()
        classification = Get-AilaInitClassification -BaselineMs $null -CandidateMs $null
    }

    if ([bool]$accuracy.automaticPassed) {
        $performanceRoot = Join-Path $root 'performance'
        Remove-Item -LiteralPath $performanceRoot -Recurse -Force -ErrorAction SilentlyContinue
        Ensure-AilaDirectory -Path $performanceRoot
        $perfConfig = Get-AilaPerfConfig -RepoRoot $script:RepoRoot -PresetsFile 'perf\presets.json'
        $preset = Get-AilaPreset -Config $perfConfig -PresetName $PerformancePreset
        $model = Get-AilaModelInfo -Config $perfConfig -Alias ([string]$preset.anchorModel) -RepoRoot $script:RepoRoot
        $allRuns = [System.Collections.Generic.List[object]]::new()
        $rawSamples = [System.Collections.Generic.List[object]]::new()
        $sequence = @(
            [pscustomobject]@{ tag = '01-baseline'; stack = $BaselineStack; build = $BaselineBuildDir },
            [pscustomobject]@{ tag = '02-candidate'; stack = $CandidateStack; build = $CandidateBuildDir },
            [pscustomobject]@{ tag = '03-candidate'; stack = $CandidateStack; build = $CandidateBuildDir },
            [pscustomobject]@{ tag = '04-baseline'; stack = $BaselineStack; build = $BaselineBuildDir }
        )
        foreach ($caseName in $PerformanceCases) {
            $benchmarkCase = @($preset.benchmarks | Where-Object name -eq $caseName)[0]
            if ($null -eq $benchmarkCase) { throw "Performance case '$caseName' not found in preset '$PerformancePreset'." }
            foreach ($item in $sequence) {
                $output = Join-Path $performanceRoot (Join-Path $item.tag $item.stack)
                $run = Invoke-AilaBenchmarkRun -BuildDir $item.build -Stack $item.stack -BenchmarkCase $benchmarkCase -Model $model -OutputDir $output -OrderTag "$caseName/$($item.tag)" -Iterations 3
                $allRuns.Add($run)
                foreach ($sample in $run.samples) { $rawSamples.Add($sample) }
            }

            $basePp = @($rawSamples | Where-Object { $_.stack -eq $BaselineStack -and $_.metric -eq 'pp' } | ForEach-Object tokPerSec)
            $candidatePp = @($rawSamples | Where-Object { $_.stack -eq $CandidateStack -and $_.metric -eq 'pp' } | ForEach-Object tokPerSec)
            $baseTg = @($rawSamples | Where-Object { $_.stack -eq $BaselineStack -and $_.metric -eq 'tg' } | ForEach-Object tokPerSec)
            $candidateTg = @($rawSamples | Where-Object { $_.stack -eq $CandidateStack -and $_.metric -eq 'tg' } | ForEach-Object tokPerSec)
            $ppComparison = Get-AilaMetricComparison -Baseline $basePp -Candidate $candidatePp -HigherIsBetter
            $tgComparison = Get-AilaMetricComparison -Baseline $baseTg -Candidate $candidateTg -HigherIsBetter
            if ((Test-AilaExpansionRequired $ppComparison) -or (Test-AilaExpansionRequired $tgComparison)) {
                foreach ($item in @(
                    [pscustomobject]@{ tag = '05-expand-baseline'; stack = $BaselineStack; build = $BaselineBuildDir },
                    [pscustomobject]@{ tag = '06-expand-candidate'; stack = $CandidateStack; build = $CandidateBuildDir }
                )) {
                    $output = Join-Path $performanceRoot (Join-Path $item.tag $item.stack)
                    $run = Invoke-AilaBenchmarkRun -BuildDir $item.build -Stack $item.stack -BenchmarkCase $benchmarkCase -Model $model -OutputDir $output -OrderTag "$caseName/$($item.tag)" -Iterations 1
                    $allRuns.Add($run)
                    foreach ($sample in $run.samples) { $rawSamples.Add($sample) }
                }
                $basePp = @($rawSamples | Where-Object { $_.stack -eq $BaselineStack -and $_.metric -eq 'pp' } | ForEach-Object tokPerSec)
                $candidatePp = @($rawSamples | Where-Object { $_.stack -eq $CandidateStack -and $_.metric -eq 'pp' } | ForEach-Object tokPerSec)
                $baseTg = @($rawSamples | Where-Object { $_.stack -eq $BaselineStack -and $_.metric -eq 'tg' } | ForEach-Object tokPerSec)
                $candidateTg = @($rawSamples | Where-Object { $_.stack -eq $CandidateStack -and $_.metric -eq 'tg' } | ForEach-Object tokPerSec)
                $ppComparison = Get-AilaMetricComparison -Baseline $basePp -Candidate $candidatePp -HigherIsBetter
                $tgComparison = Get-AilaMetricComparison -Baseline $baseTg -Candidate $candidateTg -HigherIsBetter
            }
        }
        $ppPassed = Test-AilaGate -Comparison $ppComparison -MinimumDeltaPercent -5.0
        $tgPassed = Test-AilaGate -Comparison $tgComparison -MinimumDeltaPercent -5.0
        $performancePassed = $ppPassed -and $tgPassed
        if (-not $ppPassed) { $unresolved.Add('Prefill throughput regression exceeds 5%.') }
        if (-not $tgPassed) { $unresolved.Add('Decode throughput regression exceeds 5%.') }
        $performance = [ordered]@{
            skipped = $false
            reason = $null
            preset = $PerformancePreset
            cases = $PerformanceCases
            order = @($allRuns | ForEach-Object { [pscustomobject]@{ orderTag = $_.orderTag; stack = $_.stack; outputDir = $_.outputDir } })
            rawSamples = @($rawSamples)
            metrics = [ordered]@{
                pp = [ordered]@{ baselineMedian = $ppComparison.baselineMedian; candidateMedian = $ppComparison.candidateMedian; deltaPercent = $ppComparison.deltaPercent; gateMinimumPercent = -5.0; gatePassed = $ppPassed; baselineSampleCount = $basePp.Count; candidateSampleCount = $candidatePp.Count }
                tg = [ordered]@{ baselineMedian = $tgComparison.baselineMedian; candidateMedian = $tgComparison.candidateMedian; deltaPercent = $tgComparison.deltaPercent; gateMinimumPercent = -5.0; gatePassed = $tgPassed; baselineSampleCount = $baseTg.Count; candidateSampleCount = $candidateTg.Count }
            }
        }
        $baselineLogs = @($allRuns | Where-Object stack -eq $BaselineStack | ForEach-Object logPath)
        $candidateLogs = @($allRuns | Where-Object stack -eq $CandidateStack | ForEach-Object logPath)
        $baselineWarmups = @(Get-AilaWarmupSamples -LogPaths $baselineLogs)
        $candidateWarmups = @(Get-AilaWarmupSamples -LogPaths $candidateLogs)
        if ($baselineWarmups.Count -gt 0 -and $candidateWarmups.Count -gt 0) {
            $initComparison = Get-AilaInitClassification -BaselineMs (Get-AilaMedian $baselineWarmups) -CandidateMs (Get-AilaMedian $candidateWarmups)
            $init = [pscustomobject]@{ metric = 'engine-warmup'; reliability = 'partial-phase-only'; baselineSamplesMs = $baselineWarmups; candidateSamplesMs = $candidateWarmups; classification = $initComparison }
            if ($initComparison.blocking) { $performancePassed = $false; $unresolved.Add('Engine warmup initialization is at least 2x baseline.') }
        }
    }

    $ttsBaseline = Get-AilaTtsObservation -AccuracyDir $baselineAccuracyDir -Stack $BaselineStack
    $ttsCandidate = Get-AilaTtsObservation -AccuracyDir $candidateAccuracyDir -Stack $CandidateStack
    $ttsComparison = $null
    $ttsGatePassed = $false
    if ($ttsBaseline.samples.Count -gt 0 -and $ttsCandidate.samples.Count -gt 0) {
        $ttsComparison = Get-AilaMetricComparison -Baseline $ttsBaseline.samples -Candidate $ttsCandidate.samples
        $ttsGatePassed = Test-AilaGate -Comparison $ttsComparison -MinimumDeltaPercent -7.0
        if (-not $ttsGatePassed) { $performancePassed = $false; $unresolved.Add('TTS RTF regression exceeds 7%.') }
    }
    else {
        $unresolved.Add('TTS RTF was unavailable for one or both stacks.')
        if ($null -ne $performancePassed) { $performancePassed = $false }
    }
    $status = Get-AilaTask10Status -AutomaticPassed ([bool]$accuracy.automaticPassed) -ManualReviewRequired ([bool]$accuracy.manualReviewRequired) -PerformancePassed $performancePassed
    $git = Get-AilaGitInfo -RepoRoot $script:RepoRoot
    $report = [ordered]@{
        schemaVersion = $script:ReportSchemaVersion
        startedAtUtc = $startedAtUtc
        completedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
        status = $status
        git = $git
        environment = [ordered]@{
            baseline = [ordered]@{ buildInfo = Read-AilaJsonFile (Join-Path $baselineBuildPath 'build_info.json'); verification = Read-AilaJsonFile (Join-Path $baselineBuildPath 'oneapi_verification.json') }
            candidate = [ordered]@{ buildInfo = Read-AilaJsonFile (Join-Path $candidateBuildPath 'build_info.json'); verification = Read-AilaJsonFile (Join-Path $candidateBuildPath 'oneapi_verification.json') }
        }
        accuracy = [ordered]@{ automaticPassed = [bool]$accuracy.automaticPassed; manualReviewRequired = [bool]$accuracy.manualReviewRequired; status = [string]$accuracy.status; comparisonPath = $accuracyComparisonPath; selectedCases = $AccuracyCases }
        manualLinks = $manualLinks
        performance = $performance
        ttsRtf = [ordered]@{ baseline = $ttsBaseline; candidate = $ttsCandidate; comparison = $ttsComparison; gateMinimumPercent = -7.0; gatePassed = $ttsGatePassed }
        initialization = $init
        unresolvedIssues = @($unresolved)
    }
    $paths = Write-AilaTask10Report -Root $root -Report $report
    Write-Host "oneAPI comparison report: $($paths.jsonPath)" -ForegroundColor Green
    if ($status -eq 'failed') { return 1 }
    return 0
}

function Invoke-SelfTest {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Check([bool]$Condition, [string]$Name) { if (-not $Condition) { $failures.Add($Name) } }

    $childOutput = @(Invoke-CheckedPwsh -Arguments @('-Command', "'child-output'"))
    Check ($childOutput.Count -eq 0) 'child PowerShell output does not pollute function return values'

    $throughput = Get-AilaMetricComparison -Baseline @(100, 101, 99, 100, 100) -Candidate @(96, 95, 97, 96, 96) -HigherIsBetter
    Check ($throughput.deltaPercent -eq -4.0 -and (Test-AilaGate $throughput -MinimumDeltaPercent -5.0)) 'median -4 throughput pass'
    Check (-not (Test-AilaGate (Get-AilaMetricComparison @(100) @(94) -HigherIsBetter) -MinimumDeltaPercent -5.0)) 'throughput below -5 fails'
    Check ((Get-AilaMetricComparison @(1.0) @(1.06)).deltaPercent -eq -6.0) 'RTF direction normalized'
    Check (Test-AilaGate (Get-AilaMetricComparison @(1.0) @(1.06)) -MinimumDeltaPercent -7.0) 'RTF -6 passes'
    Check (-not (Test-AilaGate (Get-AilaMetricComparison @(1.0) @(1.08)) -MinimumDeltaPercent -7.0)) 'RTF below -7 fails'
    Check ((Get-AilaInitClassification 1000 1150).status -eq 'observe') 'init observe'
    Check ((Get-AilaInitClassification 1000 1300).status -eq 'repeat-and-profile') 'init repeat profile'
    Check ((Get-AilaInitClassification 1000 2100).status -eq 'blocking') 'init 2x blocking'
    Check ((Get-AilaTask10Status $true $true $true) -eq 'manual-review-required') 'manual is not pass'
    Check ((Get-AilaTask10Status $false $false $null) -eq 'failed') 'automatic accuracy failure skips performance and fails'
    Check (Test-AilaExpansionRequired $throughput) 'borderline delta expands'

    $selfRoot = Join-Path (Get-AilaOwnedReportRoot $ReportRoot) 'selftest'
    Remove-Item -LiteralPath $selfRoot -Recurse -Force -ErrorAction SilentlyContinue
    Ensure-AilaDirectory $selfRoot
    $ttsCaseDir = Join-Path $selfRoot 'tts-fixture\case_results'
    Ensure-AilaDirectory $ttsCaseDir
    $ttsLogPath = Join-Path $selfRoot 'tts-fixture\tts.stderr.log'
    Set-Content -LiteralPath $ttsLogPath -Value '[TTS] Synthesis complete: 800 ms, 1.00 s audio, RTF=0.800' -Encoding UTF8
    [pscustomobject]@{
        logs = [pscustomobject]@{ stderr = [pscustomobject]@{ path = $ttsLogPath } }
        process = [pscustomobject]@{ durationMs = 800 }
        wav = [pscustomobject]@{ durationSeconds = 1.0 }
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ttsCaseDir 'qwen3_tts_06_base.json') -Encoding UTF8
    $ttsFixture = Get-AilaTtsObservation -AccuracyDir (Join-Path $selfRoot 'tts-fixture') -Stack 'fixture'
    Check ($ttsFixture.samples -is [object[]] -and $ttsFixture.samples.Count -eq 1) 'single TTS RTF remains an array under strict mode'
    Set-Content -LiteralPath (Join-Path $selfRoot 'report.json') -Value '{"status":"stale-pass"}' -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $selfRoot 'report.md') -Value 'stale pass' -Encoding UTF8
    Reset-AilaOwnedReports -Root $selfRoot
    Check (-not (Test-Path -LiteralPath (Join-Path $selfRoot 'report.json'))) 'stale JSON report invalidated'
    Check (-not (Test-Path -LiteralPath (Join-Path $selfRoot 'report.md'))) 'stale Markdown report invalidated'
    $abba = @('01-baseline', '02-candidate', '03-candidate', '04-baseline')
    $syntheticSamples = [System.Collections.Generic.List[object]]::new()
    foreach ($entry in $abba) {
        $stack = if ($entry -like '*baseline') { 'baseline' } else { 'candidate' }
        foreach ($metric in @('pp', 'tg')) {
            foreach ($iteration in 1..3) { $syntheticSamples.Add([pscustomobject]@{ orderTag = $entry; stack = $stack; metric = $metric; iteration = $iteration; tokPerSec = if ($stack -eq 'baseline') { 100.0 } else { 96.0 } }) }
        }
    }
    Check (@($syntheticSamples | Where-Object { $_.stack -eq 'baseline' -and $_.metric -eq 'pp' }).Count -eq 6) 'ABBA aggregates six samples per stack'
    $syntheticSamples.Add([pscustomobject]@{ orderTag = '05-expand-baseline'; stack = 'baseline'; metric = 'pp'; iteration = 1; tokPerSec = 100.0 })
    Check (@($syntheticSamples | Where-Object { $_.stack -eq 'baseline' -and $_.metric -eq 'pp' }).Count -eq 7) 'ABBA expansion reaches seven samples'

    $syntheticReport = [pscustomobject]@{
        status = 'manual-review-required'; completedAtUtc = '2026-01-01T00:00:00Z'; git = [pscustomobject]@{ fullCommit = 'abc'; branch = 'test' }
        accuracy = [pscustomobject]@{ automaticPassed = $true; manualReviewRequired = $true; comparisonPath = 'accuracy.json' }
        performance = [pscustomobject]@{
            skipped = $false
            reason = $null
            rawSamples = @($syntheticSamples)
            metrics = [pscustomobject]@{
                pp = [pscustomobject]@{ baselineMedian = $throughput.baselineMedian; candidateMedian = $throughput.candidateMedian; deltaPercent = $throughput.deltaPercent; gatePassed = $true }
                tg = [pscustomobject]@{ baselineMedian = $throughput.baselineMedian; candidateMedian = $throughput.candidateMedian; deltaPercent = $throughput.deltaPercent; gatePassed = $true }
            }
        }
        ttsRtf = [pscustomobject]@{ baseline = [pscustomobject]@{ samples = @(1.0); sampleCount = 1; source = 'synthetic' }; candidate = [pscustomobject]@{ samples = @(1.06); sampleCount = 1; source = 'synthetic' }; comparison = Get-AilaMetricComparison @(1.0) @(1.06); gatePassed = $true }
        initialization = [pscustomobject]@{ metric = 'engine-warmup'; reliability = 'synthetic'; classification = Get-AilaInitClassification 1000 1150 }
        manualLinks = @('manual.wav'); unresolvedIssues = @('manual review')
    }
    $paths = Write-AilaTask10Report -Root $selfRoot -Report $syntheticReport
    $parsed = Get-Content -LiteralPath $paths.jsonPath -Raw | ConvertFrom-Json
    $markdown = Get-Content -LiteralPath $paths.markdownPath -Raw
    Check ($parsed.status -eq 'manual-review-required') 'report JSON parses'
    Check ($markdown.Contains('manual-review-required') -and -not $markdown.Contains('Status: **pass**')) 'markdown does not print pass for manual review'
    if ($failures.Count -gt 0) { throw "compare_oneapi self-test failures: $($failures -join ', ')" }
    Write-Host 'compare_oneapi self-test: PASS' -ForegroundColor Green
}

if ($SelfTest) {
    Invoke-SelfTest
    exit 0
}

try {
    exit (Invoke-AilaFullComparison)
}
catch {
    $message = $_.Exception.Message
    try {
        $root = Get-AilaOwnedReportRoot -Requested $ReportRoot
        $git = Get-AilaGitInfo -RepoRoot $script:RepoRoot
        $failed = [pscustomobject]@{
            schemaVersion = $script:ReportSchemaVersion; startedAtUtc = $null; completedAtUtc = (Get-Date).ToUniversalTime().ToString('o'); status = 'failed'; git = $git
            environment = [pscustomobject]@{}; accuracy = [pscustomobject]@{ automaticPassed = $false; manualReviewRequired = $false; comparisonPath = $null }
            manualLinks = @(); performance = [pscustomobject]@{ skipped = $true; reason = $message; rawSamples = @(); metrics = $null }
            ttsRtf = [pscustomobject]@{ baseline = [pscustomobject]@{ samples = @(); sampleCount = 0; source = 'unavailable' }; candidate = [pscustomobject]@{ samples = @(); sampleCount = 0; source = 'unavailable' }; comparison = $null; gatePassed = $false }
            initialization = [pscustomobject]@{ metric = 'none'; reliability = 'not-isolated'; classification = Get-AilaInitClassification $null $null }
            unresolvedIssues = @($message)
        }
        Write-AilaTask10Report -Root $root -Report $failed | Out-Null
    }
    catch { }
    Write-Error $message
    exit 1
}
