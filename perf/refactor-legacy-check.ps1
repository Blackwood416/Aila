param(
    [Parameter(Mandatory = $true)][string]$BaselineBuild,
    [Parameter(Mandatory = $true)][string]$CandidateBuild,
    [Parameter(Mandatory = $true)][string]$ModelDir,
    [string]$OutputDir = 'tmp\refactor\legacy-parity',
    [string]$OneApiStack = 'oneapi-2026.2'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'PerfCommon.ps1')

$repoRoot = Get-AilaRepoRoot
$baselinePath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BaselineBuild
$candidatePath = Resolve-AilaPath -RepoRoot $repoRoot -Path $CandidateBuild
$modelPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $ModelDir
$outputPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputDir
$stackConfig = Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path 'perf\oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $stackConfig -Name $OneApiStack
$builds = @(
    @{ name = 'baseline'; path = $baselinePath },
    @{ name = 'candidate'; path = $candidatePath }
)
foreach ($build in $builds) {
    Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $build.path -Stack $stack
    Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $build.path -Stack $stack -RequireValues
    $build.executable = Join-Path $build.path 'Aila.exe'
    $build.sha256 = (Get-FileHash -LiteralPath $build.executable -Algorithm SHA256).Hash
    $build.info = Read-AilaJsonFile -Path (Join-Path $build.path 'build_info.json')
}
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)
Ensure-AilaDirectory -Path $outputPath
$pwshPath = (Get-Command pwsh -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$comparatorPath = Join-Path $PSScriptRoot 'compare-token-traces.ps1'
$traceEnv = @{ AILA_DEBUG_TOKEN_IDS = '1'; AILA_LOG_LEVEL = 'info'; AILA_KV_QUANT = '0' }
$cases = @(
    @{ name = 'fixed_seed'; input = 'smoke_text_qa.json'; options = @('--sample', '--seed', '123', '-t', '0.8', '-k', '20', '-p', '0.9', '--no-stream') },
    @{ name = 'penalties'; input = 'smoke_text_qa.json'; options = @('--sample', '--seed', '5678', '-t', '0.65', '-k', '80', '-p', '0.9', '--rep-penalty', '1.08', '--pres-penalty', '0.2', '--freq-penalty', '0.05', '--no-stream') },
    @{ name = 'stream'; input = 'smoke_text_qa.json'; options = @('--greedy', '--stream', '--stream-chunk', '4') },
    @{ name = 'thinking_budget'; input = 'smoke_text_qa.json'; options = @('--greedy', '--thinking-budget', '4', '--no-stream') },
    @{ name = 'history'; input = 'smoke_multi_turn.json'; options = @('--greedy', '--no-stream') }
)
$modelFiles = @(
    Get-ChildItem -LiteralPath $modelPath -File |
        Where-Object { $_.Extension -eq '.safetensors' -or $_.Name -in @(
            'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
            'vocab.json', 'merges.txt', 'special_tokens_map.json', 'chat_template.jinja') } |
        Sort-Object Name |
        ForEach-Object { @{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } }
)
$results = [System.Collections.Generic.List[object]]::new()
$report = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = [DateTime]::UtcNow.ToString('o')
    modelDirectory = $modelPath
    modelFiles = $modelFiles
    oneApi = Get-AilaOneApiStackMetadata -Stack $stack
    builds = $builds
    envOverrides = $traceEnv
    cases = @()
    passed = $false
}

try {
    # Keep all GPU processes sequential, including baseline/candidate pairs.
    foreach ($case in $cases) {
        $inputPath = Join-Path $repoRoot ('perf\fixtures\' + $case.input)
        $arguments = @('-m', $modelPath, '--messages-json', $inputPath,
                       '--max-seq', '4096', '--max-tokens', '32') + $case.options
        $caseResult = [ordered]@{
            name = $case.name
            inputPath = $inputPath
            inputSha256 = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
            arguments = $arguments
            executions = @()
            comparison = $null
        }
        $results.Add($caseResult)
        foreach ($build in $builds) {
            $logPath = Join-Path $outputPath ($case.name + '-' + $build.name + '.log')
            $run = Invoke-AilaProcess -Executable $build.executable -ArgumentList $arguments `
                -WorkingDirectory $build.path -EnvOverrides $traceEnv -LogPath $logPath -TimeoutSeconds 180
            $caseResult.executions += [ordered]@{
                build = $build.name
                logPath = $logPath
                exitCode = $run.exitCode
                timedOut = $run.timedOut
                durationMs = $run.durationMs
            }
            if ($run.timedOut -or $run.exitCode -ne 0) {
                throw "Legacy case '$($case.name)' failed on $($build.name); see '$logPath'."
            }
        }
        $comparisonPath = Join-Path $outputPath ($case.name + '-comparison.json')
        & $pwshPath -NoProfile -File $comparatorPath `
            -BaselineLog $caseResult.executions[0].logPath `
            -CandidateLog $caseResult.executions[1].logPath -OutputPath $comparisonPath > $null
        $comparisonExit = $LASTEXITCODE
        $caseResult.comparison = Read-AilaJsonFile -Path $comparisonPath
        if ($comparisonExit -ne 0 -or -not $caseResult.comparison.equal) {
            throw "Legacy case '$($case.name)' has incomplete or different token traces; see '$comparisonPath'."
        }
        Write-Host ("PASS {0}: {1} identical generated tokens" -f $case.name, $caseResult.comparison.baseline.generatedTokens)
    }
    foreach ($build in $builds) {
        if ((Get-FileHash -LiteralPath $build.executable -Algorithm SHA256).Hash -ne $build.sha256) {
            throw "The $($build.name) executable changed while verification was running."
        }
    }
    $report.passed = $true
} finally {
    $report.cases = $results.ToArray()
    Write-AilaJsonFile -Path (Join-Path $outputPath 'report.json') -Data $report
}
