$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$comparator = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\perf\compare-token-traces.ps1'))
$pwshPath = (Get-Command pwsh -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$tempParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/')
$testRoot = [IO.Path]::GetFullPath((Join-Path $tempParent ('Aila token trace tests ' + [guid]::NewGuid().ToString('N'))))
$tempPrefix = $tempParent + [IO.Path]::DirectorySeparatorChar
if (-not $testRoot.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Test directory is outside the intended temporary parent.'
}
$null = [IO.Directory]::CreateDirectory($testRoot)
$script:caseIndex = 0
$script:failures = [System.Collections.Generic.List[string]]::new()
$script:passed = 0

function Assert-True {
    param([bool]$Value, [string]$Message)
    if (-not $Value) { throw $Message }
}

function New-Trace {
    param([int[]]$Ids = @(10, 11, 12), [int]$PromptCount = 3, [int[]]$PromptIds = @(),
          [switch]$IncludePrompt, [int]$GeneratedCount = -1)

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('Ordinary CLI output does not participate in token parity.')
    if ($IncludePrompt) {
        $lines.Add("[DebugToken] prompt_tokens=$PromptCount (showing $($PromptIds.Count))")
        for ($index = 0; $index -lt $PromptIds.Count; ++$index) {
            $lines.Add("[DebugToken] prompt[$index]=$($PromptIds[$index]) text='prompt'")
        }
    }
    for ($index = 0; $index -lt $Ids.Count; ++$index) {
        $lines.Add("[DebugToken] step=$index id=$($Ids[$index]) text='hello'")
    }
    if ($GeneratedCount -lt 0) { $GeneratedCount = $Ids.Count }
    $lines.Add("[GenerateMessages] Prompt=$PromptCount Generated=$GeneratedCount in 1.23 ms")
    return [string]::Join([Environment]::NewLine, $lines)
}

function Invoke-Comparison {
    param([AllowEmptyString()][string]$Baseline, [AllowEmptyString()][string]$Candidate,
          [switch]$MissingBaseline, [switch]$MissingCandidate, [switch]$NoOutputPath,
          [switch]$OutputOverwritesBaseline, [switch]$Subprocess)

    ++$script:caseIndex
    $caseDirectory = Join-Path $testRoot ("case $($script:caseIndex)")
    $null = [IO.Directory]::CreateDirectory($caseDirectory)
    $baselinePath = Join-Path $caseDirectory 'baseline log.txt'
    $candidatePath = Join-Path $caseDirectory 'candidate log.txt'
    $outputPath = Join-Path $caseDirectory 'comparison result.json'
    if (-not $MissingBaseline) { [IO.File]::WriteAllText($baselinePath, $Baseline, [Text.UTF8Encoding]::new($false)) }
    if (-not $MissingCandidate) { [IO.File]::WriteAllText($candidatePath, $Candidate, [Text.UTF8Encoding]::new($false)) }
    if ($OutputOverwritesBaseline) { $outputPath = $baselinePath }

    $stdout = ''
    $stderr = ''
    $exitCode = 0
    if ($Subprocess) {
        $start = [Diagnostics.ProcessStartInfo]::new()
        $start.FileName = $pwshPath
        $start.WorkingDirectory = $caseDirectory
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        foreach ($argument in @('-NoProfile', '-File', $comparator, '-BaselineLog', $baselinePath,
                                 '-CandidateLog', $candidatePath)) {
            $start.ArgumentList.Add($argument)
        }
        if (-not $NoOutputPath) {
            $start.ArgumentList.Add('-OutputPath')
            $start.ArgumentList.Add($outputPath)
        }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $start
        try {
            $null = $process.Start()
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit(15000)) {
                $process.Kill($true)
                $process.WaitForExit()
                throw 'Offline comparator exceeded the 15-second test timeout.'
            }
            $stdout = $stdoutTask.GetAwaiter().GetResult()
            $stderr = $stderrTask.GetAwaiter().GetResult()
            $exitCode = $process.ExitCode
        } finally {
            $process.Dispose()
        }
    } else {
        $global:LASTEXITCODE = 0
        $outLines = if ($NoOutputPath) {
            & $comparator -BaselineLog $baselinePath -CandidateLog $candidatePath 2>$null
        } else {
            & $comparator -BaselineLog $baselinePath -CandidateLog $candidatePath -OutputPath $outputPath 2>$null
        }
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        $stdout = [string]::Join([Environment]::NewLine, @($outLines))
    }
    try { $json = $stdout | ConvertFrom-Json -Depth 20 }
    catch { throw "Comparator did not emit JSON. Exit=$exitCode stderr=$stderr stdout=$stdout" }
    if (-not $NoOutputPath -and -not $OutputOverwritesBaseline) {
        Assert-True (Test-Path -LiteralPath $outputPath -PathType Leaf) 'Missing comparison JSON file.'
        $fileJson = Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json -Depth 20
        Assert-True (($fileJson | ConvertTo-Json -Depth 20 -Compress) -ceq
                     ($json | ConvertTo-Json -Depth 20 -Compress)) 'File and stdout JSON differ.'
    }
    if ($OutputOverwritesBaseline) {
        Assert-True ([IO.File]::ReadAllText($baselinePath) -ceq $Baseline) 'Comparator overwrote its input log.'
    }
    return [pscustomobject]@{ exitCode = $exitCode; json = $json; stderr = $stderr }
}

function Assert-Invalid {
    param($Result, [string]$Reason)
    Assert-True ($Result.exitCode -ne 0 -and -not $Result.json.equal -and
                 -not $Result.json.integrityPassed -and $Result.json.status -eq 'invalid_trace') 'Invalid trace passed comparison.'
    Assert-True (($Result.json.errors -join ' ') -match $Reason) "Missing expected diagnostic '$Reason'."
}

function Test-Case {
    param([string]$Name, [scriptblock]$Body)
    try {
        & $Body
        ++$script:passed
        Write-Host "PASS: $Name"
    } catch {
        $script:failures.Add("$Name`: $($_.Exception.Message)")
        Write-Host "FAIL: $Name`: $($_.Exception.Message)"
    }
}

try {
    $valid = New-Trace
    Test-Case 'equal tokens with different display text and paths containing spaces' {
        $result = Invoke-Comparison $valid ($valid.Replace("text='hello'", "text='你好'")) -Subprocess
        Assert-True ($result.exitCode -eq 0 -and $result.json.equal -and $result.json.integrityPassed) 'Valid equal trace failed.'
        Assert-True ($result.json.baseline.sha256 -match '^[0-9A-F]{64}$' -and
                     $result.json.baseline.sha256 -cne $result.json.candidate.sha256) 'Log content SHA256 was not recorded.'
        Assert-True ($result.json.baseline.promptTokens -eq 3 -and $result.json.baseline.generatedTokens -eq 3) 'Wrong token counts.'
        Assert-True ($null -eq $result.json.firstDifference -and -not $result.json.promptComparison.fullPromptCompared) 'Unavailable prompt was claimed complete.'
    }
    Test-Case 'optional output path still emits JSON' {
        $result = Invoke-Comparison $valid $valid -NoOutputPath
        Assert-True ($result.exitCode -eq 0 -and $result.json.equal) 'Comparison without OutputPath failed.'
    }
    Test-Case 'different generated ID reports first difference' {
        $result = Invoke-Comparison $valid (New-Trace -Ids @(10, 99, 12))
        Assert-True ($result.exitCode -ne 0 -and $result.json.integrityPassed -and -not $result.json.equal) 'Different IDs passed.'
        Assert-True ($result.json.firstDifference.index -eq 1 -and $result.json.firstDifference.baselineId -eq 11 -and
                     $result.json.firstDifference.candidateId -eq 99) 'Wrong first token difference.'
    }
    Test-Case 'different complete generation lengths report absent ID' {
        $result = Invoke-Comparison $valid (New-Trace -Ids @(10, 11))
        Assert-True ($result.exitCode -ne 0 -and $result.json.integrityPassed -and
                     $result.json.firstDifference.index -eq 2 -and $null -eq $result.json.firstDifference.candidateId) 'Length difference was not reported.'
    }
    Test-Case 'missing step fails even when later IDs match' {
        $missing = $valid -replace '(?m)^\[DebugToken\] step=1[^\r\n]*(\r?\n|$)', ''
        Assert-Invalid (Invoke-Comparison $valid $missing) 'sequence|missing'
    }
    Test-Case 'duplicate step fails' {
        $duplicated = $valid.Replace("[DebugToken] step=1 id=11 text='hello'",
            "[DebugToken] step=1 id=11 text='hello'`n[DebugToken] step=1 id=11 text='hello'")
        Assert-Invalid (Invoke-Comparison $valid $duplicated) 'sequence|duplicated'
    }
    Test-Case 'out-of-order steps fail with unchanged IDs' {
        $reordered = $valid.Replace('step=0 id=10', 'step=1 id=10').Replace('step=1 id=11', 'step=0 id=11')
        Assert-Invalid (Invoke-Comparison $valid $reordered) 'file order'
    }
    Test-Case 'trace beyond 64-step cap is rejected' {
        Assert-Invalid (Invoke-Comparison $valid (New-Trace -Ids @(0..63) -GeneratedCount 65)) '64-step'
    }
    Test-Case 'truncated final summary is rejected' {
        Assert-Invalid (Invoke-Comparison $valid ($valid -replace '\[GenerateMessages\][^\r\n]*', '')) 'exactly one'
    }
    Test-Case 'duplicate terminal summaries are rejected' {
        Assert-Invalid (Invoke-Comparison $valid ($valid + "`n[GenerateMessages] Prompt=3 Generated=3 in 2 ms")) 'found 2'
    }
    Test-Case 'duplicate terminal markers on the same line are rejected' {
        Assert-Invalid (Invoke-Comparison $valid ($valid + ' [GenerateMessages] Prompt=3 Generated=3 in 2 ms')) 'found 2'
    }
    Test-Case 'duplicate step markers on the same line are rejected' {
        $duplicated = $valid.Replace("[DebugToken] step=1 id=11 text='hello'",
            "[DebugToken] step=1 id=11 text='hello' [DebugToken] step=1 id=11 text='hello'")
        Assert-Invalid (Invoke-Comparison $valid $duplicated) 'sequence|duplicated'
    }
    Test-Case 'terminal before steps is rejected' {
        $early = "[GenerateMessages] Prompt=3 Generated=3 in 1 ms`n" +
                 ($valid -replace '\[GenerateMessages\][^\r\n]*', '')
        Assert-Invalid (Invoke-Comparison $valid $early) 'after the terminal'
    }
    Test-Case 'empty log is rejected' {
        Assert-Invalid (Invoke-Comparison $valid '') 'exactly one'
    }
    Test-Case 'stdout-only text cannot serve as token evidence' {
        Assert-Invalid (Invoke-Comparison $valid 'The answer is 2.') 'exactly one'
    }
    Test-Case 'terminal without debug steps is rejected' {
        Assert-Invalid (Invoke-Comparison $valid '[GenerateMessages] Prompt=3 Generated=3 in 1 ms') 'does not match'
    }
    Test-Case 'zero generated tokens are rejected' {
        Assert-Invalid (Invoke-Comparison $valid (New-Trace -Ids @())) 'greater than zero'
    }
    Test-Case 'negative and malformed IDs are rejected' {
        Assert-Invalid (Invoke-Comparison $valid (New-Trace -Ids @(-1, 11, 12))) 'nonnegative'
        Assert-Invalid (Invoke-Comparison $valid ($valid.Replace('id=11', 'id=not-an-id'))) 'Malformed'
        Assert-Invalid (Invoke-Comparison $valid ($valid.Replace('id=11', 'id=2147483648'))) '32-bit integer'
    }
    Test-Case 'malformed summary is not silently ignored' {
        Assert-Invalid (Invoke-Comparison $valid ($valid.Replace('Generated=3', 'Generated=bad'))) 'Malformed'
    }
    Test-Case 'prompt count mismatch fails even with identical generated tokens' {
        $result = Invoke-Comparison $valid (New-Trace -PromptCount 4)
        Assert-True ($result.exitCode -ne 0 -and -not $result.json.equal -and $result.json.tokenSequencesEqual -and
                     -not $result.json.promptComparison.totalLengthEqual) 'Prompt count mismatch passed.'
    }
    Test-Case 'full short prompt equality is recorded' {
        $trace = New-Trace -IncludePrompt -PromptIds @(1, 2, 3)
        $result = Invoke-Comparison $trace $trace
        Assert-True ($result.exitCode -eq 0 -and $result.json.promptComparison.fullPromptCompared -and
                     $result.json.promptComparison.comparedPrefixTokens -eq 3) 'Complete prompt trace was not compared.'
    }
    Test-Case 'available prompt prefix mismatch fails' {
        $result = Invoke-Comparison (New-Trace -IncludePrompt -PromptIds @(1, 2, 3)) (New-Trace -IncludePrompt -PromptIds @(1, 8, 3))
        Assert-True ($result.exitCode -ne 0 -and $result.json.tokenSequencesEqual -and
                     $result.json.promptComparison.firstDifference.index -eq 1) 'Prompt prefix mismatch passed.'
    }
    Test-Case 'long prompt checks only recorded prefix without claiming full equality' {
        $trace = New-Trace -PromptCount 80 -IncludePrompt -PromptIds @(0..63)
        $result = Invoke-Comparison $trace $trace
        Assert-True ($result.exitCode -eq 0 -and $result.json.equal -and
                     $result.json.promptComparison.comparedPrefixTokens -eq 64 -and
                     -not $result.json.promptComparison.fullPromptCompared -and
                     -not $result.json.baseline.promptTrace.fullPromptRecorded) 'Long prompt completeness was overstated.'
    }
    Test-Case 'missing advertised prompt row is rejected' {
        $trace = New-Trace -IncludePrompt -PromptIds @(1, 2, 3)
        $missing = $trace -replace '(?m)^\[DebugToken\] prompt\[1\][^\r\n]*(\r?\n|$)', ''
        Assert-Invalid (Invoke-Comparison $trace $missing) 'advertises|contiguous'
    }
    Test-Case 'prompt header must match terminal total' {
        $trace = New-Trace -IncludePrompt -PromptIds @(1, 2, 3)
        Assert-Invalid (Invoke-Comparison $trace ($trace.Replace('prompt_tokens=3', 'prompt_tokens=4'))) 'does not match'
    }
    Test-Case 'missing logs fail with structured diagnostics' {
        $missing = Invoke-Comparison $valid $valid -MissingCandidate
        Assert-Invalid $missing 'does not exist'
        Assert-True ($null -eq $missing.json.candidate.sha256) 'Missing log had a fabricated hash.'
        Assert-Invalid (Invoke-Comparison $valid $valid -MissingBaseline) 'does not exist'
    }
    Test-Case 'JSON output cannot overwrite an input log' {
        $result = Invoke-Comparison $valid $valid -OutputOverwritesBaseline
        Assert-True ($result.exitCode -ne 0 -and -not $result.json.equal -and $result.json.status -eq 'io_error') 'Input overwrite was allowed.'
    }
} finally {
    # Delete only the unique directory allocated by this test invocation.
    $resolvedTestRoot = [IO.Path]::GetFullPath($testRoot)
    if (-not $resolvedTestRoot.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetDirectoryName($resolvedTestRoot) -cne $tempParent) {
        throw 'Refusing to remove a test directory outside its intended temporary parent.'
    }
    Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
}
Write-Host "TokenTraceComparisonTests: $($script:passed) passed, $($script:failures.Count) failed"
if ($script:failures.Count -ne 0) {
    foreach ($failure in $script:failures) { [Console]::Error.WriteLine($failure) }
    exit 1
}
exit 0
