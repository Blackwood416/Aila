[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BaselineLog,
    [Parameter(Mandatory = $true)][string]$CandidateLog,
    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# This is an offline comparator. It neither invokes Aila nor sources the
# regression runner, whose top-level code launches model inference.
function ConvertTo-TraceInteger {
    param([string]$Text, [string]$Label, [System.Collections.Generic.List[string]]$Errors)

    $value = 0
    if (-not [int]::TryParse($Text, [ref]$value)) {
        $Errors.Add("$Label is not a supported 32-bit integer: '$Text'.")
        return $null
    }
    return $value
}

function Read-TokenTrace {
    param([Parameter(Mandatory = $true)][string]$Path)

    $trace = [ordered]@{
        path = $Path
        sha256 = $null
        bytes = 0
        summaryCount = 0
        promptTokens = $null
        generatedTokens = $null
        stepCount = 0
        steps = @()
        tokenIds = @()
        complete = $false
        promptTrace = [ordered]@{
            available = $false
            headerCount = 0
            advertisedTokens = $null
            advertisedShown = $null
            recordedTokens = 0
            indices = @()
            tokenIds = @()
            fullPromptRecorded = $false
        }
        errors = @()
    }
    $errors = [System.Collections.Generic.List[string]]::new()
    try {
        $trace.path = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
        if (-not (Test-Path -LiteralPath $trace.path -PathType Leaf)) {
            throw "Log file does not exist: $($trace.path)"
        }
        # Hash exactly the bytes being parsed, even if a producer mistakenly
        # continues appending to the original file during this comparison.
        $bytes = [System.IO.File]::ReadAllBytes($trace.path)
        $trace.bytes = $bytes.LongLength
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            $trace.sha256 = [BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-', '')
        } finally {
            $hasher.Dispose()
        }
        $memory = [System.IO.MemoryStream]::new($bytes, $false)
        $reader = [System.IO.StreamReader]::new($memory, [System.Text.UTF8Encoding]::new($false, $true), $true)
        try {
            $content = $reader.ReadToEnd()
        } finally {
            $reader.Dispose()
        }
    } catch {
        $trace.errors = @("Cannot read log: $($_.Exception.Message)")
        return [pscustomobject]$trace
    }

    $summaryLines = [System.Collections.Generic.List[object]]::new()
    $stepLines = [System.Collections.Generic.List[object]]::new()
    $promptHeaders = [System.Collections.Generic.List[object]]::new()
    $promptLines = [System.Collections.Generic.List[object]]::new()
    $markerKinds = @(
        @{ pattern = '\[GenerateMessages\]\s+Prompt='; records = $summaryLines },
        @{ pattern = '\[DebugToken\]\s+step='; records = $stepLines },
        @{ pattern = '\[DebugToken\]\s+prompt_tokens='; records = $promptHeaders },
        @{ pattern = '\[DebugToken\]\s+prompt\['; records = $promptLines }
    )
    $lineNumber = 0
    foreach ($line in [regex]::Split($content, '\r\n|\n|\r')) {
        ++$lineNumber
        foreach ($kind in $markerKinds) {
            # Count markers, not merely lines: concatenated duplicate records
            # must not look like one complete summary or one generated step.
            foreach ($marker in [regex]::Matches($line, $kind.pattern)) {
                $kind.records.Add([pscustomobject]@{
                    line = $lineNumber
                    column = $marker.Index
                    text = $line.Substring($marker.Index)
                })
            }
        }
    }

    $trace.summaryCount = $summaryLines.Count
    if ($summaryLines.Count -ne 1) {
        $errors.Add("Expected exactly one GenerateMessages terminal summary; found $($summaryLines.Count).")
    } else {
        $match = [regex]::Match($summaryLines[0].text,
            '\[GenerateMessages\]\s+Prompt=(?<prompt>-?\d+)\s+Generated=(?<generated>-?\d+)(?=\s|$)')
        if (-not $match.Success) {
            $errors.Add("Malformed GenerateMessages summary at line $($summaryLines[0].line).")
        } else {
            $trace.promptTokens = ConvertTo-TraceInteger $match.Groups['prompt'].Value 'Prompt count' $errors
            $trace.generatedTokens = ConvertTo-TraceInteger $match.Groups['generated'].Value 'Generated count' $errors
            if ($null -ne $trace.promptTokens -and $trace.promptTokens -lt 0) {
                $errors.Add('Prompt count must be nonnegative.')
            }
            if ($null -ne $trace.generatedTokens -and $trace.generatedTokens -le 0) {
                $errors.Add('Generated count must be greater than zero; empty generation is not a parity baseline.')
            }
            if ($null -ne $trace.generatedTokens -and $trace.generatedTokens -gt 64) {
                $errors.Add('Generated count exceeds the 64-step CLI trace limit; the trace cannot prove complete token equality.')
            }
        }
    }

    $steps = [System.Collections.Generic.List[int]]::new()
    $ids = [System.Collections.Generic.List[int]]::new()
    $trace.stepCount = $stepLines.Count
    foreach ($record in $stepLines) {
        $match = [regex]::Match($record.text,
            '\[DebugToken\]\s+step=(?<step>-?\d+)\s+id=(?<id>-?\d+)(?=\s|$)')
        if (-not $match.Success) {
            $errors.Add("Malformed DebugToken step record at line $($record.line).")
            continue
        }
        $step = ConvertTo-TraceInteger $match.Groups['step'].Value "Step at line $($record.line)" $errors
        $id = ConvertTo-TraceInteger $match.Groups['id'].Value "Token ID at line $($record.line)" $errors
        if ($null -eq $step -or $null -eq $id) { continue }
        if ($step -ne $steps.Count) {
            $errors.Add("Step sequence must be 0..N-1 in file order: expected $($steps.Count), found $step at line $($record.line).")
        }
        if ($id -lt 0) { $errors.Add("Token ID must be nonnegative at line $($record.line).") }
        if ($summaryLines.Count -eq 1 -and
            ($record.line -gt $summaryLines[0].line -or
             ($record.line -eq $summaryLines[0].line -and $record.column -ge $summaryLines[0].column))) {
            $errors.Add("Step record at line $($record.line) occurs after the terminal summary.")
        }
        $steps.Add($step)
        $ids.Add($id)
    }
    $trace.steps = $steps.ToArray()
    $trace.tokenIds = $ids.ToArray()
    if ($null -ne $trace.generatedTokens -and
        ($stepLines.Count -ne $trace.generatedTokens -or $ids.Count -ne $trace.generatedTokens)) {
        $errors.Add("Generated count $($trace.generatedTokens) does not match all recorded steps ($($stepLines.Count) records, $($ids.Count) parsed IDs); trace is missing, duplicated or truncated.")
    }

    $promptErrors = [System.Collections.Generic.List[string]]::new()
    $trace.promptTrace.available = $promptHeaders.Count -gt 0 -or $promptLines.Count -gt 0
    $trace.promptTrace.headerCount = $promptHeaders.Count
    if ($promptHeaders.Count -gt 1) {
        $promptErrors.Add("Expected at most one prompt trace header; found $($promptHeaders.Count).")
    } elseif ($promptHeaders.Count -eq 1) {
        $match = [regex]::Match($promptHeaders[0].text,
            '\[DebugToken\]\s+prompt_tokens=(?<total>-?\d+)\s+\(showing (?<shown>-?\d+)\)(?=\s|$)')
        if (-not $match.Success) {
            $promptErrors.Add("Malformed prompt trace header at line $($promptHeaders[0].line).")
        } else {
            $total = ConvertTo-TraceInteger $match.Groups['total'].Value 'Prompt trace total' $promptErrors
            $shown = ConvertTo-TraceInteger $match.Groups['shown'].Value 'Prompt trace shown count' $promptErrors
            $trace.promptTrace.advertisedTokens = $total
            $trace.promptTrace.advertisedShown = $shown
            if ($null -ne $total -and ($total -lt 0 -or $total -ne $trace.promptTokens)) {
                $promptErrors.Add('Prompt trace total does not match the terminal prompt count.')
            }
            if ($null -ne $total -and $null -ne $shown -and
                ($shown -lt 0 -or $shown -ne [Math]::Min($total, 64))) {
                $promptErrors.Add('Prompt trace shown count must equal min(prompt total, 64).')
            }
            if ($null -ne $shown -and $shown -ne $promptLines.Count) {
                $promptErrors.Add("Prompt trace advertises $shown IDs but contains $($promptLines.Count) records.")
            }
        }
    }
    $promptIndices = [System.Collections.Generic.List[int]]::new()
    $promptIds = [System.Collections.Generic.List[int]]::new()
    foreach ($record in $promptLines) {
        $match = [regex]::Match($record.text,
            '\[DebugToken\]\s+prompt\[(?<index>-?\d+)\]=(?<id>-?\d+)(?=\s|$)')
        if (-not $match.Success) {
            $promptErrors.Add("Malformed prompt token record at line $($record.line).")
            continue
        }
        $index = ConvertTo-TraceInteger $match.Groups['index'].Value "Prompt index at line $($record.line)" $promptErrors
        $id = ConvertTo-TraceInteger $match.Groups['id'].Value "Prompt token ID at line $($record.line)" $promptErrors
        if ($null -eq $index -or $null -eq $id) { continue }
        if ($index -ne $promptIndices.Count -or $id -lt 0) {
            $promptErrors.Add("Prompt token indices must be contiguous from zero and IDs nonnegative at line $($record.line).")
        }
        if ($summaryLines.Count -eq 1 -and
            ($record.line -gt $summaryLines[0].line -or
             ($record.line -eq $summaryLines[0].line -and $record.column -ge $summaryLines[0].column))) {
            $promptErrors.Add("Prompt token at line $($record.line) occurs after the terminal summary.")
        }
        $promptIndices.Add($index)
        $promptIds.Add($id)
    }
    if ($null -ne $trace.promptTokens -and
        $promptLines.Count -gt [Math]::Min($trace.promptTokens, 64)) {
        $promptErrors.Add('Prompt trace contains more IDs than its prompt total or the 64-token trace limit.')
    }
    $trace.promptTrace.recordedTokens = $promptIds.Count
    $trace.promptTrace.indices = $promptIndices.ToArray()
    $trace.promptTrace.tokenIds = $promptIds.ToArray()
    $trace.promptTrace.fullPromptRecorded = $trace.promptTrace.available -and
        $promptErrors.Count -eq 0 -and $null -ne $trace.promptTokens -and
        $trace.promptTokens -le 64 -and $promptIds.Count -eq $trace.promptTokens
    foreach ($errorMessage in $promptErrors) { $errors.Add($errorMessage) }
    $trace.errors = $errors.ToArray()
    $trace.complete = $errors.Count -eq 0
    return [pscustomobject]$trace
}

function Find-FirstTokenDifference {
    param([int[]]$Baseline, [int[]]$Candidate, [switch]$CommonPrefixOnly)

    $count = if ($CommonPrefixOnly) { [Math]::Min($Baseline.Count, $Candidate.Count) }
             else { [Math]::Max($Baseline.Count, $Candidate.Count) }
    for ($index = 0; $index -lt $count; ++$index) {
        $baselineId = if ($index -lt $Baseline.Count) { $Baseline[$index] } else { $null }
        $candidateId = if ($index -lt $Candidate.Count) { $Candidate[$index] } else { $null }
        if ($baselineId -ne $candidateId) {
            return [pscustomobject]@{ index = $index; baselineId = $baselineId; candidateId = $candidateId }
        }
    }
    return $null
}

$baseline = Read-TokenTrace -Path $BaselineLog
$candidate = Read-TokenTrace -Path $CandidateLog
$errors = [System.Collections.Generic.List[string]]::new()
foreach ($entry in @(@('Baseline', $baseline), @('Candidate', $candidate))) {
    if (-not $entry[1].complete) {
        foreach ($errorMessage in $entry[1].errors) { $errors.Add("$($entry[0]): $errorMessage") }
    }
}
$integrityPassed = $baseline.complete -and $candidate.complete
$tokenSequencesEqual = $null
$firstDifference = $null
$promptComparison = [ordered]@{
    totalLengthEqual = $null
    comparedPrefixTokens = 0
    availablePrefixEqual = $null
    fullPromptCompared = $false
    firstDifference = $null
}
if ($integrityPassed) {
    $firstDifference = Find-FirstTokenDifference $baseline.tokenIds $candidate.tokenIds
    $tokenSequencesEqual = $null -eq $firstDifference
    if (-not $tokenSequencesEqual) {
        $errors.Add("Generated tokens differ at index $($firstDifference.index): baseline=$($firstDifference.baselineId), candidate=$($firstDifference.candidateId).")
    }
    $promptComparison.totalLengthEqual = $baseline.promptTokens -eq $candidate.promptTokens
    if (-not $promptComparison.totalLengthEqual) {
        $errors.Add("Prompt counts differ: baseline=$($baseline.promptTokens), candidate=$($candidate.promptTokens).")
    }
    if ($baseline.promptTrace.available -and $candidate.promptTrace.available) {
        $promptComparison.comparedPrefixTokens = [Math]::Min(
            $baseline.promptTrace.recordedTokens, $candidate.promptTrace.recordedTokens)
        $promptComparison.firstDifference = Find-FirstTokenDifference `
            $baseline.promptTrace.tokenIds $candidate.promptTrace.tokenIds -CommonPrefixOnly
        $promptComparison.availablePrefixEqual = $null -eq $promptComparison.firstDifference
        if (-not $promptComparison.availablePrefixEqual) {
            $errors.Add("Available prompt token prefixes differ at index $($promptComparison.firstDifference.index).")
        }
        $promptComparison.fullPromptCompared = $promptComparison.totalLengthEqual -and
            $baseline.promptTrace.fullPromptRecorded -and $candidate.promptTrace.fullPromptRecorded
    }
}
$report = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = [DateTime]::UtcNow.ToString('o')
    source = 'AILA_DEBUG_TOKEN_IDS'
    status = if (-not $integrityPassed) { 'invalid_trace' } elseif ($errors.Count -ne 0) { 'different' } else { 'equal' }
    equal = $integrityPassed -and $errors.Count -eq 0
    integrityPassed = $integrityPassed
    tokenSequencesEqual = $tokenSequencesEqual
    firstDifference = $firstDifference
    promptComparison = $promptComparison
    baseline = $baseline
    candidate = $candidate
    errors = $errors.ToArray()
}
if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
    try {
        $resolvedOutput = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
        if ($resolvedOutput.Equals($baseline.path, [StringComparison]::OrdinalIgnoreCase) -or
            $resolvedOutput.Equals($candidate.path, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'OutputPath must not overwrite either input log.'
        }
        $parentDirectory = [IO.Path]::GetDirectoryName($resolvedOutput)
        $null = [IO.Directory]::CreateDirectory($parentDirectory)
        [IO.File]::WriteAllText($resolvedOutput, ($report | ConvertTo-Json -Depth 12) + [Environment]::NewLine,
                               [Text.UTF8Encoding]::new($false))
    } catch {
        $report.status = 'io_error'
        $report.equal = $false
        $errors.Add("Cannot write comparison JSON: $($_.Exception.Message)")
        $report.errors = $errors.ToArray()
    }
}
$report | ConvertTo-Json -Depth 12
if (-not $report.equal) {
    [Console]::Error.WriteLine('Token trace comparison failed: ' + ($report.errors -join ' '))
    exit 1
}
exit 0
