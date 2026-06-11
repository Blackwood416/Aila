param(
    [string]$AilaExe = ".\build\Aila.exe",
    [string]$ModelDir = ".\models\qwen3.5-4B-bnb-nf4-offline"
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $repoRoot "perf\PerfCommon.ps1")
Initialize-AilaOneApiEnvironment

function Resolve-ToolSmokePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

$resolvedAilaExe = Resolve-ToolSmokePath -Path $AilaExe
$resolvedModelDir = Resolve-ToolSmokePath -Path $ModelDir

function Invoke-ChatJson {
    param(
        [Parameter(Mandatory = $true)]
        $Request
    )

    $json = $Request | ConvertTo-Json -Depth 32 -Compress
    $out = $json | & $resolvedAilaExe -m $resolvedModelDir --greedy --no-stream --messages-json - --chat-output-json --log-level error
    if ($LASTEXITCODE -ne 0) {
        throw "Aila exited with code $LASTEXITCODE. Output: $out"
    }

    $line = $out | Where-Object { $_.Trim().StartsWith("{") } | Select-Object -Last 1
    if (-not $line) {
        throw "No JSON object returned. Output: $out"
    }

    return $line | ConvertFrom-Json
}

function Invoke-ChatJsonStream {
    param(
        [Parameter(Mandatory = $true)]
        $Request
    )

    $json = $Request | ConvertTo-Json -Depth 32 -Compress
    $out = $json | & $resolvedAilaExe -m $resolvedModelDir --greedy --messages-json - --chat-stream-jsonl --log-level error
    if ($LASTEXITCODE -ne 0) {
        throw "Aila stream exited with code $LASTEXITCODE. Output: $out"
    }

    $events = @($out | Where-Object { $_.Trim().StartsWith("{") } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($events.Count -lt 1) {
        throw "No JSONL stream events returned. Output: $out"
    }
    return $events
}

$tools = @(
    @{
        type = "function"
        function = @{
            name = "search"
            description = "Search a small local index"
            parameters = @{
                type = "object"
                properties = @{
                    query = @{
                        type = "string"
                    }
                }
                required = @("query")
            }
        }
    }
)

$requiredReq = @{
    messages = @(
        @{
            role = "user"
            content = "Use the search tool to find the capital of France."
        }
    )
    tools = $tools
    tool_choice = "required"
    max_tokens = 128
    temperature = 0.0
}

$first = Invoke-ChatJson -Request $requiredReq
if ($null -eq $first.tool_calls -or @($first.tool_calls).Count -lt 1) {
    throw "Expected at least one tool call for required tool_choice. Warnings: $($first.warnings -join '|') Raw: $($first.raw_text)"
}

$streamFinal = $null
$streamToolDelta = $null
$streamAttemptErrors = @()
for ($attempt = 1; $attempt -le 3; $attempt++) {
    $streamEvents = Invoke-ChatJsonStream -Request $requiredReq
    $candidateFinal = @($streamEvents | Where-Object { $_.type -eq "final" }) | Select-Object -Last 1
    $candidateToolDelta = @(
        $streamEvents |
            Where-Object {
                $_.type -eq "tool_call_delta" -and
                -not [string]::IsNullOrWhiteSpace([string]$_.tool_name)
            }
    ) | Select-Object -First 1

    if ($null -ne $candidateFinal -and
        $candidateFinal.finish_reason -eq "tool_calls" -and
        $null -ne $candidateFinal.tool_calls -and
        @($candidateFinal.tool_calls).Count -ge 1 -and
        $null -ne $candidateToolDelta) {
        $streamFinal = $candidateFinal
        $streamToolDelta = $candidateToolDelta
        break
    }

    $streamAttemptErrors += "attempt ${attempt}: final=$($candidateFinal | ConvertTo-Json -Depth 8 -Compress)"
}
if ($null -eq $streamFinal -or $null -eq $streamToolDelta) {
    throw "Expected streaming tool-call handoff within retries. $($streamAttemptErrors -join ' | ')"
}

$call = @($first.tool_calls)[0]
$followReq = @{
    messages = @(
        @{
            role = "user"
            content = "Use the search tool to find the capital of France."
        },
        @{
            role = "assistant"
            content = ""
            tool_calls = @($call)
        },
        @{
            role = "tool"
            tool_call_id = $call.id
            content = "Paris is the capital of France."
        }
    )
    tools = $tools
    tool_choice = "none"
    max_tokens = 128
    temperature = 0.0
}

$final = Invoke-ChatJson -Request $followReq
if ([string]::IsNullOrWhiteSpace([string]$final.content)) {
    throw "Expected final content after tool result. Raw: $($final.raw_text)"
}

Write-Host "TOOL_WORKFLOW_OK first_finish=$($first.finish_reason) stream_finish=$($streamFinal.finish_reason) final_finish=$($final.finish_reason) final_content=$($final.content)"
