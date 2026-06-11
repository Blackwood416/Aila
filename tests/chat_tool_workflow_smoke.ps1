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

Write-Host "TOOL_WORKFLOW_OK first_finish=$($first.finish_reason) final_finish=$($final.finish_reason) final_content=$($final.content)"
