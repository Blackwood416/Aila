<#
.SYNOPSIS
Run the experimental Qwen3.8 A770 supply candidate in a single-session CLI.
.DESCRIPTION
Screening scope: representative 41/47/1833-token inputs and 48 output tokens.
This is an opt-in experiment, not a validated production dispatch. Real expert
demand loading remains enabled independently of speculative prefetch.
.EXAMPLE
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_chat.ps1 -DryRun
.EXAMPLE
pwsh -NoProfile -File perf/qwen4exp/q4e_supply_chat.ps1 -PromptFile perf/qwen4exp/prompts/story.json
#>
[CmdletBinding()]
param(
    [string]$ModelDir = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2',
    [ValidateRange(1,2560)][int]$MaxTokens = 48,
    [string]$PromptFile = '',
    [ValidateSet(512,1024,2048)][int]$PrefillChunk = 2048,
    [ValidateRange(0,512)][int]$HotsetSwaps = 0,
    [ValidateSet(0,1)][int]$DecodePrefetch = 0,
    [ValidateSet(0,1)][int]$PrefillPrefetch = 0,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../PerfCommon.ps1')
$repo = Get-AilaRepoRoot
$exe = Join-Path $repo 'build-refactor/Aila.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Aila executable not found: $exe; build with build.ps1 -BuildDir build-refactor."
}
$resolvedModel = (Resolve-Path -LiteralPath $ModelDir).Path
if (-not (Test-Path -LiteralPath $resolvedModel -PathType Container)) {
    throw "ModelDir must be a directory: $resolvedModel"
}
$profile = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'profiles/qwen38-nf4-h4-v1.profile')).Path
$arguments = @('-m', $resolvedModel, '--performance-preset', 'qwen38-a770',
    '--max-seq', '2560', '--greedy', '--thinking-budget', '0',
    '--max-tokens', [string]$MaxTokens, '--stream', '--stream-chunk', '1', '--decode-chunk', '1')
if (-not [string]::IsNullOrWhiteSpace($PromptFile)) {
    $resolvedPrompt = (Resolve-Path -LiteralPath $PromptFile).Path
    if (-not (Test-Path -LiteralPath $resolvedPrompt -PathType Leaf)) {
        throw "PromptFile must be a JSON file: $resolvedPrompt"
    }
    $null = Get-Content -LiteralPath $resolvedPrompt -Raw | ConvertFrom-Json
    $arguments += @('--messages-json', $resolvedPrompt)
}

# Pin the screened candidate in the same clean runtime environment as the probe.
# Restore the caller's environment after the child exits. The CLI preset owns
# A770/model compatibility checks.
$candidate = @{
    AILA_Q4E_PREFILL_XMX = '1'; AILA_Q4E_DIRECT_PREFILL_IO = '1'
    AILA_Q4E_DEMAND_MICROBATCH = '32'; AILA_Q4E_PHASE_SCRATCH = '1'
    AILA_Q4E_HOT_RAM_MB = '32768'; AILA_Q4E_EXPERT_READER = 'readfile'
    AILA_Q35_PREFILL_CHUNK = [string]$PrefillChunk; AILA_Q4E_INLINE_CHECKPOINT = '1'
    AILA_Q4E_STATIC_PINNING = '0'; AILA_Q4E_GROUPED_XMX_MIN_ROWS = '0'
    AILA_Q4E_GDN_CHUNK_MODE = '0'; AILA_Q4E_SMALL_ROWS_DECODE = '0'
    AILA_Q4E_SMALL_ROWS_HC = '0'; AILA_Q4E_CPU_EXPERT_DISPATCH = '0'
    AILA_Q4E_HOTSET_PROFILE = $profile; AILA_Q4E_HOTSET_INTERVAL = '4'
    AILA_Q4E_HOTSET_SWAPS = [string]$HotsetSwaps; AILA_Q4E_HOTSET_BORROW = '1'
    AILA_Q4E_HOTSET_COLD_QUEUE = '2'; AILA_Q4E_HOTSET_RING = '1'
    AILA_Q4E_HOTSET_PROMPT = '1'; AILA_Q4E_HOTSET_PROMPT_SWAPS = '0'
    AILA_Q4E_HOTSET_PROMPT_PREFILL = '0'; AILA_Q4E_HOTSET_ADOPT_READY = '1'
    AILA_Q4E_HOTSET_ADOPT_LIMIT = '4096'; AILA_Q4E_RESIDENT_PREFILL = '1'
    AILA_Q4E_RESIDENT_DECODE = '2'; AILA_Q4E_COLD_MODE = '2'
    AILA_Q4E_COLD_POLICY = '1'; AILA_Q4E_CPU_MAX_EXPERTS = '10'
    AILA_Q4E_COLD_PCIE_NUM = '141'; AILA_Q4E_COLD_BUDGET_US = '1000'
    AILA_Q4E_CPU_THREADS = '12'; AILA_Q4E_COLD_LOW_RANK = '0'
    AILA_Q4E_PREFETCH = '1'; AILA_Q4E_PREFETCH_DECODE = [string]$DecodePrefetch
    AILA_Q4E_PREFETCH_PREFILL = [string]$PrefillPrefetch; AILA_Q4E_PREDICT_STATS = '0'
    AILA_Q4E_PREFILL_ATTN_TILE = '0'; AILA_Q4E_MOE_COMPACT_ACTS = '0'
    AILA_PROFILE_Q4E_DECODE = '0'; AILA_PROFILE_Q4E_STAGES = '0'
    AILA_PROFILE_Q4E_COLD_DEVICE = '0'; AILA_Q4E_CACHE_TRACE = ''
    AILA_Q4E_TRACE_MARKERS = '0'; AILA_DEBUG_Q4E_FAKE_HITS = '0'
}
$cfg = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $cfg -Name 'oneapi-2026.1'
$stackEnvironment = Get-AilaOneApiStackEnvironment -Stack $stack
# The stack helper imports its inherited environment as well. Restore every key
# it could affect, including managed variables that it explicitly removes.
$savedEnvironment = @{}
$runtimeKeys = @([Environment]::GetEnvironmentVariables('Process').Keys |
    Where-Object { $_ -match '^(AILA_|ONEDNN_|DNNL_)' })
$affectedKeys = @($stackEnvironment.Keys) + @($candidate.Keys) +
    @(Get-AilaOneApiPathListVariableNames) + @(Get-AilaOneApiRootVariableNames) + $runtimeKeys
foreach ($key in $affectedKeys) {
    if (-not $savedEnvironment.ContainsKey($key)) {
        $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    }
}
try {
    Set-AilaProcessEnvironment -Environment $stackEnvironment
    foreach ($key in @([Environment]::GetEnvironmentVariables('Process').Keys |
        Where-Object { $_ -match '^(AILA_|ONEDNN_|DNNL_)' })) {
        [Environment]::SetEnvironmentVariable($key,
            [System.Management.Automation.Language.NullString]::Value, 'Process')
    }
    foreach ($entry in $candidate.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, 'Process')
    }
    Write-Host "Experimental supply: chunk=$PrefillChunk swaps=$HotsetSwaps prefetch(decode/prefill)=$DecodePrefetch/$PrefillPrefetch Ready-adopt=4096 host-pool=32GiB"
    Write-Host "Profile: $profile"
    if ($DryRun) {
        $selectedEnvironment = [ordered]@{}
        foreach ($key in ((@($candidate.Keys) + @('PATH', 'CMPLR_ROOT', 'DNNLROOT', 'TBBROOT', 'UMF_ROOT')) | Sort-Object -Unique)) {
            $selectedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        }
        [ordered]@{ executable = $exe; arguments = $arguments; environment = $selectedEnvironment } |
            ConvertTo-Json -Depth 5
    } else {
        & $exe @arguments
        if ($LASTEXITCODE -ne 0) { throw "Aila exited with code $LASTEXITCODE." }
    }
} finally {
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        $value = if ($null -eq $entry.Value) {
            [System.Management.Automation.Language.NullString]::Value
        } else { [string]$entry.Value }
        [Environment]::SetEnvironmentVariable($entry.Key, $value, 'Process')
    }
}
