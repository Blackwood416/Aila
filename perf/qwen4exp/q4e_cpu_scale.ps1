param(
    [ValidateSet('gpu8', 'gpu4', 'miss_seq10', 'miss_batch10', 'host_batch10',
                 'force_batch5', 'force_batch10', 'force_batch40', 'avx_batch5', 'avx_batch10',
                 'fast_miss10', 'fast_host10', 'fast_force5', 'fast_force10', 'fast_small40', 'slow_small40',
                 'data_base', 'data_fill', 'data_copy', 'data_both', 'data_miss_copy', 'data_wait')]
    [string[]]$Cases = @('gpu8', 'gpu4', 'miss_seq10', 'miss_batch10', 'host_batch10',
                       'force_batch5', 'force_batch10', 'force_batch40', 'avx_batch5', 'avx_batch10'),
    [string]$Tag = 'cpu_scale12',
    [string]$BaselineLog = 'tmp/h6-baseline.log',
    [switch]$PreparedScreen,
    [switch]$ScalePair,
    [switch]$ReverseOrder,
    [switch]$RealAcceptance,
    [switch]$DataflowScreen,
    [switch]$DataflowPair,
    [switch]$NoPrefetch,
    [switch]$NoPrefetchScreen,
    [ValidateSet('data_both', 'data_copy', 'data_wait')]
    [string]$DataflowCandidate = 'data_both',
    [switch]$DataflowWaitScreen
)
$ErrorActionPreference = 'Stop'
if ($PreparedScreen) {
    $Cases = @('gpu8', 'gpu4', 'miss_batch10', 'fast_miss10', 'fast_host10',
               'fast_force5', 'fast_force10', 'slow_small40', 'fast_small40')
}
if ($ScalePair) {
    $Cases = if ($ReverseOrder) { @('fast_small40', 'slow_small40') } else { @('slow_small40', 'fast_small40') }
}
if ($DataflowScreen) { $Cases = @('gpu8', 'data_base', 'data_fill', 'data_copy', 'data_both', 'fast_miss10', 'data_miss_copy') }
if ($DataflowPair) { $Cases = if ($ReverseOrder) { @($DataflowCandidate, 'data_base') } else { @('data_base', $DataflowCandidate) } }
if ($NoPrefetchScreen) { $Cases = @('gpu8', 'data_base', 'data_fill', 'data_copy', 'data_both'); $NoPrefetch = $true }
if ($DataflowWaitScreen) { $Cases = @('gpu8', 'data_base', 'data_wait') }
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
Push-Location $repo
try {
    foreach ($case in $Cases) {
        $vars = @{
            AILA_Q4E_EXPERT_READER = 'readfile'; AILA_DEBUG_TOKEN_IDS = '1'
            AILA_Q4E_FILL_THREADS = '4'; AILA_Q4E_CPU_THREADS = '12'
            AILA_Q4E_CPU_EXPERT_DISPATCH = '1'; AILA_Q4E_CPU_GPU_ORDER = '1'
            AILA_Q4E_CPU_OVERLAP = '1'; AILA_Q4E_CPU_BATCH = '1'
            AILA_Q4E_CPU_FIXED_CREDITS = '1'; AILA_Q4E_CPU_MAX_EXPERTS = '10'
            AILA_Q4E_CPU_PLACEMENT = '0'
            AILA_Q4E_CPU_PREPARED_INPUT = '0'; AILA_Q4E_CPU_SMALL_M = '0'
            AILA_Q4E_CPU_BATCH_PREPARE = '0'; AILA_Q4E_CPU_COALESCE_RESULTS = '0'
            AILA_Q4E_CPU_WAIT_PREFETCH = '0'
        }
        if ($case -like 'data_*') {
            $vars.AILA_Q4E_CPU_PREPARED_INPUT = '1'
            if ($case -eq 'data_miss_copy') { $vars.AILA_Q4E_CPU_COALESCE_RESULTS = '1' }
            else {
                $vars.AILA_Q4E_CPU_SMALL_M = '1'; $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_MAX_EXPERTS = '40'
                if ($case -in @('data_fill', 'data_both', 'data_wait')) { $vars.AILA_Q4E_CPU_BATCH_PREPARE = '1' }
                if ($case -in @('data_copy', 'data_both', 'data_wait')) { $vars.AILA_Q4E_CPU_COALESCE_RESULTS = '1' }
                if ($case -eq 'data_wait') { $vars.AILA_Q4E_CPU_WAIT_PREFETCH = '1' }
            }
        }
        switch ($case) {
            'gpu8' { $vars.AILA_Q4E_CPU_EXPERT_DISPATCH = '0'; $vars.AILA_Q4E_FILL_THREADS = '8' }
            'gpu4' { $vars.AILA_Q4E_CPU_EXPERT_DISPATCH = '0' }
            'miss_seq10' { $vars.AILA_Q4E_CPU_BATCH = '0' }
            'host_batch10' { $vars.AILA_Q4E_CPU_PLACEMENT = '1' }
            'force_batch5' { $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_MAX_EXPERTS = '5' }
            'force_batch10' { $vars.AILA_Q4E_CPU_PLACEMENT = '2' }
            'force_batch40' { $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_MAX_EXPERTS = '40' }
            'avx_batch5' { $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_MAX_EXPERTS = '5'; $vars.AILA_Q4E_CPU_GPU_ORDER = '0' }
            'avx_batch10' { $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_GPU_ORDER = '0' }
            'fast_miss10' { $vars.AILA_Q4E_CPU_PREPARED_INPUT = '1' }
            'fast_host10' { $vars.AILA_Q4E_CPU_PREPARED_INPUT = '1'; $vars.AILA_Q4E_CPU_PLACEMENT = '1' }
            'fast_force5' { $vars.AILA_Q4E_CPU_PREPARED_INPUT = '1'; $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_MAX_EXPERTS = '5' }
            'fast_force10' { $vars.AILA_Q4E_CPU_PREPARED_INPUT = '1'; $vars.AILA_Q4E_CPU_PLACEMENT = '2' }
            'fast_small40' { $vars.AILA_Q4E_CPU_PREPARED_INPUT = '1'; $vars.AILA_Q4E_CPU_SMALL_M = '1'; $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_MAX_EXPERTS = '40' }
            'slow_small40' { $vars.AILA_Q4E_CPU_SMALL_M = '1'; $vars.AILA_Q4E_CPU_PLACEMENT = '2'; $vars.AILA_Q4E_CPU_MAX_EXPERTS = '40' }
        }
        if ($NoPrefetch) { $vars.AILA_Q4E_PREFETCH = '0' }
        $run = $Tag + '_' + $case
        & "$PSScriptRoot/q4e_prof.ps1" -Tag $run -EnvVars $vars
        $log = "tmp/q4e_prof_$run.log"
        if (!(Select-String -Path $log -Pattern '\[GenerateMessages\] Prompt=\d+ Generated=48 in ' -Quiet)) {
            throw "Incomplete generation: $log"
        }
        & pwsh -NoProfile -File perf/compare-token-traces.ps1 -BaselineLog $BaselineLog -CandidateLog $log -OutputPath "tmp/q4e_prof_$run.tokens.json"
        $exact = $LASTEXITCODE -eq 0
        Write-Output "[$run] golden_exact=$exact (mismatches are feasibility data, not valid speedups)"
        foreach ($pattern in @('\[GenerateMessages\] Prompt=', '\[Q4E ReadyFirst\]', '\[Q4E CPU Overlap\]', '\[Q4E CPU Dataflow\]')) {
            Select-String -Path $log -Pattern $pattern | Select-Object -Last 1 | ForEach-Object { $_.Line }
        }
        if ($RealAcceptance) {
            foreach ($key in $vars.Keys) { Set-Item -Path "env:$key" -Value $vars[$key] }
            try {
                & ./build-refactor/AilaQwen4ExpGenerationTests.exe --model 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2' *> "tmp/q4e_prof_$run.real.log"
                if ($LASTEXITCODE -ne 0) { throw "Real-model acceptance failed: tmp/q4e_prof_$run.real.log" }
                Write-Output "[$run] Real-model acceptance passed"
            } finally {
                foreach ($key in $vars.Keys) { Remove-Item -Path "env:$key" -ErrorAction SilentlyContinue }
            }
        }
    }
} finally { Pop-Location }
