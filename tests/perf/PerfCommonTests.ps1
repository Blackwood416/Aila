$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\..\perf\PerfCommon.ps1')

function Assert-Equal($Expected, $Actual, [string]$Message) {
    if ($Expected -ne $Actual) {
        throw "$Message expected='$Expected' actual='$Actual'"
    }
}

$repoRoot = Get-AilaRepoRoot
$config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
$baseline = Get-AilaOneApiStack -Config $config -Name 'oneapi-2025.3'
$candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'

Assert-Equal 'sycl8.dll' $baseline.expectedSyclDll 'baseline SYCL ABI'
Assert-Equal '3.9.1' $baseline.expectedDnnlVersion 'baseline oneDNN'
Assert-Equal 'sycl9.dll' $candidate.expectedSyclDll 'candidate SYCL ABI'
Assert-Equal '3.11.2' $candidate.expectedDnnlVersion 'candidate oneDNN'

Write-Host 'PerfCommonTests PASS'
