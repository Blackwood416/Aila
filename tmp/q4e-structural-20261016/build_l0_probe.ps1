param(
    [string]$Source = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (-not $Source) { $Source = Join-Path $PSScriptRoot 'probe_l0_queue_groups.cpp' }
if (-not $Output) { $Output = [IO.Path]::ChangeExtension($Source, '.exe') }
. (Join-Path $repo 'perf/PerfCommon.ps1')
$config = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)
$cxx = Join-Path $stack.compilerRoot 'bin/icx-cl.exe'
$l0 = 'C:\Program Files\LevelZeroSDK\1.28.2'
& $cxx /nologo /TP /EHsc /O2 /Qstd=c++17 /MD "/I$l0\include" $Source "/Fe$Output" /link "$l0\lib\ze_loader.lib"
if ($LASTEXITCODE -ne 0) { throw "compile/link failed ($LASTEXITCODE)" }
Write-Host "built $Output"
