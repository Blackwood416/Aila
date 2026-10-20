# Builds the standalone host-ring visibility probe.
# It does not link AilaLib and never runs the executable.
param(
    [string]$Output = '',
    [string]$LevelZeroRoot = 'C:\Program Files\LevelZeroSDK\1.28.2',
    [switch]$CompileOnly
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $repo 'perf/PerfCommon.ps1')
$config = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

$source = (Resolve-Path (Join-Path $PSScriptRoot 'probe_host_ring.cpp')).Path
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot 'probe_host_ring.exe'
}
$objDir = Join-Path $PSScriptRoot 'obj-host-ring'
New-Item -ItemType Directory -Force -Path $objDir | Out-Null
$obj = Join-Path $objDir 'probe_host_ring.obj'

if (-not (Test-Path -LiteralPath (Join-Path $LevelZeroRoot 'include/level_zero/ze_api.h'))) {
    throw "Missing Level Zero headers under $LevelZeroRoot"
}

$compiler = Join-Path $stack.compilerRoot 'bin/icx-cl.exe'
$flags = @(
    '/nologo', '/TP', '-DNOMINMAX', '-DWIN32_LEAN_AND_MEAN',
    '/DWIN32', '/D_WINDOWS', '/EHsc', '/Qstd=c++17',
    '/O2', '/Ob2', '/DNDEBUG', '-MD',
    '-fsycl', '-fsycl-device-code-split=per_kernel',
    '-fsycl-default-sub-group-size=16',
    "/I$LevelZeroRoot/include"
)

& $compiler @flags "/Fo$obj" -c $source
if ($LASTEXITCODE -ne 0) {
    throw "compile failed ($LASTEXITCODE)"
}
if ($CompileOnly) {
    Write-Host "compiled $obj (not linked, not run)"
    exit 0
}

$loader = Join-Path $LevelZeroRoot 'lib/ze_loader.lib'
if (-not (Test-Path -LiteralPath $loader)) {
    throw "Missing Level Zero loader: $loader"
}
$libs = @(
    $loader,
    'kernel32.lib', 'user32.lib', 'advapi32.lib'
)
& $compiler /nologo -fsycl $obj @libs "/Fe$Output" `
    '/Qoption,link,/machine:x64' '/Qoption,link,/INCREMENTAL:NO' `
    '/Qoption,link,/subsystem:console'
if ($LASTEXITCODE -ne 0) {
    throw "link failed ($LASTEXITCODE)"
}

Write-Host "built $Output (not run)"
