# Builds probe_qsa_prompt_xmx.exe. This script never launches the probe.
param(
    [string]$Output = '',
    [string]$BuildDir = 'build-refactor',
    [switch]$CompileOnly,
    [switch]$Aot
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $repo 'perf/PerfCommon.ps1')
$config = Get-AilaOneApiStackConfig -RepoRoot $repo -Path 'perf/oneapi-stacks.json'
$stack = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $stack)

$source = (Resolve-Path (Join-Path $PSScriptRoot 'probe_qsa_prompt_xmx.cpp')).Path
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot 'probe_qsa_prompt_xmx.exe'
}
$objDir = Join-Path $PSScriptRoot 'obj'
New-Item -ItemType Directory -Force -Path $objDir | Out-Null
$obj = Join-Path $objDir 'probe_qsa_prompt_xmx.obj'

$cxx = Join-Path $stack.compilerRoot 'bin/icx-cl.exe'
$build = Join-Path $repo $BuildDir
$lib = Join-Path $build 'AilaLib.lib'
$l0 = 'C:\Program Files\LevelZeroSDK\1.28.2'
if (-not (Test-Path -LiteralPath $lib)) {
    throw "Missing $lib; build the primary runtime first"
}

$includes = @('include', 'src/utils', 'src/core', 'src/ops', 'src/memory',
              'src/models', 'src', 'third_party/simdjson') |
    ForEach-Object { '-I' + (Join-Path $repo $_) }
$ext = @("-external:I$($stack.dnnlRoot)/include",
         "-external:I$($stack.tbbRoot)/include", '-external:W0')
$ext += "/I$l0\include"
$flags = @('/nologo', '/TP', '-DAILA_HAS_LEVEL_ZERO=1',
           '-DAILA_STATIC_LIB', '-DNOMINMAX',
           '-DWIN32_LEAN_AND_MEAN', '-D__TBB_NO_IMPLICIT_LINKAGE=1',
           '/DWIN32', '/D_WINDOWS', '/EHsc', '-O3', '-ffast-math',
           '/Qstd=c++17', '/Qftz', '/O2', '/Ob2', '/DNDEBUG', '-MD',
           '-fsycl', '-fsycl-device-code-split=per_kernel',
           '-fsycl-default-sub-group-size=16')
$deviceTargets = @()
if ($Aot) {
    # A770/DG2 AOT device link forces IGC/ocloc to lower the probe kernel.
    # This catches device-only compiler failures without touching the GPU.
    if (-not (Get-Command ocloc.exe -ErrorAction SilentlyContinue)) {
        $oclocDir = 'C:\Program Files (x86)\Intel\oneAPI\ocloc\2026.1\bin'
        if (-not (Test-Path -LiteralPath (Join-Path $oclocDir 'ocloc.exe'))) {
            throw "ocloc.exe is required for -Aot but was not found"
        }
        $env:PATH = "$oclocDir;$env:PATH"
    }
    $deviceTargets = @('-fsycl-targets=intel_gpu_acm_g10')
}

& $cxx @flags @deviceTargets @includes @ext "/Fo$obj" -c $source
if ($LASTEXITCODE -ne 0) {
    throw "compile failed ($LASTEXITCODE)"
}
if ($CompileOnly) {
    Write-Host "compiled $obj (not linked, not run)"
    exit 0
}

$libs = @($lib,
          "$($stack.dnnlRoot)/lib/dnnl.lib",
          "$($stack.tbbRoot)/lib/tbb12.lib",
          (Join-Path $build 'simdjson.lib'),
          (Join-Path $build 'soxr.lib'),
          "$l0\lib\ze_loader.lib",
          'windowscodecs.lib', 'ole32.lib', 'kernel32.lib', 'user32.lib',
          'gdi32.lib', 'shell32.lib', 'oleaut32.lib', 'uuid.lib',
          'advapi32.lib')
& $cxx /nologo -fsycl @deviceTargets $obj @libs "/Fe$Output" `
    '/Qoption,link,/machine:x64' '/Qoption,link,/INCREMENTAL:NO' `
    '/Qoption,link,/subsystem:console'
if ($LASTEXITCODE -ne 0) {
    throw "link failed ($LASTEXITCODE)"
}

Write-Host "built $Output (not run)"
