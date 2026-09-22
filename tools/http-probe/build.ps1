[CmdletBinding()]
param(
    [string]$BuildRoot,
    [int]$Parallelism = 4,
    [switch]$DependenciesOnly,
    [switch]$ProbeOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $BuildRoot) { $BuildRoot = Join-Path $repoRoot 'build-refactor-http-probe' }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
if ($Parallelism -lt 1) { throw 'Parallelism must be positive' }
if ($DependenciesOnly -and $ProbeOnly) { throw 'DependenciesOnly and ProbeOnly are mutually exclusive' }
foreach ($directory in @('', 'logs', 'downloads', 'sources', 'prefix', 'registry-cache', 'binary-cache', 'resolve-drogon', 'deps-manifest')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $BuildRoot $directory) | Out-Null
}

function Invoke-ProbeCommand {
    param([string]$Program, [string[]]$Arguments, [string]$LogName)
    $log = Join-Path $BuildRoot "logs/$LogName.txt"
    Write-Output "Running $LogName; log: $log"
    & $Program @Arguments 2>&1 | Out-File -LiteralPath $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $log -Tail 60 | Write-Output
        throw "$Program failed with exit $LASTEXITCODE; see $log"
    }
}

# Ordinary MSVC environment; no oneAPI compiler or GPU runtime is needed.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio vswhere.exe is required' }
$vsRoot = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
if (-not $vsRoot) { throw 'A Visual Studio C++ installation is required' }
$devCommand = '""{0}\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 && set"' -f $vsRoot
$environmentStart = [Diagnostics.ProcessStartInfo]::new()
$environmentStart.FileName = $env:ComSpec
$environmentStart.Arguments = '/d /s /c ' + $devCommand
$environmentStart.UseShellExecute = $false
$environmentStart.CreateNoWindow = $true
$environmentStart.RedirectStandardOutput = $true
$environmentStart.RedirectStandardError = $true
$environmentProcess = [Diagnostics.Process]::Start($environmentStart)
$environmentLines = $environmentProcess.StandardOutput.ReadToEnd() -split "`r?`n"
$environmentError = $environmentProcess.StandardError.ReadToEnd()
$environmentProcess.WaitForExit()
if ($environmentProcess.ExitCode -ne 0) { throw "Unable to import the MSVC environment: $environmentError" }
$environmentProcess.Dispose()
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:VSLANG = '1033'
$vcpkg = Join-Path $vsRoot 'VC/vcpkg/vcpkg.exe'
if (-not (Test-Path -LiteralPath $vcpkg)) { throw 'The Visual Studio vcpkg distribution is required' }
$env:X_VCPKG_REGISTRIES_CACHE = Join-Path $BuildRoot 'registry-cache'
$env:VCPKG_DOWNLOADS = Join-Path $BuildRoot 'downloads'
$env:VCPKG_BINARY_SOURCES = "clear;files,$BuildRoot\binary-cache,readwrite"
$env:VCPKG_MAX_CONCURRENCY = "$Parallelism"
$baseline = '544a4c5c297e60e4ac4a5a1810df66748d908869'
$triplet = 'x64-windows-static-md'
$installed = Join-Path $BuildRoot 'vcpkg_installed'
$prefix = Join-Path $BuildRoot 'prefix'
$dependencyPrefix = "$prefix;$installed/$triplet"

# Keep the unmodified port's real dependency cost as evidence before choosing
# upstream's supported no-TLS/no-DNS configuration. No overlay port is used.
if (-not $ProbeOnly) {
$fullManifest = @{ name = 'aila-http-port-plan'; 'version-string' = '0.1.0'; 'builtin-baseline' = $baseline; dependencies = @(@{ name = 'drogon'; 'default-features' = $false }) }
$fullManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $BuildRoot 'resolve-drogon/vcpkg.json') -Encoding utf8
$vcpkgCommon = @('--disable-metrics', 'install', '--triplet', $triplet, "--x-install-root=$installed", "--x-buildtrees-root=$BuildRoot/buildtrees", "--x-packages-root=$BuildRoot/packages")
Invoke-ProbeCommand $vcpkg ($vcpkgCommon + @('--dry-run', "--x-manifest-root=$BuildRoot/resolve-drogon")) 'resolve-drogon'
$minimalManifest = @{ name = 'aila-http-dependencies'; 'version-string' = '0.1.0'; 'builtin-baseline' = $baseline; dependencies = @('jsoncpp', 'zlib') }
$minimalManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $BuildRoot 'deps-manifest/vcpkg.json') -Encoding utf8
Invoke-ProbeCommand $vcpkg ($vcpkgCommon + @('--dry-run', "--x-manifest-root=$BuildRoot/deps-manifest")) 'resolve-minimal'
Invoke-ProbeCommand $vcpkg ($vcpkgCommon + @("--x-manifest-root=$BuildRoot/deps-manifest")) 'install-minimal'

$sources = @(
    @{ Name = 'trantor'; Version = '1.5.24'; Sha512 = 'a875e5bf8c8d871d1a3ddeeb79891773eed4f13772aa8c09ce6bdd60fa7af621eed6986935c0b70b58e790618934ca82a7ebd4ba5faf2b534b8ecb6f74d6bad1' },
    @{ Name = 'drogon'; Version = '1.9.11'; Sha512 = 'ce61c65281bde0246e96c3558cf635241adbcab514d6a71e8267201340f577af26b1cbc69fa48652d655971122a66c70db75d1a4c034384ba0716fccf7ad9584' }
)
foreach ($source in $sources) {
    $archive = Join-Path $BuildRoot "downloads/$($source.Name)-$($source.Version).tar.gz"
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-ProbeCommand 'curl.exe' @('--fail', '--location', '--retry', '2', '--output', $archive, "https://github.com/an-tao/$($source.Name)/archive/v$($source.Version).tar.gz") "download-$($source.Name)"
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA512).Hash.ToLowerInvariant()
    if ($actualHash -ne $source.Sha512) { throw "SHA512 mismatch for $archive" }
    $sourcePath = Join-Path $BuildRoot "sources/$($source.Name)"
    if (-not (Test-Path -LiteralPath (Join-Path $sourcePath 'CMakeLists.txt'))) {
        New-Item -ItemType Directory -Force -Path $sourcePath | Out-Null
        Invoke-ProbeCommand 'tar.exe' @('-xzf', $archive, '-C', $sourcePath, '--strip-components=1') "extract-$($source.Name)"
    }
}
$sources | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $BuildRoot 'logs/source-lock.json') -Encoding utf8
$patchText = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'drogon-windows-listener-callbacks.patch')).Replace("`r`n", "`n")
$patchBytes = [Text.Encoding]::UTF8.GetBytes($patchText)
$patchDigest = [Security.Cryptography.SHA256]::Create()
try { $patchHash = [BitConverter]::ToString($patchDigest.ComputeHash($patchBytes)).Replace('-', '').ToLowerInvariant() }
finally { $patchDigest.Dispose() }
if ($patchHash -ne 'f267e88c29eaa99bb8b138fae3f8224d4843a0ef9dc393406dbd7c43498cb179') { throw 'Windows listener patch SHA256 mismatch' }
$canonicalPatch = Join-Path $BuildRoot 'drogon-windows-listener-callbacks.patch'
[IO.File]::WriteAllBytes($canonicalPatch, $patchBytes)
# A source archive nested in Aila's checkout otherwise inherits the parent
# repository prefix, and git apply can silently skip the relative patch paths.
if (-not (Test-Path -LiteralPath "$BuildRoot/sources/drogon/.git")) {
    Invoke-ProbeCommand 'git.exe' @('-C', "$BuildRoot/sources/drogon", 'init', '--initial-branch=codex/http-probe-dependency') 'init-drogon-patch-root'
}
& git.exe -C "$BuildRoot/sources/drogon" apply --reverse --check $canonicalPatch 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-ProbeCommand 'git.exe' @('-C', "$BuildRoot/sources/drogon", 'apply', '--check', $canonicalPatch) 'check-windows-listener-patch'
    Invoke-ProbeCommand 'git.exe' @('-C', "$BuildRoot/sources/drogon", 'apply', $canonicalPatch) 'apply-windows-listener-patch'
}
Invoke-ProbeCommand 'git.exe' @('-C', "$BuildRoot/sources/drogon", 'apply', '--reverse', '--check', '--verbose', $canonicalPatch) 'verify-windows-listener-patch'
$patchedFileHash = (Get-FileHash -LiteralPath "$BuildRoot/sources/drogon/lib/src/ListenerManager.cc" -Algorithm SHA256).Hash.ToLowerInvariant()
@{ source = 'Drogon 1.9.11'; patch_sha256_lf = $patchHash; patched_listener_sha256 = $patchedFileHash; patch = $canonicalPatch } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $BuildRoot 'logs/applied-patch.json') -Encoding utf8
$configureCommon = @('-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl.exe', '-DCMAKE_CXX_COMPILER=cl.exe', '-DCMAKE_CXX_STANDARD=17', '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5', "-DCMAKE_INSTALL_PREFIX=$prefix", "-DCMAKE_PREFIX_PATH=$dependencyPrefix", '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF', '-DBUILD_DOC=OFF', '-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON', '-DCMAKE_DISABLE_FIND_PACKAGE_Botan=ON', '-DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON')
Invoke-ProbeCommand 'cmake.exe' (@('-S', "$BuildRoot/sources/trantor", '-B', "$BuildRoot/build-trantor") + $configureCommon + @('-DBUILD_C-ARES=OFF', '-DTRANTOR_USE_TLS=')) 'configure-trantor'
Invoke-ProbeCommand 'cmake.exe' @('--build', "$BuildRoot/build-trantor", '--parallel', "$Parallelism") 'build-trantor'
Invoke-ProbeCommand 'cmake.exe' @('--install', "$BuildRoot/build-trantor") 'install-trantor'
Invoke-ProbeCommand 'cmake.exe' (@('-S', "$BuildRoot/sources/drogon", '-B', "$BuildRoot/build-drogon") + $configureCommon + @('-DUSE_SUBMODULE=OFF', '-DBUILD_CTL=OFF', '-DBUILD_EXAMPLES=OFF', '-DBUILD_ORM=OFF', '-DBUILD_BROTLI=OFF', '-DBUILD_YAML_CONFIG=OFF', '-DUSE_COROUTINE=OFF')) 'configure-drogon'
Invoke-ProbeCommand 'cmake.exe' @('--build', "$BuildRoot/build-drogon", '--parallel', "$Parallelism") 'build-drogon'
Invoke-ProbeCommand 'cmake.exe' @('--install', "$BuildRoot/build-drogon") 'install-drogon'
}
if ($DependenciesOnly) { return }
Invoke-ProbeCommand 'cmake.exe' @('-S', $PSScriptRoot, '-B', "$BuildRoot/build-probe", '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=cl.exe', "-DCMAKE_PREFIX_PATH=$dependencyPrefix", '-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON', '-DCMAKE_DISABLE_FIND_PACKAGE_Botan=ON') 'configure-probe'
Invoke-ProbeCommand 'cmake.exe' @('--build', "$BuildRoot/build-probe", '--parallel', "$Parallelism") 'build-probe'
Write-Output "Probe executable: $BuildRoot/build-probe/AilaHttpProbe.exe"
