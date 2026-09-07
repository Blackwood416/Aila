[CmdletBinding()]
param(
    [Alias('BuildRoot')]
    [string]$DependencyRoot,
    [int]$Parallelism = 4
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $DependencyRoot) {
    $DependencyRoot = Join-Path $repoRoot 'build-refactor-server-deps'
}
elseif (-not [IO.Path]::IsPathRooted($DependencyRoot)) {
    $DependencyRoot = Join-Path $repoRoot $DependencyRoot
}
$DependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
if ($Parallelism -lt 1) { throw 'Parallelism must be positive' }

# The recorded HTTP probe and its patched dependency build remain independent.
$probeRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'build-refactor-http-probe'))
$probePrefix = $probeRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if ($DependencyRoot.TrimEnd('\', '/').Equals($probeRoot.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase) -or
    $DependencyRoot.StartsWith($probePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'DependencyRoot must not be the original HTTP probe directory or one of its descendants'
}
foreach ($directory in @('', 'logs', 'downloads', 'sources', 'prefix', 'patches',
        'registry-cache', 'binary-cache', 'deps-manifest')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DependencyRoot $directory) | Out-Null
}

function Invoke-DependencyCommand {
    param([string]$Program, [string[]]$Arguments, [string]$LogName)
    $log = Join-Path $DependencyRoot "logs/$LogName.txt"
    Write-Host "Running $LogName; log: $log"
    & $Program @Arguments 2>&1 | Out-File -LiteralPath $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $log -Tail 60 | ForEach-Object { Write-Host $_ }
        throw "$Program failed with exit $LASTEXITCODE; see $log"
    }
}

function Get-LfPatch {
    param([string]$Path, [string]$ExpectedHash)
    $patchText = [IO.File]::ReadAllText($Path)
    $patchText = $patchText.Replace([string][char]13 + [char]10, [string][char]10)
    $patchBytes = [Text.Encoding]::UTF8.GetBytes($patchText)
    $digest = [Security.Cryptography.SHA256]::Create()
    try { $actualHash = [BitConverter]::ToString($digest.ComputeHash($patchBytes)).Replace('-', '').ToLowerInvariant() }
    finally { $digest.Dispose() }
    if ($actualHash -ne $ExpectedHash) { throw "Patch SHA256 mismatch: $Path" }
    return @{ Bytes = $patchBytes; Sha256Lf = $actualHash }
}

function Install-DependencyPatch {
    param([string]$Name, [string]$SourceName, [string]$PatchPath, [string]$ExpectedHash)
    $checked = Get-LfPatch $PatchPath $ExpectedHash
    $canonicalPatch = Join-Path $DependencyRoot "patches/$Name.patch"
    [IO.File]::WriteAllBytes($canonicalPatch, $checked.Bytes)
    $sourcePath = Join-Path $DependencyRoot "sources/$SourceName"
    # Keep git apply from resolving relative paths against Aila's parent checkout.
    if (-not (Test-Path -LiteralPath (Join-Path $sourcePath '.git'))) {
        Invoke-DependencyCommand 'git.exe' @('-C', $sourcePath, 'init',
            '--initial-branch=codex/server-dependency') "init-$SourceName-patch-root"
    }
    & git.exe -C $sourcePath apply --reverse --check $canonicalPatch 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Invoke-DependencyCommand 'git.exe' @('-C', $sourcePath, 'apply', '--check',
            $canonicalPatch) "check-$Name-patch"
        Invoke-DependencyCommand 'git.exe' @('-C', $sourcePath, 'apply',
            $canonicalPatch) "apply-$Name-patch"
    }
    Invoke-DependencyCommand 'git.exe' @('-C', $sourcePath, 'apply', '--reverse', '--check',
        '--verbose', $canonicalPatch) "verify-$Name-patch"
    return [ordered]@{
        name = $Name
        source = $SourceName
        sha256Lf = $checked.Sha256Lf
        path = $canonicalPatch
    }
}

# Import ordinary MSVC without changing the caller's persistent environment.
$programFilesX86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
$vswhere = Join-Path $programFilesX86 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio vswhere.exe is required' }
$vsArguments = @('-latest', '-products', '*', '-requires',
    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
$vsRoot = (& $vswhere @vsArguments | Select-Object -First 1)
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
try {
    $environmentLines = $environmentProcess.StandardOutput.ReadToEnd() -split '\r?\n'
    $environmentError = $environmentProcess.StandardError.ReadToEnd()
    $environmentProcess.WaitForExit()
    if ($environmentProcess.ExitCode -ne 0) { throw "Unable to import the MSVC environment: $environmentError" }
}
finally { $environmentProcess.Dispose() }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:VSLANG = '1033'
$vcpkg = Join-Path $vsRoot 'VC/vcpkg/vcpkg.exe'
if (-not (Test-Path -LiteralPath $vcpkg)) { throw 'The Visual Studio vcpkg distribution is required' }
$env:X_VCPKG_REGISTRIES_CACHE = Join-Path $DependencyRoot 'registry-cache'
$env:VCPKG_DOWNLOADS = Join-Path $DependencyRoot 'downloads'
$env:VCPKG_BINARY_SOURCES = "clear;files,$DependencyRoot\binary-cache,readwrite"
$env:VCPKG_MAX_CONCURRENCY = "$Parallelism"

$baseline = '544a4c5c297e60e4ac4a5a1810df66748d908869'
$triplet = 'x64-windows-static-md'
$installed = Join-Path $DependencyRoot 'vcpkg_installed'
$prefix = Join-Path $DependencyRoot 'prefix'
$vcpkgPrefix = Join-Path $installed $triplet
$dependencyPrefix = "$prefix;$vcpkgPrefix"

# vcpkg's Drogon port unconditionally adds TLS/DNS/compression dependencies.
# Only JsonCpp and zlib use vcpkg; the pinned framework sources are built below.
$manifest = [ordered]@{
    name = 'aila-server-dependencies'
    'version-string' = '0.1.0'
    'builtin-baseline' = $baseline
    dependencies = @('jsoncpp', 'zlib')
}
$manifest | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $DependencyRoot 'deps-manifest/vcpkg.json') -Encoding utf8
$vcpkgArguments = @('--disable-metrics', 'install', '--triplet', $triplet,
    "--x-install-root=$installed", "--x-buildtrees-root=$DependencyRoot/buildtrees",
    "--x-packages-root=$DependencyRoot/packages", "--x-manifest-root=$DependencyRoot/deps-manifest")
Invoke-DependencyCommand $vcpkg ($vcpkgArguments + @('--dry-run')) 'resolve-minimal'
Invoke-DependencyCommand $vcpkg $vcpkgArguments 'install-minimal'

$sources = @(
    [ordered]@{
        name = 'trantor'; version = '1.5.24'
        sha512 = 'a875e5bf8c8d871d1a3ddeeb79891773eed4f13772aa8c09ce6bdd60fa7af621eed6986935c0b70b58e790618934ca82a7ebd4ba5faf2b534b8ecb6f74d6bad1'
    },
    [ordered]@{
        name = 'drogon'; version = '1.9.11'
        sha512 = 'ce61c65281bde0246e96c3558cf635241adbcab514d6a71e8267201340f577af26b1cbc69fa48652d655971122a66c70db75d1a4c034384ba0716fccf7ad9584'
    }
)
foreach ($source in $sources) {
    $archive = Join-Path $DependencyRoot "downloads/$($source.name)-$($source.version).tar.gz"
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-DependencyCommand 'curl.exe' @('--fail', '--location', '--retry', '2', '--output',
            $archive, "https://github.com/an-tao/$($source.name)/archive/v$($source.version).tar.gz") "download-$($source.name)"
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA512).Hash.ToLowerInvariant()
    if ($actualHash -ne $source.sha512) { throw "Source archive SHA512 mismatch: $archive" }
    $sourcePath = Join-Path $DependencyRoot "sources/$($source.name)"
    if (-not (Test-Path -LiteralPath (Join-Path $sourcePath 'CMakeLists.txt'))) {
        New-Item -ItemType Directory -Force -Path $sourcePath | Out-Null
        Invoke-DependencyCommand 'tar.exe' @('-xzf', $archive, '-C', $sourcePath,
            '--strip-components=1') "extract-$($source.name)"
    }
}
$sources | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $DependencyRoot 'logs/source-lock.json') -Encoding utf8
$trantorPatchPath = Join-Path $PSScriptRoot 'trantor-pending-bytes.patch'
$listenerPatchPath = Join-Path $PSScriptRoot '../http-probe/drogon-windows-listener-callbacks.patch'
$patches = @(
    Install-DependencyPatch 'trantor-pending-bytes' 'trantor' $trantorPatchPath 'c1c359ef7cf9dc5ac776c7efa0339805ec90f88af6aa7bd07a585746f06cc5fc'
    Install-DependencyPatch 'drogon-windows-listener-callbacks' 'drogon' $listenerPatchPath 'f267e88c29eaa99bb8b138fae3f8224d4843a0ef9dc393406dbd7c43498cb179'
)
$patches | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $DependencyRoot 'logs/applied-patches.json') -Encoding utf8

$configureCommon = @('-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_C_COMPILER=cl.exe', '-DCMAKE_CXX_COMPILER=cl.exe', '-DCMAKE_CXX_STANDARD=17',
    '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
    "-DCMAKE_INSTALL_PREFIX=$prefix", "-DCMAKE_PREFIX_PATH=$dependencyPrefix",
    '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF', '-DBUILD_DOC=OFF',
    '-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON', '-DCMAKE_DISABLE_FIND_PACKAGE_Botan=ON',
    '-DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON')
Invoke-DependencyCommand 'cmake.exe' (@('-S', "$DependencyRoot/sources/trantor",
    '-B', "$DependencyRoot/build-trantor") + $configureCommon +
    @('-DBUILD_C-ARES=OFF', '-DTRANTOR_USE_TLS=')) 'configure-trantor'
Invoke-DependencyCommand 'cmake.exe' @('--build', "$DependencyRoot/build-trantor",
    '--parallel', "$Parallelism") 'build-trantor'
Invoke-DependencyCommand 'cmake.exe' @('--install', "$DependencyRoot/build-trantor") 'install-trantor'
Invoke-DependencyCommand 'cmake.exe' (@('-S', "$DependencyRoot/sources/drogon",
    '-B', "$DependencyRoot/build-drogon") + $configureCommon +
    @('-DUSE_SUBMODULE=OFF', '-DBUILD_CTL=OFF', '-DBUILD_EXAMPLES=OFF', '-DBUILD_ORM=OFF',
        '-DBUILD_BROTLI=OFF', '-DBUILD_YAML_CONFIG=OFF', '-DUSE_COROUTINE=OFF')) 'configure-drogon'
Invoke-DependencyCommand 'cmake.exe' @('--build', "$DependencyRoot/build-drogon",
    '--parallel', "$Parallelism") 'build-drogon'
Invoke-DependencyCommand 'cmake.exe' @('--install', "$DependencyRoot/build-drogon") 'install-drogon'

$installedArtifacts = @(
    'include/trantor/net/TcpConnection.h',
    'lib/trantor.lib',
    'lib/drogon.lib'
)
$artifacts = foreach ($relativePath in $installedArtifacts) {
    $artifact = Join-Path $prefix $relativePath
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) { throw "Missing installed artifact: $artifact" }
    [ordered]@{ path = $artifact; sha256 = (Get-FileHash -LiteralPath $artifact -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$connectionHeader = [IO.File]::ReadAllText((Join-Path $prefix 'include/trantor/net/TcpConnection.h'))
if (-not $connectionHeader.Contains('virtual size_t pendingWriteBytes() const = 0;')) {
    throw 'The installed Trantor header does not expose pendingWriteBytes()'
}

# Written only after both installs succeed. Server build.ps1 can reject an old
# probe prefix, missing extension, changed patch, or changed installed artifact.
$lock = [ordered]@{
    schemaVersion = 1
    purpose = 'aila-server-dependencies'
    pendingWriteBytesApi = 1
    compiler = 'MSVC'
    cxxStandard = 17
    msvcRuntime = 'MultiThreadedDLL'
    tls = $false
    triplet = $triplet
    vcpkgBaseline = $baseline
    prefix = $prefix
    vcpkgPrefix = $vcpkgPrefix
    sources = $sources
    packages = @(
        [ordered]@{ name = 'jsoncpp'; version = '1.9.6' },
        [ordered]@{ name = 'zlib'; version = '1.3.1' }
    )
    patches = $patches
    artifacts = @($artifacts)
}
$lockPath = Join-Path $DependencyRoot 'dependency-lock.json'
$lock | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $lockPath -Encoding utf8
Write-Host "Server dependencies ready; lock: $lockPath"
Write-Host "CMAKE_PREFIX_PATH=$dependencyPrefix"
