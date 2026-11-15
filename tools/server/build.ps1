[CmdletBinding()]
param(
    [string]$WorkerBuildDir = 'build-refactor',
    [string]$BuildRoot = 'build-refactor-server',
    [string]$DependencyRoot = 'build-refactor-server-deps',
    [int]$Parallelism = 4,
    [switch]$SkipDependencies
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if ($Parallelism -lt 1) { throw 'Parallelism must be positive' }

function Resolve-AilaBuildPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -match '[";\r\n]') {
        throw 'Build paths must be nonempty and cannot contain quotes, semicolons or line breaks'
    }
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repoRoot $Path }
    return [IO.Path]::GetFullPath($Path)
}

function Test-AilaPathContains {
    param([string]$Parent, [string]$Child)
    $normalizedParent = $Parent.TrimEnd('\', '/')
    $normalizedChild = $Child.TrimEnd('\', '/')
    return $normalizedChild.Equals($normalizedParent, [StringComparison]::OrdinalIgnoreCase) -or
        $normalizedChild.StartsWith($normalizedParent + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)
}

$WorkerBuildDir = Resolve-AilaBuildPath $WorkerBuildDir
$BuildRoot = Resolve-AilaBuildPath $BuildRoot
$DependencyRoot = Resolve-AilaBuildPath $DependencyRoot
foreach ($outputRoot in @($BuildRoot, $DependencyRoot)) {
    if (Test-AilaPathContains $outputRoot $repoRoot) {
        throw 'Server output directories cannot be the repository root or one of its ancestors'
    }
    foreach ($protectedName in @('src', 'include', 'tests', 'tools', 'docs', 'third_party',
            '.git', '.Codex', 'build', 'build-refactor-http-probe')) {
        if (Test-AilaPathContains (Join-Path $repoRoot $protectedName) $outputRoot) {
            throw "Server output directories must be independent of $protectedName"
        }
    }
}
$isolatedRoots = @($WorkerBuildDir, $BuildRoot, $DependencyRoot)
for ($left = 0; $left -lt $isolatedRoots.Count; ++$left) {
    for ($right = $left + 1; $right -lt $isolatedRoots.Count; ++$right) {
        if ((Test-AilaPathContains $isolatedRoots[$left] $isolatedRoots[$right]) -or
            (Test-AilaPathContains $isolatedRoots[$right] $isolatedRoots[$left])) {
            throw 'WorkerBuildDir, BuildRoot and DependencyRoot must be separate directories'
        }
    }
}

# The Worker may come from another checkout. Its metadata, rather than this
# checkout's HEAD, is the authority for the exact identity required by hello.
$workerInfoPath = Join-Path $WorkerBuildDir 'build_info.json'
$workerExecutable = Join-Path $WorkerBuildDir 'AilaWorker.exe'
$fakeWorkerExecutable = Join-Path $WorkerBuildDir 'AilaFakeInferenceWorker.exe'
foreach ($required in @($workerInfoPath, $workerExecutable, $fakeWorkerExecutable)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Build the external Worker and fake Worker first; missing: $required"
    }
}
$workerInfo = Get-Content -LiteralPath $workerInfoPath -Raw | ConvertFrom-Json
if ($workerInfo.schemaVersion -ne 2 -or $workerInfo.git.fullCommit -notmatch '^[0-9a-fA-F]{40}$') {
    throw 'Worker build_info.json must contain schemaVersion 2 and a full git.fullCommit'
}
$workerCommit = $workerInfo.git.fullCommit.ToLowerInvariant()
$versionFile = Join-Path $repoRoot 'VERSION.txt'
if (-not (Test-Path -LiteralPath $versionFile -PathType Leaf)) {
    throw "Missing VERSION.txt file: $versionFile"
}
$repoVersion = ([IO.File]::ReadAllText($versionFile)).Trim()
if ($repoVersion -notmatch '^\d+\.\d+\.\d+$') {
    throw "Invalid VERSION.txt file contents: '$repoVersion'"
}
$workerVersion = if ($workerInfo.PSObject.Properties['version'] -and -not [string]::IsNullOrWhiteSpace($workerInfo.version)) {
    [string]$workerInfo.version
} else {
    $repoVersion
}
$workerBuildId = if ($workerInfo.PSObject.Properties['buildId'] -and -not [string]::IsNullOrWhiteSpace($workerInfo.buildId)) {
    [string]$workerInfo.buildId
} else {
    "aila-$workerVersion-abi1-$workerCommit"
}

function Get-LfSha256 {
    param([string]$Path)
    $patchText = [IO.File]::ReadAllText($Path)
    $patchText = $patchText.Replace([string][char]13 + [char]10, [string][char]10)
    $digest = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($digest.ComputeHash(
                [Text.Encoding]::UTF8.GetBytes($patchText))).Replace('-', '').ToLowerInvariant()
    }
    finally { $digest.Dispose() }
}

function Assert-ServerDependencies {
    $lockPath = Join-Path $DependencyRoot 'dependency-lock.json'
    if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        throw "Prepare dependencies with tools/server/dependencies.ps1 first; missing: $lockPath"
    }
    $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
    if ($lock.schemaVersion -ne 1 -or $lock.purpose -ne 'aila-server-dependencies' -or
        $lock.pendingWriteBytesApi -ne 1 -or $lock.compiler -ne 'MSVC' -or
        $lock.cxxStandard -ne 17 -or $lock.msvcRuntime -ne 'MultiThreadedDLL' -or
        $lock.tls -ne $false -or $lock.triplet -ne 'x64-windows-static-md' -or
        $lock.vcpkgBaseline -ne '544a4c5c297e60e4ac4a5a1810df66748d908869') {
        throw 'The dependency lock does not describe the pinned MSVC C++17 /MD server dependencies'
    }
    $prefix = Join-Path $DependencyRoot 'prefix'
    $vcpkgPrefix = Join-Path $DependencyRoot 'vcpkg_installed/x64-windows-static-md'
    if (-not (Resolve-AilaBuildPath $lock.prefix).Equals($prefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Resolve-AilaBuildPath $lock.vcpkgPrefix).Equals($vcpkgPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The dependency lock belongs to a different install prefix'
    }
    $expectedSources = @(
        @{ name = 'trantor'; version = '1.5.24'; sha512 = 'a875e5bf8c8d871d1a3ddeeb79891773eed4f13772aa8c09ce6bdd60fa7af621eed6986935c0b70b58e790618934ca82a7ebd4ba5faf2b534b8ecb6f74d6bad1' },
        @{ name = 'drogon'; version = '1.9.11'; sha512 = 'ce61c65281bde0246e96c3558cf635241adbcab514d6a71e8267201340f577af26b1cbc69fa48652d655971122a66c70db75d1a4c034384ba0716fccf7ad9584' }
    )
    foreach ($expected in $expectedSources) {
        $actual = @($lock.sources | Where-Object { $_.name -eq $expected.name })
        if ($actual.Count -ne 1 -or $actual[0].version -ne $expected.version -or
            $actual[0].sha512 -ne $expected.sha512) {
            throw "Unexpected source lock for $($expected.name)"
        }
    }
    foreach ($expected in @(@{ name = 'jsoncpp'; version = '1.9.6' },
            @{ name = 'zlib'; version = '1.3.1' })) {
        $actual = @($lock.packages | Where-Object { $_.name -eq $expected.name })
        if ($actual.Count -ne 1 -or $actual[0].version -ne $expected.version) {
            throw "Unexpected package lock for $($expected.name)"
        }
    }
    $expectedPatches = @(
        @{
            name = 'trantor-pending-bytes'; source = 'trantor'
            path = Join-Path $PSScriptRoot 'trantor-pending-bytes.patch'
            sha256Lf = 'c1c359ef7cf9dc5ac776c7efa0339805ec90f88af6aa7bd07a585746f06cc5fc'
        },
        @{
            name = 'trantor-channel-interest'; source = 'trantor'
            path = Join-Path $PSScriptRoot 'trantor-channel-interest.patch'
            sha256Lf = 'f55f7014b88e6603d8ec643c4f46c5758c37f537579a98c1a8ef05df44526cb4'
        },
        @{
            name = 'drogon-windows-listener-callbacks'; source = 'drogon'
            path = Join-Path $PSScriptRoot '../http-probe/drogon-windows-listener-callbacks.patch'
            sha256Lf = 'f267e88c29eaa99bb8b138fae3f8224d4843a0ef9dc393406dbd7c43498cb179'
        }
    )
    foreach ($expected in $expectedPatches) {
        $actual = @($lock.patches | Where-Object { $_.name -eq $expected.name })
        $installedPatch = Join-Path $DependencyRoot "patches/$($expected.name).patch"
        if ($actual.Count -ne 1 -or $actual[0].source -ne $expected.source -or
            $actual[0].sha256Lf -ne $expected.sha256Lf -or
            -not (Resolve-AilaBuildPath $actual[0].path).Equals($installedPatch, [StringComparison]::OrdinalIgnoreCase) -or
            (Get-LfSha256 $expected.path) -ne $expected.sha256Lf -or
            (Get-LfSha256 $installedPatch) -ne $expected.sha256Lf) {
            throw "The dependency patch does not match the current server contract: $($expected.name)"
        }
    }
    foreach ($relativePath in @('include/trantor/net/TcpConnection.h', 'lib/trantor.lib', 'lib/drogon.lib')) {
        $artifactPath = Join-Path $prefix $relativePath
        $actual = @($lock.artifacts | Where-Object {
            (Resolve-AilaBuildPath $_.path).Equals($artifactPath, [StringComparison]::OrdinalIgnoreCase)
        })
        if ($actual.Count -ne 1 -or -not (Test-Path -LiteralPath $artifactPath -PathType Leaf) -or
            $actual[0].sha256 -notmatch '^[0-9a-fA-F]{64}$' -or
            (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash -ne $actual[0].sha256) {
            throw "Installed dependency changed or is incomplete: $artifactPath"
        }
    }
    $header = [IO.File]::ReadAllText((Join-Path $prefix 'include/trantor/net/TcpConnection.h'))
    if (-not $header.Contains('virtual size_t pendingWriteBytes() const = 0;')) {
        throw 'The installed Trantor header is missing the pendingWriteBytes() extension'
    }
    return $lock
}

if (-not $SkipDependencies) {
    & (Join-Path $PSScriptRoot 'dependencies.ps1') -DependencyRoot $DependencyRoot -Parallelism $Parallelism
}
$dependencyLock = Assert-ServerDependencies

# Import ordinary MSVC into this process. Do not log the inherited environment.
$programFilesX86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
$vswhere = Join-Path $programFilesX86 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { throw 'Visual Studio vswhere.exe is required' }
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
    if ($environmentProcess.ExitCode -ne 0) {
        throw "Unable to import the MSVC environment: $environmentError"
    }
}
finally { $environmentProcess.Dispose() }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:VSLANG = '1033'
foreach ($variable in @('CL', '_CL_', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS')) {
    [Environment]::SetEnvironmentVariable($variable, $null, 'Process')
}
$compiler = Join-Path $env:VCToolsInstallDir 'bin/Hostx64/x64/cl.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw 'The x64 MSVC compiler is required' }
$cmake = (Get-Command cmake.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$ninja = (Get-Command ninja.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$compiler = [IO.Path]::GetFullPath($compiler)
New-Item -ItemType Directory -Force -Path (Join-Path $BuildRoot 'logs') | Out-Null

function Invoke-ServerBuildCommand {
    param([string]$Program, [string[]]$Arguments, [string]$LogName)
    $log = Join-Path $BuildRoot "logs/$LogName.txt"
    Write-Host "Running $LogName; log: $log"
    & $Program @Arguments 2>&1 | Out-File -LiteralPath $log -Encoding utf8
    $commandExitCode = $LASTEXITCODE
    if ($commandExitCode -ne 0) {
        Get-Content -LiteralPath $log -Tail 60 | ForEach-Object { Write-Host $_ }
        throw "$Program failed with exit $commandExitCode; see $log"
    }
}

function Assert-ServerHeaderDependencies {
    # Query only current C++ outputs, not stale entries for removed targets.
    Invoke-ServerBuildCommand $ninja @('-C', $BuildRoot, '-t', 'targets', 'all') 'server-compile-targets'
    $compileTargets = @(Get-Content -LiteralPath (Join-Path $BuildRoot 'logs/server-compile-targets.txt') |
        ForEach-Object { if ($_ -match '^(.+\.obj): CXX_COMPILER__\S+$') { $matches[1] } } |
        Select-Object -Unique)
    if ($compileTargets.Count -eq 0) { throw 'Ninja did not report any Server C++ compile targets' }
    Invoke-ServerBuildCommand $ninja (@('-C', $BuildRoot, '-t', 'deps') + $compileTargets) 'server-header-dependencies'
    $dependencies = @{}
    foreach ($line in (Get-Content -LiteralPath (Join-Path $BuildRoot 'logs/server-header-dependencies.txt'))) {
        if ($line -match '^(.+\.obj): #deps ([0-9]+), deps mtime [0-9]+ \(([^)]+)\)$') {
            $dependencies[$matches[1]] = @{ count = [int]$matches[2]; status = $matches[3] }
        }
    }
    foreach ($target in $compileTargets) {
        if (-not $dependencies.ContainsKey($target) -or $dependencies[$target].count -le 0 -or
            $dependencies[$target].status -ne 'VALID') {
            throw "Missing, empty or stale Ninja header dependencies for $target; see logs/server-header-dependencies.txt"
        }
    }
    $probeDirectory = Join-Path $BuildRoot 'CMakeFiles/AilaShowIncludes'
    $prefixPath = Join-Path $probeDirectory 'prefix.bin'
    $prefixHash = (Get-Content -LiteralPath (Join-Path $probeDirectory 'prefix.sha256') -Raw).Trim()
    if ($prefixHash -notmatch '^[0-9a-f]{64}$' -or
        (Get-FileHash -LiteralPath $prefixPath -Algorithm SHA256).Hash -ne $prefixHash) {
        throw 'The MSVC include-prefix probe fingerprint is missing or inconsistent'
    }
    Write-Host "Verified nonempty Ninja header dependencies for $($compileTargets.Count) C++ objects"
    return [ordered]@{
        format = 'msvc-showIncludes'
        prefixSha256 = $prefixHash
        prefixHex = [BitConverter]::ToString([IO.File]::ReadAllBytes($prefixPath)).Replace('-', '').ToLowerInvariant()
        verifiedObjects = $compileTargets.Count
        probeDirectory = $probeDirectory
    }
}

$prefix = $dependencyLock.prefix
$vcpkgPrefix = $dependencyLock.vcpkgPrefix
# --fresh drops cached oneAPI or package search settings if this build directory
# has been configured before; it preserves the separately installed dependencies.
$configureArguments = @('--fresh', '-S', $PSScriptRoot, '-B', $BuildRoot,
    '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", "-DCMAKE_CXX_COMPILER=$compiler",
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL',
    '-DCMAKE_TOOLCHAIN_FILE=', "-DCMAKE_PREFIX_PATH=$prefix;$vcpkgPrefix",
    "-DDrogon_DIR=$prefix/lib/cmake/Drogon", "-DTrantor_DIR=$prefix/lib/cmake/Trantor",
    "-Djsoncpp_DIR=$vcpkgPrefix/share/jsoncpp",
    "-DJSONCPP_INCLUDE_DIRS=$vcpkgPrefix/include", "-DJSONCPP_LIBRARIES=$vcpkgPrefix/lib/jsoncpp.lib",
    "-DZLIB_ROOT=$vcpkgPrefix", "-DZLIB_INCLUDE_DIR=$vcpkgPrefix/include",
    "-DZLIB_LIBRARY_RELEASE=$vcpkgPrefix/lib/zlib.lib",
    '-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON', '-DCMAKE_DISABLE_FIND_PACKAGE_Botan=ON',
    "-DAILA_WORKER_BUILD_ID=$workerBuildId", "-DAILA_FAKE_WORKER_PATH=$fakeWorkerExecutable")
Invoke-ServerBuildCommand $cmake $configureArguments 'configure-server'
Invoke-ServerBuildCommand $cmake @('--build', $BuildRoot, '--config', 'Release',
    '--parallel', "$Parallelism", '--target', 'AilaServer', 'AilaServerChatTests',
    'AilaServerClientTests', 'AilaServerPipeTests', 'AilaServerChannelTests') 'build-server'
$compilerDependencies = Assert-ServerHeaderDependencies

$manifest = [ordered]@{
    schemaVersion = 1
    purpose = 'aila-server-build'
    generatedAtUtc = [DateTime]::UtcNow.ToString('o')
    sourceDirectory = $PSScriptRoot
    buildDirectory = $BuildRoot
    compiler = $compiler
    compilerDependencies = $compilerDependencies
    cxxStandard = 17
    msvcRuntime = 'MultiThreadedDLL'
    worker = [ordered]@{
        buildId = $workerBuildId
        buildInfo = $workerInfoPath
        executable = $workerExecutable
        executableSha256 = (Get-FileHash -LiteralPath $workerExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
        fakeExecutable = $fakeWorkerExecutable
        fakeExecutableSha256 = (Get-FileHash -LiteralPath $fakeWorkerExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    dependencies = [ordered]@{
        lock = Join-Path $DependencyRoot 'dependency-lock.json'
        lockSha256 = (Get-FileHash -LiteralPath (Join-Path $DependencyRoot 'dependency-lock.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        prefix = $prefix
        vcpkgPrefix = $vcpkgPrefix
    }
}
$manifest | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $BuildRoot 'server-build-info.json') -Encoding utf8
Write-Host "Server ready: $(Join-Path $BuildRoot 'AilaServer.exe')"
Write-Host "Expected Worker identity: $workerBuildId"
Write-Host ('Run CPU tests separately: ctest --test-dir "{0}" -C Release --output-on-failure' -f $BuildRoot)
