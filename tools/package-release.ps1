[CmdletBinding()]
param(
    [string]$Version = '',
    [string]$WorkerBuildDir = 'build-refactor',
    [string]$ServerBuildDir = 'build-refactor-server',
    [string]$DependencyRoot = 'build-refactor-server-deps',
    [string]$OutputDir = '',
    [string]$OneApiStack = 'oneapi-2026.2',
    [string]$OneApiStacksFile = 'perf\oneapi-stacks.json',
    [int]$Jobs = 36,
    [switch]$SkipBuild,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot '..\perf\PerfCommon.ps1')

$repoRoot = Get-AilaRepoRoot
if ([string]::IsNullOrWhiteSpace($Version)) {
    $versionFile = Join-Path $repoRoot 'VERSION.txt'
    if (-not (Test-Path -LiteralPath $versionFile -PathType Leaf)) {
        throw "Missing VERSION.txt file: $versionFile"
    }
    $Version = ([IO.File]::ReadAllText($versionFile)).Trim()
}
if ($Version -notmatch '^\d+\.\d+\.\d+$') {
    throw "Invalid release version '$Version'; expected SemVer X.Y.Z"
}
$workerBuildPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $WorkerBuildDir
$serverBuildPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $ServerBuildDir
$depRootPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $DependencyRoot

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $outputPath = Join-Path $workerBuildPath 'Release\dist'
} else {
    $outputPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputDir
}

if (-not $SkipBuild) {
    Write-Host ':: Building Worker / CLI / Shared Libraries ::' -ForegroundColor Cyan
    & (Join-Path $repoRoot 'build.ps1') `
        -BuildDir $workerBuildPath `
        -Config Release `
        -OneApiStack $OneApiStack `
        -OneApiStacksFile $OneApiStacksFile `
        -Jobs $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "build.ps1 failed with exit code $LASTEXITCODE."
    }

    Write-Host ':: Building AilaServer.exe (MSVC C++17) ::' -ForegroundColor Cyan
    & (Join-Path $repoRoot 'tools\server\build.ps1') `
        -WorkerBuildDir $workerBuildPath `
        -BuildRoot $serverBuildPath `
        -DependencyRoot $depRootPath `
        -SkipDependencies
    if ($LASTEXITCODE -ne 0) {
        throw "tools/server/build.ps1 failed with exit code $LASTEXITCODE."
    }
}

$stackConfig = Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path $OneApiStacksFile
$stack = Get-AilaOneApiStack -Config $stackConfig -Name $OneApiStack
$stackEnv = Get-AilaOneApiStackEnvironment -Stack $stack
Set-AilaProcessEnvironment -Environment $stackEnv

Write-Host ':: Staging CMake release target ::' -ForegroundColor Cyan
& cmake --build $workerBuildPath --config Release --target release
if ($LASTEXITCODE -ne 0) {
    throw "cmake --build --target release failed with exit code $LASTEXITCODE."
}

$stagedBinDir = Join-Path $workerBuildPath 'Release\bin'
$stagedRuntimeDir = Join-Path $stagedBinDir 'aila_runtime'
$serverExe = Join-Path $serverBuildPath 'AilaServer.exe'
if (-not (Test-Path -LiteralPath $serverExe -PathType Leaf)) {
    throw "Missing AilaServer.exe at $serverExe"
}

Copy-Item -LiteralPath $serverExe -Destination (Join-Path $stagedBinDir 'AilaServer.exe') -Force

# Verify split-layout isolation invariants
$forbiddenRootDllPatterns = @('*sycl*.dll', '*dnnl*.dll', '*onednn*.dll', '*ze_loader*.dll', '*ur_*.dll', '*tbb*.dll', '*intelocl*.dll', '*common_clang*.dll')
$rootDllViolations = @()
foreach ($pattern in $forbiddenRootDllPatterns) {
    $rootDllViolations += Get-ChildItem -LiteralPath $stagedBinDir -Filter $pattern -File -ErrorAction SilentlyContinue
}
if ($rootDllViolations.Count -gt 0) {
    $names = ($rootDllViolations | Select-Object -ExpandProperty Name) -join ', '
    throw "Host root directory contains forbidden GPU/oneAPI DLLs: $names"
}

foreach ($required in @(
    (Join-Path $stagedBinDir 'AilaServer.exe'),
    (Join-Path $stagedBinDir 'AilaShared.dll'),
    (Join-Path $stagedBinDir 'AilaInference.dll'),
    (Join-Path $stagedRuntimeDir 'Aila.exe'),
    (Join-Path $stagedRuntimeDir 'AilaWorker.exe')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required staged release binary: $required"
    }
}

# Smoke check staged CLI & Server binaries
Write-Host ':: Smoke checking staged binaries ::' -ForegroundColor Cyan
$cliVersion = & (Join-Path $stagedRuntimeDir 'Aila.exe') --version
if ($LASTEXITCODE -ne 0) {
    throw "Staged Aila.exe --version failed with exit code $LASTEXITCODE."
}
Write-Host ("   CLI: {0}" -f ($cliVersion -join ' '))

$null = & (Join-Path $stagedBinDir 'AilaServer.exe') --help
if ($LASTEXITCODE -ne 0) {
    throw "Staged AilaServer.exe --help failed with exit code $LASTEXITCODE."
}

# Prepare final dist directory
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$packageFolderName = "Aila-v$Version-win64"
$packageRoot = Join-Path $outputPath $packageFolderName
New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null
Copy-Item -Path (Join-Path $stagedBinDir '*') -Destination $packageRoot -Recurse -Force

# Also emit standalone top-level binaries for individual GitHub Release uploads
foreach ($standalone in @(
    (Join-Path $stagedBinDir 'AilaServer.exe'),
    (Join-Path $stagedBinDir 'AilaShared.dll'),
    (Join-Path $stagedBinDir 'AilaInference.dll'),
    (Join-Path $stagedRuntimeDir 'Aila.exe'),
    (Join-Path $stagedRuntimeDir 'AilaWorker.exe')
)) {
    Copy-Item -LiteralPath $standalone -Destination $outputPath -Force
}

$zipPath = Join-Path $outputPath "$packageFolderName.zip"
if (-not $NoZip) {
    Write-Host (":: Creating {0} ::" -f $zipPath) -ForegroundColor Cyan
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $packageRoot,
        $zipPath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )
}

$checksumLines = @()
Get-ChildItem -LiteralPath $outputPath -File | Sort-Object Name | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $checksumLines += "$hash  $($_.Name)"
}
Set-Content -LiteralPath (Join-Path $outputPath 'SHA256SUMS.txt') -Value $checksumLines -Encoding UTF8

Write-Host ':: Release package ready ::' -ForegroundColor Green
Get-ChildItem -LiteralPath $outputPath -File | Sort-Object Name | Format-Table Name, Length -AutoSize
