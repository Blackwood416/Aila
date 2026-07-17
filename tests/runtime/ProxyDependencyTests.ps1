param(
    [Parameter(Mandatory = $true)][string]$BuildDir
)

$ErrorActionPreference = 'Stop'

function Resolve-DumpbinPath {
    param([Parameter(Mandatory = $true)][string]$BuildDirPath)

    $command = Get-Command 'dumpbin.exe' -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) {
        $candidates.Add((Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64\dumpbin.exe'))
    }

    $cachePath = Join-Path $BuildDirPath 'CMakeCache.txt'
    if (Test-Path -LiteralPath $cachePath -PathType Leaf) {
        $linkerLine = Get-Content -LiteralPath $cachePath |
            Where-Object { $_ -match '^CMAKE_LINKER:FILEPATH=' } |
            Select-Object -First 1
        if ($null -ne $linkerLine) {
            $linkerPath = $linkerLine.Substring($linkerLine.IndexOf('=') + 1)
            $candidates.Add((Join-Path (Split-Path -Parent $linkerPath) 'dumpbin.exe'))
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    throw 'dumpbin.exe was not found on PATH, under VCToolsInstallDir, or beside CMAKE_LINKER.'
}

function Test-OneApiRuntimeName {
    param([Parameter(Mandatory = $true)][string]$Name)

    return $Name -match '^(?:sycl.*\.dll|dnnl\.dll|tbb.*\.dll|umf\.dll|ur_.*\.dll)$'
}

function Get-ArtifactMetadata {
    param(
        [Parameter(Mandatory = $true)]$BuildInfo,
        [Parameter(Mandatory = $true)][string]$Role
    )

    $matches = @($BuildInfo.artifacts | Where-Object { $_.role -eq $Role })
    if ($matches.Count -ne 1) {
        throw "build_info.json must contain exactly one '$Role' artifact; found $($matches.Count)."
    }
    return $matches[0]
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$buildDirPath = if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    [System.IO.Path]::GetFullPath($BuildDir)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
}
$releaseRoot = Join-Path $buildDirPath 'Release\bin'
$runtimeDir = Join-Path $releaseRoot 'aila_runtime'
$proxyPath = Join-Path $releaseRoot 'AilaShared.dll'
$workerPath = Join-Path $runtimeDir 'AilaWorker.exe'
$cliPath = Join-Path $runtimeDir 'Aila.exe'

foreach ($requiredDirectory in @($releaseRoot, $runtimeDir)) {
    if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
        throw "Required staged directory not found: $requiredDirectory"
    }
}
foreach ($requiredFile in @($proxyPath, $workerPath, $cliPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required staged artifact not found: $requiredFile"
    }
}

foreach ($forbiddenRootExecutable in @('AilaWorker.exe', 'Aila.exe')) {
    $candidate = Join-Path $releaseRoot $forbiddenRootExecutable
    if (Test-Path -LiteralPath $candidate) {
        throw "Runtime executable must be staged below aila_runtime, not beside the proxy: $candidate"
    }
}

$rootRuntimeDlls = @(Get-ChildItem -LiteralPath $releaseRoot -File |
    Where-Object { Test-OneApiRuntimeName -Name $_.Name })
if ($rootRuntimeDlls.Count -ne 0) {
    throw "oneAPI runtime DLLs must not be staged beside AilaShared.dll: $($rootRuntimeDlls.Name -join ', ')"
}

$dumpbinPath = Resolve-DumpbinPath -BuildDirPath $buildDirPath
$dependencyOutput = (& $dumpbinPath /dependents $proxyPath 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /dependents failed for '$proxyPath' with exit code $LASTEXITCODE. Output: $dependencyOutput"
}
$forbiddenImports = @([regex]::Matches(
        $dependencyOutput,
        '(?im)^\s*((?:sycl.*\.dll|dnnl\.dll|tbb.*\.dll|umf\.dll|ur_.*\.dll))\s*$') |
    ForEach-Object { $_.Groups[1].Value } |
    Sort-Object -Unique)
if ($forbiddenImports.Count -ne 0) {
    throw "AilaShared.dll imports forbidden oneAPI runtime DLLs: $($forbiddenImports -join ', ')"
}

$releaseBuildInfoPath = Join-Path $releaseRoot 'build_info.json'
if (-not (Test-Path -LiteralPath $releaseBuildInfoPath -PathType Leaf)) {
    throw "Staged build metadata not found: $releaseBuildInfoPath"
}
$buildInfo = Get-Content -LiteralPath $releaseBuildInfoPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($buildInfo.schemaVersion -ne 2) {
    throw "Unsupported build_info.json schema: $($buildInfo.schemaVersion)"
}

foreach ($expectation in @(
        @{ role = 'proxy'; relativePath = 'AilaShared.dll'; fullPath = $proxyPath },
        @{ role = 'worker'; relativePath = 'aila_runtime/AilaWorker.exe'; fullPath = $workerPath }
    )) {
    $artifact = Get-ArtifactMetadata -BuildInfo $buildInfo -Role $expectation.role
    if ($artifact.relativePath -cne $expectation.relativePath) {
        throw "Unexpected $($expectation.role) relative path: '$($artifact.relativePath)'"
    }
    $expectedHash = (Get-FileHash -LiteralPath $expectation.fullPath -Algorithm SHA256).Hash
    if (-not ([string]$artifact.sha256).Equals($expectedHash, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$($expectation.role) SHA256 does not match staged artifact."
    }
}

Write-Host 'ProxyDependencyTests PASS'
