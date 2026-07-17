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

$allowedRootFiles = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase)
$null = $allowedRootFiles.Add('AilaShared.dll')
$allowedRootDirectories = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase)
$null = $allowedRootDirectories.Add('aila_runtime')
foreach ($entry in (Get-ChildItem -LiteralPath $releaseRoot -Force)) {
    if ($entry.PSIsContainer) {
        if (-not $allowedRootDirectories.Contains($entry.Name)) {
            throw "Unexpected release root directory '$($entry.Name)'; only 'aila_runtime' is allowed."
        }
    }
    elseif (-not $allowedRootFiles.Contains($entry.Name)) {
        throw "Unexpected release root file '$($entry.Name)'; only 'AilaShared.dll' is allowed."
    }
}

$dumpbinPath = Resolve-DumpbinPath -BuildDirPath $buildDirPath
$hwlocFiles = @(Get-ChildItem -LiteralPath $runtimeDir -File -Filter 'libhwloc-*.dll')
if ($hwlocFiles.Count -eq 0) {
    throw "Staged runtime dependency closure is missing libhwloc-*.dll in '$runtimeDir'."
}

$system32 = Join-Path $env:SystemRoot 'System32'
$missingDependencies = [System.Collections.Generic.List[string]]::new()
$runtimeImages = @(
    Get-ChildItem -LiteralPath $runtimeDir -File |
        Where-Object { $_.Extension -in @('.dll', '.exe') }
)
foreach ($image in $runtimeImages) {
    $imageDependencies = (& $dumpbinPath /dependents $image.FullName 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "dumpbin /dependents failed for '$($image.FullName)' with exit code $LASTEXITCODE. Output: $imageDependencies"
    }
    $dependencyNames = @([regex]::Matches($imageDependencies, '(?im)^\s+([^\s]+\.dll)\s*$') |
        ForEach-Object { $_.Groups[1].Value } |
        Sort-Object -Unique)
    foreach ($dependencyName in $dependencyNames) {
        if ($dependencyName -match '^(?:api-ms-|ext-ms-)') {
            continue
        }
        $runtimeCandidate = Join-Path $runtimeDir $dependencyName
        $systemCandidate = Join-Path $system32 $dependencyName
        if (-not (Test-Path -LiteralPath $runtimeCandidate -PathType Leaf) -and
            -not (Test-Path -LiteralPath $systemCandidate -PathType Leaf)) {
            $missingDependencies.Add("$($image.Name) -> $dependencyName")
        }
    }
}
if ($missingDependencies.Count -ne 0) {
    throw "Staged runtime dependency closure is incomplete: $([string]::Join(', ', $missingDependencies))."
}

$dependencyOutput = (& $dumpbinPath /dependents $proxyPath 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /dependents failed for '$proxyPath' with exit code $LASTEXITCODE. Output: $dependencyOutput"
}
$forbiddenImports = @([regex]::Matches(
        $dependencyOutput,
        '(?im)^\s*((?:sycl.*|dnnl|tbb.*|umf|ur_.*|libmmd.*|OpenCL|intelocl64|common_clang64|xptifw|libhwloc-.*|tcm)\.dll)\s*$') |
    ForEach-Object { $_.Groups[1].Value } |
    Sort-Object -Unique)
if ($forbiddenImports.Count -ne 0) {
    throw "AilaShared.dll imports forbidden oneAPI runtime DLLs: $($forbiddenImports -join ', ')"
}

Write-Host 'ProxyDependencyTests PASS'
