param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$OneApiStack,
    [string]$OneApiStacksFile = 'perf\oneapi-stacks.json',
    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

function Get-RequiredOneApiMetadataValue {
    param(
        [Parameter(Mandatory = $true)]$Metadata,
        [Parameter(Mandatory = $true)][string]$PropertyName,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $property = $Metadata.PSObject.Properties[$PropertyName]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "$Context is missing required property '$PropertyName'."
    }
    return $property.Value
}

function Assert-AilaVerificationPathOwnership {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$BuildDirPath,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $leaf = Split-Path -Leaf $Path
    if (-not $leaf.Equals('oneapi_verification.json', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Verification output file must be named 'oneapi_verification.json': $Path"
    }
    if (-not (Test-AilaPathWithinRoot -Path $Path -Root $BuildDirPath)) {
        throw "Verification output path is outside selected build directory '$BuildDirPath': $Path"
    }
    if (-not (Test-AilaPathWithinRoot -Path $Path -Root $RepoRoot)) {
        throw "Verification output path is outside repository '$RepoRoot': $Path"
    }
    if (Test-Path -LiteralPath $Path -PathType Container) {
        throw "Verification output path is a directory: $Path"
    }
}

function Get-AilaApprovedVerificationPath {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$BuildDirPath,
        [AllowEmptyString()][string]$RequestedOutputPath
    )

    $verificationPath = if ([string]::IsNullOrWhiteSpace($RequestedOutputPath)) {
        Join-Path $BuildDirPath 'oneapi_verification.json'
    }
    else {
        Resolve-AilaPath -RepoRoot $RepoRoot -Path $RequestedOutputPath
    }
    $verificationPath = [System.IO.Path]::GetFullPath($verificationPath)
    Assert-AilaVerificationPathOwnership -RepoRoot $RepoRoot -BuildDirPath $BuildDirPath -Path $verificationPath

    return $verificationPath
}

function Remove-AilaPreviousVerification {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        Remove-Item -LiteralPath $Path -Force
    }
}

function Write-AilaAtomicVerificationJson {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$BuildDirPath,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Data
    )

    $parent = Split-Path -Parent $Path
    Ensure-AilaDirectory -Path $parent
    Assert-AilaVerificationPathOwnership -RepoRoot $RepoRoot -BuildDirPath $BuildDirPath -Path $Path

    $leaf = Split-Path -Leaf $Path
    $tempPath = Join-Path $parent ".$leaf.$([guid]::NewGuid().ToString('N')).tmp"
    $stream = $null
    $writer = $null
    try {
        $stream = [System.IO.FileStream]::new(
            $tempPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $writer = [System.IO.StreamWriter]::new($stream, [System.Text.UTF8Encoding]::new($false))
        $writer.Write(($Data | ConvertTo-Json -Depth 16))
        $writer.Flush()
        $stream.Flush($true)
        $writer.Dispose()
        $writer = $null
        $stream = $null

        $null = Get-Content -LiteralPath $tempPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if (Test-Path -LiteralPath $Path) {
            throw "Verification output path appeared concurrently before publication: $Path"
        }
        [System.IO.File]::Move($tempPath, $Path)
    }
    finally {
        if ($null -ne $writer) {
            $writer.Dispose()
        }
        elseif ($null -ne $stream) {
            $stream.Dispose()
        }
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force
        }
    }
}

function Invoke-AilaStableDependencyInspection {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string]$ExpectedSyclDll,
        [Parameter(Mandatory = $true)][scriptblock]$DependencyInspector
    )

    $hashBefore = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    $dependenciesText = (& $DependencyInspector $Executable | Out-String)
    $hashAfter = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    if ($hashBefore -ne $hashAfter) {
        throw "Aila executable changed during dependency inspection: '$Executable' (before '$hashBefore', after '$hashAfter')."
    }

    $syclDlls = @(
        [regex]::Matches($dependenciesText, '(?im)^\s+(sycl\d+\.dll)\s*$') |
            ForEach-Object { $_.Groups[1].Value.ToLowerInvariant() } |
            Sort-Object -Unique
    )
    $expected = $ExpectedSyclDll.ToLowerInvariant()
    if ($syclDlls.Count -ne 1 -or $syclDlls[0] -ne $expected) {
        $actual = if ($syclDlls.Count -eq 0) { '<none>' } else { [string]::Join(', ', $syclDlls) }
        throw "SYCL runtime ABI mismatch for '$Executable': expected exactly '$expected', found '$actual'."
    }

    return [pscustomobject]@{
        dependencies     = @($syclDlls)
        executableSha256 = $hashAfter
    }
}

function Invoke-AilaStableProxyDependencyInspection {
    param(
        [Parameter(Mandatory = $true)][string]$Proxy,
        [Parameter(Mandatory = $true)][scriptblock]$DependencyInspector
    )

    $hashBefore = (Get-FileHash -LiteralPath $Proxy -Algorithm SHA256).Hash
    $dependenciesText = (& $DependencyInspector $Proxy | Out-String)
    $hashAfter = (Get-FileHash -LiteralPath $Proxy -Algorithm SHA256).Hash
    if ($hashBefore -ne $hashAfter) {
        throw "Aila proxy changed during dependency inspection: '$Proxy' (before '$hashBefore', after '$hashAfter')."
    }

    $forbidden = @(
        [regex]::Matches(
            $dependenciesText,
            '(?im)^\s*((?:sycl.*\.dll|dnnl\.dll|tbb.*\.dll|umf\.dll|ur_.*\.dll))\s*$') |
            ForEach-Object { $_.Groups[1].Value.ToLowerInvariant() } |
            Sort-Object -Unique
    )
    if ($forbidden.Count -ne 0) {
        throw "AilaShared.dll has forbidden oneAPI imports: $([string]::Join(', ', $forbidden))."
    }

    return [pscustomobject]@{
        dependencies = @()
        sha256        = $hashAfter
    }
}

function Get-AilaBuildArtifactMetadata {
    param(
        [Parameter(Mandatory = $true)]$BuildInfo,
        [Parameter(Mandatory = $true)][string]$Role
    )

    $artifactsProperty = $BuildInfo.PSObject.Properties['artifacts']
    $matches = @(if ($null -eq $artifactsProperty -or $null -eq $artifactsProperty.Value) {
            @()
        }
        else {
            $artifactsProperty.Value | Where-Object { $_.role -eq $Role }
        })
    if ($matches.Count -ne 1) {
        throw "Build info must contain exactly one '$Role' artifact; found $($matches.Count)."
    }
    return $matches[0]
}

function Assert-AilaStagedArtifactMetadata {
    param(
        [Parameter(Mandatory = $true)]$BuildInfo,
        [Parameter(Mandatory = $true)][string]$Role,
        [Parameter(Mandatory = $true)][string]$ExpectedRelativePath,
        [Parameter(Mandatory = $true)][string]$ArtifactPath
    )

    $artifact = Get-AilaBuildArtifactMetadata -BuildInfo $BuildInfo -Role $Role
    $relativePath = [string](Get-RequiredOneApiMetadataValue -Metadata $artifact -PropertyName 'relativePath' -Context "Build info $Role artifact")
    if (-not $relativePath.Equals($ExpectedRelativePath, [System.StringComparison]::Ordinal)) {
        throw "Build info $Role artifact path mismatch: expected '$ExpectedRelativePath', found '$relativePath'."
    }
    $recordedHash = [string](Get-RequiredOneApiMetadataValue -Metadata $artifact -PropertyName 'sha256' -Context "Build info $Role artifact")
    $actualHash = (Get-FileHash -LiteralPath $ArtifactPath -Algorithm SHA256).Hash
    if (-not $recordedHash.Equals($actualHash, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Build info $Role artifact SHA256 mismatch: recorded '$recordedHash', staged '$actualHash'."
    }
}

function Test-AilaOneApiRuntimeDllName {
    param([Parameter(Mandatory = $true)][string]$Name)

    return $Name -match '^(?:sycl.*\.dll|dnnl\.dll|tbb.*\.dll|umf\.dll|ur_.*\.dll)$'
}

function Get-AilaDumpbinPath {
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

function Invoke-AilaOneApiBuildVerification {
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)][string]$OneApiStack,
        [string]$OneApiStacksFile = 'perf\oneapi-stacks.json',
        [string]$OutputPath = '',
        [scriptblock]$DependencyInspector,
        [string]$RepoRoot = ''
    )

    $repoRoot = if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
        [System.IO.Path]::GetFullPath($PSScriptRoot)
    }
    else {
        [System.IO.Path]::GetFullPath($RepoRoot)
    }
    $buildDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BuildDir
    $verificationPath = Get-AilaApprovedVerificationPath `
        -RepoRoot $repoRoot `
        -BuildDirPath $buildDirPath `
        -RequestedOutputPath $OutputPath
    Remove-AilaPreviousVerification -Path $verificationPath

    if (-not (Test-Path -LiteralPath $buildDirPath -PathType Container)) {
        throw "Build directory not found: $buildDirPath"
    }

    $buildInfoPath = Join-Path $buildDirPath 'build_info.json'
    if (-not (Test-Path -LiteralPath $buildInfoPath -PathType Leaf)) {
        throw "Build info file not found: $buildInfoPath"
    }

    $releaseRoot = Join-Path $buildDirPath 'Release\bin'
    $runtimeDir = Join-Path $releaseRoot 'aila_runtime'
    $proxy = Join-Path $releaseRoot 'AilaShared.dll'
    $worker = Join-Path $runtimeDir 'AilaWorker.exe'
    $exe = Join-Path $runtimeDir 'Aila.exe'

    $buildInfo = Read-AilaJsonFile -Path $buildInfoPath
    $schemaProperty = if ($null -eq $buildInfo) { $null } else { $buildInfo.PSObject.Properties['schemaVersion'] }
    if ($null -eq $schemaProperty) {
        throw "Unsupported build info schema in '$buildInfoPath': <missing>"
    }
    if ($schemaProperty.Value -ne 2) {
        throw "Unsupported build info schema in '$buildInfoPath': $($schemaProperty.Value)"
    }

    $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path $OneApiStacksFile
    $stack = Get-AilaOneApiStack -Config $config -Name $OneApiStack

    Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack
    Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack -RequireValues

    foreach ($artifactPath in @($proxy, $worker, $exe)) {
        if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
            throw "Staged release artifact not found: $artifactPath"
        }
    }
    foreach ($forbiddenRootExecutable in @('AilaWorker.exe', 'Aila.exe')) {
        $candidate = Join-Path $releaseRoot $forbiddenRootExecutable
        if (Test-Path -LiteralPath $candidate) {
            throw "Runtime executable must be staged below aila_runtime, not beside the proxy: $candidate"
        }
    }
    $rootRuntimeDlls = @(Get-ChildItem -LiteralPath $releaseRoot -File |
        Where-Object { Test-AilaOneApiRuntimeDllName -Name $_.Name })
    if ($rootRuntimeDlls.Count -ne 0) {
        throw "Found oneAPI runtime DLLs beside the proxy: $($rootRuntimeDlls.Name -join ', ')."
    }
    Assert-AilaStagedArtifactMetadata `
        -BuildInfo $buildInfo `
        -Role 'proxy' `
        -ExpectedRelativePath 'AilaShared.dll' `
        -ArtifactPath $proxy
    Assert-AilaStagedArtifactMetadata `
        -BuildInfo $buildInfo `
        -Role 'worker' `
        -ExpectedRelativePath 'aila_runtime/AilaWorker.exe' `
        -ArtifactPath $worker

    $oneApiProperty = $buildInfo.PSObject.Properties['oneApi']
    if ($null -eq $oneApiProperty -or $null -eq $oneApiProperty.Value) {
        throw "Build info '$buildInfoPath' is missing oneApi metadata."
    }
    $buildOneApi = $oneApiProperty.Value

    $stackEnvironment = Get-AilaOneApiStackEnvironment -Stack $stack
    Set-AilaProcessEnvironment -Environment $stackEnvironment
    $currentMetadata = Get-AilaOneApiStackMetadata -Stack $stack

    foreach ($propertyName in @('name', 'role', 'compilerVersion', 'dnnlVersion', 'tbbVersion')) {
        $recorded = [string](Get-RequiredOneApiMetadataValue -Metadata $buildOneApi -PropertyName $propertyName -Context 'Build info oneApi metadata')
        $current = [string](Get-RequiredOneApiMetadataValue -Metadata $currentMetadata -PropertyName $propertyName -Context 'Current oneAPI stack metadata')
        if (-not $recorded.Equals($current, [System.StringComparison]::Ordinal)) {
            throw "Build info oneApi metadata mismatch for '$propertyName': recorded '$recorded', current '$current'."
        }
    }

    foreach ($propertyName in @('compilerPath', 'dnnlRoot', 'tbbRoot', 'umfRoot')) {
        $recorded = [string](Get-RequiredOneApiMetadataValue -Metadata $buildOneApi -PropertyName $propertyName -Context 'Build info oneApi metadata')
        $current = [string](Get-RequiredOneApiMetadataValue -Metadata $currentMetadata -PropertyName $propertyName -Context 'Current oneAPI stack metadata')
        $recordedKey = Get-AilaWindowsPathKey -Path $recorded
        $currentKey = Get-AilaWindowsPathKey -Path $current
        if (-not $recordedKey.Equals($currentKey, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Build info oneApi metadata mismatch for '$propertyName': recorded '$recorded', current '$current'."
        }
    }

    $recordedSyclDll = [string](Get-RequiredOneApiMetadataValue -Metadata $buildOneApi -PropertyName 'expectedSyclDll' -Context 'Build info oneApi metadata')
    $currentSyclDll = [string](Get-RequiredOneApiMetadataValue -Metadata $currentMetadata -PropertyName 'expectedSyclDll' -Context 'Current oneAPI stack metadata')
    if (-not $recordedSyclDll.Equals($currentSyclDll, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Build info oneApi metadata mismatch for 'expectedSyclDll': recorded '$recordedSyclDll', current '$currentSyclDll'."
    }

    $recordedLegacy = Get-RequiredOneApiMetadataValue -Metadata $buildOneApi -PropertyName 'allowLegacyCompiler' -Context 'Build info oneApi metadata'
    $currentLegacy = Get-RequiredOneApiMetadataValue -Metadata $currentMetadata -PropertyName 'allowLegacyCompiler' -Context 'Current oneAPI stack metadata'
    if ($recordedLegacy -isnot [bool] -or $currentLegacy -isnot [bool] -or $recordedLegacy -ne $currentLegacy) {
        throw "Build info oneApi metadata mismatch for 'allowLegacyCompiler': recorded '$recordedLegacy', current '$currentLegacy'."
    }

    $stagedBuildInfoPath = Join-Path $releaseRoot 'build_info.json'
    if (-not (Test-Path -LiteralPath $stagedBuildInfoPath -PathType Leaf)) {
        throw "Staged build info file not found: $stagedBuildInfoPath"
    }
    $sourceMetadataHash = (Get-FileHash -LiteralPath $buildInfoPath -Algorithm SHA256).Hash
    $stagedMetadataHash = (Get-FileHash -LiteralPath $stagedBuildInfoPath -Algorithm SHA256).Hash
    if ($sourceMetadataHash -ne $stagedMetadataHash) {
        throw "Staged build_info.json does not match '$buildInfoPath'."
    }

    if ($null -eq $DependencyInspector) {
        $dumpbinPath = Get-AilaDumpbinPath -BuildDirPath $buildDirPath
        $DependencyInspector = {
            param([Parameter(Mandatory = $true)][string]$Executable)

            $output = (& $dumpbinPath /dependents $Executable 2>&1 | Out-String)
            $dumpbinExitCode = $LASTEXITCODE
            if ($dumpbinExitCode -ne 0) {
                $diagnostic = Get-AilaBoundedDiagnosticText -Text $output
                throw "dumpbin /dependents failed for '$Executable' with exit code $dumpbinExitCode. Output: $diagnostic"
            }
            return $output
        }
    }

    $inspection = Invoke-AilaStableDependencyInspection `
        -Executable $exe `
        -ExpectedSyclDll $currentSyclDll `
        -DependencyInspector $DependencyInspector
    $workerInspection = Invoke-AilaStableDependencyInspection `
        -Executable $worker `
        -ExpectedSyclDll $currentSyclDll `
        -DependencyInspector $DependencyInspector
    $proxyInspection = Invoke-AilaStableProxyDependencyInspection `
        -Proxy $proxy `
        -DependencyInspector $DependencyInspector

    $payload = [ordered]@{
        schemaVersion    = 1
        verifiedAtUtc    = (Get-Date).ToUniversalTime().ToString('o')
        buildDir         = $buildDirPath
        stack            = $buildOneApi
        dependencies     = @($inspection.dependencies)
        executableSha256 = $inspection.executableSha256
        artifacts        = @(
            [ordered]@{
                role         = 'cli'
                relativePath = 'aila_runtime/Aila.exe'
                dependencies = @($inspection.dependencies)
                sha256       = $inspection.executableSha256
            },
            [ordered]@{
                role         = 'worker'
                relativePath = 'aila_runtime/AilaWorker.exe'
                dependencies = @($workerInspection.dependencies)
                sha256       = $workerInspection.executableSha256
            },
            [ordered]@{
                role         = 'proxy'
                relativePath = 'AilaShared.dll'
                dependencies = @($proxyInspection.dependencies)
                sha256       = $proxyInspection.sha256
            }
        )
    }
    Write-AilaAtomicVerificationJson -RepoRoot $repoRoot -BuildDirPath $buildDirPath -Path $verificationPath -Data $payload

    Write-Host "Verification PASS: $OneApiStack worker -> $($workerInspection.dependencies[0]); proxy -> isolated"
}

Invoke-AilaOneApiBuildVerification @PSBoundParameters
