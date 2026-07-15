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

    $exe = Join-Path $buildDirPath 'Aila.exe'
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw "Aila executable not found: $exe"
    }

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

    if ($null -eq $DependencyInspector) {
        $DependencyInspector = {
            param([Parameter(Mandatory = $true)][string]$Executable)

            $dumpbinCommand = Get-Command 'dumpbin.exe' -CommandType Application -ErrorAction Stop | Select-Object -First 1
            $output = (& $dumpbinCommand.Source /dependents $Executable 2>&1 | Out-String)
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

    $payload = [ordered]@{
        schemaVersion    = 1
        verifiedAtUtc    = (Get-Date).ToUniversalTime().ToString('o')
        buildDir         = $buildDirPath
        stack            = $buildOneApi
        dependencies     = @($inspection.dependencies)
        executableSha256 = $inspection.executableSha256
    }
    Write-AilaAtomicVerificationJson -RepoRoot $repoRoot -BuildDirPath $buildDirPath -Path $verificationPath -Data $payload

    Write-Host "Verification PASS: $OneApiStack -> $($inspection.dependencies[0])"
}

Invoke-AilaOneApiBuildVerification @PSBoundParameters
