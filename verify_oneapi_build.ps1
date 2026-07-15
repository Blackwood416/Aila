param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$OneApiStack,
    [string]$OneApiStacksFile = 'perf\oneapi-stacks.json',
    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

$repoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$buildDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BuildDir
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

foreach ($propertyName in @('name', 'compilerVersion', 'dnnlVersion', 'tbbVersion')) {
    $recorded = [string](Get-RequiredOneApiMetadataValue -Metadata $buildOneApi -PropertyName $propertyName -Context "Build info oneApi metadata")
    $current = [string](Get-RequiredOneApiMetadataValue -Metadata $currentMetadata -PropertyName $propertyName -Context "Current oneAPI stack metadata")
    if (-not $recorded.Equals($current, [System.StringComparison]::Ordinal)) {
        throw "Build info oneApi metadata mismatch for '$propertyName': recorded '$recorded', current '$current'."
    }
}

foreach ($propertyName in @('compilerPath', 'dnnlRoot', 'tbbRoot', 'umfRoot')) {
    $recorded = [string](Get-RequiredOneApiMetadataValue -Metadata $buildOneApi -PropertyName $propertyName -Context "Build info oneApi metadata")
    $current = [string](Get-RequiredOneApiMetadataValue -Metadata $currentMetadata -PropertyName $propertyName -Context "Current oneAPI stack metadata")
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

$dumpbinCommand = Get-Command 'dumpbin.exe' -CommandType Application -ErrorAction Stop | Select-Object -First 1
$dependencies = (& $dumpbinCommand.Source /dependents $exe 2>&1 | Out-String)
$dumpbinExitCode = $LASTEXITCODE
if ($dumpbinExitCode -ne 0) {
    $diagnostic = Get-AilaBoundedDiagnosticText -Text $dependencies
    throw "dumpbin /dependents failed for '$exe' with exit code $dumpbinExitCode. Output: $diagnostic"
}

$syclDlls = @(
    [regex]::Matches($dependencies, '(?im)^\s+(sycl\d+\.dll)\s*$') |
        ForEach-Object { $_.Groups[1].Value.ToLowerInvariant() } |
        Sort-Object -Unique
)
$expectedSyclDll = $currentSyclDll.ToLowerInvariant()
if ($syclDlls.Count -ne 1 -or $syclDlls[0] -ne $expectedSyclDll) {
    $actual = if ($syclDlls.Count -eq 0) { '<none>' } else { [string]::Join(', ', $syclDlls) }
    throw "SYCL runtime ABI mismatch for '$exe': expected exactly '$expectedSyclDll', found '$actual'."
}

$verificationPath = if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    Join-Path $buildDirPath 'oneapi_verification.json'
}
else {
    Resolve-AilaPath -RepoRoot $repoRoot -Path $OutputPath
}
$payload = [ordered]@{
    schemaVersion    = 1
    verifiedAtUtc    = (Get-Date).ToUniversalTime().ToString('o')
    buildDir         = $buildDirPath
    stack            = $buildOneApi
    dependencies     = @($syclDlls)
    executableSha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
}
Write-AilaJsonFile -Path $verificationPath -Data $payload

Write-Host "Verification PASS: $OneApiStack -> $($syclDlls[0])"
