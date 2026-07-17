param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$ReleaseRoot,
    [string]$RepoRoot = ''
)

$ErrorActionPreference = 'Stop'

$repoRootPath = if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
}
else {
    [System.IO.Path]::GetFullPath($RepoRoot)
}
. (Join-Path $repoRootPath 'perf\PerfCommon.ps1')

$buildDirPath = [System.IO.Path]::GetFullPath($BuildDir)
$releaseRootPath = [System.IO.Path]::GetFullPath($ReleaseRoot)
if (-not (Test-AilaPathWithinRoot -Path $buildDirPath -Root $repoRootPath)) {
    throw "Build directory is outside repository '$repoRootPath': $buildDirPath"
}
if (-not (Test-AilaPathWithinRoot -Path $releaseRootPath -Root $buildDirPath)) {
    throw "Release root is outside build directory '$buildDirPath': $releaseRootPath"
}

$buildInfoPath = Join-Path $buildDirPath 'build_info.json'
$buildInfo = Read-AilaJsonFile -Path $buildInfoPath
if ($buildInfo.schemaVersion -ne 2) {
    throw "Unsupported build info schema in '$buildInfoPath': $($buildInfo.schemaVersion)"
}
foreach ($propertyName in @('build', 'oneApi')) {
    $property = $buildInfo.PSObject.Properties[$propertyName]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "Build info '$buildInfoPath' is missing required provenance '$propertyName'."
    }
}

$artifactDefinitions = @(
    [ordered]@{ role = 'proxy'; relativePath = 'AilaShared.dll' },
    [ordered]@{ role = 'worker'; relativePath = 'aila_runtime/AilaWorker.exe' },
    [ordered]@{ role = 'cli'; relativePath = 'aila_runtime/Aila.exe' }
)
$artifacts = @($artifactDefinitions | ForEach-Object {
    $artifactPath = Join-Path $releaseRootPath ($_.relativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
        throw "Required staged artifact not found while refreshing metadata: $artifactPath"
    }
    [ordered]@{
        role         = $_.role
        relativePath = $_.relativePath
        sha256       = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash
    }
})

$gitInfo = Get-AilaGitInfo -RepoRoot $repoRootPath
if ($LASTEXITCODE -ne 0 -or
    [string]::IsNullOrWhiteSpace([string]$gitInfo.fullCommit) -or
    [string]::IsNullOrWhiteSpace([string]$gitInfo.shortCommit) -or
    [string]::IsNullOrWhiteSpace([string]$gitInfo.branch)) {
    throw "Unable to resolve git provenance for '$repoRootPath'."
}

$buildInfo.generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
$buildInfo | Add-Member -NotePropertyName git -NotePropertyValue ([ordered]@{
    shortCommit = $gitInfo.shortCommit
    fullCommit  = $gitInfo.fullCommit
    branch      = $gitInfo.branch
}) -Force
$buildInfo | Add-Member -NotePropertyName artifacts -NotePropertyValue $artifacts -Force

$sourceTempPath = Join-Path $buildDirPath ".build_info.$([guid]::NewGuid().ToString('N')).tmp"
$stagedBuildInfoPath = Join-Path $releaseRootPath 'build_info.json'
$stagedTempPath = Join-Path $releaseRootPath ".build_info.$([guid]::NewGuid().ToString('N')).tmp"
try {
    Write-AilaJsonFile -Path $sourceTempPath -Data $buildInfo
    [System.IO.File]::Move($sourceTempPath, $buildInfoPath, $true)
    Copy-Item -LiteralPath $buildInfoPath -Destination $stagedTempPath
    [System.IO.File]::Move($stagedTempPath, $stagedBuildInfoPath, $true)
}
finally {
    foreach ($tempPath in @($sourceTempPath, $stagedTempPath)) {
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force
        }
    }
}

Write-Host "Release build metadata refreshed: $stagedBuildInfoPath"
