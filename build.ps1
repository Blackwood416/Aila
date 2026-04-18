param(
    [string]$BuildDir = 'build',
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug', 'MinSizeRel')]
    [string]$Config = 'Release',
    [switch]$Clean,
    [int]$Jobs = 36
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

$repoRoot = Get-AilaRepoRoot
$buildDirPath = Resolve-AilaPath -RepoRoot $repoRoot -Path $BuildDir
$buildDirPath = Assert-AilaPathWithinRepo -RepoRoot $repoRoot -CandidatePath $buildDirPath
$gitInfo = Get-AilaGitInfo -RepoRoot $repoRoot

if ($Clean -and (Test-Path -LiteralPath $buildDirPath)) {
    if ($buildDirPath -eq $repoRoot) {
        throw 'Refusing to clean the repo root.'
    }
    Remove-Item -LiteralPath $buildDirPath -Recurse -Force
}

Ensure-AilaDirectory -Path $buildDirPath
Initialize-AilaOneApiEnvironment

Push-Location $repoRoot
try {
    & cmake -S $repoRoot -B $buildDirPath -G Ninja "-DCMAKE_BUILD_TYPE=$Config"
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE."
    }

    $buildMeta = Get-AilaBuildMetadata -BuildDir $buildDirPath

    Write-Host ":: build info ::" -ForegroundColor Cyan
    Write-Host ("   repo root    : {0}" -f $repoRoot)
    Write-Host ("   git commit   : {0} ({1})" -f $gitInfo.shortCommit, $gitInfo.branch)
    Write-Host ("   build dir    : {0}" -f $buildDirPath)
    Write-Host ("   build type   : {0}" -f $buildMeta.buildType)
    Write-Host ("   compiler     : {0}" -f $buildMeta.compiler)
    Write-Host ("   generator    : {0}" -f $buildMeta.generator)

    & cmake --build $buildDirPath --config $Config --parallel $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$buildInfoPath = Join-Path $buildDirPath 'build_info.json'
Write-AilaJsonFile -Path $buildInfoPath -Data ([ordered]@{
    schemaVersion  = 1
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    git            = [ordered]@{
        shortCommit = $gitInfo.shortCommit
        fullCommit  = $gitInfo.fullCommit
        branch      = $gitInfo.branch
    }
    build          = [ordered]@{
        buildDir  = $buildDirPath
        buildType = $Config
        compiler  = (Get-AilaBuildMetadata -BuildDir $buildDirPath).compiler
        generator = (Get-AilaBuildMetadata -BuildDir $buildDirPath).generator
    }
})

Write-Host (":: build metadata written to {0} ::" -f $buildInfoPath) -ForegroundColor Green
