param(
    [string]$BuildDir = 'build',
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug', 'MinSizeRel')]
    [string]$Config = 'Release',
    [string]$OneApiStack = 'oneapi-2026.1',
    [string]$OneApiStacksFile = 'perf\oneapi-stacks.json',
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
$stackConfig = Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path $OneApiStacksFile
$stack = Get-AilaOneApiStack -Config $stackConfig -Name $OneApiStack
Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack
Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack
$stackEnv = Get-AilaOneApiStackEnvironment -Stack $stack
Set-AilaProcessEnvironment -Environment $stackEnv
$stackMeta = Get-AilaOneApiStackMetadata -Stack $stack
$stackCompilerPath = Join-Path $stack.compilerRoot 'bin\icx-cl.exe'
$stackDnnlCMakeDir = Join-Path $stack.dnnlRoot 'lib\cmake\dnnl'
$stackTbbCMakeDir = Join-Path $stack.tbbRoot 'lib\cmake\tbb'
$stackLegacyOption = if ($stack.allowLegacyCompiler) { 'ON' } else { 'OFF' }

Push-Location $repoRoot
try {
    $cmakeArgs = @(
        '-S', $repoRoot,
        '-B', $buildDirPath,
        '-G', 'Ninja',
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DCMAKE_CXX_COMPILER=$stackCompilerPath",
        "-Ddnnl_DIR=$stackDnnlCMakeDir",
        "-DTBB_DIR=$stackTbbCMakeDir",
        "-DAILA_ALLOW_LEGACY_ONEAPI_BASELINE=$stackLegacyOption"
    )
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE."
    }

    Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack -RequireValues
    $buildMeta = Get-AilaBuildMetadata -BuildDir $buildDirPath

    Write-Host ":: build info ::" -ForegroundColor Cyan
    Write-Host ("   repo root    : {0}" -f $repoRoot)
    Write-Host ("   git commit   : {0} ({1})" -f $gitInfo.shortCommit, $gitInfo.branch)
    Write-Host ("   build dir    : {0}" -f $buildDirPath)
    Write-Host ("   build type   : {0}" -f $buildMeta.buildType)
    Write-Host ("   oneAPI stack : {0} ({1})" -f $stackMeta.name, $stackMeta.role)
    Write-Host ("   compiler     : {0}" -f $buildMeta.compiler)
    Write-Host ("   generator    : {0}" -f $buildMeta.generator)

    & cmake --build $buildDirPath --config $Config --parallel $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed with exit code $LASTEXITCODE."
    }
    Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDirPath -Stack $stack -RequireValues
}
finally {
    Pop-Location
}

$buildInfoPath = Join-Path $buildDirPath 'build_info.json'
$artifactDefinitions = @(
    [ordered]@{
        role         = 'proxy'
        relativePath = 'AilaShared.dll'
        sourcePath   = (Join-Path $buildDirPath 'AilaShared.dll')
    },
    [ordered]@{
        role         = 'worker'
        relativePath = 'aila_runtime/AilaWorker.exe'
        sourcePath   = (Join-Path $buildDirPath 'AilaWorker.exe')
    },
    [ordered]@{
        role         = 'cli'
        relativePath = 'aila_runtime/Aila.exe'
        sourcePath   = (Join-Path $buildDirPath 'Aila.exe')
    }
)
$artifacts = @($artifactDefinitions | ForEach-Object {
    if (-not (Test-Path -LiteralPath $_.sourcePath -PathType Leaf)) {
        throw "Required build artifact not found while writing metadata: $($_.sourcePath)"
    }
    [ordered]@{
        role         = $_.role
        relativePath = $_.relativePath
        sha256       = (Get-FileHash -LiteralPath $_.sourcePath -Algorithm SHA256).Hash
    }
})
Write-AilaJsonFile -Path $buildInfoPath -Data ([ordered]@{
    schemaVersion  = 2
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    git            = [ordered]@{
        shortCommit = $gitInfo.shortCommit
        fullCommit  = $gitInfo.fullCommit
        branch      = $gitInfo.branch
    }
    build          = [ordered]@{
        buildDir  = $buildDirPath
        buildType = $Config
        compiler  = $buildMeta.compiler
        generator = $buildMeta.generator
    }
    oneApi          = $stackMeta
    artifacts       = $artifacts
})

Write-Host (":: build metadata written to {0} ::" -f $buildInfoPath) -ForegroundColor Green
