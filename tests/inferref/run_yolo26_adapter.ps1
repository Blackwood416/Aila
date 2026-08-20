param(
    [Parameter(Mandatory = $true, Position = 0)] [string] $Testcase,
    [Parameter(Mandatory = $true, Position = 1)] [string] $Output
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildDir = if ($env:AILA_INFERREF_BUILD_DIR) {
    [System.IO.Path]::GetFullPath($env:AILA_INFERREF_BUILD_DIR, $repoRoot)
} else {
    Join-Path $repoRoot "build"
}
$adapter = Join-Path $buildDir "AilaYolo26InferRefAdapter.exe"
if (-not (Test-Path -LiteralPath $adapter -PathType Leaf)) {
    Write-Error "InferRef adapter executable not found: $adapter"
    exit 2
}

& $adapter $Testcase --output $Output
exit $LASTEXITCODE
