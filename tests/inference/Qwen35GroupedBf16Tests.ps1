param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$fixtureDirectory = Join-Path $OutputRoot ('q35-grouped-bf16-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixtureDirectory | Out-Null
$referencePath = Join-Path $fixtureDirectory 'host-reference.txt'
foreach ($fixtureMode in @('host', 'gpu')) {
    & $Executable --mode $fixtureMode --reference $referencePath 2>&1 |
        Tee-Object -FilePath (Join-Path $fixtureDirectory ($fixtureMode + '.log'))
    if ($LASTEXITCODE -ne 0) {
        throw "Grouped BF16 $fixtureMode fixture failed with exit code $LASTEXITCODE. Evidence: $fixtureDirectory"
    }
}
Write-Output "Grouped BF16 evidence: $fixtureDirectory"
