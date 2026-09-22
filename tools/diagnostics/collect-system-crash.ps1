[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../../tmp/refactor/stability-20260908/admin-collection'),
    [datetime]$Since = '2026-09-08T00:00:00'
)

$ErrorActionPreference = 'Stop'
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this read-only collector in an Administrator PowerShell. It copies dumps and diagnostic logs; it does not change access controls, drivers, registry or GPU settings.'
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$files = @()
foreach ($dumpRoot in @('C:/Windows/Minidump', 'C:/Windows/LiveKernelReports')) {
    if (-not (Test-Path -LiteralPath $dumpRoot)) { continue }
    foreach ($dump in Get-ChildItem -LiteralPath $dumpRoot -File -Recurse -Filter '*.dmp') {
        if ($dump.LastWriteTime -lt $Since) { continue }
        $relative = $dump.FullName.Substring($dumpRoot.Length).TrimStart('\', '/')
        $destination = Join-Path (Join-Path $outputPath (Split-Path $dumpRoot -Leaf)) $relative
        New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
        if (Test-Path -LiteralPath $destination) {
            throw "Refusing to overwrite existing evidence: $destination. Use a new OutputDirectory."
        }
        Copy-Item -LiteralPath $dump.FullName -Destination $destination
        $files += [ordered]@{source=$dump.FullName;copy=$destination;bytes=$dump.Length;
            lastWriteUtc=$dump.LastWriteTimeUtc.ToString('o');sha256=(Get-FileHash -LiteralPath $destination).Hash}
    }
}
$events = @(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$Since} |
    Where-Object { $_.ProviderName -in @('Microsoft-Windows-WER-SystemErrorReporting',
        'Microsoft-Windows-Kernel-Power','Display','Microsoft-Windows-WHEA-Logger',
        'Microsoft-Windows-Resource-Exhaustion-Detector') -or $_.Id -in @(6008,161,162) } |
    ForEach-Object { [ordered]@{time=$_.TimeCreated.ToString('o');provider=$_.ProviderName;
        id=$_.Id;record=$_.RecordId;message=$_.Message;xml=$_.ToXml()} })
$report = [ordered]@{
    collectedAtUtc=[DateTime]::UtcNow.ToString('o')
    since=$Since.ToString('o')
    system=Get-CimInstance Win32_OperatingSystem | Select-Object LastBootUpTime,Version,BuildNumber
    graphics=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate)
    dumps=$files
    events=$events
}
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $outputPath 'collection.json') -Encoding utf8
Write-Output "Crash evidence copied locally to $outputPath. No GPU work was started and no system setting was changed."
