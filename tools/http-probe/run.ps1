[CmdletBinding()]
param([string]$BuildRoot, [int]$Port = 0)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $BuildRoot) { $BuildRoot = Join-Path $repoRoot 'build-refactor-http-probe' }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
$executable = Join-Path $BuildRoot 'build-probe/AilaHttpProbe.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw 'Build the probe with tools/http-probe/build.ps1 first' }
if ($Port -eq 0) {
    $reservation = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    try { $reservation.Start(); $Port = $reservation.LocalEndpoint.Port }
    finally { $reservation.Stop() }
}
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Invalid port' }
$runName = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runRoot = Join-Path $BuildRoot "runs/$runName"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$serverReport = Join-Path $runRoot 'server.json'
$clientReport = Join-Path $runRoot 'client.json'
$processReport = Join-Path $runRoot 'process.json'
$probeProcess = $null
$forcedCleanup = $false
$runPassed = $false
$runTimer = [Diagnostics.Stopwatch]::StartNew()
try {
    $probeProcess = Start-Process -FilePath $executable -ArgumentList @('--port', "$Port", '--report', ('"' + $serverReport + '"')) -WorkingDirectory $runRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runRoot 'server.stdout.txt') -RedirectStandardError (Join-Path $runRoot 'server.stderr.txt')
    @{ pid = $probeProcess.Id; host = '127.0.0.1'; port = $Port; executable = $executable; started = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'launch.json') -Encoding utf8
    Write-Output "Probe PID $($probeProcess.Id), 127.0.0.1:$Port; artifacts: $runRoot"
    & python.exe (Join-Path $PSScriptRoot 'client.py') --port $Port --server-report $serverReport --report $clientReport
    if ($LASTEXITCODE -ne 0) { throw "Socket acceptance failed; see $clientReport" }
    if (-not $probeProcess.WaitForExit(4000)) { throw 'Server did not exit within four seconds after acceptance checks' }
    if ($probeProcess.ExitCode -ne 0) { throw "Server exited with code $($probeProcess.ExitCode)" }
    $runPassed = $true
} finally {
    if ($probeProcess) {
        if (-not $probeProcess.HasExited) {
            try { Invoke-WebRequest -Uri "http://127.0.0.1:$Port/shutdown" -Method Post -TimeoutSec 2 | Out-Null } catch {}
            if (-not $probeProcess.WaitForExit(3000)) {
                $forcedCleanup = $true
                Stop-Process -Id $probeProcess.Id -Force
                if (-not $probeProcess.WaitForExit(2000)) { throw 'Unable to stop the known probe process' }
            }
        }
        $result = @{ pid = $probeProcess.Id; exited = $probeProcess.HasExited; exit_code = $probeProcess.ExitCode; forced_cleanup = $forcedCleanup; passed = ($runPassed -and -not $forcedCleanup); elapsed_seconds = $runTimer.Elapsed.TotalSeconds; run_directory = $runRoot }
        $result | ConvertTo-Json | Set-Content -LiteralPath $processReport -Encoding utf8
        $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $BuildRoot 'latest-run.json') -Encoding utf8
        $probeProcess.Dispose()
    }
}
Write-Output "All socket scenarios passed; server stopped. Results: $runRoot"
