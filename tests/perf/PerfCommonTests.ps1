$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\..\perf\PerfCommon.ps1')

$script:TestFailures = [System.Collections.Generic.List[string]]::new()

function Assert-Equal($Expected, $Actual, [string]$Message) {
    if ($Expected -ne $Actual) {
        throw "$Message expected='$Expected' actual='$Actual'"
    }
}

function Assert-IsNull {
    param(
        [AllowNull()]$Actual,
        [Parameter(Mandatory = $true)][string]$Message
    )

    if ($null -ne $Actual) {
        throw "$Message expected a null value but received type '$($Actual.GetType().FullName)' with value '$Actual'"
    }
}

function Restore-ProcessEnvironmentVariable {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [AllowNull()]$Value
    )

    if ($null -eq $Value) {
        [System.Environment]::SetEnvironmentVariable($Name, [System.Management.Automation.Language.NullString]::Value, 'Process')
        return
    }
    [System.Environment]::SetEnvironmentVariable($Name, [string]$Value, 'Process')
}

function Get-ProcessEnvironmentSnapshot {
    $snapshot = @{}
    [System.Environment]::GetEnvironmentVariables('Process').GetEnumerator() | ForEach-Object {
        $snapshot[[string]$_.Key] = [string]$_.Value
    }
    return $snapshot
}

function Restore-ProcessEnvironmentSnapshot {
    param([Parameter(Mandatory = $true)][hashtable]$Snapshot)

    $currentKeys = @([System.Environment]::GetEnvironmentVariables('Process').Keys | ForEach-Object { [string]$_ })
    foreach ($key in $currentKeys) {
        if (-not $Snapshot.ContainsKey($key)) {
            Restore-ProcessEnvironmentVariable -Name $key -Value $null
        }
    }
    foreach ($entry in $Snapshot.GetEnumerator()) {
        Restore-ProcessEnvironmentVariable -Name $entry.Key -Value $entry.Value
    }
}

function Invoke-ChildPowerShellWithTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][int]$TimeoutMs
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = Join-Path $PSHOME 'pwsh.exe'
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    foreach ($argument in @('-NoProfile', '-File', $ScriptPath) + $ArgumentList) {
        $startInfo.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start child PowerShell process: $($startInfo.FileName)"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit($TimeoutMs)
    if (-not $completed) {
        $process.Kill($true)
        $process.WaitForExit()
    }

    return [pscustomobject]@{
        completed = $completed
        exitCode = if ($completed) { $process.ExitCode } else { $null }
        stdout = $stdoutTask.GetAwaiter().GetResult()
        stderr = $stderrTask.GetAwaiter().GetResult()
    }
}

function Assert-Throws([scriptblock]$Action, [string]$ExpectedMessageFragment, [string]$Message) {
    $caught = $null
    try {
        $null = & $Action
    }
    catch {
        $caught = $_
    }

    if ($null -eq $caught) {
        throw "$Message expected an exception containing '$ExpectedMessageFragment'"
    }

    $actual = [string]$caught.Exception.Message
    if (-not $actual.Contains($ExpectedMessageFragment)) {
        throw "$Message expected message containing='$ExpectedMessageFragment' actual='$actual'"
    }
}

function Invoke-Test([string]$Name, [scriptblock]$Test) {
    try {
        & $Test
    }
    catch {
        $script:TestFailures.Add("${Name}: $($_.Exception.Message)")
    }
}

function Get-NormalizedTestPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path.Trim().Trim('"')).TrimEnd([char[]]@('\', '/'))
}

function Assert-PathContainsSegment([string]$PathValue, [string]$Expected, [string]$Message) {
    $expectedNormalized = Get-NormalizedTestPath -Path $Expected
    $segments = @($PathValue -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object {
        Get-NormalizedTestPath -Path $_
    })
    if ($segments -notcontains $expectedNormalized) {
        throw "$Message expected PATH segment='$expectedNormalized'"
    }
}

function Assert-PathSegmentsUnique([string]$PathValue, [string]$Message) {
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($segment in ($PathValue -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        $normalized = Get-NormalizedTestPath -Path $segment
        if (-not $seen.Add($normalized)) {
            throw "$Message duplicate PATH segment='$normalized'"
        }
    }
}

function Assert-PathContainsRoot([string]$PathValue, [string]$ExpectedRoot, [string]$Message) {
    foreach ($segment in ($PathValue -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        if (Test-AilaPathWithinRoot -Path $segment -Root $ExpectedRoot) {
            return
        }
    }
    throw "$Message expected a path under '$ExpectedRoot'"
}

function Assert-PathExcludesRoot([string]$PathValue, [string]$ExcludedRoot, [string]$Message) {
    foreach ($segment in ($PathValue -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        if (Test-AilaPathWithinRoot -Path $segment -Root $ExcludedRoot) {
            throw "$Message contains path from excluded root '$ExcludedRoot': '$segment'"
        }
    }
}

function New-TestStack([string]$Root) {
    return [pscustomobject]@{
        role                    = 'test'
        compilerRoot            = $Root
        dnnlRoot                = $Root
        tbbRoot                 = $Root
        umfRoot                 = $Root
        expectedCompilerVersion = '0.0.0'
        expectedDnnlVersion     = '0.0.0'
        expectedSyclDll         = 'sycl-test.dll'
        allowLegacyCompiler     = $false
    }
}

function New-VerifierBuildFixture {
    param(
        [Parameter(Mandatory = $true)][string]$SourceBuildDir,
        [Parameter(Mandatory = $true)][string]$DestinationBuildDir
    )

    New-Item -ItemType Directory -Path $DestinationBuildDir -Force | Out-Null
    foreach ($fileName in @('Aila.exe', 'build_info.json', 'CMakeCache.txt')) {
        Copy-Item -LiteralPath (Join-Path $SourceBuildDir $fileName) -Destination (Join-Path $DestinationBuildDir $fileName)
    }
    return $DestinationBuildDir
}

function New-AccuracyBuildFixture {
    param(
        [Parameter(Mandatory = $true)][string]$SourceBuildDir,
        [Parameter(Mandatory = $true)][string]$DestinationBuildDir
    )

    New-Item -ItemType Directory -Path $DestinationBuildDir -Force | Out-Null
    foreach ($fileName in @('build_info.json', 'CMakeCache.txt')) {
        Copy-Item -LiteralPath (Join-Path $SourceBuildDir $fileName) -Destination (Join-Path $DestinationBuildDir $fileName)
    }

    $sourcePath = Join-Path $DestinationBuildDir 'FakeAila.cs'
    Set-Content -LiteralPath $sourcePath -Encoding UTF8 -Value @'
using System;

public static class FakeAila
{
    public static int Main(string[] args)
    {
        Console.Error.WriteLine("[fake diagnostic]");
        Console.WriteLine("Hello, WORLD!");
        return 0;
    }
}
'@
    $compiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
        throw "C# fixture compiler not found: $compiler"
    }
    $exePath = Join-Path $DestinationBuildDir 'Aila.exe'
    $compilerOutput = (& $compiler /nologo /target:exe "/out:$exePath" $sourcePath 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
        throw "Failed to compile fake Aila fixture: $compilerOutput"
    }
    Remove-Item -LiteralPath $sourcePath -Force
    return $DestinationBuildDir
}

$repoRoot = Get-AilaRepoRoot
$repoScratch = Join-Path $repoRoot "build-verifier-tests-$([guid]::NewGuid().ToString('N'))"
if (-not (Test-AilaPathWithinRoot -Path $repoScratch -Root $repoRoot)) {
    throw "Refusing to create verifier test files outside the repository: $repoScratch"
}
New-Item -ItemType Directory -Path $repoScratch -Force | Out-Null
$accuracyScratch = Join-Path $repoRoot "tmp\perf\accuracy-runner-tests-$([guid]::NewGuid().ToString('N'))"
if (-not (Test-AilaPathWithinRoot -Path $accuracyScratch -Root (Join-Path $repoRoot 'tmp'))) {
    throw "Refusing to create accuracy test files outside repository tmp: $accuracyScratch"
}
New-Item -ItemType Directory -Path $accuracyScratch -Force | Out-Null
$tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\') + '\'
$tempRoot = [System.IO.Path]::GetFullPath((Join-Path $tempBase "Aila-PerfCommonTests-$([guid]::NewGuid().ToString('N'))"))
if (-not $tempRoot.StartsWith($tempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to create test files outside the temporary directory: $tempRoot"
}

New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$validRoot = Join-Path $tempRoot 'root'
New-Item -ItemType Directory -Path $validRoot -Force | Out-Null
$ordinaryFile = Join-Path $tempRoot 'not-a-root.txt'
Set-Content -LiteralPath $ordinaryFile -Value 'not a directory' -Encoding UTF8
$unsupportedSchemaPath = Join-Path $tempRoot 'unsupported-schema.json'
Set-Content -LiteralPath $unsupportedSchemaPath -Value '{"schemaVersion":2,"stacks":{}}' -Encoding UTF8
$missingSchemaPath = Join-Path $tempRoot 'missing-schema.json'
Set-Content -LiteralPath $missingSchemaPath -Value '{"stacks":{}}' -Encoding UTF8
$failingBatchPath = Join-Path $tempRoot 'fail-with-37.bat'
Set-Content -LiteralPath $failingBatchPath -Value "@echo off`r`nexit /b 37" -Encoding ASCII
$diagnosticBatchPath = Join-Path $tempRoot 'fail-with-diagnostic.bat'
Set-Content -LiteralPath $diagnosticBatchPath -Value @(
    '@echo off'
    'echo diagnostic-from-stdout'
    '>&2 echo diagnostic-from-batch'
    'exit /b 23'
) -Encoding ASCII
$saturationBatchPath = Join-Path $tempRoot 'saturate-batch-pipes.bat'
Set-Content -LiteralPath $saturationBatchPath -Value @(
    '@echo off'
    'for /L %%i in (1,1,8000) do ('
    '  echo saturation-stdout-%%i-abcdefghijklmnopqrstuvwxyz-0123456789'
    '  >&2 echo saturation-stderr-%%i-abcdefghijklmnopqrstuvwxyz-0123456789'
    ')'
    'set CMPLR_ROOT=C:\saturation-compiler'
    'exit /b 0'
) -Encoding ASCII
$saturationChildPath = Join-Path $tempRoot 'invoke-saturation-import.ps1'
Set-Content -LiteralPath $saturationChildPath -Value @'
param(
    [Parameter(Mandatory = $true)][string]$PerfCommonPath,
    [Parameter(Mandatory = $true)][string]$BatchPath
)
$ErrorActionPreference = 'Stop'
. $PerfCommonPath
$environment = Import-AilaBatchEnvironment -Scripts @($BatchPath)
if ([string]::IsNullOrWhiteSpace([string]$environment.PATH) -or
    [string]::IsNullOrWhiteSpace([string]$environment.CMPLR_ROOT)) {
    throw 'Saturation import did not return required PATH and CMPLR_ROOT values.'
}
Write-Output 'SATURATION_IMPORT_PASS'
'@ -Encoding UTF8
$partialBatchPath = Join-Path $tempRoot 'missing-compiler-root.bat'
Set-Content -LiteralPath $partialBatchPath -Value "@echo off`r`nset CMPLR_ROOT=`r`nexit /b 0" -Encoding ASCII
$missingPathBatchPath = Join-Path $tempRoot 'missing-path.bat'
Set-Content -LiteralPath $missingPathBatchPath -Value "@echo off`r`nset PATH=`r`nset CMPLR_ROOT=C:\compiler`r`nexit /b 0" -Encoding ASCII
$invalidDnnlRoot = Join-Path $tempRoot 'invalid-dnnl'
$invalidDnnlVersionDir = Join-Path $invalidDnnlRoot 'lib\cmake\dnnl'
New-Item -ItemType Directory -Path $invalidDnnlVersionDir -Force | Out-Null
$invalidDnnlVersionPath = Join-Path $invalidDnnlVersionDir 'dnnl-config-version.cmake'
Set-Content -LiteralPath $invalidDnnlVersionPath -Value 'set(PACKAGE_VERSION_BROKEN "0.0.0")' -Encoding UTF8
$failingCompilerRoot = Join-Path $tempRoot 'failing-compiler'
$failingCompilerBin = Join-Path $failingCompilerRoot 'bin'
New-Item -ItemType Directory -Path $failingCompilerBin -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $env:SystemRoot 'System32\where.exe') -Destination (Join-Path $failingCompilerBin 'icx-cl.exe')

try {
    Invoke-Test 'normalizes Aila text across punctuation case and whitespace' {
        Assert-Equal 'hello world 中文 测试' (Normalize-AilaText -Text "  HELLO,`tWorld!  中文—测试。 ") 'normalized text'
        Assert-Equal '' (Normalize-AilaText -Text '') 'empty normalized text'
    }

    Invoke-Test 'parses typed ASR transcript and metrics' {
        $asr = Parse-AilaAsrOutput -OutputText "这是一个测试`n[Language] Chinese`n[Audio: 1.0s, Latency: 500ms, Speed: 2.0x, 8 tok/s]"
        Assert-Equal '这是一个测试' $asr.text 'ASR text'
        Assert-Equal 500.0 $asr.latencyMs 'ASR latency'
        Assert-Equal 'Chinese' $asr.language 'ASR language'
        Assert-Equal 1.0 $asr.audioSeconds 'ASR audio duration'
        Assert-Equal 2.0 $asr.speed 'ASR speed'
        Assert-Equal 8.0 $asr.tokensPerSecond 'ASR token rate'
    }

    Invoke-Test 'preserves multiline ASR transcript while excluding bracketed diagnostics' {
        $asr = Parse-AilaAsrOutput -OutputText "First line.`n第二行！`n[Language] Chinese`n[Audio: 2.5s, Latency: 1250.25ms, Speed: 2.0x, 12.5 tok/s]"
        Assert-Equal "First line.`n第二行！" $asr.text 'multiline ASR text'
        Assert-Equal 1250.25 $asr.latencyMs 'fractional ASR latency'
    }

    Invoke-Test 'rejects ASR output with a missing metric line clearly' {
        Assert-Throws {
            Parse-AilaAsrOutput -OutputText "transcript`n[Language] English"
        } 'Failed to parse ASR metrics' 'missing ASR metrics'
    }

    Invoke-Test 'rejects ASR output with invalid metric values clearly' {
        Assert-Throws {
            Parse-AilaAsrOutput -OutputText "transcript`n[Language] English`n[Audio: 1.0s, Latency: nope, Speed: 2.0x, 8 tok/s]"
        } 'Failed to parse ASR metrics' 'invalid ASR metrics'
    }

    Invoke-Test 'parses typed alignment rows including escaped and Unicode text' {
        $aligned = Parse-AilaAlignmentOutput -OutputText '  "这"  0ms - 80ms'
        Assert-Equal 1 $aligned.Count 'alignment count'
        Assert-Equal 80 $aligned[0].endMs 'alignment end'

        $rows = Parse-AilaAlignmentOutput -OutputText @'
  "a \"quoted\" 字"  80ms - 160ms
"第二行" 160ms - 320ms
'@
        Assert-Equal 2 $rows.Count 'multiple alignment count'
        Assert-Equal 'a \"quoted\" 字' $rows[0].text 'escaped alignment text'
        Assert-Equal 80 $rows[0].startMs 'first alignment start'
        Assert-Equal 320 $rows[1].endMs 'second alignment end'
        Assert-Equal 'System.Int32' $rows[0].startMs.GetType().FullName 'alignment start type'
    }

    Invoke-Test 'returns an empty array when alignment output has no matching rows' {
        $aligned = @(Parse-AilaAlignmentOutput -OutputText "no timestamps here`n[diagnostic]")
        Assert-Equal 0 $aligned.Count 'empty alignment count'
    }

    $accuracySourceBuild = Join-Path $repoRoot 'build-oneapi-2025.3'
    $accuracyBuild = New-AccuracyBuildFixture `
        -SourceBuildDir $accuracySourceBuild `
        -DestinationBuildDir (Join-Path $repoScratch 'accuracy-build')
    $accuracyModel = Join-Path $repoScratch 'accuracy-model'
    New-Item -ItemType Directory -Path $accuracyModel -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $accuracyModel 'config.json') -Value '{"model_type":"fake"}' -Encoding UTF8
    $accuracyInputOne = Join-Path $repoScratch 'accuracy-input-one.json'
    $accuracyInputTwo = Join-Path $repoScratch 'accuracy-input-two.json'
    Set-Content -LiteralPath $accuracyInputOne -Value '{"messages":[{"role":"user","content":"one"}]}' -Encoding UTF8
    Set-Content -LiteralPath $accuracyInputTwo -Value '{"messages":[{"role":"user","content":"two"}]}' -Encoding UTF8
    $accuracyCasesPath = Join-Path $repoScratch 'accuracy-cases.json'
    Write-AilaJsonFile -Path $accuracyCasesPath -Data ([ordered]@{
        schemaVersion = 1
        cases = @(
            [ordered]@{ name = 'case_one'; kind = 'chat'; model = $accuracyModel; input = $accuracyInputOne; maxTokens = 4; mode = 'greedy'; expectRegex = 'hello' },
            [ordered]@{ name = 'case_two'; kind = 'chat'; model = $accuracyModel; input = $accuracyInputTwo; maxTokens = 8; mode = 'greedy' }
        )
    })
    $regressPath = Join-Path $repoRoot 'regress.ps1'

    Invoke-Test 'accuracy runner preserves sentinel on wrong stack preflight failure' {
        $outputDir = Join-Path $accuracyScratch 'wrong-stack'
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        $sentinelPath = Join-Path $outputDir 'accuracy.json'
        Set-Content -LiteralPath $sentinelPath -Value 'wrong-stack-sentinel' -Encoding UTF8
        $result = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $regressPath `
            -ArgumentList @('-BuildDir', $accuracyBuild, '-OneApiStack', 'oneapi-2026.1', '-CasesFile', $accuracyCasesPath, '-OutputDir', $outputDir) `
            -TimeoutMs 30000
        Assert-Equal $true $result.completed 'wrong stack runner completion'
        if ($result.exitCode -eq 0) {
            throw 'Wrong stack runner unexpectedly succeeded.'
        }
        Assert-Equal $true (($result.stderr + $result.stdout).Contains("belongs to 'oneapi-2025.3'")) 'wrong stack diagnostic'
        Assert-Equal 'wrong-stack-sentinel' (Get-Content -LiteralPath $sentinelPath -Raw).Trim() 'wrong stack sentinel'
    }

    Invoke-Test 'accuracy runner preserves sentinel on missing build schema failure' {
        $buildDir = Join-Path $repoScratch 'accuracy-missing-schema'
        New-AccuracyBuildFixture -SourceBuildDir $accuracySourceBuild -DestinationBuildDir $buildDir | Out-Null
        Set-Content -LiteralPath (Join-Path $buildDir 'build_info.json') -Value '{"oneApi":{"name":"oneapi-2025.3"}}' -Encoding UTF8
        $outputDir = Join-Path $accuracyScratch 'missing-schema'
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        $sentinelPath = Join-Path $outputDir 'accuracy.json'
        Set-Content -LiteralPath $sentinelPath -Value 'missing-schema-sentinel' -Encoding UTF8
        $result = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $regressPath `
            -ArgumentList @('-BuildDir', $buildDir, '-OneApiStack', 'oneapi-2025.3', '-CasesFile', $accuracyCasesPath, '-OutputDir', $outputDir) `
            -TimeoutMs 30000
        Assert-Equal $true $result.completed 'missing schema runner completion'
        if ($result.exitCode -eq 0) {
            throw 'Missing schema runner unexpectedly succeeded.'
        }
        Assert-Equal $true (($result.stderr + $result.stdout).Contains('Unsupported build info schema')) 'missing schema diagnostic'
        Assert-Equal 'missing-schema-sentinel' (Get-Content -LiteralPath $sentinelPath -Raw).Trim() 'missing schema sentinel'
    }

    Invoke-Test 'accuracy runner preserves sentinel on unknown and duplicate selections' {
        foreach ($selection in @('missing_case', 'case_one,case_one')) {
            $label = $selection.Replace(',', '-')
            $outputDir = Join-Path $accuracyScratch "selection-$label"
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
            $sentinelPath = Join-Path $outputDir 'accuracy.json'
            Set-Content -LiteralPath $sentinelPath -Value "selection-$label-sentinel" -Encoding UTF8
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $regressPath `
                -ArgumentList @('-BuildDir', $accuracyBuild, '-OneApiStack', 'oneapi-2025.3', '-CasesFile', $accuracyCasesPath, '-CaseNames', $selection, '-OutputDir', $outputDir) `
                -TimeoutMs 30000
            Assert-Equal $true $result.completed "selection $selection runner completion"
            if ($result.exitCode -eq 0) {
                throw "Selection '$selection' unexpectedly succeeded."
            }
            $expected = if ($selection -eq 'missing_case') { 'Unknown accuracy case selection' } else { 'Duplicate accuracy case selection' }
            Assert-Equal $true (($result.stderr + $result.stdout).Contains($expected)) "selection $selection diagnostic"
            Assert-Equal "selection-$label-sentinel" (Get-Content -LiteralPath $sentinelPath -Raw).Trim() "selection $selection sentinel"
        }
    }

    Invoke-Test 'accuracy runner preserves sentinel on missing input preflight failure' {
        $casesPath = Join-Path $repoScratch 'accuracy-missing-input-cases.json'
        Write-AilaJsonFile -Path $casesPath -Data ([ordered]@{
            schemaVersion = 1
            cases = @([ordered]@{ name = 'missing_input'; kind = 'chat'; model = $accuracyModel; input = (Join-Path $repoScratch 'absent.json'); maxTokens = 4; mode = 'greedy' })
        })
        $outputDir = Join-Path $accuracyScratch 'missing-input'
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        $sentinelPath = Join-Path $outputDir 'accuracy.json'
        Set-Content -LiteralPath $sentinelPath -Value 'missing-input-sentinel' -Encoding UTF8
        $result = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $regressPath `
            -ArgumentList @('-BuildDir', $accuracyBuild, '-OneApiStack', 'oneapi-2025.3', '-CasesFile', $casesPath, '-OutputDir', $outputDir) `
            -TimeoutMs 60000
        Assert-Equal $true $result.completed 'missing input runner completion'
        if ($result.exitCode -eq 0) {
            throw 'Missing input runner unexpectedly succeeded.'
        }
        Assert-Equal $true (($result.stderr + $result.stdout).Contains('input file not found')) 'missing input diagnostic'
        Assert-Equal 'missing-input-sentinel' (Get-Content -LiteralPath $sentinelPath -Raw).Trim() 'missing input sentinel'
    }

    Invoke-Test 'accuracy runner rejects output outside repository tmp without modifying it' {
        $outputDir = Join-Path $repoScratch 'accuracy-outside-tmp'
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        $sentinelPath = Join-Path $outputDir 'accuracy.json'
        Set-Content -LiteralPath $sentinelPath -Value 'outside-tmp-sentinel' -Encoding UTF8
        $result = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $regressPath `
            -ArgumentList @('-BuildDir', $accuracyBuild, '-OneApiStack', 'oneapi-2025.3', '-CasesFile', $accuracyCasesPath, '-OutputDir', $outputDir) `
            -TimeoutMs 30000
        Assert-Equal $true $result.completed 'outside tmp runner completion'
        if ($result.exitCode -eq 0) {
            throw 'Outside tmp runner unexpectedly succeeded.'
        }
        Assert-Equal $true (($result.stderr + $result.stdout).Contains('inside repository tmp')) 'outside tmp diagnostic'
        Assert-Equal 'outside-tmp-sentinel' (Get-Content -LiteralPath $sentinelPath -Raw).Trim() 'outside tmp sentinel'
    }

    Invoke-Test 'accuracy runner writes typed subset artifacts and preserves unrelated files across reruns' {
        $outputDir = Join-Path $accuracyScratch 'success'
        $unrelatedPath = Join-Path $outputDir 'outputs\keep.local'
        New-Item -ItemType Directory -Path (Split-Path -Parent $unrelatedPath) -Force | Out-Null
        Set-Content -LiteralPath $unrelatedPath -Value 'keep-me' -Encoding UTF8
        $parentCompilerRoot = [System.Environment]::GetEnvironmentVariable('CMPLR_ROOT', 'Process')

        $result = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $regressPath `
            -ArgumentList @('-BuildDir', $accuracyBuild, '-OneApiStack', 'oneapi-2025.3', '-CasesFile', $accuracyCasesPath, '-CaseNames', 'case_one,case_two', '-OutputDir', $outputDir) `
            -TimeoutMs 60000
        Assert-Equal $true $result.completed 'successful runner completion'
        Assert-Equal 0 $result.exitCode "successful runner exit stderr='$($result.stderr)' stdout='$($result.stdout)'"
        Assert-Equal $parentCompilerRoot ([System.Environment]::GetEnvironmentVariable('CMPLR_ROOT', 'Process')) 'runner child environment isolation'
        Assert-Equal 'keep-me' (Get-Content -LiteralPath $unrelatedPath -Raw).Trim() 'unrelated output preserved after first run'

        $accuracy = Read-AilaJsonFile -Path (Join-Path $outputDir 'accuracy.json')
        Assert-Equal 1 $accuracy.schemaVersion 'accuracy schema'
        Assert-Equal 'oneapi-2025.3' $accuracy.oneApi.name 'accuracy stack'
        Assert-Equal 2 $accuracy.cases.Count 'accuracy selected case count'
        foreach ($caseName in @('case_one', 'case_two')) {
            $caseResultPath = Join-Path $outputDir "case_results\$caseName.json"
            Assert-Equal $true (Test-Path -LiteralPath $caseResultPath -PathType Leaf) "$caseName result path"
            $caseResult = Read-AilaJsonFile -Path $caseResultPath
            Assert-Equal 'chat' $caseResult.kind "$caseName result kind"
            Assert-Equal 'hello world' $caseResult.normalizedText "$caseName normalized text"
            Assert-Equal 0 $caseResult.process.exitCode "$caseName exit code"
            Assert-Equal $true (Test-Path -LiteralPath $caseResult.output.path -PathType Leaf) "$caseName output artifact"
            Assert-Equal $caseResult.output.sha256 (Get-FileHash -LiteralPath $caseResult.output.path -Algorithm SHA256).Hash "$caseName output hash"
            Assert-Equal $true (Test-Path -LiteralPath $caseResult.logs.stdout.path -PathType Leaf) "$caseName stdout log"
            Assert-Equal $true (Test-Path -LiteralPath $caseResult.logs.stderr.path -PathType Leaf) "$caseName stderr log"
        }

        $rerun = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $regressPath `
            -ArgumentList @('-BuildDir', $accuracyBuild, '-OneApiStack', 'oneapi-2025.3', '-CasesFile', $accuracyCasesPath, '-CaseNames', 'case_one', '-OutputDir', $outputDir) `
            -TimeoutMs 60000
        Assert-Equal $true $rerun.completed 'rerun completion'
        Assert-Equal 0 $rerun.exitCode "rerun exit stderr='$($rerun.stderr)' stdout='$($rerun.stdout)'"
        Assert-Equal 'keep-me' (Get-Content -LiteralPath $unrelatedPath -Raw).Trim() 'unrelated output preserved after rerun'
        Assert-Equal $true (Test-Path -LiteralPath (Join-Path $outputDir 'case_results\case_two.json') -PathType Leaf) 'unselected result preserved after rerun'
        $rerunAccuracy = Read-AilaJsonFile -Path (Join-Path $outputDir 'accuracy.json')
        Assert-Equal 1 $rerunAccuracy.cases.Count 'rerun selected case count'
        Assert-Equal 'case_one' $rerunAccuracy.cases[0].name 'rerun selected case name'
    }

    Invoke-Test 'verifier rejects a missing build directory' {
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @('-BuildDir', 'missing-build', '-OneApiStack', 'oneapi-2026.1') `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } 'Build directory not found:' 'missing build verification'
    }

    Invoke-Test 'verifier rejects a missing build info file' {
        $buildDir = Join-Path $repoScratch 'verifier-missing-build-info'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @('-BuildDir', $buildDir, '-OneApiStack', 'oneapi-2026.1') `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } 'Build info file not found:' 'missing build info verification'
    }

    Invoke-Test 'verifier rejects a different requested stack without writing output' {
        $buildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        if (-not (Test-Path -LiteralPath $buildDir -PathType Container)) {
            throw "Required verifier fixture build directory not found: $buildDir"
        }
        $wrongStackOutputDir = Join-Path $buildDir 'wrong-stack-output'
        $outputPath = Join-Path $wrongStackOutputDir 'oneapi_verification.json'
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @(
                    '-BuildDir', $buildDir,
                    '-OneApiStack', 'oneapi-2026.1',
                    '-OutputPath', $outputPath
                ) `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } "but 'oneapi-2026.1'" 'wrong stack verification'
        Assert-Equal $false (Test-Path -LiteralPath $outputPath) 'wrong stack output must not be written'
    }

    Invoke-Test 'verifier rejects an outside output path without modifying its sentinel' {
        $outsideOutputDir = Join-Path $tempRoot 'outside-output'
        New-Item -ItemType Directory -Path $outsideOutputDir -Force | Out-Null
        $outputPath = Join-Path $outsideOutputDir 'oneapi_verification.json'
        Set-Content -LiteralPath $outputPath -Value 'outside-sentinel' -Encoding UTF8
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @(
                    '-BuildDir', 'missing-build',
                    '-OneApiStack', 'oneapi-2026.1',
                    '-OutputPath', $outputPath
                ) `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } 'outside selected build directory' 'outside output path verification'
        Assert-Equal 'outside-sentinel' (Get-Content -LiteralPath $outputPath -Raw).Trim() 'outside sentinel content'
    }

    Invoke-Test 'verifier rejects owned filename outside the selected build without modifying it' {
        $unrelatedDir = Join-Path $repoScratch 'unrelated-output'
        New-Item -ItemType Directory -Path $unrelatedDir -Force | Out-Null
        $outputPath = Join-Path $unrelatedDir 'oneapi_verification.json'
        Set-Content -LiteralPath $outputPath -Value 'inside-repo-sentinel' -Encoding UTF8
        $sentinelHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @(
                    '-BuildDir', (Join-Path $repoScratch 'missing-selected-build'),
                    '-OneApiStack', 'oneapi-2026.1',
                    '-OutputPath', $outputPath
                ) `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } 'outside selected build directory' 'selected build output boundary'
        Assert-Equal 'inside-repo-sentinel' (Get-Content -LiteralPath $outputPath -Raw).Trim() 'inside repo sentinel content'
        Assert-Equal $sentinelHash (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash 'inside repo sentinel hash'
    }

    Invoke-Test 'verifier rejects a wrong output filename inside the selected build' {
        $sourceBuildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        $buildDir = New-VerifierBuildFixture -SourceBuildDir $sourceBuildDir -DestinationBuildDir (Join-Path $repoScratch 'wrong-output-name')
        $outputPath = Join-Path $buildDir 'not-owned.json'
        Set-Content -LiteralPath $outputPath -Value 'wrong-name-sentinel' -Encoding UTF8
        $sentinelHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @(
                    '-BuildDir', $buildDir,
                    '-OneApiStack', 'oneapi-2025.3',
                    '-OutputPath', $outputPath
                ) `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } "must be named 'oneapi_verification.json'" 'owned output filename'
        Assert-Equal 'wrong-name-sentinel' (Get-Content -LiteralPath $outputPath -Raw).Trim() 'wrong filename sentinel content'
        Assert-Equal $sentinelHash (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash 'wrong filename sentinel hash'
    }

    Invoke-Test 'verifier publishes owned output in a selected build subdirectory' {
        $sourceBuildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        $buildDir = New-VerifierBuildFixture -SourceBuildDir $sourceBuildDir -DestinationBuildDir (Join-Path $repoScratch 'owned-subdirectory')
        $outputPath = Join-Path $buildDir 'attestations\oneapi_verification.json'
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        $result = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $verifierPath `
            -ArgumentList @(
                '-BuildDir', $buildDir,
                '-OneApiStack', 'oneapi-2025.3',
                '-OutputPath', $outputPath
            ) `
            -TimeoutMs 30000
        Assert-Equal $true $result.completed 'owned subdirectory verifier completion'
        Assert-Equal 0 $result.exitCode "owned subdirectory verifier exit stderr='$($result.stderr)'"
        $verification = Read-AilaJsonFile -Path $outputPath
        Assert-Equal 1 $verification.schemaVersion 'owned subdirectory schema'
        Assert-Equal 'oneapi-2025.3' $verification.stack.name 'owned subdirectory stack'
        Assert-Equal 'sycl8.dll' $verification.dependencies[0] 'owned subdirectory dependency'
        $tempFiles = @(Get-ChildItem -LiteralPath (Split-Path -Parent $outputPath) -Filter '.oneapi_verification.json.*.tmp' -Force)
        Assert-Equal 0 $tempFiles.Count 'owned subdirectory temp cleanup'
    }

    Invoke-Test 'verifier rejects a junction output escaping the selected build' {
        $sourceBuildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        $buildDir = New-VerifierBuildFixture -SourceBuildDir $sourceBuildDir -DestinationBuildDir (Join-Path $repoScratch 'junction-output')
        $outsideDir = Join-Path $tempRoot 'junction-output-target'
        New-Item -ItemType Directory -Path $outsideDir -Force | Out-Null
        $outsideSentinel = Join-Path $outsideDir 'oneapi_verification.json'
        Set-Content -LiteralPath $outsideSentinel -Value 'junction-outside-sentinel' -Encoding UTF8
        $sentinelHash = (Get-FileHash -LiteralPath $outsideSentinel -Algorithm SHA256).Hash
        $junctionPath = Join-Path $buildDir 'junction-out'
        New-Item -ItemType Junction -Path $junctionPath -Target $outsideDir | Out-Null
        try {
            $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
            Assert-Throws {
                $result = Invoke-ChildPowerShellWithTimeout `
                    -ScriptPath $verifierPath `
                    -ArgumentList @(
                        '-BuildDir', $buildDir,
                        '-OneApiStack', 'oneapi-2025.3',
                        '-OutputPath', (Join-Path $junctionPath 'oneapi_verification.json')
                    ) `
                    -TimeoutMs 30000
                if (-not $result.completed) {
                    throw 'Verifier child process timed out.'
                }
                if ($result.exitCode -ne 0) {
                    throw ($result.stderr + $result.stdout).Trim()
                }
            } 'outside selected build directory' 'junction output boundary'
            Assert-Equal 'junction-outside-sentinel' (Get-Content -LiteralPath $outsideSentinel -Raw).Trim() 'junction sentinel content'
            Assert-Equal $sentinelHash (Get-FileHash -LiteralPath $outsideSentinel -Algorithm SHA256).Hash 'junction sentinel hash'
        }
        finally {
            if (Test-Path -LiteralPath $junctionPath) {
                Remove-Item -LiteralPath $junctionPath -Force
            }
        }
    }

    Invoke-Test 'verifier removes stale output when role metadata is missing' {
        $sourceBuildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        $buildDir = New-VerifierBuildFixture -SourceBuildDir $sourceBuildDir -DestinationBuildDir (Join-Path $repoScratch 'missing-role')
        $buildInfoPath = Join-Path $buildDir 'build_info.json'
        $buildInfo = Read-AilaJsonFile -Path $buildInfoPath
        $buildInfo.oneApi.PSObject.Properties.Remove('role')
        Write-AilaJsonFile -Path $buildInfoPath -Data $buildInfo
        $outputPath = Join-Path $buildDir 'oneapi_verification.json'
        Set-Content -LiteralPath $outputPath -Value '{"stale":true}' -Encoding UTF8

        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @('-BuildDir', $buildDir, '-OneApiStack', 'oneapi-2025.3') `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } "missing required property 'role'" 'missing role verification'
        Assert-Equal $false (Test-Path -LiteralPath $outputPath) 'missing role must invalidate stale output'
    }

    Invoke-Test 'verifier removes stale output when role metadata is wrong' {
        $sourceBuildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        $buildDir = New-VerifierBuildFixture -SourceBuildDir $sourceBuildDir -DestinationBuildDir (Join-Path $repoScratch 'wrong-role')
        $buildInfoPath = Join-Path $buildDir 'build_info.json'
        $buildInfo = Read-AilaJsonFile -Path $buildInfoPath
        $buildInfo.oneApi.role = 'candidate'
        Write-AilaJsonFile -Path $buildInfoPath -Data $buildInfo
        $outputPath = Join-Path $buildDir 'oneapi_verification.json'
        Set-Content -LiteralPath $outputPath -Value '{"stale":true}' -Encoding UTF8

        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $verifierPath `
                -ArgumentList @('-BuildDir', $buildDir, '-OneApiStack', 'oneapi-2025.3') `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } "metadata mismatch for 'role'" 'wrong role verification'
        Assert-Equal $false (Test-Path -LiteralPath $outputPath) 'wrong role must invalidate stale output'
    }

    Invoke-Test 'verifier atomically replaces stale output with complete JSON' {
        $sourceBuildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        $buildDir = New-VerifierBuildFixture -SourceBuildDir $sourceBuildDir -DestinationBuildDir (Join-Path $repoScratch 'atomic-success')
        $outputDir = Join-Path $buildDir 'attestations'
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        $outputPath = Join-Path $outputDir 'oneapi_verification.json'
        Set-Content -LiteralPath $outputPath -Value 'stale-sentinel' -Encoding UTF8
        $verifierPath = Join-Path $repoRoot 'verify_oneapi_build.ps1'
        $result = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $verifierPath `
            -ArgumentList @(
                '-BuildDir', $buildDir,
                '-OneApiStack', 'oneapi-2025.3',
                '-OutputPath', $outputPath
            ) `
            -TimeoutMs 30000
        Assert-Equal $true $result.completed 'atomic verifier completion'
        Assert-Equal 0 $result.exitCode "atomic verifier exit stderr='$($result.stderr)'"

        $verification = Read-AilaJsonFile -Path $outputPath
        Assert-Equal 1 $verification.schemaVersion 'atomic verification schema'
        Assert-Equal 'oneapi-2025.3' $verification.stack.name 'atomic verification stack'
        Assert-Equal 1 @($verification.dependencies).Count 'atomic verification dependency count'
        Assert-Equal 'sycl8.dll' $verification.dependencies[0] 'atomic verification dependency'
        Assert-Equal (Get-FileHash -LiteralPath (Join-Path $buildDir 'Aila.exe') -Algorithm SHA256).Hash $verification.executableSha256 'atomic verification hash'
        $tempFiles = @(Get-ChildItem -LiteralPath $outputDir -Filter '.oneapi_verification.json.*.tmp' -Force)
        Assert-Equal 0 $tempFiles.Count 'atomic verification temp cleanup'
    }

    Invoke-Test 'verifier rejects an executable that changes during dependency inspection' {
        $sourceBuildDir = Join-Path $repoRoot 'build-oneapi-2025.3'
        $buildDir = New-VerifierBuildFixture -SourceBuildDir $sourceBuildDir -DestinationBuildDir (Join-Path $repoScratch 'identity-change')
        $outputPath = Join-Path $buildDir 'oneapi_verification.json'
        Set-Content -LiteralPath $outputPath -Value '{"stale":true}' -Encoding UTF8
        $childScriptPath = Join-Path $tempRoot 'invoke-verifier-identity-change.ps1'
        Set-Content -LiteralPath $childScriptPath -Encoding UTF8 -Value @'
param(
    [Parameter(Mandatory = $true)][string]$VerifierPath,
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $VerifierPath) 'perf\PerfCommon.ps1')
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($VerifierPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) {
    throw "Unable to parse verifier functions: $($parseErrors[0].Message)"
}
$functionDefinitions = $ast.FindAll({
    param($node)
    return $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $false)
foreach ($definition in $functionDefinitions) {
    Invoke-Expression $definition.Extent.Text
}
$inspector = {
    param([Parameter(Mandatory = $true)][string]$Executable)
    [System.IO.File]::AppendAllText($Executable, 'identity-change')
    return "    sycl8.dll`r`n"
}
Invoke-AilaOneApiBuildVerification `
    -BuildDir $BuildDir `
    -OneApiStack 'oneapi-2025.3' `
    -OutputPath $OutputPath `
    -RepoRoot (Split-Path -Parent $VerifierPath) `
    -DependencyInspector $inspector
'@

        Assert-Throws {
            $result = Invoke-ChildPowerShellWithTimeout `
                -ScriptPath $childScriptPath `
                -ArgumentList @(
                    '-VerifierPath', (Join-Path $repoRoot 'verify_oneapi_build.ps1'),
                    '-BuildDir', $buildDir,
                    '-OutputPath', $outputPath
                ) `
                -TimeoutMs 30000
            if (-not $result.completed) {
                throw 'Verifier identity child process timed out.'
            }
            if ($result.exitCode -ne 0) {
                throw ($result.stderr + $result.stdout).Trim()
            }
        } 'changed during dependency inspection' 'executable identity verification'
        Assert-Equal $false (Test-Path -LiteralPath $outputPath) 'identity change must invalidate stale output'
    }

    Invoke-Test 'loads the complete real stack contract' {
        $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
        $baseline = Get-AilaOneApiStack -Config $config -Name 'oneapi-2025.3'
        $candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'

        Assert-Equal 'baseline' $baseline.role 'baseline role'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/compiler/2025.3' $baseline.compilerRoot 'baseline compiler root'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/dnnl/2025.3' $baseline.dnnlRoot 'baseline oneDNN root'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/tbb/2022.3' $baseline.tbbRoot 'baseline TBB root'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/umf/1.0' $baseline.umfRoot 'baseline UMF root'
        Assert-Equal '2025.3.3' $baseline.expectedCompilerVersion 'baseline compiler version'
        Assert-Equal '3.9.1' $baseline.expectedDnnlVersion 'baseline oneDNN'
        Assert-Equal 'sycl8.dll' $baseline.expectedSyclDll 'baseline SYCL ABI'
        Assert-Equal $true $baseline.allowLegacyCompiler 'baseline legacy compiler policy'

        Assert-Equal 'candidate' $candidate.role 'candidate role'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/compiler/2026.1' $candidate.compilerRoot 'candidate compiler root'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/dnnl/2026.0' $candidate.dnnlRoot 'candidate oneDNN root'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/tbb/2023.1' $candidate.tbbRoot 'candidate TBB root'
        Assert-Equal 'C:/Program Files (x86)/Intel/oneAPI/umf/1.1' $candidate.umfRoot 'candidate UMF root'
        Assert-Equal '2026.1.0' $candidate.expectedCompilerVersion 'candidate compiler version'
        Assert-Equal '3.11.2' $candidate.expectedDnnlVersion 'candidate oneDNN'
        Assert-Equal 'sycl9.dll' $candidate.expectedSyclDll 'candidate SYCL ABI'
        Assert-Equal $false $candidate.allowLegacyCompiler 'candidate legacy compiler policy'

        $baselineMeta = Get-AilaOneApiStackMetadata -Stack $baseline
        Assert-Equal 'oneapi-2025.3' $baselineMeta.name 'baseline metadata name'
        Assert-Equal 'baseline' $baselineMeta.role 'baseline metadata role'
        Assert-Equal (Join-Path $baseline.compilerRoot 'bin\icx-cl.exe') $baselineMeta.compilerPath 'baseline compiler path'
        Assert-Equal '2025.3.3' $baselineMeta.compilerVersion 'baseline compiler version'
        Assert-Equal $baseline.dnnlRoot $baselineMeta.dnnlRoot 'baseline oneDNN root'
        Assert-Equal '3.9.1' $baselineMeta.dnnlVersion 'baseline oneDNN version'
        Assert-Equal $baseline.tbbRoot $baselineMeta.tbbRoot 'baseline TBB root'
        Assert-Equal '2022.3' $baselineMeta.tbbVersion 'baseline TBB version'
        Assert-Equal $baseline.umfRoot $baselineMeta.umfRoot 'baseline UMF root'
        Assert-Equal 'sycl8.dll' $baselineMeta.expectedSyclDll 'baseline SYCL DLL'
        Assert-Equal $true $baselineMeta.allowLegacyCompiler 'baseline legacy flag'

        $candidateMeta = Get-AilaOneApiStackMetadata -Stack $candidate
        Assert-Equal 'oneapi-2026.1' $candidateMeta.name 'candidate metadata name'
        Assert-Equal 'candidate' $candidateMeta.role 'candidate metadata role'
        Assert-Equal (Join-Path $candidate.compilerRoot 'bin\icx-cl.exe') $candidateMeta.compilerPath 'candidate compiler path'
        Assert-Equal '2026.1.0' $candidateMeta.compilerVersion 'compiler version'
        Assert-Equal $candidate.dnnlRoot $candidateMeta.dnnlRoot 'candidate oneDNN root'
        Assert-Equal '3.11.2' $candidateMeta.dnnlVersion 'oneDNN version'
        Assert-Equal $candidate.tbbRoot $candidateMeta.tbbRoot 'candidate TBB root'
        Assert-Equal '2023.1' $candidateMeta.tbbVersion 'TBB version'
        Assert-Equal $candidate.umfRoot $candidateMeta.umfRoot 'candidate UMF root'
        Assert-Equal 'sycl9.dll' $candidateMeta.expectedSyclDll 'SYCL DLL'
        Assert-Equal $false $candidateMeta.allowLegacyCompiler 'candidate legacy flag'
    }

    Invoke-Test 'rejects a missing compiler metadata executable' {
        $stack = New-TestStack -Root $validRoot
        Assert-Throws {
            Get-AilaOneApiStackMetadata -Stack $stack
        } 'Compiler executable not found:' 'missing compiler executable'
    }

    Invoke-Test 'rejects a failed compiler metadata query' {
        $stack = New-TestStack -Root $validRoot
        $stack.compilerRoot = $failingCompilerRoot
        Assert-Throws {
            Get-AilaOneApiStackMetadata -Stack $stack
        } 'Compiler version query failed' 'failed compiler query'
    }

    Invoke-Test 'rejects a missing oneDNN metadata version file' {
        $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
        $candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
        $candidate.dnnlRoot = $validRoot
        Assert-Throws {
            Get-AilaOneApiStackMetadata -Stack $candidate
        } 'oneDNN version file not found:' 'missing oneDNN version file'
    }

    Invoke-Test 'rejects an invalid oneDNN metadata version file' {
        $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
        $candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
        $candidate.dnnlRoot = $invalidDnnlRoot
        Assert-Throws {
            Get-AilaOneApiStackMetadata -Stack $candidate
        } 'Unable to parse oneDNN version from' 'invalid oneDNN version file'
    }

    Invoke-Test 'rejects an installed compiler version that does not match the stack' {
        $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
        $candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
        $candidate.expectedCompilerVersion = '0.0.0'
        Assert-Throws {
            Get-AilaOneApiStackMetadata -Stack $candidate
        } "Compiler version mismatch for oneAPI stack 'oneapi-2026.1'" 'compiler version mismatch'
    }

    Invoke-Test 'rejects an installed oneDNN version that does not match the stack' {
        $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
        $candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
        $candidate.expectedDnnlVersion = '0.0.0'
        Assert-Throws {
            Get-AilaOneApiStackMetadata -Stack $candidate
        } "oneDNN version mismatch for oneAPI stack 'oneapi-2026.1'" 'oneDNN version mismatch'
    }

    Invoke-Test 'rejects an unknown dotted stack name explicitly' {
        $config = [pscustomobject]@{
            stacks = [pscustomobject]@{ 'synthetic.1' = (New-TestStack -Root $validRoot) }
        }
        Assert-Throws {
            Get-AilaOneApiStack -Config $config -Name 'missing.9'
        } "oneAPI stack 'missing.9' not found." 'unknown stack'
    }

    Invoke-Test 'rejects a missing stacks collection explicitly' {
        Assert-Throws {
            Get-AilaOneApiStack -Config ([pscustomobject]@{}) -Name 'missing.9'
        } "oneAPI stack 'missing.9' not found." 'missing stacks collection'
    }

    Invoke-Test 'rejects an unsupported schema explicitly' {
        Assert-Throws {
            Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path $unsupportedSchemaPath
        } 'Unsupported oneAPI stack config schema: 2' 'unsupported schema'
    }

    Invoke-Test 'rejects a missing schema explicitly' {
        Assert-Throws {
            Get-AilaOneApiStackConfig -RepoRoot $repoRoot -Path $missingSchemaPath
        } 'Unsupported oneAPI stack config schema: <missing>' 'missing schema'
    }

    Invoke-Test 'rejects a missing required root property explicitly' {
        $stack = [pscustomobject]@{
            dnnlRoot = $validRoot
            tbbRoot  = $validRoot
            umfRoot  = $validRoot
        }
        $config = [pscustomobject]@{ stacks = [pscustomobject]@{ 'synthetic.1' = $stack } }
        Assert-Throws {
            Get-AilaOneApiStack -Config $config -Name 'synthetic.1'
        } "missing required root property 'compilerRoot'" 'missing root property'
    }

    Invoke-Test 'rejects a blank required root explicitly' {
        $stack = New-TestStack -Root $validRoot
        $stack.compilerRoot = '   '
        $config = [pscustomobject]@{ stacks = [pscustomobject]@{ 'synthetic.1' = $stack } }
        Assert-Throws {
            Get-AilaOneApiStack -Config $config -Name 'synthetic.1'
        } 'has blank compilerRoot' 'blank root'
    }

    Invoke-Test 'rejects an ordinary file as a root' {
        $stack = New-TestStack -Root $validRoot
        $stack.compilerRoot = $ordinaryFile
        $config = [pscustomobject]@{ stacks = [pscustomobject]@{ 'synthetic.1' = $stack } }
        Assert-Throws {
            Get-AilaOneApiStack -Config $config -Name 'synthetic.1'
        } 'compilerRoot is not a directory' 'file root'
    }

    Invoke-Test 'does not mutate the source stack object' {
        $sourceStack = New-TestStack -Root $validRoot
        $config = [pscustomobject]@{ stacks = [pscustomobject]@{ 'synthetic.1' = $sourceStack } }
        $result = Get-AilaOneApiStack -Config $config -Name 'synthetic.1'

        Assert-Equal 'synthetic.1' $result.name 'returned stack name'
        if ($null -ne $sourceStack.PSObject.Properties['name']) {
            throw 'source stack unexpectedly contains an injected name property'
        }
    }

    Invoke-Test 'rejects build metadata from a different oneAPI stack' {
        $buildDir = Join-Path $tempRoot 'build-info-mismatch'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        Write-AilaJsonFile -Path (Join-Path $buildDir 'build_info.json') -Data ([ordered]@{
            schemaVersion = 2
            oneApi = [ordered]@{ name = 'oneapi-2025.3' }
        })
        $stack = [pscustomobject]@{ name = 'oneapi-2026.1' }

        Assert-Throws {
            Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $buildDir -Stack $stack
        } "contains oneAPI stack 'oneapi-2025.3'" 'build info stack mismatch'
    }

    Invoke-Test 'accepts build metadata for the selected oneAPI stack' {
        $buildDir = Join-Path $tempRoot 'build-info-match'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        Write-AilaJsonFile -Path (Join-Path $buildDir 'build_info.json') -Data ([ordered]@{
            schemaVersion = 2
            oneApi = [ordered]@{ name = 'oneapi-2026.1' }
        })
        $stack = [pscustomobject]@{ name = 'oneapi-2026.1' }

        Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $buildDir -Stack $stack
    }

    Invoke-Test 'rejects a cached compiler outside the selected stack' {
        $buildDir = Join-Path $tempRoot 'compiler-cache-mismatch'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $selectedRoot = Join-Path $tempRoot 'selected'
        $otherRoot = Join-Path $tempRoot 'other'
        $stack = [pscustomobject]@{
            name = 'selected'
            compilerRoot = Join-Path $selectedRoot 'compiler\1'
            dnnlRoot = Join-Path $selectedRoot 'dnnl\1'
            tbbRoot = Join-Path $selectedRoot 'tbb\1'
            allowLegacyCompiler = $false
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $otherRoot 'compiler\2\bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $stack.dnnlRoot 'lib\cmake\dnnl')"
            "TBB_DIR:PATH=$(Join-Path $stack.tbbRoot 'lib\cmake\tbb')"
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack
        } 'CMAKE_CXX_COMPILER' 'cached compiler mismatch'
    }

    Invoke-Test 'rejects a cached oneDNN directory outside the selected stack' {
        $buildDir = Join-Path $tempRoot 'dnnl-cache-mismatch'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $selectedRoot = Join-Path $tempRoot 'selected-dnnl'
        $otherRoot = Join-Path $tempRoot 'other-dnnl'
        $stack = [pscustomobject]@{
            name = 'selected'
            compilerRoot = Join-Path $selectedRoot 'compiler\1'
            dnnlRoot = Join-Path $selectedRoot 'dnnl\1'
            tbbRoot = Join-Path $selectedRoot 'tbb\1'
            allowLegacyCompiler = $false
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $stack.compilerRoot 'bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $otherRoot 'dnnl\2\lib\cmake\dnnl')"
            "TBB_DIR:PATH=$(Join-Path $stack.tbbRoot 'lib\cmake\tbb')"
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack
        } 'dnnl_DIR' 'cached oneDNN mismatch'
    }

    Invoke-Test 'rejects a cached TBB directory outside the selected stack' {
        $buildDir = Join-Path $tempRoot 'tbb-cache-mismatch'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $selectedRoot = Join-Path $tempRoot 'selected-tbb'
        $otherRoot = Join-Path $tempRoot 'other-tbb'
        $stack = [pscustomobject]@{
            name = 'selected'
            compilerRoot = Join-Path $selectedRoot 'compiler\1'
            dnnlRoot = Join-Path $selectedRoot 'dnnl\1'
            tbbRoot = Join-Path $selectedRoot 'tbb\1'
            allowLegacyCompiler = $false
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $stack.compilerRoot 'bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $stack.dnnlRoot 'lib\cmake\dnnl')"
            "TBB_DIR:PATH=$(Join-Path $otherRoot 'tbb\2\lib\cmake\tbb')"
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack
        } 'TBB_DIR' 'cached TBB mismatch'
    }

    Invoke-Test 'requires all selected stack cache values after configure' {
        $buildDir = Join-Path $tempRoot 'cache-missing-tbb'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $selectedRoot = Join-Path $tempRoot 'selected-required'
        $stack = [pscustomobject]@{
            name = 'selected'
            compilerRoot = Join-Path $selectedRoot 'compiler\1'
            dnnlRoot = Join-Path $selectedRoot 'dnnl\1'
            tbbRoot = Join-Path $selectedRoot 'tbb\1'
            allowLegacyCompiler = $true
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $stack.compilerRoot 'bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $stack.dnnlRoot 'lib\cmake\dnnl')"
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack -RequireValues
        } 'missing TBB_DIR' 'missing configured TBB value'
    }

    Invoke-Test 'rejects candidate cache with legacy baseline mode enabled' {
        $buildDir = Join-Path $tempRoot 'candidate-legacy-cache-on'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $selectedRoot = Join-Path $tempRoot 'candidate-legacy-selected'
        $stack = [pscustomobject]@{
            name = 'candidate'
            compilerRoot = Join-Path $selectedRoot 'compiler\1'
            dnnlRoot = Join-Path $selectedRoot 'dnnl\1'
            tbbRoot = Join-Path $selectedRoot 'tbb\1'
            allowLegacyCompiler = $false
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $stack.compilerRoot 'bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $stack.dnnlRoot 'lib\cmake\dnnl')"
            "TBB_DIR:PATH=$(Join-Path $stack.tbbRoot 'lib\cmake\tbb')"
            'AILA_ALLOW_LEGACY_ONEAPI_BASELINE:BOOL=ON'
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack
        } 'AILA_ALLOW_LEGACY_ONEAPI_BASELINE' 'candidate cached legacy mode'
    }

    Invoke-Test 'rejects baseline cache with legacy baseline mode disabled' {
        $buildDir = Join-Path $tempRoot 'baseline-legacy-cache-off'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $selectedRoot = Join-Path $tempRoot 'baseline-legacy-selected'
        $stack = [pscustomobject]@{
            name = 'baseline'
            compilerRoot = Join-Path $selectedRoot 'compiler\1'
            dnnlRoot = Join-Path $selectedRoot 'dnnl\1'
            tbbRoot = Join-Path $selectedRoot 'tbb\1'
            allowLegacyCompiler = $true
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $stack.compilerRoot 'bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $stack.dnnlRoot 'lib\cmake\dnnl')"
            "TBB_DIR:PATH=$(Join-Path $stack.tbbRoot 'lib\cmake\tbb')"
            'AILA_ALLOW_LEGACY_ONEAPI_BASELINE:BOOL=OFF'
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack
        } 'AILA_ALLOW_LEGACY_ONEAPI_BASELINE' 'baseline cached legacy mode'
    }

    Invoke-Test 'requires the legacy baseline cache option after configure' {
        $buildDir = Join-Path $tempRoot 'baseline-legacy-cache-missing'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $selectedRoot = Join-Path $tempRoot 'baseline-legacy-missing-selected'
        $stack = [pscustomobject]@{
            name = 'baseline'
            compilerRoot = Join-Path $selectedRoot 'compiler\1'
            dnnlRoot = Join-Path $selectedRoot 'dnnl\1'
            tbbRoot = Join-Path $selectedRoot 'tbb\1'
            allowLegacyCompiler = $true
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $stack.compilerRoot 'bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $stack.dnnlRoot 'lib\cmake\dnnl')"
            "TBB_DIR:PATH=$(Join-Path $stack.tbbRoot 'lib\cmake\tbb')"
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack -RequireValues
        } 'missing AILA_ALLOW_LEGACY_ONEAPI_BASELINE' 'missing cached legacy option'
    }

    Invoke-Test 'imports an isolated baseline oneAPI environment' {
        $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
        $baseline = Get-AilaOneApiStack -Config $config -Name 'oneapi-2025.3'
        $envMap = Get-AilaOneApiStackEnvironment -Stack $baseline

        Assert-Equal '2025.3' (Split-Path $envMap.CMPLR_ROOT -Leaf) 'baseline compiler root'
        if ($envMap.PATH -notmatch [regex]::Escape('compiler\2025.3\bin')) {
            throw 'baseline PATH does not contain compiler 2025.3 bin'
        }
        if ($envMap.PATH -match 'compiler\\2026\.1') {
            throw 'baseline PATH contains candidate compiler'
        }
        if ($envMap.PATH.Contains('System.Object[]')) {
            throw 'baseline PATH contains a stringified nested array'
        }
        Assert-PathContainsSegment $envMap.PATH (Join-Path $env:SystemRoot 'System32') 'baseline inherited system path'
        Assert-PathContainsSegment $envMap.PATH (Join-Path $baseline.compilerRoot 'lib\ocloc') 'baseline selected compiler ocloc path'
        Assert-PathSegmentsUnique $envMap.PATH 'baseline PATH'
        Assert-Equal (Get-NormalizedTestPath $baseline.umfRoot) (Get-NormalizedTestPath $envMap.UMF_ROOT) 'baseline UMF root'
        foreach ($key in @('PATH', 'LIB', 'INCLUDE', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH')) {
            if ($envMap.ContainsKey($key) -and $envMap[$key] -match '(?i)umf[\\/](latest|1\.1)') {
                throw "baseline $key retains a non-selected UMF path"
            }
        }
    }

    Invoke-Test 'imports an isolated candidate oneAPI environment' {
        $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
        $candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'
        $envMap = Get-AilaOneApiStackEnvironment -Stack $candidate

        Assert-Equal '2026.1' (Split-Path $envMap.CMPLR_ROOT -Leaf) 'candidate compiler root'
        if ($envMap.PATH -notmatch [regex]::Escape('compiler\2026.1\bin')) {
            throw 'candidate PATH does not contain compiler 2026.1 bin'
        }
        if ($envMap.PATH -match 'compiler\\2025\.3') {
            throw 'candidate PATH contains baseline compiler'
        }
        if ($envMap.PATH.Contains('System.Object[]')) {
            throw 'candidate PATH contains a stringified nested array'
        }
        Assert-PathContainsSegment $envMap.PATH (Join-Path $env:SystemRoot 'System32') 'candidate inherited system path'
        Assert-PathContainsSegment $envMap.PATH (Join-Path $candidate.compilerRoot 'lib\ocloc') 'candidate selected compiler ocloc path'
        Assert-PathSegmentsUnique $envMap.PATH 'candidate PATH'
        Assert-Equal (Get-NormalizedTestPath $candidate.umfRoot) (Get-NormalizedTestPath $envMap.UMF_ROOT) 'candidate UMF root'
        foreach ($key in @('PATH', 'LIB', 'INCLUDE', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH')) {
            if ($envMap.ContainsKey($key) -and $envMap[$key] -match '(?i)umf[\\/]latest') {
                throw "candidate $key retains an unresolved UMF latest path"
            }
        }
    }

    Invoke-Test 'propagates a failing batch script exit code' {
        Assert-Throws {
            Import-AilaBatchEnvironment -Scripts @($failingBatchPath)
        } 'exit code 37' 'failing batch script'
    }

    Invoke-Test 'reports stderr and script context for a failing batch import' {
        $caught = $null
        try {
            Import-AilaBatchEnvironment -Scripts @($diagnosticBatchPath)
        }
        catch {
            $caught = $_
        }

        if ($null -eq $caught) {
            throw 'diagnostic batch expected an exception'
        }
        $message = [string]$caught.Exception.Message
        foreach ($fragment in @('exit code 23', 'stdout: <empty>', 'stderr: diagnostic-from-batch', 'fail-with-diagnostic.bat')) {
            if (-not $message.Contains($fragment)) {
                throw "diagnostic batch expected message containing='$fragment' actual='$message'"
            }
        }
        if ($message.Length -gt 7000) {
            throw "diagnostic batch error was not bounded: length=$($message.Length)"
        }
    }

    Invoke-Test 'bounds native diagnostic text before adding it to errors' {
        $bounded = Get-AilaBoundedDiagnosticText -Text ('x' * 5000) -MaxLength 256
        if ($bounded.Length -gt 400) {
            throw "bounded diagnostic remained too large: length=$($bounded.Length)"
        }
        if (-not $bounded.Contains('<truncated')) {
            throw "bounded diagnostic did not report truncation: '$bounded'"
        }
        if ($bounded.Contains('x' * 1000)) {
            throw 'bounded diagnostic retained an unbounded payload.'
        }
    }

    Invoke-Test 'drains saturated batch stdout and stderr pipes without deadlock' {
        $child = Invoke-ChildPowerShellWithTimeout `
            -ScriptPath $saturationChildPath `
            -ArgumentList @('-PerfCommonPath', (Join-Path $repoRoot 'perf\PerfCommon.ps1'), '-BatchPath', $saturationBatchPath) `
            -TimeoutMs 15000

        if (-not $child.completed) {
            throw 'Saturation import child timed out after 15000 ms; batch output pipes were not drained concurrently.'
        }
        if ($child.exitCode -ne 0) {
            throw "Saturation import child failed with exit code $($child.exitCode). stdout='$($child.stdout)' stderr='$($child.stderr)'"
        }
        if (-not $child.stdout.Contains('SATURATION_IMPORT_PASS')) {
            throw "Saturation import child did not report success. stdout='$($child.stdout)' stderr='$($child.stderr)'"
        }
    }

    Invoke-Test 'rejects a partial batch environment' {
        Assert-Throws {
            Import-AilaBatchEnvironment -Scripts @($partialBatchPath)
        } 'missing required variable: CMPLR_ROOT' 'partial batch environment'
    }

    Invoke-Test 'rejects a batch environment without PATH' {
        Assert-Throws {
            Import-AilaBatchEnvironment -Scripts @($missingPathBatchPath)
        } 'missing required variable: PATH' 'missing PATH batch environment'
    }

    Invoke-Test 'recognizes selected compiler paths in a custom installation root' {
        $selectedRoot = 'D:\oneAPI\compiler\2026.1'
        Assert-Equal $true (Test-AilaPathWithinRoot -Path 'D:\oneAPI\compiler\2026.1\lib\ocloc' -Root $selectedRoot) 'selected custom compiler path'
        Assert-Equal $false (Test-AilaPathWithinRoot -Path 'D:\oneAPI\compiler\2025.3\bin' -Root $selectedRoot) 'other custom compiler path'
    }

    Invoke-Test 'rejects a lexical child that traverses a junction outside the selected root' {
        $selectedRoot = Join-Path $tempRoot 'junction-selected-root'
        $outsideRoot = Join-Path $tempRoot 'junction-outside-root'
        $realChild = Join-Path $selectedRoot 'real-child'
        $junctionPath = Join-Path $selectedRoot 'escape'
        New-Item -ItemType Directory -Path $realChild,$outsideRoot -Force | Out-Null

        try {
            try {
                New-Item -ItemType Junction -Path $junctionPath -Target $outsideRoot | Out-Null
            }
            catch {
                Write-Host "SKIP junction containment test: junction creation unavailable: $($_.Exception.Message)" -ForegroundColor Yellow
                return
            }

            Assert-Equal $true (Test-AilaPathWithinRoot -Path $realChild -Root $selectedRoot) 'real selected-root child'
            $escapedPath = Join-Path $junctionPath 'nonexistent-tail'
            Assert-Equal $false (Test-AilaPathWithinRoot -Path $escapedPath -Root $selectedRoot) 'junction escape path'
        }
        finally {
            if (Test-Path -LiteralPath $junctionPath) {
                Remove-Item -LiteralPath $junctionPath -Force
            }
        }
    }

    Invoke-Test 'rejects an in-repo build directory junction resolving outside the repo' {
        $outsideBuildRoot = Join-Path $tempRoot 'outside-build-junction-target'
        New-Item -ItemType Directory -Path $outsideBuildRoot -Force | Out-Null
        $sentinelPath = Join-Path $outsideBuildRoot 'must-survive.txt'
        Set-Content -LiteralPath $sentinelPath -Value 'keep' -Encoding UTF8
        $junctionPath = Join-Path $repoRoot ('.aila-build-junction-test-' + [guid]::NewGuid().ToString('N'))

        try {
            try {
                New-Item -ItemType Junction -Path $junctionPath -Target $outsideBuildRoot | Out-Null
            }
            catch {
                Write-Host "SKIP build-dir junction test: junction creation unavailable: $($_.Exception.Message)" -ForegroundColor Yellow
                return
            }

            $caught = $null
            try {
                Assert-AilaPathWithinRepo -RepoRoot $repoRoot -CandidatePath $junctionPath
            }
            catch {
                $caught = $_
            }
            if (-not (Test-Path -LiteralPath $sentinelPath -PathType Leaf)) {
                throw 'Build-dir containment guard removed data through the junction.'
            }
            if ($null -eq $caught) {
                throw "Build-dir containment guard accepted junction '$junctionPath' resolving outside '$repoRoot'."
            }
            if (-not ([string]$caught.Exception.Message).Contains('outside repo')) {
                throw "Build-dir containment guard returned unexpected error: $($caught.Exception.Message)"
            }
        }
        finally {
            if (Test-Path -LiteralPath $junctionPath) {
                Remove-Item -LiteralPath $junctionPath -Force
            }
        }
    }

    Invoke-Test 'removes the UMF transitive TCM path without clearing unrelated oneAPI paths' {
        $oneApiRoot = Join-Path $tempRoot 'oneapi-path-filter'
        $systemPath = Join-Path $tempRoot 'system-bin'
        $tcmPath = Join-Path $oneApiRoot 'tcm\latest\lib'
        $unrelatedPath = Join-Path $oneApiRoot 'mkl\latest\lib'
        $filtered = Remove-AilaManagedOneApiPathSegments -Value ([string]::Join(';', @($systemPath, $tcmPath, $unrelatedPath))) -OneApiRoot $oneApiRoot

        Assert-PathContainsSegment $filtered $systemPath 'oneAPI path filter system path'
        Assert-PathContainsSegment $filtered $unrelatedPath 'oneAPI path filter unrelated component'
        Assert-PathExcludesRoot $filtered (Join-Path $oneApiRoot 'tcm') 'oneAPI path filter transitive TCM path'
    }

    Invoke-Test 'sets process environment entries without leaking test state' {
        $variableName = "AILA_PERF_COMMON_TEST_$([guid]::NewGuid().ToString('N'))"
        $environmentSnapshot = Get-ProcessEnvironmentSnapshot
        try {
            [System.Environment]::SetEnvironmentVariable($variableName, 'before', 'Process')
            Set-AilaProcessEnvironment -Environment @{ $variableName = 'after' }
            Assert-Equal 'after' ([System.Environment]::GetEnvironmentVariable($variableName, 'Process')) 'process environment value'
        }
        finally {
            Restore-ProcessEnvironmentSnapshot -Snapshot $environmentSnapshot
        }
    }

    Invoke-Test 'clears a previously managed environment key when the next stack omits it' {
        $environmentSnapshot = Get-ProcessEnvironmentSnapshot
        try {
            Set-AilaProcessEnvironment -Environment @{ PATH = $env:PATH; DNNLROOT = 'stale-dnnl-root' }
            Set-AilaProcessEnvironment -Environment @{ PATH = $env:PATH }
            Assert-IsNull -Actual ([System.Environment]::GetEnvironmentVariable('DNNLROOT', 'Process')) -Message 'cleared managed DNNLROOT'
        }
        finally {
            Restore-ProcessEnvironmentSnapshot -Snapshot $environmentSnapshot
        }
    }

    Invoke-Test 'clears omitted managed keys after PerfCommon is dot-sourced again' {
        $environmentSnapshot = Get-ProcessEnvironmentSnapshot
        $sentinelName = "AILA_UNRELATED_SENTINEL_$([guid]::NewGuid().ToString('N'))"
        try {
            [System.Environment]::SetEnvironmentVariable($sentinelName, 'keep-me', 'Process')
            Set-AilaProcessEnvironment -Environment @{ PATH = $env:PATH; DNNLROOT = 'stale-after-redot' }
            . (Join-Path $repoRoot 'perf\PerfCommon.ps1')
            Set-AilaProcessEnvironment -Environment @{ PATH = $env:PATH }

            Assert-IsNull -Actual ([System.Environment]::GetEnvironmentVariable('DNNLROOT', 'Process')) -Message 're-dot-source cleared managed DNNLROOT'
            Assert-Equal 'keep-me' ([System.Environment]::GetEnvironmentVariable($sentinelName, 'Process')) 're-dot-source preserves unrelated sentinel'
        }
        finally {
            Restore-ProcessEnvironmentSnapshot -Snapshot $environmentSnapshot
        }
    }

    Invoke-Test 'switches baseline candidate baseline without stale oneAPI paths' {
        $environmentSnapshot = @{}
        [System.Environment]::GetEnvironmentVariables('Process').GetEnumerator() | ForEach-Object {
            $environmentSnapshot[[string]$_.Key] = [string]$_.Value
        }
        $pathVariables = @('PATH', 'LIB', 'INCLUDE', 'CMAKE_PREFIX_PATH', 'PKG_CONFIG_PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH')
        $rootVariables = @('CMPLR_ROOT', 'DNNLROOT', 'TBBROOT', 'UMF_ROOT', 'ONEAPI_ROOT')
        $expectedComponents = @{
            PATH                = @('compilerRoot', 'dnnlRoot', 'tbbRoot', 'umfRoot')
            LIB                 = @('compilerRoot', 'dnnlRoot', 'tbbRoot', 'umfRoot')
            INCLUDE             = @('compilerRoot', 'dnnlRoot', 'tbbRoot', 'umfRoot')
            CMAKE_PREFIX_PATH   = @('compilerRoot', 'dnnlRoot', 'tbbRoot')
            PKG_CONFIG_PATH     = @('compilerRoot', 'dnnlRoot', 'tbbRoot')
            CPATH               = @('umfRoot')
            C_INCLUDE_PATH      = @('compilerRoot', 'tbbRoot', 'umfRoot')
            CPLUS_INCLUDE_PATH  = @('compilerRoot', 'tbbRoot', 'umfRoot')
        }

        $assertSelectedStack = {
            param($Selected, $Other, [string]$Label)

            foreach ($variable in $pathVariables) {
                $value = [System.Environment]::GetEnvironmentVariable($variable, 'Process')
                Assert-PathSegmentsUnique $value "$Label $variable"
                foreach ($property in $expectedComponents[$variable]) {
                    Assert-PathContainsRoot $value ([string]$Selected.$property) "$Label $variable selected $property"
                }
                foreach ($property in @('compilerRoot', 'dnnlRoot', 'tbbRoot', 'umfRoot')) {
                    Assert-PathExcludesRoot $value ([string]$Other.$property) "$Label $variable"
                }
            }

            Assert-Equal (Get-NormalizedTestPath $Selected.compilerRoot) (Get-NormalizedTestPath ([System.Environment]::GetEnvironmentVariable('CMPLR_ROOT', 'Process'))) "$Label compiler root"
            Assert-Equal (Get-NormalizedTestPath $Selected.dnnlRoot) (Get-NormalizedTestPath ([System.Environment]::GetEnvironmentVariable('DNNLROOT', 'Process'))) "$Label oneDNN root"
            Assert-Equal (Get-NormalizedTestPath $Selected.tbbRoot) (Get-NormalizedTestPath ([System.Environment]::GetEnvironmentVariable('TBBROOT', 'Process'))) "$Label TBB root"
            Assert-Equal (Get-NormalizedTestPath $Selected.umfRoot) (Get-NormalizedTestPath ([System.Environment]::GetEnvironmentVariable('UMF_ROOT', 'Process'))) "$Label UMF root"
            $oneApiRoot = Split-Path -Parent (Split-Path -Parent $Selected.compilerRoot)
            Assert-Equal (Get-NormalizedTestPath $oneApiRoot) (Get-NormalizedTestPath ([System.Environment]::GetEnvironmentVariable('ONEAPI_ROOT', 'Process'))) "$Label oneAPI root"
        }

        try {
            $config = Get-AilaOneApiStackConfig -RepoRoot $repoRoot
            $baseline = Get-AilaOneApiStack -Config $config -Name 'oneapi-2025.3'
            $candidate = Get-AilaOneApiStack -Config $config -Name 'oneapi-2026.1'

            Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $baseline)
            & $assertSelectedStack $baseline $candidate 'baseline first switch'
            $baselineFirst = @{}
            foreach ($variable in ($pathVariables + $rootVariables)) {
                $baselineFirst[$variable] = [System.Environment]::GetEnvironmentVariable($variable, 'Process')
            }

            Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $candidate)
            & $assertSelectedStack $candidate $baseline 'candidate switch'

            Set-AilaProcessEnvironment -Environment (Get-AilaOneApiStackEnvironment -Stack $baseline)
            & $assertSelectedStack $baseline $candidate 'baseline second switch'
            foreach ($variable in ($pathVariables + $rootVariables)) {
                Assert-Equal $baselineFirst[$variable] ([System.Environment]::GetEnvironmentVariable($variable, 'Process')) "stable repeated baseline $variable"
            }
        }
        finally {
            $currentKeys = @([System.Environment]::GetEnvironmentVariables('Process').Keys | ForEach-Object { [string]$_ })
            foreach ($key in $currentKeys) {
                if (-not $environmentSnapshot.ContainsKey($key)) {
                    Restore-ProcessEnvironmentVariable -Name $key -Value $null
                }
            }
            foreach ($entry in $environmentSnapshot.GetEnumerator()) {
                Restore-ProcessEnvironmentVariable -Name $entry.Key -Value $entry.Value
            }
        }
    }
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $repoScratch) {
        Remove-Item -LiteralPath $repoScratch -Recurse -Force
    }
    if (Test-Path -LiteralPath $accuracyScratch) {
        Remove-Item -LiteralPath $accuracyScratch -Recurse -Force
    }
}

if ($script:TestFailures.Count -gt 0) {
    $details = ($script:TestFailures | ForEach-Object { " - $_" }) -join [System.Environment]::NewLine
    throw "PerfCommonTests FAIL$([System.Environment]::NewLine)$details"
}

Write-Host 'PerfCommonTests PASS'
