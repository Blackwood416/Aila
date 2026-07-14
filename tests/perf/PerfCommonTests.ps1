$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\..\perf\PerfCommon.ps1')

$script:TestFailures = [System.Collections.Generic.List[string]]::new()

function Assert-Equal($Expected, $Actual, [string]$Message) {
    if ($Expected -ne $Actual) {
        throw "$Message expected='$Expected' actual='$Actual'"
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

$repoRoot = Get-AilaRepoRoot
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
$partialBatchPath = Join-Path $tempRoot 'missing-compiler-root.bat'
Set-Content -LiteralPath $partialBatchPath -Value "@echo off`r`nset CMPLR_ROOT=`r`nexit /b 0" -Encoding ASCII
$missingPathBatchPath = Join-Path $tempRoot 'missing-path.bat'
Set-Content -LiteralPath $missingPathBatchPath -Value "@echo off`r`nset PATH=`r`nset CMPLR_ROOT=C:\compiler`r`nexit /b 0" -Encoding ASCII

try {
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

    Invoke-Test 'sets process environment entries without leaking test state' {
        $variableName = "AILA_PERF_COMMON_TEST_$([guid]::NewGuid().ToString('N'))"
        $previousValue = [System.Environment]::GetEnvironmentVariable($variableName, 'Process')
        try {
            [System.Environment]::SetEnvironmentVariable($variableName, 'before', 'Process')
            Set-AilaProcessEnvironment -Environment @{ $variableName = 'after' }
            Assert-Equal 'after' ([System.Environment]::GetEnvironmentVariable($variableName, 'Process')) 'process environment value'
        }
        finally {
            [System.Environment]::SetEnvironmentVariable($variableName, $previousValue, 'Process')
        }
    }
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}

if ($script:TestFailures.Count -gt 0) {
    $details = ($script:TestFailures | ForEach-Object { " - $_" }) -join [System.Environment]::NewLine
    throw "PerfCommonTests FAIL$([System.Environment]::NewLine)$details"
}

Write-Host 'PerfCommonTests PASS'
