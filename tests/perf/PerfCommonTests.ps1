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
