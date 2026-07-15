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
$diagnosticBatchPath = Join-Path $tempRoot 'fail-with-diagnostic.bat'
Set-Content -LiteralPath $diagnosticBatchPath -Value "@echo off`r`n>&2 echo diagnostic-from-batch`r`nexit /b 23" -Encoding ASCII
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
        }
        Set-Content -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Encoding UTF8 -Value @(
            "CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $stack.compilerRoot 'bin\icx-cl.exe')"
            "dnnl_DIR:PATH=$(Join-Path $stack.dnnlRoot 'lib\cmake\dnnl')"
        )

        Assert-Throws {
            Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $buildDir -Stack $stack -RequireValues
        } 'missing TBB_DIR' 'missing configured TBB value'
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
        foreach ($fragment in @('exit code 23', 'diagnostic-from-batch', 'fail-with-diagnostic.bat')) {
            if (-not $message.Contains($fragment)) {
                throw "diagnostic batch expected message containing='$fragment' actual='$message'"
            }
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
        $previousValue = [System.Environment]::GetEnvironmentVariable($variableName, 'Process')
        try {
            [System.Environment]::SetEnvironmentVariable($variableName, 'before', 'Process')
            Set-AilaProcessEnvironment -Environment @{ $variableName = 'after' }
            Assert-Equal 'after' ([System.Environment]::GetEnvironmentVariable($variableName, 'Process')) 'process environment value'
        }
        finally {
            Restore-ProcessEnvironmentVariable -Name $variableName -Value $previousValue
        }
    }

    Invoke-Test 'clears a previously managed environment key when the next stack omits it' {
        $previousDnnlRoot = [System.Environment]::GetEnvironmentVariable('DNNLROOT', 'Process')
        try {
            Set-AilaProcessEnvironment -Environment @{ PATH = $env:PATH; DNNLROOT = 'stale-dnnl-root' }
            Set-AilaProcessEnvironment -Environment @{ PATH = $env:PATH }
            Assert-IsNull -Actual ([System.Environment]::GetEnvironmentVariable('DNNLROOT', 'Process')) -Message 'cleared managed DNNLROOT'
        }
        finally {
            Restore-ProcessEnvironmentVariable -Name 'DNNLROOT' -Value $previousDnnlRoot
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
}

if ($script:TestFailures.Count -gt 0) {
    $details = ($script:TestFailures | ForEach-Object { " - $_" }) -join [System.Environment]::NewLine
    throw "PerfCommonTests FAIL$([System.Environment]::NewLine)$details"
}

Write-Host 'PerfCommonTests PASS'
