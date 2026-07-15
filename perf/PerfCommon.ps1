Set-StrictMode -Version Latest

$script:AilaOneApiInitialized = $false

function Get-AilaRepoRoot {
    $perfDir = Split-Path -Parent $PSScriptRoot
    return [System.IO.Path]::GetFullPath($perfDir)
}

function Get-AilaOneApiStackConfig {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$Path = 'perf\oneapi-stacks.json'
    )

    $fullPath = Resolve-AilaPath -RepoRoot $RepoRoot -Path $Path
    $config = Read-AilaJsonFile -Path $fullPath
    $schemaProperty = if ($null -eq $config) { $null } else { $config.PSObject.Properties['schemaVersion'] }
    if ($null -eq $schemaProperty) {
        throw 'Unsupported oneAPI stack config schema: <missing>'
    }

    $schemaVersion = $schemaProperty.Value
    if ($schemaVersion -ne 1) {
        throw "Unsupported oneAPI stack config schema: $schemaVersion"
    }
    return $config
}

function Get-AilaOneApiStack {
    param(
        [Parameter(Mandatory = $true)]$Config,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $stacksProperty = $Config.PSObject.Properties['stacks']
    if ($null -eq $stacksProperty -or $null -eq $stacksProperty.Value) {
        throw "oneAPI stack '$Name' not found."
    }

    $stackProperty = $stacksProperty.Value.PSObject.Properties[$Name]
    if ($null -eq $stackProperty -or $null -eq $stackProperty.Value) {
        throw "oneAPI stack '$Name' not found."
    }
    $stack = $stackProperty.Value

    foreach ($property in @('compilerRoot', 'dnnlRoot', 'tbbRoot', 'umfRoot')) {
        $rootProperty = $stack.PSObject.Properties[$property]
        if ($null -eq $rootProperty) {
            throw "oneAPI stack '$Name' is missing required root property '$property'."
        }

        $path = [string]$rootProperty.Value
        if ([string]::IsNullOrWhiteSpace($path)) {
            throw "oneAPI stack '$Name' has blank $property."
        }
        if (-not (Test-Path -LiteralPath $path -PathType Container)) {
            throw "oneAPI stack '$Name' $property is not a directory: '$path'."
        }
    }

    $result = [ordered]@{ name = $Name }
    foreach ($sourceProperty in $stack.PSObject.Properties) {
        $result[$sourceProperty.Name] = $sourceProperty.Value
    }
    return [pscustomobject]$result
}

function Get-AilaOneApiStackMetadata {
    param([Parameter(Mandatory = $true)]$Stack)

    $compiler = Join-Path $Stack.compilerRoot 'bin\icx-cl.exe'
    if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
        throw "Compiler executable not found: $compiler"
    }

    $versionText = (& $compiler --version 2>&1 | Out-String)
    $compilerExitCode = $LASTEXITCODE
    if ($compilerExitCode -ne 0) {
        throw "Compiler version query failed for $compiler with exit code $compilerExitCode."
    }

    $compilerVersionMatch = [regex]::Match($versionText, 'Version\s+(\d+\.\d+\.\d+)')
    if (-not $compilerVersionMatch.Success) {
        throw "Unable to parse compiler version from $compiler"
    }
    $compilerVersion = $compilerVersionMatch.Groups[1].Value
    if ($compilerVersion -ne [string]$Stack.expectedCompilerVersion) {
        throw "Compiler version mismatch for oneAPI stack '$($Stack.name)': expected '$($Stack.expectedCompilerVersion)', actual '$compilerVersion'."
    }

    $dnnlVersionFile = Join-Path $Stack.dnnlRoot 'lib\cmake\dnnl\dnnl-config-version.cmake'
    if (-not (Test-Path -LiteralPath $dnnlVersionFile -PathType Leaf)) {
        throw "oneDNN version file not found: $dnnlVersionFile"
    }

    $dnnlText = Get-Content -LiteralPath $dnnlVersionFile -Raw
    $dnnlVersionMatch = [regex]::Match($dnnlText, 'set\(PACKAGE_VERSION\s+"([^"]+)"\)')
    if (-not $dnnlVersionMatch.Success) {
        throw "Unable to parse oneDNN version from $dnnlVersionFile"
    }
    $dnnlVersion = $dnnlVersionMatch.Groups[1].Value
    if ($dnnlVersion -ne [string]$Stack.expectedDnnlVersion) {
        throw "oneDNN version mismatch for oneAPI stack '$($Stack.name)': expected '$($Stack.expectedDnnlVersion)', actual '$dnnlVersion'."
    }

    return [pscustomobject]@{
        name                = $Stack.name
        role                = $Stack.role
        compilerPath        = $compiler
        compilerVersion     = $compilerVersion
        dnnlRoot            = [string]$Stack.dnnlRoot
        dnnlVersion         = $dnnlVersion
        tbbRoot             = [string]$Stack.tbbRoot
        tbbVersion          = Split-Path $Stack.tbbRoot -Leaf
        umfRoot             = [string]$Stack.umfRoot
        expectedSyclDll     = [string]$Stack.expectedSyclDll
        allowLegacyCompiler = [bool]$Stack.allowLegacyCompiler
    }
}

function Assert-AilaBuildInfoMatchesOneApiStack {
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)]$Stack
    )

    $buildInfoPath = Join-Path $BuildDir 'build_info.json'
    if (-not (Test-Path -LiteralPath $buildInfoPath -PathType Leaf)) {
        return
    }

    $buildInfo = Read-AilaJsonFile -Path $buildInfoPath
    $oneApiProperty = $buildInfo.PSObject.Properties['oneApi']
    if ($null -eq $oneApiProperty -or $null -eq $oneApiProperty.Value) {
        return
    }

    $nameProperty = $oneApiProperty.Value.PSObject.Properties['name']
    if ($null -eq $nameProperty -or [string]::IsNullOrWhiteSpace([string]$nameProperty.Value)) {
        return
    }

    $existingName = [string]$nameProperty.Value
    if ($existingName -ne [string]$Stack.name) {
        throw "Build directory '$BuildDir' contains oneAPI stack '$existingName', but '$($Stack.name)' was requested. Use -Clean or a different BuildDir."
    }
}

function Assert-AilaCMakeCacheMatchesOneApiStack {
    param(
        [Parameter(Mandatory = $true)][string]$BuildDir,
        [Parameter(Mandatory = $true)]$Stack,
        [switch]$RequireValues
    )

    $cachePath = Join-Path $BuildDir 'CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf)) {
        if ($RequireValues) {
            throw "Configured build directory '$BuildDir' is missing CMakeCache.txt."
        }
        return
    }

    $buildMeta = Get-AilaBuildMetadata -BuildDir $BuildDir
    $checks = @(
        [pscustomobject]@{ key = 'CMAKE_CXX_COMPILER'; value = $buildMeta.compiler; root = [string]$Stack.compilerRoot },
        [pscustomobject]@{ key = 'dnnl_DIR'; value = Read-AilaCMakeCacheValue -CachePath $cachePath -Key 'dnnl_DIR'; root = [string]$Stack.dnnlRoot },
        [pscustomobject]@{ key = 'TBB_DIR'; value = Read-AilaCMakeCacheValue -CachePath $cachePath -Key 'TBB_DIR'; root = [string]$Stack.tbbRoot }
    )

    foreach ($check in $checks) {
        $value = [string]$check.value
        if ([string]::IsNullOrWhiteSpace($value)) {
            if ($RequireValues) {
                throw "Configured build directory '$BuildDir' is missing $($check.key)."
            }
            continue
        }

        if (-not (Test-AilaPathWithinRoot -Path $value -Root $check.root)) {
            throw "Build directory '$BuildDir' has $($check.key) '$value' outside selected oneAPI stack '$($Stack.name)' root '$($check.root)'. Use -Clean or a different BuildDir."
        }
    }

    $legacyKey = 'AILA_ALLOW_LEGACY_ONEAPI_BASELINE'
    $legacyValue = [string](Read-AilaCMakeCacheValue -CachePath $cachePath -Key $legacyKey)
    if ([string]::IsNullOrWhiteSpace($legacyValue)) {
        if ($RequireValues) {
            throw "Configured build directory '$BuildDir' is missing $legacyKey."
        }
        return
    }

    $normalizedLegacyValue = switch -Regex ($legacyValue.Trim().ToUpperInvariant()) {
        '^(1|ON|TRUE|YES|Y)$' { 'ON'; break }
        '^(0|OFF|FALSE|NO|N)$' { 'OFF'; break }
        default { $null }
    }
    $expectedLegacyValue = if ([bool]$Stack.allowLegacyCompiler) { 'ON' } else { 'OFF' }
    if ($normalizedLegacyValue -ne $expectedLegacyValue) {
        throw "Build directory '$BuildDir' has $legacyKey '$legacyValue', but selected oneAPI stack '$($Stack.name)' requires '$expectedLegacyValue'. Use -Clean or a different BuildDir."
    }
}

function Get-AilaWindowsPathKey {
    param([Parameter(Mandatory = $true)][string]$Path)

    $cleaned = $Path.Trim().Trim('"').Replace('/', '\')
    return [System.IO.Path]::GetFullPath($cleaned).TrimEnd('\')
}

function Get-AilaCanonicalWindowsPathKey {
    param([Parameter(Mandatory = $true)][string]$Path)

    $cleaned = $Path.Trim().Trim('"').Replace('/', '\')
    $fullPath = [System.IO.Path]::GetFullPath($cleaned)
    $pathRoot = [System.IO.Path]::GetPathRoot($fullPath)
    if ([string]::IsNullOrWhiteSpace($pathRoot)) {
        throw "Path has no filesystem root: $Path"
    }

    $current = $pathRoot
    $relative = $fullPath.Substring($pathRoot.Length)
    $segments = @($relative -split '[\\/]' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    for ($index = 0; $index -lt $segments.Count; $index++) {
        $candidate = Join-Path $current $segments[$index]
        try {
            $item = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
        }
        catch [System.Management.Automation.ItemNotFoundException] {
            for ($tailIndex = $index; $tailIndex -lt $segments.Count; $tailIndex++) {
                $current = Join-Path $current $segments[$tailIndex]
            }
            break
        }
        catch {
            throw "Failed to inspect path component '$candidate': $($_.Exception.Message)"
        }

        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            try {
                $target = $item.ResolveLinkTarget($true)
            }
            catch {
                throw "Failed to resolve reparse point '$candidate': $($_.Exception.Message)"
            }
            if ($null -eq $target) {
                throw "Failed to resolve reparse point '$candidate'."
            }
            $current = [System.IO.Path]::GetFullPath($target.FullName)
            continue
        }

        $current = $item.FullName
    }

    return [System.IO.Path]::TrimEndingDirectorySeparator([System.IO.Path]::GetFullPath($current))
}

function Test-AilaCanonicalPathKeyWithinRootKey {
    param(
        [Parameter(Mandatory = $true)][string]$PathKey,
        [Parameter(Mandatory = $true)][string]$RootKey
    )

    $rootPrefix = if ([System.IO.Path]::EndsInDirectorySeparator($RootKey)) { $RootKey } else { $RootKey + '\' }
    return $PathKey.Equals($RootKey, [System.StringComparison]::OrdinalIgnoreCase) -or
        $PathKey.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-AilaPathWithinRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $pathKey = Get-AilaCanonicalWindowsPathKey -Path $Path
    $rootKey = Get-AilaCanonicalWindowsPathKey -Path $Root
    return Test-AilaCanonicalPathKeyWithinRootKey -PathKey $pathKey -RootKey $rootKey
}

function Get-AilaUniquePathSegments {
    param([Parameter(Mandatory = $true)][string[]]$Segments)

    $result = New-Object System.Collections.Generic.List[string]
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($segment in $Segments) {
        if ([string]::IsNullOrWhiteSpace($segment)) {
            continue
        }

        $trimmed = $segment.Trim()
        if ($seen.Add((Get-AilaWindowsPathKey -Path $trimmed))) {
            $result.Add($trimmed)
        }
    }
    return $result.ToArray()
}

function Get-AilaOneApiPathListVariableNames {
    return @('PATH', 'LIB', 'INCLUDE', 'CMAKE_PREFIX_PATH', 'PKG_CONFIG_PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH')
}

function Get-AilaOneApiRootVariableNames {
    return @('CMPLR_ROOT', 'DNNLROOT', 'TBBROOT', 'UMF_ROOT', 'ONEAPI_ROOT')
}

function Remove-AilaManagedOneApiPathSegments {
    param(
        [AllowEmptyString()][string]$Value,
        [Parameter(Mandatory = $true)][string]$OneApiRoot
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ''
    }

    $familyRoots = @('compiler', 'dnnl', 'tbb', 'umf', 'tcm') | ForEach-Object { Join-Path $OneApiRoot $_ }
    $kept = New-Object System.Collections.Generic.List[string]
    foreach ($segment in ($Value -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        $isManaged = $false
        foreach ($familyRoot in $familyRoots) {
            if (Test-AilaPathWithinRoot -Path $segment -Root $familyRoot) {
                $isManaged = $true
                break
            }
        }
        if (-not $isManaged) {
            $kept.Add($segment.Trim())
        }
    }

    if ($kept.Count -eq 0) {
        return ''
    }
    return [string]::Join(';', (Get-AilaUniquePathSegments -Segments $kept.ToArray()))
}

function Get-AilaBoundedDiagnosticText {
    param(
        [AllowNull()][string]$Text,
        [ValidateRange(64, 65536)][int]$MaxLength = 2048
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return '<empty>'
    }

    $trimmed = $Text.Trim()
    if ($trimmed.Length -le $MaxLength) {
        return $trimmed
    }

    $omitted = $trimmed.Length - $MaxLength
    return $trimmed.Substring(0, $MaxLength) + "... <truncated $omitted characters>"
}

function Import-AilaBatchEnvironment {
    param([Parameter(Mandatory = $true)][string[]]$Scripts)

    $vsDevCmd = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat'
    if (-not (Test-Path -LiteralPath $vsDevCmd)) {
        throw "Visual Studio environment script not found: $vsDevCmd"
    }

    $calls = New-Object System.Collections.Generic.List[string]
    $calls.Add("call `"$vsDevCmd`" amd64 >nul")
    $oneApiRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Scripts[0])))
    $calls.Add("set `"ONEAPI_ROOT=$oneApiRoot`"")
    foreach ($script in $Scripts) {
        if (-not (Test-Path -LiteralPath $script)) {
            throw "Environment script not found: $script"
        }
        $calls.Add("call `"$script`" >nul")
    }
    $calls.Add('set')

    $cmdPath = Join-Path $env:SystemRoot 'System32\cmd.exe'
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $cmdPath
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    foreach ($key in (Get-AilaOneApiPathListVariableNames)) {
        if ($startInfo.Environment.ContainsKey($key)) {
            $startInfo.Environment[$key] = Remove-AilaManagedOneApiPathSegments -Value $startInfo.Environment[$key] -OneApiRoot $oneApiRoot
        }
    }
    foreach ($key in (Get-AilaOneApiRootVariableNames)) {
        $startInfo.Environment.Remove($key) | Out-Null
    }
    $compoundCommand = $calls -join ' && '
    $startInfo.Arguments = '/d /s /c "' + $compoundCommand + '"'

    $result = @{}
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start batch environment process: $cmdPath"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $standardOutput = $stdoutTask.GetAwaiter().GetResult()
    $standardError = $stderrTask.GetAwaiter().GetResult()
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) {
        $scriptContext = [string]::Join(', ', @($Scripts | ForEach-Object { "'$_'" }))
        $outputContext = Get-AilaBoundedDiagnosticText -Text $standardOutput
        $errorContext = Get-AilaBoundedDiagnosticText -Text $standardError
        throw "Batch environment import failed with exit code $exitCode while running cmd.exe /d /s /c for scripts: $scriptContext. stdout: $outputContext. stderr: $errorContext"
    }

    $standardOutput -split '\r?\n' | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $result[$matches[1]] = $matches[2]
        }
    }
    foreach ($required in @('PATH', 'CMPLR_ROOT')) {
        if (-not $result.ContainsKey($required) -or [string]::IsNullOrWhiteSpace([string]$result[$required])) {
            throw "Batch environment import missing required variable: $required"
        }
    }
    return $result
}

function Get-AilaOneApiStackEnvironment {
    param([Parameter(Mandatory = $true)]$Stack)

    $scripts = @(
        (Join-Path $Stack.umfRoot 'env\vars.bat'),
        (Join-Path $Stack.tbbRoot 'env\vars.bat'),
        (Join-Path $Stack.dnnlRoot 'env\vars.bat'),
        (Join-Path $Stack.compilerRoot 'env\vars.bat')
    )
    $envMap = Import-AilaBatchEnvironment -Scripts $scripts
    $umfRoot = [System.IO.Path]::GetFullPath([string]$Stack.umfRoot)
    $umfLatest = Join-Path (Split-Path -Parent $umfRoot) 'latest'
    $umfLatestPattern = [regex]::Escape($umfLatest)
    foreach ($key in (Get-AilaOneApiPathListVariableNames)) {
        if ($envMap.ContainsKey($key)) {
            $normalizedValue = [regex]::Replace(
                [string]$envMap[$key],
                $umfLatestPattern,
                $umfRoot,
                [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
            )
            $segments = @($normalizedValue -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            $envMap[$key] = if ($segments.Count -eq 0) { '' } else { [string]::Join(';', (Get-AilaUniquePathSegments -Segments $segments)) }
        }
    }
    $envMap['UMF_ROOT'] = $umfRoot

    $compilerRoot = [System.IO.Path]::GetFullPath([string]$Stack.compilerRoot)
    $compilerFamilyRoot = Split-Path -Parent $compilerRoot
    $inheritedPath = @($envMap['PATH'] -split ';' | Where-Object {
        if ([string]::IsNullOrWhiteSpace($_)) {
            return $false
        }

        return -not (Test-AilaPathWithinRoot -Path $_ -Root $compilerFamilyRoot) -or
            (Test-AilaPathWithinRoot -Path $_ -Root $compilerRoot)
    })
    $pathSegments = @(
        (Join-Path $Stack.compilerRoot 'bin'),
        (Join-Path $Stack.dnnlRoot 'bin'),
        (Join-Path $Stack.tbbRoot 'bin'),
        (Join-Path $Stack.umfRoot 'bin')
    ) + $inheritedPath
    $envMap['PATH'] = [string]::Join(';', (Get-AilaUniquePathSegments -Segments $pathSegments))
    return $envMap
}

function Set-AilaProcessEnvironment {
    param([Parameter(Mandatory = $true)][hashtable]$Environment)

    foreach ($key in ((Get-AilaOneApiPathListVariableNames) + (Get-AilaOneApiRootVariableNames))) {
        if (-not $Environment.ContainsKey($key)) {
            [System.Environment]::SetEnvironmentVariable($key, [System.Management.Automation.Language.NullString]::Value, 'Process')
        }
    }
    foreach ($entry in $Environment.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, 'Process')
    }
}

function Resolve-AilaPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function Assert-AilaPathWithinRepo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$CandidatePath
    )

    $candidateFull = [System.IO.Path]::GetFullPath($CandidatePath)
    $repoKey = Get-AilaCanonicalWindowsPathKey -Path $RepoRoot
    $candidateKey = Get-AilaCanonicalWindowsPathKey -Path $candidateFull

    if ($candidateKey.Equals($repoKey, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-AilaCanonicalPathKeyWithinRootKey -PathKey $candidateKey -RootKey $repoKey)) {
        throw "Refusing to operate on path outside repo: $candidateFull (resolved: $candidateKey)"
    }

    return $candidateFull
}

function Ensure-AilaDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Initialize-AilaOneApiEnvironment {
    [System.Environment]::SetEnvironmentVariable('MSYS2_ARG_CONV_EXCL', '*', 'Process')

    if ($script:AilaOneApiInitialized) {
        return
    }

    $setvars = 'C:\Program Files (x86)\Intel\oneAPI\setvars.bat'
    $vsInstaller = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer'
    if (-not (Test-Path -LiteralPath $setvars)) {
        throw "oneAPI setvars.bat not found: $setvars"
    }

    cmd /c "set `"PATH=.;$vsInstaller;%PATH%`" && call `"$setvars`" && set" |
        ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
            }
        }

    $script:AilaOneApiInitialized = $true
    Write-Host ':: oneAPI environment initialized ::' -ForegroundColor Green
}

function Get-AilaGitInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $short = (git -C $RepoRoot rev-parse --short HEAD).Trim()
    $full = (git -C $RepoRoot rev-parse HEAD).Trim()
    $branch = (git -C $RepoRoot rev-parse --abbrev-ref HEAD).Trim()

    return [pscustomobject]@{
        shortCommit = $short
        fullCommit  = $full
        branch      = $branch
    }
}

function Read-AilaCMakeCacheValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CachePath,

        [Parameter(Mandatory = $true)]
        [string]$Key
    )

    if (-not (Test-Path -LiteralPath $CachePath)) {
        return $null
    }

    $escaped = [regex]::Escape($Key)
    $match = Select-String -Path $CachePath -Pattern "^$escaped(:[^=]+)?=(.*)$" | Select-Object -First 1
    if (-not $match) {
        return $null
    }

    return $match.Matches[0].Groups[2].Value
}

function Read-AilaCMakeScriptValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string]$Key
    )

    if (-not (Test-Path -LiteralPath $FilePath)) {
        return $null
    }

    $escaped = [regex]::Escape($Key)
    $match = Select-String -Path $FilePath -Pattern "^\s*set\($escaped\s+`"(?<value>.*)`"\)\s*$" | Select-Object -First 1
    if (-not $match) {
        return $null
    }

    return $match.Matches[0].Groups['value'].Value
}

function Get-AilaCompilerPathFromBuildDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BuildDir
    )

    $compilerFile = Get-ChildItem -Path (Join-Path $BuildDir 'CMakeFiles') -Recurse -Filter 'CMakeCXXCompiler.cmake' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $compilerFile) {
        return $null
    }

    return Read-AilaCMakeScriptValue -FilePath $compilerFile.FullName -Key 'CMAKE_CXX_COMPILER'
}

function Get-AilaBuildMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BuildDir
    )

    $cachePath = Join-Path $BuildDir 'CMakeCache.txt'
    $compiler = Read-AilaCMakeCacheValue -CachePath $cachePath -Key 'CMAKE_CXX_COMPILER'
    if ([string]::IsNullOrWhiteSpace($compiler)) {
        $compiler = Get-AilaCompilerPathFromBuildDir -BuildDir $BuildDir
    }

    return [pscustomobject]@{
        buildDir   = $BuildDir
        cachePath  = $cachePath
        buildType  = Read-AilaCMakeCacheValue -CachePath $cachePath -Key 'CMAKE_BUILD_TYPE'
        compiler   = $compiler
        generator  = Read-AilaCMakeCacheValue -CachePath $cachePath -Key 'CMAKE_GENERATOR'
    }
}

function Get-AilaPerfConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [string]$PresetsFile = 'perf\presets.json'
    )

    $fullPath = Resolve-AilaPath -RepoRoot $RepoRoot -Path $PresetsFile
    if (-not (Test-Path -LiteralPath $fullPath)) {
        throw "Perf preset file not found: $fullPath"
    }

    return Get-Content -LiteralPath $fullPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-AilaPreset {
    param(
        [Parameter(Mandatory = $true)]
        $Config,

        [Parameter(Mandatory = $true)]
        [string]$PresetName
    )

    $preset = $Config.presets.$PresetName
    if ($null -eq $preset) {
        throw "Perf preset '$PresetName' not found."
    }

    return $preset
}

function Get-AilaModelInfo {
    param(
        [Parameter(Mandatory = $true)]
        $Config,

        [Parameter(Mandatory = $true)]
        [string]$Alias,

        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $model = $Config.models.$Alias
    if ($null -eq $model) {
        throw "Model alias '$Alias' not found in perf config."
    }

    return [pscustomobject]@{
        alias      = $Alias
        path       = Resolve-AilaPath -RepoRoot $RepoRoot -Path $model.path
        maxSeqLen  = $model.maxSeqLen
        description = $model.description
    }
}

function New-AilaOutputDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$OutputRoot,

        [Parameter(Mandatory = $true)]
        [string]$Phase,

        [Parameter(Mandatory = $true)]
        [string]$ShortCommit
    )

    $root = Resolve-AilaPath -RepoRoot $RepoRoot -Path $OutputRoot
    $dir = Join-Path (Join-Path $root $Phase) $ShortCommit
    Ensure-AilaDirectory -Path $dir
    return $dir
}

function Write-AilaJsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        $Data
    )

    $parent = Split-Path -Parent $Path
    if ($parent) {
        Ensure-AilaDirectory -Path $parent
    }
    $Data | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Read-AilaJsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "JSON file not found: $Path"
    }

    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Invoke-AilaProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,

        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,

        [string]$LogPath,

        [hashtable]$EnvOverrides = @{},

        [ValidateRange(0, 86400)]
        [int]$TimeoutSeconds = 0
    )

    $previousEnv = @{}
    foreach ($key in $EnvOverrides.Keys) {
        $previousEnv[$key] = [System.Environment]::GetEnvironmentVariable($key, 'Process')
        [System.Environment]::SetEnvironmentVariable($key, [string]$EnvOverrides[$key], 'Process')
    }

    if ($LogPath) {
        $parent = Split-Path -Parent $LogPath
        if ($parent) {
            Ensure-AilaDirectory -Path $parent
        }
        Set-Content -LiteralPath $LogPath -Value '' -Encoding UTF8
        Add-Content -LiteralPath $LogPath -Value ("[{0}] process starting: {1} {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Executable, ($ArgumentList -join ' ')) -Encoding UTF8
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $captured = New-Object System.Collections.Generic.List[string]
    $process = $null
    $stdout = ''
    $stderr = ''
    $exitCode = $null
    $timedOut = $false

    try {
        $process = New-Object System.Diagnostics.Process
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = Resolve-AilaExecutablePath -WorkingDirectory $WorkingDirectory -Executable $Executable
        $startInfo.WorkingDirectory = $WorkingDirectory
        $startInfo.UseShellExecute = $false
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $startInfo.Arguments = Join-AilaProcessArguments -ArgumentList $ArgumentList
        $process.StartInfo = $startInfo
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $completed = if ($TimeoutSeconds -gt 0) {
            $process.WaitForExit([int]([math]::Min([int]::MaxValue, [int64]$TimeoutSeconds * 1000)))
        }
        else {
            $process.WaitForExit()
            $true
        }
        if (-not $completed) {
            $timedOut = $true
            try {
                $process.Kill($true)
            }
            catch {
                try { $process.Kill() } catch { }
            }
            $process.WaitForExit()
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode

        $mergedText = (($stdout, $stderr) | Where-Object { -not [string]::IsNullOrEmpty($_) }) -join "`n"
        foreach ($line in ($mergedText -split "`r?`n")) {
            if ([string]::IsNullOrEmpty($line)) {
                continue
            }
            $captured.Add($line)
            if ($LogPath) {
                Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
            }
            Write-Host $line
        }
    }
    finally {
        $stopwatch.Stop()
        foreach ($key in $EnvOverrides.Keys) {
            [System.Environment]::SetEnvironmentVariable($key, $previousEnv[$key], 'Process')
        }
    }

    $outputText = [string]::Join("`n", $captured.ToArray())
    return [pscustomobject]@{
        exitCode     = $exitCode
        stdoutText   = $stdout
        stderrText   = $stderr
        outputLines  = $captured.ToArray()
        outputText   = $outputText
        logPath      = $LogPath
        durationMs   = [math]::Round($stopwatch.Elapsed.TotalMilliseconds, 3)
        timedOut     = $timedOut
        timeoutSeconds = $TimeoutSeconds
        commandLine  = ($Executable + ' ' + ($ArgumentList -join ' ')).Trim()
        envOverrides = [pscustomobject]$EnvOverrides
    }
}

function Resolve-AilaExecutablePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,

        [Parameter(Mandatory = $true)]
        [string]$Executable
    )

    if ([System.IO.Path]::IsPathRooted($Executable)) {
        return [System.IO.Path]::GetFullPath($Executable)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $WorkingDirectory $Executable))
}

function Join-AilaProcessArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    $escapedArgs = foreach ($arg in $ArgumentList) {
        Format-AilaProcessArgument -Value $arg
    }

    return [string]::Join(' ', $escapedArgs)
}

function Format-AilaProcessArgument {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Normalize-AilaText {
    param([AllowEmptyString()][string]$Text)

    return (($Text.ToLowerInvariant() -replace '[\p{P}\p{S}\s]+', ' ').Trim())
}

function Parse-AilaAsrOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputText
    )

    $metricPattern = '(?m)^\s*\[Audio:\s*(?<audio>\d+(?:\.\d+)?)s,\s*Latency:\s*(?<latency>\d+(?:\.\d+)?)ms,\s*Speed:\s*(?<speed>\d+(?:\.\d+)?)x,\s*(?<tokps>\d+(?:\.\d+)?)\s+tok/s\]\s*$'
    $metricMatch = [regex]::Match($OutputText, $metricPattern)
    if (-not $metricMatch.Success) {
        throw 'Failed to parse ASR metrics from Aila output.'
    }

    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    try {
        $audioSeconds = [double]::Parse($metricMatch.Groups['audio'].Value, $culture)
        $latencyMs = [double]::Parse($metricMatch.Groups['latency'].Value, $culture)
        $speed = [double]::Parse($metricMatch.Groups['speed'].Value, $culture)
        $tokensPerSecond = [double]::Parse($metricMatch.Groups['tokps'].Value, $culture)
    }
    catch {
        throw "Failed to parse ASR metrics from Aila output: $($_.Exception.Message)"
    }

    $languageMatch = [regex]::Match($OutputText, '(?m)^\s*\[Language\]\s*(?<language>[^\r\n]*)\s*$')
    $transcriptLines = New-Object System.Collections.Generic.List[string]
    foreach ($line in ($OutputText -split '\r?\n')) {
        if ($line -match '^\s*\[[^\]]+\]') {
            continue
        }
        $transcriptLines.Add($line)
    }

    return [pscustomobject]@{
        text            = ([string]::Join("`n", $transcriptLines.ToArray())).Trim()
        language        = if ($languageMatch.Success) { $languageMatch.Groups['language'].Value.Trim() } else { '' }
        audioSeconds    = $audioSeconds
        latencyMs       = $latencyMs
        speed           = $speed
        tokensPerSecond = $tokensPerSecond
    }
}

function Parse-AilaAlignmentOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputText
    )

    $pattern = '(?m)^\s*"(?<text>.*)"\s+(?<start>\d+)ms\s+-\s+(?<end>\d+)ms\s*$'
    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($match in [regex]::Matches($OutputText, $pattern)) {
        $rows.Add([pscustomobject]@{
            text    = $match.Groups['text'].Value
            startMs = [int]$match.Groups['start'].Value
            endMs   = [int]$match.Groups['end'].Value
        })
    }
    return $rows.ToArray()
}

function Get-AilaWavMetadata {
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    $reader = [System.IO.BinaryReader]::new($stream)
    try {
        if ($stream.Length -lt 12) { throw "WAV output is too small: $Path" }
        $riff = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
        $riffSize = [long]$reader.ReadUInt32()
        $wave = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
        if ($riff -ne 'RIFF' -or $wave -ne 'WAVE') { throw "Invalid WAV RIFF/WAVE header: $Path" }
        if ($riffSize -lt 4 -or $riffSize + 8 -ne $stream.Length) { throw "Invalid WAV RIFF chunk bounds: $Path" }

        $fmt = $null
        $dataOffset = $null
        $dataBytes = $null
        while ($stream.Position + 8 -le $stream.Length) {
            $chunkId = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
            $chunkSize = [long]$reader.ReadUInt32()
            $chunkStart = $stream.Position
            if ($chunkStart + $chunkSize -gt $stream.Length) {
                throw "Invalid WAV chunk bounds: $Path"
            }
            if ($chunkId -eq 'fmt ') {
                if ($chunkSize -lt 16) { throw "Invalid WAV fmt chunk: $Path" }
                $fmt = [ordered]@{
                    audioFormat   = [int]$reader.ReadUInt16()
                    channels      = [int]$reader.ReadUInt16()
                    sampleRate    = [long]$reader.ReadUInt32()
                    byteRate      = [long]$reader.ReadUInt32()
                    blockAlign    = [int]$reader.ReadUInt16()
                    bitsPerSample = [int]$reader.ReadUInt16()
                }
            }
            elseif ($chunkId -eq 'data' -and $null -eq $dataOffset) {
                $dataOffset = $chunkStart
                $dataBytes = $chunkSize
            }
            $next = $chunkStart + $chunkSize + ($chunkSize % 2)
            if ($next -gt $stream.Length) { throw "Invalid WAV padded chunk bounds: $Path" }
            $stream.Position = $next
        }

        if ($null -eq $fmt -or $null -eq $dataOffset -or $null -eq $dataBytes) {
            throw "WAV output is missing fmt or data chunk: $Path"
        }
        if ($fmt.audioFormat -ne 1 -and $fmt.audioFormat -ne 3) {
            throw "Unsupported WAV format '$($fmt.audioFormat)': $Path"
        }
        if ($fmt.channels -le 0 -or $fmt.sampleRate -le 0) {
            throw "WAV channels and sampleRate must be positive: $Path"
        }
        $supportedBits = if ($fmt.audioFormat -eq 1) { @(8, 16, 24, 32) } else { @(32, 64) }
        if ($supportedBits -notcontains $fmt.bitsPerSample) {
            throw "Unsupported WAV bitsPerSample '$($fmt.bitsPerSample)': $Path"
        }
        $expectedBlockAlign = [int64]$fmt.channels * [int64]($fmt.bitsPerSample / 8)
        $expectedByteRate = [int64]$fmt.sampleRate * $expectedBlockAlign
        if ($fmt.blockAlign -ne $expectedBlockAlign) {
            throw "WAV blockAlign mismatch: expected $expectedBlockAlign, actual $($fmt.blockAlign): $Path"
        }
        if ($fmt.byteRate -ne $expectedByteRate) {
            throw "WAV byteRate mismatch: expected $expectedByteRate, actual $($fmt.byteRate): $Path"
        }
        if ($dataBytes -le 0 -or $dataBytes % $fmt.blockAlign -ne 0) {
            throw "WAV data chunk is empty or not frame-aligned: $Path"
        }

        $frameCount = [long]($dataBytes / $fmt.blockAlign)
        $duration = [double]$frameCount / [double]$fmt.sampleRate
        if ([double]::IsNaN($duration) -or [double]::IsInfinity($duration) -or $duration -le 0) {
            throw "WAV output has invalid duration: $Path"
        }
        if ($fmt.audioFormat -eq 3) {
            $stream.Position = $dataOffset
            $sampleCount = [long]$dataBytes / [long]($fmt.bitsPerSample / 8)
            for ($index = 0L; $index -lt $sampleCount; $index++) {
                $sample = if ($fmt.bitsPerSample -eq 32) { [double]$reader.ReadSingle() } else { $reader.ReadDouble() }
                if ([double]::IsNaN($sample) -or [double]::IsInfinity($sample)) {
                    throw "WAV float sample is non-finite at index ${index}: $Path"
                }
            }
        }

        $fmt['dataBytes'] = [long]$dataBytes
        $fmt['frameCount'] = $frameCount
        $fmt['durationSeconds'] = [math]::Round($duration, 6)
        return [pscustomobject]$fmt
    }
    finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function Parse-AilaBenchmarkOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputText
    )

    $results = @{}
    $pattern = '(?m)^\s*(pp|tg)\s+(?<tokens>\d+)\s+(?<tokps>[\d.]+)\s+(?<mspt>[\d.]+)\s+(?<msrun>[\d.]+)\s+(?<stddev>[\d.]+)\s*$'
    foreach ($match in [regex]::Matches($OutputText, $pattern)) {
        $name = $match.Groups[1].Value
        $results[$name] = [pscustomobject]@{
            tokens          = [int]$match.Groups['tokens'].Value
            tokPerSec       = [double]$match.Groups['tokps'].Value
            msPerToken      = [double]$match.Groups['mspt'].Value
            msPerRun        = [double]$match.Groups['msrun'].Value
            stddev          = [double]$match.Groups['stddev'].Value
        }
    }

    if (-not $results.ContainsKey('pp') -or -not $results.ContainsKey('tg')) {
        throw "Failed to parse benchmark output."
    }

    return [pscustomobject]@{
        prefill = $results['pp']
        decode  = $results['tg']
    }
}

function Parse-AilaDecodeProfiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputText
    )

    $pattern = '\[Q35DecodeProfile\] tokens=(?<tokens>\d+) total=(?<total>[\d.]+) embed=(?<embed>[\d.]+) linear_proj=(?<linear_proj>[\d.]+) linear_delta=(?<linear_delta>[\d.]+) linear_o=(?<linear_o>[\d.]+) full_qkv=(?<full_qkv>[\d.]+) full_split=(?<full_split>[\d.]+) qk_rope=(?<qk_rope>[\d.]+) kv_copy=(?<kv_copy>[\d.]+) attn=(?<attn>[\d.]+) attn_gate=(?<attn_gate>[\d.]+) full_o=(?<full_o>[\d.]+) post_attn=(?<post_attn>[\d.]+) ffn_proj=(?<ffn_proj>[\d.]+) ffn_act=(?<ffn_act>[\d.]+) down=(?<down>[\d.]+) post_mlp=(?<post_mlp>[\d.]+) lm_head=(?<lm_head>[\d.]+)'
    $stageNames = @(
        'total', 'embed', 'linear_proj', 'linear_delta', 'linear_o',
        'full_qkv', 'full_split', 'qk_rope', 'kv_copy', 'attn', 'attn_gate',
        'full_o', 'post_attn', 'ffn_proj', 'ffn_act', 'down', 'post_mlp', 'lm_head'
    )

    $samples = New-Object System.Collections.Generic.List[object]
    foreach ($match in [regex]::Matches($OutputText, $pattern)) {
        $sample = [ordered]@{
            tokens = [int]$match.Groups['tokens'].Value
        }
        foreach ($stage in $stageNames) {
            $sample[$stage] = [double]$match.Groups[$stage].Value
        }
        $samples.Add([pscustomobject]$sample)
    }

    return $samples.ToArray()
}

function Get-AilaDecodeProfileSummary {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Samples
    )

    if ($Samples.Count -eq 0) {
        throw 'No decode profile samples were parsed.'
    }

    $stageNames = @(
        'total', 'embed', 'linear_proj', 'linear_delta', 'linear_o',
        'full_qkv', 'full_split', 'qk_rope', 'kv_copy', 'attn', 'attn_gate',
        'full_o', 'post_attn', 'ffn_proj', 'ffn_act', 'down', 'post_mlp', 'lm_head'
    )

    $average = [ordered]@{}
    foreach ($stage in $stageNames) {
        $avg = ($Samples | Measure-Object -Property $stage -Average).Average
        $average[$stage] = [math]::Round([double]$avg, 6)
    }

    $hotspots = foreach ($stage in $stageNames | Where-Object { $_ -ne 'total' }) {
        [pscustomobject]@{
            stage = $stage
            avgMs = $average[$stage]
        }
    }

    $hotspots = $hotspots | Sort-Object -Property avgMs -Descending

    return [pscustomobject]@{
        sampleCount = $Samples.Count
        average     = [pscustomobject]$average
        hotspots    = $hotspots
        lastSample  = $Samples[-1]
    }
}

function Get-AilaResponseText {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$OutputLines
    )

    $responseLines = New-Object System.Collections.Generic.List[string]
    foreach ($line in $OutputLines) {
        if ($line -match '^\[Context\]') {
            break
        }
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $responseLines.Add($line.Trim())
    }

    return ([string]::Join("`n", $responseLines.ToArray())).Trim()
}
