param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$OneApiStack,
    [string]$CasesFile = 'perf\oneapi-regression-cases.json',
    [string[]]$CaseNames = @(),
    [string]$OutputDir = '',
    [ValidateRange(1, 86400)][int]$CaseTimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')
$script:AilaRegressionModelIdentityCache = @{}

function Get-AilaRegressionTokenTrace {
    param([Parameter(Mandatory = $true)][string]$StderrText)
    $matches = [regex]::Matches($StderrText, '(?m)^.*\[DebugToken\] step=(?<step>\d+) id=(?<id>-?\d+)')
    $steps = @($matches | ForEach-Object { [int]$_.Groups['step'].Value })
    $ids = @($matches | ForEach-Object { [int]$_.Groups['id'].Value })
    $generated = [regex]::Match($StderrText, '\[GenerateMessages\] Prompt=\d+ Generated=(?<count>\d+)')
    $generatedCount = if ($generated.Success) { [int]$generated.Groups['count'].Value } else { -1 }
    $complete = $generated.Success -and $steps.Count -eq $generatedCount -and $ids.Count -eq $generatedCount
    if ($complete) {
        for ($i = 0; $i -lt $generatedCount; $i++) {
            if ($steps[$i] -ne $i -or $ids[$i] -lt 0) { $complete = $false; break }
        }
    }
    return [pscustomobject]@{
        tokenIds = $ids
        steps = $steps
        generatedCount = $generatedCount
        complete = [bool]$complete
        source = 'AILA_DEBUG_TOKEN_IDS'
    }
}

function Get-AilaRegressionProperty {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "$Context is missing required property '$Name'."
    }
    return $property.Value
}

function Get-AilaRegressionString {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $value = Get-AilaRegressionProperty -Object $Object -Name $Name -Context $Context
    if ($value -isnot [string] -or [string]::IsNullOrWhiteSpace($value)) {
        throw "$Context property '$Name' must be a non-empty string."
    }
    return [string]$value
}

function Get-AilaRegressionPositiveInt {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $value = Get-AilaRegressionProperty -Object $Object -Name $Name -Context $Context
    $isInteger = $value -is [byte] -or $value -is [sbyte] -or $value -is [int16] -or
        $value -is [uint16] -or $value -is [int32] -or $value -is [uint32] -or
        $value -is [int64] -or $value -is [uint64]
    if (-not $isInteger -or [decimal]$value -le 0 -or [decimal]$value -gt [int]::MaxValue) {
        throw "$Context property '$Name' must be a positive integer."
    }
    return [int]$value
}

function Get-AilaRegressionMainRepoRoot {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)

    $commonDirText = (& git -C $RepoRoot rev-parse --path-format=absolute --git-common-dir 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commonDirText)) {
        return $RepoRoot
    }
    $commonDir = [System.IO.Path]::GetFullPath($commonDirText)
    if ((Split-Path $commonDir -Leaf) -ieq '.git') {
        return [System.IO.Path]::GetFullPath((Split-Path -Parent $commonDir))
    }
    return $RepoRoot
}

function Test-AilaRegressionPathInApprovedRoots {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Roots
    )

    foreach ($root in $Roots) {
        if (Test-AilaPathWithinRoot -Path $Path -Root $root) {
            return $true
        }
    }
    return $false
}

function Resolve-AilaRegressionInputPath {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string[]]$ApprovedRoots,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][ValidateSet('Leaf', 'Container')][string]$PathType
    )

    $resolved = Resolve-AilaPath -RepoRoot $RepoRoot -Path $Path
    if (-not (Test-AilaRegressionPathInApprovedRoots -Path $resolved -Roots $ApprovedRoots)) {
        throw "$Label resolves outside approved repository roots: $resolved"
    }
    if (-not (Test-Path -LiteralPath $resolved -PathType $PathType)) {
        $kind = if ($PathType -eq 'Leaf') { 'file' } else { 'directory' }
        throw "$Label $kind not found: $resolved"
    }
    return [System.IO.Path]::GetFullPath($resolved)
}

function Get-AilaApprovedRegressionOutputDir {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [AllowEmptyString()][string]$RequestedOutputDir,
        [Parameter(Mandatory = $true)][string]$StackName
    )

    $tmpRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot 'tmp'))
    $candidate = if ([string]::IsNullOrWhiteSpace($RequestedOutputDir)) {
        Join-Path $tmpRoot ("perf\oneapi-compare\accuracy\{0}" -f $StackName)
    }
    else {
        Resolve-AilaPath -RepoRoot $RepoRoot -Path $RequestedOutputDir
    }
    $candidate = [System.IO.Path]::GetFullPath($candidate)

    $tmpPrefix = $tmpRoot.TrimEnd('\') + '\'
    if (-not $candidate.StartsWith($tmpPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-AilaPathWithinRoot -Path $candidate -Root $tmpRoot)) {
        throw "Accuracy OutputDir must be inside repository tmp: $candidate"
    }
    return $candidate
}

function Assert-AilaRegressionBuildStack {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$BuildDirPath,
        [Parameter(Mandatory = $true)][string]$StackName
    )

    if (-not (Test-Path -LiteralPath $BuildDirPath -PathType Container)) {
        throw "Build directory not found: $BuildDirPath"
    }
    $buildInfoPath = Join-Path $BuildDirPath 'build_info.json'
    if (-not (Test-Path -LiteralPath $buildInfoPath -PathType Leaf)) {
        throw "Build info file not found: $buildInfoPath"
    }
    $exePath = Join-Path $BuildDirPath 'Aila.exe'
    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
        throw "Aila executable not found: $exePath"
    }

    $buildInfo = Read-AilaJsonFile -Path $buildInfoPath
    $schemaProperty = $buildInfo.PSObject.Properties['schemaVersion']
    if ($null -eq $schemaProperty) {
        throw "Unsupported build info schema in '$buildInfoPath': <missing>"
    }
    if ($schemaProperty.Value -ne 2) {
        throw "Unsupported build info schema in '$buildInfoPath': $($schemaProperty.Value)"
    }
    $buildOneApi = Get-AilaRegressionProperty -Object $buildInfo -Name 'oneApi' -Context "Build info '$buildInfoPath'"
    $recordedName = Get-AilaRegressionString -Object $buildOneApi -Name 'name' -Context 'Build info oneApi metadata'
    if (-not $recordedName.Equals($StackName, [System.StringComparison]::Ordinal)) {
        throw "Build '$BuildDirPath' belongs to '$recordedName', not '$StackName'."
    }

    $stackConfig = Get-AilaOneApiStackConfig -RepoRoot $RepoRoot
    $stack = Get-AilaOneApiStack -Config $stackConfig -Name $StackName
    Assert-AilaBuildInfoMatchesOneApiStack -BuildDir $BuildDirPath -Stack $stack
    Assert-AilaCMakeCacheMatchesOneApiStack -BuildDir $BuildDirPath -Stack $stack -RequireValues

    $stackEnvironment = Get-AilaOneApiStackEnvironment -Stack $stack
    Set-AilaProcessEnvironment -Environment $stackEnvironment
    $currentMetadata = Get-AilaOneApiStackMetadata -Stack $stack

    foreach ($propertyName in @('name', 'role', 'compilerVersion', 'dnnlVersion', 'tbbVersion')) {
        $recorded = [string](Get-AilaRegressionProperty -Object $buildOneApi -Name $propertyName -Context 'Build info oneApi metadata')
        $current = [string](Get-AilaRegressionProperty -Object $currentMetadata -Name $propertyName -Context 'Current oneAPI stack metadata')
        if (-not $recorded.Equals($current, [System.StringComparison]::Ordinal)) {
            throw "Build info oneApi metadata mismatch for '$propertyName': recorded '$recorded', current '$current'."
        }
    }
    foreach ($propertyName in @('compilerPath', 'dnnlRoot', 'tbbRoot', 'umfRoot')) {
        $recorded = Get-AilaWindowsPathKey -Path ([string](Get-AilaRegressionProperty -Object $buildOneApi -Name $propertyName -Context 'Build info oneApi metadata'))
        $current = Get-AilaWindowsPathKey -Path ([string](Get-AilaRegressionProperty -Object $currentMetadata -Name $propertyName -Context 'Current oneAPI stack metadata'))
        if (-not $recorded.Equals($current, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Build info oneApi metadata mismatch for '$propertyName'."
        }
    }
    $recordedSycl = [string](Get-AilaRegressionProperty -Object $buildOneApi -Name 'expectedSyclDll' -Context 'Build info oneApi metadata')
    $currentSycl = [string](Get-AilaRegressionProperty -Object $currentMetadata -Name 'expectedSyclDll' -Context 'Current oneAPI stack metadata')
    if (-not $recordedSycl.Equals($currentSycl, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Build info oneApi metadata mismatch for 'expectedSyclDll'."
    }
    $recordedLegacy = Get-AilaRegressionProperty -Object $buildOneApi -Name 'allowLegacyCompiler' -Context 'Build info oneApi metadata'
    $currentLegacy = Get-AilaRegressionProperty -Object $currentMetadata -Name 'allowLegacyCompiler' -Context 'Current oneAPI stack metadata'
    if ($recordedLegacy -isnot [bool] -or $currentLegacy -isnot [bool] -or $recordedLegacy -ne $currentLegacy) {
        throw "Build info oneApi metadata mismatch for 'allowLegacyCompiler'."
    }

    return [pscustomobject]@{
        buildInfo     = $buildInfo
        buildInfoPath = $buildInfoPath
        executable    = $exePath
        stack         = $stack
        currentOneApi = $currentMetadata
    }
}

function Test-AilaRegressionIntegerValue {
    param($Value)

    return $Value -is [byte] -or $Value -is [sbyte] -or $Value -is [int16] -or
        $Value -is [uint16] -or $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
}

function Assert-AilaRegressionAllowedFields {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string[]]$AllowedFields,
        [Parameter(Mandatory = $true)][string]$Context,
        [switch]$TopLevel
    )

    $allowed = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($field in $AllowedFields) { $null = $allowed.Add($field) }
    foreach ($property in $Object.PSObject.Properties) {
        if (-not $allowed.Contains($property.Name)) {
            if ($TopLevel) { throw "Accuracy cases config has unknown top-level field '$($property.Name)'." }
            throw "$Context has unknown field '$($property.Name)'."
        }
    }
}

function Assert-AilaRegressionCaseSchema {
    param([Parameter(Mandatory = $true)]$Case)

    if ($Case -isnot [pscustomobject]) { throw 'Each accuracy case must be a JSON object.' }
    $name = Get-AilaRegressionString -Object $Case -Name 'name' -Context 'Accuracy case'
    $context = "Accuracy case '$name'"
    $kind = (Get-AilaRegressionString -Object $Case -Name 'kind' -Context $context).ToLowerInvariant()
    if ($kind -notin @('chat', 'asr', 'align', 'tts')) { throw "$context has unsupported kind '$kind'." }

    $specific = switch ($kind) {
        'chat' { @('input', 'maxTokens', 'mode', 'expectRegex') }
        'asr' { @('audio', 'forcedLanguage', 'expectedText') }
        'align' { @('audio', 'text', 'language', 'timestampFrameMs', 'timestampToleranceMs') }
        'tts' { @('text', 'referenceAudio', 'language', 'maxTokens') }
    }
    Assert-AilaRegressionAllowedFields -Object $Case -AllowedFields (@('name', 'kind', 'model') + $specific) -Context $context
    $null = Get-AilaRegressionString -Object $Case -Name 'model' -Context $context

    switch ($kind) {
        'chat' {
            $null = Get-AilaRegressionString -Object $Case -Name 'input' -Context $context
            $null = Get-AilaRegressionPositiveInt -Object $Case -Name 'maxTokens' -Context $context
            $mode = Get-AilaRegressionString -Object $Case -Name 'mode' -Context $context
            if ($mode -ne 'greedy') { throw "$context property 'mode' must be the string 'greedy'." }
            if ($null -ne $Case.PSObject.Properties['expectRegex']) {
                $pattern = Get-AilaRegressionString -Object $Case -Name 'expectRegex' -Context $context
                try { $null = [regex]::new($pattern) } catch { throw "$context property 'expectRegex' is invalid: $($_.Exception.Message)" }
            }
        }
        'asr' {
            foreach ($field in @('audio', 'forcedLanguage', 'expectedText')) { $null = Get-AilaRegressionString -Object $Case -Name $field -Context $context }
        }
        'align' {
            foreach ($field in @('audio', 'text', 'language')) { $null = Get-AilaRegressionString -Object $Case -Name $field -Context $context }
            $null = Get-AilaRegressionPositiveInt -Object $Case -Name 'timestampFrameMs' -Context $context
            $null = Get-AilaRegressionPositiveInt -Object $Case -Name 'timestampToleranceMs' -Context $context
        }
        'tts' {
            foreach ($field in @('text', 'referenceAudio', 'language')) { $null = Get-AilaRegressionString -Object $Case -Name $field -Context $context }
            $null = Get-AilaRegressionPositiveInt -Object $Case -Name 'maxTokens' -Context $context
        }
    }
}

function Get-AilaRegressionSelection {
    param(
        [Parameter(Mandatory = $true)]$CasesConfig,
        [string[]]$RequestedNames = @()
    )

    if ($CasesConfig -isnot [pscustomobject]) { throw 'Accuracy cases config must be a JSON object.' }
    Assert-AilaRegressionAllowedFields -Object $CasesConfig -AllowedFields @('schemaVersion', 'cases') -Context 'Accuracy cases config' -TopLevel
    $schemaProperty = $CasesConfig.PSObject.Properties['schemaVersion']
    if ($null -eq $schemaProperty) { throw 'Unsupported accuracy cases schema: <missing>' }
    if (-not (Test-AilaRegressionIntegerValue -Value $schemaProperty.Value) -or $schemaProperty.Value -ne 1) {
        throw "Unsupported accuracy cases schema: $($schemaProperty.Value)"
    }
    $casesProperty = $CasesConfig.PSObject.Properties['cases']
    if ($null -eq $casesProperty -or $null -eq $casesProperty.Value) {
        throw "Accuracy cases config is missing required property 'cases'."
    }
    if ($casesProperty.Value -isnot [System.Array]) { throw 'Accuracy cases must be a JSON array.' }
    $allCases = @($casesProperty.Value)
    if ($allCases.Count -eq 0) {
        throw 'Accuracy cases config contains no cases.'
    }

    $byName = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($case in $allCases) {
        Assert-AilaRegressionCaseSchema -Case $case
        $name = Get-AilaRegressionString -Object $case -Name 'name' -Context 'Accuracy case'
        if (-not $byName.TryAdd($name, $case)) {
            throw "Duplicate accuracy case name in cases file: '$name'."
        }
    }

    $normalized = New-Object System.Collections.Generic.List[string]
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $RequestedNames) {
        foreach ($part in ([string]$entry -split ',')) {
            $name = $part.Trim()
            if ([string]::IsNullOrWhiteSpace($name)) {
                continue
            }
            if (-not $seen.Add($name)) {
                throw "Duplicate accuracy case selection: '$name'."
            }
            $normalized.Add($name)
        }
    }
    if ($normalized.Count -eq 0) {
        return $allCases
    }

    $selected = New-Object System.Collections.Generic.List[object]
    foreach ($name in $normalized) {
        if (-not $byName.ContainsKey($name)) {
            throw "Unknown accuracy case selection: '$name'."
        }
        $selected.Add($byName[$name])
    }
    return $selected.ToArray()
}

function Resolve-AilaRegressionCases {
    param(
        [Parameter(Mandatory = $true)][object[]]$Cases,
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string[]]$ApprovedRoots
    )

    $resolvedCases = New-Object System.Collections.Generic.List[object]
    foreach ($case in $Cases) {
        $name = Get-AilaRegressionString -Object $case -Name 'name' -Context 'Accuracy case'
        $context = "Accuracy case '$name'"
        if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or $name.Contains('..')) {
            throw "$context has an unsafe name. Use only letters, digits, dot, underscore, and hyphen."
        }
        $kind = (Get-AilaRegressionString -Object $case -Name 'kind' -Context $context).ToLowerInvariant()
        if ($kind -notin @('chat', 'asr', 'align', 'tts')) {
            throw "$context has unsupported kind '$kind'."
        }
        $modelValue = Get-AilaRegressionString -Object $case -Name 'model' -Context $context
        $modelPath = Resolve-AilaRegressionInputPath -RepoRoot $RepoRoot -ApprovedRoots $ApprovedRoots -Path $modelValue -Label "$context model" -PathType Container

        $resolved = [ordered]@{
            name          = $name
            kind          = $kind
            config        = $case
            modelPath     = $modelPath
            modelIdentity = Get-AilaRegressionModelIdentity -ModelPath $modelPath -ApprovedRoots $ApprovedRoots
        }
        switch ($kind) {
            'chat' {
                $inputValue = Get-AilaRegressionString -Object $case -Name 'input' -Context $context
                $resolved.inputPath = Resolve-AilaRegressionInputPath -RepoRoot $RepoRoot -ApprovedRoots $ApprovedRoots -Path $inputValue -Label "$context input" -PathType Leaf
                $resolved.maxTokens = Get-AilaRegressionPositiveInt -Object $case -Name 'maxTokens' -Context $context
                $modeProperty = $case.PSObject.Properties['mode']
                if ($null -ne $modeProperty -and ($modeProperty.Value -isnot [string] -or [string]$modeProperty.Value -ne 'greedy')) {
                    throw "$context property 'mode' must be the string 'greedy'."
                }
                $regexProperty = $case.PSObject.Properties['expectRegex']
                if ($null -ne $regexProperty) {
                    if ($regexProperty.Value -isnot [string]) {
                        throw "$context property 'expectRegex' must be a string."
                    }
                    try { $null = [regex]::new([string]$regexProperty.Value) }
                    catch { throw "$context property 'expectRegex' is invalid: $($_.Exception.Message)" }
                }
            }
            'asr' {
                $audioValue = Get-AilaRegressionString -Object $case -Name 'audio' -Context $context
                $resolved.audioPath = Resolve-AilaRegressionInputPath -RepoRoot $RepoRoot -ApprovedRoots $ApprovedRoots -Path $audioValue -Label "$context audio" -PathType Leaf
                $resolved.forcedLanguage = Get-AilaRegressionString -Object $case -Name 'forcedLanguage' -Context $context
                $expectedProperty = $case.PSObject.Properties['expectedText']
                if ($null -ne $expectedProperty -and $expectedProperty.Value -isnot [string]) {
                    throw "$context property 'expectedText' must be a string."
                }
            }
            'align' {
                $audioValue = Get-AilaRegressionString -Object $case -Name 'audio' -Context $context
                $resolved.audioPath = Resolve-AilaRegressionInputPath -RepoRoot $RepoRoot -ApprovedRoots $ApprovedRoots -Path $audioValue -Label "$context audio" -PathType Leaf
                $resolved.text = Get-AilaRegressionString -Object $case -Name 'text' -Context $context
                $resolved.language = Get-AilaRegressionString -Object $case -Name 'language' -Context $context
                $resolved.timestampFrameMs = Get-AilaRegressionPositiveInt -Object $case -Name 'timestampFrameMs' -Context $context
                $resolved.timestampToleranceMs = Get-AilaRegressionPositiveInt -Object $case -Name 'timestampToleranceMs' -Context $context
            }
            'tts' {
                $referenceValue = Get-AilaRegressionString -Object $case -Name 'referenceAudio' -Context $context
                $resolved.referenceAudioPath = Resolve-AilaRegressionInputPath -RepoRoot $RepoRoot -ApprovedRoots $ApprovedRoots -Path $referenceValue -Label "$context referenceAudio" -PathType Leaf
                $resolved.text = Get-AilaRegressionString -Object $case -Name 'text' -Context $context
                $resolved.language = Get-AilaRegressionString -Object $case -Name 'language' -Context $context
                $resolved.maxTokens = Get-AilaRegressionPositiveInt -Object $case -Name 'maxTokens' -Context $context
            }
        }
        $resolvedCases.Add([pscustomobject]$resolved)
    }
    return $resolvedCases.ToArray()
}

function Write-AilaRegressionAtomicText {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowEmptyString()][string]$Text
    )

    $parent = Split-Path -Parent $Path
    Ensure-AilaDirectory -Path $parent
    $tempPath = Join-Path $parent ('.' + (Split-Path $Path -Leaf) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        [System.IO.File]::WriteAllText($tempPath, $Text, [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $tempPath -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force
        }
    }
}

function Write-AilaRegressionAtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Data
    )

    Write-AilaRegressionAtomicText -Path $Path -Text ($Data | ConvertTo-Json -Depth 24)
}

function Get-AilaRegressionFileArtifact {
    param([Parameter(Mandatory = $true)][string]$Path)

    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    return [ordered]@{
        path   = $item.FullName
        bytes  = [long]$item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash
    }
}

function Get-AilaRegressionModelIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)][string[]]$ApprovedRoots
    )

    $modelKey = Get-AilaCanonicalWindowsPathKey -Path $ModelPath
    if ($script:AilaRegressionModelIdentityCache.ContainsKey($modelKey)) {
        return $script:AilaRegressionModelIdentityCache[$modelKey]
    }

    $metadataNames = @(
        'config.json', 'generation_config.json', 'tokenizer_config.json', 'processor_config.json',
        'preprocessor_config.json', 'special_tokens_map.json', 'added_tokens.json', 'tokenizer.json',
        'model.safetensors.index.json', 'pytorch_model.bin.index.json', 'vocab.json', 'merges.txt'
    )
    $files = New-Object System.Collections.Generic.List[object]
    foreach ($name in $metadataNames) {
        $path = Join-Path $ModelPath $name
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $artifact = Get-AilaRegressionFileArtifact -Path $path
            $artifact['name'] = $name
            $files.Add([pscustomobject]$artifact)
        }
    }
    foreach ($path in @(Get-ChildItem -LiteralPath $ModelPath -File -Filter '*.index.json' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)) {
        $name = Split-Path $path -Leaf
        if ($metadataNames -contains $name) { continue }
        $artifact = Get-AilaRegressionFileArtifact -Path $path
        $artifact['name'] = $name
        $files.Add([pscustomobject]$artifact)
    }
    $weightExtensions = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($extension in @('.safetensors', '.bin', '.gguf', '.pt', '.pth')) { $null = $weightExtensions.Add($extension) }
    $weightManifest = New-Object System.Collections.Generic.List[object]
    foreach ($directory in @(Get-ChildItem -LiteralPath $ModelPath -Directory -Recurse -Force -ErrorAction Stop)) {
        if (($directory.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -and
            (-not (Test-AilaPathWithinRoot -Path $directory.FullName -Root $ModelPath) -or
             -not (Test-AilaRegressionPathInApprovedRoots -Path $directory.FullName -Roots $ApprovedRoots))) {
            throw "Model weight directory escapes approved model root: $($directory.FullName)"
        }
    }
    $weightFiles = @(Get-ChildItem -LiteralPath $ModelPath -File -Recurse -Force -ErrorAction Stop | Where-Object {
        $weightExtensions.Contains($_.Extension) -and $_.FullName -notmatch '[\\/](cache|outputs?|output)[\\/]'
    } | Sort-Object -Property FullName)
    if ($weightFiles.Count -eq 0) {
        throw "Model '$ModelPath' contains no supported weight files."
    }
    Write-Host (":: hashing {0} model weight file(s): {1} ::" -f $weightFiles.Count, $ModelPath)
    foreach ($weightFile in $weightFiles) {
        if (-not (Test-AilaPathWithinRoot -Path $weightFile.FullName -Root $ModelPath) -or
            -not (Test-AilaRegressionPathInApprovedRoots -Path $weightFile.FullName -Roots $ApprovedRoots)) {
            throw "Model weight escapes approved model root: $($weightFile.FullName)"
        }
        $relativePath = [System.IO.Path]::GetRelativePath($ModelPath, $weightFile.FullName).Replace('\', '/')
        $weightManifest.Add([pscustomobject]@{
            relativePath = $relativePath
            size         = [long]$weightFile.Length
            sha256       = (Get-FileHash -LiteralPath $weightFile.FullName -Algorithm SHA256).Hash
        })
    }

    $identity = [ordered]@{
        path           = $ModelPath
        metadataFiles  = $files.ToArray()
        weightManifest = @($weightManifest.ToArray() | Sort-Object -Property relativePath)
    }
    $script:AilaRegressionModelIdentityCache[$modelKey] = $identity
    return $identity
}

function Remove-AilaRegressionOwnedCaseFiles {
    param(
        [Parameter(Mandatory = $true)]$Case,
        [Parameter(Mandatory = $true)][string]$LogsDir,
        [Parameter(Mandatory = $true)][string]$OutputsDir,
        [Parameter(Mandatory = $true)][string]$ResultsDir
    )

    $ownedPaths = @(
        (Join-Path $LogsDir "$($Case.name).stdout.txt"),
        (Join-Path $LogsDir "$($Case.name).stderr.txt"),
        (Join-Path $OutputsDir "$($Case.name).txt"),
        (Join-Path $OutputsDir "$($Case.name).wav"),
        (Join-Path $OutputsDir "$($Case.name).alignment.json"),
        (Join-Path $ResultsDir "$($Case.name).json")
    )
    foreach ($path in $ownedPaths) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Remove-Item -LiteralPath $path -Force
        }
    }
}

$repoRoot = Get-AilaRepoRoot
$mainRepoRoot = Get-AilaRegressionMainRepoRoot -RepoRoot $repoRoot
$approvedInputRoots = @($repoRoot, $mainRepoRoot | Select-Object -Unique)
$outputDirPath = Get-AilaApprovedRegressionOutputDir -RepoRoot $repoRoot -RequestedOutputDir $OutputDir -StackName $OneApiStack
$buildDirPath = Resolve-AilaRegressionInputPath -RepoRoot $repoRoot -ApprovedRoots $approvedInputRoots -Path $BuildDir -Label 'Build directory' -PathType Container
$casesPath = Resolve-AilaRegressionInputPath -RepoRoot $repoRoot -ApprovedRoots $approvedInputRoots -Path $CasesFile -Label 'Accuracy cases' -PathType Leaf
$casesConfig = Read-AilaJsonFile -Path $casesPath
$selectedCases = @(Get-AilaRegressionSelection -CasesConfig $casesConfig -RequestedNames $CaseNames)
$resolvedCases = @(Resolve-AilaRegressionCases -Cases $selectedCases -RepoRoot $repoRoot -ApprovedRoots $approvedInputRoots)
$buildContext = Assert-AilaRegressionBuildStack -RepoRoot $repoRoot -BuildDirPath $buildDirPath -StackName $OneApiStack

$logsDir = Join-Path $outputDirPath 'case_logs'
$outputsDir = Join-Path $outputDirPath 'outputs'
$resultsDir = Join-Path $outputDirPath 'case_results'
Ensure-AilaDirectory -Path $outputDirPath
Ensure-AilaDirectory -Path $logsDir
Ensure-AilaDirectory -Path $outputsDir
Ensure-AilaDirectory -Path $resultsDir
foreach ($case in $resolvedCases) {
    Remove-AilaRegressionOwnedCaseFiles -Case $case -LogsDir $logsDir -OutputsDir $outputsDir -ResultsDir $resultsDir
}
$accuracyPath = Join-Path $outputDirPath 'accuracy.json'
if (Test-Path -LiteralPath $accuracyPath -PathType Leaf) {
    Remove-Item -LiteralPath $accuracyPath -Force
}

$results = New-Object System.Collections.Generic.List[object]
foreach ($case in $resolvedCases) {
    $stdoutPath = Join-Path $logsDir "$($case.name).stdout.txt"
    $stderrPath = Join-Path $logsDir "$($case.name).stderr.txt"
    $args = @()
    switch ($case.kind) {
        'chat' {
            $args = @('-m', $case.modelPath, '--messages-json', $case.inputPath, '--max-tokens', [string]$case.maxTokens, '--greedy', '--no-stream')
        }
        'asr' {
            $args = @('-m', $case.modelPath, '--transcribe', $case.audioPath, '--forced-lang', [string]$case.forcedLanguage, '--no-stream')
        }
        'align' {
            $args = @('-m', $case.modelPath, '--align-audio', $case.audioPath, '--align-text', [string]$case.text, '--align-lang', [string]$case.language)
        }
        'tts' {
            $wavPath = Join-Path $outputsDir ("{0}.wav" -f $case.name)
            $args = @('-m', $case.modelPath, '--synthesize', [string]$case.text, '--ref', $case.referenceAudioPath, '--language', [string]$case.language, '--max-tokens', [string]$case.maxTokens, '--output-wav', $wavPath)
        }
        default { throw "Unsupported accuracy case kind '$($case.kind)'." }
    }

    $caseResultPath = Join-Path $resultsDir "$($case.name).json"
    $envOverrides = @{}
    if ($case.kind -eq 'chat') {
        # Engine.hpp emits prompt/generated token IDs under this diagnostic
        # flag.  It is bounded to the first 64 generated steps by the engine;
        # the result records availability so comparison never claims strict
        # token equality when the trace is incomplete.
        $envOverrides['AILA_DEBUG_TOKEN_IDS'] = '1'
    }
    $run = Invoke-AilaProcess -Executable '.\Aila.exe' -ArgumentList $args -WorkingDirectory $buildDirPath -EnvOverrides $envOverrides -TimeoutSeconds $CaseTimeoutSeconds
    Write-AilaRegressionAtomicText -Path $stdoutPath -Text ([string]$run.stdoutText)
    Write-AilaRegressionAtomicText -Path $stderrPath -Text ([string]$run.stderrText)
    $logArtifacts = [ordered]@{
        stdout = Get-AilaRegressionFileArtifact -Path $stdoutPath
        stderr = Get-AilaRegressionFileArtifact -Path $stderrPath
    }
    $processResult = [ordered]@{
        executable     = $buildContext.executable
        arguments      = $args
        commandLine    = $run.commandLine
        exitCode       = $run.exitCode
        durationMs     = $run.durationMs
        timedOut       = $run.timedOut
        timeoutSeconds = $run.timeoutSeconds
    }
    if ($run.timedOut) {
        $diagnostic = [ordered]@{
            schemaVersion        = 1
            name                 = $case.name
            kind                 = $case.kind
            config               = $case.config
            model                = $case.modelIdentity
            executionPassed      = $false
            expectationPassed    = $null
            passed               = $false
            status               = 'timed-out'
            manualReviewRequired = $true
            process              = $processResult
            logs                 = $logArtifacts
            diagnostic           = "Case exceeded timeout of $CaseTimeoutSeconds second(s); the process tree was terminated."
        }
        Write-AilaRegressionAtomicJson -Path $caseResultPath -Data $diagnostic
        Write-Host (":: accuracy TIMED OUT {0} ({1}) ::" -f $case.name, $case.kind) -ForegroundColor Red
        throw "Accuracy case '$($case.name)' timed out after $CaseTimeoutSeconds second(s). See '$caseResultPath'."
    }
    if ($run.exitCode -ne 0) {
        throw "Accuracy case '$($case.name)' failed with exit code $($run.exitCode). See '$stdoutPath' and '$stderrPath'."
    }

    $combinedOutput = (([string]$run.stdoutText, [string]$run.stderrText) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"
    $result = [ordered]@{
        schemaVersion = 1
        name          = $case.name
        kind          = $case.kind
        config        = $case.config
        model         = $case.modelIdentity
        executionPassed = $true
        expectationPassed = $null
        passed         = $true
        status         = 'passed'
        manualReviewRequired = $false
        process        = $processResult
        logs          = $logArtifacts
        expectation   = $null
    }

    switch ($case.kind) {
        'chat' {
            $responseText = ([string]$run.stdoutText).Trim()
            if ([string]::IsNullOrWhiteSpace($responseText)) {
                throw "Accuracy chat case '$($case.name)' produced no response text on stdout."
            }
            $textPath = Join-Path $outputsDir "$($case.name).txt"
            Write-AilaRegressionAtomicText -Path $textPath -Text $responseText
            $outputArtifact = Get-AilaRegressionFileArtifact -Path $textPath
            $outputArtifact['rawText'] = $responseText
            $result['input'] = Get-AilaRegressionFileArtifact -Path $case.inputPath
            $result['rawText'] = $responseText
            $result['normalizedText'] = Normalize-AilaText -Text $responseText
            $result['output'] = $outputArtifact
            $tokenTrace = Get-AilaRegressionTokenTrace -StderrText ([string]$run.stderrText)
            $result['generatedTokenIds'] = @($tokenTrace.tokenIds)
            $result['tokenSequenceAvailable'] = [bool]$tokenTrace.complete
            $result['tokenSequenceTrace'] = [ordered]@{ source = $tokenTrace.source; tracedCount = $tokenTrace.tokenIds.Count; generatedCount = if ($tokenTrace.generatedCount -ge 0) { $tokenTrace.generatedCount } else { $null }; steps = @($tokenTrace.steps); complete = [bool]$tokenTrace.complete }
            $expectRegexProperty = $case.config.PSObject.Properties['expectRegex']
            $result['expectation'] = if ($null -eq $expectRegexProperty) { $null } else {
                [ordered]@{ type = 'regex'; pattern = [string]$expectRegexProperty.Value; passed = ($responseText -match [string]$expectRegexProperty.Value) }
            }
        }
        'asr' {
            try {
                $parsed = Parse-AilaAsrOutput -OutputText ([string]$run.stdoutText)
            }
            catch {
                $parsed = Parse-AilaAsrOutput -OutputText $combinedOutput
            }
            if ([string]::IsNullOrWhiteSpace([string]$parsed.text)) {
                throw "Accuracy ASR case '$($case.name)' produced no transcript text."
            }
            $textPath = Join-Path $outputsDir "$($case.name).txt"
            Write-AilaRegressionAtomicText -Path $textPath -Text ([string]$parsed.text)
            $outputArtifact = Get-AilaRegressionFileArtifact -Path $textPath
            $outputArtifact['rawText'] = [string]$parsed.text
            $result['audio'] = Get-AilaRegressionFileArtifact -Path $case.audioPath
            $result['rawText'] = [string]$parsed.text
            $result['normalizedText'] = Normalize-AilaText -Text ([string]$parsed.text)
            $result['asr'] = $parsed
            $result['output'] = $outputArtifact
            $expectedProperty = $case.config.PSObject.Properties['expectedText']
            $result['expectation'] = if ($null -eq $expectedProperty) { $null } else {
                $expectedNormalized = Normalize-AilaText -Text ([string]$expectedProperty.Value)
                [ordered]@{ type = 'normalized_text'; expected = $expectedNormalized; passed = ($result.normalizedText -eq $expectedNormalized) }
            }
        }
        'align' {
            $alignment = @(Parse-AilaAlignmentOutput -OutputText $combinedOutput)
            if ($alignment.Count -eq 0) {
                throw "Accuracy alignment case '$($case.name)' produced no alignment rows."
            }
            $alignmentPath = Join-Path $outputsDir "$($case.name).alignment.json"
            Write-AilaRegressionAtomicJson -Path $alignmentPath -Data $alignment
            $result['audio'] = Get-AilaRegressionFileArtifact -Path $case.audioPath
            $result['alignment'] = $alignment
            $result['output'] = Get-AilaRegressionFileArtifact -Path $alignmentPath
        }
        'tts' {
            if (-not (Test-Path -LiteralPath $wavPath -PathType Leaf) -or (Get-Item -LiteralPath $wavPath).Length -le 0) {
                throw "Accuracy TTS case '$($case.name)' did not produce a non-empty WAV file: $wavPath"
            }
            $wavMetadata = Get-AilaWavMetadata -Path $wavPath
            $result['referenceAudio'] = Get-AilaRegressionFileArtifact -Path $case.referenceAudioPath
            $result['wav'] = $wavMetadata
            $result['output'] = Get-AilaRegressionFileArtifact -Path $wavPath
        }
    }

    $expectationPassed = if ($null -eq $result.expectation) { $null } else { [bool]$result.expectation.passed }
    $result['expectationPassed'] = $expectationPassed
    $result['passed'] = ($expectationPassed -ne $false)
    $result['status'] = if ($expectationPassed -eq $false) { 'expectation-failed' } else { 'passed' }
    $result['manualReviewRequired'] = ($expectationPassed -eq $false)
    Write-AilaRegressionAtomicJson -Path $caseResultPath -Data $result
    $result['caseResult'] = Get-AilaRegressionFileArtifact -Path $caseResultPath
    $results.Add([pscustomobject]$result)
    if ($expectationPassed -eq $false) {
        Write-Host (":: accuracy EXPECTATION FAILED {0} ({1}) ::" -f $case.name, $case.kind) -ForegroundColor Yellow
    }
    else {
        Write-Host (":: accuracy PASSED {0} ({1}) ::" -f $case.name, $case.kind) -ForegroundColor Green
    }
}

$gitInfo = Get-AilaGitInfo -RepoRoot $repoRoot
$expectationValues = @($results | ForEach-Object { $_.expectationPassed } | Where-Object { $null -ne $_ })
$combinedExpectationPassed = if (@($expectationValues | Where-Object { $_ -eq $false }).Count -gt 0) {
    $false
}
elseif ($expectationValues.Count -gt 0) {
    $true
}
else {
    $null
}
$combinedPassed = ($combinedExpectationPassed -ne $false)
$payload = [ordered]@{
    schemaVersion     = 1
    executionPassed   = $true
    expectationPassed = $combinedExpectationPassed
    passed            = $combinedPassed
    status            = if ($combinedPassed) { 'passed' } else { 'completed-with-expectation-failures' }
    manualReviewRequired = (-not $combinedPassed)
    generatedAtUtc    = (Get-Date).ToUniversalTime().ToString('o')
    git               = [ordered]@{
        shortCommit = $gitInfo.shortCommit
        fullCommit  = $gitInfo.fullCommit
        branch      = $gitInfo.branch
    }
    build             = $buildContext.buildInfo.build
    buildInfo         = Get-AilaRegressionFileArtifact -Path $buildContext.buildInfoPath
    executable        = Get-AilaRegressionFileArtifact -Path $buildContext.executable
    oneApi            = $buildContext.buildInfo.oneApi
    currentOneApi     = $buildContext.currentOneApi
    casesFile         = Get-AilaRegressionFileArtifact -Path $casesPath
    selectedCaseNames = @($resolvedCases | ForEach-Object { $_.name })
    cases             = $results.ToArray()
}
Write-AilaRegressionAtomicJson -Path $accuracyPath -Data $payload
if ($combinedPassed) {
    Write-Host (":: accuracy PASSED; results written to {0} ::" -f $accuracyPath) -ForegroundColor Green
}
else {
    Write-Host (":: accuracy COMPLETE WITH EXPECTATION FAILURES; results written to {0} ::" -f $accuracyPath) -ForegroundColor Yellow
}
