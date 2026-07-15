param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$OneApiStack,
    [string]$CasesFile = 'perf\oneapi-regression-cases.json',
    [string[]]$CaseNames = @(),
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

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

function Get-AilaRegressionSelection {
    param(
        [Parameter(Mandatory = $true)]$CasesConfig,
        [string[]]$RequestedNames = @()
    )

    $schemaProperty = $CasesConfig.PSObject.Properties['schemaVersion']
    if ($null -eq $schemaProperty) {
        throw 'Unsupported accuracy cases schema: <missing>'
    }
    if ($schemaProperty.Value -ne 1) {
        throw "Unsupported accuracy cases schema: $($schemaProperty.Value)"
    }
    $casesProperty = $CasesConfig.PSObject.Properties['cases']
    if ($null -eq $casesProperty -or $null -eq $casesProperty.Value) {
        throw "Accuracy cases config is missing required property 'cases'."
    }
    $allCases = @($casesProperty.Value)
    if ($allCases.Count -eq 0) {
        throw 'Accuracy cases config contains no cases.'
    }

    $byName = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($case in $allCases) {
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
        if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
            throw "$context has an unsafe name. Use only letters, digits, dot, underscore, and hyphen."
        }
        $kind = (Get-AilaRegressionString -Object $case -Name 'kind' -Context $context).ToLowerInvariant()
        if ($kind -notin @('chat', 'asr', 'align', 'tts')) {
            throw "$context has unsupported kind '$kind'."
        }
        $modelValue = Get-AilaRegressionString -Object $case -Name 'model' -Context $context
        $modelPath = Resolve-AilaRegressionInputPath -RepoRoot $RepoRoot -ApprovedRoots $ApprovedRoots -Path $modelValue -Label "$context model" -PathType Container

        $resolved = [ordered]@{
            name      = $name
            kind      = $kind
            config    = $case
            modelPath = $modelPath
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
    param([Parameter(Mandatory = $true)][string]$ModelPath)

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
    return [ordered]@{
        path          = $ModelPath
        metadataFiles = $files.ToArray()
    }
}

function Get-AilaRegressionWavMetadata {
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    $reader = [System.IO.BinaryReader]::new($stream)
    try {
        if ($stream.Length -lt 44) { throw "WAV output is too small: $Path" }
        $riff = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
        $null = $reader.ReadUInt32()
        $wave = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
        if ($riff -ne 'RIFF' -or $wave -ne 'WAVE') { throw "Invalid WAV RIFF header: $Path" }

        $format = $null
        $dataBytes = 0L
        while ($stream.Position + 8 -le $stream.Length) {
            $chunkId = [System.Text.Encoding]::ASCII.GetString($reader.ReadBytes(4))
            $chunkSize = [long]$reader.ReadUInt32()
            $chunkStart = $stream.Position
            if ($chunkStart + $chunkSize -gt $stream.Length) { throw "Invalid WAV chunk size in: $Path" }
            if ($chunkId -eq 'fmt ' -and $chunkSize -ge 16) {
                $format = [ordered]@{
                    audioFormat   = [int]$reader.ReadUInt16()
                    channels      = [int]$reader.ReadUInt16()
                    sampleRate    = [long]$reader.ReadUInt32()
                    byteRate      = [long]$reader.ReadUInt32()
                    blockAlign    = [int]$reader.ReadUInt16()
                    bitsPerSample = [int]$reader.ReadUInt16()
                }
            }
            elseif ($chunkId -eq 'data') {
                $dataBytes = $chunkSize
            }
            $stream.Position = $chunkStart + $chunkSize + ($chunkSize % 2)
        }
        if ($null -eq $format -or $dataBytes -le 0 -or $format.sampleRate -le 0 -or $format.blockAlign -le 0) {
            throw "WAV output is missing valid format or audio data: $Path"
        }
        $duration = [double]$dataBytes / ([double]$format.sampleRate * [double]$format.blockAlign)
        if ([double]::IsNaN($duration) -or [double]::IsInfinity($duration) -or $duration -le 0) {
            throw "WAV output has invalid duration: $Path"
        }
        $format['dataBytes'] = $dataBytes
        $format['durationSeconds'] = [math]::Round($duration, 6)
        return [pscustomobject]$format
    }
    finally {
        $reader.Dispose()
        $stream.Dispose()
    }
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
$buildContext = Assert-AilaRegressionBuildStack -RepoRoot $repoRoot -BuildDirPath $buildDirPath -StackName $OneApiStack
$resolvedCases = @(Resolve-AilaRegressionCases -Cases $selectedCases -RepoRoot $repoRoot -ApprovedRoots $approvedInputRoots)

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

    $run = Invoke-AilaProcess -Executable '.\Aila.exe' -ArgumentList $args -WorkingDirectory $buildDirPath
    Write-AilaRegressionAtomicText -Path $stdoutPath -Text ([string]$run.stdoutText)
    Write-AilaRegressionAtomicText -Path $stderrPath -Text ([string]$run.stderrText)
    $logArtifacts = [ordered]@{
        stdout = Get-AilaRegressionFileArtifact -Path $stdoutPath
        stderr = Get-AilaRegressionFileArtifact -Path $stderrPath
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
        model         = Get-AilaRegressionModelIdentity -ModelPath $case.modelPath
        process       = [ordered]@{
            executable  = $buildContext.executable
            arguments   = $args
            commandLine = $run.commandLine
            exitCode    = $run.exitCode
            durationMs  = $run.durationMs
        }
        logs          = $logArtifacts
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
            $wavMetadata = Get-AilaRegressionWavMetadata -Path $wavPath
            $result['referenceAudio'] = Get-AilaRegressionFileArtifact -Path $case.referenceAudioPath
            $result['wav'] = $wavMetadata
            $result['output'] = Get-AilaRegressionFileArtifact -Path $wavPath
        }
    }

    $caseResultPath = Join-Path $resultsDir "$($case.name).json"
    Write-AilaRegressionAtomicJson -Path $caseResultPath -Data $result
    $result['caseResult'] = Get-AilaRegressionFileArtifact -Path $caseResultPath
    $results.Add([pscustomobject]$result)
    Write-Host (":: accuracy PASS {0} ({1}) ::" -f $case.name, $case.kind) -ForegroundColor Green
}

$gitInfo = Get-AilaGitInfo -RepoRoot $repoRoot
$payload = [ordered]@{
    schemaVersion     = 1
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
Write-Host (":: accuracy results written to {0} ::" -f $accuracyPath) -ForegroundColor Green
