[CmdletBinding()]
param(
    [string]$BaselineDir = '',
    [string]$CandidateDir = '',
    [string]$OutputPath = '',
    [switch]$SelfTest,
    [switch]$SkipSpeakerEmbedding
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'perf\PerfCommon.ps1')

$script:RepoRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$script:TmpRoot = [System.IO.Path]::GetFullPath((Join-Path $script:RepoRoot 'tmp'))
$commonDir=(& git -C $script:RepoRoot rev-parse --path-format=absolute --git-common-dir 2>$null|Out-String).Trim()
$script:MainRepoRoot=if(-not [string]::IsNullOrWhiteSpace($commonDir) -and (Split-Path $commonDir -Leaf) -ieq '.git'){[IO.Path]::GetFullPath((Split-Path -Parent $commonDir))}else{$script:RepoRoot}
$script:ApprovedRepoRoots=@($script:RepoRoot,$script:MainRepoRoot|Select-Object -Unique)
$script:HashCache = @{}
$script:AudioThresholds = [ordered]@{
    durationDeltaFramesMaximum = 2000
    correlationManualBelow = 0.999
    relativeRmsDeltaManualAbove = 0.02
    logMelMaeManualAbove = 0.05
    speakerEmbeddingCosineManualBelow = 0.995
    clippingRatioDeltaManualAbove = 0.01
    silenceRatioDeltaManualAbove = 0.01
}

function Test-PathWithinRoot {
    param([string]$Path, [string]$Root)
    return Test-AilaPathWithinRoot -Path $Path -Root $Root
}

function Resolve-ExistingPath {
    param([string]$Path, [string]$Label, [ValidateSet('Leaf','Container')][string]$PathType = 'Leaf')
    if ([string]::IsNullOrWhiteSpace($Path)) { throw "$Label path is empty." }
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) { $Path = Join-Path $script:RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $Path -PathType $PathType)) { throw "$Label not found: $Path" }
    $resolved = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).Path)
    $insideApproved=$false;foreach($root in $script:ApprovedRepoRoots){if(Test-PathWithinRoot -Path $resolved -Root $root){$insideApproved=$true;break}}
    if (-not $insideApproved) {
        throw "$Label resolves outside repository: $resolved"
    }
    return $resolved
}

function Get-FileSha256 {
    param([string]$Path)
    $key = [System.IO.Path]::GetFullPath($Path).ToLowerInvariant()
    if (-not $script:HashCache.ContainsKey($key)) {
        $script:HashCache[$key] = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
    }
    return $script:HashCache[$key]
}

function Get-PropertyValue {
    param($Object, [string]$Name, [string]$Context)
    if ($null -eq $Object) { throw "$Context is null." }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) { throw "$Context is missing '$Name'." }
    return $property.Value
}

function Get-OptionalPropertyValue {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Test-FiniteNumber {
    param([AllowNull()]$Value)
    if ($null -eq $Value) { return $false }
    $numeric = $Value -is [byte] -or $Value -is [sbyte] -or $Value -is [short] -or $Value -is [ushort] -or
        $Value -is [int] -or $Value -is [uint] -or $Value -is [long] -or $Value -is [ulong] -or
        $Value -is [float] -or $Value -is [double] -or $Value -is [decimal]
    if (-not $numeric) { return $false }
    try { $number = [double]$Value } catch { return $false }
    return -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)
}

function Test-IntegerNumber {
    param([AllowNull()]$Value)
    if (-not (Test-FiniteNumber $Value)) { return $false }
    if($Value -is [decimal]){return $Value -eq [decimal]::Truncate($Value)}
    if($Value -is [float]){return [double]$Value -eq [Math]::Truncate([double]$Value)}
    if($Value -is [double]){return $Value -eq [Math]::Truncate($Value)}
    return $true
}

function Test-Int64Integer {
    param([AllowNull()]$Value)
    if (-not (Test-IntegerNumber $Value)) { return $false }
    try { $null = [int64]$Value; return $true } catch { return $false }
}

function Test-TtsNumericProperty {
    param($Result,$Object,[string]$Name,[string]$Context,[AllowNull()]$Minimum=$null,[AllowNull()]$Maximum=$null,[switch]$Integer,[switch]$Required)
    $value=Get-OptionalPropertyValue $Object $Name
    if($null -eq $value){
        if($Required){Set-CaseFailure $Result "$Context '$Name' is missing or null";return $false}
        return $true
    }
    if(-not (Test-FiniteNumber $value)){Set-CaseFailure $Result "$Context '$Name' must be a finite number";return $false}
    if($Integer -and -not (Test-IntegerNumber $value)){Set-CaseFailure $Result "$Context '$Name' must be an integer";return $false}
    if($Integer){
        try{$number=[decimal]$value}catch{Set-CaseFailure $Result "$Context '$Name' is outside the supported integer range";return $false}
        if($null -ne $Minimum -and $number -lt [decimal]$Minimum){Set-CaseFailure $Result "$Context '$Name' must be at least $Minimum";return $false}
        if($null -ne $Maximum -and $number -gt [decimal]$Maximum){Set-CaseFailure $Result "$Context '$Name' must be at most $Maximum";return $false}
    }else{
        $number=[double]$value
        if($null -ne $Minimum -and $number -lt [double]$Minimum){Set-CaseFailure $Result "$Context '$Name' must be at least $Minimum";return $false}
        if($null -ne $Maximum -and $number -gt [double]$Maximum){Set-CaseFailure $Result "$Context '$Name' must be at most $Maximum";return $false}
    }
    return $true
}

function Assert-IntegerOne {
    param($Value, [string]$Context)
    if ($Value -isnot [int] -and $Value -isnot [long]) { throw "$Context must be integer 1." }
    if ([long]$Value -ne 1) { throw "$Context must be 1." }
}

function Assert-Artifact {
    param($Artifact, [string]$Label, [string]$RequiredRoot = '')
    $path = Resolve-ExistingPath -Path ([string](Get-PropertyValue $Artifact 'path' $Label)) -Label $Label
    if (-not [string]::IsNullOrWhiteSpace($RequiredRoot) -and -not (Test-PathWithinRoot $path $RequiredRoot)) {
        throw "$Label must remain inside '$RequiredRoot': $path"
    }
    $bytes = Get-PropertyValue $Artifact 'bytes' $Label
    if ([long]$bytes -ne (Get-Item -LiteralPath $path).Length) { throw "$Label byte count is stale: $path" }
    $sha = [string](Get-PropertyValue $Artifact 'sha256' $Label)
    if (-not $sha.Equals((Get-FileSha256 $path), [StringComparison]::OrdinalIgnoreCase)) { throw "$Label hash is stale: $path" }
    return $path
}

function ConvertTo-StableJson {
    param($Value)
    return ($Value | ConvertTo-Json -Depth 100 -Compress)
}

function ConvertTo-CaseAuthorityJson {
    param($Case)
    $copy=(ConvertTo-StableJson $Case | ConvertFrom-Json -Depth 100)
    $copy.PSObject.Properties.Remove('caseResult')
    return ConvertTo-StableJson $copy
}

function Normalize-Text {
    param([AllowNull()][string]$Text)
    if ($null -eq $Text) { return '' }
    return (($Text.Normalize([Text.NormalizationForm]::FormKC).ToLowerInvariant() -replace '[\p{P}\p{S}\s]+', ' ').Trim())
}

function Get-CharacterErrorRate {
    param([string]$Actual, [string]$Expected)
    $a = @($Actual.EnumerateRunes() | ForEach-Object ToString)
    $b = @($Expected.EnumerateRunes() | ForEach-Object ToString)
    if ($b.Count -eq 0) { return $(if ($a.Count -eq 0) { 0.0 } else { 1.0 }) }
    $previous = [int[]]::new($b.Count + 1)
    for ($j = 0; $j -le $b.Count; $j++) { $previous[$j] = $j }
    for ($i = 1; $i -le $a.Count; $i++) {
        $current = [int[]]::new($b.Count + 1); $current[0] = $i
        for ($j = 1; $j -le $b.Count; $j++) {
            $cost = if ($a[$i - 1] -ceq $b[$j - 1]) { 0 } else { 1 }
            $current[$j] = [Math]::Min([Math]::Min($current[$j - 1] + 1, $previous[$j] + 1), $previous[$j - 1] + $cost)
        }
        $previous = $current
    }
    return [double]$previous[$b.Count] / [double]$b.Count
}

function Test-TokenTraceIntegrity {
    param([int[]]$Steps,[int[]]$Ids,[int]$GeneratedCount)
    if($GeneratedCount -lt 0 -or $Steps.Count -ne $GeneratedCount -or $Ids.Count -ne $GeneratedCount){return $false}
    for($i=0;$i -lt $GeneratedCount;$i++){if($Steps[$i] -ne $i -or $Ids[$i] -lt 0){return $false}}
    return $true
}

function Test-ComparatorOnlyChanges {
    param([string[]]$Paths)
    foreach($path in $Paths){$normalized=$path.Replace('\','/');if($normalized -notin @('compare_accuracy.ps1','tools/compare_audio.py')){return $false}}
    return $true
}

function Test-CaseTokenSequenceAvailable {
    param($Case)
    if((Get-OptionalPropertyValue $Case 'tokenSequenceAvailable') -ne $true){return $false}
    $trace=Get-OptionalPropertyValue $Case 'tokenSequenceTrace';$ids=@(Get-OptionalPropertyValue $Case 'generatedTokenIds')
    if($null -eq $trace){return $false}
    $steps=@(Get-OptionalPropertyValue $trace 'steps');$count=Get-OptionalPropertyValue $trace 'generatedCount'
    if($null -eq $count -or (Get-OptionalPropertyValue $trace 'complete') -ne $true){return $false}
    return Test-TokenTraceIntegrity $steps $ids ([int]$count)
}

function Get-CaseMap {
    param($Accuracy, [string]$Label)
    $map = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    foreach ($case in @((Get-PropertyValue $Accuracy 'cases' $Label))) {
        $name = [string](Get-PropertyValue $case 'name' "$Label case")
        if($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or $name.Contains('..')){throw "$Label contains unsafe case name '$name'."}
        if (-not $map.TryAdd($name, $case)) { throw "$Label has duplicate case '$name'." }
    }
    if ($map.Count -eq 0) { throw "$Label contains no cases." }
    return $map
}

function Assert-ModelIdentity {
    param($Model, [string]$Label)
    $modelPath = Resolve-ExistingPath -Path ([string](Get-PropertyValue $Model 'path' $Label)) -Label "$Label model" -PathType Container
    foreach ($metadata in @((Get-PropertyValue $Model 'metadataFiles' $Label))) {
        $null = Assert-Artifact $metadata "$Label metadata" $modelPath
    }
    foreach ($weight in @((Get-PropertyValue $Model 'weightManifest' $Label))) {
        $relative = [string](Get-PropertyValue $weight 'relativePath' "$Label weight")
        $weightPath = Resolve-ExistingPath -Path (Join-Path $modelPath $relative) -Label "$Label weight"
        if (-not (Test-PathWithinRoot $weightPath $modelPath)) { throw "$Label weight escapes its model: $weightPath" }
        if ([long](Get-PropertyValue $weight 'size' "$Label weight") -ne (Get-Item $weightPath).Length) { throw "$Label weight size is stale: $weightPath" }
        if (-not ([string](Get-PropertyValue $weight 'sha256' "$Label weight")).Equals((Get-FileSha256 $weightPath), [StringComparison]::OrdinalIgnoreCase)) {
            throw "$Label weight hash is stale: $weightPath"
        }
    }
    return $modelPath
}

function Assert-CaseArtifacts {
    param($Case, [string]$Label, [string]$AccuracyDir)
    $null = Assert-Artifact (Get-PropertyValue $Case 'output' $Label) "$Label output" $AccuracyDir
    $caseResultPath = Assert-Artifact (Get-PropertyValue $Case 'caseResult' $Label) "$Label caseResult" $AccuracyDir
    $authoritative=Get-Content -Raw -LiteralPath $caseResultPath | ConvertFrom-Json -Depth 100
    if((ConvertTo-CaseAuthorityJson $Case) -cne (ConvertTo-CaseAuthorityJson $authoritative)){throw "$Label embedded case does not match its authoritative caseResult JSON."}
    $logs = Get-PropertyValue $Case 'logs' $Label
    $null = Assert-Artifact (Get-PropertyValue $logs 'stdout' "$Label logs") "$Label stdout" $AccuracyDir
    $null = Assert-Artifact (Get-PropertyValue $logs 'stderr' "$Label logs") "$Label stderr" $AccuracyDir
    $null = Assert-ModelIdentity (Get-PropertyValue $Case 'model' $Label) $Label
    foreach ($field in @('input','audio','referenceAudio')) {
        $property = $Case.PSObject.Properties[$field]
        if ($null -ne $property -and $null -ne $property.Value) { $null = Assert-Artifact $property.Value "$Label $field" }
    }
}

function Assert-ComparableAccuracy {
    param([string]$BaselinePath, [string]$CandidatePath)
    $baseDir = Resolve-ExistingPath $BaselinePath 'BaselineDir' 'Container'
    $candDir = Resolve-ExistingPath $CandidatePath 'CandidateDir' 'Container'
    if ($baseDir.Equals($candDir, [StringComparison]::OrdinalIgnoreCase)) { throw 'BaselineDir and CandidateDir must differ.' }
    $baseFile = Resolve-ExistingPath (Join-Path $baseDir 'accuracy.json') 'baseline accuracy.json'
    $candFile = Resolve-ExistingPath (Join-Path $candDir 'accuracy.json') 'candidate accuracy.json'
    $base = Get-Content -Raw -LiteralPath $baseFile | ConvertFrom-Json -Depth 100
    $cand = Get-Content -Raw -LiteralPath $candFile | ConvertFrom-Json -Depth 100
    Assert-IntegerOne (Get-PropertyValue $base 'schemaVersion' 'baseline accuracy') 'baseline schemaVersion'
    Assert-IntegerOne (Get-PropertyValue $cand 'schemaVersion' 'candidate accuracy') 'candidate schemaVersion'
    foreach($pair in @(@($base,'baseline'),@($cand,'candidate'))){
        $doc,$label=$pair
        $execution=Get-PropertyValue $doc 'executionPassed' "$label accuracy"
        if($execution -isnot [bool]){throw "$label accuracy executionPassed must be boolean."}
        $status=[string](Get-PropertyValue $doc 'status' "$label accuracy")
        if($status -notin @('passed','completed-with-expectation-failures','execution-failed')){throw "$label accuracy has unsupported status '$status'."}
        $build=Get-PropertyValue $doc 'build' "$label accuracy"
        $buildDir=Resolve-ExistingPath ([string](Get-PropertyValue $build 'buildDir' "$label build")) "$label buildDir" 'Container'
        if([string]::IsNullOrWhiteSpace([string](Get-PropertyValue $build 'buildType' "$label build")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyValue $build 'generator' "$label build"))){throw "$label build metadata is incomplete."}
        $null=$buildDir
    }
    $baseOneApi = Get-PropertyValue $base 'oneApi' 'baseline accuracy'
    $candOneApi = Get-PropertyValue $cand 'oneApi' 'candidate accuracy'
    if ([string](Get-PropertyValue $baseOneApi 'role' 'baseline oneApi') -cne 'baseline') { throw 'Baseline accuracy oneApi role must be baseline.' }
    if ([string](Get-PropertyValue $candOneApi 'role' 'candidate oneApi') -cne 'candidate') { throw 'Candidate accuracy oneApi role must be candidate.' }
    if ([string](Get-PropertyValue $baseOneApi 'name' 'baseline oneApi') -cne 'oneapi-2025.3') { throw 'Baseline stack name must be oneapi-2025.3.' }
    if ([string](Get-PropertyValue $candOneApi 'name' 'candidate oneApi') -cne 'oneapi-2026.1') { throw 'Candidate stack name must be oneapi-2026.1.' }
    if((ConvertTo-StableJson (Get-PropertyValue $base 'currentOneApi' 'baseline accuracy')) -cne (ConvertTo-StableJson $baseOneApi)){throw 'Baseline currentOneApi does not match recorded oneApi.'}
    if((ConvertTo-StableJson (Get-PropertyValue $cand 'currentOneApi' 'candidate accuracy')) -cne (ConvertTo-StableJson $candOneApi)){throw 'Candidate currentOneApi does not match recorded oneApi.'}
    $baseCommit = [string](Get-PropertyValue (Get-PropertyValue $base 'git' 'baseline accuracy') 'fullCommit' 'baseline git')
    $candCommit = [string](Get-PropertyValue (Get-PropertyValue $cand 'git' 'candidate accuracy') 'fullCommit' 'candidate git')
    if ($baseCommit -cne $candCommit) { throw "Accuracy git commits differ: $baseCommit vs $candCommit" }
    $null = & git -C $script:RepoRoot cat-file -e "${baseCommit}^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Accuracy git commit does not exist locally: '$baseCommit'." }
    $null=& git -C $script:RepoRoot diff --quiet -- .;if($LASTEXITCODE -ne 0){throw 'Tracked working tree must be clean before comparing accuracy artifacts.'}
    $null=& git -C $script:RepoRoot diff --cached --quiet -- .;if($LASTEXITCODE -ne 0){throw 'Git index must be clean before comparing accuracy artifacts.'}
    $changed=@(& git -C $script:RepoRoot diff --name-only $baseCommit HEAD --);if($LASTEXITCODE -ne 0){throw 'Failed to inspect changes since accuracy artifact commit.'}
    if(-not (Test-ComparatorOnlyChanges $changed)){throw "Accuracy artifacts are stale because non-comparator tracked files changed since $baseCommit`: $($changed -join ', ')"}
    foreach ($pair in @(@($base,'baseline',$baseDir), @($cand,'candidate',$candDir))) {
        $doc, $label, $dir = $pair
        $null = Assert-Artifact (Get-PropertyValue $doc 'buildInfo' "$label accuracy") "$label buildInfo"
        $null = Assert-Artifact (Get-PropertyValue $doc 'executable' "$label accuracy") "$label executable"
        $null = Assert-Artifact (Get-PropertyValue $doc 'casesFile' "$label accuracy") "$label casesFile"
        foreach ($case in @((Get-PropertyValue $doc 'cases' "$label accuracy"))) { Assert-CaseArtifacts $case "$label case '$($case.name)'" $dir }
    }
    if ([string](Get-PropertyValue (Get-PropertyValue $base 'casesFile' 'baseline') 'sha256' 'baseline casesFile') -cne
        [string](Get-PropertyValue (Get-PropertyValue $cand 'casesFile' 'candidate') 'sha256' 'candidate casesFile')) { throw 'Accuracy cases-file hashes differ.' }
    $baseNames = @((Get-PropertyValue $base 'selectedCaseNames' 'baseline accuracy'))
    $candNames = @((Get-PropertyValue $cand 'selectedCaseNames' 'candidate accuracy'))
    if ((ConvertTo-StableJson $baseNames) -cne (ConvertTo-StableJson $candNames)) { throw 'Selected case sets or order differ.' }
    $baseMap = Get-CaseMap $base 'baseline accuracy'; $candMap = Get-CaseMap $cand 'candidate accuracy'
    if ($baseMap.Count -ne $candMap.Count) { throw 'Accuracy case sets differ.' }
    foreach ($name in $baseMap.Keys) {
        if (-not $candMap.ContainsKey($name)) { throw "Candidate is missing case '$name'." }
        $b, $c = $baseMap[$name], $candMap[$name]
        if ([string]$b.kind -cne [string]$c.kind) { throw "Case '$name' kind differs." }
        if ((ConvertTo-StableJson $b.config) -cne (ConvertTo-StableJson $c.config)) { throw "Case '$name' config differs." }
        $bModel = $b.model | Select-Object -Property metadataFiles,weightManifest
        $cModel = $c.model | Select-Object -Property metadataFiles,weightManifest
        if ((ConvertTo-StableJson $bModel) -cne (ConvertTo-StableJson $cModel)) { throw "Case '$name' model manifest differs." }
        foreach ($field in @('input','audio','referenceAudio')) {
            $bp, $cp = $b.PSObject.Properties[$field], $c.PSObject.Properties[$field]
            if (($null -eq $bp) -ne ($null -eq $cp)) { throw "Case '$name' $field provenance differs." }
            if ($null -ne $bp -and [string]$bp.Value.sha256 -cne [string]$cp.Value.sha256) { throw "Case '$name' $field hash differs." }
        }
    }
    return [pscustomobject]@{ BaselineDir=$baseDir; CandidateDir=$candDir; Baseline=$base; Candidate=$cand; BaselineMap=$baseMap; CandidateMap=$candMap; Commit=$baseCommit }
}

function New-CaseResult {
    param($BaselineCase, $CandidateCase)
    return [ordered]@{
        name = [string]$BaselineCase.name
        kind = [string]$BaselineCase.kind
        passed = $true
        automaticPassed = $true
        manualReviewRequired = $false
        status = 'pass'
        reasons = [Collections.Generic.List[string]]::new()
        baseline = [ordered]@{ status=$BaselineCase.status; executionPassed=[bool]$BaselineCase.executionPassed; expectationPassed=(Get-OptionalPropertyValue $BaselineCase 'expectationPassed'); normalizedText=(Get-OptionalPropertyValue $BaselineCase 'normalizedText'); output=$BaselineCase.output }
        candidate = [ordered]@{ status=$CandidateCase.status; executionPassed=[bool]$CandidateCase.executionPassed; expectationPassed=(Get-OptionalPropertyValue $CandidateCase 'expectationPassed'); normalizedText=(Get-OptionalPropertyValue $CandidateCase 'normalizedText'); output=$CandidateCase.output }
        metrics = [ordered]@{}
    }
}

function Set-CaseFailure { param($Result,[string]$Reason) $Result.passed=$false; $Result.automaticPassed=$false; $Result.status='failed'; $Result.reasons.Add($Reason) }
function Set-CaseManual { param($Result,[string]$Reason) $Result.manualReviewRequired=$true; if($Result.passed){$Result.status='manual-review-required'}; $Result.reasons.Add($Reason) }

function Test-VisionCase {
    param($Case)
    $config=Get-OptionalPropertyValue $Case 'config'
    $pattern=Get-OptionalPropertyValue $config 'expectRegex'
    return -not [string]::IsNullOrWhiteSpace([string]$pattern)
}

function Compare-ChatCase {
    param($BaselineCase,$CandidateCase,$Result)
    if (-not [bool]$BaselineCase.executionPassed -or -not [bool]$CandidateCase.executionPassed) { Set-CaseFailure $Result 'chat execution failed'; return }
    $isVision=Test-VisionCase $BaselineCase
    $sameText = [string]$BaselineCase.normalizedText -ceq [string]$CandidateCase.normalizedText
    $sameArtifact = [string]$BaselineCase.output.sha256 -ceq [string]$CandidateCase.output.sha256
    $baseTokensAvailable=Test-CaseTokenSequenceAvailable $BaselineCase
    $candidateTokensAvailable=Test-CaseTokenSequenceAvailable $CandidateCase
    $tokenEqual=$false
    if($baseTokensAvailable -and $candidateTokensAvailable){$tokenEqual=((ConvertTo-StableJson (Get-OptionalPropertyValue $BaselineCase 'generatedTokenIds')) -ceq (ConvertTo-StableJson (Get-OptionalPropertyValue $CandidateCase 'generatedTokenIds')))}
    $Result.metrics.normalizedTextEqual = $sameText; $Result.metrics.outputArtifactEqual = $sameArtifact; $Result.metrics.tokenSequenceAvailable=($baseTokensAvailable -and $candidateTokensAvailable);$Result.metrics.tokenSequenceEqual=$tokenEqual
    if ($isVision) {
        $pattern=[string]$BaselineCase.config.expectRegex
        try{$baseExpected=([string]$BaselineCase.rawText -match $pattern);$candExpected=([string]$CandidateCase.rawText -match $pattern)}catch{Set-CaseFailure $Result "vision expectRegex is invalid: $($_.Exception.Message)";return}
        $Result.metrics.baselineRegexPassed=$baseExpected; $Result.metrics.candidateRegexPassed=$candExpected
        if (-not $baseExpected -or -not $candExpected) { Set-CaseFailure $Result 'vision output failed expectRegex'; return }
        if (-not $sameText) { Set-CaseManual $Result 'vision outputs both satisfy expectRegex but normalized texts differ' }
        if (-not $sameArtifact) { Set-CaseManual $Result 'vision outputs both satisfy expectRegex but raw text artifacts differ' }
        if($baseTokensAvailable -and $candidateTokensAvailable){if(-not $tokenEqual){Set-CaseManual $Result 'vision outputs both satisfy expectRegex but generated token sequences differ'}}else{Set-CaseManual $Result 'vision generated token sequence is unavailable for one or both stacks'}
        return
    }
    if($baseTokensAvailable -ne $candidateTokensAvailable){Set-CaseFailure $Result 'chat token-sequence provenance is present for only one stack';return}
    if($baseTokensAvailable -and -not $tokenEqual){Set-CaseFailure $Result 'greedy chat generated token sequences differ';return}
    if(-not $sameText -or -not $sameArtifact){Set-CaseFailure $Result 'greedy chat normalized text or output artifact differs';return}
    if(-not $baseTokensAvailable){Set-CaseManual $Result 'generated token sequence is unavailable; exact raw output matches but token equality cannot be claimed'}
}

function Compare-AsrCase {
    param($BaselineCase,$CandidateCase,$Result)
    if (-not [bool]$BaselineCase.executionPassed -or -not [bool]$CandidateCase.executionPassed) { Set-CaseFailure $Result 'ASR execution failed'; return }
    $expected = Normalize-Text ([string]$BaselineCase.config.expectedText)
    $baseText = Normalize-Text ([string]$BaselineCase.normalizedText); $candText = Normalize-Text ([string]$CandidateCase.normalizedText)
    $baseCer = Get-CharacterErrorRate $baseText $expected; $candCer = Get-CharacterErrorRate $candText $expected
    $Result.metrics.expectedNormalizedText=$expected; $Result.metrics.baselineCer=$baseCer; $Result.metrics.candidateCer=$candCer
    $Result.metrics.exactTextEqual=($baseText -ceq $candText); $Result.metrics.baselineExpectationPassed=($baseText -ceq $expected); $Result.metrics.candidateExpectationPassed=($candText -ceq $expected)
    if ($candCer -gt $baseCer + 1e-12) { Set-CaseFailure $Result 'candidate ASR character error rate is worse than baseline' }
    if ($baseText -cne $expected -or $candText -cne $expected -or $BaselineCase.manualReviewRequired -eq $true -or $CandidateCase.manualReviewRequired -eq $true) {
        Set-CaseManual $Result 'ASR absolute expected-text check requires review'
    }
}

function Compare-AlignCase {
    param($BaselineCase,$CandidateCase,$Result)
    if (-not [bool]$BaselineCase.executionPassed -or -not [bool]$CandidateCase.executionPassed) { Set-CaseFailure $Result 'aligner execution failed'; return }
    $a=@($BaselineCase.alignment); $b=@($CandidateCase.alignment); $tol=[int]$BaselineCase.config.timestampToleranceMs
    $margin=[Math]::Max(1,[Math]::Min(10,[Math]::Ceiling($tol*0.1)))
    $Result.metrics.timestampToleranceMs=$tol; $Result.metrics.boundaryMarginMs=$margin; $Result.metrics.wordCountEqual=($a.Count -eq $b.Count)
    if ($a.Count -eq 0 -or $b.Count -eq 0) { Set-CaseFailure $Result 'aligner output is empty'; return }
    if ($a.Count -ne $b.Count) { Set-CaseFailure $Result 'aligner word counts differ'; return }
    [int64]$maxStart=0; [int64]$maxEnd=0; $boundary=$false
    for($i=0;$i -lt $a.Count;$i++){
        $baseStart=Get-OptionalPropertyValue $a[$i] 'startMs';$baseEnd=Get-OptionalPropertyValue $a[$i] 'endMs';$candidateStart=Get-OptionalPropertyValue $b[$i] 'startMs';$candidateEnd=Get-OptionalPropertyValue $b[$i] 'endMs'
        foreach($timestamp in @(
            [pscustomobject]@{label='baseline startMs';value=$baseStart},[pscustomobject]@{label='baseline endMs';value=$baseEnd},
            [pscustomobject]@{label='candidate startMs';value=$candidateStart},[pscustomobject]@{label='candidate endMs';value=$candidateEnd}
        )){
            if(-not (Test-Int64Integer $timestamp.value) -or [int64]$timestamp.value -lt 0){Set-CaseFailure $Result "aligner $($timestamp.label) must be a nonnegative integer within Int64 range at index $i";return}
        }
        $baseStart=[int64]$baseStart;$baseEnd=[int64]$baseEnd;$candidateStart=[int64]$candidateStart;$candidateEnd=[int64]$candidateEnd
        if ([string]$a[$i].text -cne [string]$b[$i].text) { Set-CaseFailure $Result "aligner word mismatch at index $i"; return }
        if($baseEnd -lt $baseStart -or $candidateEnd -lt $candidateStart){Set-CaseFailure $Result "aligner interval has end before start at index $i";return}
        if($i -gt 0 -and (($baseStart -lt [int64]$a[$i-1].startMs) -or ($baseEnd -lt [int64]$a[$i-1].endMs) -or ($candidateStart -lt [int64]$b[$i-1].startMs) -or ($candidateEnd -lt [int64]$b[$i-1].endMs))){Set-CaseFailure $Result "aligner timestamps are nonmonotonic at index $i";return}
        $sd=[Math]::Abs($baseStart-$candidateStart); $ed=[Math]::Abs($baseEnd-$candidateEnd)
        $maxStart=[Math]::Max($maxStart,$sd); $maxEnd=[Math]::Max($maxEnd,$ed); if(($sd -ge ($tol-$margin) -and $sd -le $tol) -or ($ed -ge ($tol-$margin) -and $ed -le $tol)){$boundary=$true}
        if($sd -gt $tol -or $ed -gt $tol){Set-CaseFailure $Result "aligner timestamp delta exceeds ${tol}ms at index $i"; return}
    }
    $Result.metrics.maxStartDeltaMs=$maxStart; $Result.metrics.maxEndDeltaMs=$maxEnd; $Result.metrics.atToleranceBoundary=$boundary
    $Result.metrics.nearToleranceBoundary=$boundary
    if($boundary){Set-CaseManual $Result 'aligner timestamp delta is within the configured near-boundary margin'}
}

function Invoke-PythonJson {
    param([string[]]$Arguments)
    $lines = & python @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python audio helper failed ($LASTEXITCODE): $($lines -join [Environment]::NewLine)" }
    return ($lines -join [Environment]::NewLine | ConvertFrom-Json -Depth 100)
}

function Extract-SpeakerEmbedding {
    param($Context,$Case,[string]$Role,[string]$OutputDir)
    $doc = if($Role -eq 'baseline'){$Context.Baseline}else{$Context.Candidate}
    $buildDir=[string]$doc.build.buildDir; $model=[string]$Case.model.path; $audio=[string]$Case.output.path
    $path=Join-Path $OutputDir ("speaker_embedding_{0}_{1}.json" -f $Role,$Case.name)
    $args=@((Join-Path $script:RepoRoot 'tools/compare_audio.py'),'--extract-embedding','--build-dir',$buildDir,'--model',$model,'--audio',$audio,'--output',$path)
    $output=& python @args 2>&1
    if($LASTEXITCODE -ne 0){throw "Speaker embedding extraction failed for ${Role}: $($output -join [Environment]::NewLine)"}
    return Get-Content -Raw -LiteralPath $path | ConvertFrom-Json -Depth 100
}

function Apply-TtsMetricGates {
    param($Result,$Metrics,[bool]$AudioHashEqual,[AllowNull()]$SpeakerCosine,[switch]$SkipEmbedding)
    $valid=$true
    foreach($spec in @(
        [pscustomobject]@{name='duration_delta_frames';min=0;max=[int64]::MaxValue;integer=$true;required=$true},
        [pscustomobject]@{name='correlation';min=-1;max=1;integer=$false;required=$true},
        [pscustomobject]@{name='relative_rms_delta';min=0;max=$null;integer=$false;required=$true},
        [pscustomobject]@{name='log_mel_mae';min=0;max=$null;integer=$false;required=$true},
        [pscustomobject]@{name='clipping_ratio_delta';min=0;max=1;integer=$false;required=$true},
        [pscustomobject]@{name='silence_ratio_delta';min=0;max=1;integer=$false;required=$true},
        [pscustomobject]@{name='sample_rate';min=1;max=[int]::MaxValue;integer=$true;required=$false},
        [pscustomobject]@{name='reference_frames';min=1;max=[int64]::MaxValue;integer=$true;required=$false},
        [pscustomobject]@{name='candidate_frames';min=1;max=[int64]::MaxValue;integer=$true;required=$false},
        [pscustomobject]@{name='reference_rms';min=0;max=$null;integer=$false;required=$false},
        [pscustomobject]@{name='candidate_rms';min=0;max=$null;integer=$false;required=$false},
        [pscustomobject]@{name='reference_clipping_ratio';min=0;max=1;integer=$false;required=$false},
        [pscustomobject]@{name='candidate_clipping_ratio';min=0;max=1;integer=$false;required=$false},
        [pscustomobject]@{name='reference_silence_ratio';min=0;max=1;integer=$false;required=$false},
        [pscustomobject]@{name='candidate_silence_ratio';min=0;max=1;integer=$false;required=$false}
    )){
        if(-not (Test-TtsNumericProperty $Result $Metrics $spec.name 'TTS audio metric' $spec.min $spec.max -Integer:$spec.integer -Required:$spec.required)){$valid=$false}
    }
    if($null -ne $SpeakerCosine -and (-not (Test-FiniteNumber $SpeakerCosine) -or [double]$SpeakerCosine -lt -1 -or [double]$SpeakerCosine -gt 1)){Set-CaseFailure $Result 'TTS speaker embedding cosine must be finite and within [-1, 1]';$valid=$false}
    if(-not $valid){return}
    if([int64]$Metrics.duration_delta_frames -gt $script:AudioThresholds.durationDeltaFramesMaximum){Set-CaseFailure $Result 'TTS duration delta exceeds 2000 frames'}
    $embeddingMaterial=(-not $SkipEmbedding -and $null -ne $SpeakerCosine -and [double]$SpeakerCosine -lt $script:AudioThresholds.speakerEmbeddingCosineManualBelow)
    if(-not $AudioHashEqual -or $embeddingMaterial){
        $material=([double]$Metrics.correlation -lt $script:AudioThresholds.correlationManualBelow -or [double]$Metrics.relative_rms_delta -gt $script:AudioThresholds.relativeRmsDeltaManualAbove -or [double]$Metrics.log_mel_mae -gt $script:AudioThresholds.logMelMaeManualAbove -or [double]$Metrics.clipping_ratio_delta -gt $script:AudioThresholds.clippingRatioDeltaManualAbove -or [double]$Metrics.silence_ratio_delta -gt $script:AudioThresholds.silenceRatioDeltaManualAbove)
        if($embeddingMaterial){$material=$true}
        if($material){Set-CaseManual $Result 'TTS waveform or speaker embedding differs materially; listening review required'}
    }
}

function Apply-TtsFormatGates {
    param($Result,$Reference,$Candidate)
    $valid=$true
    foreach($side in @([pscustomobject]@{label='reference';value=$Reference},[pscustomobject]@{label='candidate';value=$Candidate})){
        foreach($spec in @(
            [pscustomobject]@{name='channels';min=1;required=$true},[pscustomobject]@{name='audioFormat';min=$null;required=$true},[pscustomobject]@{name='bitsPerSample';min=1;required=$true},
            [pscustomobject]@{name='sampleRate';min=1;required=$false},[pscustomobject]@{name='frameCount';min=1;required=$false},[pscustomobject]@{name='blockAlign';min=1;required=$false},[pscustomobject]@{name='byteRate';min=1;required=$false}
        )){
            if(-not (Test-TtsNumericProperty $Result $side.value $spec.name "TTS WAV $($side.label)" $spec.min ([int]::MaxValue) -Integer -Required:$spec.required)){$valid=$false}
        }
        $format=Get-OptionalPropertyValue $side.value 'audioFormat';$bits=Get-OptionalPropertyValue $side.value 'bitsPerSample'
        $formatSafe=(Test-IntegerNumber $format) -and [double]$format -ge [int]::MinValue -and [double]$format -le [int]::MaxValue
        $bitsSafe=(Test-IntegerNumber $bits) -and [double]$bits -ge [int]::MinValue -and [double]$bits -le [int]::MaxValue
        if($formatSafe -and [int]$format -notin @(1,3)){Set-CaseFailure $Result "TTS WAV $($side.label) audioFormat is unsupported: $format";$valid=$false}
        if($formatSafe -and $bitsSafe){
            if(([int]$format -eq 1 -and [int]$bits -notin @(8,16,24,32)) -or ([int]$format -eq 3 -and [int]$bits -notin @(32,64))){Set-CaseFailure $Result "TTS WAV $($side.label) bit depth $bits is unsupported for audioFormat $format";$valid=$false}
        }
    }
    if(-not $valid){return}
    if([int]$Reference.channels -ne [int]$Candidate.channels){Set-CaseFailure $Result 'TTS WAV channel counts differ'}
    if([int]$Reference.audioFormat -ne [int]$Candidate.audioFormat){Set-CaseFailure $Result 'TTS WAV audio formats differ'}
    if([int]$Reference.bitsPerSample -ne [int]$Candidate.bitsPerSample){Set-CaseFailure $Result 'TTS WAV bit depths differ'}
    if($null -ne $Reference.PSObject.Properties['blockAlign'] -and $null -ne $Candidate.PSObject.Properties['blockAlign'] -and [int]$Reference.blockAlign -ne [int]$Candidate.blockAlign){Set-CaseFailure $Result 'TTS WAV block alignment differs'}
}

function Compare-SpeakerEmbeddings {
    param($Result,$BaselineEmbedding,$CandidateEmbedding,[string]$WorkDir,[string]$CaseName)
    if([int]$BaselineEmbedding.dimension -ne [int]$CandidateEmbedding.dimension){Set-CaseFailure $Result 'speaker embedding dimensions differ';return}
    $tmp=Join-Path $WorkDir ("embedding_pair_{0}.json" -f $CaseName)
    [ordered]@{reference=$BaselineEmbedding.values;candidate=$CandidateEmbedding.values}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $tmp -Encoding utf8
    $cosScript='import json,sys;from tools.compare_audio import cosine_similarity;x=json.load(open(sys.argv[1],encoding="utf-8-sig"));print(cosine_similarity(x["reference"],x["candidate"]))'
    $cosOut=@(& python -c $cosScript $tmp 2>&1); if($LASTEXITCODE -ne 0){Set-CaseFailure $Result "speaker cosine calculation failed: $($cosOut -join ' ')";return}
    try{$cosine=[double]::Parse([string]($cosOut | Select-Object -Last 1),[Globalization.CultureInfo]::InvariantCulture)}catch{Set-CaseFailure $Result "speaker cosine is not numeric: $($cosOut -join ' ')";return}
    if([double]::IsNaN($cosine)-or[double]::IsInfinity($cosine)-or$cosine -lt -1.0000001-or$cosine -gt 1.0000001){Set-CaseFailure $Result "speaker cosine is invalid: $cosine";return}
    $Result.metrics.speakerEmbeddingCosine=$cosine; $Result.metrics.speaker_embedding_cosine=$cosine
    $Result.metrics.baselineEmbedding=[ordered]@{dimension=$BaselineEmbedding.dimension;sha256=$BaselineEmbedding.sha256;norm=$BaselineEmbedding.norm}
    $Result.metrics.candidateEmbedding=[ordered]@{dimension=$CandidateEmbedding.dimension;sha256=$CandidateEmbedding.sha256;norm=$CandidateEmbedding.norm}
}

function Compare-TtsCase {
    param($Context,$BaselineCase,$CandidateCase,$Result,[string]$WorkDir,[switch]$SkipEmbedding)
    if (-not [bool]$BaselineCase.executionPassed -or -not [bool]$CandidateCase.executionPassed) { Set-CaseFailure $Result 'TTS execution failed'; return }
    try{$audio=Invoke-PythonJson @((Join-Path $script:RepoRoot 'tools/compare_audio.py'),'--reference',[string]$BaselineCase.output.path,'--candidate',[string]$CandidateCase.output.path)}catch{Set-CaseFailure $Result "TTS WAV validation or metric calculation failed: $($_.Exception.Message)";return}
    $m=$audio.metrics; $Result.metrics.audio=$m; $Result.metrics.audioContentHashEqual=([string]$BaselineCase.output.sha256 -ceq [string]$CandidateCase.output.sha256)
    Apply-TtsFormatGates $Result $audio.reference $audio.candidate
    $Result.metrics.speakerEmbeddingCosine=$null; $Result.metrics.speaker_embedding_cosine=$null
    if(-not $SkipEmbedding){
        try{$eb=Extract-SpeakerEmbedding $Context $BaselineCase 'baseline' $WorkDir; $ec=Extract-SpeakerEmbedding $Context $CandidateCase 'candidate' $WorkDir}catch{Set-CaseFailure $Result "speaker embedding extraction failed: $($_.Exception.Message)";return}
        Compare-SpeakerEmbeddings $Result $eb $ec $WorkDir ([string]$BaselineCase.name)
    } else { $Result.metrics.speakerEmbeddingSkipped=$true }
    Apply-TtsMetricGates $Result $m ([bool]$Result.metrics.audioContentHashEqual) $Result.metrics.speakerEmbeddingCosine -SkipEmbedding:$SkipEmbedding
}

function Get-ApprovedOutputPath {
    param([string]$Requested)
    $ownedRoot=[IO.Path]::GetFullPath((Join-Path $script:TmpRoot 'perf\oneapi-compare'));New-Item -ItemType Directory -Path $ownedRoot -Force|Out-Null
    $path=if([string]::IsNullOrWhiteSpace($Requested)){Join-Path $ownedRoot 'accuracy_comparison.json'}elseif([IO.Path]::IsPathFullyQualified($Requested)){$Requested}else{Join-Path $script:RepoRoot $Requested}
    $path=[IO.Path]::GetFullPath($path); if(-not (Test-PathWithinRoot $path $ownedRoot) -or $path.Equals($ownedRoot,[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($path) -ine '.json'){throw "OutputPath must be an owned JSON file inside '$ownedRoot': $path"}
    return $path
}

function Write-AtomicJson {
    param([string]$Path,$Data)
    $parent=Split-Path -Parent $Path; New-Item -ItemType Directory -Path $parent -Force|Out-Null
    $temp=Join-Path $parent ('.'+[IO.Path]::GetFileName($Path)+'.'+[guid]::NewGuid().ToString('N')+'.tmp')
    try{$Data|ConvertTo-Json -Depth 100|Set-Content -LiteralPath $temp -Encoding utf8; Move-Item -LiteralPath $temp -Destination $Path -Force}finally{if(Test-Path -LiteralPath $temp){Remove-Item -LiteralPath $temp -Force}}
}

function Invoke-Comparison {
    param([string]$Baseline,[string]$Candidate,[string]$Output,[switch]$SkipEmbedding)
    $outPath=Get-ApprovedOutputPath $Output
    if(Test-Path -LiteralPath $outPath -PathType Leaf){Remove-Item -LiteralPath $outPath -Force}
    $context=Assert-ComparableAccuracy $Baseline $Candidate
    $workDir=Join-Path (Split-Path -Parent $outPath) 'accuracy_comparison_artifacts'; New-Item -ItemType Directory -Path $workDir -Force|Out-Null
    $results=[Collections.Generic.List[object]]::new()
    foreach($name in @($context.Baseline.selectedCaseNames)){
        $b=$context.BaselineMap[[string]$name];$c=$context.CandidateMap[[string]$name];$r=New-CaseResult $b $c
        switch([string]$b.kind){'chat'{Compare-ChatCase $b $c $r};'asr'{Compare-AsrCase $b $c $r};'align'{Compare-AlignCase $b $c $r};'tts'{Compare-TtsCase $context $b $c $r $workDir -SkipEmbedding:$SkipEmbedding};default{Set-CaseFailure $r "unsupported case kind '$($b.kind)'"}}
        $results.Add([pscustomobject]$r)
    }
    $passed=-not @($results|Where-Object{-not $_.passed}).Count; $manual=[bool]@($results|Where-Object{$_.manualReviewRequired}).Count
    $status=if(-not $passed){'failed'}elseif($manual){'manual-review-required'}else{'pass'}
    $payload=[ordered]@{schemaVersion=1;passed=$passed;automaticPassed=$passed;manualReviewRequired=$manual;status=$status;generatedAtUtc=[DateTime]::UtcNow.ToString('o');gitCommit=$context.Commit;baseline=[ordered]@{directory=$context.BaselineDir;oneApi=$context.Baseline.oneApi};candidate=[ordered]@{directory=$context.CandidateDir;oneApi=$context.Candidate.oneApi};thresholds=$script:AudioThresholds;cases=$results}
    Write-AtomicJson $outPath $payload
    if(-not $passed){Write-Host ":: accuracy comparison FAILED; results written to $outPath ::" -ForegroundColor Red;return 1}
    if($manual){Write-Host ":: accuracy comparison MANUAL REVIEW REQUIRED; results written to $outPath ::" -ForegroundColor Yellow;return 0}
    Write-Host ":: accuracy comparison PASSED; results written to $outPath ::" -ForegroundColor Green;return 0
}

function Invoke-SelfTest {
    $failures=[Collections.Generic.List[string]]::new()
    function Check([bool]$Condition,[string]$Name){if(-not $Condition){$failures.Add($Name)}}
    Check ((Get-CharacterErrorRate '这是错事' '这是测试') -gt 0) 'Unicode CER'
    Check ((Normalize-Text ' A，B。 ') -ceq 'a b') 'Unicode normalization'
    Check (-not (Test-FiniteNumber $null)) 'finite-number helper rejects null safely'
    $base=[pscustomobject]@{name='chat';kind='chat';status='passed';executionPassed=$true;expectationPassed=$null;manualReviewRequired=$false;normalizedText='same';config=[pscustomobject]@{};output=[pscustomobject]@{sha256='A'}}
    $cand=$base.PSObject.Copy(); $r=New-CaseResult $base $cand; Compare-ChatCase $base $cand $r; Check $r.passed 'exact chat pass'
    $cand=$base.PSObject.Copy();$cand.normalizedText='different';$cand.output=[pscustomobject]@{sha256='B'};$r=New-CaseResult $base $cand;Compare-ChatCase $base $cand $r;Check (-not $r.passed) 'chat mismatch fail'
    $tokenBase=$base.PSObject.Copy();$tokenBase|Add-Member -NotePropertyName tokenSequenceAvailable -NotePropertyValue $true;$tokenBase|Add-Member -NotePropertyName generatedTokenIds -NotePropertyValue @(1,2);$tokenBase|Add-Member -NotePropertyName tokenSequenceTrace -NotePropertyValue ([pscustomobject]@{steps=@(0,1);generatedCount=2;complete=$true})
    $tokenCand=$tokenBase.PSObject.Copy();$tokenCand.generatedTokenIds=@(1,3);$r=New-CaseResult $tokenBase $tokenCand;Compare-ChatCase $tokenBase $tokenCand $r;Check (-not $r.passed) 'chat token mismatch fail'
    $malformed=$tokenBase.PSObject.Copy();$malformed.tokenSequenceTrace=[pscustomobject]@{steps=@(0,0);generatedCount=2;complete=$true};$r=New-CaseResult $malformed $malformed;Compare-ChatCase $malformed $malformed $r;Check ($r.passed -and $r.manualReviewRequired -and -not $r.metrics.tokenSequenceAvailable) 'malformed token trace downgraded to manual'
    $r=New-CaseResult $base $base;Compare-ChatCase $base $base $r;Check ($r.passed -and $r.manualReviewRequired -and -not $r.metrics.tokenSequenceAvailable) 'chat unavailable token sequence manual'
    Check (Test-TokenTraceIntegrity @(0,1,2) @(10,11,12) 3) 'token trace contiguous valid'
    Check (-not (Test-TokenTraceIntegrity @(0,2,2) @(10,11,12) 3)) 'token trace duplicate/gap rejected'
    Check (-not (Test-TokenTraceIntegrity @(1,0,2) @(10,11,12) 3)) 'token trace out-of-order rejected'
    Check (-not (Test-TokenTraceIntegrity @(0,1,2,3) @(10,11,12,13) 3)) 'token trace cap/count mismatch rejected'
    Check (Test-ComparatorOnlyChanges @('compare_accuracy.ps1','tools/compare_audio.py')) 'comparator-only provenance accepted'
    Check (-not (Test-ComparatorOnlyChanges @('compare_accuracy.ps1','regress.ps1'))) 'runner provenance change rejected'
    try{$null=Get-CaseMap ([pscustomobject]@{cases=@([pscustomobject]@{name='../escape'})}) 'unsafe';$failures.Add('unsafe case name accepted')}catch{}
    $authority=[pscustomobject]@{name='case';kind='chat';passed=$true};$embedded=[pscustomobject]@{name='case';kind='chat';passed=$false;caseResult=[pscustomobject]@{path='x'}};Check ((ConvertTo-CaseAuthorityJson $authority) -cne (ConvertTo-CaseAuthorityJson $embedded)) 'caseResult authority detects edited combined case'
    $visionBase=$base.PSObject.Copy();$visionBase.name='vision';$visionBase.config=[pscustomobject]@{expectRegex='ok'};$visionBase.expectationPassed=$true;$visionBase|Add-Member -NotePropertyName rawText -NotePropertyValue 'ok one';$visionBase|Add-Member -NotePropertyName tokenSequenceAvailable -NotePropertyValue $true;$visionBase|Add-Member -NotePropertyName generatedTokenIds -NotePropertyValue @(1,2)
    $visionCand=$visionBase.PSObject.Copy();$visionCand.normalizedText='other';$visionCand.rawText='ok two';$visionCand.output=[pscustomobject]@{sha256='B'};$visionCand.generatedTokenIds=@(1,3);$r=New-CaseResult $visionBase $visionCand;Compare-ChatCase $visionBase $visionCand $r;Check ($r.passed -and $r.manualReviewRequired -and $r.status -eq 'manual-review-required') 'vision valid-different text-artifact-token manual'
    $visionCand=$visionBase.PSObject.Copy();$visionCand.expectationPassed=$false;$visionCand.rawText='no';$r=New-CaseResult $visionBase $visionCand;Compare-ChatCase $visionBase $visionCand $r;Check (-not $r.passed) 'vision regex fail'
    $visionLike=$tokenBase.PSObject.Copy();$visionLike.name='vision_like_chat';$r=New-CaseResult $visionLike $visionLike;try{Compare-ChatCase $visionLike $visionLike $r;Check ($r.passed -and -not $r.manualReviewRequired) 'vision-like ordinary chat exact pass'}catch{$failures.Add("vision-like ordinary chat threw: $($_.Exception.Message)")}
    $visionLikeCandidate=$visionLike.PSObject.Copy();$visionLikeCandidate.normalizedText='different';$visionLikeCandidate.output=[pscustomobject]@{sha256='B'};$visionLikeCandidate.generatedTokenIds=@(1,3);$r=New-CaseResult $visionLike $visionLikeCandidate;try{Compare-ChatCase $visionLike $visionLikeCandidate $r;Check (-not $r.passed -and $r.status -eq 'failed' -and $r.reasons.Count -gt 0) 'vision-like ordinary chat mismatch structured fail'}catch{$failures.Add("vision-like ordinary chat mismatch threw: $($_.Exception.Message)")}
    $asrBase=[pscustomobject]@{name='asr';kind='asr';status='expectation-failed';executionPassed=$true;expectationPassed=$false;manualReviewRequired=$true;normalizedText='错';config=[pscustomobject]@{expectedText='对'};output=[pscustomobject]@{}}
    $asrCand=$asrBase.PSObject.Copy();$r=New-CaseResult $asrBase $asrCand;Compare-AsrCase $asrBase $asrCand $r;Check ($r.passed -and $r.automaticPassed -and $r.manualReviewRequired -and $r.status -eq 'manual-review-required') 'ASR equal-wrong manual pass'
    $asrCand=$asrBase.PSObject.Copy();$asrCand.normalizedText='更错';$r=New-CaseResult $asrBase $asrCand;Compare-AsrCase $asrBase $asrCand $r;Check (-not $r.passed) 'ASR worse CER fail'
    $alignBase=[pscustomobject]@{name='align';kind='align';status='passed';executionPassed=$true;expectationPassed=$null;manualReviewRequired=$false;normalizedText=$null;config=[pscustomobject]@{timestampToleranceMs=80};alignment=@([pscustomobject]@{text='字';startMs=0;endMs=80});output=[pscustomobject]@{}}
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=80;endMs=160});$r=New-CaseResult $alignBase $alignCand;Compare-AlignCase $alignBase $alignCand $r;Check ($r.passed -and $r.manualReviewRequired -and $r.metrics.atToleranceBoundary) 'align boundary manual'
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=81;endMs=80});$r=New-CaseResult $alignBase $alignCand;Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed) 'align over boundary fail'
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=71;endMs=151});$r=New-CaseResult $alignBase $alignCand;Compare-AlignCase $alignBase $alignCand $r;Check ($r.passed -and -not $r.manualReviewRequired) 'align ordinary delta pass'
    foreach($delta in 72,79,80){$alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=$delta;endMs=$delta+80});$r=New-CaseResult $alignBase $alignCand;Compare-AlignCase $alignBase $alignCand $r;Check ($r.passed -and $r.manualReviewRequired) "align near-boundary $delta manual"}
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=-1;endMs=80});$r=New-CaseResult $alignBase $alignCand;try{Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed -and $r.status -eq 'failed' -and [string]$r.reasons[0] -match 'nonnegative integer') 'align first-row negative start structured fail'}catch{$failures.Add("align first-row negative start threw: $($_.Exception.Message)")}
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=0;endMs=-1});$r=New-CaseResult $alignBase $alignCand;try{Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed -and $r.status -eq 'failed' -and [string]$r.reasons[0] -match 'endMs must be a nonnegative integer') 'align negative end structured fail'}catch{$failures.Add("align negative end threw: $($_.Exception.Message)")}
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=1e100;endMs=1e100});$r=New-CaseResult $alignBase $alignCand;try{Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed -and $r.status -eq 'failed') 'align oversized integer structured fail'}catch{$failures.Add("align oversized integer threw: $($_.Exception.Message)")}
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=[Math]::Pow(2,63);endMs=[Math]::Pow(2,63)});$r=New-CaseResult $alignBase $alignCand;try{Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed -and $r.status -eq 'failed') 'align Int64 boundary overflow structured fail'}catch{$failures.Add("align Int64 boundary overflow threw: $($_.Exception.Message)")}
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=[int64]3000000000;endMs=[int64]3000000080});$r=New-CaseResult $alignBase $alignCand;try{Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed -and $r.status -eq 'failed' -and [string]$r.reasons[0] -match 'delta exceeds') 'align large Int64 delta structured fail'}catch{$failures.Add("align large Int64 delta threw: $($_.Exception.Message)")}
    $largeFraction=[decimal]::Parse('9007199254740992.1',[Globalization.CultureInfo]::InvariantCulture);$alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=$largeFraction;endMs=$largeFraction});$r=New-CaseResult $alignBase $alignCand;try{Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed -and [string]$r.reasons[0] -match 'nonnegative integer') 'align large fractional decimal structured fail'}catch{$failures.Add("align large fractional decimal threw: $($_.Exception.Message)")}
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='字';startMs=0;endMs=80},[pscustomobject]@{text='二';startMs=-1;endMs=90});$alignBase.alignment=@([pscustomobject]@{text='字';startMs=0;endMs=80},[pscustomobject]@{text='二';startMs=80;endMs=160});$r=New-CaseResult $alignBase $alignCand;Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed) 'align nonmonotonic fail'
    $alignCand=$alignBase.PSObject.Copy();$alignCand.alignment=@([pscustomobject]@{text='词';startMs=0;endMs=80});$r=New-CaseResult $alignBase $alignCand;Compare-AlignCase $alignBase $alignCand $r;Check (-not $r.passed) 'align word mismatch fail'
    $ttsBase=[pscustomobject]@{name='tts';kind='tts';status='passed';executionPassed=$true;expectationPassed=$null;manualReviewRequired=$false;output=[pscustomobject]@{}}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$identical=[pscustomobject]@{duration_delta_frames=0;correlation=1.0;relative_rms_delta=0.0;log_mel_mae=0.0;clipping_ratio_delta=0.0;silence_ratio_delta=0.0};Apply-TtsMetricGates $ttsResult $identical $true $null -SkipEmbedding;Check ($ttsResult.passed -and -not $ttsResult.manualReviewRequired) 'TTS identical metrics'
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$long=$identical.PSObject.Copy();$long.duration_delta_frames=2001;Apply-TtsMetricGates $ttsResult $long $false $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS duration fail'
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$material=$identical.PSObject.Copy();$material.correlation=0.9;Apply-TtsMetricGates $ttsResult $material $false $null -SkipEmbedding;Check ($ttsResult.passed -and $ttsResult.manualReviewRequired) 'TTS material manual'
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid.correlation=[double]::NaN;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS NaN correlation structured fail'}catch{$failures.Add("TTS NaN correlation threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid.log_mel_mae=[double]::PositiveInfinity;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS infinite log-mel structured fail'}catch{$failures.Add("TTS infinite log-mel threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;try{Apply-TtsMetricGates $ttsResult $identical $true 1.1;Check (-not $ttsResult.passed) 'TTS out-of-range speaker cosine structured fail'}catch{$failures.Add("TTS out-of-range speaker cosine threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid.duration_delta_frames=-1;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS negative duration structured fail'}catch{$failures.Add("TTS negative duration threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid.duration_delta_frames=1e100;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS oversized duration structured fail'}catch{$failures.Add("TTS oversized duration threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid.duration_delta_frames=[Math]::Pow(2,63);try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS Int64 boundary duration structured fail'}catch{$failures.Add("TTS Int64 boundary duration threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid.relative_rms_delta=-0.1;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS negative RMS delta structured fail'}catch{$failures.Add("TTS negative RMS delta threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid|Add-Member -NotePropertyName sample_rate -NotePropertyValue 0;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS nonpositive sample rate structured fail'}catch{$failures.Add("TTS nonpositive sample rate threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid|Add-Member -NotePropertyName reference_frames -NotePropertyValue $largeFraction;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS large fractional frame count structured fail'}catch{$failures.Add("TTS large fractional frame count threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$invalid=$identical.PSObject.Copy();$invalid|Add-Member -NotePropertyName reference_clipping_ratio -NotePropertyValue 1.1;try{Apply-TtsMetricGates $ttsResult $invalid $true $null -SkipEmbedding;Check (-not $ttsResult.passed) 'TTS out-of-range clipping ratio structured fail'}catch{$failures.Add("TTS out-of-range clipping ratio threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;Apply-TtsFormatGates $ttsResult ([pscustomobject]@{channels=1;audioFormat=3;bitsPerSample=32}) ([pscustomobject]@{channels=2;audioFormat=3;bitsPerSample=32});Check (-not $ttsResult.passed) 'TTS channel mismatch fail'
    $ttsResult=New-CaseResult $ttsBase $ttsBase;Apply-TtsFormatGates $ttsResult ([pscustomobject]@{channels=1;audioFormat=1;bitsPerSample=16;blockAlign=2}) ([pscustomobject]@{channels=1;audioFormat=3;bitsPerSample=32;blockAlign=4});Check (-not $ttsResult.passed) 'TTS PCM-float format mismatch fail'
    $ttsResult=New-CaseResult $ttsBase $ttsBase;try{Apply-TtsFormatGates $ttsResult ([pscustomobject]@{channels=1;audioFormat=9;bitsPerSample=16}) ([pscustomobject]@{channels=1;audioFormat=9;bitsPerSample=16});Check (-not $ttsResult.passed) 'TTS unsupported WAV format structured fail'}catch{$failures.Add("TTS unsupported WAV format threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;try{Apply-TtsFormatGates $ttsResult ([pscustomobject]@{channels=1;audioFormat=1e100;bitsPerSample=16}) ([pscustomobject]@{channels=1;audioFormat=1;bitsPerSample=16});Check (-not $ttsResult.passed) 'TTS oversized WAV format structured fail'}catch{$failures.Add("TTS oversized WAV format threw: $($_.Exception.Message)")}
    $ttsResult=New-CaseResult $ttsBase $ttsBase;$ttsResult.metrics.speakerEmbeddingCosine=$null;$ttsResult.metrics.speaker_embedding_cosine=$null;Compare-SpeakerEmbeddings $ttsResult ([pscustomobject]@{dimension=2;values=@(1,0);sha256='a';norm=1}) ([pscustomobject]@{dimension=3;values=@(1,0,0);sha256='b';norm=1}) $script:TmpRoot 'dimension-test';Check (-not $ttsResult.passed -and $null -eq $ttsResult.metrics.speaker_embedding_cosine) 'embedding dimension structured fail'
    $ttsResult=New-CaseResult $ttsBase $ttsBase;Apply-TtsMetricGates $ttsResult $identical $true 0.9;Check ($ttsResult.passed -and $ttsResult.manualReviewRequired) 'exact TTS WAV low cosine manual'
    $badWav=Join-Path $script:TmpRoot ('bad-wav-'+[guid]::NewGuid().ToString('N')+'.wav');New-Item -ItemType Directory -Path $script:TmpRoot -Force|Out-Null;Set-Content -LiteralPath $badWav -Value 'not wav' -Encoding ascii
    try{$badBase=$ttsBase.PSObject.Copy();$badBase.output=[pscustomobject]@{path=$badWav;sha256='x'};$badCandidate=$badBase.PSObject.Copy();$r=New-CaseResult $badBase $badCandidate;Compare-TtsCase $null $badBase $badCandidate $r $script:TmpRoot -SkipEmbedding;Check (-not $r.passed) 'invalid TTS WAV structured fail'}finally{Remove-Item -LiteralPath $badWav -Force -ErrorAction SilentlyContinue}
    $validWav=Join-Path $script:TmpRoot ('valid-wav-'+[guid]::NewGuid().ToString('N')+'.wav');$wavCode='import sys,wave; w=wave.open(sys.argv[1],"wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\0\0"*64); w.close()';& python -c $wavCode $validWav|Out-Null
    try{$validBase=$ttsBase.PSObject.Copy();$validBase.output=[pscustomobject]@{path=$validWav;sha256='x'};$validCandidate=$validBase.PSObject.Copy();$r=New-CaseResult $validBase $validCandidate;Compare-TtsCase $null $validBase $validCandidate $r $script:TmpRoot;Check (-not $r.passed) 'embedding extraction structured fail'}finally{Remove-Item -LiteralPath $validWav -Force -ErrorAction SilentlyContinue}
    try{$null=Get-ApprovedOutputPath (Join-Path (Split-Path -Parent $script:RepoRoot) 'outside.json');$failures.Add('outside output rejected')}catch{}
    $external=Join-Path ([IO.Path]::GetTempPath()) ('aila-compare-escape-'+[guid]::NewGuid().ToString('N'));$junction=Join-Path $script:TmpRoot 'perf\oneapi-compare\escape-junction';New-Item -ItemType Directory -Path $external -Force|Out-Null
    try{if(Test-Path -LiteralPath $junction){Remove-Item -LiteralPath $junction -Force};New-Item -ItemType Junction -Path $junction -Target $external|Out-Null;try{$null=Get-ApprovedOutputPath (Join-Path $junction 'report.json');$failures.Add('junction output escape accepted')}catch{}}finally{if(Test-Path -LiteralPath $junction){Remove-Item -LiteralPath $junction -Force};Remove-Item -LiteralPath $external -Recurse -Force -ErrorAction SilentlyContinue}
    $stalePath=Join-Path $script:TmpRoot 'perf\oneapi-compare\selftest-stale.json';New-Item -ItemType Directory -Path (Split-Path -Parent $stalePath) -Force|Out-Null;Set-Content -LiteralPath $stalePath -Value '{"passed":true}' -Encoding utf8
    try{$null=Invoke-Comparison (Join-Path $script:TmpRoot 'missing-baseline') (Join-Path $script:TmpRoot 'missing-candidate') $stalePath -SkipEmbedding}catch{};Check (-not (Test-Path -LiteralPath $stalePath)) 'stale comparison removed before preflight failure'
    if($failures.Count){throw "compare_accuracy self-test failures: $($failures -join ', ')"}
    & python (Join-Path $script:RepoRoot 'tools/compare_audio.py') --self-test
    if($LASTEXITCODE -ne 0){throw 'compare_audio self-test failed'}
    Write-Host 'compare_accuracy self-test: PASS' -ForegroundColor Green
}

if($SelfTest){Invoke-SelfTest;exit 0}
if([string]::IsNullOrWhiteSpace($BaselineDir)-or[string]::IsNullOrWhiteSpace($CandidateDir)){throw 'BaselineDir and CandidateDir are required unless -SelfTest is used.'}
exit (Invoke-Comparison $BaselineDir $CandidateDir $OutputPath -SkipEmbedding:$SkipSpeakerEmbedding)
