Set-StrictMode -Version Latest

$script:AilaOneApiInitialized = $false

function Get-AilaRepoRoot {
    $perfDir = Split-Path -Parent $PSScriptRoot
    return [System.IO.Path]::GetFullPath($perfDir)
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

    $repoFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    $candidateFull = [System.IO.Path]::GetFullPath($CandidatePath)

    if (-not $candidateFull.StartsWith($repoFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate on path outside repo: $candidateFull"
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

        [hashtable]$EnvOverrides = @{}
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
        $process.WaitForExit()
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
        outputLines  = $captured.ToArray()
        outputText   = $outputText
        logPath      = $LogPath
        durationMs   = [math]::Round($stopwatch.Elapsed.TotalMilliseconds, 3)
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
