[CmdletBinding()]
param(
    [string]$Endpoint = 'http://127.0.0.1:8080/v1',
    [string]$ModelId = 'qwen4exp',
    [int]$ContextWindow = 262144,
    [int]$MaxTokens = 4096,
    [string]$Thinking = 'off',
    [string]$IsolatedRoot = '',
    [string]$WorkDir = '',
    [string]$Tools = 'read,bash,powershell,edit,write,grep,find,ls',
    [string]$Prompt = '',
    [switch]$Print,
    [string]$Mode = 'text',
    [double]$Temperature = 0.0
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))

$rootDir = if ([string]::IsNullOrWhiteSpace($IsolatedRoot)) {
    Join-Path $repoRoot 'tmp/pi-isolated'
} else {
    [IO.Path]::GetFullPath($IsolatedRoot)
}
$agentDir = Join-Path $rootDir 'agent'
$sessionDir = Join-Path $rootDir 'sessions'
$targetWorkDir = if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    Join-Path $rootDir 'work'
} else {
    [IO.Path]::GetFullPath($WorkDir)
}

New-Item -ItemType Directory -Force -Path $agentDir, $sessionDir, $targetWorkDir | Out-Null

$settings = @{
    defaultProvider = 'aila'
    defaultModel    = $ModelId
    defaultTools    = @($Tools.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    compaction      = @{
        reserveTokens    = 4096
        keepRecentTokens = 8192
        modelOverrides   = @{
            "aila/$ModelId" = @{
                reserveTokens    = 4096
                keepRecentTokens = 8192
            }
        }
    }
}
$models = @{
    providers = @{
        aila = @{
            baseUrl = $Endpoint.TrimEnd('/')
            api     = 'openai-completions'
            apiKey  = 'aila-isolated-local'
            models  = @(
                @{
                    id            = $ModelId
                    name          = "Qwen3.8-Flash-Next (Aila Isolated)"
                    reasoning     = $true
                    input         = @('text')
                    output        = @('text')
                    contextWindow = $ContextWindow
                    maxTokens     = $MaxTokens
                    samplingParams = @{
                        temperature = $Temperature
                    }
                    cost          = @{ input = 0; output = 0; cacheRead = 0; cacheWrite = 0 }
                }
            )
        }
    }
}

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $agentDir 'settings.json'), ($settings | ConvertTo-Json -Depth 10), $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $agentDir 'models.json'), ($models | ConvertTo-Json -Depth 10), $utf8NoBom)

# Locate the global pi cli.js bundle without inheriting any global config/auth/extensions.
$npmRoot = (npm root -g).Trim()
$cliJs = Join-Path $npmRoot '@earendil-works/pi-coding-agent/dist/bundle/cli.js'
if (-not (Test-Path -LiteralPath $cliJs)) {
    throw "Cannot find pi cli.js at $cliJs"
}

$savedEnv = @{}
$stripPrefixes = @(
    'PI_', 'OPENAI_', 'ANTHROPIC_', 'OPENROUTER_', 'GOOGLE_', 'GEMINI_',
    'MISTRAL_', 'GROQ_', 'XAI_', 'DEEPSEEK_', 'AWS_', 'AZURE_', 'OPENCODE_',
    'KIMI_', 'MOONSHOT_', 'MINIMAX_', 'ZAI_', 'QWEN_', 'XIAOMI_', 'NVIDIA_'
)
foreach ($entry in Get-ChildItem Env:) {
    $upper = $entry.Name.ToUpperInvariant()
    if ($stripPrefixes | Where-Object { $upper.StartsWith($_) }) {
        $savedEnv[$entry.Name] = $entry.Value
        Remove-Item "Env:$($entry.Name)" -ErrorAction SilentlyContinue
    }
}

try {
    $env:PI_CODING_AGENT_DIR = $agentDir
    $env:PI_CODING_AGENT_SESSION_DIR = $sessionDir
    $env:PI_OFFLINE = '1'
    $env:PI_SKIP_VERSION_CHECK = '1'
    $env:PI_TELEMETRY = '0'

    $piArgs = @(
        $cliJs,
        '--no-extensions',
        '--no-skills',
        '--no-prompt-templates',
        '--no-themes',
        '--no-context-files',
        '--no-mcp',
        '--provider', 'aila',
        '--model', $ModelId,
        '--thinking', $Thinking,
        '--tools', $Tools
    )
    if ($Mode -ne 'text') {
        $piArgs += @('--mode', $Mode)
    }
    if ($Print) {
        $piArgs += '--print'
    }
    if (-not [string]::IsNullOrWhiteSpace($Prompt)) {
        $piArgs += @('--', $Prompt)
    }

    Push-Location $targetWorkDir
    try {
        & node @piArgs
        exit $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    foreach ($entry in Get-ChildItem Env:) {
        if ($entry.Name.StartsWith('PI_')) {
            Remove-Item "Env:$($entry.Name)" -ErrorAction SilentlyContinue
        }
    }
    foreach ($kv in $savedEnv.GetEnumerator()) {
        Set-Item "Env:$($kv.Key)" $kv.Value
    }
}
