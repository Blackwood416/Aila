$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here "..\..")
$routeProfile = Join-Path $repo "tmp\q4e-sprint-r1-20261014\prefill\ttft14k-export.profile"
$decodeProfile = Join-Path $repo "perf\qwen4exp\profiles\qwen38-nf4-agent6k-v1.profile"
$jsonOut = Join-Path $here "out\current-profile-only.json"
$csvOut = Join-Path $here "out\current-profile-only.csv"

python (Join-Path $here "prompt_union_replay.py") `
  --route-profile $routeProfile `
  --decode-profile $decodeProfile `
  --slots 2792,2400 `
  --promotion-slots 512 `
  --json-out $jsonOut `
  --csv-out $csvOut
