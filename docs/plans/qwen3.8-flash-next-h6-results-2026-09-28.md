# Qwen4Exp H6a/b implementation and acceptance

Base: `d39c164`. Platform: Windows / Intel Arc A770 / oneAPI 2026.1.
All builds, tests and inference processes run serially.

## H6a: explicit request lifetime

- `GenerationConfig::snapshot_policy` defaults to `ReusablePrefix`.
- CLI: `--snapshot-policy none|reusable_prefix`. Chat JSON accepts the same
  strings in `snapshot_policy`; invalid values are rejected.
- `none` disables incremental reuse, stable-prefix anchor rendering/splitting,
  and Q4E snapshot capture on both prefill and decode. Completion drains and
  resets backend state, without replaying the final sampled token for reuse.
- `ForwardOptions` carries snapshot policy and the per-forward logits demand.
  Generation still requires logits for sampling; H6b uses the demand flag for
  intermediate prefill and context-alignment replay.
- No public C ABI structure or IPC wire schema changed. This implementation
  targets the Q4E legacy engine / CLI / in-process chat JSON path.

Validation: full `build.ps1 -BuildDir build-refactor -Jobs 8` succeeded; existing
Jinja fast-math infinity warnings remain. `AilaChatTests`,
`AilaQwen4ExpModuleTests`, and `AilaBuildIdentityCMakeTests` passed (3/3).

Real model: `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`, max-seq=512, greedy,
thinking-budget=0, max-tokens=48, `perf/qwen4exp/prompts/story.json`.
Both runs used `AILA_Q4E_EXPERT_READER=readfile`, `AILA_DEBUG_TOKEN_IDS=1`, and
`AILA_PROFILE_Q4E_DECODE=1`. The 48 generated token IDs matched exactly.

| Single run | Prefill | GenerateMessages timer |
|---|---|---|
| Original baseline | 26 + 4 | 5836.22 ms |
| H6a, policy none | 30 | 5470.39 ms |

Logs: `tmp/h6-baseline.log`, `tmp/h6a-oneshot.log`, `tmp/h6a-build.log`,
`tmp/h6a-ctest.log` (local, not versioned). These are single-run observations,
not a stable speedup claim. The existing GenerateMessages timer excludes
post-generation alignment/reset and process initialization.

## H6b

Pending: skip unused logits, preallocate token staging and snapshot slots,
then verify multi-turn reuse and rollback against a fresh replay.
