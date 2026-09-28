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

## H6b: completed

H6a commit: `ac5e7df`. H6b retains the default reusable-prefix contract:

- Non-final prefill slices and context-alignment replay set `need_logits=false`.
  Q4E skips the final hyper-connection mixer and LM head, returns an invalid
  tensor, and still advances recurrent, KV, QSA and PLE state. Only the final
  prefill slice returns sampling logits. Other backends retain their existing
  forward through the default options overload.
- Explicit anchor capture also handles a one-token slice; checkpoint cadence
  cannot accidentally suppress the requested boundary. Snapshot policy `none`
  still overrides capture requests.
- Engine initialization allocates one max-context device token staging tensor.
  Chunked and tokenwise prefill reuse it, draining on failure before reuse.
- Session creation allocates bounded snapshot slots, including GDN state,
  convolution history and PLE history. Capture publishes the length only after
  copies finish. Eviction, rollback and reset invalidate slots without freeing
  their backing vectors. The existing oldest-position eviction rule remains.
- Calculated payload for the real 48-layer model is 118,038,528 bytes per
  snapshot; the default four slots reserve 450.28125 MiB of ordinary host RAM.
  This moves the existing maximum payload to session creation, including when
  that session later receives a one-shot request. It is not pinned host USM.
  Token staging adds max_seq * 4 device bytes (2 KiB at max_seq=512).
- The allocation claim is limited to token input staging and snapshot storage;
  this change does not claim to remove every allocation in the entire backend.
  Runtime scratch growth and other existing forward paths are outside H6b.

### Verification

Hardware: Arc A770, driver `32.0.101.8860`; oneAPI 2026.1 Release build.
Full builds succeeded. Existing deprecation and floating-point option warnings
remain; no compiler flags were changed.

1. Q4E/chat/SafeTensors/build-identity CTest selection: **11/11 passed**, 63.15 s.
   Includes the new `AilaQwen4ExpGenerationTests` recording-backend test for
   26+4, single-token anchors, one-shot, capacity slicing, state-only replay,
   staging pointer/live-allocation checks and injected failure recovery.
   Module tests verify stable snapshot backing addresses across capture,
   eviction, truncate and reset, forced single-token capture, zero-budget
   behavior, recurrent restoration and QSA compressed-block restoration.
2. CPU/ABI selection: **18 passed, 1 skipped**, 131.33 s. The skipped test is
   `AilaInferenceCApiGpuTests`, which requires its separate model configuration.
3. Final generation/build-identity tests after test refinement: **2/2 passed**.
4. Explicit real-model acceptance:

   ```powershell
   $env:AILA_Q4E_EXPERT_READER='readfile'
   $env:AILA_PROFILE_Q4E_DECODE='1'
   ./build-refactor/AilaQwen4ExpGenerationTests.exe --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2
   ```

   **Passed**: all 248,320 final BF16 logit bit patterns match between the
   ordinary 26+4 path, skipped-head 26+4 path, and checkpoint-26 rollback/replay.
   Skipping the head leaves the previous logits buffer untouched and returns
   no valid logits. Two-turn generation reuses 38 tokens; edited assistant
   history restores the retained checkpoint at 30 and prefills 32 tokens.
   Both generation paths match fresh replay token IDs. Switching reusable ->
   none -> reusable clears/rebuilds state correctly.
5. Final 48-token story: **48/48 token IDs match the original baseline** in
   both policies. Reusable logs show count=26/logits=0, then count=4/logits=1;
   its final context-alignment token also has logits=0. One-shot remains a
   single count=30/logits=1 prefill.

| Single run | Prefill | GenerateMessages timer |
|---|---|---|
| Original baseline | 26 + 4 | 5836.22 ms |
| H6a | 30, policy none | 5470.39 ms |
| H6b | 26 + 4, reusable_prefix | 5689.77 ms |
| H6b | 30, policy none | 5454.87 ms |

Same model, story and CLI parameters as H6a. Story runs used ReadFile,
token-ID logging and Q4E profiling. These single-run timers are observations,
not a repeated-run speedup claim, and exclude load and final alignment/reset.

Final logs (local): `tmp/h6b-build-acceptance.log`, `tmp/h6b-ctest.log`,
`tmp/h6b-cpu-abi.log`, `tmp/h6b-real-acceptance.log`,
`tmp/h6b-story-reusable.log`, `tmp/h6b-story-none.log`.
Earlier `tmp/h6b-real.log` failed an incorrect test expectation: fully matched
history reused all 38 tokens and did not need rollback. The final fixture
explicitly edits history while checkpoint 30 is retained, and asserts that
checkpoint survives, preventing a reset-to-zero from passing as rollback.

H6a/b acceptance is complete. H6c (capturing intermediate state within one
full-length prefill), state sharding, raw-K ring and AOT remain conditional
future stages; none is implemented or claimed by this change.
