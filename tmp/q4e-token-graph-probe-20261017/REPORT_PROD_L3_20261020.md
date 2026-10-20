# Production Session L3 Probe Report

Date: 2026-10-20  
Repository: `E:\RiderProjects\Aila`  
Revision: `53cc5ba`  
Scope: real model weights + real tokenizer + production `Qwen4ExpBnb4Backend::Session`; no GPU run in this task.

## Result

The production-Session probe is implemented and compiles:

- `tmp/q4e-token-graph-probe-20261017/probe_prod_l3.cpp`
- `tmp/q4e-token-graph-probe-20261017/build_prod_l3.ps1`
- produced `tmp/q4e-token-graph-probe-20261017/probe_prod_l3.exe` (10,410,496 bytes)
- build script completed successfully; `probe_prod_l3.exe --help` exits 0 under the oneAPI 2026.1 environment.
- a synthetic direct/graph file pair passed `--compare` (`bit_exact=1`,
  `graph_nodes=10`), validating the binary format and compare gates without
  loading a model.
- No model load, SYCL device context, GPU kernel, or test was executed.

This is a real executable probe, not a report-only patch point.

Artifact hashes (SHA-256):

```text
AA6D916AF818113C017F7D9A38FFBFC13EFD753A5E2D7476C3C1CC5088D76D42  probe_prod_l3.cpp
29090AAD4730549F76C6ABC94B9456338FB7EB852EE23FDA307FB2C24F7FD58F  build_prod_l3.ps1
1C687FD0E41608752556305830C52DA68F1EF8C09F8556A180C1295A15033E7A  probe_prod_l3.exe
```

## What The Existing Probe Does

`probe_qsa_layer.cpp` already uses real Qwen3.8 layer-3 weights through
`LoadModelWeightsLazy` (`tmp/q4e-token-graph-probe-20261017/probe_qsa_layer.cpp:132,147`),
but it then builds two private `Lane` objects with deterministic `x`, seeded
K/V/indexer buffers, and directly calls the QSA operations
(`probe_qsa_layer.cpp:231,585`).

It therefore proves the layer math and graph body, but does not exercise:

- `InferenceEngine::init` / real tokenizer loading
- `Qwen4ExpBnb4Backend::Session` allocation and ownership
- the production 48-layer prefix path and Session block counters
- the production `AILA_Q4E_QSA_LAYER_GRAPH` dispatch in `RunAttentionLayerTile`

The existing probe is a valid kernel/sublayer oracle, but not a production
Session receipt.

## New Probe Path

The new probe uses:

1. `InferenceEngine::init(model, capacity)` with real `tokenizer.json` and lazy
   Qwen4Exp weights (`src/engine/EngineLoad.cpp:3,274,334`).
2. A tokenizer-produced prefix of exactly 2052 tokens. The default corpus is
   repeated realistic text; `--prompt-file` can supply a production prefix.
3. The normal `InferenceEngine::forward_prompt_tokens` path for the 2052-token
   prefix, with `SnapshotPolicy::None` and logits disabled.
4. Four ordinary `Qwen4ExpBnb4Backend::Forward` calls at positions
   2052..2055 through the production `Session`.
5. Direct/graph separation by process:
   - direct lane: `AILA_Q4E_QSA_LAYER_GRAPH=0`
   - graph lane: `AILA_Q4E_QSA_LAYER_GRAPH=1`

Separate processes are required because the production graph gate is a
function-local static in `src/models/Qwen4ExpPrefill.cpp:27-30`; changing the
environment inside one process after the direct lane has run is not reliable.
The graph lane therefore exercises the real P10 dispatch at
`Qwen4ExpPrefill.cpp:210` and `ForwardQsaLayerGraph` at
`Qwen4ExpPrefill.cpp:803`.

Capacity is restricted to 4096 or 8192. The probe does not copy a full KV
cache.

## Compared Rows And Gates

For each position `2052..2055`, the direct and graph files contain:

- K cache: the two written KV-head rows, 2 x 256 BF16 values
- V cache: the two written KV-head rows, 2 x 256 BF16 values
- indexer raw K: the written 128-value row
- compressed indexer K: only the completed block row, written at position 2055
- graph control: all 16 words, completion sequence, phase, executable presence,
  session block counter, and expected block counter
- sentinel rows: position 2056 for K/V/indexer and compressed block 514

The compare mode requires exact row equality, exact session counters, no abort,
graph sequence `1..4`, graph phase `Ready`, ten captured graph nodes, and
unchanged sentinels. A graph-lane fallback to direct therefore fails the
receipt rather than silently passing.

The expected ten nodes are: `qkv`, `index_qk`, `qgkv`, QSA prepare, compress,
score, select, validator, attention, and `o_proj`.

## Build Receipt

Command:

```powershell
pwsh -NoProfile -File tmp/q4e-token-graph-probe-20261017/build_prod_l3.ps1
```

Result:

```text
built E:\RiderProjects\Aila\tmp\q4e-token-graph-probe-20261017\probe_prod_l3.exe (not run)
```

No source file was modified. The only changed tracked state remains the
pre-existing unrelated `docs/plans/qwen38-r1-whole-token-engineering-audit-2026-10-20.md`;
the new probe files are under ignored `tmp/`.

## Remaining Gaps

1. **No real-model receipt yet.** No GPU run was allowed in this task. The
   direct and graph binaries must still be generated and compared.
2. **Two model loads.** Direct and graph lanes are separate processes, so model
   load and prefix prefill are paid twice. This is a correctness probe, not a
   throughput measurement.
3. **Upstream determinism is assumed, not separately instrumented.** Both lanes
   use the same tokenizer prefix, environment, capacity, and model. The probe
   compares the L3 writes; it does not dump preceding-layer hidden states.
   A mismatch would identify L3 cache divergence but not the upstream producer.
4. **No full KV snapshot is taken by design.** This keeps the on-disk receipt
   small, but it cannot detect an out-of-bounds write outside the sentinel rows.
5. **The default prompt is tokenizer-real but synthetic.** A production
   `--prompt-file` should be used for the formal receipt.

## Next Run

Run only under `Global\AilaGpuSingleton`, after `wsl.exe --shutdown`, with no
`AilaWorker`/`AilaServer` process and at least 54 GiB free:

```powershell
$exe = 'tmp/q4e-token-graph-probe-20261017/probe_prod_l3.exe'
$model = 'C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2'
$dir = 'tmp/q4e-token-graph-probe-20261017/prod-l3-4096'

New-Item -ItemType Directory -Force -Path $dir | Out-Null
& $exe --lane direct --model $model --capacity 4096 `
    --output "$dir/direct.bin"
& $exe --lane graph --model $model --capacity 4096 `
    --output "$dir/graph.bin"
& $exe --compare "$dir/direct.bin" "$dir/graph.bin"
```

If and only if that returns `bit_exact=1`, repeat with capacity 8192. A
performance comparison is a separate A/B step and should not reuse this
correctness result as a timing receipt.
