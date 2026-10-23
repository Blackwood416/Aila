# Qwen3.8 R1 whole-token graph implementation status (2026-10-21)

## Status

Whole-token graph replay is implemented and runs on the real Qwen3.8 NF4
model, but it is still **opt-in**. It does not meet the +5% end-to-end
default-promotion gate, so `tools/server/start-qwen38.ps1` is unchanged.

Enable it for experiments with:

```powershell
$env:AILA_Q4E_TOKEN_GRAPH = '1'
$env:AILA_Q4E_TOKEN_GRAPH_TRACE = '1'
```

The production worker profile already supplies the required
`AILA_Q4E_CPU_MAX_EXPERTS=10` and `AILA_Q4E_COLD_MODE=2`.

## Implemented boundary

- One captured 48-layer executable per eligible long-context session.
- Embedding, PLE preparation, LM head and sampling remain outside replay.
- The graph contains hidden repeat, HC, 36 GDN layers, 12 QSA layers,
  router, resident/shared experts, 48 host packet handoffs and final bridge.
- Each decode token uses one graph launch. The graph trace reports one
  capture, one replay per decode token and `service_done=48 * tokens`.
- PLE and GDN use device-authoritative head state; host mirrors are restored
  only at explicit synchronization or teardown boundaries.

## Correctness finding

The first integrated replay changed the PLE graph view to:

```text
(hidden_states=hidden4, out=hidden4, residual=null)
```

The direct path uses:

```text
(hidden_states=hidden4, out=null, residual=hidden4)
```

Those are not equivalent: the graph path overwrote `hidden4` with the PLE
projection instead of adding the projection to the residual stack. This
caused the first token to look plausible and subsequent tokens to collapse to
EOS. The graph view now matches the direct residual semantics.

## Receipts

All receipts are under
`tmp/q4e-token-graph-default-20261021/`.

- `results-aa-deterministic/receipt.json`: direct/direct A/A, 4 tokens.
  Output was identical. Wall time was 169.866 s vs 169.369 s and generation
  time 106791.95 ms vs 105969.53 ms (1.008x generation-speed noise band).
- `results-token-plefix/receipt.json`: PLE-fixed whole-token graph A/B.
  The graph emitted the canonical A/A stream
  `[16, 13, 24624, 279]`. The direct arm jittered at one token to
  `[16, 13, 32451, 279]`; the same direct jitter was observed across
  otherwise identical runs. Graph generation time was 110277.16 ms vs
  107398.83 ms direct.
- `results-token-final-smoke/receipt.json`: current binary, 2 tokens,
  2108-token prompt. Output and stdout hashes are identical. Graph trace:
  `capture_ready=1`, `replay=2`, `service_done=96`, `gate_fail=0`.
  Graph generation time was 108301.36 ms vs 106319.59 ms direct.
- `results-token-frontfused/receipt.json`: router and shared-gate projection
  now use the production `bf16_moe_front_fused` kernel in the graph body.
  The 4-token A/B was bit-identical. The production-profile A/B
  `results-token-production-frontfused/receipt.json` was also identical
  (4/4), but remained slower in wall time.
- `results-token-nodecut/receipt.json`: removed the redundant zero nodes and
  all intermediate done publications; the merge now validates the device
  resident/cold partition before reading either source. The 4-token
  production A/B remained bit-identical. Trace showed `nodes=352` and
  per-token replay times of 1304-1314 ms, with `route_wait` still about
  1267 ms and CPU cold work only 27-37 ms.
- `results-token-overlap/receipt.json`: tested early route publication,
  final mixer in graph and one final done publication. It did not improve
  wall or generation time, so the experiment was reverted.
- `results-token-fused-hc/receipt.json`: an experimental pending-residual
  fused HC path changed the first token and was rejected. The forced fusion
  and pending-residual graph rewrite were reverted.
- `results-token-auditfix-4-8/receipt.json`: current structural audit fix.
  Token-graph epoch accesses are device-scoped, the input/route doorbell is
  published before resident/shared experts, graph cold work executes directly
  in the service thread, and snapshot synchronization preserves the executable.
  The 8-token run captured once at `pos=2108` and emitted continuous
  `service_done=48,96,...,384`. Median values were:
  `route_wait=98.2 ms`, `cold=20.8 ms`, `replay=120.9 ms`; Direct decode
  median was `59.7 ms`. This is an approximately 8x reduction from the
  previous `route_wait=981 ms` / `replay=1009 ms`, but the graph remains about
  2x slower than Direct and is still not default-ready.

The dominant residual root cause was **not** packet payload bandwidth. The
existing packet-copy probe measured a 5-row, 50 KiB host-USM copy at a
`0.477 ms/layer` median (`tmp/q4e-token-graph-probe-20261017`). The large
bucket was QSA graph control: block scoring and tile attention re-read the
same device-scope atomic abort word in every work-item. Those reads now happen
once per work-group and are broadcast through local memory. The QSA graph and
layer-composite kernel tests remain bitwise/pass.

GPU lifecycle gates also pass:

```text
AilaQwen4ExpModuleTests --token-graph-runtime-only
AilaQwen4ExpModuleTests --token-graph-key-only
```

The final tree also completed the full 292-target CMake/Ninja build. The
`cpu|abi` CTest selection passed 20/20; `AilaInferenceCApiGpuTests` was the
expected skipped GPU-labeled case.

## Why it is not default

The current graph is still about 2x slower than Direct decode in the tested
4K/T1 envelope. The QSA control fix removed the dominant atomic storm, but
the remaining per-layer graph work, host handoff and packet bridge add roughly
60 ms/token over Direct. The observed graph and Direct outputs also still
drift across separate processes, so default promotion needs same-process
boundary validation rather than only a serial A/B screen.

Do not enable the graph by default until a real 2052+ prompt A/B reaches at
least `1.05x` wall speedup with no graph trace rejects or gate failures.

## Next candidates

1. Fuse packet-copy into ordered merge while carrying rank rows in the device
   epoch, eliminating one captured node and the full 16-row device mirror.
2. Restore GDN device-head history fusion so 36 standalone history kernels
   disappear while preserving device ring-head authority.
3. Inline the final mixer into the graph tail and remove the remaining
   blocking tail D2H bubble after consolidating gate/control publication.
4. Consider pending MLP residual only after the safe GDN/mixer work, keeping
   Layer 0 materialized before PLE as required by the Direct path.
