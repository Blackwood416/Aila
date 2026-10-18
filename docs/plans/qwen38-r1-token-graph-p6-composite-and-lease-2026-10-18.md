# Qwen3.8 R1 token graph P6: composite QSA layer and graph key (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2.

## 0. Verdict

The full-attention/QSA decode primitives now execute as one captured graph:

```text
graph QGKV prep
  -> QSA query/raw-K update
  -> completed-block compression
  -> compressed-block scoring
  -> fixed-grid top-512 selector
  -> fixed-grid block-native attention
```

The same change set adds the production graph-key and generation skeleton used
to invalidate captured graphs at state and residency boundaries.

The composite test uses synthetic buffers rather than real model projections.
It is not yet an L3 model-layer receipt and not a whole-token graph.

## 1. Composite receipt

Log: `tmp/q4e-token-graph-probe-20261017/gpu-qsa-layer-composite.log`

```text
start_pos=2052..2055
compressed blocks=513..514
visible length=2053..2056
active tiles=17/17/17/16
selector: complement x3, exact cutoff x1
six graph nodes in one executable
all compared state bitwise
abort=1 with invalid position leaves all outputs unchanged
```

The comparison covers:

- packed Q -> Q/gate;
- K/V scratch and full BF16 K/V cache;
- indexer query/raw K;
- compressed block 513;
- block scores and active block IDs;
- attention partials and final output.

The selector output is fed directly to attention; the test no longer relies on
a separately fabricated contiguous active-block list.

Full QSA suite receipt:

`tmp/q4e-token-graph-probe-20261017/gpu-qsa-suite-composite.log`

```text
QSA graph, attention graph, QGKV graph, composite layer graph
legacy 1428-case selector
prefill scoring/selection
block-native decode
all PASSED
```

## 2. Graph control fail-closed contract

All graph operators use the same 16-word table. GraphControl now provides:

- atomic sticky abort load/store;
- position/count consistency:

```text
compressed_blocks == floor((start_pos + 1) / 4)
```

- capture-time dtype and capacity checks for QSA scoring, selector and QSA
  preparation;
- QGKV position bounds and abort propagation;
- attention geometry validation that publishes abort instead of returning a
  stale-success state.

The composite abort test sets `abort=1` and an invalid position before replay;
none of the tracked outputs or caches change.

## 3. Production graph key skeleton

New POD header: `src/models/Qwen4ExpTokenGraph.hpp`

The key contains:

```text
backend/session identity
session generation
session state generation
runtime generation and capacity
residency generation
descriptor generation
topology generation and flags
body + attention topology
```

The lease currently contains only POD state: key, phase, sequence counters,
executable-ready and abort-requested flags. Executable/event/ring ownership is
intentionally not added until the first real model-layer capture.

Generation receipt:

`tmp/q4e-token-graph-probe-20261017/gpu-graph-key-generation.log`

```text
ModelSpec synthetic + real config passed
AdaptiveCache residency/descriptor generation passed
snapshot/truncate/reset/poison key invalidation passed
GDN authority transition test passed
```

Invalidation is wired at:

- runtime buffer and LayerMajor window growth;
- direct forward, Reset/Truncate and embedding/mRoPE mutation;
- MTP capture/restore;
- GDN host/device authority transitions;
- adaptive resident/staging/slot-table changes.

## 4. What is not proven

- No real 2052-token prefill checkpoint or real L3 projections are in the
  composite graph.
- The composite does not include attention hyper-connection, QSA index_qk
  projection or o_projection.
- There is no production `ForwardTokenGraph`, executable lease, event,
  in-flight state machine or CPU worker lifecycle yet.
- Per-full-layer compressed-block counters are not implemented; the current
  control slot is valid for one-layer proof only.
- Dense QSA topology and INT8/FP8 QGKV graph paths remain out of scope.
- `current_length_for_testing()` can still bypass generation invalidation.

## 5. Next gate

Capture one real L3 sparse-QSA attention sublayer from a 2052-token prefill
checkpoint:

```text
HC -> packed QKV -> graph QGKV prep -> index_qk projection
   -> QSA q/raw-K/compress/score/select
   -> attention -> o_proj
```

Replay four tokens over visible lengths 2053..2056 and compare every
intermediate bitwise. Do not start whole-token capture until this real model
layer and its session commit/poison path pass.
