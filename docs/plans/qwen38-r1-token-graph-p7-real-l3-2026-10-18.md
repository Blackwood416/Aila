# Qwen3.8 R1 token graph P7: real-weight L3 QSA layer (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2.

## 0. Verdict

The first real-weight L3 sparse-QSA attention sublayer now executes as one
captured graph and matches the direct path bitwise for four consecutive
positions:

```text
start_pos=2052..2055
visible length=2053..2056
compressed blocks=513..514
active blocks=512
attention tiles=17/17/17/16
all tracked intermediates bitwise
```

This clears the P6 real-layer gate. It is still a sublayer proof, not a
production token graph or an upstream-prefill checkpoint.

## 1. Captured body

The executable contains:

```text
real L3 packed QKV projection
  -> real L3 index_qk projection
  -> graph QGKV preparation
  -> QSA query/raw-K preparation
  -> one completed-block compression
  -> compressed-block scoring
  -> fixed-grid top-512 selector
  -> fixed-grid block-native attention
  -> real L3 o_projection
```

Both lanes use the real BF16/NV4 tensors from
`Qwen3.8-Flash-Next-BNB-NF4-v2`. The QKV and o projections use the production
packed decode graph path. The indexer uses its real BF16 `index_qk` weight and
the production single-row GEMV path.

The initial hidden row, K/V cache and indexer caches are deterministic seeded
data. This isolates graph replay from upstream prefill while retaining the real
layer weights, dimensions and QSA threshold. No HC, PLE, MoE, sampler or
upstream 2052-token prefill is included.

## 2. Receipt

Log: `tmp/q4e-token-graph-probe-20261017/gpu-qsa-real-l3.log`

```text
[QSA L3 WEIGHTS] real=1 layer=3 qkv=2560x13312 index_qk=2560x640 graph_ready=1
[QSA L3 STEP] pos=2052 visible=2053 blocks=513 active=512 abort=0 bitwise=1
[QSA L3 STEP] pos=2053 visible=2054 blocks=513 active=512 abort=0 bitwise=1
[QSA L3 STEP] pos=2054 visible=2055 blocks=513 active=512 abort=0 bitwise=1
[QSA L3 STEP] pos=2055 visible=2056 blocks=514 active=512 abort=0 bitwise=1
[QSA L3 RESULT] bit_exact=1 steps=4 start_pos=2052..2055
blocks=513..514 selector=complement_x3+cutoff_x1 tiles=17,17,17,16
nodes=qkv,index_qk,qgkv,qsa_prepare,compress,score,select,attention,o_proj
```

Run conditions:

```text
global GPU mutex
wsl.exe --shutdown
free physical memory >= 54 GiB (observed 56.57 GiB)
capacity=2060
wall time=28.2 s
```

## 3. Compared state

Each replay compares direct and graph lanes byte-for-byte for:

- fused QKV output;
- normalized Q, gate, K and V;
- full K/V cache rows;
- `index_qk` projection and prepared indexer Q;
- raw indexer K cache row;
- compressed indexer K block;
- block scores and active block IDs;
- attention partials and final attention output;
- o-projection output.

The selector output feeds attention directly inside the same executable. The
test therefore exercises the real sparse complement/cutoff dispatch rather
than a fabricated contiguous block list.

## 4. Limits

- The cache is seeded rather than produced by a real 2052-token prefill.
- Fail-closed is not yet a whole-layer transaction. QGKV and QSA preparation
  can write cache rows before a later stale-count check aborts the graph.
- Compression still uses its independent invalid flag; the unified
  `GraphControl.abort` repair is a separate source change.
- There is no production `ForwardTokenGraph`, event lease, one-in-flight
  state machine or host commit/poison lifecycle yet.
- The graph remains the sparse, exact head-dim 256 QSA topology.

## 5. Next gate

Implement the minimal production lifecycle around this proven body:

```text
prepare fixed runtime/PLE state before key construction
  -> one executable, one in-flight event
  -> same-thread CPU packet service
  -> explicit abort/packet gate
  -> host mirror commit only after success
  -> poison and drain after any failure
```

Do not capture `ForwardImpl`. Do not extend to multiple full-attention layers
until the single global compressed-block counter is replaced by per-layer
device counters.
