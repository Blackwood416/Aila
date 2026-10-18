# Qwen3.8 R1 token graph P4: replayable state foundation (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2.

This gate adds replayable state and fixed-launch primitives needed by a
whole-token graph. It does not yet capture the QSA selector or attention layer.

## 0. Verdict

Three decode-state blockers now have opt-in graph-safe foundations:

- GDN has a production per-layer device conv-head table and explicit host
  authority transitions.
- PLE can prepare into a fixed staging slot and consume through a fixed device
  view with a device 9-row conv-ring head.
- QSA query preparation reads position from device memory, and completed
  4-token block compression runs from device position/count in a fixed
  one-workgroup launch.

All focused GPU gates pass. Production defaults remain unchanged.

## 1. Receipts

Log: `tmp/q4e-token-graph-probe-20261017/gpu-post-merge-gates-2.log`

```text
GDN device conv head:
  steps=5 final=1 prior=0
  state/history/output bitwise

PLE graph device conv head:
  replays=11
  head=5 through ring wrap
  output/state/head bitwise

QSA graph device position + block advance:
  pos=2051..2055
  complete blocks=513..514
  q/raw-K/compressed-K/count bitwise

Production GDN session state:
  allocation and authority transitions verified
  mixed authority fail-closed verified
  reset/checkpoint synchronization verified
```

The real-weight PLE module gate also passes:

```text
PLE ticket pipeline bit-exact rows=4,65,4,1
PLE lookahead exact outputs/state, no new USM
```

Log: `tmp/q4e-token-graph-probe-20261017/gpu-ple-module.log`

The existing QSA baseline also passes:

```text
selector cases=1428
513 complement selector
600 -> 512 + 3-tail sparse selector
batched scoring/selection
block-native attention at cached_len=2400
```

Log: `tmp/q4e-token-graph-probe-20261017/gpu-qsa-baseline.log`

The two-layer P3 graph was rerun against the merged AILA library:

```text
layers=4,5 tokens=4
heads=2->0->1->2->0
all parity fields=1
abort_write_gate=1
all_ok=1
```

Log: `tmp/q4e-token-graph-probe-20261017/gpu-two-layer-p3-postmerge.log`

## 2. GDN production state

`AILA_Q4E_GRAPH_GDN_STATE=1` allocates one three-int slot per layer:

```text
[head, previous_head, arrival_count]
```

Decode sets `DeltaNetState::conv_head_dev` only for one-token graph decode.
Host prefill, mixed authority, and deferred GDN history fail closed. Snapshot,
truncate, Reset, and MTP restore/capture synchronize the device slot and host
mirror at explicit boundaries. Steady decode does not D2H the head.

The gate is off by default and adds no allocation on the normal path.

## 3. PLE fixed staging and ring head

`Qwen4ExpPle::prepare_fixed_slot()` and `wait_prepared()` move host work before
graph launch. `create_graph_view()` exposes only fixed device pointers, and
`consume_graph()` performs no ticket wait, H2D wait, workspace growth, host
head update, or commit.

The graph variant reads `conv_head_dev[0]` in the dilated convolution and
advances the 9-row ring in a separate one-task kernel. The focused test replays
11 times from head 5 and compares output, full conv state, and head bitwise.

Host PLE commit remains outside the graph.

## 4. QSA device position and block advance

`qsa_prepare_q_and_cache_k_graph()` keeps the fixed 4-head, 128-thread launch
but reads the absolute position from device memory.

`qsa_compress_one_completed_block_graph()` uses one fixed 128-thread workgroup.
It reads position and `num_compressed_blocks_dev`, compresses only when exactly
one complete block became visible, and publishes the incremented count after a
global/local barrier. A larger gap or capacity overflow sets an optional
device invalid flag instead of silently dropping blocks.

The focused test covers:

```text
position:          2051, 2052, 2053, 2054, 2055
visible length:    2052, 2053, 2054, 2055, 2056
complete blocks:    513,  513,  513,  513,  514
compression:        no,   no,   no,   no,  yes
```

## 5. What is not proven

- QSA compressed-block scoring still uses a host-sized grid.
- The 513 complement and 514+ radix selectors still choose their branch on the
  host.
- Block-native attention still derives effective length, tail and tile count
  on the host.
- Dense and sparse QSA remain separate executable topologies.
- No production `ForwardTokenGraph` entry point or graph lease exists yet.
- PLE MTP integration still uses host PLE state unless explicitly synchronized.
- A failure after GDN/PLE/front work is detected but is not transactionally
  rolled back. Production must poison the session or restore a pre-token
  checkpoint.

CPU/ABI regression remains clean (`20/20`, one GPU test skipped by label).

## 6. Next gate

Implement fixed-grid graph variants for:

1. compressed-block scoring with a `max_blocks * 128` grid;
2. one 256-thread selector that handles both 513 and 514+ cases on device;
3. exact head-dim 256 block-native attention with a fixed maximum tile grid.

The first real test should capture one full-attention layer above the QSA
threshold and replay four tokens from `start_pos=2052..2055`, comparing q,
KV row, indexer K, compressed block 513, active blocks, attention output and
o-projection output bitwise.
