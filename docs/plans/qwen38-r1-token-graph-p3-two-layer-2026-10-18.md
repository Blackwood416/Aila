# Qwen3.8 R1 token graph P3: two complete GDN layers (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2, real
`Qwen3.8-Flash-Next-BNB-NF4-v2` weights.

Probe: `tmp/q4e-token-graph-probe-20261017/probe_token_graph.cpp`

Receipts:

- `gpu-two-layer-p3.log`
- `gpu-two-layer-p3-final.log` (final abort-gate build)
- `gpu-two-layer-p3-postmerge.log` (full merged runtime)

## 0. Verdict

One Limited Graph now executes two complete consecutive GDN layers, L4 and
L5, with two host CPU-packet handoffs and one in-flight replay:

```text
L4 front/router -> route publish -> resident + real CPU packet
  -> ordered merge -> shared expert add -> hidden residual bridge
L5 front/router -> route publish -> resident + real CPU packet
  -> ordered merge -> shared expert add -> hidden residual bridge
```

The graph and a direct submission of the same kernels started from identical
cloned state are bitwise equal for every measured boundary.

## 1. Result

Run configuration:

```text
layers=4,5
tokens=4
top_k=10
hidden=2560
intermediate=640
shared_intermediate=640
resident_slots=2800
cpu_max_experts=16
host ring=zeMemAllocHost uncached bias
host arena=native large pages
```

| token | L4 CPU rows | L5 CPU rows | L4 handoff | L5 handoff | replay |
|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 10 | 4,994.2 us | 5,481.9 us | 48,557.3 us |
| 1 | 7 | 10 | 4,001.1 us | 5,727.3 us | 47,507.6 us |
| 2 | 7 | 9 | 3,983.7 us | 4,992.1 us | 46,341.2 us |
| 3 | 7 | 9 | 4,050.4 us | 4,973.5 us | 46,623.0 us |

Parity:

```text
hidden_l4=1 hidden_l5=1 state=1 history=1 heads=1
routes=1 selected=1 moe_h=1 resident_down=1 packet=1
routed_merge=1 shared=1 y_shared=1 cpu_rows=1
```

Health:

```text
timeouts=0 stale=0 mismatches=0
channel_mismatches=0 packet_seq_mismatches=0
packet_shape_errors=0 gate_aborts=0 host_route_errors=0
graph_cpu_rows=68 direct_cpu_rows=68
heads=2->0->1->2->0
all_ok=1 bit_exact=1 head_sequence=1 service_ok=1 ring_ok=1
```

The timings are correctness-run observations only. This run used the
generic CPU worker configuration plus system-scope gate checks and is not
comparable to the earlier tuned P2 latency receipt.

## 1.1 Abort negative control

The final executable also performs one deliberate stale-result replay after
the four valid tokens:

```text
gate_aborts=1 mismatches=1
front_committed=1 standalone_final_gate=1
session_poisoned=1
```

The attention residual is already committed by the MLP-HC portion of the
front before the host packet is available, so a rejected token cannot restore
the entire pre-token state. The gate prevents routed merge, shared add and the
final residual bridge from committing further writes. Production must poison
the session or restore a pre-token checkpoint after this class of failure.

## 2. Device GDN conv head

`DeltaNetState::conv_head_dev` is an optional three-int device slot:

```text
[0] head
[1] previous_head
[2] arrival_count
```

Both decode implementations read the device head before writing the ring and
advance it after the history write is visible. The grouped fast path uses an
acq-rel arrival counter so the last workgroup publishes `previous_head` and
the next head. Host-int callers keep the original behavior when the pointer is
null.

The P3 test starts both layers at head 2 and replayed four tokens:

```text
2 -> 0 -> 1 -> 2 -> 0
```

The state, conv history, output, previous-head slot and zero arrival counter
match the direct path bitwise.

## 3. Two-handoff protocol

Both layers share one uncached control/route ring and one CPU packet ring.
Sequence numbers advance once per layer:

```text
L4 seq = 2N + 1
L5 seq = 2N + 2
```

Only the first route publication clears `sticky_abort`. A failure in L4
therefore prevents L5 merge from becoming valid, while the host service loop
can still finish the second handoff and reach the final error receipt.

The per-layer captures are copied to independent device mirrors inside the
graph, so L5 cannot overwrite L4 verification data.

## 4. What is proven

- One executable can run two sequential whole-layer GDN segments with fixed
  pointers and one graph replay.
- Device `conv_head` advances correctly in a captured graph across the
  3-row ring wrap.
- Layer-specific slot-table offsets and a global descriptor table built from
  each resident slot's owning layer work for both layers.
- Real `RunGraphColdPacket` service can be called twice sequentially through
  one replay, including resident and CPU ranks in the same route.
- Shared expert, routed merge, shared add and `hc_inject_add` are all present
  before the next layer consumes `hidden4`.
- Sticky abort, packet sequence and packet-shape gates remain clean during
  fresh two-layer replays.

## 5. What is not proven

- No full-attention, QSA, PLE, final mixer or LM-head node is included.
- The probe uses synthetic cloned L4/L5 state rather than a production
  post-prefill session.
- The backend still does not own per-layer device head slots or synchronize
  them into snapshot/truncate/MTP state.
- Residency and the descriptor table are frozen for the replay.
- A post-GDN abort prevents the remaining routed/shared/final writes, but
  cannot undo recurrent/conv or attention-residual writes already performed
  by the graph. Production needs session poison or a pre-token state restore
  before another token is admitted.
- The graph is not a performance claim. A whole-token graph must still show
  an e2e gain after PLE/QSA support is present.

## 6. Next gate

The next minimal graph variant is one full-attention layer exercised above the
QSA enable threshold (`prompt_tokens >= 2052`). It must add fixed device scalar
state for position, QSA block counts and fixed-grid masked scan/selector work.
Dense and sparse QSA topologies remain separate executables; do not branch
between them on the host during replay.
