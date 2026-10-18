# Qwen3.8 R1 token graph P2 partial: real masked resident MoE (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2, Hyper-V enabled.
Probe: `tmp/q4e-token-graph-probe-20261017/probe_token_graph.cpp`

Receipt: `gpu-real-moe-p2-forced.log`

## 0. Verdict

The graph now captures a real layer-0 GDN front **and** a real masked resident
MoE dispatch:

```text
GDN front + router
  -> route publish to uncached host ring
  -> device resident classify
  -> fixed top-10 masked NF4 gate/up
  -> fixed top-10 masked NF4 down
  -> bounded device wait
  -> mapped-host result copy
  -> graph verification
```

The test route contains five GPU-resident experts and five cold experts. The
`selected[k] < 0` entries exit in the existing resident kernels; no dynamic
launch count is required.

## 1. Result

| item | result |
|---|---:|
| measured graph replays | 1,000 |
| front/state parity | bitwise |
| real router ids/weights parity | exact |
| resident selected ranks | 5 / 10 |
| graph vs direct MoE down | bitwise |
| timeouts / stale / mismatch | 0 / 0 / 0 |
| handoff median / p99 | 155.1 / 258.0 us |
| replay wall median / p99 | 653.4 / 927.5 us |

For comparison, the front-only P1 replay was 523.8 us and its zero-CPU
handoff was 28.8 us. Adding the real masked gate/up/down raises the replay to
653.4 us; the 155 us handoff includes completion of those resident kernels
before the graph's bounded wait node.

## 2. What is proven

- The adaptive cache's device slot table can produce a dynamic resident mask
  inside a captured graph.
- The graph can index a stable all-slot descriptor table and call the existing
  `moe_resident_gate_up/down` fixed-top-10 kernels.
- A mixed resident/cold mask preserves the original rank positions and the
  graph MoE down output matches the direct resident path bitwise.
- Removing the repeated graph-event status query from every payload poll keeps
  the handshake valid; the dynamic batch is released once after launch.

## 3. What is not proven

This is deliberately not a full decode layer yet:

- CPU classification still uses a forced test route, not the real router route
  for the MoE portion;
- no real `CpuWorker` job or host-arena cold computation;
- no 64-byte rank map plus 16-row CPU packet copied through a mapped-ring kernel;
- no shared-expert add or residual merge;
- no real-route layer-0 comparison, QSA layer, two-layer chain, or 48-layer graph;
- the descriptor table assumes frozen residency (no swap) and is only valid for
  layer 0's nested metadata.

The existing result ring is only 320 bytes of float data. The next probe must
carry a real CPU packet (roughly 50-100 KiB/layer) and must not use
`queue.memcpy` inside the graph.

## 4. Next gate

One full real-route layer must add:

1. device route ids from the actual router to both classification and host ring;
2. host CPU classification and a real `CpuWorker` job;
3. uncached host packet publication with `rank_to_row`, CPU rows and epoch;
4. graph mapped-copy kernel into a fixed device packet;
5. existing rank-ordered merge plus shared/residual add;
6. abort/timeout fail-closed behavior and cancellation drain.

Only after that layer matches the direct path bitwise should the probe add a QSA
layer and then two consecutive layers.

No production default changed.
