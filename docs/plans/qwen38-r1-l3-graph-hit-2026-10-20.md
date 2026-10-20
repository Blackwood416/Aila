# Qwen4Exp L3 graph hit and host visibility receipts (2026-10-20)

Scope: A770 production Qwen4Exp path and token-graph probes. This note only
consolidates existing logs and receipts; it changes no source and runs no GPU.

## 0. Verdict

- The production Session now reaches the L3 QSA graph: the 2082-token run logged
  `eligible=1`, a `nodes=10` capture, and replay `gate=1`.
- The runner receipt records `output_exact=true`.
- The cached-pinned host-ring design is NO-GO for graph handoff: device-to-host
  payload visibility failed on the first round.
- The production 2052..2055 probe did hit and replay the graph, but its
  cross-process cache rows were not bitwise. A direct A/A rerun shows
  equivalent nondeterminism, so this is not evidence of an L3 graph error and
  not a row-level correctness pass.

## 1. Eligibility defects and fixes

Two root eligibility defects prevented the intended L3 graph hit.

### 1.1 Missing layer guard

Problem: the decode tile path called `ForwardQsaLayerGraph()` whenever the QSA
graph environment switch was enabled and the tile had one token. It did not
first require the current layer to be the single supported QSA graph layer.

Effect: graph eligibility and lifecycle code could be entered from an
unsupported attention layer instead of being restricted to
`kQsaGraphLayer`.

Fix: the dispatch condition now requires `L == kQsaGraphLayer` before calling
`ForwardQsaLayerGraph()`.

### 1.2 Q-only width used as the Q+gate / fused width

Problem: the graph geometry check treated `full_q_proj_dim_` as if it were
`full_q_dim_`. The buffer-shape check then required `buf_.q`, `buf_.z`, and
`buf_.attn_out` to have the Q+gate projection width.

These tensors contain Q-only data; they are not laid out at the Q+gate width.
The check therefore rejected every otherwise valid L3 capture through the
`buffer_shape` path.

Fix: the width domains are now explicit.

- Geometry requires the expected Q-only width and Q+gate width separately.
- `buf_.q`, `buf_.z`, and `buf_.attn_out` are checked against the Q-only width.
- `buf_.full_qkv` remains checked against the fused QKV width.
- The temporary `q_gate_view` continues to use the Q+gate width.

The fixes are in the current working-tree
[`Qwen4ExpPrefill.cpp`](../../src/models/Qwen4ExpPrefill.cpp). The production
receipt records commit `53cc5ba`; the eligibility edits were uncommitted at
consolidation time, so commit identity alone does not contain the patch.

## 2. Production Session hit at 2082 tokens

Source: [`graph-stderr.log`](../../tmp/q4e-prod-l3-session-20261020/graph-stderr.log)
and [`receipt.json`](../../tmp/q4e-prod-l3-session-20261020/receipt.json).

The request file is named `request-2052.json`, but the production log records
the actual tokenizer-prefix length:

```text
[GenerateMessages] Full prefill: 2082 tokens
```

The L3 graph then produced the required positive hit evidence:

```text
[Q4E QsaGraph] eligible=1 start_pos=2082 current=2082 blocks=520
[Q4E QsaGraph] capture_ready start_pos=2082 nodes=10
[Q4E QsaGraph] replay start_pos=2082 blocks=520 target=520 gate=1
```

The same run completed generation and recorded:

```text
[GenerateMessages] Prompt=2082 Generated=1 in 105831.43 ms
```

Runner receipt facts:

```text
commit=53cc5baedc03dd21c7ef750ebdfd383f05c912d0; free_gib=58.7
graph_only=true; arm=graph; graph_enabled=1; exit_code=0
wall_s=175.792; output_exact=true
```

This is a real production Session hit, not the standalone L3 probe. It proves
that the graph became eligible, captured ten nodes, replayed, passed the replay
gate, and produced the runner's exact output verdict. It does not replace
row-level cache and layer-boundary checks.

## 3. Packet-copy and active-row receipts

Common configuration from all three copy logs:

```text
device="Intel(R) Arc(TM) A770 Graphics"
top_k=10 warmup=20 replays=300 wg=256 grid=4096
control_bytes=64 control_alloc=uncached
payload_bytes_full=163840
host_mapping=usm_host rank_map=scalar
queue=in_order_profiling
```

All reported `wg_stride_vec4` arms were bitwise correct, including rank map,
active payload, and untouched tail.

| receipt | rows | payload_alloc | payload_bytes | scalar median | wg_stride_vec4 median | speedup |
|---|---:|---|---:|---:|---:|---:|
| `copy-pinned-r10.log` | 10 | pinned | 102400 | 9439.114 us | 947.500 us | 9.962x |
| `copy-uncached-r10.log` | 10 | uncached | 102400 | 9436.771 us | 947.448 us | 9.960x |
| `copy-pinned-r3.log` | 3 | pinned | 30720 | 2831.614 us | 290.521 us | 9.747x |

The logs show `payload_bytes_full=163840`, while active-row copies use `102400`
bytes at ten rows and `30720` bytes at three rows. All three
`wg_stride_vec4` runs beat the scalar and linear/grid-stride alternatives under
the same logged configuration.

## 4. Host-ring visibility NO-GO

Sources: [`host-ring.log`](../../tmp/q4e-token-graph-probe-20261017/visibility-20261020/host-ring.log)
and [`host-ring.err.log`](../../tmp/q4e-token-graph-probe-20261017/visibility-20261020/host-ring.err.log).

The setup separated the control block from the bulk payload:

```text
probe=start rounds=1000/1000 control_bytes=64
payload_slots=4 payload_bytes_per_slot=16384
backend=level_zero control=uncached_64B
payload=pinned_default payload_bytes=131072
overlap=0 handoff=sealed
capture=ok graph=1 kernel=1 host_service=in_place
```

The graph captured, but the first device-to-host payload check failed:

```text
visibility=NO-GO handoff=disabled
error="device->host payload mismatch at round 1 byte 0"
graph_done=1
snapshot="state=failed reason=host_payload_bitwise round=1
host_pub=1 device_done_seq=1 host_ack=1
device_visibility=1 host_visibility=0"
```

The first correction split D2H into its own uncached allocation and changed
the device result write to `uncached_store(L1+L3)`. That passed four rounds,
then failed when the cached-pinned H2D slot wrapped:

```text
reason=device_payload_bitwise round=5
host_pub=5 device_done_seq=4 host_ack=4
device_visibility=4 host_visibility=4
```

Changing the device-side H2D check to `uncached_load(L1+L3)` did not remove the
stale-slot failure. Although a one-shot cached-pinned H2D copy is bitwise, it
cannot be reused in the mid-graph ring after the four-slot wrap. The 64-byte
uncached control split remains valid, but the reusable bulk payload must stay
uncached. The structural gain should therefore come from `wg_stride_vec4` and
active-row copying, not from cached-pinned payload residency.

## 5. Production 2052..2055 probe and A/A noise

The production Session probe did execute the intended graph:

```text
pos=2052..2055 blocks=513,513,513,514
eligible=1 capture=10 nodes replay_seq=1..4 gate=1
```

Fresh direct and graph outputs nevertheless failed selected-row bitwise
comparison at K row 2052. A second direct-only run was therefore compared with
the first direct run. The A/A noise is of the same order as direct/graph:

| rows | A/A equal | direct/graph equal | A/A mean abs bit delta | d/g mean abs bit delta |
|---|---:|---:|---:|---:|
| K | 83 / 2048 | 103 / 2048 | 456.07 | 356.81 |
| V | 67 / 2048 | 86 / 2048 | 570.66 | 646.50 |
| indexer K | 26 / 512 | 47 / 512 | 268.96 | 205.97 |
| compressed K | 394 / 512 | 400 / 512 | 66.95 | 66.82 |

The production two-process path is therefore not a bitwise oracle. This
receipt neither proves nor disproves L3 graph boundary parity. A same-process
direct/graph oracle, or a frozen deterministic upstream state, is required for
that claim. Do not describe the 2082 output match as row-level bitwise proof.

Receipts:

- `tmp/q4e-token-graph-probe-20261017/prod-l3-4096-20261020/receipt.json`
- `tmp/q4e-token-graph-probe-20261017/prod-l3-4096-20261020/graph.stderr.log`
- `tmp/q4e-token-graph-probe-20261017/prod-l3-4096-aa2-20261020/receipt.json`

## 6. Next structural priorities

1. Build a same-process (or frozen deterministic) L3 boundary oracle. Do not
   repeat cross-process cache-row bitwise runs under the asynchronous
   production environment.
2. Integrate the validated `wg_stride_vec4` active-row packet copy as an opt-in
   experiment; keep the small control block separate from the bulk payload and
   preserve the scalar rank map.
3. Keep the reusable bulk payload uncached. Integrate the validated
   `wg_stride_vec4` active-row copy into the P3 packet path and test the full
   1000/1000 ring again; do not retry cached-pinned slot reuse without a new
   coherence mechanism.
4. Only after those gates pass, advance the production async service and
   whole-token graph. The bounded QSA prompt XMX candidate remains subordinate
   to this structural path; use [`qwen38-r1-qsa-prompt-xmx-bound-2026-10-20.md`](qwen38-r1-qsa-prompt-xmx-bound-2026-10-20.md)
   for its probe and admission bounds.
