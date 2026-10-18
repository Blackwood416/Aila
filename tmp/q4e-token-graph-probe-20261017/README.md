# Qwen4Exp token-graph P0 probe

Date: 2026-10-17
Status: **protocol GO after switching the host ring to uncached Level Zero
memory; production integration not yet tested**

This directory tests the highest-leverage structural candidate from
`docs/plans/qwen38-r1-strata-kernel-transfer-2026-10-17.md`:

```text
captured graph
  route publish D->H (uncached L1+L3)
  bounded device wait on host flag
  CPU result written to mapped pinned host ring
  copy-from-mapped kernel (not queue.memcpy)
  bitwise verify
  completion publish D->H (uncached L1+L3)
```

The probe uses Aila's `Context`, in-order queue and Limited SYCL Graph. It
does not modify main-project source and does not use `queue.memcpy` inside the
captured graph.

## Files

- `probe_token_graph.cpp`: the complete P0 probe.
- `build.ps1`: compiles against `build-refactor/AilaLib.lib`; it does not run
  the probe.
- `run.ps1`: GPU-mutex, >=45 GiB free-memory and production L0 environment
  gate; it does not run until explicitly invoked.
- `RESULT_TEMPLATE.md`: receipt template for the eventual GPU run.

## Safety

- Device wait is bounded by `--spin-max` (default 20,000 reads).
- Host polling is bounded by `--host-timeout-ms` (default 2,000 ms).
- After `ext_oneapi_graph`, the probe issues one graph-event
  `command_execution_status` query to discharge the Level Zero dynamic batch;
  periodic `ext_oneapi_empty` queries are only a bounded fallback.
- There are no sleep calls. CPU workload uses `_mm_pause` busy-wait.
- The host service ring is `zeMemAllocHost` with
  `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`; ordinary `Context::alloc_host` is
  retained only as a fallback and is not valid evidence for the doorbell path.
- Every device access to route, result, flag and completion words uses an
  Intel `annotated_ptr` with uncached L1+L3 cache-control hints.
- The graph's mapped-host transfer is a kernel, never `queue.memcpy`.

## Build

```powershell
pwsh -NoProfile -File tmp/q4e-token-graph-probe-20261017/build.ps1
```

## Run later

Do not run until the GPU mutex and memory gate are clear:

```powershell
pwsh -NoProfile -File tmp/q4e-token-graph-probe-20261017/run.ps1 `
  -Replays 1000 -CpuUs '0,50,200,500' `
  -Log tmp/q4e-token-graph-probe-20261017/gpu-run.log
```

Each CPU arm runs:

1. `--warmup 10` replays.
2. 1,000 measured graph replays.
3. One deliberate stale-result replay to prove the stale detector is live.

## Output

The probe prints one line per arm:

```text
[TOKEN GRAPH ARM] cpu_us=... replays=... ok=...
  route_to_flag_us_median=... route_to_flag_us_p99=...
  handoff_us_median=... handoff_us_p99=...
```

`handoff_us` is measured from the host observing `route_seq` until it observes
the device completion word after copy+verify. It includes the requested CPU
work and the device resume/copy/verify path.

## Pass gates

P0 passes only if all of the following hold for every CPU arm:

- at least 1,000 measured replays;
- zero timeout, stale and mismatch deltas on fresh replays;
- stale negative-control replay increments both stale and mismatch counters;
- copied values are bit-identical to the host-written expected values;
- host observes route before completion and device observes the host flag;
- handoff median <= 20 us and p99 <= 100 us.

If the ring is not visible before graph completion, or the bounded device spin
expires, record **NO-GO** and do not implement the token graph.

## Actual result

### 2026-10-17: ordinary host USM (superseded)

Limited Graph capture succeeds, but the route payload is not visible to the
host until graph completion. Both 20,000 and 2,000,000 device spins timed out;
the host saw `route_seq` and then needed ~376-378 ms for the payload, by which
point the graph had exited. Event-status polling and host `_mm_clflush` did not
change the result. Receipts: `gpu-run3.log`, `gpu-run5.log`.

Do not use a host-serviced, whole-token graph on this path. A per-layer graph
that ends before the host reads the route is a different design and would need
its own probe.

### 2026-10-18: uncached Level Zero host ring

The failure above was caused by ordinary host USM. With
`zeMemAllocHost(..., ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED)` and the existing
uncached L1+L3 device accesses, all four CPU-service arms pass 1,000 replays:

| cpu service | handoff median | p99 | timeouts |
|---:|---:|---:|---:|
| 0 us | 28.9 us | 40.0 us | 0 |
| 50 us | 78.9 us | 167.0 us | 0 |
| 200 us | 228.9 us | 358.6 us | 0 |
| 500 us | 528.9 us | 667.0 us | 0 |

Fresh results are bit-exact and the stale negative control fires. Receipts:
`gpu-run-uncached-smoke.log`, `gpu-run-uncached-sweep.log`.

This opens the P1 real-layer capture gate. It does not yet prove an end-to-end
performance win.

## P1 real GDN front

`--real-front-model DIR` loads the model and captures layer 0's real GDN front
plus the route/result doorbell. Router ids and weights are written to device
memory inside the graph, then republished to the uncached host ring.

Receipt `gpu-real-front-p1-timed.log`:

```text
front_parity=1 route_parity=1
handoff_us_median=28.8
direct_front_us_median=538.2
graph_replay_us_median=523.8
replays=1000 timeouts=0
```

The one-layer graph is 2.7% faster than the direct front; the important result
is correctness of real state/router traffic through the mid-graph ring. P1
stops before MoE. The next gate is one full decode layer with device-count
resident dispatch plus the CPU packet, then two layers, then the whole token.

## P2 partial: real masked resident MoE

The probe now also captures device resident classification and the existing
fixed-top-10 `moe_resident_gate_up/down` kernels. With a test route containing
five resident and five cold experts:

```text
front_parity=1 route_parity=1 resident_count=5 moe_down_parity=1
replays=1000 replay_wall_us_median=653.4
handoff_us_median=155.1 timeouts=0
```

This proves the graph-safe masked dispatch. It does not yet contain the real
CPU worker, real packet, merge, shared expert or residual.

## P2 packet bridge

The probe now allocates a 163,968-byte uncached packet ring with a 16-rank map
and up to 16 x 2,560 FP32 rows. A graph copy kernel transfers it to a fixed
device packet, then the existing ordered merge consumes it.

```text
resident_count=5 cpu_rows=5
moe_down_parity=1 packet_merge_parity=1
replays=1000 replay_wall_us_median=712.6
handoff_us_median=190.3 timeouts=0
```

The CPU rows are still synthetic. Replacing them with a real `CpuWorker` job is
the next gate before a full-layer graph claim.

## Real CPU pipeline

`--real-front-model` now publishes the real router route and the device
activation row to the uncached ring, then the host calls
`Qwen4ExpExpertExecutor::RunGraphColdPacket` using the existing `CpuWorker` and
host arena. The graph copies the real packet into device memory and merges it.

Receipt `gpu-real-cpu-p2-classify.log`:

```text
resident_count=0 cpu_rows=10
front_parity=1 route_parity=1
moe_down_parity=1 packet_merge_parity=1
replays=1000
route_to_flag_us_median=348.4
handoff_us_median=409.0 p99=594.0
replay_wall_us_median=927.7
first_cold_handoff_ms=14.6
timeouts=0
```

The real route is all-cold for this fixture; mixed resident/CPU coverage is
retained by the earlier forced mask receipt.

## P3 two complete GDN layers

`--two-layer --layers 4,5` captures L4 and L5 in one executable. Both layers
use the real router, resident/CPU classification, `RunGraphColdPacket`, shared
expert, routed merge and final hidden residual bridge.

Receipt `gpu-two-layer-p3-final.log`:

```text
tokens=4
heads=2->0->1->2->0
graph_cpu_rows=68 direct_cpu_rows=68
hidden_l4=1 hidden_l5=1 state=1 history=1 routes=1
selected=1 moe_h=1 resident_down=1 packet=1
routed_merge=1 shared=1 y_shared=1
timeouts=0 stale=0 gate_aborts=0
verdict all_ok=1 bit_exact=1
abort_gate: gate_aborts=1 mismatches=1
            standalone_final_gate=1 session_poisoned=1
```

Run with `AILA_Q4E_COLD_MODE=2` and
`AILA_Q4E_CPU_MAX_EXPERTS=16`. This is a correctness gate, not a performance
comparison. A failed handoff poisons the session; pre-handoff attention
residual writes are not rolled back by the probe.

## P4 replayable state foundation

The merged runtime now has opt-in fixed-state primitives for a future
whole-token graph:

- `AILA_Q4E_GRAPH_GDN_STATE=1`: per-layer device GDN
  `[head, previous_head, arrivals]`, with explicit snapshot/truncate/MTP
  synchronization.
- PLE fixed slot 0 prepare/wait plus a graph view whose consume path has no
  host waits and advances a device 9-row conv head.
- QSA device position and fixed one-workgroup 4-token block compression.

Focused receipts:

```text
AilaQwen4ExpKernelTests --gdn-order-only
AilaQwen4ExpKernelTests --ple-device-head-only
AilaQwen4ExpKernelTests --qsa-graph-only
AilaQwen4ExpModuleTests --gdn-graph-state-only
AilaQwen4ExpModuleTests --ple-only
```

These P4 receipts cover replayable state only. P5/P6 below add fixed-grid QSA
scoring, selector and block-native attention, but still stop before a real L3
model layer.

## P5/P6 fixed-grid QSA attention

Focused graph tests:

```text
AilaQwen4ExpKernelTests --qsa-graph-only
AilaQwen4ExpKernelTests --attention-graph-only
AilaQwen4ExpKernelTests --qgkv-graph-only
AilaQwen4ExpKernelTests --qsa-layer-graph-only
AilaQwen4ExpModuleTests --token-graph-key-only
```

`--qsa-layer-graph-only` captures six QSA nodes in one executable and replays
`start_pos=2052..2055` with 513..514 compressed blocks. The graph-derived
selector output feeds block-native attention directly. All compared buffers
are bitwise equal to the direct path, and `abort=1` is a no-op.

This is still a synthetic composite layer. It does not include real L3 model
projections or the production graph lease.

## P7 real-weight L3 QSA layer

`probe_qsa_layer.cpp` loads the real L3 packed QKV, indexer and o-projection
weights from `Qwen3.8-Flash-Next-BNB-NF4-v2`. It captures the full sparse-QSA
attention sublayer in one executable and compares every intermediate against
the direct path for `start_pos=2052..2055`.

Build:

```powershell
pwsh -NoProfile -File tmp/q4e-token-graph-probe-20261017/build_qsa_layer.ps1
```

Run only under the global GPU mutex, after `wsl.exe --shutdown`, with at least
54 GiB free:

```powershell
tmp/q4e-token-graph-probe-20261017/probe_qsa_layer.exe `
  --model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 `
  --capacity 2060
```

Receipt `gpu-qsa-real-l3.log`:

```text
real=1 layer=3 qkv=2560x13312 index_qk=2560x640 graph_ready=1
pos=2052..2055 visible=2053..2056 blocks=513..514
active=512 bitwise=1 tiles=17,17,17,16
[QSA L3 RESULT] bit_exact=1
```

The initial hidden/cache data is deterministic seeded data rather than a real
2052-token prefill checkpoint. This proves the real-weight sublayer and graph
topology, not whole-token integration. See
`docs/plans/qwen38-r1-token-graph-p7-real-l3-2026-10-18.md`.
