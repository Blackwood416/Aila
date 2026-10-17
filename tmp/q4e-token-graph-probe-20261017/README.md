# Qwen4Exp token-graph P0 probe

Date: 2026-10-17
Status: **code prepared only; not built or executed**

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
- The host service ring is `Context::alloc_host` pinned USM.
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
