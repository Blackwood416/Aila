# Qwen3.8 R1: token-graph doorbell revalidation (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2, Hyper-V enabled.
No model was loaded. This re-tests the protocol rejected by
`qwen38-r1-copyq-graph-falsification-2026-10-17.md`.

Probe: `tmp/q4e-token-graph-probe-20261017/probe_token_graph.cpp`

Receipts:

- Previous ordinary-host-USM failure: `gpu-run5.log`
- Uncached L0 host ring smoke: `gpu-run-uncached-smoke.log`
- Uncached L0 host ring sweep: `gpu-run-uncached-sweep.log`

## 0. Verdict

The previous whole-token-graph NO-GO was caused by the **host allocation**, not
by a fundamental Windows/Level-Zero limitation.

The probe now passes with:

```text
host ring: zeMemAllocHost(..., ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED)
device accesses: annotated uncached L1+L3
graph: route publish -> bounded device wait -> mapped copy kernel -> verify
```

Four CPU service arms, 1,000 measured graph replays each:

| synthetic CPU service | handoff median | handoff p99 | timeouts | bit-exact |
|---:|---:|---:|---:|---:|
| 0 us | 28.9 us | 40.0 us | 0 | yes |
| 50 us | 78.9 us | 167.0 us | 0 | yes |
| 200 us | 228.9 us | 358.6 us | 0 | yes |
| 500 us | 528.9 us | 667.0 us | 0 | yes |

The stale-result negative control fired in every arm. The host observed the
route while the graph was still running, the device observed the host flag,
and the copied result was bit-identical.

## 1. What changed

The failed probe used `Context::alloc_host`, i.e. ordinary pinned host USM. Its
device stores already used uncached L1+L3 hints, but the host mapping itself
remained cacheable.

Strata's working A770 path allocates polled host memory with:

```cpp
ze_host_mem_alloc_desc_t desc{
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
zeMemAllocHost(context, &desc, bytes, alignment, &ptr);
```

The probe now uses that allocation for the control/route/result ring, keeps the
same uncached annotated device accesses, and links `ze_loader.lib`.

## 2. Probe result

Smoke arm (`cpu_us=0`, 1,001 total replays including warmup):

```text
[TOKEN GRAPH RING] allocation=zeMemAllocHost_uncached_bias
launch_to_route_us_median=77.5
route_to_flag_us_median=2.7
handoff_us_median=29.0
replay_wall_us_median=129.0
timeouts=0
verdict all_ok=1 bit_exact=1
```

The 29 us handoff includes launch visibility, the device wake-up, mapped copy,
verification, and completion publication. This is the same order as Strata's
reported handoff constant, not the previous 376 ms graph-exit delay.

## 3. Consequence

This removes the protocol blocker for a host-serviced device-resident graph. It
does **not** yet prove an end-to-end win:

- the graph must still contain enough real nodes to amortize replay;
- the real CPU expert pool must run at the existing roughly 200 us/layer cost;
- topology, fixed pointers, session state, cancellation and bounded-spin
  recovery must survive real capture;
- the current `Limited Graph` device exposes no mutable kernel arguments, so a
  fixed-address parameter table remains required.

The next step is the existing P1: capture one or two real decode layers and
verify route ids/weights, intermediate state, greedy token ids and cancellation
before considering a whole-token graph.

## 4. Decision ledger

| item | previous verdict | revised verdict |
|---|---|---|
| mid-graph host/device handshake | NO-GO | **GO on uncached L0 host ring** |
| ordinary `malloc_host` ring | failed | do not use for doorbell traffic |
| whole-token graph | blocked before P1 | **P1 protocol gate now open** |
| end-to-end adoption | no claim | still unmeasured |

No production default changed. Aila still uses the ordinary event/copy path.
