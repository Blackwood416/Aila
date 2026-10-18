# Qwen3.8 R1 structural receipts: copy-q event threading, post-router staging, token graph

Date: 2026-10-17

> 2026-10-18 correction: the whole-token-graph NO-GO below was caused by
> ordinary host USM. Re-running with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`
> passes 4x1,000 captured-graph handoffs with zero timeouts and bit-exact
> results. See
> [token-graph doorbell revalidation](qwen38-r1-token-graph-doorbell-revalidation-2026-10-18.md).
> The production integration remains unproven; the protocol gate is open.

Scope: canonical Qwen4Exp decode (`count == 1`, adaptive pipeline, int8 KV,
agent6k hotset) and the 14K prefill follow-up. All GPU arms used the paired
Worker/Server build recorded below. No default was changed without the
charter's >=5% end-to-end gate.

## 0. Verdict

- `AILA_Q4E_MOE_COPY_VIA_COPYQ=1` plus HC up-mix event threading is a real
  mechanism improvement, but is not an end-to-end win. Production A/B on the
  same build: `17.056 -> 16.787 tok/s` (`-1.58%`). Keep the feature opt-in.
- Moving the activation D2H after the mandatory router wait is a clear
  **NO-GO**: isolated A/B `16.827 -> 15.598 tok/s` (`-7.30%`). The pre-router
  overlap is necessary.
- F4 (`prev_cpu_h2d_ev_ -> CpuWorker::Job::transfer_reader`) is neutral within
  session noise and remains gated behind the copy-q switch.
- A whole-token Limited SYCL Graph with a host-visible route payload and a
  device-side doorbell is **NO-GO on the current Windows/L0 path**. Graph
  capture succeeds, but a payload published by the graph is not host-visible
  until the graph exits. The host sees `route_seq`, waits 376-378 ms for the
  payload, and the device has already exhausted even a 2,000,000-read spin.
- Prefill remains `53.2-54.1 s` at 14K. Its only credible next structural probe
  is the `4 x 4096` layer-major cold-union path, not more staging tuning.

## 1. Build and receipts

Source/candidate commits:

| commit | purpose |
|---|---|
| `53f1f8b` | HC up-mix event returned to order the x D2H |
| `93dc878` | source-only receipt for the event threading |
| `eb05f89` | F4: previous CPU-result H2D handed to the worker `transfer_reader` |
| `9d928ea` | `AILA_Q4E_STAGEIN_AFTER_ROUTE`, default off |
| `8f4655c` | token-graph P0 probe |

The production A/B binaries were built at `93dc878`
(`AilaWorker.exe` `2c35e9fb...`, `AilaServer.exe` `56a37531...`). The
post-router arm was built at `9d928ea`
(`AilaWorker.exe` `6d43ec96...`, `AilaServer.exe` `5e1163b4...`).
CPU/ABI tests on the first pair: 20/20 passed.

Receipts:

- `tmp/q4e-sprint-r1-20261014/ev-a1`
- `tmp/q4e-sprint-r1-20261014/ev-b1`
- `tmp/q4e-sprint-r1-20261014/evd-b2` and `evd-b2-worker.log`
- `tmp/q4e-sprint-r1-20261014/ar-a1`
- `tmp/q4e-sprint-r1-20261014/ar-b1`
- `tmp/q4e-token-graph-probe-20261017/gpu-run3.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-run5.log`

## 2. Copy-q + HC event

The x D2H producer is now the actual `hc_decode_up_mix` submission:

```text
RunHyperConnectionCombined
  -> hc_decode_up_mix returns sycl::event
  -> StageResidentInput(..., producer)
  -> copy_q.memcpy(..., depends_on(producer))
```

This removes the per-layer compute-queue `ext_oneapi_submit_barrier()`.
The fallback is unchanged when the copy-q switch is off or the HC branch
cannot return an event.

Mechanism receipt, diagnostic arm `evd-b2`, 1K prompt (above the QSA enable
threshold), 48 measured decode tokens:

| HC submission | before | event-threaded copy-q |
|---|---:|---:|
| norm mean | 20.47 us | 16.80 us |
| norm fallback mean | 26.64 us | 21.42 us |
| down mean | - | 24.13 us |
| up mean | - | 15.58 us |

The submit cost fell by 18-20%, so the event dependency is correct and active.
It did not reach the 8-12 us batched regime because router waits and the
remaining per-layer handoffs keep most submissions in the drained path.

Production A/B, 14K continue:

| arm | copy-q | after-route | decode tok/s |
|---|---:|---:|---:|
| `ev-a1` | 0 | 0 | **17.056** |
| `ev-b1` | 1 | 0 | **16.787** |

Delta: `-1.58%`. This is below the >=5% default gate and is not adopted. The
prior two B samples and one A sample hinted higher, but the adjacent paired
run is the valid decision receipt.

## 3. F4 and post-router staging

F4 removes this submit-thread wait:

```cpp
prev_cpu_h2d_ev_.wait_and_throw();
```

and passes the event through the already-existing
`CpuWorker::Job::transfer_reader`. It is gated by
`AILA_Q4E_MOE_COPY_VIA_COPYQ=1`; the default path is byte-for-byte unchanged.

The next candidate moved `StageResidentInput` after `wait_actual_route()`.
The idea was to avoid a cross-queue dependency inside the router submission
window. The isolated A/B used the same build and the same F4 code:

| arm | copy-q | after-route | decode tok/s |
|---|---:|---:|---:|
| `ar-a1` | 1 | 0 | **16.827** |
| `ar-b1` | 1 | 1 | **15.598** |

Delta: `-7.30%`. **NO-GO.** Issuing the activation copy before the router is
necessary to hide its handoff; moving it after the router creates a new CPU
input stall that is larger than any batch-coherency benefit.

F4 alone is effectively neutral: `ar-a1` 16.827 vs the earlier event-threaded
`ev-b1` 16.787. Keep F4 as an opt-in sub-change, not a default.

## 4. Whole-token graph P0

The P0 probe captures four ordered nodes in a Limited SYCL Graph:

1. route payload and sequence publish;
2. bounded device spin on a host flag;
3. copy-from-mapped host result using a kernel, not `queue.memcpy`;
4. bitwise result verification and completion publish.

Capture itself succeeds:

```text
[TOKEN GRAPH CAPS] limited=1 full=0 device=Intel(R) Arc(TM) A770 Graphics
[TOKEN GRAPH CAPTURE] status=ok nodes=4
```

The failure is not a timeout caused by a short spin bound. With both the
20,000-read and A770-sized 2,000,000-read bounds:

```text
timeout_delta=1
timeout_spins=2000000
route_us=377779
done_us=377949
```

The host observes `route_seq`, but the route id/weight payload remains the
pre-launch value while the graph is running. Polling with a per-iteration
event-status driver entry, volatile host reads, and explicit `_mm_clflush`
does not make it visible. The payload becomes complete only when the device
spin expires and the graph reaches its exit.

Therefore the current Windows L0/Limited-Graph path cannot provide the
mid-graph host route handoff required by this design. The safe conclusion is:

- do not implement a whole-token token graph with host-side route servicing;
- do not spend more time on event-query frequency, spin bounds, or host cache
  hints for this protocol;
- if graphs are revisited, use a per-layer graph boundary where the graph ends
  before the host reads its route, or wait for a driver/backend primitive that
  proves mid-graph payload visibility.

The corrected probe remains under `tmp/q4e-token-graph-probe-20261017/` with
payload-completeness polling and the A770 spin default.

## 5. Prefill status

Current 14K production A/A:

| run | TTFT |
|---|---:|
| b1 | 53.202 s |
| b2 | 54.057 s |

`router_wait` is `26.227/27.059 s` (1.88/1.94 ms per row). It is queue-depth
time for attention/HC/router, not just top-k. `experts_host` is
`26.261/26.303 s`; H2D volume is `174.2-174.6 GiB/prompt`, with an 18.5-18.6 s
PCIe floor. Making all H2D free gives only `34.6-35.5 s`, so supply-side
tuning alone has no margin at the 35 s target.

The next prefill probe is a true `4 x 4096` layer-major cold-union pass:

- target staged bytes <=60 GiB/prompt;
- gate TTFT <=48 s and raw token ids equal to chunk-major;
- reject if non-expert residual or bank wait still dominates.

The CPU-only promotion replay is committed in
`tmp/q4e-prefill-union-20261017/`, but it is blocked on a per-block route-ID
trace: the existing run-level hotset export cannot reconstruct the four-block
sequence without inventing data.

## 6. Defaults and next actions

No performance default changed:

- `AILA_Q4E_MOE_COPY_VIA_COPYQ` remains opt-in;
- `AILA_Q4E_STAGEIN_AFTER_ROUTE` remains opt-in and is NO-GO;
- F4 only acts inside the opt-in copy-q path;
- `AILA_Q4E_STAGEIN_HOIST` and all router waits are unchanged.

The next useful GPU slot is the layer-major cold-union prefill probe. Further
decode work should target a mechanism that removes the router host round trip
without requiring mid-graph host payload visibility; copy queue and post-router
staging are now closed by receipts.
