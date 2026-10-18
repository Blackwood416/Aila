# Qwen3.8 R1: Strata/Graph 结构性迁移候选

Date: 2026-10-17
Scope: read-only source and receipt study. No source change, no GPU run.
Target: canonical Qwen4Exp decode, `count == 1`, adaptive pipeline.

> 2026-10-18 update: the follow-up P0 now passes when the host control ring is
> allocated with `zeMemAllocHost` + `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED`.
> See
> [token-graph doorbell revalidation](qwen38-r1-token-graph-doorbell-revalidation-2026-10-18.md).
> The analysis below remains the design basis; the "mid-graph visibility"
> risk in its NO-GO path has been removed.

## 0. Decision

**最高杠杆候选：把整个 decode token 录成一张 Limited SYCL Graph；路由后的
CPU 专家 handoff 改用“uncached pinned ring + 有界设备侧等待 + copy kernel”，
把每 token 约 984 次 kernel 提交和 48 次 router host wait 从主动提交/轮询路径
移入一次 graph replay。**

Verdict: **CONDITIONAL-GO to one probe only**. Do not integrate the token graph
until the device doorbell probe passes visibility, bounded-wait and bit-exactness
gates. This is not a request for another copy-site move or kernel micro-tuning.

The minimum successful result would be:

```text
one graph launch/token
+ bounded host/device handshake around the CPU expert pool
+ unchanged arithmetic and one final wait/token
```

The current copy queue remains useful inside this design, but it is no longer the
main mechanism: the mechanism is changing who owns the 48-layer submission loop.

## 1. Why this is the highest-leverage remaining structural change

Current best canonical receipt is `cq-b1`: **17.085 tok/s**, equivalent to
**58.53 ms/token**. The audit's latest per-token ledger is:

| item | ms/token | evidence |
|---|---:|---|
| kernel submission block | **18.7** | bottleneck audit section 3.2 |
| router other host work | 1.9 | same |
| MoE coordination | **11.5** | same |
| router wait | **11.4** | same |
| CPU join | **6.8** | same |
| sampling | 3.5 | same |

The flush inventory shows why the submission block does not behave like a normal
queue: the canonical layer has two host waits, activation D2H, CPU-result H2D,
and route-weight H2D; the CPU-result upload is one copy per contiguous rank run,
so the real operation count is roughly **400 flush-class operations/token**, not
4-6. All 286 measured HC submits/token sit in the drained regime:
20.3 us mean for norm, 24.8 us for the fallback.

The relevant upper bounds are therefore:

| “if X is free” | resulting token time | resulting rate |
|---|---:|---:|
| submission block -> 0 | 58.53 - 18.7 = **39.83 ms** | **25.1 tok/s** |
| submission + MoE coordination -> 0 | 58.53 - 30.2 = **28.33 ms** | **35.3 tok/s** |

The 20 tok/s gate is 50 ms/token. Even if only **45%** of the measured submission
block is removed, the target is reached. This makes the submission/handoff
structure a larger and more robust lever than any remaining single kernel.

The device-side wait is not assumed free. Strata's measured 5.2 us/handoff gives
48 x 5.2 us = **0.25 ms/token**; its WDDM graph-launch observation is
0.3-0.4 ms/replay. Even subtracting about 0.65 ms of replacement overhead, the
conservative submission-free bound remains about **25 tok/s**. Those are
Strata-platform numbers, not an Aila measurement; the Aila probe must measure
them again.

## 2. The proposed structure

For a fixed `count == 1` shape, one session owns:

1. A fixed-address pinned control ring per layer:
   `seq`, route ids, route weights, activation row, CPU-result rows, and a
   host-written completion `flag`.
2. A graph containing all 48 layer bodies in order.
3. A router/publish node that writes the route payload and ring with
   uncached L1+L3 stores.
4. A one-thread device wait node that spins, bounded, until the host has written
   the matching completion flag.
5. A copy kernel that moves the CPU result from mapped pinned memory to the
   device buffer. Do **not** use a `queue.memcpy` node here.
6. The resident-hit and merge nodes, followed by the next layer.

The host service loop does only:

```text
poll ring -> read route -> run existing CPU expert work -> write result -> write flag
```

It does not submit 20 kernels per layer, wait a SYCL event, or perform a
cross-queue copy. One final graph completion wait remains at the end of the
token.

The two graphs-per-layer shape used by Strata is the fallback if whole-token
capture is too large. It is not the preferred target: Strata records 96 graph
launches/token under WDDM and measured 48.6 ms/token for 12 ms of kernels plus
27 ms of pool. A whole-token graph is specifically what removes that per-layer
launch tax.

## 3. Evidence that it is feasible

### 3.1 Strata has the mechanism on the same toolchain family

`tmp/Strata/sycl/include/strata/core/session.hpp:392-415` defines a whole-token
graph with a device-side doorbell wait. The host polls rings, runs the CPU pool
and writes flags; the graph contains one wait kernel per layer. The measured
handoff constant is **5.2 us**.

`tmp/Strata/sycl/src/core/session.cpp:1214-1271` captures all 48 layer bodies
into one executable graph. It explicitly uses:

- `doorbell_wait(...)` as a captured device node;
- `copy_from_mapped(...)` as a **kernel**, not a copy-engine memcpy node;
- a fixed mapped-pinned staging buffer.

The comment at `session.cpp:1266` is directly relevant to Aila after the copyq
work: a copy-engine node split the WDDM submission, measured at 67 flushes/token.
A kernel copy avoided that split.

`tmp/Strata/sycl/include/strata/sycl_doorbell.hpp:46-88` documents the Intel
visibility result: system-scope atomics can still be served from cache on Arc;
the working implementation uses explicit uncached L1+L3 cache-control reads and
writes. A770 measurements are included there. Every device spin is bounded
(`:114-148`) so a missing flag does not wedge the GPU indefinitely.

This is materially stronger than the older 2026-09-30 Aila review note that
rejected “whole-window CUDA graph + doorbell/flag” as a generic CUDA port.
That review predates the current Intel cache-control implementation and the
current Aila flush evidence.

### 3.2 Aila already supports the graph primitive and fixes captured pointers

`tests/models/Qwen4ExpFrontGraphProbe.hpp` uses the same
`sycl::ext::oneapi::experimental::command_graph` path, records three real graph
fronts, replays them, and compares state, history, output, router ids and
weights with `same_bits`. The existing test reports
`state_router_output=bitwise`.

The graph experiment on this machine establishes the other half:

- full/updatable graph and mutable kernel arguments are unavailable;
- Limited SYCL Graph replay works;
- fixed-address host USM parameter tables are the viable dynamic-parameter
  mechanism;
- a single-node graph is slower, but a long chain of short kernels wins because
  host submission is amortized.

The graph count is one per decode-token topology, not one per kernel. With one
graph per token, the native-L0-list advantage is no longer decisive: only one
execute remains. Use Limited SYCL Graph first; native L0 lists are a later
replacement only if the one launch itself is measured above about 0.5 ms or
capture fails.

### 3.3 Aila already has the route payload, but not mid-graph visibility

The canonical router writes ids and weights into host USM:
`src/ops/MoeExpertOps.cpp:256-270`. Today the host waits for the kernel event
(`src/models/Qwen4ExpBnb4Backend.cpp:2857`) before reading them. The buffer is
allocated as host memory at `Qwen4ExpBnb4Backend.cpp:1146-1147`.

The existing plain stores are sufficient only when the kernel has completed.
The token graph requires the same values to become host-visible while the graph
is still running. That is a new, narrow requirement; it does not change the
router arithmetic.

## 4. Why this is not an already-refuted item

| prior item | why this is different |
|---|---|
| out-of-order queue | OOO was measured slower and does not attack the per-layer wait. This proposal keeps the existing in-order execution semantics and removes the submit loop. |
| `UR_L0_BATCH_SIZE` | Static batching held kernels until a full batch and regressed 61%. Graph capture replays the exact nodes without using the L0 batch as the synchronization mechanism. |
| Strata C1/C2 | C1/C2 were transfer/compute-overlap experiments. This candidate uses the already-created copy path only as part of a captured token; the primary change is one graph replay and a bounded device handshake. |
| copy queue alone | `AILA_Q4E_MOE_COPY_VIA_COPYQ` moves individual copies; it does not remove the 48 router waits or the host submission loop. This candidate can consume the copy-queue mechanism inside the graph. |
| single-kernel graph replay | Single-node graph replay was 32-49 us and not a win. The graph experiment explicitly shows the economics improve with a long chain; this proposal has approximately one token's worth of nodes per replay. |
| old “whole token CUDA graph” rejection | It predates the current Intel cache-control implementation, bounded device waits, the 400-flush inventory, and the current graph experiments. The only valid new discriminator is whether Aila's own Windows L0 path exposes the uncached ring before graph completion. |

The old note “events are enough” is not evidence for the current shape: the
current event/copy implementation itself measured every HC submit in the
drained regime. This proposal does not claim that a graph makes waits disappear;
it moves the required handoff and keeps it bounded.

## 5. Minimum probe before any implementation

No main-source edit until this probe passes.

### P0: device doorbell in one captured graph

Build a `tmp/` probe using Aila's current `Context`/Limited Graph path:

1. Allocate one small uncached host USM block.
2. Capture a graph containing:
   - a kernel that writes a sequence, 10 route ids and 10 weights to the block;
   - a device wait kernel on a host-written flag;
   - a copy kernel from mapped host result rows to a device buffer;
   - a final checksum kernel.
3. The host service thread polls the sequence without a fixed sleep, computes a
   deterministic synthetic result, writes the flag, and lets the graph finish.
4. Sweep the synthetic CPU work over 0, 50, 200 and 500 us.
5. Run at least 1,000 graph replays; inject no arbitrary long sleeps.

Success gates:

- 100% of replays complete, with no spin timeout and no device loss;
- host observes the ring before graph completion in at least the CPU-work arms;
- ring/result/checksum are bit-identical to the reference;
- measured handoff median is <= 20 us and no p99 exceeds 100 us;
- repeated replays do not read stale route or stale result data.

If the host cannot see the ring until graph completion, or the device cannot see
the host flag during the bounded spin, write **NO-GO** and return to per-layer
graph or ordinary event scheduling. Do not implement a token graph on faith.

### P1: one-layer real-model capture

Only after P0 passes, capture the existing GDN decode layer (including HC,
attention, router, CPU handoff, resident-hit merge and shared expert) behind an
opt-in flag. Run one prompt with one generated token and compare:

- greedy token sequence/content sha;
- route ids/weights;
- all intermediate tensors exposed by the existing tolerance probes;
- cancellation and session-destruction paths.

The first P1 graphs should be at most two layers, not 48. A two-layer graph is
enough to prove that the fixed pointers, route ring and flag survive capture;
the 48-layer graph is only an integration step.

### P2: end-to-end A/B

Once the two-layer graph is bit-exact, run a single A/B on the canonical decode
bundle. Default-on requires the existing **+5% e2e** gate and identical content
sha. The mechanism check is HC submit n/mean plus graph replay count; an arm
that merely hides work in a background thread does not pass.

## 6. Risks that could make the idea NO-GO

| risk | consequence | mitigation |
|---|---|---|
| host USM write is not visible until graph end | CPU expert pool cannot receive route mid-graph | P0; use uncached L1+L3 hints, then stop if still stale |
| host flag is not visible to device spin | graph waits to its bound | bounded spin; query driver status only on timeout; P0 |
| captured kernel arguments contain token-varying values | stale position/length | use fixed-address Host USM parameter tables or pinned staging; no mutable kernel args |
| 48-layer graph capture memory/time | capture may exceed practical budget | first two layers; measure graph bytes and finalize time; do not assume |
| CPU result copy breaks WDDM submission | graph stalls at a copy node | use a copy kernel, not `queue.memcpy`; retain copyq only for non-graph edges |
| cancellation leaves a spinning graph | device reset or hang | one bounded wait per layer, host always publishes a cancel flag, drain before free |
| single graph is not reentrant | overlapping requests corrupt ring | one in-flight graph per session/slot; lease and sequence checks |

## 7. Ranking

| candidate | upper bound / evidence | verdict |
|---|---|---|
| whole-token graph + device CPU handoff | submission-free **25.1 tok/s**; with coordination-free ceiling 35.3 tok/s; Strata has same-stack SYCL implementation | **highest; one P0 probe** |
| two graphs/layer, no device spin | removes per-kernel submits but leaves 48 router waits; Strata's 96 launches are expensive | fallback only |
| fused-HC/shared tuning | measured -19.8% and content change | NO-GO |
| ordinary kernel tuning | audit shows GPU kernels are not the dominant decode term | low leverage |

## 8. Bottom line

The structural problem is not “one more copy is on the wrong queue.” It is that
the model has a host-driven 48-layer loop whose synchronization cost is larger
than the current 20 tok/s margin. The only single change that attacks that
shape at scale is a captured token graph with a bounded device-side CPU
handoff. The evidence is strong enough to spend one small probe on it and not
strong enough to integrate it before that probe.
