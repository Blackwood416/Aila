# Qwen3.8 R1 whole-token engineering audit (2026-10-20)

## Scope and evidence rules

This is a read-only source and receipt audit for the next host-ring/whole-token
step. No GPU was run and no implementation was changed.

- `[E]` means directly observed in the cited source, document, or receipt.
- `[I]` means an engineering inference from the observed code. Inferences are
  deliberately kept separate from the proof.
- XeStrata is used only as a reference for ordering and failure semantics. No
  XeStrata implementation, comment block, or near-line-for-line layout was
  copied. The relevant XeStrata files are mixed LGPL-3.0/MIT.

## 0. Verdict

`[E]` P3 already proves the hard control-plane idea: one captured executable
can contain multiple layer handoffs; the device waits for host service inside
the graph; the host returns a CPU packet; and the graph continues only after a
sequence and gate check. The final P3 receipt is
`all_ok=1 bit_exact=1 service_ok=1 ring_ok=1 abort_write_gate=1`
(`docs/plans/qwen38-r1-token-graph-p3-two-layer-2026-10-18.md:8-70`).

`[E]` Production L3 has a different shape. It is one synchronous layer graph:
stage a device control table, launch, wait for the graph event, copy the
control table back, then validate and commit the compressed-block count
(`src/models/Qwen4ExpPrefill.cpp:1001-1034`). It has no host route ring and no
CPU-packet service loop.

`[I]` The next useful change is therefore not a generic graph framework and not
a 48-layer capture first. It is one backend-owned host-service boundary with
P3's sequence/gate semantics, followed by the already-scoped real L3 receipt.
Only after that should the whole-token executable reuse the same boundary.

## 1. XeStrata execution, Doorbell, and CPU service

### 1.1 Capture-time order

`[E]` `session_capture_token` requires a mapped doorbell and mapped staging,
then records the whole token in one graph. For each layer the captured order is:

```text
pre[layer]
  -> optional expert-hit work
  -> doorbell_wait
  -> copy-from-mapped kernel for the CPU result
  -> optional hit add
  -> post[layer]
```

(`tmp/XeStrata/src/core/session.cpp:829-898`; the protocol summary is also in
`tmp/XeStrata/include/strata/core/session.hpp:384-402`.)

`[E]` The host is not notified by an event record in the middle of the graph.
The notification is a kernel that writes mapped host memory. XeStrata records
that an event record inside capture can be silently dropped, while a mid-graph
doorbell kernel is captured and the host sees the pinned handoff early in the
graph (`tmp/XeStrata/include/strata/core/layer.hpp:405-423`). The Aila probes
use the same rule: the graph copies from mapped memory with a kernel rather
than using a copy-engine node (`tmp/q4e-token-graph-probe-20261017/probe_token_graph.cpp:709-719`).

### 1.2 Replay-time order

`[E]` `session_run_token` performs this exact host/device order
(`tmp/XeStrata/src/core/session.cpp:900-963`):

1. Stage token and position data into mapped host buffers.
2. Reset the host-visible sequence and completion flag with `doorbell_reset`.
3. Launch the captured graph.
4. Call `stream_idle` before polling the mapped ring. This forces a driver
   entry; a pure host memory spin is not assumed to advance the GPU.
5. For each layer, poll until `h_seq` reaches `layer + 1`. The loop uses a
   bounded pause, periodically checks stream status, and has a 20-second
   deadline. It does not use a fixed sleep.
6. On the ring, run the CPU pool for the layer, write the CPU result, issue
   fences, and publish `h_flag = want` last.
7. After all layer handoffs, synchronize the graph stream.

`[E]` The device side is symmetric:

```text
device publishes x/ids/weights
  -> increments d_seq in memory
  -> shared/expert work continues
  -> doorbell_wait spins until d_flag == d_seq
  -> copy-from-mapped kernel reads the CPU result
```

(`tmp/XeStrata/src/core/layer.cpp:380-395`;
`tmp/XeStrata/src/kernels/xe/elementwise.cpp:145-164`.)

`[E]` The sequence is deliberately read and incremented in memory. A captured
literal would ring the same sequence on every replay. The same file states
that the publish/ring order is a stream-ordering guarantee, not a timing
assumption (`tmp/XeStrata/src/core/layer.cpp:389-395`).

### 1.3 Pipelined verification variant

`[E]` XeStrata's pipelined verifier separates the wait into a host service
phase: it launches the window, records a completion event, calls `stream_idle`,
services ring arrivals, and only then queries the completion event
(`tmp/XeStrata/src/core/verify.cpp:2665-2800`). `done()` does not report success
while the event is pending, and it drains the copy stream before declaring the
window complete (`tmp/XeStrata/src/core/verify.cpp:2789-2800`).

`[I]` This is the closest safe shape for Aila: not an event recorded mid-graph,
and not a synchronous wait inserted between the launch and the host service.

## 2. Aila P3: what the control plane already has and what it lacks

| Control-plane semantic | P3 probe | Aila production state |
|---|---|---|
| Stable graph identity | Probe-local ring and epoch. | `TokenGraphKey` already carries backend/session/runtime/capacity/topology generations and body/attention variant (`src/models/Qwen4ExpTokenGraph.hpp:28-49`). |
| One in-flight lifetime | Probe-local event. | Backend-wide `TokenGraphRuntime` owns one event and permits one in-flight launch (`src/models/Qwen4ExpTokenGraphRuntime.hpp:250-301`; `src/models/Qwen4ExpBnb4Backend.cpp:4172-4199`). |
| Host-visible route ring | 576-byte `HostRing`: 64-byte control, route IDs/weights, result values. | `TokenGraphHostRing` is only a POD declaration; the comment explicitly says wait/copy kernels are future work (`src/models/Qwen4ExpTokenGraphRuntime.hpp:70-143`). |
| Uncached control access | `zeMemAllocHost` with uncached bias for the ring, plus uncached load/store hints (`probe_token_graph.cpp:425-443,262-285`). | No production ring allocation or visibility preflight. |
| System-scope synchronization | Device publish is followed by a release fence; host packet publication uses fences and `_mm_sfence`; device wait validates sequence and packet shape. | Production L3 uses device-side `GraphControl` atomics, but not the P3 ring protocol. |
| CPU packet service | Real two-layer `service_two_layer_replay` runs CPU expert work twice in one graph. | No production whole-token service. |
| Sticky failure propagation | First route publication clears `sticky_abort`; later handoffs must not turn a failed epoch valid. | L3 has abort/start/blocks checks, but no multi-handoff sticky epoch. |
| Final sequence/done gate | `done_seq` is published by a device verification kernel; host waits for it and then waits for the graph event. | L3 waits for the graph event, copies `GraphControl` to host, checks `abort`, `start_pos`, and block count, then calls `Complete` (`Qwen4ExpPrefill.cpp:1011-1034`). |
| Failure cleanup | P3 proves a stale packet is gated and the session must be poisoned; front state may already be committed. | Production `AbortAndPoisonTokenGraph` exists and P10 drains before destroying the executable (`Qwen4ExpPrefill.cpp:1035-1039`; `Qwen4ExpBnb4Backend.cpp:4135-4157`). |
| Memory-domain split | The probe currently allocates both the 576-byte ring and the 169088-byte CPU packet with uncached bias (`probe_token_graph.cpp:1989-2003`). | Missing for production. The next implementation should not make the entire packet uncached by default. |
| Payload copy kernel | Current probe copy is a scalar 1D `parallel_for` over `kMaxPacketRows + rows * kHidden` (`probe_token_graph.cpp:721-742`). | Missing production implementation; do not carry the scalar copy shape into the hot path unchanged. |

`[E]` The P3 probe's graph launch and service loop already contain the key
control-plane order: graph launch, one event-status query, route poll, CPU
packet publication, done poll, then graph event wait
(`probe_token_graph.cpp:1050-1151`). The two-layer variant repeats that handoff
for two sequence numbers in one executable
(`probe_token_graph.cpp:1790-1884`).

`[I]` P3's synthetic `result_seq`/`result.values` channel is probe scaffolding,
not automatically part of production. The production minimal channel is route
metadata plus the CPU packet and the final gate. Keeping the synthetic result
ring in production would add synchronization without serving the model.

## 3. Safe landing for one event-status query, ring polling, and final commit

The following is the recommended boundary. The phase names are the important
part; the method names are intentionally local to this audit.

### 3.1 Before submission

1. `[I]` Require the runtime to be `Ready`, with one in-flight event only.
2. `[I]` Stage all replay-varying values in device-visible memory, including
   position, QSA block counts, abort flag, token/position values, and the
   current sequence. Do not bake them into captured kernel arguments.
3. `[I]` Reset the host ring's sequence and flag only after the previous event
   is known complete and the previous executable cannot read the ring.
4. `[I]` Check `Ready` and the expected sequence before calling
   `ext_oneapi_graph`. The current `Launch` happens after submission, so a
   rejected launch can otherwise leave a live local event outside the runtime
   state machine. If that happens, drain the local event before poisoning.

### 3.2 Submission and L0 release

5. `[E]` Create the graph event with `queue.ext_oneapi_graph(executable)`.
6. `[E]` Move a copy into `TokenGraphRuntime::Launch(sequence, event)`. `Launch`
   must be the state transition that makes the graph `InFlight`; it increments
   the expected sequence and clears the previous gate state
   (`Qwen4ExpTokenGraphRuntime.hpp:250-263`).
7. `[E]` Immediately perform one non-blocking event-status query through the
   runtime. `event_status()` is deliberately non-throwing and maps a query
   failure to `Failed`, never to `Pending`
   (`Qwen4ExpTokenGraphRuntime.hpp:265-285`). This is the L0 dynamic-batch
   release point. It is a query, not a wait.
8. `[I]` If the query returns `Failed` or `NoEvent`, transition to aborting,
   drain the local event, poison the session, and release the executable.

### 3.3 Ring service

9. `[E]` Poll the mapped route sequence with `_mm_pause` and a bounded host
   deadline. The probe's polling contract is exact sequence equality, with a
   queue-end fallback every 1024 polls
   (`probe_token_graph.cpp:921-953`).
10. `[E]` Read payload only after a release/acquire boundary and only after the
    sequence word says it is published. The probe also invalidates the host
    route lines before trusting the payload
    (`probe_token_graph.cpp:954-998`).
11. `[E]` Run the CPU expert service, publish packet payload and metadata, then
    publish `ready_seq` and `cpu_flag` last. The host must not publish a flag
    before the bytes it authorizes are visible
    (`probe_token_graph.cpp:1000-1035`).
12. `[I]` A layer with zero CPU rows must still publish a valid empty packet or
    consume a dedicated empty-plan flag. A device wait must not depend on the
    host deciding that a layer is "not worth servicing".
13. `[I]` Do not call event status in the inner poll loop. Poll the ring; use
    the bounded deadline and the final event query. If a liveness fallback is
    needed, make it a single terminal check, not a per-iteration driver call.

### 3.4 Final event done and gate commit

14. `[I]` After the last host flag is published, query the same in-flight event.
    If it is `Pending`, wait only now: all ring obligations have been
    discharged, so a wait cannot deadlock on an unserviced handoff.
15. `[E]` The current L3 path correctly waits
    before reading the control table
    (`Qwen4ExpPrefill.cpp:1011-1027`), but its `D2H` copy must not be replaced
    by an early host read of mutable ring data.
16. `[I]` Do not commit before `Done`. Read the final gate surfaces:
    `done_seq`, sticky abort, per-layer
    compressed counts, `start_pos`, and any GDN head/counter authority.
17. `[E]` Call `Complete(sequence, gate_valid)` only after event completion and
    gate validation. `Complete` enters `Aborting` on a stale sequence, an
    aborted runtime, or a rejected gate
    (`Qwen4ExpTokenGraphRuntime.hpp:287-301`).
18. `[E]` Commit host mirrors only after the gate is accepted. In P10 this is
    the `num_compressed_blocks` assignment after `Complete`
    (`Qwen4ExpPrefill.cpp:1028-1034`).
19. `[I]` On a host timeout, sequence mismatch, packet-shape failure, or gate
    failure, publish a terminal release value (or rely on a deliberately
    bounded device spin), then poison the session, drain the event, and only
    then release the executable. Do not silently continue direct execution
    after partial KV/indexer/recurrent writes.

The resulting ordered contract is:

```text
stage device data
  -> submit graph
  -> Runtime::Launch
  -> one event-status query (L0 release)
  -> host ring poll and CPU packet service
  -> final event done/status
  -> final gate read
  -> Runtime::Complete
  -> host mirror commit
  -> or poison + drain + release
```

The concrete production insertion point is the current L3 replay block:

`[E]` `ForwardQsaLayerGraph` creates the graph event, derives the sequence,
calls `Launch`, waits for the event, then copies `GraphControl` back
(`src/models/Qwen4ExpPrefill.cpp:1011-1022`).

`[I]` Keep event creation and `Launch` at that location. Replace the immediate
blocking wait/copy with the host-service boundary above. Move the final
event-done check and control read to the point after the last ring flag, and
move the host-mirror commit after the gate is accepted. `ReleaseQsaLayerGraph`
remains the only destruction path and must receive `false` until the event is
known complete.

## 4. Values that must live in device data, not capture-time scalars

`[E]` `GraphControl` states the rule directly: pointers are baked at capture
time, but values are updated before every replay
(`src/ops/GraphControl.hpp:5-16`). XeStrata independently records the same
failure mode: a captured ring literal would repeat forever
(`tmp/XeStrata/src/core/layer.cpp:389-395`).

### 4.1 Mandatory per-replay device data

| Value | Why it cannot be a captured scalar | Existing evidence |
|---|---|---|
| Graph sequence and CPU completion flag | Changes every handoff; the device waits on the in-memory pair. | P3 control ring and device gate (`Qwen4ExpTokenGraphRuntime.hpp:72-122`; `probe_token_graph.cpp:547-705`). |
| `start_pos` / visible length | Changes every decode token; QSA validation derives visible length from it. | `GraphControl.hpp:22-96`; `Qwen4ExpPrefill.cpp:1001-1008`. |
| Per-full-layer compressed block counts | Changes as blocks complete; a graph may advance exactly one pending block. | `GraphControl.hpp:14-28`; `Qwen4ExpPrefill.cpp:995-1027`. |
| Abort flag and sticky abort reason | Multiple workgroups can fail and later handoffs must stay invalid. | `GraphControl.hpp:31-45`; P3 first-handoff clearing rule (`docs/plans/qwen38-r1-token-graph-p3-two-layer-2026-10-18.md:120-132`). |
| Token ID, position, mRoPE positions | The input token and position differ per replay; positions drive RoPE and cache writes. | XeStrata stages token/position before launch (`session.cpp:900-963`); Aila's graph-control rule above. |
| PLE previous/history values if read by the graph | They are token-dependent host-visible state and can change without changing geometry. | `TokenGraphKey` has PLE workspace/view generations (`Qwen4ExpTokenGraph.hpp:42-45`); whole-token must not copy L3's zeroing of those generations. |
| GDN conv head, previous head, arrival count | State advances across replay and across the 3-row ring. | P3 receipt (`docs/plans/...p3...md:94-116`). |
| Route IDs and weights | Produced by the graph and consumed by the CPU; they are not capture constants. | P3 route publish and service loop (`probe_token_graph.cpp:486-527,1050-1151`). |
| CPU packet `ready_seq`, rows, rank map | Each layer's cold-row shape and sequence change. | P3 packet validation (`probe_token_graph.cpp:643-700`). |
| CPU packet payload and input activation | CPU results and the activation row are per-token/per-layer data. | P3 packet publication and copy (`probe_token_graph.cpp:1000-1020,1101-1122`). |
| Done sequence, counters, and gate state | Completion and failure are runtime facts; they cannot be inferred from a captured scalar. | P3 `done_seq`, counters and `gate_state` (`probe_token_graph.cpp:857-887`). |

### 4.2 Values that may remain capture-time constants

`[E]` Hidden size, top-k, head dimensions, block size, capacities, and fixed
grid sizes are topology or capacity facts, not per-replay values. The dense and
sparse QSA bodies are separate graph variants; the P3 plan explicitly says
not to branch between them on the host during replay
(`docs/plans/qwen38-r1-token-graph-p3-two-layer-2026-10-18.md:165-171`).

`[I]` A scalar such as `count == 1`, a selector mode, or a pointer base may be
captured only when it is part of the graph key or a separate executable
variant. If the same executable is intended to handle more than one value, the
value must be loaded from device data. Pointer identity is not runtime data:
the pointee contents still have to be updated or protected by a generation key.

`[E]` The L3 key deliberately zeroes PLE, residency, descriptor, and
session-state generations because that body does not read them
(`Qwen4ExpPrefill.cpp:786-800`).

`[I]` A whole-token body must not copy that exclusion unless it proves the
same non-dependency. Otherwise expert residency/descriptor changes, PLE view
changes, Reset/Truncate, and MTP restore must invalidate or update the graph.

## 5. Mechanisms not to introduce

`[E]` The following are already excluded by the current transfer record
(`docs/plans/qwen38-r1-xestrata-transfer-2026-10-20.md:88-106`):

- No copy-only queue for the graph handoff. The measured A770 result is
  negative, and the graph protocol uses a compute-queue kernel reading mapped
  memory. A copy-engine node can split the WDDM submission.
- No second compute queue.
- No MTP side stream.
- No `stream_fork`/`stream_join` parallel graph branches.
- No per-layer graph chain. The whole-token target is one executable with
  multiple internal handoffs, not 48 separately launched graphs.
- No mid-graph event record as the host notification. The device-to-host
  notification is a kernel writing mapped memory.
- No silent direct-path fallback after a graph has partially written KV,
  indexer, recurrent, or attention state. Poison and drain.
- No device-side cold lane rescue. The current all-CPU cold policy remains the
  reference; XeStrata also has no independent device cold lane.
- No commit-batch graph until Aila has a separate accepted-prefix/MTP commit
  stage.
- No LM-head/sampling node in the first whole-token executable. Keep the head
  outside the graph and keep replay-invariant graph capture limited to the
  transformer body.

`[I]` Two additional exclusions follow from the source audit:

- Do not make the entire 169 KB CPU packet allocation uncached. Keep the small
  control/route/done lines uncached and use a separate pinned bulk payload;
  otherwise the payload pays the same cache-bypass cost as the doorbell.
- Do not use the synthetic P3 `result_seq`/`result.values` channel as a
  production requirement. It is useful for probe verification, but it is not
  the model's data path.

## 6. Minimum patch order (five steps maximum)

### Step 1: Add one owned host-service ring and visibility preflight

Add a backend-owned ring object with:

- 64-byte uncached control/doorbell storage;
- small uncached route/done metadata;
- a separate pinned bulk CPU packet;
- stable device mirror pointers and an explicit lifetime longer than the
  executable;
- a startup visibility test: device-to-host sequence visibility and
  host-to-device packet bitwise visibility, with fail-closed rejection if
  either direction fails.

Acceptance: a failed visibility or allocation test leaves the graph unready;
no graph body is captured against uncertain mapped memory.

### Step 2: Replace the scalar CPU-packet copy with a gated vector copy

Keep the rank map scalar. Copy the packet payload with a fixed group/grid
stride and `uint4`/vec4 loads and stores. Both the rank-map path and payload
path must check the same valid gate before reading host memory. Preserve exact
rank order and byte values.

Acceptance: packet rows, rank mapping, and bytes match the P3 scalar path
bitwise; invalid gate performs no read and no write.

### Step 3: Put the async service boundary behind `TokenGraphRuntime`

Add one narrow backend method that:

1. preflights `Ready` and the expected sequence;
2. submits the executable;
3. calls `Launch` and immediately performs the one release status query;
4. polls the route ring and services CPU packets;
5. waits for the final event only after all handoffs are serviced;
6. validates the final gate, calls `Complete`, and commits host mirrors;
7. on any fault, poisons, drains, and releases.

Do not add another queue, a per-layer graph chain, or a generic graph service
framework. Use the existing one-in-flight runtime and `ReleaseQsaLayerGraph`
ownership rules.

### Step 4: Obtain the real production L3 receipt through the same boundary

Use real weights and a real tokenizer prefix at or above the QSA threshold:

```text
capacity: 4096 or 8192
start_pos: 2052..2055
expected complete blocks: 513..514
scope: direct-vs-graph bitwise rows plus per-layer control counters
```

Snapshot only the KV/indexer rows the body may write plus sentinels; do not
copy a 32K KV cache. Run one A/B receipt, not a long validation matrix. Keep
`AILA_Q4E_QSA_LAYER_GRAPH` opt-in until the existing end-to-end threshold is
met.

### Step 5: Assemble the whole-token executable only after Step 4 passes

Capture one 48-layer executable with fixed residency and descriptor views, all
handoffs inside the graph, and the LM head/sampling outside. Use the same
single orchestration thread for CPU packet service. Before capture, freeze and
key the session/runtime/residency/descriptor/PLE/topology identities. Do not
build per-layer executables while assembling the whole-token body.

## 7. Evidence summary

`[E]` Current receipts:

- `tmp/q4e-token-graph-probe-20261017/gpu-two-layer-p3-final.log`:
  bitwise two-layer P3 completion and stale-gate abort.
- `tmp/q4e-token-graph-probe-20261017/gpu-xestrata-transfer-suite.log`:
  selector, attention, QSA, runtime, and real-L3 suite all exited zero.
- `tmp/q4e-token-graph-probe-20261017/gpu-p11-postbuild-smoke.log`:
  selector exactness and token-graph runtime lifecycle passed after rebuild.

`[E]` The real L3 receipt is already bit-exact at `start_pos=2052..2055` with
blocks `513..514` (`docs/plans/qwen38-r1-xestrata-transfer-2026-10-20.md:67-73`).
It does not yet prove the asynchronous ring service, because production L3
still uses the synchronous wait/copy/commit path.

`[I]` The first measurable success criterion for the next patch is therefore
not "whole token is faster." It is: one in-flight executable reaches its ring,
the host services every entry without a graph wait in the inner loop, the
final event and gate agree, and a failed visibility/handoff test poisons
instead of falling through to direct execution.
