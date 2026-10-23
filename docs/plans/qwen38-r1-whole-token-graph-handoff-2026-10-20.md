# Qwen3.8 R1 Whole-Token Graph Handoff (2026-10-20)

## 1. Repository State

- Repository: `E:\RiderProjects\Aila`
- HEAD: `331264d`
- Worktree is intentionally dirty. Do not revert unrelated or existing changes.
- No GPU process is currently running.
- Last observed free physical memory: about `59.3 GiB`.
- Current build directory: `build-refactor`
- Whole-token graph remains opt-in:
  - `AILA_Q4E_TOKEN_GRAPH=0` by default.
  - Do not enable it by default; it still does not meet the `+5%` gate.

Important uncommitted files:

```text
M CMakeLists.txt
M src/core/Context.hpp
M src/models/Qwen4ExpBnb4Backend.cpp
M src/models/Qwen4ExpBnb4Backend.hpp
M src/models/Qwen4ExpExpertExecutor.cpp
M src/models/Qwen4ExpPle.cpp
M src/models/Qwen4ExpPle.hpp
M src/models/Qwen4ExpPrefill.cpp
M src/models/Qwen4ExpTokenGraphRuntime.hpp
M src/ops/AttentionOps.cpp
M src/ops/QsaIndexerOps.cpp
?? docs/plans/qwen38-r1-whole-token-callgraph-2026-10-21.md
?? docs/plans/qwen38-r1-whole-token-graph-implementation-2026-10-21.md
?? docs/plans/qwen38-r1-whole-token-graph-handoff-2026-10-20.md
?? src/models/Qwen4ExpTokenGraphKernels.hpp
?? src/models/Qwen4ExpTokenGraphService.hpp
?? src/models/Qwen4ExpWholeTokenGraph.cpp
```

Do not commit unless the user explicitly requests it.

## 2. Current Graph Boundary

The graph currently contains one executable for all 48 layers:

- Hidden repeat and hyper-connections.
- 36 GDN layers.
- 12 QSA layers.
- Router and resident/shared experts.
- 48 host cold-expert handoffs.
- Gated merge, shared add and final bridge.

Still outside the graph:

- Embedding.
- PLE preparation.
- LM head and sampling.
- Final mixer.

Each decode token is one graph replay. The graph is eligible only for
`count == 1` at long-context positions above the QSA threshold.

## 3. Structural Fixes Already Landed

### 3.1 QSA graph-control atomic removal: dominant win

The previous graph spent about `981 ms/token` in `route_wait`. The main cause
was not the CPU cold packet. QSA block scoring and tile attention re-read the
same device-scope atomic abort word from every work-item, producing roughly
millions of contended atomic loads per token.

Fixed paths:

- `src/ops/QsaIndexerOps.cpp:1365`
- `src/ops/AttentionOps.cpp:1412`
- `src/ops/AttentionOps.cpp:1541`

Each Work-Group now:

1. Lets only `local_id == 0` read/validate `GraphControl`.
2. Stores geometry/validity in local memory.
3. Uses a Work-Group barrier.
4. Lets all work-items consume the broadcast values.

This preserves fail-closed behavior and does not change arithmetic.

### 3.2 Whole-token service

- `src/models/Qwen4ExpTokenGraphService.hpp`
- `src/models/Qwen4ExpTokenGraphKernels.hpp`

Current behavior:

- `TokenGraphDeviceEpoch` is pure Device USM and uses device-scope
  acquire/release.
- Host-visible control words remain uncached system-scope communication.
- Input activation and route publication are combined in one 256-thread
  Work-Group.
- Packet payload copy currently uses 64-byte vector transactions.
- Gated consumers use one gate read per Work-Group, not one per work-item.
- Merge failure publication is limited to one Work-Group.

Important: a relaxed polling experiment without the system acquire fence
crashed with `0xC0000409`. It was reverted. Do not remove the acquire fence
from the CPU-doorbell polling path without a genuinely different mechanism
and a new negative-control probe.

### 3.3 Doorbell moved before GPU experts

- `src/models/Qwen4ExpWholeTokenGraph.cpp:303`

The graph now publishes input and route immediately after router top-k and
before resident classification/experts. The CPU cold service can overlap with
GPU resident/shared expert work instead of waiting for it to finish.

### 3.4 CPU cold packet direct execution

- `src/models/Qwen4ExpExpertExecutor.cpp:494`
- `src/models/Qwen4ExpExpertExecutor.cpp:1061`

`RunGraphColdPacket` now:

- Copies the uncached packet activation into cached pinned `x_host_`.
- Executes the existing TBB arena directly from the service thread.
- Avoids `submit()`/`finish()` condition-variable round trips.
- Writes the expert result directly into `packet->values`.

### 3.5 Snapshot no longer destroys graph

- `src/models/Qwen4ExpBnb4Backend.cpp:340`
- `src/models/Qwen4ExpBnb4Backend.cpp:3869`
- `src/models/Qwen4ExpWholeTokenGraph.cpp:769`

`CaptureSnapshot` and `SynchronizeGdnDeviceHeadsToHost` gained a
`preserve_token_graph` mode. Whole-token graph snapshots refresh host mirrors
without aborting the executable or incrementing the session-state generation.

This fixed the previous behavior where position `2112` triggered a second
capture and reset graph sequence to 1.

Do not use preserve mode for reset, truncate, snapshot restore, or any path
that rebinds/modifies graph-visible device state.

### 3.6 Duplicate head D2H and dead H2D removed

`EmbedTokens` now accepts an already-known host token ID. The whole-token graph
passes the token ID read back for PLE preparation and no longer performs a
second blocking D2H. The unused `graph.token_id` H2D was removed.

## 4. A/B Runner Fixes

File:

```text
tmp/q4e-token-graph-default-20261021/run_ab.ps1
```

Changes:

- `SYCL_CACHE_PERSISTENT=1`.
- Parses direct decode rows:
  `[Q4E AdaptiveWork] ... count=1 ... submitted_ms=...`
- Parses graph decode rows:
  `[Q4E TokenGraph] replay pos=... replay_ms=...`
- Parses:
  `[Q4E TokenGraphTiming] route_wait_ms, cold_ms, publish_ms, service_ms`
- Records all capture positions and service-done samples.
- Validates service-done progression per capture segment instead of incorrectly
  requiring the final value to equal `tokens * 48` across a recapture.
- Uses decode-only speedup for the screening speed gate.
- Keeps wall speedup in the receipt as secondary evidence.

## 5. Latest Real Receipt

Receipt:

```text
tmp/q4e-token-graph-default-20261021/results-token-auditfix-4-8/receipt.json
```

Latest production-profile 8-token result:

```text
captures=1
capture_pos=2108
service_done=48,96,144,192,240,288,336,384
route_wait_p50=98.182 ms/token
cold_p50=20.770 ms/token
replay_p50=120.868 ms/token
direct_decode_p50=59.707 ms/token
gate_failures=0
```

Graph token IDs:

```text
16,13,24624,279,1622,5435,6681,7810
```

This is about an `8x` reduction from the prior graph result:

```text
route_wait ~981 ms
replay     ~1009 ms
```

The graph is still about 2x slower than Direct decode, so it is not ready for
default promotion.

## 6. Packet Payload Is Not the Main Remaining Cost

Existing packet-copy probe:

```text
tmp/q4e-token-graph-probe-20261017/probe_packet_copy.exe
```

Measured with 5 active rows, 50 KiB uncached payload, 200 replays:

```text
best wg_stride Vec4 median: 0.477 ms/layer
```

Therefore the old `981 ms/token` bucket was not explained by packet payload
copy bandwidth. Even after the QSA fix, packet service is still a meaningful
but non-dominant part of the remaining gap.

## 7. Remaining Known Issues

### 7.1 Graph is still not default-ready

Steady graph replay is about `121 ms/token` versus Direct about `60 ms/token`.
The remaining gap is roughly:

- GPU graph body work.
- 48 synchronized layer boundaries.
- Host packet bridge.
- GDN history nodes.
- Final mixer outside the graph.
- Tail D2H/gate validation.

### 7.2 Cross-process correctness comparison is weak

Direct output still jitters across separate process runs. The latest graph run
is internally continuous and emits the canonical graph prefix, but serial
Direct-vs-Graph process comparison does not provide a reliable bitwise oracle.

For boundary correctness, use the same-process frozen-prefix/cloned-state
oracle from the existing graph probes. Top-1 agreement `>=95%` is the MTP
acceptance threshold; current graph decode work still needs a proper
same-process comparison.

### 7.3 Final mixer and tail D2H remain outside

The final mixer can potentially move inside the graph tail, but:

- Keep `submit_publish_done` after the mixer.
- Preserve graph event ordering.
- Keep gate/control validation fail-closed.
- Do not create different graph topology based on `need_logits` without adding
  that flag to the graph key.

### 7.4 GDN history fusion is disabled in graph mode

The graph currently emits separate device-head history commit nodes for GDN
layers. Direct path history fusion must be adapted to read the device ring head
before this can be removed safely.

### 7.5 Pending MLP residual remains high risk

Do not retry unconditional fused HC/pending residual. Layer 0 must be
materialized before Layer 1 PLE consume because PLE reads `hidden4`. A correct
implementation would keep Layer 0 explicit and defer only from Layer 1 onward.

### 7.6 Event sentinel comparisons

Unknown/empty `sycl::event` must not be tested with `event != sycl::event{}`.
Use an explicit boolean or `std::optional<sycl::event>`. The snapshot paths
touched in this work were changed to explicit booleans; audit any new event
guard similarly.

## 8. Recommended Next Order

1. Fuse packet-copy into ordered merge:
   - Move rank rows into `TokenGraphDeviceEpoch::reserved[]`.
   - Read active host rows directly in the merge kernel with uncached Vec4/Vec8.
   - Remove the separate packet-copy node and the 16-row device mirror.
   - Keep partition validation fail-closed before publishing the gate.

2. Restore GDN device-head history fusion:
   - Remove 36 standalone history commit nodes.
   - Read the previous ring row from `conv_head_dev`.
   - Require bitwise history-row validation across consecutive tokens.

3. Move final mixer into the graph tail and consolidate tail gate/control
   publication. This is lower risk than pending MLP residual.

4. Add a same-process frozen-prefix correctness oracle for 4/8/16 tokens.

5. Only after that consider pending MLP residual, with the Layer 0 PLE
   materialization boundary preserved.

Do not spend another round on packet vector width, zero-fill nodes, scalar
spin limits, or small launch-count tweaks until the items above are measured.

## 9. Required Validation Commands

Build:

```powershell
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor -Jobs 12
```

Graph lifecycle:

```powershell
.\build-refactor\AilaQwen4ExpModuleTests.exe --token-graph-runtime-only
.\build-refactor\AilaQwen4ExpModuleTests.exe --token-graph-key-only
```

QSA graph and composite correctness:

```powershell
.\build-refactor\AilaQwen4ExpKernelTests.exe --qsa-graph-only
.\build-refactor\AilaQwen4ExpKernelTests.exe --qsa-layer-graph-only
```

Real production-profile A/B:

```powershell
& .\tmp\q4e-token-graph-default-20261021\run_ab.ps1 `
  -GraphMode token `
  -MaxTokens 8 `
  -OutputDir tmp/q4e-token-graph-default-20261021/results-next
```

The runner itself acquires `Global\AilaGpuSingleton`, shuts down WSL, checks
for busy Aila processes, and enforces the free-memory gate.

CPU/ABI regression:

```powershell
ctest --test-dir build-refactor -L 'cpu|abi' --output-on-failure
```

The full CPU/ABI set passed 20/20 before the final QSA control change. The
final QSA change was validated with the two QSA GPU tests above and the full
8-token model A/B. Rerun the broad CTest set if making changes outside QSA.

## 10. Operating Discipline

- Use Windows PowerShell only. Do not invoke bash/WSL for repository work.
  `wsl.exe --shutdown` before GPU work is the only expected WSL command.
- Hold `Global\AilaGpuSingleton` for any GPU run.
- Require at least `54 GiB` free physical memory before GPU work.
- Only one GPU workload at a time.
- No fixed long sleeps. Poll process/result/log completion with bounded waits.
- Preserve existing `tmp/*-aborted` and experiment artifacts.
- Default promotion requires at least `+5%` end-to-end improvement.
- A single A/B round is acceptable for routine screening; use A/A or
  A/B/B/A only when the result is near the decision boundary.
- QSA tests must use a prompt at or above the QSA activation threshold
  (`>=2052` tokens in the current fixture). Do not use short prompts to test
  QSA kernels.
- Prefer measured probes over micro-tuning. The QSA control fix was found by
  following the actual per-token timing bucket, not by changing spin constants.

## 11. Important References

- `tmp/q4e-token-graph-bottleneck-audit-20261020/REPORT.md`
- `docs/plans/qwen38-r1-whole-token-graph-implementation-2026-10-21.md`
- `docs/plans/qwen38-r1-whole-token-callgraph-2026-10-21.md`
- `tmp/q4e-token-graph-default-20261021/results-token-auditfix-4-8/receipt.json`
- `tmp/q4e-token-graph-probe-20261017/probe_packet_copy.exe`
- `tmp/q4e-token-graph-probe-20261017/gpu-two-layer-p3-final.log`
- `tmp/q4e-strata-principle/REPORT.md`
- `tmp/XeStrata`

## 12. Bottom Line

The earlier `~981 ms/token` handoff path was dominated by QSA graph-control
atomic rereads, not by CPU expert math. Fixing that brought `route_wait` to
about `98 ms` and replay to about `121 ms`, a roughly 8x improvement.

The graph is now diagnosable and structurally connectable, but still not
faster than Direct. Continue with packet/merge fusion, GDN history fusion,
graph-tail mixer integration, and a same-process correctness oracle. Do not
promote the graph to default until a real `>=2052`-token A/B clears `+5%`.
