# Qwen3.8 R1 whole-token call-graph audit (2026-10-21)

Implementation status and current receipts:
`qwen38-r1-whole-token-graph-implementation-2026-10-21.md`.

## Scope

Read-only audit of `Qwen4ExpBnb4Backend::ForwardImpl` for the production
decode shape `count == 1`, `decode_verification == false`,
`need_logits == true`, and `prefill_remaining < 0`.

Evidence marks:

- `[E]`: directly observed in the cited source.
- `[I]`: engineering inference or proposed implementation boundary.

No GPU work was run and no source file was changed. Current source revision is
`331264d`.

## 0. Verdict

`[E]` The active count-1 path is not capture-safe as one function call.
`ForwardImpl` contains mandatory host waits, pinned-host route writes, a CPU
worker, H2D/D2H activation and result traffic, host-only cache bookkeeping,
and an optional nested L3 graph that waits immediately.

`[I]` The minimal whole-token boundary is:

```text
host embedding + PLE prepare
  -> one 48-layer executable
  -> 48 host ring services for router/CPU-cold packets
  -> final device gate + event completion
  -> host commit
  -> LM head and sampling outside the graph
```

`[I]` Start with a fixed post-QSA-sparse topology. Capturing across the
dense/sparse QSA branch or across checkpoint/MTP modes creates different
executables and should not be part of the first landing.

## 1. Entry and final boundaries

`[E]` `Model::ForwardImpl` copies the one-token ID H2D before calling the
backend, calls `Qwen4ExpBnb4Backend::Forward`, performs LM-head sampling in
`ReadOutput`, invokes MTP `AfterForward` when present, and then synchronizes
the context (`src/inference/Model.cpp:340-411`).

`[E]` Greedy/device sampling and optional captured logits use blocking D2H
copies in `ReadOutput` (`src/inference/Model.cpp:553-595`). The unconditional
final context synchronize is at `src/inference/Model.cpp:407-411`.

`[E]` `ForwardImpl` writes `session.currentLength_ = cached_len` on the host
immediately after submitting the current graph/direct work
(`src/models/Qwen4ExpBnb4Backend.cpp:3454-3457`). A graph path must move this
commit after the final graph gate.

## 2. Count-1 host/device call graph

| Stage | Device submissions | Host waits / CPU work / transfers | Mutable host state |
|---|---|---|---|
| Preamble | None until embedding | `ValidateSession`; optional graph abort/invalidate; GDN validation; `ensure_runtime_buffers` may grow buffers (`Backend.cpp:2458-2542`). | `cold_split_credit_`, graph epochs, runtime capacity if grown. |
| Embedding | Either `ops::embedding_lookup`, or a 4x repeat kernel | Default host-mmap table does blocking token-ID D2H, host gather, pinned H2D (`Backend.cpp:2240-2257`). Repeat vector kernel follows (`Backend.cpp:2600-2656`). | `embed_host_ids_`, `embed_host_rows_`. |
| PLE prepare | Worker-side projection preparation is not yet submitted; worker issues pinned H2D | `ple_.prepare` can wait a prior slot task and prior H2D before reusing staging (`Qwen4ExpPle.cpp:686-686,745-780`). | `ple_ticket_`, `ple_next_ticket_`, slot task IDs and staging slot. |
| PLE consume at L1 | Key projection, value projection, three grouped norms, gate/value, conv1d+SiLU+residual (`Qwen4ExpPle.cpp:816-908`) | Consume waits worker completion and PLE H2D (`Qwen4ExpPle.cpp:835-842`). | `ple_state_.prev_tokens` and ticket validity committed at `Qwen4ExpPle.cpp:1003-1009`. |
| Attention HC | Grouped norm, down/inject, up/mix (`Backend.cpp:2325-2446`) | No mandatory host wait on the normal fused path. | `pending_mlp` counters/state only. |
| Linear attention | `linear_all_proj`, one grouped GDN decode kernel, optional fused history commit, `linear_o_proj` (`Qwen4ExpPrefill.cpp:100-153`) | No mandatory host wait in decode. | `linear_conv_head`, authority/sync flags, `device_state_dirty` (`Qwen4ExpPrefill.cpp:202-213`). |
| Full attention, direct | QKV GEMV, indexer projection, QGKV prepare, QSA prepare/select, `attention_decode`, `o_proj` (`Qwen4ExpPrefill.cpp:215-332`) | No host wait except optional test/profile paths. | `cache.num_compressed_blocks` advances on completed blocks (`Qwen4ExpPrefill.cpp:267-280`). |
| Full attention, L3 graph | Nested executable: QKV, indexer, QGKV, QSA, selector, validator, attention, o_proj (`Qwen4ExpPrefill.cpp:913-994`) | `H2D` control, immediate graph wait, blocking `D2H` control (`Qwen4ExpPrefill.cpp:1038-1059`). | L3 compressed count committed after gate (`Qwen4ExpPrefill.cpp:1060-1077`). |
| MLP HC | Grouped norm plus fused down/up mix; alternate legacy path is multiple kernels (`Backend.cpp:2936-2955`) | No mandatory host wait. | `pending_mlp` ownership and debug counters. |
| Activation staging | Device-to-host copy of `buf_.x` through `StageResidentInput` | Calls `resident_input_event_.wait_and_throw()` before reuse, then submits D2H (`Qwen4ExpExpertExecutor.cpp:760-775`). | `resident_input_event_`, pinned `x_host_`. |
| Router | Router GEMV, `router_softmax_topk_device`, optional resident classify (`Backend.cpp:3018-3064`) | Kernel writes IDs/weights directly to pinned host; host calls `actual_route_event.wait_and_throw()` (`Backend.cpp:3042-3062,3094-3097`). | `router_host_ids_`, `router_host_weights_`, `resident_selected_host_`; `adaptive_cache_.ObserveRouting`. |
| Shared expert | Gate GEMV, fused gate/up/SwiGLU, down (`Backend.cpp:3174-3213`) | No mandatory wait; issued before the mandatory router wait to overlap. | None except optional log-once flags. |
| Routed experts | Resident gate/up/down kernels; optional GPU cold staging; CPU cold worker; H2D result packet; ranked merge (`Backend.cpp:1452-1793`) | `StartArenaCold` may wait prior result H2D; `JoinResidentCold` blocks in `cpu_worker_->finish()`; route weights may be H2D (`Backend.cpp:1690-1789`; `Qwen4ExpExpertExecutor.cpp:936-999,1044-1178`). | Adaptive route stats/residency; executor job/event fields; `cold_count_`; result-reader event. |
| Residual | `hc_inject_add`, or deferred materialization (`Backend.cpp:3329-3344`) | No mandatory wait. | `pending_mlp.pending`. |
| Final mixer | Final HC norm/down/up (`Backend.cpp:3377-3382`) | No mandatory wait. | `pending_mlp` cleared/materialized. |
| LM head | BF16 head, optional Q8 head (`Backend.cpp:3389-3420`) | No explicit wait; later sampling/logit readback follows in `Model`. | `lmHeadQ8ForwardCount_`, `lmHeadQ8RowCount_`, `lastTokenLogits_`. |
| Snapshot / return | Optional D2H of recurrent state | `CaptureSnapshot` does full `ctx.synchronize()`, head D2H, state D2H, then waits the last D2H (`Backend.cpp:3847-3899`). | Snapshot slots, device-dirty flags, current length. |

`[E]` `RunResidentDecode` is selected by the count-1 branch because
`use_adaptive_pipeline_` is true and the adaptive cache is allocated
(`Backend.cpp:2753-2759,3292-3308`; header default at
`Qwen4ExpBnb4Backend.hpp:696`).

`[E]` For count 1 it starts the compact CPU batch at
`Qwen4ExpBnb4Backend.cpp:1666-1691`, joins at `:1766-1769`, uploads the CPU
rank rows at `:1776-1789`, and only then returns the merge event.

## 3. Capture classification

### Direct capture after replacing scalar arguments with device control

- `Bnb4BitLinear::ForwardDecodeRows` for packed, non-LoRA projections
  (`src/ops/Bnb4BitLinear.cpp:2306-2385`).
- Linear-attention decode and graph-safe history commit
  (`Qwen4ExpPrefill.cpp:125-153`).
- Full-attention QKV, QGKV preparation, QSA preparation/selection,
  block-native attention, and `o_proj` when using graph/device-control variants
  (`Qwen4ExpPrefill.cpp:226-332,938-994`).
- Hyper-connection norm/down/up kernels (`Backend.cpp:2325-2446`).
- Router math, device classification, resident expert kernels, shared expert
  kernels, and ranked merge, provided route weights/IDs and packet addresses
  remain device-resident (`Backend.cpp:3018-3213`).
- PLE `consume_graph` only after its fixed staging slot and device head are
  prepared; host n-gram preparation stays outside (`Qwen4ExpPle.cpp:911-1001`).

### Must become device-control / host-ring service

- Router publication: replace pinned-host outputs and `wait_and_throw` with
  device-written route IDs/weights plus a host-visible sequence, as proven by
  the P3 route ring (`tmp/q4e-token-graph-probe-20261017/probe_token_graph.cpp:512-635`).
- Activation input: replace `StageResidentInput` D2H with a graph kernel that
  copies `buf_.x` to the mapped packet and publishes its sequence
  (`probe_token_graph.cpp:556-563,776-820`).
- CPU cold service: host calls the existing
  `RunGraphColdPacket`-style worker after route publication
  (`Qwen4ExpExpertExecutor.cpp:1002-1041`; probe service at
  `probe_token_graph.cpp:1142-1215`).
- CPU result: host fills a fixed mapped packet, publishes payload after
  metadata, and the graph uses active-row copy plus ordered merge
  (`probe_token_graph.cpp:776-870`).
- QSA compressed counts: keep all 12 full-layer counters in the 16-word
  `GraphControl` table, not host `cache.num_compressed_blocks`
  (`src/ops/GraphControl.hpp:7-38`).
- GDN head advancement: device-head authority must be enabled and persisted;
  capture-time host writes do not replay (`Backend.cpp:402-426,4122-4133`).

### Must stay outside the graph

- Host embedding lookup and its D2H/CPU-gather/H2D sequence
  (`Backend.cpp:2240-2257`).
- PLE n-gram lookup and worker preparation; only `consume_graph` is recorded.
- LM head and sampling (`Backend.cpp:3389-3428`; `Model.cpp:553-595`).
- MTP draft/verification and rollback.
- `ensure_runtime_buffers`, `TrimDecodeRuntimeBuffers`, snapshot capture,
  reset/truncate, and any allocation-capacity change.
- Logging, performance counters, debug dumps, and test aliases.

`[E]` `ForwardQsaLayerGraph` is itself an executable and has a mandatory wait
and control D2H, so it cannot be called as a node inside a whole-token
capture. Its constituent graph variants must be inlined into the whole-token
body (`Qwen4ExpPrefill.cpp:904-1008,1047-1077`).

## 4. Required boundaries

- **Embedding:** outside by default. The resident-table alternative is one
  optional later extension; it is not needed for the first product graph.
- **LM head:** outside. The graph ends after the final mixer and leaves
  `buf_.x` valid for the existing head.
- **Sampling:** outside, in `Model::ReadOutput`.
- **MTP:** outside. `ForwardWindow` invokes a separate count-2 target forward
  with `decode_verification=true` and its own rollback/commit state
  (`Qwen4ExpMtpRuntime.cpp:132-188`). Do not reuse the count-1 executable.
- **Checkpoint:** outside. It is a full synchronize plus state D2H
  (`Backend.cpp:3847-3899`). If a decode token lands on a checkpoint
  boundary, run the graph to completion, validate the gate, then snapshot.
- **PLE prepare:** outside. The graph consumes only fixed device staging;
  host n-gram preparation must finish before launch.

## 5. Minimal refactor signature and order

Add one dispatch method and one graph-body helper:

```cpp
Tensor& Qwen4ExpBnb4Backend::ForwardWholeTokenGraph(
    Context& ctx, Session& session, const int* tokenIdsDevice,
    int count, bool lastTokenOnly, const ForwardOptions& options);

sycl::event Qwen4ExpBnb4Backend::RecordWholeTokenBody(
    Context& ctx, Session& session, WholeTokenGraphIo& io,
    std::size_t& captured_nodes);
```

`WholeTokenGraphIo` is backend-owned and contains only fixed addresses:
uncached mapped control/route/done ring, mapped CPU packet, device epoch,
device route IDs/weights, device selected-slot table, device packet copy, and
the 16-word `GraphControl` table. The graph key must include the exact
topology, top-k, QSA phase, residency/descriptor generations, PLE workspace
generation, and GDN device-head generation. Per-token sequence, position,
block counts, and packet contents are values, never key fields.

Capture order:

1. Run eligibility and all capacity growth. Require count 1, true decode,
   fixed sparse-QSA phase, top-k compatible with the 10-rank ring, no
   overrides, no checkpoint/MTP mode, and graph-safe weights.
2. Perform host embedding and PLE prepare. Publish the fixed PLE stage to the
   device and establish one pre-launch dependency barrier; do not bake the
   host ticket or current position into lambda arguments.
3. Initialize `GraphControl`, device epoch, route rings, packet sentinels, and
   the GDN head table. Validate the mapped-ring visibility preflight.
4. Begin recording. Record a device control-validity/fence node followed by
   all 48 layers in order.
5. At layer 1, record `Qwen4ExpPle::consume_graph` with the fixed device view.
6. For every layer record HC, attention, MLP-HC, router math, resident/shared
   expert work, a ring wait, active-row packet copy, and ranked merge.
   Linear layers use the GDN device-head path; full layers use inline graph
   QSA ops, not `ForwardQsaLayerGraph`.
7. Record the final mixer and a final device gate/done publication. Stop
   before the LM head.
8. End recording, finalize one executable, and publish it through
   `TokenGraphRuntime::MarkExecutableReady`.

Replay order:

1. Submit host-only embedding/PLE preparation and set the one-token/position
   control values; preserve one dependency point before graph launch.
2. Launch the executable, then make exactly one non-blocking event-status
   query to release L0 (`Qwen4ExpTokenGraphRuntime.hpp:250-285`).
3. Service 48 route arrivals with bounded polling, no fixed sleep. Publish an
   explicit empty packet for a layer with no CPU rows; never skip a sequence.
4. After the last flag, wait/query the graph event and read the final done,
   sticky-abort, compressed-count, GDN-head, and session-state gate.
5. Only after a valid gate call `TokenGraphRuntime::Complete`, update
   `currentLength_`, compressed counts, GDN host mirrors, route/residency
   observers, PLE state, and checkpoint bookkeeping.
6. Launch the existing LM head on the same in-order compute queue. Sampling
   and D2H logits remain in `Model::ReadOutput`.

## 6. Highest immediate risks

1. **Host round trips inside the supposed body.** Router publication and
   `actual_route_event.wait_and_throw` (`Backend.cpp:3042-3097`), activation
   staging (`Qwen4ExpExpertExecutor.cpp:760-775`), and CPU result upload
   (`Backend.cpp:1766-1789`) cannot be represented by graph nodes. If left in
   the capture function, the graph records their submission-time effects but
   replay has neither the pinned-host publication nor the worker join.
2. **Capture-only state mutations.** `ObserveRouting`, slot/residency
   bookkeeping, GDN authority flags, PLE tickets, and
   `cache.num_compressed_blocks` execute on the host while recording and do
   not execute on replay (`Backend.cpp:3066-3092,3313-3319`;
   `Qwen4ExpPle.cpp:1003-1009`; `Qwen4ExpPrefill.cpp:202-213`). They must be
   represented by device state plus a post-gate host commit.
3. **Nested/conditional QSA and PLE control flow.** `ForwardQsaLayerGraph` is
   a graph with its own event wait and D2H (`Qwen4ExpPrefill.cpp:847-891,
   1047-1077`), and the dense/sparse projection branch depends on host
   `visible_len`/`num_complete_blocks` (`Qwen4ExpPrefill.cpp:247-330`). A
   single whole-token executable must either inline the chosen QSA variant
   and use device control, or remain ineligible for that phase.

`[I]` Do not start by capturing `ForwardImpl` wholesale. The first useful
patch is the host-service boundary: mapped ring, active-row packet copy,
device publish/wait, bounded service loop, final gate, and post-gate commit.
