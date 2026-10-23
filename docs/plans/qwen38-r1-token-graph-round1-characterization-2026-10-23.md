# Qwen3.8 R1 Whole-Token Graph — Round 1 Characterization (2026-10-23)

Status: **characterization / attribution round**. No production code changed.
Companion probe artifacts (not committed, `tmp/` is gitignored):

```text
tmp/q4e-token-graph-fusion-20261022/
  analysis-packet-merge.md        (design analysis, packet-copy -> merge fusion)
  analysis-final-mixer-tail.md    (design analysis, final mixer + tail D2H)
  results-baseline/               (8-token A/B, contaminated arm)
  results-abab-2/                 (8-token A/B, second sample)
  results-t1-run1/                (1-token A/B, sample 1)
  results-t1-run2/                (1-token A/B, sample 2)
```

## 1. Verified entry state

| Item | Handoff claim | Measured |
|---|---|---|
| HEAD | `331264d` | `331264d` on `main` |
| Dirty worktree | 11 modified + 6 untracked | identical, file by file |
| GPU processes | none | none |
| Free physical memory | ~59.3 GiB | 58.61 GiB (gate 54) |
| `AILA_Q4E_TOKEN_GRAPH` default | 0 | `read_flag(..., false)` at `Qwen4ExpBnb4Backend.cpp:248`, `Qwen4ExpWholeTokenGraph.cpp:35` |
| Layer split | 36 GDN / 12 QSA / 48 | `config.json` `layer_types`: `linear_attention`=36, `full_attention`=12 |
| Build currency | current | binaries 19:12-19:14, newest source 19:10 |

Code anchors named by the handoff all check out: QSA work-group broadcast of the graph
control word (`QsaIndexerOps.cpp:1366`, `AttentionOps.cpp:1416`, `AttentionOps.cpp:1552`),
doorbell-before-resident-experts (`Qwen4ExpWholeTokenGraph.cpp:303-316`),
direct CPU execution in the service thread (`Qwen4ExpExpertExecutor.cpp:1059`),
`preserve_token_graph` snapshot mode (`Qwen4ExpBnb4Backend.cpp:340`, `:3869`,
`Qwen4ExpWholeTokenGraph.cpp:769`).

Verified additionally: GDN history fusion really is disabled in graph mode.
`Qwen4ExpPrefill.cpp:134-135` requires `s_gdn_history_fused && allowQ8 &&
!use_device_conv_head`, and `use_device_conv_head = count == 1 &&
session.graph_gdn_state_enabled_` (`Qwen4ExpPrefill.cpp:112-113`), which the token
graph enables by default. Result: 36 standalone conv-history kernels per token.

## 2. Reproducibility experiments

All runs: 2108-token prompt, greedy, `--max-tokens` as noted, single serial A/B.

### 2.1 Eight-token runs

| Run | Direct token ids | Graph token ids | direct p50 | graph p50 |
|---|---|---|---|---|
| baseline | `16,13,24624,279,1622,72346,20735,13` | `2720,62,15,15,23,21,25,9307` | 63.8 ms | 130.3 ms |
| abab-2 | `16,13,24624,279,2100,999,6513,321` | `16,13,24624,279,1622,5435,6681,7810` | 68.6 ms | 173.9 ms |

Run 1's graph stream is degenerate (`15,15,23,21,25`); run 2's graph stream is exactly the
canonical stream recorded in the 2026-10-20 handoff. Protocol was clean in both runs:
`captures=1 @ pos=2108`, `gate_fail=0`, `service_done=48,96,...,384`, monotone sequence,
no abort/timeout lines.

### 2.2 One-token runs (first-decode-token isolation)

| Run | Direct | Graph | direct p50 | graph p50 | route_wait | cold |
|---|---|---|---|---|---|---|
| t1-run1 | `16` | `16` | 90.9 ms | 134.2 ms | - | - |
| t1-run2 | `16` | `16` | 83.9 ms | 133.1 ms | 84.9 ms | 44.9 ms |

Findings:

1. **The graph's first decode token reproduces** across independent processes and matches
   Direct. The degenerate 8-token stream is therefore not a first-replay fault.
2. The 8-token graph p50 spread (130.3 vs 173.9) exceeds anything seen in the 1-token
   runs (133.1 / 134.2) and coincided with four background analysis agents running
   repo-wide `grep`. Treat the 8-token numbers as contaminated.
3. Direct is also not reproducible across processes (`...279,1622` vs `...279,2100`).
   Consistent with order-dependent CPU cold-expert reduction, and it is the reason a
   serial two-process A/B cannot serve as a Top-1 oracle.

### 2.3 Measurement discipline adopted

Background analysis agents must not run during GPU measurements. A repo-wide `grep`
alongside the graph arm inflates the paced CPU service path; measured effect in this
round was +33% on graph p50 (130.3 -> 173.9).

## 3. Quantitative decomposition of the ~2x gap

Direct per-token, from `[Q4E AdaptiveWork] count=1` (`pos=2112..2115`):

```text
submitted_ms      65.6 / 61.2 / 60.7 / 62.9     host submit interval
cpu_compute_ms    21.0 / 17.5 / 16.8 / 20.6     CPU cold expert math
cpu_join_ms       11.7 /  7.7 /  9.2 / 12.1     exposed CPU join
hc_host_ms         7.4 /  7.3 /  7.4 /  7.3     hyper-connection submit
router_wait_ms    10.3 /  9.6 /  9.7 /  9.1     per-layer router barrier
experts_host_ms   30.3 / 26.5 / 25.6 / 29.1     expert submit
full_host_ms       3.3 /  3.4 /  3.5 /  3.6
```

Direct is **host-submit bound**: ~62 ms of its ~72 ms token is host submission, so its GPU
work overlaps inside that window and its CPU cold math is hidden behind it.

Graph per-token, 1 token, `results-t1-run2`:

```text
replay      133.1 ms
route_wait   84.9 ms   CPU idle waiting for the device doorbell
cold         44.9 ms   CPU cold expert math (device is stalled here)
service     130.7 ms
```

Identity: `route_wait + cold + 3.3 ms = replay`, i.e. the CPU is either busy or idle for
the whole replay and only **3.3 ms/token (2.5%)** is unattributed handoff latency.

Consequences:

- `route_wait` is **not** a GPU stall bucket; it is the CPU's idle time while the device
  works. Reading it as "exposed wait" is the mistake that produced the earlier
  `981 ms` diagnosis.
- The graph removes Direct's ~62 ms host submission, yet lands at 133 ms, so the graph's
  recurring cost is dominated by two terms that Direct hides: exposed CPU cold expert
  work and device body time.
- Per-layer handoff latency is not a lever: even eliminating all of it buys 2.5%.

## 4. Prior-plan items re-examined against the measurements

### 4.1 Packet-copy -> ordered-merge fusion (handoff 8.1): premise invalid, value sub-1%

The handoff proposes publishing rank rows into
`TokenGraphDeviceEpoch::reserved[]` written by the CPU service. `device_epoch` is a
**Device USM** allocation (`Tensor::allocate` -> `ctx.alloc_device` ->
`sycl::malloc_device`, allocated at `Qwen4ExpWholeTokenGraph.cpp:560-564`); the CPU
cannot write it. Moving the epoch to host-uncached USM would force the four per-work-group
gate reads (`Qwen4ExpTokenGraphService.hpp:208-213`) onto `memory_scope::system`, which
re-opens exactly the failure class that produced `0xC0000409` on 2026-10-20.

A safe replacement exists and was designed in this round (P0 in
`tmp/.../analysis-packet-merge.md`): the merge reads the already-CPU-published
`packet->rank_to_row[16]` directly with uncached Vec4 loads, deleting the
`submit_packet_copy` node, the 160 KiB `packet_dev` mirror and the `0xcdcdcdcd` fill,
with no epoch change and no doorbell-fence change.

Value is still bounded by section 3: total handoff latency is 2.5% of a token.

### 4.2 Final mixer into the graph tail (handoff 8.3): value <= 0.33%

`analysis-final-mixer-tail.md` localises the mixer at
`Qwen4ExpWholeTokenGraph.cpp:773` (`RunHyperConnection(ctx, mixer_, 1, false)`), not PLE,
and shows it writes no `hc_inject` (`w_inject == nullptr` -> `total_rows == lowrank`,
`HyperConnectionOps.cpp:499`). Measured/estimated end-to-end benefit:
-0.08..-0.20 ms/token. Its real value is unblocking head-in-graph later.

### 4.3 GDN history fusion (handoff 8.2): 36 nodes, but the device is the wrong target

Deleting 36 kernels is correct and cheap, but at ~30-50 us per kernel it is ~1-2 ms/token
against a 53 ms decision margin. Keep it as a bundled cleanup, not as the round's main
lever.

## 5. Corrected priority

The +5% gate needs the graph at <= 0.95 x Direct, i.e. <= ~80 ms/token from 133 ms.
Section 3 says only two terms can supply that, and they must be split by measurement:

1. **Exposed CPU cold expert work (~45 ms/token)** — Direct hides the identical work
   behind its host-submit window. In the graph the service is a single thread doing 48
   sequential `wait -> cold -> publish` rounds, so the device stalls for the full cold
   time. Any fix must give the device other work during cold compute, or shrink cold work.
2. **Device body time (~85 ms/token)** — Direct's device body cannot exceed its 72 ms
   token, so the graph body is at least +18% and probably larger. Candidates: the two
   direct-path fusions the graph does not carry (`AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL`
   is Direct-only, `Bnb4Backend.cpp:2776-2778`; GDN history fusion disabled), plus the
   graph-mode QSA/GDN kernel variants.

Attribution tool chosen for round 2: the repository already carries device gap
instrumentation built on `sycl::info::event_profiling` (`Bnb4Backend.cpp:3508-3514`),
gated by `AILA_Q4E_DEVICE_QUEUE_PROFILING` (`Model.cpp:44-47`) and
`AILA_Q4E_DECODE_STAGE_EVENTS`. It covers the Direct path only; the first action of
round 2 is to obtain the Direct device busy/idle split with these flags and then add the
equivalent marks to the graph capture.

## 6. Open items carried forward

- Same-process frozen-prefix Top-1 oracle is still not implemented. Acceptance policy for
  this project is Top-1 agreement with a 90% floor and 95% target; the serial two-process
  A/B cannot produce that number, and section 2.2 shows why.
- The degenerate 8-token graph stream from the baseline run is **unexplained**. It was not
  reproducible in three later runs, and the first token was correct every time, so it is
  presently classified as a timing-sensitive defect that must be reproduced deliberately
  (contention ladder) before it is declared absent.
- `AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=1` is set in the production profile, so the Direct
  baseline already carries that fusion while the graph does not. Any graph/Direct
  comparison must state this explicitly.

## 7. Sub-agent findings that changed the plan

| Report | Consequence |
|---|---|
| `analysis-packet-merge.md` | handoff 8.1 as written is impossible (device-USM epoch); P0 alternative designed; value sub-1% |
| `analysis-final-mixer-tail.md` | handoff 8.3 value <= 0.33%; mixer is HC, not PLE; no graph-key change needed if recorded unconditionally |
| GDN history design | timed out, no report; blocker confirmed independently in section 1 (`Prefill.cpp:112-135`) |

## 8. Rules in force for subsequent rounds

1. Top-1 token agreement is the correctness yardstick: 90% floor, 95% target. Bitwise
   equality is no longer required.
2. Routine screening is a single A/B. A/B/B/A interleaving is for production-config
   freezing. A single-config change may run B/A against a previously measured A.
3. A single-config improvement must reach >= 5% to enter the production configuration.
4. Probes stay in `tmp/`; every round writes a committed document like this one.
5. GPU work is serialised under `Global\AilaGpuSingleton`; nothing else may run
   CPU-heavy work during a measured run.

## 9. Round-1 addendum: device side is the real gap, not the doorbell

Instrumentation added this round (committed, zero cost when unused): the packet-wait
kernel accumulates its spin iterations in `TokenGraphDeviceEpoch::reserved[2]`
(`kEpochSpinItersWord`), `submit_publish_done` relays the total into
`TokenGraphControlRing::reserved[0]`, and `[Q4E TokenGraphTiming]` prints it as
`spin_iters`.

First measurement (1 token, pos=2108, direct 70.5 ms / graph 115.5 ms):

```text
route_wait_ms=84.658 cold_ms=27.484 publish_ms=0.175 service_ms=112.967 spin_iters=17527
```

`spin_iters=17527` over 48 layers is ~365 iterations per layer. The device therefore
does **not** block on the CPU; the CPU's `route_wait` idle really is the device working
through the 48 doorbells. This removes the doorbell-latency hypothesis entirely and
confirms the earlier sub-1% verdict on packet-copy fusion.

Comparing the two arms:

| Metric | Direct | Graph |
|---|---:|---:|
| forward span | 88.05 ms | 112.97 ms (`service_ms`) |
| barrier-to-barrier device span (`dev_*_ms` sum) | 82.47 ms | ~85 ms (`route_wait`) |
| exposed CPU cold service | overlapped | 27.5 ms (of which ~17.5 ms is device spin) |

**Correction (same day).** An earlier reading of this data concluded that Direct's device
busy time is only ~5.6 ms and that the graph inflates device work by ~12x. That reading was
wrong. `dev_marks` are `ext_oneapi_submit_barrier()` events
(`Qwen4ExpBnb4Backend.cpp:2856-2860`), so `dev_ms[bucket]` is the *span* between two
consecutive barriers, which already contains any wait for host-side work. It is neither a
kernel-duration sum nor an idle-gap sum. The corrected reading is that Direct and the graph
spend essentially the same ~82-85 ms of barrier-to-barrier device time.

What the numbers do support:

- The graph's deficit versus Direct (113.0 vs 88.0 ms) is accounted for by the exposed CPU
  cold service, not by graph device work.
- `spin_iters=17527` (~365 per layer) shows the device does block on the CPU packet, so the
  two sides are not overlapped in practice even though the device has resident/shared expert
  work scheduled between the doorbell and the packet wait.
- `cold_ms` itself is dominated by pushing the expert result rows into the uncached host
  packet ring (10 rows x 2560 floats per layer, ~100 KiB over PCIe), not by TBB arithmetic:
  `publish_ms` is only 0.175 ms per token because `RunGraphColdPacket` writes the rows inline.

Consequence: the dominant lever is overlapping that host-to-device result traffic with
device work, and the correct attribution tool is per-node device timing of the graph body
(see section 10), not the barrier-span buckets.

## 10. Build environment change (blocking, fixed)

`C:\Program Files\LevelZeroSDK\1.28.2` no longer exists; the machine now carries
`1.33.1`, whose headers live under `include/level_zero/`. `build-refactor/CMakeCache.txt`
still pinned `AILA_LEVEL_ZERO_INCLUDE_DIR` and `AILA_LEVEL_ZERO_LOADER_LIBRARY` to the
removed 1.28.2 tree, so linking `Aila.exe` failed with

```text
ninja: error: 'C:/Program Files/LevelZeroSDK/1.28.2/lib/ze_loader.lib' ... missing
```

Both cache entries were repointed at `1.33.1` and the tree rebuilt 292/292. Anyone
reusing this build directory on a machine with a different Level Zero SDK version must
expect the same failure and repoint those two cache variables.

## 11. Round-1 conclusion: the command graph and the serialised submission were the regression

Two probes built this round resolved the "graph is ~2x slower than Direct" question
completely, and the answer is neither the doorbell nor the CPU experts.

### 11.1 Per-node device timing (`AILA_Q4E_TOKEN_GRAPH_NODE_PROF`)

A captured command graph yields no per-node events, so the body is executed un-captured on
the live queue in this mode (`Qwen4ExpWholeTokenGraph.cpp`, `WholeTokenNodeProfiler`). A
barrier is inserted after the last submission of each of the seven node groups per layer and
labelled with that group; the interval between two marks is the device span of the group
between them. Same technique as the Direct path's `dev_*_ms`.

Measured device spans per token (48 layers, pos 2108-2110):

| group | total | per layer |
|---|---:|---:|
| attn_hc | 5 ms | 0.10 |
| attention | 13-14 ms | 0.29 |
| mlp_hc | 4 ms | 0.08 |
| route + doorbell publish | 3 ms | 0.06 |
| resident experts | 5 ms | 0.10 |
| shared expert | 2 ms | 0.04 |
| **compute total** | **32-33 ms** | 0.68 |

The `tail` bucket is 107-416 ms because it contains the packet-wait spin, i.e. the handoff,
not compute. Compute is therefore ~33 ms/token.

### 11.2 The graph executable costs ~66 ms/token on its own

Same body, same nodes, same handoff, two launch mechanisms:

| launch | `route_wait` | `replay_ms` |
|---|---:|---:|
| `ext_oneapi_graph` replay | 96.3-97.1 ms | 115-133 ms |
| direct in-order submission | 28.0-28.4 ms | 103-113 ms |

`route_wait` measures the segment from publishing packet(L) to observing doorbell(L+1), so
it excludes both `cold` and the spin. It drops by 3.4x when the identical node sequence is
submitted directly instead of replayed from an executable. With compute at ~33 ms, the
captured replay pays roughly **66 ms/token of graph-execution overhead** (1.35 ms/layer,
~190 us/node), which is what made the graph look device-bound.

### 11.3 Serialised submission: the second, independent defect

Un-captured submission initially measured `replay_ms=103-113` while `service_ms` was only
47 ms, with `spin_iters=272164` on the first token. The submission pass is node-by-node and
costs the host tens of milliseconds, and `ServiceWholeTokenGraphReplay` was invoked *after*
it, so the device stalled at layer 0's packet wait for the whole submission. The submission
and the service are now run concurrently (service on a worker thread started before the body
is submitted). `event` is consumed only by the service's final wait and the following in-order
`memcpy_d2h` re-establishes that ordering.

```text
before (serialised):  replay_ms=102.9-113.0  spin_iters=56581-272164
after  (concurrent):  replay_ms=48.4-50.3    spin_iters=8725-11132
```

### 11.4 Result

```text
AILA_Q4E_TOKEN_GRAPH=1 AILA_Q4E_TOKEN_GRAPH_UNROLLED=1, 4 tokens, pos 2108-2111
direct p50 = 73.40 ms   unrolled p50 = 49.45 ms   (-32.7%)
token ids identical in both arms: 16,13,24624,279   gate_fail = 0
route_wait 27.9-30.2   cold 17.9-34.3   publish ~0.16   service 46.4-48.5
```

The remaining budget of the unrolled path is balanced: service is 46-48 ms and the host
submission pass is of the same order, so the two threads are both close to saturated. The
next levers are therefore submission cost (~0.16 ms/node x 352) and `cold`, not node fusion.

### 11.5 Status of the earlier node-cutting plan

The handoff document's section 8 items are now known to be worth far less than the launch
mechanism: packet-merge fusion removes ~4.8 MB of host traffic and one node per layer
(~1-2 ms), GDN conv-history fusion removes 36 small kernels per token (~0.2-1 ms), and the
final mixer is <=0.33%. None of them should be prioritised over the launch-mechanism fix.

## 12. Post-fix bottleneck: host submission vs host service

After 11.4 the token is `max(host submission, host service)` plus ~2 ms, so both terms
were measured directly. A `submit_ms` probe now brackets `RecordWholeTokenGraphBody`.

```text
pos  submit_ms  nodes  route_wait  cold    service  replay
2109  40.394    352    28.293      17.780  46.712   48.573
2110  40.897    352    28.570      18.460  47.683   49.451
2111  41.720    352    28.188      19.890  48.864   50.902
pip50: direct 55.35  unrolled 51.03
```

`nodes=352` counts logical node groups; the actual `queue.submit` calls are roughly 750-800
per token (15-17 per layer), so a submission costs ~53 us and the whole pass is ~40 ms.

The service is the limiter and it is fully serial:

```text
service 46.7 = route_wait 28.2 (device work) + cold 17.8 (CPU experts) + publish 0.16
```

`spin_iters` (8.7k-13.8k) shows the device is idle during most of `cold`, so the two phases
do not overlap: the device's post-doorbell work is only the resident plus shared expert
groups, 5 ms + 2 ms per token. The remaining lever is therefore not node fusion but making
the device own more of the expert work so that the CPU service hides behind it.

Rejected this round: `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` together with
`SYCL_PI_LEVEL_ZERO_BATCH_SIZE=16`, proposed as a way to cut host submission to 15-20 ms.
It changed nothing (`submit_ms` 38.9-47.5, `replay` 50.6), so the per-submit cost here is
not immediate-command-list / WDDM submission overhead.

Also noted for later review, from an independent review of the node inventory: the four
tail kernels (`submit_packet_copy`, `submit_gated_ordered_merge`,
`submit_add_shared_expert_to_y`, `submit_gated_final_bridge`) are four launches per layer
and `submit_packet_copy` moves the full 160 KiB row block from uncached host memory every
layer. Gating the empty rows out of that copy is *not* safe as stated, because
`submit_gated_ordered_merge` decides membership from `packet_dev[rank] >= 0` and the sentinel
write is what clears stale rows; any fusion here must rewrite that check together with it.

### 12.1 Determinism status

`AILA_Q4E_HOTSET_INTERVAL=0` removed one source of cross-process drift (Direct's fifth
token became stable), but a second source remains and affects both arms equally. Both arms
diverge from themselves at token 5-6 across processes, so cross-process Top-1 comparison
cannot adjudicate correctness at all. The leading candidate is floating-point reduction
order in the CPU-served expert path (TBB parallelism). The decision recorded for this
project is that micro-precision alignment is not a requirement: weights are NF4, and the
gate that matters is whether the produced text degrades. A same-process frozen-prefix
comparison, or a text-quality comparison over a longer generation, is therefore the correct
acceptance test, not per-token id equality.
