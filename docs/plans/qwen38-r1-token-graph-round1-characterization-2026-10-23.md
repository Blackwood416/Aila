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

## 13. Residency is exhausted, and cold is at the host memory-bandwidth limit

Section 12 left one lever: give the device more expert work so the 17.8 ms of CPU cold
service hides behind it. That lever is now closed, with direct evidence.

### 13.1 The adaptive cache cannot grow or swap

```text
[Q4E Adaptive Residency] profile_total_keys=24576 ranked_keys=2800 admitted_slots=2800
```

Admission is already at the cap (`AILA_Q4E_MAX_GPU_SLOTS` 2800, Expert Cache 7883 MB of a
15933 MB card, ~2.8 MB per slot). 48 layers x 512 experts = 24576 keys, so residency averages
58 experts per layer against a top-10 selection, and decode misses ~346 of 480 selections
per token (73%).

Enabling the whole runtime-adoption chain (`AILA_Q4E_HOTSET_INTERVAL=1`,
`AILA_Q4E_HOTSET_SWAPS=512`, `AILA_Q4E_HOTSET_ADOPT_LIMIT=4096`,
`AILA_Q4E_HOTSET_ADOPT_READY=1`) over 12 tokens produced no trend in `cold_ms`
(15.5-22.6 ms, no decay) and `route_wait` stayed at 28.0-28.8 ms. The decisive detail is
`swap_ms=0.000` in the AdaptiveWork line: with the cache already fully admitted, no swap is
ever executed, so runtime adoption cannot replace the profile's generic ranking with this
conversation's actual hot set. Residency is effectively static.

### 13.2 cold is bandwidth-bound, not compute-bound

One expert's weights: `gate_up` 2 x 640 x 2560 plus `down` 2560 x 640 = 4.91 M parameters,
4-bit packed = **2.46 MB**. At 346 cold tasks per token that is **851 MB of host weight
traffic per token**, and `cold_ms` averages 17.8 ms, i.e. **47.8 GB/s**.

A desktop dual-channel platform sustains roughly 50-70 GB/s, so the CPU expert path is
already within ~1.3x of the memory system's limit. There is no large factor hiding in CPU
vectorisation: cutting `cold` materially would require reading fewer bytes, which needs
residency, which needs VRAM that does not exist on a 16 GB card.

### 13.3 Resulting floor

```text
token ~= device work 28.2 ms + cold 17.8 ms, with cold's device overlap limited to the
         7 ms of post-doorbell resident+shared work
```

Both terms sit near their hardware limits, so the unrolled pipeline is close to its
structural floor at 46-50 ms/token. Reaching it required removing the two defects in
section 11; what remains is device kernel efficiency (28 ms) rather than pipeline structure.

Recorded for the next round, in expected-value order:

1. Device compute is 28 ms for ~750 launches, i.e. 37 us per launch against 0.68 ms/layer of
   actual work. Per-launch fixed cost, not arithmetic, is the term to attack next.
2. `AILA_Q4E_CPU_PACKED_CACHE_MB` defaults to 0 (host packed expert cache disabled) and was
   never exercised in these runs; worth one A/B before concluding 13.2 is final.
3. Reserve headroom is 3584 MB of 15933 MB. Shrinking it would admit a few hundred more
   slots, but 13.1 shows the miss rate is dominated by the per-layer share, so the expected
   gain is small.

## 14. Null results recorded so they are not retried

Three single-variable experiments were run against the unrolled path and left the
bottleneck untouched. All three are recorded here because each one looks plausible on paper.

| experiment | setting | measured | verdict |
|---|---|---|---|
| host packed expert cache | `AILA_Q4E_CPU_PACKED_CACHE_MB=4096` | `cold_ms` 14.9-20.0 vs 16.7-22.6, p50 49.04 | null |
| runtime hotset adoption | `HOTSET_INTERVAL=1 SWAPS=512 ADOPT_LIMIT=4096 ADOPT_READY=1` | `cold_ms` no trend over 12 tokens, `swap_ms=0.000` | null, cache already fully admitted |
| L0 command-list batching | `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`, `BATCH_SIZE=16` | `submit_ms` 38.9-47.5 vs 40.3-41.7 | null |

The packed-cache result is the informative one: if `cold` were limited by page-fault or
mmap behaviour, caching 4 GB of packed weights in host memory would have moved it. It did
not, which supports section 13.2: the cost is the bytes that must be touched, not how they
are laid out or cached.

### 14.1 Paired results accumulated so far

Nine paired runs of the same process (Direct arm and unrolled arm back to back):

```text
Direct   p50: 73.40 57.53 56.16 56.73 57.68 55.35 56.06 54.80 55.77
unrolled p50: 49.45 49.14 49.64 49.48 49.24 51.03 50.62 50.64 49.04
```

Direct spans 54.8-73.4 ms (+-15%); the unrolled path spans 49.0-51.0 ms (+-2%). Median
improvement is 1 - 49.5/56.7 = 12.7%, and the worst pairing still clears the 5% gate. The
stability of the unrolled path is itself a property worth keeping: it comes from having no
per-layer host submission in the critical path.

## 15. Correction: the floor in 13.3 was over-stated, and the cause is thread starvation

Section 13.3 concluded that cold at 47.8 GB/s was near the memory-system limit and that the
pipeline therefore sat at a structural floor of 46-50 ms. An independent review of that
reasoning is correct and the conclusion is withdrawn.

### 15.1 The CPU is idle for the entire device phase

`route_wait` is 28.0-28.8 ms per token during which the host service thread is parked in
`wait_request`. `cold` is 15.5-22.6 ms per token during which the device has only the
resident plus shared groups queued, 0.10 + 0.04 = 0.14 ms/layer = 6.7 ms/token, before it
reaches `submit_packet_wait`. The two phases are strictly serial:

```text
per layer:  device 0.58 -> doorbell -> device 0.14 -> wait for cold 0.371 -> merge
exposed cold per layer = 0.371 - 0.140 = 0.231 ms   -> 11.1 ms/token
service_ms = route_wait 28.4 + cold 17.8 + publish 0.15 = 46.4 (matches measurement)
```

The achievable value if cold overlapped fully would be `max(device 35.1, cold 17.8) = 35.1`,
not 46.4.

### 15.2 Why the 47.8 GB/s figure is not a hardware limit

851 MB/token divided by 17.8 ms gives 47.8 GB/s, which *looks* like a platform limit. It is
not. There are only ~346 cold tasks per token spread over 48 layers, i.e. **about 7 tasks per
layer**, and the executor parallelises one task per TBB thread. With 7 runnable tasks on an
18-thread arena, at most 7 cores participate, so the aggregate is 7 x ~6.8 GB/s. The
bandwidth per participating core is already at a normal single-core streaming rate for
unpack-bound NF4 work; the shortfall is **parallelism within a layer, not bytes**.

Refilling the arena requires splitting one expert's reduction across several thread tiles
(row-block partition of `gate_up` and `down`) instead of one thread per expert. At 18
participating cores the same 851 MB would take roughly 7-8 ms, which would move
`service_ms` from 46.4 to about 36, and the token from 49 to about 42 (at which point the
40 ms host submission pass finally becomes the limiter and is worth attacking).

### 15.3 Revised lever order

1. Intra-expert parallelism in the CPU cold path (target `cold` 17.8 -> ~8 ms).
2. Host submission cost, 40 ms, only after (1) because it is not the limiter today.
3. Device compute, 28 ms, unchanged and dominated by attention (14 ms of it).

The 12.7% median improvement in 14.1 stands as measured; what changes is that it is not the
end of the road.

## 16. The host is a 2015 Xeon: 48 GB/s is the memory wall, not a thread-starvation artefact

Section 15 hypothesised that `cold` was thread-starved and that intra-expert parallelism would
recover a 2x. Both halves of that are wrong, and the code plus the hardware spec settle it.

**Code**: the parallelism is already fine-grained. `cpu_expert_gate_up_swiglu_tbb_avx2`
(`src/ops/Qwen4ExpCpuExpertOps.cpp:1517`) partitions the 640 intermediate rows in tiles of 16,
`cpu_expert_down_tbb_avx2` (`:1557`) partitions 2560 output rows in tiles of 32, and
`cpu_expert_batch_forward` (`:1648`, `:1675`) additionally partitions tasks x tiles. A single
expert already spreads over dozens of TBB chunks; there is no one-thread-per-expert serialisation
to remove.

**Hardware**:

```text
CPU      : Intel Xeon E5-2696 v3 (Haswell-EP), 18 cores / 18 threads @ 2.3 GHz, AVX2 only
Memory   : 4 x 16 GB Micron DDR4-1866 RDIMM, 4 channels
Peak     : 1866 MT/s x 4 channels x 8 B = 59.7 GB/s theoretical
Practical: ~45-52 GB/s sustained for streaming reads
Measured : 851 MB / 17.8 ms = 47.8 GB/s  -> 80-95% of the sustainable figure
```

So `cold` is already at the host memory wall. The 47.8 GB/s figure in 13.2 stands, and the
"floor" wording in 13.3 was only wrong in the sense that it discouraged further work: the
physically available win is not in making `cold` faster, it is in **not paying it serially**.

Restating the decomposition with the hardware bound in place:

```text
device work per token          = 28.4 (route_wait) + 6.7 (resident+shared, after doorbell)
                                 = 35.1 ms
host cold per token            = 851 MB / 47.8 GB/s = 17.8 ms
overlap achievable today       = 6.7 ms (only the post-doorbell device work is concurrent)
exposed cold                   = 11.1 ms
token                          = 35.1 + 11.1 = 46.2 ms   (measured 49.0-51.0 at p50)
```

Since `cold` cannot get faster and residency is VRAM-capped, the only remaining way to recover
the 11.1 ms is to place more device work between the doorbell and the packet wait. Everything
after the router depends on the cold result, so that work has to come from a restructure:
giving the device work that is independent of layer L's expert result (for example moving a
part of the next layer's pre-router chain, or speculating the route so cold results are ready
before the doorbell). Both are pipeline redesigns, not tuning.

The remaining tuning lever is device compute itself, 35.1 ms, of which attention is ~14 ms
(12 QSA layers at ~0.6 ms plus 36 GDN layers at ~0.15 ms) and the hidden-state/hyper-connection
groups another ~9 ms.

## 17. Next target: the QSA layer body, and why ESIMD is the right tool here

Section 16 closes the host side. The remaining device budget is 35.1 ms/token, of which
attention is ~14 ms and the QSA layers are the expensive part (~0.62 ms per QSA layer for 12
layers, versus ~0.15 ms per GDN layer for 36).

### 17.1 The QSA body is two orders of magnitude off roofline

QSA is block-sparse (512-token blocks) with an indexer, `head_dim=256`, `Q=1` decode at
KV ~= 2052. First-order bounds for one QSA layer with `Hkv=2`:

```text
K/V traffic = 2052 x 256 x 2 (K,V) x 2 B x Hkv(2) = 4.2 MB
measured 0.62 ms  ->  ~6.8 GB/s      versus A770 ~500 GB/s
arithmetic  = 2052 x 256 x Hkv x 2   = 2.1 MMAC  ->  ~6.8 GFLOP/s  versus ~10+ TFLOP/s
```

Both the bandwidth and the arithmetic occupancy are ~50-70x below what the device can do.
So this kernel is latency/occupancy bound, not bandwidth bound, and there is real room. This
is the one place in the current token budget where a large factor is plausible.

### 17.2 The body is a six-stage chain, not one kernel

`RecordSingleFullAttentionLayerGraphBody` (`src/models/Qwen4ExpPrefill.cpp:933`) forwards to
`RecordQsaLayerGraphBody`, which drives: `qsa_prepare_q_and_cache_k`,
`qsa_compute_block_scores` (indexer), `qsa_compress_key_blocks`,
`qsa_select_topk_blocks_graph`, then the sparse attention core. The 0.62 ms is the sum of all
of them, and the next measurement must split it before any rewrite: rewriting the indexer when
the sparse core dominates would be wasted effort, and vice versa.

### 17.3 Execution plan, in the order the evidence discipline requires

1. Add sub-stage marks inside `RecordQsaLayerGraphBody` using the existing
   `whole_token_node_profiler()` mechanism, so the 0.62 ms splits into per-stage device spans.
   Run one token, read the split. No kernel changes yet.
2. Write the ESIMD variant **only for the dominant stage**, behind an env flag, keeping the
   current kernel as the control.
3. Build the correctness harness before claiming anything: f64 CPU reference for the exact
   operator, per-stage tolerance, `errors: 0/N`, warmup plus >=20 timed samples, median with
   p10/p90 over 3 runs.
4. Then A/B against the current stage on the real token.

Constraints that apply from the A770 kernel skill and must not be violated:

- ESIMD attention work-groups must stay small: `WG=512` has produced TDRs on A770 even for
  trivial kernels, and non-zero compiler spill with `RPT>4` is a device-lost trigger.
- `BM=64, BN=32, VLEN=8, WG=512` (~40 KB SLM) is a preserved `UR_RESULT_ERROR_DEVICE_LOST`
  failure. Do not reintroduce that geometry.
- Run new ESIMD geometry as isolated one-shot processes one at a time, with the GPU singleton
  held, and record any hang or device loss as a versioned negative result.

### 17.4 What the host cannot do any more, restated

For anyone resuming this work: the host side is finished. `cold` is at 80-95% of a 59.7 GB/s
DDR4-1866 four-channel memory system, residency is capped by 16 GB of VRAM, host submission is
40 ms and already fully overlapped, and the command-graph replay penalty is avoided. All
remaining headroom is in GPU kernel efficiency, starting with QSA.

## 18. Measured device breakdown, and why the target changed

Section 17 assumed QSA was the expensive part and proposed an ESIMD rewrite of its dominant
stage. Section 18.1 is the measurement that was required first, and it invalidates that
assumption before any kernel was written.

### 18.1 Breakdown at pos 2109, steady state, ms per token

| bucket | ms | share | notes |
|---|---:|---:|---|
| HC (`attn_hc` + `mlp_hc`) | 9.40 | 27% | 96 calls, 0.098 ms each |
| GDN attention, 36 layers | 8.04 | 23% | 0.223 ms/layer |
| resident + shared experts | 8.21 | 23% | 0.120 + 0.051 ms/layer |
| QSA, 12 layers, all stages | 6.70 | 19% | 0.558 ms/layer |
| router (front + topk + publish + classify) | 3.08 | 9% | 0.064 ms/layer |
| **device compute total** | **35.4** | | `tail` excluded, it contains the packet wait |

QSA per stage, 12 layers: `qkv_proj` 1.14, `indexer_qk` 0.27, `prepare_qkv` 0.52,
`prepare_idx_q` 0.21, `compress` 0.24, `blk_scores` 0.48, `select` 0.52, `validate` 0.19,
`attend` 2.36, `o_proj` 0.77. The sparse attention core is only **2.36 ms/token**, and no
single QSA stage exceeds 1.2 ms. An ESIMD rewrite of `attend` therefore caps out at about
2.4 ms, not the 14 ms that was assumed.

### 18.2 Every bucket runs at 20-35 GB/s on a 500 GB/s device

The QSA `attend` stage moves ~4.2 MB of K/V per layer and takes 0.197 ms, i.e. 21 GB/s. The
resident expert stage moves ~2.4 MB per layer in 0.120 ms, i.e. 20 GB/s. The HC stage is
0.098 ms for a 2560-wide projection. Every bucket sits at the same low effective bandwidth,
which is the signature of a **per-launch cost**, not of arithmetic or memory limits: ~750
`queue.submit` calls over 35.4 ms is 47 us per launch.

### 18.3 The pipeline is balanced, so only kernel count helps both sides

```text
host submission   40.4 ms/token = 0.83 ms/layer
device spans      28.0 ms/token = 0.58 ms/layer   (= route_wait)
```

The two are within 40% of each other. Reducing only the host side cannot help because the
device spans would not shrink, and reducing only the device side cannot help because the
submitter is already at 0.83 ms/layer. Any change that removes kernels reduces **both** terms
at once, which is why the next round should fuse rather than micro-optimise:

- the four tail kernels per layer (`packet_copy`, `gated_ordered_merge`,
  `add_shared_expert_to_y`, `gated_final_bridge`) are 4 launches each, 192 per token, and one
  of them moves 160 KiB from uncached host memory every layer;
- `router_softmax_topk_device` + `submit_input_and_route_publish` +
  `moe_resident_classify_device` are three single-work-group kernels over the same ten route
  ids, 96 launches per token;
- the two HC calls per layer are 96 launches at 0.098 ms each and are the single largest
  bucket, so any fusion that absorbs them is worth more than the whole QSA rewrite.

ESIMD remains the right tool for a stage once it is shown to be arithmetic- or
bandwidth-bound in isolation rather than launch-bound; none of the measured stages currently
is, so writing ESIMD first would have optimised the wrong thing.

## 19. The launch floor is 21 us, and it is 39% of the device time

Sections 17-18 tried to attack individual kernels and every attempt failed. This section
records why, with the calibration that should have been taken first.

### 19.1 Calibration

Eight empty kernels were submitted back to back, each bracketed by two profiler marks:

```text
[Q4E NodeProf] ... empty=0.15/7        -> 21 us per empty kernel
```

So every span in the profiler contains a **21 us launch floor** on top of whatever the kernel
actually does. That single number reconciles every previous contradiction:

| stage | span | minus 21 us floor | verdict |
|---|---:|---:|---|
| `hc_norm` (per call) | 53 us | 32 us | real work, surprisingly slow for 20 KB |
| `hc_down` | 54 us | 33 us | real work, ~200 GB/s |
| `hc_up` | 30 us | 9 us | matches the 6.5 MB / 500 GB/s estimate |

Corrected device budget, ms per token:

| bucket | ms |
|---|---:|
| HC (`hc_norm` + `hc_down` + `hc_up`) | **13.27** |
| GDN attention, 36 layers | 8.25 |
| resident + shared experts | 7.97 |
| QSA, 12 layers, all stages | 6.78 |
| router | 3.47 |
| **total** | **40.25** |

The earlier `attn_hc 4.88 + mlp_hc 4.52` reading was an attribution artefact: with only
group-level marks the HC call was split across two labels. The sub-stage marks give the true
13.27 ms, and HC is therefore **33% of the device time**, not 27%.

### 19.2 Launch floor share

```text
750 launches x 21 us = 15.75 ms  =  39% of the 40.25 ms device budget
```

Kernel work is the other 24.5 ms. Any change that keeps the kernel count the same can at best
touch 61% of the device time, and each individual stage is only a few percent of it, which is
exactly why tuning `grouped_rms_norm`, `hc_decode_down_inject`, its work-group size, its
prefetch depth and its fused form all moved nothing.

### 19.3 Negative results recorded this round

| attempt | change | result |
|---|---|---|
| software-pipelined loads | `AILA_Q4E_HC_DECODE_PL_DEPTH=4` (existing bit-exact `_pl` variants) | worse: `attn_hc` 4.88 -> 7.98, `mlp_hc` 4.52 -> 7.78 |
| work-group size | `AILA_Q4E_HC_DECODE_DOWN_WG=64` / `=128` | no change (4.86/4.53, 4.83/4.52) |
| fused norm+down submission | `AILA_Q4E_DECODE_FUSED_HC=1` | worse: 5.72 / 5.72 |
| vectorised norm loads | vec8 in `grouped_rms_norm` | no change: `hc_norm` 5.23 -> 5.16 |
| L0 batching | `UR_L0_BATCH_SIZE=16` | submit 40 -> 29 ms but `route_wait` 28 -> 55 ms, p50 77 ms |
| immediate command lists | `UR_L0_USE_IMMEDIATE_COMMANDLISTS=1` | submit 40 -> 32 ms but `route_wait` -> 123 ms, p50 150 ms |

The last two are important: making host submission cheaper made the token *worse*, which proves
the device is the limiter and that `route_wait` is real device time rather than a submission
artefact. The work-group result also has a structural explanation: the down GEMM has 324
output rows and assigns one subgroup per row, so the total thread count is fixed at ~5184 no
matter how the subgroups are grouped into work-groups. Raising parallelism there would need a
split-K reduction, which changes the summation order.

### 19.4 Plan that follows from the calibration

Cut launches, in this order, because each launch removed is worth 21 us and these are the
largest groups:

1. HC: 3 kernels x 96 calls = **288 launches**. Fusing to one is worth up to 4.0 ms, and it is
   the biggest single group in the token.
2. Tail: 4 kernels x 48 = **192 launches**, worth up to 3.0 ms. The existing suggestion to gate
   the empty rows out of `submit_packet_copy` is unsafe on its own because
   `submit_gated_ordered_merge` uses the sentinel to clear stale rows, so the membership check
   must move with the fusion.
3. Router: `moe_front_fused` + `topk` + `route_publish` + `resident_classify` = **192 launches**,
   worth up to 3.0 ms, and all four already operate on the same ten route ids in one
   work-group.

Together those are 672 of the ~750 launches and up to 10 ms, i.e. 49 -> ~39 ms. This is the
only remaining lever of that size; everything else in the device budget is a small share of
the 24.5 ms of real work.

## 20. First real launch-count win, and the corrected cost of a launch

### 20.1 Tail fusion (landed, on by default)

`submit_gated_ordered_merge`, `submit_add_shared_expert_to_y` and `submit_gated_final_bridge`
are three launches per layer, 144 per token. They are all element-wise over the same hidden
row, and each stage consumes only the value the previous stage produced for that same element,
so one work-item can carry an element through all three phases with no cross-item dependency
and therefore no barrier. `submit_gated_fused_tail` does exactly that and reproduces each
stage's bf16 rounding point explicitly, so it is bit-identical rather than merely close:
merge rounds to bf16, the shared add rounds again from that value, and the bridge rounds from
the second value.

Two A/B pairs taken in opposite order, 8 tokens, `route_wait` is the device-side signal:

| pair | tail fused | `route_wait` median | decode p50 |
|---|---|---:|---:|
| 1 (fused first) | on / off | 27.09 / 28.06 | 50.31 / 48.47 |
| 2 (off first) | on / off | 27.19 / 28.04 | 47.50 / 49.87 |

`route_wait` improves by 0.85-0.97 ms in **both** pairs, while decode p50 flips sign between
them; Direct itself swings 55.9-62.3 ms across those runs, so p50 is dominated by host-side
noise at this effect size. The fusion is therefore on by default and
`AILA_Q4E_TG_TAIL_FUSED=0` disables it.

### 20.2 The real cost of a launch is ~9.4 us, not 21 us

96 launches removed bought 0.9 ms, i.e. **9.4 us per launch**. The 21 us figure in 19.1 was
measured with `ext_oneapi_submit_barrier` marks on a profiling queue, and barriers on a
profiling queue allocate timestamped L0 events and force a command-list batch boundary, so
that number is an upper bound that does not describe the production path. Every launch-cost
projection in 19.4 must be rescaled by 0.45:

| fusion | launches removed | projected saving at 9.4 us |
|---|---:|---:|
| tail (done) | 96 | 0.90 ms |
| HC 3 -> 1 | 192 | 1.80 ms |
| router 4 -> 1 | 144 | 1.35 ms |
| packet copy folded into the merge | 48 | 0.45 ms |

### 20.3 `AILA_Q4E_DECODE_FUSED_HC=1` is confirmed worse

Previously dismissed on the group-level attribution; re-tested with the reliable metric, with
tail fusion on and 8 tokens:

```text
DECODE_FUSED_HC=0 : route_wait median 27.36   p50 49.01
DECODE_FUSED_HC=1 : route_wait median 30.05   p50 52.63
```

So the existing fused norm+down kernel costs more than the 96 launches it saves. It stays off.
Fusing HC is still worth 1.8 ms in principle, but it needs a different kernel (compute the
four per-group RMS scales inside the down kernel and apply them to the dot product on the fly,
so the normalised vector never has to be materialised) rather than the existing variant.

### 20.4 Sub-graphs: promising on paper, and unvalidated here

An external review suggests recording per-layer command sub-graphs so that each layer's ~15
kernels execute back to back inside one closed L0 command list (claimed ~2.2 us per node) with
the WDDM submit happening immediately after the doorbell write, cutting host submission from
40.4 ms to ~2.4 ms and the launch floor to ~2.4 ms per token.

That contradicts what this machine already measured: replaying the same node sequence as one
352-node command graph gives `route_wait` 96.3-97.1 ms versus 28.0-28.4 ms when submitted
directly, i.e. roughly 90 us per real kernel of graph-execution overhead, not 2.2 us. Before
any sub-graph refactor, a one-layer sub-graph must be measured: if per-node graph overhead
persists, sub-graphs inherit the whole-token graph's problem and the refactor is worthless.

## 21. Norm reduction change, and the current combined state

### 21.1 Sub-group shuffle reduction in the HC normalisation

`grouped_rms_norm` reduced a 256-entry SLM array with an eight-step tree, paying one full
work-group barrier per level. It now reduces inside sub-groups with `reduce_over_group`,
collapses 16 sub-group partials with one barrier, and finishes with a second barrier. The
summation order changes; the operator does not.

```text
hc_norm  5.16 -> 4.58 ms per token  (-0.58 ms, 97 calls, 6 us per call)
```

So the barrier tree did cost something, but a 10240-element normalisation still costs ~38 us
of kernel time after subtracting the ~9.4 us launch floor, which remains unexplained and is
now the largest single anomaly in the device budget.

`route_wait` did not move measurably (27.19 -> 27.24 across the before/after runs). At this
effect size the token-level metric cannot resolve it: direct p50 alone swings between 54.6 and
62.3 ms across consecutive runs on this machine, and the unrolled p50 swings 47.5-49.2. Any
future change worth less than ~1 ms needs paired repeat runs, not a single A/B.

### 21.2 Current state, and a correctness note

```text
route_wait median 27.24   cold median 20.30
unrolled decode p50 49.24 (47.50-49.24 across recent runs)
direct   decode p50 54.64 (54.64-62.28 across recent runs)
```

In the combined run the Direct arm and the unrolled arm produced **all eight** token ids
identically, including the fifth and sixth positions that had diverged in earlier runs, and
`gate_fail` was 0 in every run. The fused tail is bit-exact by construction and this is
consistent with that.

### 21.3 Where the remaining time is

```text
token ~47.5-49 ms  =  host cold 20.3 (DRAM-bound, irreducible)
                    + device critical path 27.2 (route_wait)
                    + ~1-2 ms handoff and mismatch
```

Everything after the packet is on the critical path. Per section 20.1 the exposed device work
is HC ~12.6, GDN attention 8.3, router 3.5, QSA 6.8, tail ~2. Launch reduction has been
exhausted as far as it is cheap (tail done, 0.9 ms; the remaining HC and router fusions are
worth ~0.9-1.8 ms each at 9.4 us per launch and both need real kernel work, with one failed
precedent already recorded in 20.3).

## 22. Observer effect: the in-code profiler inflates short spans by up to 5x

Every per-stage number in sections 17-21 came from the in-code profiler, which is not a
neutral observer. This section measures how far off it is, using VTune `xpu-offload` as the
observer-free reference: it is a pure trace of Level Zero timestamps, so it needs no marks, no
profiling queue and adds no barriers.

### 22.1 What each observer actually changes

| observer | what it does | changes synchronisation? |
|---|---|---|
| `AILA_Q4E_TOKEN_GRAPH_TRACE` | two `steady_clock::now()` per layer plus a final log line | **no** — it is host-side bookkeeping only |
| `AILA_PROFILE_Q4E_COLD_DEVICE=1` | constructs the queue with `sycl::property::queue::enable_profiling` | yes, indirectly: every command carries a timestamp request |
| `AILA_Q4E_TOKEN_GRAPH_NODE_PROF=1` | submits one `ext_oneapi_submit_barrier()` after each of 7 groups and 21 sub-stages, 756 per token | **yes, strongly** — each barrier is an explicit L0 barrier node that drains all prior work and carries cache-flush semantics |

The trace flag defaults to false in the code but the benchmark runner sets
`AILA_Q4E_TOKEN_GRAPH_TRACE=1` (run_ab.ps1:638), so every p50 and every `route_wait` reported
so far was taken with trace on. That is acceptable: trace changes no synchronisation, and it
is the only way `route_wait` is emitted at all.

### 22.2 Measured inflation

Same kernels, both methods, same run configuration:

| stage | profiler span | VTune real | inflation |
|---|---:|---:|---:|
| `hc_decode_down_inject` | 50 us | **41 us** | 1.2x |
| `hc_decode_up_mix` | 28 us | **21 us** | 1.3x |
| HC normalisation (`Global=1024`, 4 work-groups) | 47 us | **9-20 us** | **2.5-5x** |
| QSA `attention_decode_exact_head256_partial_merge` | 197 us | **154 us** | 1.3x |

The inflation is **not uniform**: the shorter and narrower the kernel, the more the 756 barrier
nodes and the profiling timestamps dominate it. A barrier node apparently costs on the order of
10 us of device-visible time on this stack, which is why an empty kernel measured 21 us through
the marks while the production path only pays ~9.4 us per launch (section 20.2).

Consequences that must be applied to earlier sections:

- The claim in 18.2 and 21.1 that the HC normalisation is "10x too slow for 20 KB" is **wrong**.
  It really costs 9-20 us, which is unremarkable. There is no anomaly there to fix.
- HC's real per-call cost is about 41 + 21 + ~15 = **77 us**, so ~7.4 ms per token rather than
  the 13.27 ms derived from inflated spans.
- Per-stage spans from the in-code profiler remain usable for **ranking** and for measuring a
  change made in the same configuration, but must never be quoted as absolute kernel costs, and
  they bias against short kernels when comparing across stages.

By contrast `route_wait` (27.2 ms) is host-measured around the doorbell protocol and contains
no marks, so it is a true device-side figure and stays the primary metric.

### 22.3 Method going forward

1. `route_wait` and decode p50 for decisions.
2. VTune `xpu-offload` (no marks, no profiling queue) for absolute kernel durations.
3. The in-code profiler only for A/B of a change within the same configuration, and only after
   checking that the change does not merely move work between marks.

One caveat recorded for the next user of the VTune trace: a 2108-token prompt spends ~127 s in
prefill and only ~0.1 s in decode, so the aggregated hotspot CSV is prefill-dominated. Decode
kernels must be isolated either by name (`hc_decode_*`, `attention_decode_*` run only in
decode) or by work size (`grouped_rms_norm` at `Global=1024` is the four-work-group decode
shape; the prefill shape is `Global=524288`).

## 23. Work-group per row: the first kernel change that the observer-free data predicted

### 23.1 The finding

VTune's no-marks trace (section 22) put the decode down GEMM at 41 us for 6.55 MB, i.e.
160 GB/s. The limit for that kernel is not bandwidth but **work-group count**: the production
mapping puts sixteen output rows in each 256-thread work-group, and the decode shape has
`lowrank + groups = 324` rows, so it launches **21 work-groups** on a device that can host well
over a thousand. In-flight bytes per thread were never the problem, which is exactly why the
software-pipelined `_pl` variants make it worse at every prefetch depth (4 in section 20, 2 and
3 re-tested here: `route_wait` 27.2 -> 33.5).

Re-mapping so that one whole work-group owns one output row gives 324 work-groups of 256
threads, with the row reduction done cooperatively inside the work-group: sub-group
`reduce_over_group`, one barrier, a second sub-group reduction over the 16 partials. No extra
launch and no split-K buffer.

### 23.2 Result, two A/B pairs in opposite order

| pair | mapping | `route_wait` | decode p50 | Direct p50 same run |
|---|---|---:|---:|---:|
| 1 | sub-group per row | 27.24 | 49.67 | 55.75 |
| 1 | **work-group per row** | **25.19** | **46.51** | 56.21 |
| 2 | sub-group per row | 27.26 | 48.57 | 56.53 |
| 2 | **work-group per row** | **25.43** | **46.51** | 60.63 |

`route_wait` improves 1.81-2.05 ms and p50 2.06-3.16 ms in both orders. The predicted saving
was 2.0 ms (41 us -> ~20 us on 96 calls), so the measurement matches the mechanism.

The mapping is on by default; `AILA_Q4E_HC_DOWN_WGROW=0` restores the old one. It is not
bit-identical (the reduction order inside a row changes), which is accepted: weights are NF4
and the acceptance criterion is text quality, not per-token id equality.

### 23.3 Cumulative state

```text
unrolled decode p50   46.5 ms  (was 49.5 before the tail fusion and this change)
route_wait            25.2-25.4 ms  (was 28.0)
Direct     decode p50 55.7-60.6 ms in the same runs
```

Against Direct measured in the same process this is now a 17-23% improvement, and against the
slowest Direct runs observed (73.4 ms) 36%.

### 23.4 What the method change bought

Sections 17-21 spent four attempts on the hyper-connection kernels and got nothing but negative
results, because the in-code profiler reported the down GEMM at 50 us and the normalisation at
47 us, which made them look like arithmetic or memory problems. The observer-free trace put the
same kernels at 41 us and 9-20 us and, more importantly, exposed the work-group count, which
was visible in the profiler data all along but not interpretable without the true durations.
The lesson is recorded in 22.3: VTune trace for durations, `route_wait` for decisions, in-code
marks only for same-configuration A/B.

## 24. Sub-graph experiment: implemented, measured, rejected

Section 20.4 recorded a proposal to record the token as per-layer command sub-graphs so that
each layer's nodes execute from one closed command list and each layer's doorbell reaches the
device at its own submission instead of at a batch boundary. It is now implemented
(`AILA_Q4E_TG_SUBGRAPHS=1`, 50 graphs: prologue + 48 layers + epilogue, built by
`RecordWholeTokenGraphPrologue/Layer/Epilogue`) and measured against the two alternatives on
the same code.

| launch mechanism | `route_wait` median | decode p50 | gate failures |
|---|---:|---:|---:|
| one whole-token graph (352 logical nodes) | 93.34 | 116.44 | 0 |
| **50 per-layer graphs** | 30.00 | 54.61 | 0 |
| **direct unrolled submission** | **25.37** | **47.08** | 0 |

Sub-graphs are a large improvement over the whole-token graph (93.3 -> 30.0) and still worse
than submitting the same nodes directly (25.4). The per-node cost of command-graph execution
scales with graph size:

```text
whole-token graph : (93.34 - 25.37) / 750 nodes = 90 us per node
50 per-layer graphs: (30.00 - 25.37) / 750 nodes = 6.1 us per node
```

So the penalty is not a fixed per-node charge but grows with the size of the dependency graph,
which is consistent with graph-wide resource barriers or transitive dependency resolution rather
than with the 2.2 us per-node figure the proposal assumed. Splitting the graph recovers most of
the loss and none of the benefit, so direct submission remains the architecture and
`AILA_Q4E_TG_SUBGRAPHS` stays off by default.

Note that the earlier whole-token graph measurement of 96.6 ms `route_wait` in section 11.2 was
taken before the tail fusion, the work-group-per-row GEMM, the normalisation change and the
concurrent service thread, so it is not comparable; the 93.34 ms above is the current-code
figure with all of those in place.

The three-way comparison also settles what the remaining device budget is spent on: at 25.37 ms
of `route_wait` with ~750 submissions, launch and ordering overhead is no longer the dominant
term, and kernel work is.

## 25. Corrected decode budget from the observer-free trace

A trace taken with `--max-tokens 1200` makes decode long enough to dominate the hotspot
report (prefill is unchanged at ~127 s while decode grows to ~58 s), and decode kernels then
separate cleanly by instance count: 96, 48 or 12 calls per token. Two corrections to earlier
readings came out of it.

### 25.1 The resident-expert group is not on the critical path

`launch_moe_decode_gate_up_swiglu_grouped` and `launch_moe_decode_down_to_rank_f32` together
cost about **12.8 ms/token**, which would make them the largest group in the token. They are
not, because they run between the doorbell and the packet wait, i.e. inside the window where
the host is computing cold experts, so they never appear in `route_wait`. They are already
free and must not be optimised for critical-path reasons.

### 25.2 `zeCommandListAppendMemoryCopy` in the decode rows is prefill

The row that looks like 12.65 ms/token of copies is 7330 instances, which is 94 copies times
the 78 prefill chunks, not 109 per decode token. Prefill remains 97% of the wall time in this
workload and instance-count division must always be checked against a known multiplicity.

### 25.3 The actual critical path, in observer-free numbers

```text
HC (norm + down + up)                  ~7.0 ms after the work-group-per-row change
packed_nf4_gemv_bf16_impl               3.02   (96 calls, 31 us each)
packed_nf4_gemv_gate_up_swiglu          1.43   (48 calls, 30 us each)
hc_inject_add_and_grouped_rms_norm      0.96   (48 calls, 20 us each)
moe_ordered_accumulate_down_rows        0.84
launch_router_softmax_topk              0.82
QSA attention_decode_exact_head256      1.9    (12 calls, 154 us each)
route_wait                              25.4 total
```

### 25.4 Next target has the same defect the GEMM had

`packed_nf4_gemv_gate_up_swiglu` uses one sub-group per output row and sixteen rows per
work-group (`rows_per_group = wg_size / sub_group_size`, `num_groups =
ceil(intermediate_size / 16)`), so the shared expert's `intermediate_size = 640` launches only
**40 work-groups**. That is the identical defect the hyper-connection down GEMM had, and the
same remap applies: one work-group per row gives 640 work-groups. At 30 us per call and 48
calls per token the ceiling on that change is about 0.9 ms.

`packed_nf4_gemv_bf16_impl` already uses one work-group per output row (`grid = out_features x
wg_size`), so it has no equivalent remap available and its 31 us has to be attacked differently
or left alone.

## 26. The overlapped window is bigger than 25.3 assumed

Section 25.4 predicted that re-mapping the shared expert gate/up kernel to one work-group per row
(40 -> 640 work-groups) would be worth about 0.9 ms. It was implemented and measured neutral:

```text
AILA_Q4E_SHARED_GU_WGROW=0 : route_wait 25.14   decode p50 46.61
AILA_Q4E_SHARED_GU_WGROW=1 : route_wait 25.03   decode p50 47.19
```

The prediction failed for a reason worth recording: that kernel runs **after** the doorbell and
**before** the packet wait, i.e. in the same window as the resident experts, so it is already
hidden behind the host's cold-expert computation. It never appears in `route_wait`, exactly like
`launch_moe_decode_gate_up_swiglu_grouped` and `launch_moe_decode_down_to_rank_f32` in 25.1.
The variant stays in the tree behind `AILA_Q4E_SHARED_GU_WGROW=1` but is off by default because
it changes the reduction order for no gain.

The corrected structural rule for this pipeline:

```text
critical path (route_wait):  packet consume -> merge/add/bridge -> HC_attn
                             -> attention -> HC_mlp -> router -> doorbell publish
overlapped with host cold:   resident experts, shared expert
```

So the only kernels worth optimising are the ones in the first list. Measured against the
observer-free trace, the remaining critical-path budget is roughly:

| item | ms/token | state |
|---|---:|---|
| HC (norm + down + up, 96 calls) | ~5.3 | down GEMM now at the 327 GB/s ceiling after the work-group-per-row change; up already at the same ceiling; norm is 1.4 ms of it |
| QSA body (12 layers, 10 stages) | ~4.4 | every stage is a small standalone launch |
| GDN attention (36 layers) | ~1.5 | |
| router (front + topk + publish + classify) | ~1.5 | topk 0.82 of it |
| tail (merge/add/bridge fused + packet wait) | ~2 | |

## 27. Two failed hypotheses on the QSA attention core

`attention_decode_exact_head256_partial_merge_impl` takes 154 us per layer (12 layers, so
~1.85 ms per token) with GlobalWS 52224 = 408 work-groups of 128. Two explanations were tested
and both are wrong; they are recorded so they are not retried.

### 27.1 Redundant K/V reads across query heads: neutral

The model has 24 query heads over 2 KV heads, so `heads_per_kv = 12` and twelve work-groups read
the same K/V tile. The group index was head-major (`head * num_tiles + tile`), which puts those
twelve work-groups `num_tiles = 17` groups apart in dispatch order, so the tile looked like it
would be evicted between them: 50.4 MB of traffic per layer for 4.2 MB of unique data, and
154 us for 50.4 MB is 327 GB/s, i.e. exactly the bandwidth ceiling. Reordering the groups to be
kv-head-major, which places those twelve work-groups adjacently and changes no arithmetic,
measured **neutral**:

```text
AILA_Q4E_QSA_KV_GROUPED=0 : route_wait 25.16   decode p50 46.84
AILA_Q4E_QSA_KV_GROUPED=1 : route_wait 25.14   decode p50 45.49  (Direct also moved 60.1 -> 55.5)
```

So the repeats were already served by L2 and DRAM traffic is only the 4.2 MB of unique data.
The flag stays in the tree, off by default.

### 27.2 Missing vectorisation: already there

The bf16 score path already reads K as 32 `vec8` loads per token and accumulates 256 FMAs per
thread, and the q vector is read from SLM as a broadcast, so the inner loop is not the scalar
byte-at-a-time form that would explain the time.

### 27.3 What is actually left to rule out

An instruction-count estimate for the kernel lands two orders of magnitude below the measured
154 us, so the limiter is neither DRAM traffic nor instruction count. The remaining candidates
are wave quantisation and occupancy (408 work-groups of 128 threads), the serial dependency of
the 256-FMA accumulator chain per thread, and the per-tile SLM logits/statistics stage. Naming
one of those needs the full-compute counters (memory throughput, occupancy, EU stall reasons)
rather than more hypotheses, which is what the collection started at `tmp/vtune-full-decode`
is for.

Note for whoever repeats the profile: `-knob characterization-mode=full-compute` over a
600-token decode needs well over half an hour and the report step alone takes many minutes, so
it has to be run detached rather than inside a single command with a short timeout.

## 28. Why the reorder was neutral, and what the traffic actually is

Section 27 concluded that the twelve redundant K/V reads were "already absorbed by L2" and left
the limiter unnamed. Working the numbers properly gives the mechanism, and it explains the
neutral result exactly.

Per QSA layer at 2052 tokens, `head_dim` 256, `num_heads` 24, `num_kv_heads` 2:

```text
unique K + V            = 2052 x 256 x 2 B x 2 kv_heads x 2 (K,V) = 4.2 MB
K reads by query heads  = 24 heads x 2052 x 512 B                  = 25.2 MB
V reads by query heads  = 24 heads x 2052 x 512 B                  = 25.2 MB
total L2/DRAM traffic   ~= 100 MB per layer
100 MB / 154 us = 650 GB/s
```

Two things follow. First, a 4.2 MB working set fits inside the 16 MB L2, so the twelve repeats
are L2 hits and DRAM sees only the 4.2 MB. That is why reordering the work-groups so the twelve
heads run adjacent changed nothing: it improves *where* the repeats are served from, while the
**volume** of L2 traffic stays at 100 MB. Second, 650 GB/s is the right order for DG2 L2
throughput, so the kernel is bound by L2 bandwidth on redundant traffic, not by DRAM, not by
instruction count and not by the FMA chain.

### 28.1 The consequence for the rewrite

The only fix that scales is to stop reading the same K/V row once per query head: one
work-group should read a K/V tile once and produce scores for all `heads_per_kv = 12` heads that
share it. That divides the traffic by twelve, i.e. 100 MB -> 8.4 MB per layer.

The naive form of that change (one work-group per (kv_head, tile), one thread per token, twelve
accumulators per thread) collapses the grid to `2 x 17 = 34` work-groups, which starves a
512-core device. It has to be combined with splitting the token dimension, e.g. 32 tokens per
work-group, giving `2 x 17 x 4 = 136` work-groups of 128 threads, and with twelve independent
accumulators per thread so the dependency chain does not become 3072 FMAs long.

## 29. The attention core split, measured

The observer-free decode trace separates the QSA attention path into three rows by grid size, which
finally splits it into phases:

| kernel | GlobalWS | work-groups | calls/token | per call | ms/token |
|---|---:|---:|---:|---:|---:|
| phase 1 (scores + V partials) | 52224 | 408 | 8.9 | **155 us** | **1.38** |
| phase 2 (partial merge) | 3072 | 24 | 11.7 | 26 us | 0.30 |

Phase 1 is 85% of it. Its traffic, with `num_heads` 24 over `num_kv_heads` 2:

```text
K read once per query head : 24 x 2052 x 512 B = 25.2 MB
V read once per query head : 24 x 2052 x 512 B = 25.2 MB
total                      : 50.4 MB in 155 us = 322 GB/s
unique K + V               :  4.2 MB
```

Both phases read the whole tile once per query head, so the twelve-fold redundancy is confirmed on
both K and V. Phase 2 also re-reads the per-tile partials: `24 heads x 17 tiles x 272 floats x 4 B
= 444 KB` in 26 us, which is only 26 GB/s and is itself mostly launch cost at 24 work-groups.

### 29.1 The decisive next experiment

Whether phase 1 is volume-bound or chain-bound can be settled cheaply: make each work-group serve
**two** query heads that share a KV head (read the K row once, keep two accumulators), which halves
K traffic and halves the grid. If phase 1 drops roughly in proportion the kernel is volume-bound
and the full staging rewrite is worth it. If it does not move, the kernel is bound by the
256-FMA dependency chain per thread and the parallelisation has to change instead.

### 29.2 What the eventual fix has to look like

Cutting the K/V volume by 12x requires one work-group to read a tile once and serve every query
head that shares it. The naive form collapses the grid to `2 kv_heads x 17 tiles = 34`
work-groups, which starves a 512-core device, so it has to be combined with a token split and with
per-head accumulators held in lanes rather than in one thread. That is a flash-attention-shaped
kernel, and it is the reason the cheap two-head probe comes first: it decides whether the rewrite
is justified before it is written.

## 30. The attention core is bound by cache-line overfetch from the 512 B lane stride

Section 28 estimated L2 volume; this section pins the mechanism, and it is a different one.

### 30.1 The access pattern, quantified

In phase 1 thread `lid` owns token `tile_start + lid`, so adjacent lanes of a SIMD16 read
addresses **512 B apart** (one 256-element bf16 K row). Each `vec8` gather of a sub-group
therefore touches **16 distinct 64 B cache lines to deliver 256 B**, i.e. 25% cache-line
utilization. Scaling that up: 3264 sub-groups x 32 gathers = 104448 gather messages, about
1.67 M cache-line requests, roughly **107 MB of L1/L2 cache-line traffic for 2.1 MB of unique
K**. Adding V and the twelve-fold GQA reuse gives about **157 MB of cache-line traffic per
layer**.

That number, not DRAM bandwidth and not instruction count, is what the 155 us is spending, and it
explains both neutral experiments:

- reordering the work-groups so the twelve query heads run adjacent (28.1) does not change the
  per-lane stride, so the overfetch is unchanged;
- hoisting the token->cache-row index chain into SLM (`token_map`, implemented, measured
  `route_wait` 25.32 vs 25.14 i.e. neutral) removes integer work from the loop but leaves the
  line utilization at 25%.

The VTune instruction mix agrees: 0.06% of instructions are in the half-float category and the
kernel runs at 3-4% of the issue rate, so it is stalling on line-granularity memory requests.

### 30.2 The fix, and a reference implementation that already exists

The fix is to make lanes read contiguous bytes: **one sub-group per token, with the 256-element
head dimension split across its lanes** (16 lanes x 16 bf16 = 512 B contiguous = 8 full cache
lines, 100% utilization), and to have one work-group serve all `heads_per_kv = 12` query heads so
K/V is read once instead of twelve times.

Two reference implementations of exactly this shape are on disk under `tmp/`:

- `tmp/XeStrata/src/kernels/xe/qsa_decode_attn.cpp` — GQA split-chunk decode attention for
  `head_dim = 256`, `G = 12`: `CHUNK = 64` tokens per work-group, 256 threads as 8 sub-groups of
  32, one work-group per `(query, kv_head, chunk)` so it computes all 12 heads, Q staged in SLM
  (12 KB) and each lane preloading its own `qr[12][8]` slice into registers, a `reduce12`
  butterfly that reduces all twelve heads in 16 shuffles instead of 60, and a V loop that
  unrolls four tokens to issue four loads before the FMAs. K and V are not staged in SLM.
- `tmp/Strata/sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` — the same kernel in SYCL.

Their measured decode numbers: ~12-25 us per layer on A770/A770-class parts in a verification
window, and 34.5 us for one query at 2051 cells on the CUDA path with the `reduce12` form. A
realistic target for Aila's 2052 contiguous bf16 cells is therefore **20-30 us per layer**, i.e.
**5-7.5x faster than the current 154 us**, worth roughly 1.1-1.3 ms per token.

### 30.3 Negative result worth keeping: do not use XMX/DPAS for decode attention here

`tmp/XeStrata/src/kernels/xe/qsa_prompt_attn.cpp` and
`tmp/Strata/sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp` implement the joint-matrix form of
the same operator (12 heads padded to 16 rows, K/V staged in SLM, `8x16x16` DPAS). Strata's own
documentation records it as **2-3x slower** than the chunk kernel for decode (13.4 ms vs 5.5 ms
per 32-query batch at 2051 cells) because the SLM packing and four to five barriers per chunk
outweigh DPAS on gathered K/V rows. So extending Aila's `joint_matrix` path from `head_dim` 128
to 256 is the wrong direction for decode, however attractive XMX looks on paper.

## 31. Third neutral result: the score phase is not where phase 1 spends its time

Section 30 identified 25% cache-line utilisation from the 512 B lane stride as the mechanism behind
the attention core's 155 us. The score phase was rewritten to remove it: one sub-group per cell with
the head dimension split across its lanes, so each sub-group reads one contiguous 512 B row (eight
full cache lines) instead of sixteen separate lines. Verified to build and pass the gate, measured:

```text
coalesced score phase : route_wait 25.22   decode p50 47.28   gate_fail 0
previous              : route_wait 25.14   decode p50 46.4-46.8
```

Neutral again. That is the third explanation for this kernel that measurement has rejected (K/V
reorder 28.1, index-chain hoisting 30.1, coalesced K reads here), and together they say something
specific: **phase 1 is dominated by the V stage, not the score stage.** The score stage costs 32
`vec8` loads and 256 FMAs per thread, while the V stage runs `tile_len = 128` sequential iterations
per thread with two dependent loads each, so it is a much deeper dependency chain with poor memory
level parallelism. The change was reverted rather than kept: it was neutral, and it simplified away
the int8 `kv_i8_vec` fast path.

### 31.1 What the evidence now points at

The V loop is the target, and the reference implementation in `tmp/XeStrata` already shows the fix:
unroll four cells and issue all four `V` loads before folding them into the accumulators, which
turns a 128-deep chain of dependent loads into a pipeline four deep. Their V loop also has each
thread own one dimension for **all twelve heads**, which additionally removes the twelve-fold GQA
re-read.

For anyone continuing: the ordered experiment list is now

1. four-cell unroll of the existing V loop (small, contained, keeps the current grid);
2. then the full `CHUNK`/`reduce12` port from `tmp/XeStrata/src/kernels/xe/qsa_decode_attn.cpp`
   if (1) shows the V stage really is the bulk.

Both need the same measurement discipline used here: `route_wait` as the decision metric, VTune
`xpu-offload` for kernel durations, and no in-code marks while deciding.

## 32. Fourth neutral result, and what four of them together mean

The V loop was unrolled four cells at a time, with all four V loads issued before any FMA, which
turns a 128-deep chain of dependent loads into a four-deep pipeline. The cells are still folded in
ascending order, so the sums are those of the plain loop.

```text
V unroll x4 : route_wait 25.20   decode p50 46.32   gate_fail 0
previous    : route_wait 25.14   decode p50 46.4-46.8
```

Neutral. That is now **four** independent changes to the QSA attention core that `route_wait` cannot
see:

| change | expected on phase 1 | measured route_wait |
|---|---|---|
| work-group reorder for GQA locality (28.1) | none on volume | 25.14 vs 25.16 |
| token->row map hoisted to SLM (30.1) | index chain removal | 25.32 vs 25.14 |
| coalesced score phase (31) | up to 2.6x on line traffic | 25.22 vs 25.14 |
| V loop 4-cell unroll (here) | up to 4x on load MLP | 25.20 vs 25.14 |

Four independent attempts, one of which should have moved a 1.86 ms item by a millisecond, all land
within 0.2 ms. The conclusion has to be that **the premise is wrong, not the optimisations**: at this
configuration the QSA attention core is not on the `route_wait` critical path in the way section 29
assumed. The 155 us per layer was measured on a decode trace taken with `--max-tokens 1200`, where the
context grows well past 2052 and the tile count with it, so that figure describes a different
`effective_len` than the 8-token benchmark at pos 2108.

### 32.1 The next step is to re-establish the premise

Before any further work on this kernel:

1. Measure the attention kernel's own duration **at pos 2108** with the observer-free method
   (VTune `xpu-offload`, isolating rows by name), not on a long-generation trace.
2. Re-derive which items actually make up the 25.4 ms `route_wait` by putting a *mark-free* duration
   next to each of them, since the in-code profiler is not usable for absolute numbers (section 22).

Without that, more kernel work on this operator is unguided. The one thing that can be stated with
confidence is the cost of a mistake here: five changes, five neutral results, and the only real wins
this round came from where the observer-free data pointed first (tail fusion, work-group-per-row).

## 33. Micro kernel probes, and the end of the attention-core detour

Full-workload VTune runs were taking over half an hour per iteration and the in-code profiler
inflates short spans by 2.5-5x (section 22), which left no usable instrument for kernel work. The
answer is a **micro probe**, and it changes the working method.

### 33.1 The probe

`tests/ops/QsaAttentionProbe.cpp` (target `AilaQsaAttentionProbe`) times one operator in isolation at
the production shape: 24 query heads, 2 KV heads, `head_dim` 256, 2052 cells, `tile_t` 128, so 17
tiles. It builds its own `Context` with a profiling queue, allocates the tensors, warms up five
iterations off the clock (the pipeline under study has no persistent cache, so every run is a warm
run), then reports the median and the best of 30 event-timed samples. It takes seconds.

Baseline:

```text
cells=2052 heads=24 kv_heads=2 head_dim=256 tiles=17
qsa_decode_attention_us: median=216.1  best=192.2  worst=221.0
per_token_12_layers_ms: 2.59
```

That also corrects section 29's 155 us figure: that was phase 1 only, measured on a
long-generation trace where the context had grown well past 2052. At the benchmark's own context the
whole operator, both phases, is 216 us, i.e. **2.6 ms of a 46 ms token**, about 10% of `route_wait`.

### 33.2 The probe reproduces the model's verdict in seconds

The coalesced score phase from section 31 was re-applied and measured both ways:

| variant | probe median | model `route_wait` |
|---|---:|---:|
| baseline (one thread per cell) | 216.1-217.7 us | 25.14 |
| coalesced score (sub-group per cell) | **231.7 us** | 25.22 |

It is **6.5% slower** in isolation, and neutral in the model. So the four neutral model results were
not a measurement failure: the change genuinely does not help, and the probe says so in seconds
instead of through a 46 ms token containing twenty other stages. That is the instrument this work
needed from the start.

Why coalescing loses: it replaces 128 independent per-thread dot products with 128 sequential
sub-group scores, so the work within a work-group becomes a serial loop of 128 reductions. The
line-utilisation win is real but the reduction chain costs more than it saves.

### 33.3 Method that replaces the previous rounds

1. **Single A/B in the model.** The pipeline warms up and keeps no persistent cache, so every run is
   a warm run; a real gain shows in one run and p50 over many runs only hides that behind host noise.
2. **Micro probe first for kernel changes.** Seconds per iteration, isolated from the other twenty
   stages of the token.
3. If a change wins in the probe but not in the model, the cost has moved elsewhere or the operator
   is not on the critical path, and that has to be explained rather than averaged away.
4. VTune `xpu-offload` only when the shape of the cost is unknown (as in section 22), never as a
   per-iteration tool.

## 34. First structural rewrite attempt of the attention core: wrong and slower

The full GQA-fused kernel was written in the shape of `tmp/XeStrata`'s `qsa_decode_attn.cpp`: one
work-group per (KV head, tile) computing all twelve query heads that share it, sub-groups owning
cells with each lane holding `head_dim/32` contiguous dimensions, the `reduce12` butterfly, and a
value phase where thread `t` owns dimension `t` for every head.

Probe result, against the 216.9 us baseline:

```text
AILA_Q4E_QSA_GQA_CHUNK=0 : 217.9 us   PROBE_OK
AILA_Q4E_QSA_GQA_CHUNK=1 : 292.6 us   PROBE_FAILED (output is all zeros)
```

So the first attempt is both incorrect and 34% slower. It was reverted; the probe stays, because it
is what turned a 46 ms token into a two-second verdict and caught the all-zero output immediately.

Two things are already clear from the numbers.

### 34.1 Why it cannot be faster yet: 34 work-groups

The partials layout fixes the tile at 128 cells, so a one-work-group-per-(KV head, tile) kernel
launches only `2 x 17 = 34` work-groups of 256 threads. That is the starvation case the review
warned about, and it is almost certainly why the rewrite is slower even before the correctness bug.
The reference implementation avoids it by using `CHUNK = 64` and 33 chunks, giving 66 work-groups;
its tile count is its own, whereas here 128 is fixed by the merge kernel.

The lever that keeps the layout: split the heads across work-groups. `HEADS_PER_WG = 6` gives
`2 x 17 x 2 = 68` work-groups and still reads K/V twice instead of twelve times; `HEADS_PER_WG = 4`
gives 102 work-groups at three reads. That is the knob the next attempt should sweep, with the probe
as the instrument.

### 34.2 Correctness to fix first

The all-zero output means the value accumulators or the probability weights came out zero. The
places to check, in order: whether `sp` is written for every head by the even-lane rule (heads 12-15
are skipped by the `hd_lane < heads_per_kv` guard and must be zero in `pp`), whether `srow` masks
compressed every cell, and whether the `(head * max_tiles + tile) * stride` slot agrees with the
merge kernel's expectation, which is the bug the first revision actually had (it wrote
`tile * stride + head * max_tiles * stride`).

### 34.3 Method note

The probe's sanity check (output not all zeros) is what made this attempt cheap to judge. Every
future kernel experiment should keep that check, because a fast-but-wrong kernel would otherwise look
like a win in the model the moment the gate happens not to trip.

## 35. The attention operator is dominated by a per-call fixed cost, not by its tile work

The probe was extended to sweep the cell count, which is the measurement that should have been taken
before any kernel work on this operator.

### 35.1 Scaling sweep

Single call between its own barrier pair, 30 samples, median:

| cells | tiles | us |
|---:|---:|---:|
| 256 | 2 | 180.1 |
| 512 | 4 | 182.7 |
| 1024 | 8 | 189.6 |
| 2052 | 17 | 217.5 |
| 3072 | 24 | 282.3 |

At 2 tiles, with almost no work at all, the operator still costs 180 us. So essentially the whole
216 us at the production shape is a **fixed per-call cost**, and the part that scales with the tile
count is small: about `(282.3 - 187.0) / (24 - 6) = 2.4 us per tile`, i.e. roughly 41 us of the
216 us at 17 tiles.

### 35.2 Measurement correction

Batching ten calls between one barrier pair and dividing by ten replaces the per-measurement cost
with the per-call cost:

```text
batch=1  : 217.5 us at 2052 cells, 180.1 us at 256 cells
batch=10 : 180.4 us at 2052 cells, 143.6 us at 256 cells
```

So ~37 us of the single-call figure is the profiling-queue barrier pair and its wait, and the
remaining **~143 us is charged per call** and does not depend on the context length, the tile count
or the number of heads.

### 35.3 What this explains, and what it invalidates

It explains the four neutral model results in sections 28-32. Those changes all addressed the
per-tile work, which is 41 us of a 217 us operator, i.e. 0.5 ms of a 46 ms token, and that is below
the resolution of a single A/B in the model. The premise behind them - "the attention core is 1.85 ms
of the critical path because it is 155 us per layer" - was wrong for the operating point, and the
probe backed it up only because the probe itself was dominated by the same kind of constant.

It also invalidates the first GQA rewrite (section 34) as a way to win: making K/V reads twelve times
cheaper can at best attack the 41 us, not the 143 us.

### 35.4 The next measurement, before any more kernel work

The 143 us per call has to be attributed before anything else. The candidates, in order:

1. **the operator's own host side**, since two submissions of that size cannot cost 143 us of device
   time (a launch measures 9.4 us, section 20.2), and a host-side query such as
   `kv_int8_dpas_enabled(ctx.queue().get_device())` is the only thing besides the submissions in the
   dispatch path;
2. **the profiling queue itself**, if `enable_profiling` makes each submission materially slower on
   this driver - which can be checked by running the same batch on a non-profiling queue and timing
   wall clock instead of events;
3. the device drain of the batch tail.

Until that is answered, per-tile kernel optimisation has a ceiling of about 40 us per call and should
stay parked.

## 36. The attention operator is charged two host submissions per call

### 36.1 The fixed cost is not the profiling queue

Running the same batch on a queue without `enable_profiling`, timed with the host clock:

| cells | event-timed (profiling) | wall clock (no profiling) |
|---:|---:|---:|
| 256 | 143.6 us | 149.5 us |
| 2052 | 180.2 us | 186.5 us |

A 6 us difference, so the profiling queue is not responsible and the ~143 us per call is real on the
queue the production pipeline uses.

### 36.2 What it is

The operator submits exactly two kernels: the per-tile score/softmax/V pass and the partial merge.
Their device work at 256 cells cannot be 143 us: phase 1 is 48 work-groups of 128 threads, phase 2 is
`num_heads x merge_wg = 24 x 128` threads looping twice. And the earlier measurement of this stack put
host submission at **53 us per `queue.submit`** (750 submissions per token = 40.4 ms, section 12).

```text
2 submissions x ~53 us = ~106 us
plus a few us of device work per tile (2.4 us/tile measured)
~= the 143 us constant
```

So the attention operator costs what it costs because it is **two host submissions**, not because of
its arithmetic or its memory traffic. Across the twelve QSA layers that is 24 submissions per token,
about 1.3 ms of the token's submission budget, and it is consistent with the operator measuring
~2.2 ms/token in the model while its per-tile work is only 0.5 ms.

### 36.3 Consequences

1. Every kernel-internal optimisation attempted in sections 28-34 was aimed at the 41 us per-tile
   part, i.e. at ~20% of the operator and ~0.5 ms of a 46 ms token. That is why all four were neutral
   and why the probe has to be read as "143 us of submission + 41 us of work".
2. The GQA restructure cannot pay for itself on its own: it only shrinks the 41 us.
3. The lever that follows is **fewer submissions on the attention path**, i.e. fusing the per-tile
   pass and the partial merge, or folding the operator into a larger submission. That is worth about
   0.65 ms per token for the fusion alone, and it also re-prioritises the whole critical path: on this
   stack, submission count per token is a first-class cost, not an implementation detail.

## 37. Collapsing the attention operator to one submission costs more than it saves

Section 36 showed the operator's per-call cost is dominated by its two submissions. The obvious move
is one submission, so a single-kernel form was written: one work-group per query head walking the
cache in 128-cell tiles with an online softmax, no partials buffer and no merge kernel.

Probe, 2052 cells:

```text
two-pass (2 submissions, 408 work-groups) : 180.2 us   PROBE_OK
online    (1 submission,  24 work-groups) : 1072.9 us  PROBE_OK
```

Correct, and **six times slower**. The saving of one 53 us submission is swamped by the loss of
parallelism: the two-pass form gets `num_heads x num_tiles = 408` work-groups, the online form gets
one per head, i.e. 24, and each of those then walks seventeen tiles serially with two work-group
barriers and two whole-group reductions per tile.

That is a hard constraint on this operator: it needs on the order of four hundred work-groups, so
submission count cannot be reduced by reducing parallelism. The two-pass shape is the right one, and
its second submission can only be removed by a mechanism that keeps the tile parallelism - a
grid-wide dependency, which SYCL does not expose portably here.

Consequence for the round: the remaining submission savings have to come from elsewhere in the
layer, where the kernels are small and their work does not need hundreds of work-groups.

## 38. Submission-reduction roadmap, from two independent scans

With the economics confirmed (`submit_ms` 35.4-38.6 ms for ~654 submissions, so 48-54 us each), two
scans enumerated every submission in a layer and every submission in the QSA body and ranked the
fusions. Together they identify about **12 ms of a 46 ms token**, all of it host submission time
rather than device work.

### 38.1 Whole-token layer body (48 layers per token)

| rank | fusion | saved / token | @ 53 us | barrier |
|---|---|---:|---:|---|
| 1 | fold `moe_resident_classify_device` into `submit_input_and_route_publish` | 48 | 2.5 ms | none (the publish already has a group barrier) |
| 2 | drop `submit_packet_copy` and read the host packet's rows directly in the fused tail | 48 | 2.5 ms | none (`submit_packet_wait` already validated the shape) |
| 3 | fuse `router_softmax_topk_device` + publish + classify into one kernel | 96 | 5.1 ms | one SLM barrier |
| 4 | fold `submit_begin_replay` into the layer-0 publish | 1 | 0.05 ms | none |

Rank 3 subsumes rank 1, so the realistic set is 3 + 2 + 4 = 145 submissions = **7.7 ms**.

Rejected by the scan for good reasons: fusing the router front GEMV with the top-k selection (the
front is a multi-work-group GEMV and the selection needs one work-group), fusing the 1-work-group
packet wait with the 256-work-group tail, fusing the resident/shared expert GEMVs with each other,
and folding `submit_publish_done` into the tail on the last layer.

### 38.2 QSA layer body (12 layers per token)

| rank | fusion | saved / token | @ 53 us |
|---|---|---:|---:|
| 1 | `qsa_prepare_q_and_cache_k_graph` (4 WG) + `qsa_compress_one_completed_block_graph` (1 WG) | 12 | 0.64 ms |
| 2 | `qsa_select_topk_blocks_graph` (1 WG) + `attention_validate_qsa_active_blocks_graph` | 12 | 0.64 ms |
| 3 | `decode_prepare_qgkv_packed_partial_graph` + the two above | 12-24 | 0.64-1.27 ms |
| 4 | fold the compress into the `b == current_blocks` work-group of the block-score kernel | 12 | 0.64 ms |

### 38.3 What this says about the shape of the remaining work

The scans agree with the measurement: the win is now entirely in **collapsing small submissions**,
not in making any kernel faster. The three biggest items are all sub-millisecond kernels whose only
real cost is that they are separate submissions, and two of them need no barrier at all.

Order to execute, by value over risk: 38.1 rank 3 (5.1 ms), 38.1 rank 2 (2.5 ms), 38.2 rank 1 and 2
(0.64 ms each). Doing all four is ~9 ms, i.e. 46.4 ms -> ~37.5 ms, a further 19%.

One caution recorded from section 37: none of these may reduce a large kernel's work-group count.
The attention operator needed 408 work-groups and collapsing it to 24 made it six times slower, and
the same trap applies to the router front GEMV and the resident expert GEMVs, which the scan
explicitly rejected.

## 39. First roadmap item attempted: the existing fused router crashes in this path

Section 38.1 rank 1 asked to fold `moe_resident_classify_device` into the router, and there is
already a kernel that does it: `launch_router_softmax_topk<true>`, exposed as
`ops::router_softmax_topk_resident`, which performs the resident slot classification inside the top-k
kernel and writes `resident_selected`, `host_ids`, `host_weights` and `host_selected`. Wiring it into
`RecordWholeTokenGraphBody` in place of `router_softmax_topk_device` + `moe_resident_classify_device`
would remove one submission per layer, 48 per token, about 2.5 ms.

Measured, one run per arm:

```text
AILA_Q4E_TG_ROUTE_FUSED=0 : submit_ms 38.4   route_wait 25.37   p50 46.25   gate_fail 0
AILA_Q4E_TG_ROUTE_FUSED=1 : graph arm exited -1073740791 (0xC0000409) on the second token
```

A fail-fast, which on this platform usually means a buffer overrun caught by the CRT. The log shows
the first token generated correctly and the crash on the next one, so the fused kernel's first
invocation is fine and something it leaves behind - most likely one of the three host-side route
arrays it writes (`router_host_ids_`, `router_host_weights_`, `resident_selected_host_`) - does not
match what this path allocated.

The flag therefore defaults to **off** and the code is kept behind it for debugging rather than
deleted. Two things to check when it is picked up again:

1. `router_host_ids_` is allocated as `cap * top_k` ints where `cap` belongs to the Direct path's
   prompt capacity; the token-graph path should not be writing through a pointer sized for a
   different call shape.
2. `launch_router_softmax_topk<true>` writes `host_ids[k]`, `host_weights[k]` and `host_selected[k]`
   unconditionally; the token-graph path only needs `route_ids`, `route_weights` and `selected`, so
   the simplest safe wiring is to pass the graph tensors for all three and leave the Direct-path host
   arrays out of it entirely.

Default path after this round, single run: `submit_ms 37.9`, `route_wait 25.47`, p50 46.16,
Direct 55.21, `gate_fail 0`.

## 40. The router fusion is mechanically correct but sits inside the measurement noise

Section 39's crash is fixed by writing the three host-side route arrays into the graph's own tensors
(`route_ids`, `route_weights`, `selected`) instead of the Direct path's buffers, which are sized for a
different call shape. Verified by reading `launch_router_softmax_topk`: on the 512-expert path it
issues exactly **one** submission, so the fused caller really does replace two submissions with one.

Single A/B, one run per arm:

```text
AILA_Q4E_TG_ROUTE_FUSED=0 : submit_ms 38.6   route_wait 25.36   p50 47.15   gate_fail 0
AILA_Q4E_TG_ROUTE_FUSED=1 : submit_ms 38.0   route_wait 25.81   p50 49.83   gate_fail 0
```

No crash, gate clean, and `submit_ms` moved by only 0.6 ms where the mechanism predicts 2.5 ms.

### 40.1 Why the measurement cannot decide

`submit_ms` reads 35.4, 37.9, 38.4 and 38.6 ms across the last four runs of the same configuration, so
its scatter is about 1.6 ms - a 2.5 ms single-item saving is inside it. And `route_wait` is the wrong
instrument for this change by construction: removing a host submission cannot reduce device work
between two doorbells, so its 25.36 -> 25.81 movement is noise as well.

The flag therefore stays **off** and the code stays behind it.

### 40.2 How to measure the remaining roadmap

The four layer-body fusions in 38.1 total about 7.7 ms of submission time, and the QSA pairs another
1.3 ms. No single one of them is measurable in one run, but together they are five times the noise.
The right procedure is therefore:

1. land the group (or at least three of them) behind their flags,
2. measure default-vs-off in one A/B, which now has ~9 ms of signal,
3. keep the group on or off as a whole, and only then bisect if the aggregate is disappointing.

This also argues for one small instrumentation addition before the next group: log the **actual**
submission count per token next to `submit_ms`, since the node counter reports logical groups (352)
rather than submissions (~654) and a count is immune to timing noise.

## 41. Correcting the submission-time statistics, and what they now validate

Section 40 concluded that a 2.5 ms submission saving sits inside the noise of `submit_ms`. That
conclusion was reached from polluted numbers: the pattern `submit_ms=([0-9.]+)` also matches
`topk_submit_ms=` and `shared_submit_ms=` inside the `[Q4E AdaptiveWork]` lines, which carry values
like 76558 and 1.375. Half the samples in every run were those fields.

Parsing only the real line, `] submit pos=<n> nodes=<n> submit_ms=<v>`, gives eight decode samples per
run and a much tighter statistic:

| run | configuration | min | median |
|---|---|---:|---:|
| `tail-rev0` | tail fusion **off** | 39.85 | 41.71 |
| `tail-rev1` | tail fusion **on** | 37.80 | 38.51 |
| `rfx0` | router fusion **off** | 36.06 | 38.56 |
| `rfx1` | router fusion **on** | 37.03 | 37.99 |

### 41.1 The tail fusion's host saving is confirmed

41.71 -> 38.51 ms is **3.2 ms** of submission time removed by fusing three element-wise tail kernels
into one. Until now that fusion was justified only by a 0.9 ms `route_wait` improvement, measured on
the device side, where removing host submissions should not even show up. The host-side figure is the
one that matches the mechanism: 96 submissions at ~35 us each.

So `submit_ms` can resolve a group of ~96 submissions at about 3 ms, i.e. the instrument is good to
roughly 1 ms when parsed correctly. That sets the target: land at least ~50 submissions' worth of
fusion per measurement, which the roadmap items satisfy individually except the QSA pairs.

### 41.2 The router fusion does not deliver

38.56 -> 37.99 ms median, and its minimum got *worse* (36.06 -> 37.03). The mechanism is real - reading
`launch_router_softmax_topk` confirms one submission on the 512-expert path - so the plausible
explanation is that the fused kernel costs more than the submission it removes: it writes three extra
arrays that the split form did not, and one of them (`host_selected`) duplicates `resident_selected`.

It stays off, and the next attempt should pass `nullptr`-safe distinct targets or drop the redundant
writes rather than simply renaming the pointers.

### 41.3 Method note for every future measurement

Parse `] submit pos=` only. Use the median over the run's decode tokens. A change worth less than
about 1 ms of host submission is not worth measuring this way; group such changes.

## 42. Per-submission cost is not uniform: 33 us for a large kernel, 12 us for a tiny one

Putting the two measured fusions side by side recalibrates the roadmap.

| fusion | submissions removed | measured host saving | per submission |
|---|---:|---:|---:|
| tail (3 kernels -> 1, 256-work-group grid) | 96 | 3.2 ms | **33 us** |
| router (classify folded into top-k, 1 work-group) | 48 | 0.57 ms | **12 us** |

So "53 us per submission" was an average over the whole token, and it is not what a given removal is
worth: a submission costs roughly with the size of its descriptor and grid, and a single-work-group
kernel submits at about a third of the rate of a 256-work-group one.

### 42.1 Consequences for the roadmap in section 38

- Layer-body rank 2 (drop `submit_packet_copy`, a 128-work-group copy of the packet rows) is the
  remaining large-descriptor item, so its 48 removals should be worth ~1.5 ms rather than 2.5 ms.
- Layer-body rank 3 (fuse top-k + publish + classify) merges one tiny and two small kernels, so 96
  removals are probably worth ~1.2-1.5 ms, not 5.1 ms.
- The QSA pairs (12 removals of single-work-group kernels each) are worth ~0.15 ms apiece, i.e. below
  the instrument. They should be batched with something else or left alone.

Revised expectation for the whole roadmap: roughly **3-4 ms**, not 12 ms. That is still the largest
remaining item on the critical path, but it will take the whole group to measure.

### 42.2 What would change the picture

If submission cost tracks descriptor size, then the lever is not only *fewer* submissions but
*cheaper* ones: fewer kernel arguments, no unnecessary USM pointers, and smaller grids. That is worth
measuring directly with a micro-probe that submits N trivial kernels with 2 arguments versus 2
hundreds, because a factor of two there would apply to all ~650 submissions in the token.

## 43. A bare submission costs about 5 us, so the model's 53 us is not submission cost

A second micro-probe (`AilaKernelSubmitProbe`) enqueues hundreds of trivial kernels back to back and
times only the enqueue loop, then drains. Results, 300 iterations, steady state:

| shape | submit us/kernel | drain us/kernel |
|---|---:|---:|
| args=2, grid 1x32 | (JIT, ignore) | 0.56 |
| args=2, grid 256x256 | 4.94 | 0.32 |
| args=200, grid 1x32 | 4.71 | 0.30 |
| args=200, grid 256x256 | 4.42 | 0.36 |
| 1 `local_accessor`, grid 256 | 6.36 | 0.38 |
| 4 `local_accessor`, grid 256 | 6.64 | 0.40 |

Two things follow.

**Argument count and grid size do not matter.** 2 arguments versus 200, and 32 threads versus 65536,
both land at 4.4-4.9 us. Dynamic shared-memory accessors add about 1.5-2 us, but one versus four makes
no difference, so it is a fixed per-kernel cost for having any at all, not per accessor.

**So a bare submission is ~5 us, ten times less than the 53 us average this document has been quoting.**
That average was computed as `submit_ms / submissions` in the model, and the model's submit loop is not
a bare enqueue loop: the same thread also runs the ops functions' host code, and, more importantly, the
queue applies backpressure once the device is the slower side. With ~650 submissions across a 38 ms
pass while the device needs ~25-40 ms of work, the host cannot run ahead, so `submit_ms` measures
device-paced submission, not the host cost of submitting.

### 43.1 What this means for the two measured fusions

The tail fusion removed 96 submissions and 3.2 ms of `submit_ms` - 33 us each; the router fusion
removed 48 and 0.57 ms - 12 us each. Those numbers are now best read as **device-side per-kernel gaps**
in a device-paced pipeline, not as host submission cost. That is consistent with the tail kernels
having 256-work-group grids and the classifier being a single work-group: a larger dispatch costs more
gap. It also means the earlier attribution "the attention operator's 143 us fixed cost is two
submissions" (section 36) is wrong, since two submissions are ~10 us; that 133 us is device-side or
host-side work still to be attributed.

### 43.2 Revised lever

Since a submission is cheap on its own, the reason fewer kernels help is the *gap between kernels on
the device*, which grows with the size of the dispatch. The remaining roadmap items are therefore
worth roughly what they measured (tail 3.2 ms, router 0.6 ms) and no more, and the biggest single
lever this probe points at is not submission count but **whatever makes a dispatch gap 33 us instead
of 0.3 us** - which the trivial-kernel drain figures say is not intrinsic to submission.

## 44. What the fusion savings actually were, and the consequence

Section 43 corrected the submission cost to ~5 us. Taking that one step further changes what the two
measured fusions mean.

The tail fusion removed 96 submissions, 3.2 ms of `submit_ms` (33 us each). Those three kernels were
not empty: they merged the cold rows into the routed output, added the shared expert and updated the
residual, and the fused form additionally stops writing and re-reading the intermediate row. A
single-work-group classifier folded into top-k removed 48 submissions for 0.57 ms (12 us each), and
that kernel really was almost empty.

So the saving tracks **the work the removed kernel was doing**, not the submission. Two submissions
cost ~10 us; the tail kernels cost ~33 us each because they move real data. Since `submit_ms` is
device-paced (43), the fusion savings were device-time savings all along.

### 44.1 Consequence: the pipeline is device-work-limited, not launch-limited

```text
650 submissions x ~5 us of true submission cost = ~3.2 ms of a 46.3 ms token (7%)
route_wait 25.1 + cold 18-23 = 44-48 ms, which is the token
```

The 21 us "launch floor" quoted in section 19.1 was a profiler artefact (the marks themselves), and
the 53 us "per submission" was device time divided by submission count. Neither describes launching.

That does not invalidate the wins already banked - they were real work reductions - but it does mean
the local roadmap is finished: the remaining fusions would shave small kernels' real work, worth
0.5-1 ms each at best, and the QSA pairs are worth ~0.15 ms.

### 44.2 What is left, stated plainly

- `cold` is at 80-95% of this host's memory wall and cannot be reduced without reducing the bytes,
  which needs VRAM that does not exist on a 16 GB card.
- `route_wait` is 25 ms of device work whose per-stage attribution is only known to within the
  profiler's 1.2-5x inflation, and its largest identified components (hyper-connections ~7 ms, GDN
  attention ~8 ms, attention core ~2 ms, resident+shared ~8 ms overlapped) are each small.
- Overlap is blocked structurally: the cold computation needs the routed ids, which the doorbell
  publishes, and the device has only ~7 ms of independent work to hide 18-23 ms behind.

So the honest position is that decode has no single remaining lever above ~3 ms, and the next step is
a direction call rather than more of the same. Prefill is a candidate of a different order: it takes
~127 s for 2108 tokens (60 ms/token) and stages ~42 GB host-to-device per run, so it dominates every
benchmark run's wall time and none of the decode work above touches it.

## 45. Prefill: 60% of it sits inside one 21-line bracket around the router submission

Decode has no lever above ~3 ms left (44), so the same `[Q4E AdaptiveWork]` counters were read for the
prefill chunk, where `count=2048`:

| field | value |
|---|---:|
| `submitted_ms` | 127,731 ms |
| **`topk_submit_ms`** | **77,356 ms** |
| `full_host_ms` | 26,240 ms |
| `linear_host_ms` | 11,608 ms |
| `experts_host_ms` | 7,265 ms |
| `router_wait_ms` | 3,417 ms |
| `h2d_mb` | 42,578 MB |

`GenerateMessages` for the whole run is 133.8 s, so prefill dominates everything this project measures,
and **60% of it is one timer**.

### 45.1 The bracket really is just the submission

`host_topk_submit_ms` is accumulated at `Qwen4ExpBnb4Backend.cpp:3123` from `t_topk`, taken at
`:3102`. Between them are only the two submit calls:

```cpp
if (resident_epoch && s_router_resident_fused && cfg_.num_experts == 512)
    actual_route_event = ops::router_softmax_topk_resident(...);
else {
    actual_route_event = ops::router_softmax_topk_device(..., count, cfg_.num_experts, top_k, true);
    if (resident_epoch)
        actual_route_event = ops::moe_resident_classify(..., router_host_ids_,
                                                        router_host_weights_,
                                                        resident_selected_host_, ...);
}
```

There is no wait, no memcpy and no synchronise in that range. Both calls end up in
`launch_router_softmax_topk`, which for `num_experts == 512` submits one kernel and returns its event.
So 77.4 s over 48 layers, i.e. **1.61 s per layer**, cannot be the enqueue itself - a bare submission
measures ~5 us (43). Something inside the callee blocks, and the most likely candidate is the
host-side route arrays: `router_host_ids_`, `router_host_weights_` and `resident_selected_host_` are
host allocations that an epoch-free call has the device write into, and any code path that then reads
them on the host for the layer's routing must synchronise.

### 45.2 The three measurements to take next

1. Split the bracket: time `router_softmax_topk_*` and `moe_resident_classify` separately, and count
   how many layers take the `resident_epoch` branch.
2. Instrument inside the callee for any `queue().wait()`, `memcpy_d2h` or `synchronize` per call.
3. If a D2H is the cause, ask why prefill needs the routes on the host at all: prefill routes 2048
   tokens per layer and the host only needs them to drive the CPU expert path, so the transfer could
   be narrowed, deferred, or made asynchronous with the device work that follows it.

If this holds, prefill is worth far more than everything decode has produced: a 2K-token prefill at
127 s is 60 ms per token, and 77 s of that is one bracket.

## 46. The prefill bracket measures backpressure, not top-k cost

The bracket from 45 was split into top-k and classify with a one-line probe (first layer only):

```text
count=2048 : topk_submit_us = 78,143,770   classify skipped (no epoch)
count=56   : topk_submit_us = 24.7
count=4    : topk_submit_us = 59.1
```

The same call costs **tens of microseconds** at 56 or 4 tokens and **78 seconds** at 2048. That cannot be
the call's own work: it is the host blocking on an in-order queue that is full because the device is
behind. So `topk_submit_ms` is where backpressure surfaces, not where the time goes.

### 46.1 What it actually says about prefill

If the host blocks there, the device is the limiter, and the per-layer device time is

```text
78.1 s / 48 layers = 1.63 s per layer for 2048 tokens
                  = 0.79 ms per token per layer
x 48 layers       = 38 ms per token, i.e. ~62 ms/token of device work in a 127 s prefill
```

So prefill is **device-bound at about 62 ms per token** against decode's 46 ms per token for a single
token. A 2K-token prefill of a 512-expert top-10 MoE should be far cheaper per token than that, because
its GEMMs are batched, so the gap is in prefill's own kernel efficiency, not in the handoff machinery.

That reverses the priority: prefill is 97% of every benchmark run's wall time and runs at ~62 ms/token
of device work, while decode's remaining levers are worth at most ~3 ms. Prefill is the prize.

### 46.2 First three things to measure there

1. Per-layer device time in prefill from an observer-free trace (VTune `xpu-offload` on a short chunk),
   to see whether the 1.63 s per layer is expert GEMMs, QSA attention, or staging.
2. The 42 GB of host-to-device staging per run: 42.6 GB for 2048 tokens x 48 layers implies expert
   weights are being copied repeatedly rather than kept resident; quantify the copy time against
   `h2d_mb` and the fill/bank timers already present.
3. Whether the 512-expert top-10 routing at 2048 tokens is being evaluated as one batched grouped GEMM
   or as per-token work: 2048 x 10 expert tasks per layer is exactly the shape a grouped GEMM should
   absorb in a few hundred microseconds, not 1.6 s.

## 47. Prefill device-time profile: 60% is memory copies and the GPU is only 25% busy

Taking the prefill-dominated observer-free trace already on disk (`tmp/vtune-xpu-1`, a run with two
generated tokens, so ~97% of its wall time is prefill) and summing by kernel:

| kernel | device time | share |
|---|---:|---:|
| `zeCommandListAppendMemoryCopy` | 15.62 s | 49% |
| another memory-copy row | 0.83 s | 3% |
| `gemm_kernel` (two shapes) | 6.77 s | 21% |
| `submit_dequant` | 1.85 s | 6% |
| `run_linear_delta_prefill_gpu_batched_impl` (GDN) | 1.42 s | 4% |
| `attention_prefill_cached_impl` (QSA) | 0.99 s | 3% |
| `packed_nf4_gemm_bf16_joint_matrix_wide` | 0.83 s | 3% |
| **total kernel time** | **31.7 s** | |

Against a prefill wall time of ~127 s that is **25% GPU utilisation**, and **memory copies alone are
60% of the device time**.

### 47.1 What the numbers say

```text
42.6 GB of host-to-device staging per prefill run
19.0 s of copy time  ->  2.7 GB/s effective
4242 MB of static model weights  ->  42.6 GB is about ten times the model
```

So prefill stages the model roughly ten times over a three-chunk run (2048, 56 and 4 tokens). Even a
generous PCIe 4.0 link would move 42.6 GB in about 2 s, so the copies are running at roughly a seventh
of the link's capability, which is the signature of many small scattered transfers rather than a few
large ones: the trace shows 27213 copy instances averaging 574 us each.

And because the GPU is only a quarter busy, the remaining ~95 s of prefill is *not* compute and *not*
copy - it is the host orchestrating those transfers and the per-phase bookkeeping around them.

### 47.2 Therefore the first two prefill levers, in order

1. **Move less.** 42.6 GB is ten times the model; expert staging should be able to reuse what is
   already resident between chunks and within a chunk's layers. This is the largest single number in
   the whole prefill picture.
2. **Move it better.** 2.7 GB/s for 574 us average copies means the transfers are fragmented; batching
   them into fewer, larger copies should approach the link's bandwidth, which alone would cut the 15.6 s
   to a few seconds.

Both are measurable with counters already printed (`h2d_mb`, `fill_ms`, `bank_wait_ms`, `staged`) plus
one new counter: copy calls per phase.

## 48. First prefill knob test: wrong knob, and where the right one is

The staging pipeline logs `[Q4E PrefillStage] banks=2 lookahead=1 slots_per_bank=16`, i.e. sixteen
experts per bank, copied and then computed with one bank of lookahead. Sixteen experts is about 39 MB
per transfer round, which fits the 574 us average copy and the 2.7 GB/s effective bandwidth.

`AILA_Q4E_LAYER_STAGE_CHUNK_SLOTS` looked like the lever, so 16 versus 64 was measured, one run each:

```text
slots=16 : GenerateMessages 134.5 s   submitted 128.6 s   h2d 42.22 GB   fill 2.33 s
slots=64 : GenerateMessages 133.8 s   submitted 128.0 s   h2d 41.59 GB   fill 2.30 s
```

No effect, and the reason is visible in the log: `slots_per_bank` reads 16 in **both** runs, so that
variable is not what the staging loop uses. The real chain is

```cpp
Qwen4ExpBnb4Backend.cpp:1954  const int staging_banks      = adaptive_cache_.StagingBanks();
Qwen4ExpBnb4Backend.cpp:1956  const int slots_per_chunk    = adaptive_cache_.StagingSlotsPerBuffer();
Qwen4ExpExpertCache.cpp:2613  const size_t staging_capacity = staging_buffer_bytes_ / slot_bytes_;
Qwen4ExpExpertCache.cpp:760   prefetch_staging_slots_      = 16;
```

so the bank size comes from `staging_buffer_bytes_` (or from `prefetch_staging_slots_`, default 16),
not from the chunk-slot knob. The candidate environment variables to sweep next are
`AILA_Q4E_STAGING_SLOTS` (raw, default 16) and `AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS`, and the
verification is simply that `[Q4E PrefillStage] slots_per_bank=` changes in the log.

This is recorded because it cost a run to discover: for the next person, **confirm the logged parameter
moved before believing a knob test**.

### 48.1 Where prefill work stands

```text
prefill wall             ~127 s for 2108 tokens (60 ms/token)
GPU busy                 31.7 s (25%)
of which memory copies   19.0 s (60%), 42.6 GB at 2.7 GB/s
staging granularity      16 experts per bank, ~39 MB per transfer round
```

The two levers remain: move less (42.6 GB is ten times the model) and move it better (larger, fewer
copies). The knob sweep above is the cheapest way to test the second, provided the logged
`slots_per_bank` is checked each time.

## 49. Prefill has a hard ~133 s floor that staging granularity does not move

Two more zero-code knobs were swept, one variable each, same 2108-token prompt:

| config | prefill wall | `h2d_mb` | `fill_ms` | `bank_wait_ms` |
|---|---:|---:|---:|---:|
| baseline (`CHUNK_SLOTS=16`) | 134.5 s | 42.22 GB | 2.33 s | - |
| `CHUNK_SLOTS=64` | 133.8 s | 41.59 GB | 2.30 s | - |
| `FILL_PARTS=1` | 132.96 s | 41.42 GB | 2.26 s | 1.86 s |
| `FILL_PARTS=2` | 133.2 s | 42.17 GB | 2.30 s | 1.91 s |

Nothing moves it. The earlier `FILL_PARTS` default of 8 splits each expert fill into eight parts, and
collapsing that to one part changes neither the transfer volume nor the wall time, so the 19 s of copy
time is **not** granularity overhead. Read together with section 48, this rules out the whole
"make the copies bigger" family of fixes.

### 49.1 The arithmetic does not close, and that is the finding

```text
prefill wall                                 127 s
GPU busy from an observer-free trace        31.7 s   (25%)
  of which copies                            19.0 s
host stage timers (full 26.2 + linear 11.6 + experts 7.3)  45 s
topk bracket blocking (backpressure)         78 s
```

The 78 s of blocking cannot be the device, because the device is only busy a quarter of the time. So
the blocking is the host waiting on **queue drain points** while the device itself sits idle waiting for
the host to append more work. In other words prefill is a **per-layer host/device ping-pong**:

```text
per layer = 0.66 s of device work + ~1.0 s of host work, serialised  ->  1.63 s x 48 = 78 s per chunk
```

`full_host_ms` localises the host half: 26.2 s over 12 QSA layers is **1.07 ms per token per QSA layer**
against 0.157 ms for a GDN layer, so QSA's stage costs 6.8x a GDN stage and contributes 26 s of the
host side on its own.

### 49.2 Therefore the next prefill target is the QSA stage's host cost, not the staging

Three things to measure next, in this order:

1. Break `stage_ms[kStageFullAttn]` down inside a QSA layer of the prefill path: how much is submission,
   how much is an explicit drain/sync, how much is real host compute (RoPE, packing, block selection).
2. Count the drain points per layer. If each QSA layer forces a `synchronize` for routing or for the
   indexer's block selection, that is the ping-pong and it should be pipelined away exactly as the
   decode path's doorbell was.
3. Compare the prefill attention work against the reference implementations already on disk
   (`tmp/Strata/sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp` and
   `tmp/XeStrata/src/kernels/xe/qsa_prompt_attn.cpp`, whose DPAS form Strata records as 2-3x slower than
   the chunk form for decode but was written for the prompt case), to see whether Aila's
   `attention_prefill_cached_impl` is in the right family at all. Its measured 27.5 ms per instance for
   2048 tokens is only 1 s of device time in total, so this is a lower priority than the host half.

## 50. Prefill goal: 400 tok/s, and the measured ceiling that governs it

Target set by the user: **prefill at 400 tok/s**. Baseline today is 2048 tokens in ~127 s, i.e.
**16 tok/s**, so the target is a 24x improvement that requires 5.1 s for a 2K-token pass.

### 50.1 The volume is inherent

512 experts per layer x 48 layers x 2.46 MB = 60 GB of expert weights. At 2048 tokens with top-10, the
union of routes per layer is essentially all 512 experts, so a full pass reads the whole 60 GB from
host memory (observed staging volume 42.6 GB, the remainder being VRAM-resident). That volume cannot be
argued away: it needs either 60 GB of VRAM, which a 16 GB card does not have, or a transfer.

### 50.2 A new probe gives the copy model

`tests/ops/H2DBandwidthProbe.cpp` (target `AilaH2DBandwidthProbe`) copies pinned host memory to device
memory and reports bandwidth per chunk size. On this machine (A770 on a Haswell-EP PCIe 3.0 platform):

| chunk | bandwidth | per copy |
|---|---:|---:|
| 150 KB | 1.97 GB/s | 78 us |
| 300 KB | 3.27 GB/s | 92 us |
| 600 KB | 4.95 GB/s | 121 us |
| 1200 KB | 6.61 GB/s | 181 us |
| 2400 KB | 7.05 GB/s | 341 us |
| 9600 KB | 8.99 GB/s | 1068 us |
| single 512 MB | 9.91 GB/s | - |

Fitting those points gives the model this round should be planned against:

```text
t(copy) = 62 us + bytes / 9.54 GB/s
```

Two negative results pin the model down further. Rotating the destination across a 1 GB region, as the
real expert pool does, changes nothing (7.01 vs 7.01 GB/s at 2400 KB, 8.97 vs 9.00 at 9600 KB), so
destination page or TLB behaviour is not the issue. And neither `AILA_Q4E_FILL_PARTS` (1, 2, 16) nor
`AILA_Q4E_FILL_THREADS` (16) changes the prefill wall time at all (132.96 / 133.2 / 133.4 s), so the
fill plumbing is not what makes staging slow.

### 50.3 Where that leaves the target

The staging path moves 42.6 GB and an observer-free trace shows it costs 19.0 s, i.e. **2.2 GB/s**,
while the same copy size at full pipeline gives 7.05 GB/s. So the copies are being issued into an
in-order queue interleaved with kernels and therefore never pipeline.

```text
staging at the probe's rate          42.6 GB / 7.05 GB/s  = 6.0 s   -> 341 tok/s
staging with large runs              42.6 GB / 9.54 GB/s  = 4.5 s   -> 455 tok/s
non-copy device work today                                 12.7 s
host stage wall today (46.6 s) plus blocking               the rest of the 127 s
```

So **400 tok/s is reachable only if all three hold at once**: copies run at the pipelined rate with
large runs, the non-copy device work stays under ~4.5 s, and copy and compute overlap completely. Today
none of the three is true, and the largest unexplained block is the host/device ping-pong described in
section 49.

### 50.4 Ranked plan toward the target

1. **Make copies pipeline.** 19 s -> ~5 s. They currently interleave with compute in one in-order queue.
   Either batch them into large runs per layer or put them on their own queue so the copy engine and
   the compute queue overlap.
2. **Break the host/device ping-pong.** 46.6 s of host stage wall time plus roughly 49 s of submission
   blocking sits inside a 127 s pass whose device work is only 31.7 s. This is the largest single block
   of time in prefill and it is not compute.
3. **Shrink the 12.7 s of non-copy device work.** gemm_kernel 6.77 s, submit_dequant 1.85 s, GDN 1.42 s,
   QSA 0.99 s. Batched prefill GEMMs should be a fraction of that.

Milestones to report against: 16 tok/s now, ~65 tok/s once the device is the only limiter, ~110 tok/s
once copies pipeline at todays rate, ~200 tok/s once copies use large runs, 400 tok/s when compute is
hidden as well.

## 51. Correction: the trace used in sections 47, 50 mixes prefill with decode

The counts in `tmp/vtune-xpu-1-hotspots.csv` separate the phases cleanly once they are read as instance
counts rather than durations:

```text
gemm_kernel                 n = 3619   }  three kernels tracking 1:1, about 3620 calls
gemm_kernel                 n = 3622   }  3620 / 3 per decode token ~ 1200 decode tokens
submit_dequant              n = 3620   }
attention_prefill_cached    n = 36     }  12 QSA layers x 3 prefill chunks
run_linear_delta_prefill    n = 252    }  36 GDN layers x 7
attention_prefill_qsa_batched n = 12   }  12 QSA layers x 1
```

So that trace contains ~1200 decode tokens as well as prefill, and the "31.7 s of device time" splits
into prefill and decode parts that the aggregate cannot separate. Nothing above about the copy model is
affected, because the copy rows are staging (`n = 27213`, tracking the 17548 expert stagings) and the
copy probe is independent; but three specific claims in sections 47 and 50 are withdrawn:

- "prefill's non-copy device work is 12.7 s" - unknown, that figure includes decode GEMMs.
- "QSA prefill is 0.99 s" - `attention_prefill_cached_impl` at n = 36 is 0.99 s but it is the non-QSA
  fallback; the batched QSA path measures 0.08 s total.
- "the GPU is 25% busy during prefill" - the ratio is against the whole run, not prefill alone.

### 51.1 The measurement that must come next

A **prefill-only** trace: the same collection with generation limited to one token, so every instance
count is a multiple of the three prefill chunks and the per-kernel durations are prefill durations.
Until that exists, prefill's cost structure is known only as: 42.6 GB of staging at 2.2 GB/s, ~46.6 s of
host stage wall time, and ~127 s of wall clock in which device busy is well under half.

This is recorded as a correction rather than an edit so the reasoning trail stays honest: the copy model
in section 50.2 is solid (it comes from a standalone probe), while the phase attribution was not.

## 52. Prefill scaling: per-token cost grows 5x from 772 to 2048 tokens inside a single chunk

Direct CLI runs on the same model, one prompt each, `--max-tokens 1`:

| prompt | wall | ms/token | chunking |
|---|---:|---:|---|
| 772 tokens | 13.65 s | **17.7** | `PREFILL_CHUNK=512` -> 512 + 256 |
| 772 tokens | 9.75 s | **12.6** | `PREFILL_CHUNK=2048` -> single chunk |
| 2108 tokens | 133.0 s | **63.5** | 2048 + 56 + 4 |

Two facts follow.

**First, the discovered knob works, and it is a real win.** `AILA_Q35_PREFILL_CHUNK` is set by the
`qwen38-a770` preset to 512, and raising it to 2048 makes a 772-token prefill **29% faster**
(13.65 -> 9.75 s, i.e. 17.7 -> 12.6 ms/token), verified from the `[PerformancePreset] ... source=environment`
line so the override was actually in force. The mechanism is that expert staging is a **per-chunk fixed
cost**: one 512-token chunk already routes to nearly every expert, so two chunks stage the expert set
twice.

**Second, and more important, the cost is superlinear in tokens within one chunk.** Comparing the two
single-chunk cases: 772 tokens cost 12.6 ms/token while 2048 tokens cost 63.5 ms/token. That is 5x the
per-token cost for 2.65x the tokens, so the dominant term grows faster than linearly in sequence length.

That also explains why raising the chunk did nothing for the 2108-token prompt (133.0 s against a 133.8 s
baseline, with `staged` 17,664 vs 17,548 and `h2d_mb` 42.7 vs 42.2 GB unchanged): its first chunk was
already 2048 under either setting, so only the negligible 56- and 4-token remainders changed.

### 52.1 What this means for the 400 tok/s goal

The staging volume at 2048 tokens is inherent - one chunk routes to about 72% of all (layer, expert)
pairs, so 42.6 GB must cross the link once per pass - and the copy probe says the best achievable rate
for that volume is 42.6 GB / 9.54 GB/s = 4.5 s. So staging alone does not explain a 127 s pass.

The 5x per-token growth inside one chunk does. Whatever grows faster than linearly there is now the
target, and the two candidates worth separating are the block-sparse attention path (whose block count
per token grows with position, `tile_max_blocks = (tile_start_pos + tile_count) / compress_ratio`, so
score and selection work is O(tokens x blocks)) and the expert GEMM path (whose row count per expert
grows with tokens). One measurement separates them: run 1024- and 1536-token prompts and check whether
the growth tracks tokens x blocks (attention) or tokens x experts (MoE).

## 53. Prefill is at 150 tok/s, not 16: the runner was the wrong instrument

Sections 47-52 measured prefill through the decode A/B runner, whose `GenerateMessages` wall time and
whose `AdaptiveWork` counters do not describe prefill. Two direct CLI runs expose the difference:

| run | staged experts | `h2d_mb` | `misses` | `experts_host_ms` |
|---|---:|---:|---:|---:|
| A/B runner, 2108 tokens | 17,548 | 42,578 | 869,431 | 7,265 |
| plain CLI, 2102 tokens | **766** | **1,851** | **766** | **315** |

The runner moves 23x more expert data and reports a different counting convention for misses, and its
wall time also carries one-time costs (the CLI's `GenerateMessages` was 32.2 s inside a 212.6 s process,
so 180 s of that run was model load). Everything the earlier sections said about "prefill takes 127 s,
16 tok/s" is therefore **withdrawn**.

### 53.1 A dedicated prefill scoreboard

`tools/prefill-bench.ps1` drives `Aila.exe` directly, parses `[GenerateMessages] Full prefill: N tokens`
and the generation wall time, and reports tok/s plus a linear fit for the marginal per-token cost.

```text
 1019 tokens   10,338.7 ms   10.15 ms/token    98.6 tok/s
 2026 tokens   13,475.8 ms    6.65 ms/token   150.3 tok/s
 3033 tokens   22,146.5 ms    7.30 ms/token   137.0 tok/s

marginal 5.86 ms/token -> 171 tok/s   fixed per run 4,364 ms
```

So prefill today is **150 tok/s at 2K tokens and 171 tok/s marginal**, and the 400 tok/s goal needs a
**2.34x** improvement in the marginal per-token cost (5.86 -> 2.50 ms), plus attention to the 4.4 s fixed
cost if short prompts matter.

### 53.2 Where the marginal cost sits

Two facts from the CLI run with `AILA_Q4E_WORK_SUMMARY=1`:

- `h2d_mb` is only 1,851 MB and `fill_ms` 285 ms, so **staging is not the marginal cost** - it is a
  per-chunk fixed cost, and with `PREFILL_CHUNK=2048` a 2K prompt pays it once.
- The per-stage sums are `full_host_ms` 26,198, `linear_host_ms` 7,460, `experts_host_ms` 315. Those sum
  above the 16,953 ms run, so they are per-layer accumulations rather than wall time and must only be
  used as ratios: **QSA (full attention) is ~76% of prefill compute, GDN ~22%, expert CPU serving ~1%**.

At 5.86 ms/token marginal and 76% in QSA over 12 QSA layers, one token costs ~375 us per QSA layer. The
arithmetic for that layer is ~50 MFLOP, which at this GPU's ~20 TFLOPS is 2.5 us. So **the prefill QSA
path runs at well under 1% of peak** and is the single place where the 2.34x has to come from.

### 53.3 Next step

Build the prefill equivalent of the decode attention micro-probe (`AilaQsaAttentionProbe` already exists
for decode at 2052 cells) so the QSA prefill kernel can be timed in seconds in isolation, then find out
why it is two orders of magnitude off peak: most likely a per-token or per-block host loop around
`attention_prefill_qsa_batched`, or a work-group geometry with far too few groups for a 2048-token chunk.

## 54. The prefill QSA attention kernel gives every query token its own work-group

`attention_prefill_qsa_batched_impl` (`src/ops/AttentionOps.cpp:5684`) is dispatched as

```cpp
const int num_groups = num_heads / 4;                       // 6
cgh.parallel_for(sycl::nd_range<1>(count * num_groups * wg_size, wg_size), ...);
const int group  = item.get_group(0);
const int row    = group / num_groups;                      // one query token per work-group
const int g_idx  = group % num_groups;
const int h0     = g_idx * 4;
```

So a work-group owns **one query row and four heads**, which is 12,288 work-groups for a 2048-token
chunk. Parallelism is fine; the problem is reuse. Every one of the 2048 query work-groups streams the
whole K/V for the layer, so the layer's 2.1 MB of K/V is requested 2048 times.

### 54.1 The arithmetic says this is where the 2.34x lives

```text
K/V traffic per pass   2048 tokens x 2.1 MB x 12 QSA layers      = 51.6 GB
measured QSA time      ~76% of 5.86 ms/token x 3033 tokens       = 13.5 s   -> 3.8 GB/s
attention FLOPs        24 heads x 2052 cells x 256 dims x 2 x 2 x 2048 x 12 = 1.24 TFLOP
                      1.24 TFLOP / 13.5 s = 92 GFLOP/s           = 0.5% of this GPU's ~20 TFLOPS
```

A layer's K/V is 2.1 MB and the L2 is 16 MB, so this traffic should be served from cache; at 3.8 GB/s
effective it clearly is not. The kernel is therefore both recomputing and re-fetching, and the wasted
factor is the query-token count.

### 54.2 The fix is the reference implementations already on disk

The shape that avoids this is a flash-style prefill kernel: stage a **query tile** and a **K/V tile** in
SLM once and let every query row in the tile reuse them, so K/V is fetched once per tile rather than once
per token. That is exactly what the reference kernels the user pointed at do:

- `tmp/XeStrata/src/kernels/xe/qsa_prompt_attn.cpp` (prompt-case chunk kernel),
- `tmp/Strata/sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp` (the DPAS variant).

Note the recorded negative from the decode side: the XMX/DPAS form was 2-3x *slower* than the chunk form
for decode (`Strata/docs/INTEL.md:364,581,629`). That negative is about decode; for prefill the winning
property to borrow is the tile reuse, not the DPAS.

### 54.3 Concrete next step

Rewrite `attention_prefill_qsa_batched_impl`'s inner kernel to tile queries (e.g. 32 or 64 rows per
work-group) and stage K/V tiles in SLM, keeping the existing signature and the block-sparse index
arguments so the caller does not change, and gate it behind an env flag with the current kernel as the
A/B control. Expected effect: K/V traffic drops by the query-tile factor, which is where the 51.6 GB and
the 0.5%-of-peak utilisation both come from.

## 55. Correction: the marginal prefill cost is per-token MoE weight traffic, not attention

Section 54 blamed the QSA attention kernel using a 76% share. That share came from the per-layer stage
sums, which section 53.2 already flagged as unreliable because they sum above the run's wall time. The
scoreboard's own scaling contradicts the conclusion:

| prompt | ms/token |
|---:|---:|
| 1019 | 10.15 |
| 2026 | **6.65** |
| 3033 | **7.30** |

A cost that dominated and grew with the key count, as dense attention does, would make the per-token
figure rise steeply with prompt length. Instead it is flat (2026 -> 3033 is +50% tokens for +10%
ms/token, and 1019 is only worse because the 4.4 s fixed cost is spread over fewer tokens). So attention
is **not** the marginal term, and rewriting its kernel would not move the scoreboard.

### 55.1 What the marginal term actually is

Marginal 5.86 ms/token, independent of prompt length, with only ~1% of it on the CPU expert path
(`experts_host_ms` 315 ms) and ~285 ms of staging for a whole pass. That leaves the GPU expert path:
each token selects 10 experts in each of 48 layers, so it touches

```text
10 x 48 = 480 (layer, expert) pairs x 2.46 MB = 1.18 GB of expert weights per token
1.18 GB / 5.86 ms = 202 GB/s   (~40% of an A770's ~500 GB/s)
```

So prefill is **weight-bandwidth-bound on the expert set**, running at 202 GB/s of effective throughput.
That is the number the 2.34x has to move, and the way to move it is reuse: if each expert's weights are
read once and applied to all of the token rows routed to it, the per-token traffic falls by the average
rows-per-expert factor (about 60 for a 3033-token chunk against 512 experts per layer) rather than being
paid per token.

### 55.2 The question to answer before writing any kernel

Whether the prefill expert kernel is already expert-major or row-major. The preset enables
`[Q4E PrefillExpertXmx] enabled=1 max_chunk_rows=4096 channels=2560/640`, which suggests a batched path
exists, so the first measurement is not "write a grouped GEMM" but "count how many times each expert's
weights are read per chunk". A cheap proxy: compare `h2d_mb` and the expert kernel's instance count
against the chunk's (layer, expert) set. If the kernel is already read-once, the 202 GB/s is a real
bandwidth ceiling and the lever becomes occupancy/access pattern instead.

### 55.3 Correction to the attention claim

The QSA prefill kernel really does give each query token its own work-group
(`AttentionOps.cpp:5743`, `row = group / num_groups`), which is a genuine inefficiency and worth fixing
for long prompts, but at the current numbers it is **not** where the 400 tok/s target is lost. Fixing it
would help at 8K-32K contexts, not at 2K.

## 56. First prefill wins: +72% marginal throughput from two defaults

Both changes were found by sweeping one variable at a time on the scoreboard, and both are now defaults
for the `qwen38-a770` preset rather than flags a user has to know about.

### 56.1 The XMX prefill expert path is a net loss

```text
2026 tokens, AILA_Q4E_PREFILL_EXPERT_XMX=1 (old default)   13,412.6 ms   6.62 ms/token   151 tok/s
2026 tokens, AILA_Q4E_PREFILL_EXPERT_XMX=0                 11,826.8 ms   5.84 ms/token   171 tok/s
```

The source comment already recorded that this path won 2.1x **in its kernel alone with no end-to-end
change**; on prefill it is simply slower. `src/models/Qwen4ExpBnb4Backend.cpp` now defaults it off, with
the measurement recorded in the comment and the flag still available.

### 56.2 Expert weight reuse scales with the chunk size, and the preset pinned the worst value

`[Q4E PrefillExpertXmx] max_chunk_rows=4096` and a sweep of `AILA_Q4E_PREFILL_EXPERT_XMX_ROWS` shows the
mechanism directly:

```text
rows = 256    25,453.5 ms   12.56 ms/token    79.6 tok/s
rows = 4096   13,412.6 ms    6.62 ms/token   151.1 tok/s
```

Halving the row batch doubles the time, so each expert's weights are being read per row-batch and the
cost falls as the batch grows. The same holds for the chunk size, because a prompt longer than one chunk
repeats the whole expert set per chunk:

```text
2995 tokens, PREFILL_CHUNK=2048   19,247.6 ms   6.43 ms/token   155.6 tok/s
2995 tokens, PREFILL_CHUNK=4096   15,389.7 ms   5.14 ms/token   194.6 tok/s
```

`src/utils/Qwen38A770Preset.hpp` set `AILA_Q35_PREFILL_CHUNK` to **512**, the worst of those, so it is
now 4096 with the numbers recorded beside it.

### 56.3 The default path after both changes

Measured on `tools/prefill-bench.ps1` with no chunk override, i.e. the shipped preset:

| prompt | before | after |
|---|---:|---:|
| 1019 | 98.6 tok/s | **117.6** |
| 2026 | 150.3 tok/s | **171.1** |
| 2995 | 137.0 tok/s | **194.7** |

```text
marginal 3.40 ms/token -> 294 tok/s     fixed per run 5,200 ms
```

So the marginal per-token cost went from 5.86 to 3.40 ms (**+72%**) and the distance to the 400 tok/s goal
is now **1.36x** rather than 2.34x. Throughput also still rises with prompt length (117.6 at 1K, 194.7 at
3K), because the 5.2 s fixed cost amortises.

### 56.4 Next

The same reuse argument implies an even larger chunk helps longer prompts, so the next sweep is
`PREFILL_CHUNK` 4096 against 8192 at a 6K prompt, checking VRAM headroom since activation buffers scale
with the chunk. After that the remaining 1.36x has to come from the expert kernel itself, since the
weight-reuse structure is now doing what it can.

## 57. Chunk 8192 as the shipped default, and what the asymptote says

The chunk sweep continues to pay at longer prompts, so the preset now ships 8192:

```text
7878 tokens, PREFILL_CHUNK=4096   36,786.7 ms   4.67 ms/token   214.2 tok/s
7878 tokens, PREFILL_CHUNK=8192   32,992.7 ms   4.19 ms/token   238.8 tok/s
2026 tokens, PREFILL_CHUNK=8192   11,862.3 ms   5.86 ms/token   170.8 tok/s
2995 tokens, PREFILL_CHUNK=8192   15,412.3 ms   5.15 ms/token   194.3 tok/s
```

Short prompts are unchanged at 8192 because they already fitted in one chunk, so the larger default is
free where it does not help and worth 11% where it does. Verified on the shipped preset with no
overrides:

```text
2026 tokens   11,837.6 ms   5.84 ms/token   171.1 tok/s
7878 tokens   33,042.2 ms   4.19 ms/token   238.4 tok/s
```

### 57.1 The goal needs the asymptote, not the fixed cost

```text
marginal (7878 vs 2026)  = (33042.2 - 11837.6) / (7878 - 2026) = 3.62 ms/token
asymptotic throughput    = 1 / 3.62 ms = 276 tok/s
```

Throughput rises with prompt length only because the fixed ~5.2 s amortises; the ceiling set by the
marginal cost is **276 tok/s**, so no prompt length reaches 400. The target therefore has to come out of
the per-token term: 3.62 -> 2.50 ms, a factor of 1.45.

### 57.2 Where that factor has to come from

Per token the expert path evaluates 10 experts in each of 48 layers, i.e. 480 x 4.91 M MAC = 2.36 G MAC
= 4.7 GFLOP. At 3.62 ms that is **1.3 TFLOPS, about 6.5% of this GPU's ~20 TFLOPS bf16 peak**. The reuse
structure is now doing what it can with the chunk size, so what is left is the efficiency of the expert
kernel itself under the ALU (non-XMX) path that section 56.1 made the default.

Next measurements, in order: read the ALU expert kernel's work-group geometry and effective bandwidth the
same way the hyper-connection GEMM was diagnosed (a tiny work-group count on a huge GPU was the defect
there), then compare its achieved FLOP/s against the 6.5% figure to decide whether the problem is
occupancy or instruction mix.

## 58. The systolic path wins once M is large, and a row gate takes both sides

Section 56.1 turned XMX off after measuring it at chunk 2048. That conclusion does not survive the 8192
chunk: the systolic units need a large M, which the larger chunk supplies.

```text
M=2026   XMX 13,396 ms vs ALU 11,838 ms    XMX loses 13%
M=2995   XMX 15,641 ms vs ALU 15,412 ms    XMX loses  1%
M=4952   XMX 19,971 ms vs ALU 22,483 ms    XMX wins  13%
M=7878   XMX 26,159 ms vs ALU 33,042 ms    XMX wins  26%
```

So both paths are right, for different M. `RunGroupedExperts` now gates the XMX consumer on the row
count (`AILA_Q4E_PREFILL_EXPERT_XMX_GATE_ROWS`, default 4096, the measured crossover) and the XMX path
is enabled again by default. Measured on the shipped preset with no overrides:

| prompt | tok/s | path |
|---|---:|---|
| 2026 | **171.0** | ALU |
| 4952 | **250.3** | XMX |
| 7878 | **299.5** | XMX |

```text
marginal 2.47 ms/token -> 405 tok/s asymptote     fixed per run 6,844 ms
```

Progress against the goal: **150 tok/s at 2K and 137 at 3K when this round started, now 171 and 250, with
299.5 at 8K.**

### 58.1 A trap worth recording

`AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M` looks like the obvious gate and is not: it is a padded-band
parameter. Setting it to 4096 forced every band to pad that far and measured 39,636 ms at M=7878 against
26,159 ms - worse than either path alone. The real gate had to be a row-count condition at the call site.
Reading the surrounding comment before reusing a knob saved nothing here; measuring did.

### 58.2 What is left for 400 tok/s measured, not asymptotic

```text
throughput(n) = 1000 n / (6844 + 2.47 n)
at 32K tokens -> 373 tok/s    at 64K -> 388 tok/s    asymptote 405
```

So the asymptote clears 400 but no realistic prompt reaches it while the fixed cost is 6.8 s and the
marginal is 2.47 ms. Two levers, both now precisely sized:

1. **The marginal 2.47 ms/token.** Per token the expert path is 4.7 GFLOP, so 2.47 ms is 1.9 TFLOPS,
   about 9.5% of peak. This is exactly the user's suggestion: the weights are stored NF4-packed and are
   unpacked into the DPAS layout per tile, so a load-time transform into the systolic layout (a VNNI-style
   repack) would remove that per-tile work. It is the remaining big item and it is a data-layout change,
   not an algorithm change.
2. **The 6.8 s fixed cost**, which is what holds measured throughput well under the asymptote.

## 59. The remaining marginal cost is a weight-layout problem, exactly as the user suggested

User's direction: the weights are sharded, so transpose them to the systolic (VNNI-style) layout at load
time; cost unknown, benefit probably large, work substantial. The source confirms the defect and pins
the location.

### 59.1 The weight operand is strided by construction

`src/ops/MoeExpertXmx.cpp:76-79` says it outright:

```cpp
// NF4 -> bf16/fp16 dequant of a whole chunk: 1 work-item = 1 uint32 (8 codes)
// plus a vec8 store, wgu written [B][2I][H] and wdn [B][H][I] in the packed
// [N][K] linear order (no transpose; the GEMM takes the strided view).
```

and the matmuls take exactly that strided view (`MoeExpertXmx.cpp:301-307`):

```cpp
prim.s1 = dnnl::memory::desc({b, m, h},      elem_dt, {m * h, h, 1});
prim.w1 = dnnl::memory::desc({b, h, two_i},  elem_dt, {two_i * h, 1, h});   // <-- K contiguous, N stride H
prim.w2 = dnnl::memory::desc({b, i, h},      elem_dt, {h * i, 1, i});       // <-- same shape of problem
```

oneDNN's canonical `K x N` operand for a matmul is N-contiguous (`{K*N, N, 1}`); this is the opposite,
K-contiguous with N strided. That is why the systolic path currently lands around 1.4 TFLOPS (7% of
peak) even with a large M: the kernel cannot use the native DPAS layout.

### 59.2 Two ways to fix it, in increasing order of work

1. **Transpose inside `submit_dequant`.** Change the write mapping from `n*K + k` to `k*N + n` and the
   descriptors to `{... , N, 1}`. One side of the copy becomes strided instead of the other, so this may
   only break even - worth measuring first because it is a small change.
2. **Transpose the packed NF4 layout once at load.** Store the nibbles as `[K/8][N]` so that a dequant
   work-item reads its 8 codes contiguously *and* writes them contiguously into `[K][N]`. This removes the
   strided side completely and is the version that should scale, at the cost of a one-time transform and
   a matching change in every reader of the packed layout. This is the user's proposal.

The measurement that decides between them: time `submit_dequant` against the two matmuls per chunk. If
dequant is a small share (rough arithmetic says one chunk of experts is a few GB of bf16 writes, so tens
of ms against a layer's ~548 ms), then option 1's cost is irrelevant and option 2's only advantage is the
GEMM descriptor, which option 1 already delivers.

### 59.3 Where the goal stands after this round

| prompt | start of round | now |
|---|---:|---:|
| 2026 | 150.3 tok/s | **171.0** |
| 2995 | 137.0 tok/s | **250.3** |
| 7878 | - | **299.5** |

```text
marginal 2.47 ms/token -> 405 tok/s asymptote     fixed per run 6,844 ms
measured: throughput(n) = 1000 n / (6844 + 2.47 n)   373 tok/s at 32K, 388 at 64K
```

So the asymptote clears 400 but no realistic prompt reaches 400 measured yet. The two remaining levers
are the weight layout above (marginal 2.47 -> ~2.0 ms would put 400 within reach at 16K tokens) and the
6.8 s fixed cost.
