# R1 decode submit-path audit: the ~20 submissions per layer and where the 5x host inflation is

Date: 2026-10-16. Repo `E:\RiderProjects\Aila`, base HEAD `bb07130`.
Scope: read-only. No GPU run, no build, no commit, no source edit; this document is the
only file written. Sources: the working tree as it stands (hashes below),
`tmp/q4e-sprint-r1-20261014/e10-bundle{,-b2}/result.json` + `e10-bundle-worker.log`
(`worker_environment` pins every flag), `docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md`,
`docs/plans/qwen38-r1-decode-tail-p7-2026-10-16.md`,
`docs/plans/qwen38-r1-decode-kernel-roofline-2026-10-15.md`, and the probe receipts in the
bottleneck audit's `tmp/q4e-bottleneck-audit-20261016/`.

Worktree caveat: the tree at audit time carries two uncommitted owner patches that are **not**
part of HEAD (they landed at 16:00:02 / 16:00:45, after the 15:55:49 commit):
`src/models/Qwen4ExpBnb4Backend.cpp` +15 (`AILA_Q4E_JOIN_RELEASE` release gate in
`RunResidentDecode`, sha256 `03646fd1...`) and `src/ops/MoeExpertOps.cpp` +40/-8
(`*_submitted` flags replacing `!= sycl::event{}` fence selection, sha256 `2d779c80...`).
Every line anchor below is LF-based (`rg -n` / python) against **that** tree; reverting either
patch shifts anchors in the 1200-1620 / 5900-6290 ranges.

Reference config = the "e10 bundle" (best measured decode arm): model
`C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2`, 48 layers (36 GDN + 12 full), 512 experts
top-10, NF4, adaptive pipeline, `count == 1` (T1). Its env: `MOE_FRONT_FUSED=1`,
`GDN_HISTORY_FUSED=1`, `DECODE_PENDING_MLP_RESIDUAL=1`, `KV_QUANT_FORMAT=int8`,
`ROUTER_RESIDENT_FUSED=0`, `ATTN_NF4_Q8=0`, `GPU_NF4_Q8=0`, `MOE_SHARED_FUSED` **unset**,
`HC_FUSED` default, `DECODE_FUSED_HC` **unset**, `CPU_ASYNC_ROUTE=1` (inert on the adaptive
pipeline: `Qwen4ExpBnb4Backend.cpp:2533` requires `!use_adaptive_pipeline_`),
`CPU_DEFER_ARENA_WAIT=0`, `WORK_SUMMARY=1`.

---

## 0. Conclusions

1. **Exact per-layer submission count: 19 on a GDN layer, 25 on a full (QSA) layer**
   (+1 every 4th token for indexer block compression). Over 48 layers that is **984-987
   submissions/token = 20.5-20.6 per layer**, which reconciles the ~20 the bottleneck audit
   used in its probe. Section 1 is the full inventory with `file:line`, the submitted op, the
   result event's use, and whether the op is already fused.
2. **Only 6 of the 19/25 are "obvious" data-path kernels.** 6 are Hyper-Connection, 4 are the
   routed/shared MoE glue (2 resident projections, 1 accumulator, 1 shared add), and the full
   layers add 9 (QSA + attention). Three of the already-fused ops (MoE front, gate/up SwiGLU,
   GDN history commit) are the result of exactly this kind of exercise.
3. **The 5x is real but it is not inside `sycl::queue::submit` itself.** Measured host cost per
   submission, group by group, is **12.9-35.5 us** (Section 2.1) against **5.9-7.4 us** in the
   probe and **6.4 us** for the same HC kernels in the E5 bench (194 kernels / 1.24 ms). The
   inflation tracks *how* the submission is issued: 2 host SYCL waits per layer end the L0 v1
   dynamic batch, 4 submits per layer carry explicit `depends_on` events, and the MoE submits
   marshal ~1 KB of descriptor arrays per call.
4. **Two structural facts dominate and both are already priced.** (a) the probe's empty
   kernel + wait round trip is 59 us; the code has **2 waits per layer** (`actual_route_event`
   and `resident_input_event_`), i.e. ~96 round trips/token of which one (the resident-input
   wait) exists only to reuse a pinned buffer - `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1` already
   removes it. (b) same kernels, per-layer wait cadence: 13.8-14.6 ms/token; one wait at the
   end: 8.3 ms/token.
5. **Recommendation order:** price the two wait/flow changes first (no arithmetic change, both
   already implemented as flags), then the two in-tree HC/shared fusions, then one VTune arm to
   split UR/L0 self time from our own block glue. Section 5 has the 5 ranked experiments.
6. **Graph replay has a hard ceiling of ~13% here** (probe: 12.7 vs 14.6 ms/token at the same
   per-layer wait cadence) and on DG2 requires a device-side parameter block because kernel
   parameters cannot be updated in place. The natural graph units are the attention subchain
   and the MoE subchain, because the routing readback plus CPU cold-expert join splits the
   layer in the middle (Section 4).

---

## 1. Complete submission inventory (count == 1, e10 bundle)

How the count was obtained (reproducible):

1. Every `ctx.queue().submit(...)` / `queue().memcpy(...)` site was enumerated in
   `src/models/Qwen4ExpBnb4Backend.cpp` (`ForwardImpl` `:2414-2881` layer loop,
   `RunResidentDecode` `:1235-1540`), `src/models/Qwen4ExpPrefill.cpp`
   (`RunAttentionLayerTile` `:14-256`) and every op they call (python body scan per function).
2. The runtime branch of each `if` was pinned by the e10 env and by evidence lines in
   `e10-bundle-worker.log` (`[Q4E MoE Front Fused] actual_enabled=1`, `[Q4E GDN History Fused]
   actual_enabled=1 layer=0`, `[Q4E ResidentDecode] mode=2 adaptive_pipeline=1`, `staged=0`).
3. Cross-check with the log's own counters: `hits ~ 237-257`, `misses ~ 190-306`, `staged = 0`,
   `cpu_tasks ~ 190-306` per token. No cold expert in decode reaches the GPU staging banks
   (misses <= CPU credit cap 10), so `StageColdExpertsAsync` H2D submits are **0/token**, not
   2/layer.

### 1a. Block shared by every layer (16)

| # | per-layer caller | submitted op -> submit site | result event use | already fused |
|---:|---|---|---|---|
| 1 | `Qwen4ExpBnb4Backend.cpp:2449` (attn HC) | `try_hc_decode_inject_add_and_grouped_rms_norm` `HyperConnectionOps.cpp:314`; L0: `grouped_rms_norm` `:12` | dropped (`void`) | yes: inject + grouped RMSNorm, 1 kernel |
| 2 | same -> `:2013` | `hc_decode_down_inject` `HyperConnectionOps.cpp:380` | dropped | no |
| 3 | same -> `:2019` | `hc_decode_up_mix` `HyperConnectionOps.cpp:460` | dropped | no |
| 4 | `:2522` (hoisted) or `:2546` | `StageResidentInput` -> `queue.memcpy` D2H `Qwen4ExpExpertExecutor.cpp:746` | stored `resident_input_event_`; host-waited at `:745` on the *next* layer; passed to the CPU job as `job.activation` `:936` | no (5 KB D2H) |
| 5 | `:2561` | `ops::bf16_moe_front_fused` `Bnb4BitLinear.cpp:2811` | dropped | yes: router GEMM + shared-gate GEMV, 1 kernel |
| 6 | `:2594` | `ops::router_softmax_topk_device` `MoeExpertOps.cpp:123` | `actual_route_event`; host-waited `:2637` inside `wait_actual_route` `:2620-2634` | yes: softmax + top-k + normalization, writes pinned host ids/weights in-kernel |
| 7 | `:2722` | `Bnb4BitLinear::try_forward_decode_gate_up_swiglu` `Bnb4BitLinear.cpp:1236` | dropped | yes: gate + up + SwiGLU, 1 kernel |
| 8 | `:2729` | `shared_down_proj.ForwardDecodeRows` -> `packed_nf4_gemv_bf16_impl` `Bnb4BitLinear.cpp:1493` | dropped | no |
| 9 | `:2501` (mlp HC) | `hc_inject_add_and_grouped_rms_norm` `HyperConnectionOps.cpp:241` | dropped (residual ownership moves to `pending_mlp`) | yes: inject + RMSNorm, 1 kernel |
| 10 | same -> `:2013` | `hc_decode_down_inject` `:380` | dropped | no |
| 11 | same -> `:2019` | `hc_decode_up_mix` `:460` | dropped | no |
| 12 | `RunResidentDecode:1474` | `moe_decode_gate_up_swiglu_grouped` `MoeExpertOps.cpp:736` (dispatch `:5056`) | returned as `last`; dependency of #13 and of #15 | yes: gate+up+SwiGLU over <=16 resident experts, 1 launch |
| 13 | `:1480` | `moe_decode_down_to_rank_f32` `MoeExpertOps.cpp:1126` (dispatch `:5154`) | `depends_on(last)`; recorded as the resident reader `:1484` | yes: down + rank-indexed store |
| 14 | `:1526` | `queue.memcpy(resident_weights_, t_weights, 64 B)` H2D | dropped | no |
| 15 | `:1531` | `moe_ordered_accumulate_down_rows` `MoeExpertOps.cpp:6379` (host-rows variant `:6428`) | returned `merge` -> `FinishResidentCpuRead` `:1533`; gates next-layer `StageResidentInput` (`:745`) and the CPU result read `:1273-1277` | yes: rank merge + routed add (+ shared add when `MOE_SHARED_FUSED=1`) |
| 16 | `:2855` | `add_shared_expert_to_y` `Qwen4ExpBnb4Backend.cpp:137` | dropped | no (only while `MOE_SHARED_FUSED=0`) |

### 1b. GDN (linear) layers add 3 -> **19/layer**, x36

| # | caller | op -> site | event use | fused |
|---:|---|---|---|---|
| 17 | `Qwen4ExpPrefill.cpp:64` | `project_decode` `:29` -> `ForwardDecodeRows` `Bnb4BitLinear.cpp:2306` -> `forward` `:2411` -> `packed_nf4_gemv_bf16` `:1493` | dropped | no |
| 18 | `:77` | `ops::run_linear_delta_decode_gpu` `GatedDeltaNetOps.cpp:414` -> fastpath submit `:115` (the `:273` non-fastpath submit is not taken) | dropped | yes: conv + QK + delta-rule + gate, 1 kernel |
| 19 | `:81` | `linear_o_proj.TryForwardDecodeCommitHistory` `Bnb4BitLinear.cpp:2389` -> `packed_nf4_gemv_bf16<true>` `:1493` | dropped | yes: o_proj GEMV + conv-history ring write in one nd_range |

(Fallback pair, not taken in e10: `:88` `commit_linear_conv_history` + `:90` another
`project_decode` - that would make a GDN layer 20.)

### 1c. Full (QSA) layers add 9 (10 in the compression token) -> **25 (26)/layer**, x12

| # | caller | op -> site | event use | fused |
|---:|---|---|---|---|
| 20 | `Qwen4ExpPrefill.cpp:144` | `project_decode(qkv_proj)` -> NF4 GEMV `:1493` | dropped | no |
| 21 | `:150` | `decode_prepare_qgkv_packed_partial` `ElementwiseOps.cpp:1312` | dropped | yes: q/k RMSNorm + partial RoPE + KV cache write |
| 22 | `:209` | `DenseLinear(index_qk_proj)` `Qwen4ExpBnb4Backend.cpp:1881` -> `bf16_gemv_bf16` `Bnb4BitLinear.cpp:2881` | dropped | no (sparse branch: `dense_phase` false because `num_complete_blocks 1486 > block_topk 512`) |
| 23 | `:213` | `qsa_prepare_causal_query_and_select_blocks` `QsaIndexerOps.cpp:1451` -> `qsa_prepare_q_and_cache_k` `:204` | dropped | yes: Q norm + RoPE + indexer-K write |
| 24 | `:213` | same call -> `qsa_compute_block_scores_compressed` `QsaIndexerOps.cpp:1082` | dropped | no |
| 25 | `:213` | same call -> `qsa_select_topk_blocks` `QsaIndexerOps.cpp:1301` (radix variants `:62`, `:1225`, `:1246`; exactly one runs) | its return value `num_active_blocks` is a **host** value consumed by #26 | no |
| 26 | `:248` | `attention_decode` `AttentionOps.cpp:2120` -> head-256 dispatch `:2210-2225` -> `...partial_merge_impl` phase 1 `:476` | dropped | yes: QK^T + softmax + V over 128-token tiles |
| 27 | same | merge kernel `AttentionOps.cpp:686` | dropped | yes: partial merge to bf16 |
| 28 | `:256` | `project_decode(o_proj)` -> NF4 GEMV `:1493` | dropped | no |
| + | `:213` every 4th token | `qsa_compress_key_blocks` `QsaIndexerOps.cpp:944` | dropped | no |

### 1d. Totals

| kind | per layer | x layers | submissions/token |
|---|---:|---:|---:|
| GDN (layers 36) | 19 | 36 | 684 |
| full (layers 12) | 25 | 12 | 300 |
| indexer block compression (12 layers x 1/4 tokens) | 3 | - | 3 |
| **total** | **20.5-20.6** | 48 | **984-987** |

Not counted (off or not per-layer): `dev_mark()` `ext_oneapi_submit_barrier`
`Qwen4ExpBnb4Backend.cpp:2401` (needs `DECODE_STAGE_EVENTS=1` **and** a profiling queue);
`verification_*` memcpys `:2444-2468`, `:2576-2581`; `[Q4E AdaptiveWork]` (1 line/forward);
`dump_hidden_row`; the once-per-forward embed-id D2H `:2133`; PLE (`:2436`, layer 1 only,
2-3 submits); the head `forward_head` `:2943` (1-2 submits, inside `head_host_ms=0.15`).

**Dead code found:** the second `RunResidentDecode` overload
(`Qwen4ExpBnb4Backend.cpp:1542-1614`, `GpuReadyLease*`/`ResidentColdWork*` form) has **no call
site** in `src/` - it is unreachable since the legacy `expert_cache_` pipeline was removed
(that branch now throws at `:2838`). Its 6 submits/layer (`:1566`, `:1569`, `:1583`, `:1586`,
`:1597`, `:1603`) and the `moe_resident_gate_up/down` submits at `:2682/:2684` belong to that
dead path, not to the e10 bundle.

---

## 2. Where the 5x host inflation lives

### 2.1 The arithmetic against the inventory

`[log]` e10 ledger, `pos=5945-5951` decode steps (`e10-bundle-worker.log`), split by the
inventory above. "us/submit" = host ms / submissions in that group; "x" = that value / 6.4 us
(the E5 bench's HC-kernel host cost) with the probe's 5.9-7.4 us range as the other end.

| group | submits/token | host ms/token | us/submit | x vs 5.9-7.4 us |
|---|---:|---:|---:|---:|
| HC (48x6 + 3 mixer) | 291 | 6.62 | 22.8 | 3.1-3.9 |
| GDN attention (36x3) | 108 | 3.31 | 30.6 | 4.1-5.2 |
| full attention (12x9 + 3) | 111 | 2.73 | 24.6 | 3.3-4.2 |
| shared expert (48x2) | 96 | 2.89 | 30.1 | 4.1-5.1 |
| shared add (48x1) | 48 | 0.62 | 12.9 | 1.7-2.2 |
| MoE front (48x1) | 48 | 1.70 | 35.4 | 4.8-6.0 |
| router top-k (48x1) | 48 | 0.89 | 18.5 | 2.5-3.1 |
| MoE stage (4/layer) | 192 | inside `experts_host 19.6-20.4` (= `cpu_compute 13.3` + `cpu_join 9.2` + submits + coordination) | - | - |
| ple (L1) / head | 3 / 1-2 | 0.24 / 0.15 | - | head is dominated by the 1271 MB bf16 LM head GEMV, not a submit metric |

These are **block** costs, so part of each is dispatch glue rather than `submit` (candidate
L5). The honest reading: the per-call submit cost in the production path is 3-5x the bare
probe on *every* group, and the ratio scales with how much non-submit work and how many
waits/dependencies surround the call.

### 2.2 Candidates (falsifiable, with the cheapest test)

| id | candidate | where | hypothesis | cheapest test |
|---|---|---|---|---|
| L1 | **batch-cold submits after host waits** | `Qwen4ExpBnb4Backend.cpp:2637` (`actual_route_event.wait_and_throw`), `Qwen4ExpExpertExecutor.cpp:745` (`resident_input_event_`), `:927`/`:1332` (`prev_cpu_h2d_ev_`) | L0 v1 + dynamic batching holds submitted kernels in a batch until a wait/event query; every wait ends the batch, so the submits after it pay the full launch path (~20-35 us) instead of a batch append. 2 waits/layer => 96/token. | (a) the probe already has both arms: per-layer wait 13.8-14.6 vs one wait 8.3 ms/token, empty-kernel+wait round trip 59 us; (b) **run one decode arm with `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1`** (removes only the `:745` wait) and read `hc_host_ms`/`linear_host_ms`/`full_host_ms` medians |
| L2 | **dependency-carrying submits** | `MoeExpertOps.cpp:1126-1128` (down depends on gate/up), `:6379-6381` (accumulate depends on down), `:736-738` (gate/up depends on `resident_ready`); `Qwen4ExpAdaptiveCache.cpp:652` (H2D depends on staging reader; prefill only) | a non-empty `depends_on` cannot join the dynamic batch and forces a standalone launch per MoE submit (4/layer). | probe arm: same kernel, `submit` with vs without `cgh.depends_on`; count host us; 2 GPU-min |
| L3 | **functor width / descriptor marshalling** | `MoeExpertOps.cpp:5079-5086`, `:5179-5188` (16 `MoeExpertSlotDev` ~ 6 pointers each + 16 ranks + 16 tokens ~ 0.8-1 KB per submit) | the runtime allocates+marshals the functor per submit; the probe kernel carries ~32 B. | micro-bench: same kernel with 1 vs 16 experts (the launch path accepts either); or print `sizeof(functor)` next to the counter |
| L4 | **`sycl::get_pointer_type` per MoE submit** | `MoeExpertOps.cpp:5073`, `:5173` | a USM allocation query per projection (2/layer) costs 1-3 us and may walk the L0 allocation map. | CPU-only micro-bench: call it on a device USM pointer in a tight loop; 0 GPU-min |
| L5 | **block glue misread as submit** | `RunHyperConnectionCombined` `Qwen4ExpBnb4Backend.cpp:1926-2045` (branch tests, `pending_mlp->ConsumeNorm`, `verify_rows_alignment_`/`s_small_rows_hc`, `data_as`), `add_shared_expert_to_y` `:130-158` | >=30% of the 18.6 ms is dispatch/glue, so "20 us x N" overstates the win from deleting submissions. | one env-gated counter arm wrapping each `submit` with `steady_clock` (same pattern as the `AILA_Q4E_ROUTE_REPLAY_COUNT` patch); or VTune self-time on our symbols |
| L6 | **hsplit/`stage` timers** | `:2364-2366` + 5 pairs/layer (`:2559/:2574`, `:2584/:2605`, `:2717/:2755`, `:2853/:2858`, `:2861/:2873`) + `stage()` `:2346`; on in the measured arms (`WORK_SUMMARY=1`) | ~12 clock reads/layer ~ 0.02-0.05 ms/token, not the 5x. | price `steady_clock::now()` (~20-30 ns here); or run one A/B with `WORK_SUMMARY=0` and read `result.json` rate |
| L7 | **`Context::queue()` accessor** | `Context.hpp:57` (`async_errors_->Rethrow()` + null-trace `CheckTrace`), called 2-4x/submit | one acquire-load atomic + null check per call => <0.1 ms/token. | count calls with a static counter in the probe; 0 GPU-min |
| L8 | **RAII / exception guards** | `RunResidentDecode:1253-1254` (`ResidentColdWork`), `:2528-2530` (`std::optional<GpuResidentLayerLease>`, dead in e10), `DecodePendingMlpResidual` ctor/dtor `:2311-2312` / `:1881-1885` | small (<0.05 ms/token); only the pending-MLP dtor can submit. | counters; low priority |

### 2.3 Already excluded (do not re-measure)

| refuted | evidence |
|---|---|
| CPU contention as the cause | probe: same-kernel submit 5.9 -> 7.0-7.4 us (+20%) with an 18-thread DRAM stream running |
| immediate command lists | probe: 41.5 ms/token (worse); e10 keeps `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` |
| execution trace / per-call logging | `AILA_EXECUTION_TRACE_DIR` unset in the e10 env => `ExecutionTrace::Check()` never runs (`Context.hpp:225`, `ExecutionTrace.cpp:172`); every `AILA_LOG_INFO` on this path is behind a one-shot `*Logged_` flag |
| uncached env reads | every env read reachable from the count==1 path is a function-local `static` (`Bnb4BitLinear.cpp:1484`, `:2336`, `:2356-2357`; `GatedDeltaNetOps.cpp:101`; `Qwen4ExpBnb4Backend.cpp:2276-2338`; `QsaIndexerOps.cpp:24`). The uncached `_dupenv_s` form (`EnvUtils.hpp:21-38`) has **no** call site in the loop. |
| oneDNN descriptor rebinding | no oneDNN primitive executes with `count==1`: `Bnb4BitLinear::forward` takes `packed_nf4_gemv_bf16` whenever `seq_len==1` and the weight is packed NF4 (`:1337-1352`, `:2408-2416`); `DenseLinear` takes `ops::bf16_gemv_bf16` whenever a raw weight is passed (`Qwen4ExpBnb4Backend.cpp:1876-1884`), which is how the QSA indexer is called (`Qwen4ExpPrefill.cpp:209`) |
| `data_as<T>()` cost | pure `static_cast` (`Tensor.hpp:105`), ~0 ns |
| event bookkeeping | `RecordResidentConsumer`/`RecordStagingConsumer` are reference assignments (`Qwen4ExpAdaptiveCache.hpp:218-221`); `FinishResidentCpuRead` is 2 stores (`Qwen4ExpExpertExecutor.cpp:1273-1277`) |

### 2.4 The structural reading

One layer = `[6 submits] -> [3/9 submits] -> 1 D2H submit -> 1 front submit -> 1 top-k submit ->
(host wait) -> 2 shared submits -> (CPU route classification) -> 4 MoE submits -> (CPU join) ->
1 accumulate -> 1 shared add`. There are **2 host SYCL waits** (router publication and the
previous layer's resident-input copy) and **4 dependency-carrying submits** in that chain.
Submissions that sit between two waits can batch (~6 us each); submissions adjacent to a wait,
a dependency, or a long host-only gap pay the standalone launch path (~20-35 us each). That is
the whole 5x, and it is why graph replay (which turns ~20 submits into 1 launch per layer) only
bought 13% in the probe: the waits themselves are not removed by a graph.

---

## 3. Fusion opportunities (ranked by estimated win x confidence / risk)

| rank | change | submits removed /token | anchors | win [est] | bit-exact (SSE sha)? | risk |
|---:|---|---:|---|---:|---|---|
| 1 | **Remove the per-layer resident-input wait** by double-buffering `x_host_` (2 pinned banks, rotate on the `cold_count_`-clean guard; `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1` is the zero-code version of the same idea) | 48 waits (0 submits) | `Qwen4ExpExpertExecutor.cpp:740-747`; guard `:741` already excludes unfinished CPU readers | -1.5...-2.8 ms | yes (bytes unchanged; the D2H stays but no longer blocks) | low-med: needs a second pinned buffer + rotation |
| 2 | `AILA_Q4E_DECODE_FUSED_HC=1` (in-tree, default off): (inject + norm + down) in one kernel, (up + mix + hidden update) in another | 96 | `Qwen4ExpBnb4Backend.cpp:1933-1974`; kernels `Qwen4ExpDecodeFusion.cpp:63`, `:225` | -1.0...-2.2 ms | yes - in-code "Bit-exact mirror of ..." of the ops it replaces | low (flag exists; first e2e measurement here) |
| 3 | `AILA_Q4E_MOE_SHARED_FUSED=1` (in-tree, default off): shared add folded into the ordered accumulate | 48 | call site `:2839-2842`; fused expression `MoeExpertOps.cpp:6402-6406`; split expression `Qwen4ExpBnb4Backend.cpp:143-153` | -0.3...-0.7 ms | yes by expression identity: both sides compute `bf16(float(bf16(total_val)) + gate_act*float(shared_out))`; the split path's intermediate store is an exact bf16 round-trip | low |
| 4 | Drop the 64 B `resident_weights_` H2D (`:1526`) and have the accumulate read `router_host_weights_` directly | 48 | `:1526` -> `MoeExpertOps.cpp:6379` (it already accepts a host `cpu_result_packet`, `:6376-6377`) | -0.3...-0.6 ms | yes (same floats) - keep the host source, never the device copy written by another kernel | med: USM-host reads from a kernel may be slow; measure before adopting |
| 5 | Merge the QSA indexer chain `prepare_q_and_cache_k` + `compute_block_scores_compressed` + `select_topk` into 1-2 launches (all three are device-side, no host readback between them) | 24 | `QsaIndexerOps.cpp:204`, `:1082`, `:1301`; chain `:1479-1511` | -0.5...-0.8 ms | **no by construction** - a fused kernel must reproduce the exact per-kernel float reduction order of the block-score dot and the exact selection; gate on SSE sha | med |
| - | Attention phase-1 + merge in one launch | 0 | `AttentionOps.cpp:476`, `:686` | - | - | **do not attempt**: phase 1 spans `(heads/G)*tiles ~ 204` work-groups at the production attended length 2049; the merge needs all partials. Only the <128-token regime has the fused JM kernel. |
| - | Shared gate/up + SwiGLU + down in one launch | 0 | `Bnb4BitLinear.cpp:1236`, `:1493` | - | - | **do not attempt**: two NF4 matvecs chained through a 640-wide intermediate; against the known-negative register/SLM list |
| - | HC (norm+down+up) into a single kernel | 0 | `HyperConnectionOps.cpp:241`/`:380`/`:460` | - | - | **do not attempt**: the up projection needs the complete down result (2560 columns); a cross-work-group dependency cannot live in one launch |

Notes: the shared expert is *already* submitted before the router wait (`:2717-2755` before
`:2776`), so it needs no reordering. The resident MoE kernels cannot move before the route is
known; overlapping them with the wait requires speculative execution (P7's C2), out of scope.
Anything in Section 4 must not silently change the arithmetic of row 5 or any "do not attempt"
row above.

---

## 4. Graph / device-side-parameter feasibility

What the probe implies: static graph replay of the same `20 kernels + 1 wait` loop measured
**12.7 vs 14.6 ms/token (-13%)**, i.e. the graph removes per-submit host cost but not the
per-layer wait nor the CPU join. On DG2 there is no in-place kernel-parameter update, so a graph
cannot be replayed with different pointers/scalars unless the *kernels read them from a device
buffer*.

Per-layer values that must become device-readable before a graph can span a layer:

| parameter | today | live value |
|---|---|---|
| token position `start_pos` | host argument to `decode_prepare_qgkv_packed_partial` `ElementwiseOps.cpp:1486`, QSA `Qwen4ExpPrefill.cpp:213`, GDN `:77` | changes every token => device scalar written before launch |
| `cached_len`, `num_complete_blocks`, `num_active_blocks` | host ints (`Qwen4ExpPrefill.cpp:19`, `:175`, `:213-245`) | derived from position => device scalar |
| `resident_selected_[]`, `hit_ranks[]`, miss list | host re-derived inside `RunResidentDecode` `:1392-1404` from `router_host_ids_` | id/weight arrays are already device-written by the top-k kernel (`MoeExpertOps.cpp:123`); slots/ranks are host classification |
| `MoeExpertSlotDev` descriptor arrays | host stack arrays `:1463-1466`, `:1482-1485` | layer/residency state => small device table |
| per-layer base pointers (weights, `buf_.x/y/hidden4`, `moe_h`, `resident_down`, shared buffers) | captured by value per submit | constant per layer => one graph **per layer** (or per layer kind) makes these graph constants |
| CPU cold result packet (`cpu_result_packet`) and the CPU join | host (`expert_executor_`) | **hard split**: cannot be inside a graph |

So the layer decomposes into two graphable units around the CPU join:

* **S1 (attention subchain, GDN layers only, 36/48 layers):** attn HC (3) + all_proj + GDN decode
  + o_proj/history (3) = 6 submits -> 1 graph launch. No host readback inside. Needs the
  `start_pos` scalar in device memory and a per-layer graph instance (or a 36-instance graph
  array). Pre-registered gates: bit-exact SSE sha over a 32-token stream; `hc_host_ms +
  linear_host_ms` drop >= 40%; e2e >= +2% on the 14K-continue arm; A/A <= 1.5%.
* **S2 (MoE subchain):** resident gate/up + down + weights memcpy + accumulate (4) with the
  descriptor table and rank arrays device-resident; the routing ids are already device-written.
  Gate: sha identical; `experts_host_ms` - >= 20%. This is also the unit where the L2
  dependency submits disappear (a graph edge replaces them).
* **S3 (full-layer graph):** only after S1+S2 plus a CPU-cold strategy that does not need the
  host in the middle (or an accept of a *two-graph* layer). Gate: sha identical, >= +5% e2e,
  no new SYCL waits per layer. VRAM: each static graph instance owns its command buffers -
  price it against the current admission (static 4522 MB + session 622 MB + reserve 3584 MB +
  7204 MB experts of 15933 MB) before scaling to 48 instances.

---

## 5. Next experiments (ranked, ready to execute)

All arms use the existing harness (`python tmp/q4e-sprint-r1-20261014/run.py <label> --prime
--diag --generate=32 ...`) with the e10 bundle env on every arm; gate = SSE **content sha
identical** in all arms, A/A spread <= 1.5%, adopt at >= +3.0% e2e rate.

| # | experiment | cost | expected e2e | why it is the right next move |
|---:|---|---:|---|---|
| 1 | A/B/B/A on `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1` (0 build) | ~16 GPU-min | -1.5...-2.8 ms/token (-3...-5%) | removes 48 per-layer SYCL waits with zero arithmetic change; the flag already exists and e10 ran it at 0 |
| 2 | A/B/B/A on `AILA_Q4E_JOIN_RELEASE=1` (the in-tree `RunResidentDecode` release gate; needs the worker rebuilt once) | build + ~16 GPU-min | -4...-10 ms/token [est] | the probe measured -14% for the same release-before-condvar pattern; the code is already in the tree pending this e2e gate |
| 3 | A/B/B/A on `AILA_Q4E_DECODE_FUSED_HC=1 AILA_Q4E_MOE_SHARED_FUSED=1` (0 build) | ~16 GPU-min | -1.3...-2.9 ms/token | deletes 144 of 987 submissions with in-code bit-exactness arguments; cheapest test of "submit count is the lever" |
| 4 | 1x VTune GPU-Offload / Level-Zero attribution on a decode arm (no build) | ~6 GPU-min | 0 (picks the fix) | splits the 18.6 ms between UR/L0 submit self-time, our block glue (L5) and wait time; decides whether to chase per-call cost or call count |
| 5 | Build the S1 attention-subchain graph (36 GDN layers) behind an env flag and run 3 arms | build + ~12 GPU-min | -1...-3 ms/token [est] | prices graph replay on the *live* path (the 13% probe is the bound); also produces the device-parameter machinery S2/S3 need |

Fallback if experiment 4 shows the cost is inside our own glue: then experiment 3's win is the
ceiling, and the next step is L3/L5 micro-work (functor slimming, `get_pointer_type` hoisting)
rather than more fusion.

---

## 6. Open items / risks

1. `[est]` numbers above reuse the probe's 5.9-7.4 us and 59 us constants from the bottleneck
   audit (same machine, same queue env); if experiment 4 contradicts them, re-derive before
   trusting ranks 2-5 of Section 3.
2. The measured arms run with `AILA_Q4E_WORK_SUMMARY=1`; the counters themselves (L6) should be
   re-priced once, because every A/B gate in Section 5 compares "same config + one flag".
3. The two uncommitted patches are load-bearing for this document's anchors (see header). If the
   `MoeExpertOps` fence fix is reverted, the prefill-only fence bug from the bottleneck audit
   returns; if the `JOIN_RELEASE` patch is reverted, experiment 2 has no code to gate.
4. `count == 2..4` (verify windows / P7 batching) has a different submission inventory (extra
   per-row attention and verification memcpys) and was not counted here.
