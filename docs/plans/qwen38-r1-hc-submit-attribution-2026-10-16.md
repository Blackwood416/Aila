# R1 HC submit attribution: the 5x is not in our HC call sites (2026-10-16)

Read-only source study. No GPU, no build, no src edit, no commit; this file is the only write.
Tree read: `ecf6c69`, clean (`git status --porcelain` empty). All anchors were re-derived today
with `rg -n` against this tree; the earlier audits' anchors (base `bb07130` + 2 uncommitted
patches) do not all match and are not reused. Numbers quoted from
`docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md` (here **A16**, sections 1/2),
`docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md` (**B16**, section 3.3 receipts) and
`docs/plans/qwen38-r1-decode-misc-2026-10-15.md` (**E5**, sections 2/3, the 194-kernel bench).

Denominator note. A16 counts **291** HC submissions/token (97 calls x 3 submits); E5 divides the
same `hc_host` by **194** (97 pairs x 2 = the bench's replay). So "22.8 us/submit" (6.62/291) and
"34 us/submit" (6.62/194) are the same measurement with different denominators. This document
keeps 291 for production and 194 for the bench, and reports both.

## 0. Conclusions

1. Production HC submits/token = **291** = 97 x (fused inject+norm, down, up); 97 = 48 attn_hc +
   48 mlp_hc + 1 mixer. The bench replays only the **194** down+up kernels (E5 `kHcPairCalls`,
   `perf/qwen4exp/q4e_decode_misc_bench.cpp:68`), i.e. it never submits the fused norm kernel.
2. The production per-submit host work that is *ours* is ~10^-7 s: the ops bodies are the same
   code the bench runs, the three submits are straight-line calls, and there is **no** per-submit
   allocation, try/catch, mutex, virtual/factory dispatch, event copy, env read or log macro
   (section 3, with anchors). The bench even adds work production lacks (a `std::function` hop,
   `q4e_decode_misc_bench.cpp:359-370`). "Our call-site work" cannot contain ~28 us.
3. What the bench does not reproduce is **queue state and adjacency**: (a) the third (norm)
   submit per call; (b) per-layer distinct USM pointers instead of 8 reused buffers; (c) the
   per-layer memcpys/waits/event query on the same in-order queue; (d) host at parity with the
   device (A16/B16: host ~60 ms vs device ~57 ms per token) instead of host-behind-device
   (bench: host 1.24 ms for 194 kernels whose device span is ~3.3 ms).
4. Therefore the 5x is located **inside `sycl::queue::submit` as a function of queue state**, or
   in the *attribution* (the block timer is not 291 submit timers). The code cannot distinguish
   these two; one cheap experiment can (section 5).
5. Ranked candidates: C1 queue-boundary pacing inside submit (high x high), C3 block-vs-submit
   attribution (high x high), C2 the un-replayed norm submit (med-high x high), C5 pointer
   working-set/L0 allocation-map lookups (med x med), C4 our call-site work (low x very high).
6. Cheapest decisive experiment: an env-gated `steady_clock` around the three `submit` calls in
   `HyperConnectionOps.cpp` (plus a counter), printed in the existing `[Q4E AdaptiveWork]` line;
   one decode arm on the e10 bundle. Read: ~5-7 us/submit => submit is warm and the 5x is in the
   block around/near it (C3/C4); ~20-35 us => it is the SYCL/UR submit itself (C1/C2/C5), and the
   per-op split says which.

## 1. Complete HC submission inventory (decode T1, e10 env: HC_FUSED=1 default, DECODE_FUSED_HC=0)

Per-layer call sites: attn HC `Qwen4ExpBnb4Backend.cpp:2621-2624`, MLP HC `:2674-2682`, mixer
`RunHyperConnectionCombined(ctx, mixer_, 1, false, ...)` `:3083-3086`. All three go through
`RunHyperConnectionCombined` (`:2114-2233`).

| # | caller -> submit site | op (kernel) | T1 geometry | constructs/captures per submit | in bench? |
|---:|---|---|---|---|---|
| A1 | `:2165` `ConsumeNorm` (`:2088-2106`) -> `HyperConnectionOps.cpp:312` | `try_hc_decode_inject_add_and_grouped_rms_norm` (:301) | `nd_range<1>(4*256, 256)`; `local_accessor<float,1>(256)` = 1 KiB SLM | alignment tests on `H`/`y` (:306-309); CG lambda `[&]` (:312); kernel `[=]` captures `H,y,inj,norm_w` + `eps,out_N,group_dim,plus_one` | **no** |
| A2 | `:2201-2206` -> `:584` -> dispatch `:436-446` -> impl `:367`, submit `:380` | `hc_decode_down_inject` | `nd_range<1>(ceil(324/16)*256, 256)` = 5376 thr | template switch on `work_group` (3 instantiations); kernel captures `N,w_down,w_inject,act,inj` + `lowrank,k_dim,k8,total_rows,num_wgs,inv_groups`; no SLM | yes |
| A3 | `:2207-2212` -> `:590` -> impl `:448`, submit `:460` | `hc_decode_up_mix` | `nd_range<1>(160*256, 256)`, `num_wgs=ceil(2560/16)` | kernel captures `act,w_up,N,x` + `lowrank,groups,k8,num_wgs,dims_per_wg,rows,...`; no SLM in the non-streams variant | yes |
| B1 | `:2674-2676` (pending_y/pending_inj path) -> `:2168-2175` -> `HyperConnectionOps.cpp:241` | `hc_inject_add_and_grouped_rms_norm` (:224) | `nd_range<1>(4*256, 256)`; SLM 256 floats | `red_slm` local_accessor; per-thread `bf16 v_reg[16]` register cache; kernel captures `H,y,inj,norm_w,out_N,rows,groups,group_dim,eps,plus_one` (`inj` always non-null here) | **no** |
| B2/B3 | `:2168` block falls through to the same `:2201`/`:2207` calls | `hc_decode_down_inject` / `hc_decode_up_mix` | as A2/A3 | as A2/A3 (different weight pointers per layer) | yes |
| M1-M3 | `:3083-3084` (mixer, `with_inject=false`) | norm path + down with `w_inject=nullptr` (`total_rows=320`) + up | as A1-A3 | as A1-A3 | pair only |

Per-call scalar work in `RunHyperConnectionCombined`, all function-local `static`s (one guarded
byte check after the first call; `read_flag` runs once per process, `src/utils/EnvUtils.hpp:18-38`):
`s_fused_hc` `:2119`, `s_decode_fused_hc` `:2128`, `s_small_rows_hc` `:2189`, `s_hc_down_wg`
`:2190-2194` (raw int + 64/128/256 validation), `s_hc_up_streams` `:2195`, plus member tests
`decodeHcReferenceForTest_`, `verify_rows_alignment_` `:2197-2199` and 6 branches on
`hc.mix_down_w/mix_up_w/inject_w`. Nothing here allocates, locks, formats or copies.
Not taken at T1: `hc_decode_*_rows` (`:2191-2198`, count 2-4 only), `DenseLinear` fallback
`:2222-2233`, `HcDecodeVariant` (bench-only, `perf/qwen4exp/q4e_decode_misc_bench.cpp:29`),
`hc_decode_up_mix_streams_impl` (`hc_up_streams=false`).

## 2. The bench arm vs the production path

| aspect | bench `q4e_decode_misc_bench.cpp` | production |
|---|---|---|
| kernels replayed | 194 = 97 down+up pairs (`:68`, `:280-285`, `:359-377`) | 291 = 97 norm+down+up trios (section 1) |
| queue | `Context ctx(nullptr, true)` `:704` -> **profiling** in-order queue | worker: plain in-order queue, profiling only if `AILA_Q4E_DEVICE_QUEUE_PROFILING=1` (`src/inference/Model.cpp:42-60`) |
| USM pointers | one `HcFixture` (`:234-270`): 8 buffers reused for all 97 calls | per-layer weights/activations (`buf_*`, `layers_[L].*_hc`, KV/state caches): thousands of distinct pointers per token |
| queue neighbours | 2 `ext_oneapi_submit_barrier` per sample (`:163-167`), nothing else | per layer: D2H copy + `wait_and_throw` (`Qwen4ExpExpertExecutor.cpp:745-746`, called at `Qwen4ExpBnb4Backend.cpp:2695/2719`), route wait (`:2808-2810`, `:2827`, `:2949`), 40 B H2D memcpy (`:1571`), `try{get_info}` release (`:1558`), shared/MoE submits |
| host/device balance | host 1.24 ms vs device span ~3.3 ms -> host behind device the whole arm | host ~60 ms/token vs device ~57 ms/token (A16/B16) -> host at parity, i.e. every batch launch is paced by the GPU |
| per-call host wrapper | `std::function` hop per pair (`:359-370`) | direct call, no indirection |

The bench therefore already pays everything our code does per submit (`ctx.queue()` accessor,
command-group lambda, `local_accessor` construction for the kernels that have SLM, kernel
argument marshalling) and pays one extra virtual-ish hop. A 5x cannot come from "the bench is
missing our call-site work"; it must come from the two rows that differ in *state* (pointers,
neighbours) and the one row that differs in *content* (the 97 norm submits the bench never does).

## 3. `Context`/queue wrapper and the suspect checklist

`Context::queue()` (`src/core/Context.hpp:57`) = `async_errors_->Rethrow(); CheckTrace(); return q_;`.
`Rethrow` (`src/core/AsyncErrorState.hpp:31-33`) is one acquire load on the fast path (`Failed()`),
the mutex is taken only after a recorded fault; `CheckTrace` (`Context.hpp:224-225`) is a null
check on a borrowed pointer. No batching flush, no event bookkeeping, no queue copy. The ops call
it once per submit and the bench uses the same accessor.

| requested suspect | verdict | anchor / evidence |
|---|---|---|
| per-submit heap allocation | **absent** | `HyperConnectionOps.cpp` contains no `vector/string/new/allocate`; all HC buffers are preallocated once in `ensure_runtime_buffers` (`Qwen4ExpBnb4Backend.cpp:1062-1063`); `Tensor::data_as` is a cast (`src/core/Tensor.hpp:105-108`) |
| per-submit `std::optional`/`std::string` | absent in HC ops | `std::optional` exists only for the resident leases, once per layer (`:2701-2702`) |
| per-submit try/catch | absent in HC ops | only `:2072` (dtor, 1/forward), `:1558` (1/layer, JOIN_RELEASE release), `:3488` (per-request) |
| per-submit mutex/lock | absent on this path | `DeviceState::mutex` (`Model.cpp:66`) is not taken by `ForwardImpl`; `AsyncErrorState` mutex only on fault; audit A16 2.3 refuted logging/locks generally |
| per-submit virtual dispatch / factory | absent | ops are direct calls; the only dispatch is the `switch(work_group)` `:436-446` (branch, 3 template cases) and `parallel_streams` `:590-594` (`false` at T1); no `std::function` in the production path |
| per-submit `sycl::event` copies | no large copies | the ops discard the returned event (temporary); `resident_input_event_` = 1 assign + 1 `!=` compare per layer (`Qwen4ExpExpertExecutor.cpp:745-746`); `std::vector<DevMark>` only when `AILA_Q4E_DECODE_STAGE_EVENTS=1` (`:2566-2571`), off in e10 |
| per-submit `range<3>`/`nd_range` recomputation | integer arithmetic only | e.g. `:371-375`, `:456-458` (~tens of ns) |
| per-submit `read_flag`/env statics | cached | 5 function-local statics per HC call + ~25 in `ForwardImpl` (`:2449-2513`); guarded static = byte test, `read_flag` runs once (`EnvUtils.hpp:18-38`) |
| per-submit logging at O2 | absent | zero `AILA_LOG` in `HyperConnectionOps.cpp`; `ForwardImpl` logs sit behind one-shot `*Logged_` flags / `s_work_summary` (A16 2.3) |
| real per-call constructs near HC (per **layer**, not per submit) | exist but small | `std::array<ops::MoeExpertSlotDev,16> hot{}/cold{}` zero-init ~1 KiB each (`:1515`, `:1534`, `:1603`, `kMaxTopK=16` `:1283`); 40 B `queue().memcpy` H2D (`:1571`); `DecodePendingMlpResidual` ctor (`:2499`, 1/forward); `hsplit` clock pairs (`:2537-2541`, active in the measured arm, ~ns) |

## 4. Candidates ranked by plausibility x testability

| rank | id | candidate | mechanism | why it can add ~28 us | cheapest falsification |
|---:|---|---|---|---|---|
| 1 | C1 | **submit is paced by the in-flight batch** (UR L0 v1, non-immediate, in-order queue) | the queue holds one open command list; a submit that opens/refills a batch launched right after a wait/copy/query pays the previous batch's launch/completion, a submit that appends to an already-running batch is ~6 us (B16 3.3: empty kernel+wait 59 us; 48x20 kernels + per-layer wait 13.8-14.6 ms vs one wait 8.3 ms; batch=1 = 43.9 ms/token) | production has >=3 boundaries/layer (D2H wait `:745`, route wait `:2810`, 40 B H2D `:1571`, event query `:1558`); if 1 of every 3 HC submits is boundary-cold at ~55 us, (55+6.4+6.4)/3 = 22.6 us ~= measured 22.8 us; a uniform fill `cost = 6.4 + boundary_cost/submits_per_batch` also fits | per-submit timer (section 5): expect a bimodal distribution (cold ~40-60 us, warm ~6 us). Replaces nothing else |
| 2 | C3 | **attribution**: `hc_host_ms` is a block timer, not a sum of 291 submit timers | `stage_ms[kStageHc]` accumulates wall time between `stage()`s (`:2528-2542`, `:2633`, `:2689`); dividing by 291 assumes every ns is submit cost, which is unverified (A16 2.1 "these are block costs") | if ~2-3 ms/token of the block is host work outside `submit` (std::function-free but still block glue), the true per-submit inflation is smaller than 5x and the fix is different | the same per-submit timer: `sum(submit_us)` vs `hc_host_ms`; no extra run needed |
| 3 | C2 | **the un-replayed fused norm submit** | bench replays 2/3 of the HC kernels; the norm submit is the first submit of each trio, i.e. the one adjacent to the previous layer's last flush point, and it is the only HC op production runs that the bench never runs (`:2168`, `:241`, `:312`) | if the 97 norm submits cost ~49 us each (cold launch + 1 KiB SLM + 4-WG kernel), they alone add 4.8 ms to a 1.86 ms warm rest = the whole 6.6 ms | per-op timer breakdown (norm vs down vs up) in the same run |
| 4 | C5 | **pointer working set / L0 allocation lookups inside submit** | production marshals ~6 pointers/kernel from thousands of distinct USM allocations per token, the bench marshals the same 8 pointers 97x; any per-argument allocation/property lookup in UR/L0 has a hot vs cold cache behaviour | ~6 pointer lookups x a few us/miss x 291 submits = ms-scale | bench-only arm: rotate 48 distinct weight/out pointer sets through the same 97 calls; if `host_us_per_kernel` rises to ~20 us the mechanism is real and worker-free |
| 5 | C4 | **our call-site work in the HC ops** | none found: same ops, same captures, fewer constructs than the bench (section 1/3) | cannot: the enumerated work is ~10^-7 s/submit; would need 28 us of unexplained arithmetic | `-ftime-report`-style codegen check is unnecessary; the timer in section 5 would show up as `hc_host - sum(submit_us) > 0` with no candidate |

Fit arithmetic (all three fit "6.62 ms / 291 = 22.8 us" and cannot be separated without per-submit
data): (a) 1 cold + 2 warm per trio = 22.6 us; (b) uniform 16 us of added pacing on top of the
bench's 6.4 us; (c) the norm submits alone carrying the whole gap (4.8 ms) on top of a warm rest.

## 5. Single cheapest experiment: split "SYCL/UR submit" vs "our work"

Primary (one small env-gated patch + one decode arm; no arithmetic change, no default-on code):

1. In `src/ops/HyperConnectionOps.cpp`, wrap each `ctx.queue().submit(...)` of the HC ops
   (`:312`, `:380`, `:460`, and `:241` for the non-pending path) with two `steady_clock` reads
   guarded by `static const bool s_hc_submit_timing = aila::env::read_flag("AILA_Q4E_HC_SUBMIT_TIMING", false);`
   and accumulate `count/min/median/max` per op id (same read-once pattern as the existing
   `AILA_Q4E_ROUTE_REPLAY_COUNT` counter cited in A16 2.2/L5). No new includes beyond `<chrono>`.
2. Print the aggregates in the existing `[Q4E AdaptiveWork]` line (`Qwen4ExpBnb4Backend.cpp:3214-3250`)
   next to `hc_host_ms` (same env gate as `WORK_SUMMARY`, which the e10 bundle already sets).
3. Run exactly one decode arm with the e10 bundle env; read `hc_submit_us/count` and the per-op
   split against `hc_host_ms` = 6.62 ms.

Read map: submit ~5-7 us => the 5x is **not** inside `submit`; the block's remaining time is
non-submit host work and the attribution (C3) plus "no candidate in our code" (C4) is the answer,
so the next target is the block rather than the kernel count. Submit ~20-35 us => the 5x is the
**UR/L0 submit under production queue state** (C1/C2/C5); the per-op split then separates C2
(norm alone) from C1/C5 (all three).

Zero-build pre-step (10 min, no code): `AILA_Q4E_DEVICE_QUEUE_PROFILING=1`
(`Model.cpp:44`) + `AILA_Q4E_DECODE_STAGE_EVENTS=1` + `AILA_Q4E_WORK_SUMMARY=1` gives
`hc_dev_ms` next to `hc_host_ms`. Caveat from B16 3.4: marker-to-marker spans attribute idle time
to the next bucket, so this only bounds device vs host; it does not split submit from call-site.
Second-stage if the timer says UR-side: add a bench arm that (a) rotates 48 distinct pointer sets
and (b) puts a GPU-heavy filler before the 194 kernels, and re-read `hc_token_prod
host_us_per_kernel` (identical ops, no worker).

## 6. Do not re-measure (carried from A16 2.3 / B16 3.3)

CPU contention (+20% max), execution trace/logging, uncached env reads, oneDNN descriptors,
immediate command lists (41.5 ms/token), and `data_as` are all refuted with receipts.
Also note `docs/plans/qwen38-r1-fused-regression-analysis-2026-10-16.md`: the in-tree 2-submit
`AILA_Q4E_DECODE_FUSED_HC` path was a NO-GO (-19.8%, numeric defect) - deleting HC submissions
is not a substitute for moving the host-side cost; do not re-screen it before the defect is fixed.

Sources: `src/ops/HyperConnectionOps.cpp`, `src/models/Qwen4ExpBnb4Backend.cpp`,
`src/models/Qwen4ExpExpertExecutor.cpp`, `src/core/Context.hpp`, `src/core/AsyncErrorState.hpp`,
`src/inference/Model.cpp`, `perf/qwen4exp/q4e_decode_misc_bench.cpp`, A16, B16, E5.
