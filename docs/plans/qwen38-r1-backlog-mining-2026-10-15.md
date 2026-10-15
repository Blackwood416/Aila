# R1 backlog mining: proposed-but-unfinished candidates and measurement gaps

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, HEAD `9abb858`, pair
`retained23` (worker `1b988254...`, server `ed71ae3d...`).

Role B2 (history mining). Scope: read-only sweep of `docs/plans/qwen38-*`,
`docs/Qwen38FlashNext_A770_Handoff.md`, `tmp/q4e-*` reports/logs/scripts and
`tmp/Strata/` with the goal of recovering items that were **proposed but never
implemented / never measured / shelved**, plus the measurement gaps that block
ranking them.

Discipline: no `src`/`tests`/`tools` edits, no build, no GPU run, no git write.
Only this document plus `tmp/q4e-sprint-r1-20261014/backlog/`. Every item
carries a `file:line` or artifact path; every projection is tagged `[estimate]`;
numbers are copied from measured artifacts, never invented. Text is ASCII.

Exclusion rules applied: nothing from the charter S2 negated list
(`docs/plans/qwen38-performance-research-charter-2026-10-14.md` S2) is listed
unless a mechanism difference is stated; the five already-closed items named in
the task (L6 CPU/GPU cap, L14 C2 PLE lookahead, L12 router_wait
re-attribution, S service batching, chunk 8192 / single chunk, layer-major
current v1-v3 shapes) are treated as closed and appear only as backlog
*next steps* where a report explicitly names one.

---

## 0. Three highest-value unfinished items

**#0.1 - Cross-layer pipelined layer-major prefill (v4).** The layer-major
*mechanism* is proven (H2D and uploads drop ~75%) but every measured shape is
wall-negative because each layer's staging forms a barrier with its expert
compute (`docs/plans/qwen38-r1-cost-model-2026-10-14.md:451-464`, L27-L31;
`docs/plans/qwen38-r1-layer-major-proto-2026-10-15.md` S8). The named next
action - stage layer L+1 while layer L computes - is the only remaining
untried form and is already commissioned to subagent A1-v4
(`tmp/q4e-sprint-r1-20261014/agents/a1v4-pipeline.txt:5-9`). Ranking basis: the
P-1 export floor is 55.98 GiB vs 179.5 GiB observed for the 14K request, i.e.
128.1 GiB = 14.95 s at 9.2 GB/s = `-21% TTFT` if H2D were the only term
(`docs/plans/qwen38-r1-prefill-supply-2026-10-14.md` S3.1-S3.2).

**#0.2 - Decode-attention GQA K/V sharing (12x re-read).** `attention_decode`
at len 14336 / Hq24 / Hkv2 / D256 measures 0.792 ms/row against a 0.101 ms
minimum-byte floor (7.85x), with partial L2 absorption (0.412 ms at 854.6 GB/s
predicts 1.92x below measured) `[estimate]`
(`docs/plans/qwen38-r1-kernel-2026-10-14.md` S2.2, table row e citing
`docs/plans/qwen38-int8kv-vec-2026-10-12.md:23`). 12 full-attention layers of a
decode token are ~9.5 ms/token `[estimate]` against a 66 ms/token wall. This is
not on the charter S2 negated list, the QSA cross-head route is a *different*
lever (closed at 4096, `qwen38-r1-cost-model-2026-10-14.md` S6.8), and a
dedicated bench arm plus design is commissioned (`agents/k3-decode-attn.txt`).

**#0.3 - Close the decode cost-model residuals with current instrumentation.**
The two largest unmeasured decode terms are the unprinted MoE-front/shared
residual (9.09 ms/token, 13.8%) and the device time of "hc + linear + full"
(12.46 ms/token of host submits with no device split)
(`docs/plans/qwen38-r1-cost-model-2026-10-14.md:126-136`, table rows
"unprinted residual", "hc + linear + full submits";
`docs/plans/qwen38-r1-decode-2026-10-14.md:50`). Both are pure instrumentation
additions on an existing pair (the `kStageRouter`/`kStageShared` prints and a
decode-safe timing arm), and the stage-serialized arm is explicitly not usable
at decode scale (28-54 ms/token of sync, `r1-cost-model` S3 and S7 item 1).

---

## 1. Candidate table (ranked by expected gain x verifiability, 19 items)

Legend: **Gain** is an upper bound with its basis; `[estimate]` marks anything
derived. **Oracle** is the cheapest falsification route. `--` means not
applicable. Line references to `docs/plans/...` are abbreviated by filename
only where the file is unambiguous.

| # | Candidate | Where proposed | Status today | Why revisit | Minimal verification (command / field / criterion) | Gain |
|---|---|---|---|---|---|---|
| C1 | Cross-layer pipelined layer-major prefill (v4) | `qwen38-r1-layer-major-proto-2026-10-15.md` S6/S8.3 ("the next step is a cross-layer pipelined variant"); `r1-cost-model` S7 item 4 | v1-v3 built and measured; mechanism valid, wall -67%; committed opt-in/default-off (`9abb858`, L31). v4 is in flight in another subagent, no result yet | Validated H2D floor; prefetch/staging overlap is already the chunk-major norm | `python tmp/q4e-sprint-r1-20261014/run.py lm-v4-a --diag --chunk=8192 --prompts=6000,8192 AILA_Q4E_LAYER_MAJOR=1 AILA_Q4E_LAYER_MAJOR_PIPELINE=1`; read 48x `[Q4E LayerMajorLayer] wall_ms/bank_wait_ms`, `[Q4E LayerMajor] h2d_mb`, run-log `first_content_seconds`; gate TTFT <= 35.2 s and H2D <= 54.7 GiB | chunk-major 8192 baseline 37.06 s; floor would be ~29.7 s at the -21% bound `[estimate]` |
| C2 | Decode-attention GQA K/V sharing (kill the 12x re-read) | `qwen38-r1-kernel-2026-10-14.md` S2.2 + ranking row 3; `qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md` unimplemented item 3 | Measured 0.792 ms/row/layer; no variant exists; K3 bench arm designed but not merged (`agents/k3-decode-attn.txt`) | Largest measured distance-to-bound among probe-ready kernels; not in S2 | New `--decode-attn-bench` (len 4096/8192/14336, Hq24/Hkv2/D256, rows 1-2, JSON): treat >=2x as a lever; then 12-layer account vs 66 ms/token wall | 0.792 -> 0.412 ms/row at L2 bound = `-0.38 ms/row`; 12 layers ~`-4.6 ms/token` = `-7%` `[estimate]` |
| C3 | PLE first-chunk exposure after F1 (fixture-specific residual) | `qwen38-r1-cost-model-2026-10-14.md` S6.7/S7 item 8; `qwen38-r1-ple-firstchunk-2026-10-15.md` S2/S4 | F1 landed default-on (-37% on the fixture first request, L26); the residual cause on the fixture corpus was not isolated (`ple_host` 20.5 s before F1, 0.5 ms from chunk 2) | Only remaining first-request cost after warmup; mechanism unknown (content-dependent page set vs gather scheduling) | `run.py plepf-* --diag --prompts=6000 AILA_INIT_WARMUP=1` with/without `AILA_Q4E_PLE_PREFETCH`; read first `[Q4E AdaptiveWork] pos=0 count=4096` `ple_host_ms`; criterion: <=0.25x baseline with cold-page control between arms | Pre-F1 gap 20.5 s/chunk; F1 measured -18.7 s on the fixture `[measured]`, residual unquantified `[estimate 0-5 s]` |
| C4 | Decode/kernel device-time split for hc + linear + full (12.46 ms/token of submits) | `qwen38-r1-decode-2026-10-14.md` S2 table rows `hc_host`,`linear_host`,`full_host` + S4 R3; `r1-cost-model` S7 item 1 | No decode-safe arm; the stage-serialized arm costs 28-54 ms/token and is diagnostic-only (L11 caveat) | 18.9% of the decode wall is submit time with unknown device content; gates any K-scope work | Add a decode arm that syncs only after hc+linear+full (not per stage) or use event intervals; field = new `stage_ms` pair for decode; criterion: report device vs host split at <2 ms perturbation | If device time is small the 12.46 ms is host-side and addressable; if large it is kernel scope `[estimate]` |
| C5 | MoE-front / shared-expert submit breakdown | `r1-cost-model` table row "unprinted residual" 9.091 ms (13.8%); `r1-decode` S2 row "MoE front + shared submit (derived, not printed)" 8.684 | Derived by subtraction only; `kStageRouter` (Backend.cpp:2315) and `kStageShared` are not printed | 13.5-13.8% of decode unexplained is the single largest attribution hole; `AILA_Q4E_MOE_FRONT_FUSED` exists and is off | Print the existing stage enum for router/shared in `[Q4E Stages]`/`AdaptiveWork`; read `router gate GEMV` vs `top-k` vs `shared gate_up/down` vs `hc_inject_add`; criterion: residual < 1 ms | Splits 9.09 ms/token into 4 terms; upper bound of any of them is `-13.8%` of the wall if it went to zero `[estimate]` |
| C6 | Layer-major 8192 residual decomposition | `r1-cost-model` S6.10; `r1-layer-major-proto` S7.3 residual risk 2 / S8.3 | v3: `experts_ms` 22.4 s, `wall_ms` sum 61.6 s, i.e. 38.7 s non-expert residual; suspicion is 8192-row capacity scratch + no cross-layer overlap; not measured per module | Explains why the mechanism's H2D win never reaches the wall; needed to decide whether v4 or per-module shape is the fix | Read `[Q4E LayerMajorLayer] experts_ms/wall_ms` per layer plus a per-module variant (PLE / attn HC / MLP HC / mixer at 4096 vs 8192); criterion: attribute >=80% of the residual to named modules | Residual 38.7 s on a 37.1 s baseline is `>100%` of the target window - this is the whole layer-major GO/NO-GO `[estimate]` |
| C7 | Consumer-side (not source-side) first-chunk PLE warming (F2/F3) | `qwen38-r1-ple-firstchunk-2026-10-15.md` S2 (F2 4-thread gather, F3 smaller first slice) | F1 chosen; F2 explicitly "only if F1 < 2x", F3 deferred; F1 was adopted without the P2 arm being run (no record of an F2 test) | F1's 12x was measured on a CPU replica; if the live arm shows <2x the fallback already exists; F3 changes chunk boundaries (QSA threshold discipline) | F2: same P1 protocol with `AILA_Q4E_PLE_PREFETCH=0` + 4-thread gather arm; F3: run.py with a 1024-token first slice and check QSA stays active (`prompt_tokens >= 2052`); criterion F2 <= 0.5x of F1 residual | F2 same class as F1 (20.5 -> 3-6 s `[estimate]`); F3 proportional to slice (20.5 -> ~5 s `[estimate]`) |
| C8 | Decode attention measurement for the fused multi-row path (`attention_decode_rows`, head_dim 128 < 128 tokens) | `docs/plans/qwen38-r1-kernel-2026-10-14.md` S1.1 note (partial-merge switch at 1024); `qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md` decode grouping note | Only single-row/near-single-row numbers exist; the fused multi-row JM path is described in AGENTS S3 but has no current-code timing | Decode is 66 ms/token; any 5%+ kernel item must be measurable at its real shape before K work is scheduled | Extend the C2 bench convention to `rows=2..4` with the head_dim 128 fused path and the >=128-token switch; criterion: single-shape measurement of >=3% change | Unquantified; bound is the same 12x re-read family as C2 `[estimate]` |
| C9 | Opt-in cold-start explicit read (`AILA_Q4E_ARENA_READ_THREADS`) hardening | `qwen38-strata-c7-fill-pipeline-2026-10-13.md`; `qwen38-c7-deep-load-breakdown-and-directstorage-2026-10-13.md`; `qwen38-strata-candidates-execution-2026-10-13.md` C7 row | Measured 116.6 -> 63.8 s cold (-45%) but hot cache regressed 68 -> 82 s, so it is opt-in/default-off and only for cold starts | Startup dominates the operator loop (warm 68 s, cold 116 s); the hot-cache regression is the only blocker to a conditional default | Policy arm: enable explicit read only when the OS standby state is cold (probe the same runner twice, `AILA_Q4E_LOAD_TIMING=1`); criterion: cold <= 65 s AND hot <= 70 s on the same binary | -45% cold start when the guard holds; -0 on hot `[measured]` |
| C10 | `GDN_CHUNK_MODE` routing-drift cause (decode -4..7%) | `qwen38-optimization-roadmap-2026-10-10.md` A3-1 verdict; `qwen38-qwen4exp-pipeline-status-2026-10-13.md` S3.2 | TTFT -2.7% reproduced twice, decode -4..7% attributed to "numerical change -> routing drift -> lower expert hit rate" but never measured directly | Attribution is a hypothesis; if the drift is really hit-rate, a hotset/profile fix could unlock the -2.7% TTFT for free | Run both modes with `AILA_Q4E_WORK_SUMMARY=1` and compare `hits`/`misses`/`staged` per token plus `[Q4E Adaptive Profile]`; criterion: routing overlap >= 99% or hit-rate delta < 1pp | If drift is the cause: recover ~2.7% TTFT at zero decode cost `[estimate]` |
| C11 | Re-verify INT8 KV at the production reserve | `qwen38-int8kv-prod-2026-10-12.md`: `resident = min(slots, ranking.size)` caps intake at the 2792-key H4 profile | The +156-slot budget (2792 -> 2948) never materialised; measured +1.8% decode vs the <=+5% gate; `plan:259-277` | The intended mechanism was never actually exercised; the report itself proposes regenerating a >=2949-key profile | Regenerate the profile with >=2949 keys and rerun the production A/B (same `MAX_GPU_SLOTS` both arms); criterion: `[Qwen4Exp] Memory Admission Accounting` shows the delta and decode >= +5% | Expected +1.5-2.5% at 0.45-0.5%/100 slots `[estimate]`, below the default gate on its own |
| C12 | Hotset online adoption (`AILA_Q4E_HOTSET_INTERVAL`, `_BORROW`, `_ADOPT_*`) re-screen on the current pair | `qwen38-qwen4exp-pipeline-status-2026-10-13.md` S2.1 env table (interval 4, borrow 1, adopt_ready 1, adopt_limit 4096) | Switches are active in the production profile JSON and are never mentioned in any R1 screen or ledger line | Cheap multi-variable surface already wired; profile ordering is worth slices worth ~0.45-0.5%/100 slots | One A/B/B/A pair with `AILA_Q4E_HOTSET_INTERVAL=0` (adoption off) vs production; criterion: >=5% end-to-end or the surface is closed with evidence | Bounded by the profile-mix experiment (+1.1pp, closed) but the *online* mechanisms were never measured `[estimate 0-3%]` |
| C13 | JIT/warmup residual after the opt-in warmup | `r1-cost-model` S6.7/S7 item 2 (first large request 109.9 -> 51.2 s, startup +111 s); `qwen38-r1-ple-firstchunk` S1.3 | Adopted opt-in; residual ~20.5 s first-chunk PLE was later root-caused and fixed by F1; the +111 s startup cost was never reduced | Warmup is the only lever that removes the one-time ~80 s; its cost is a net loss for one-shot CLI use | Try a reduced warmup (fewer rows / decode-only) and read `[Q4E Warmup] total_ms` plus the first-request penalty; criterion: first request within 5 s of steady state at <= 60 s startup | Startup 247 -> ~180 s at constant first-request latency `[estimate]` |
| C14 | AOT device-image build (`-fsycl-targets=intel_gpu_acm_g10,spir64`) | `docs/Qwen38FlashNext_A770_Handoff.md:386-387` "Recommended next steps" item 5 | Never attempted; `SYCL_CACHE_PERSISTENT=1` still crashes on load (issue #4, reproduced L15) so there is no cache-based shortcut | The ~80 s first-use cost is uncacheable today; AOT removes JIT from the product path entirely | Quantify build time/size first (no GPU): `pwsh -File build.ps1` with the AOT target in a scratch build dir, then one cold-start run with `AILA_Q4E_LOAD_TIMING=1`; criterion: first-request penalty < 10 s | Removes the ~80 s first-use window and the +111 s warmup trade `[estimate]` |
| C15 | MTP closure options (a)/(b) | `qwen38-optimization-roadmap-2026-10-10.md` B4-6 closure; `qwen38-mtp-acceptance-2026-10-11.md` "next steps"; `qwen38-mtp-align-2026-10-11.md` | Equivalence gate failed at pos=1492 (rejected verify window row 0); MTP stays opt-in/default-off; throughput 15.0-15.3 vs T1 15.6 tok/s | Acceptance 74.5-74.8% is real; the cost is verify-window cost (1.70x step) and one operator, not the idea | Either (a) live stage probe at the first diverging row, or (b) force per-row `count==1` verify; criterion: interleaved MTP/T1 outputs identical on all arms | If (b) works: MTP becomes adoptable only where its verify cost falls; measured net is currently negative `[measured]` |
| C16 | Cross-head QSA sharing at 4096 (reopen only with an 8192-row window) | `r1-cost-model` S6.8 (4096: +0.6..+5.4%, round-dependent; 8192: +36.3..+37.0%) | Closed for the 4096 production shape (marginally under the 3%/two-shape rule) | The decisive 8192 number is `+36%` of that attention step; pairs naturally with any 8192-row prefill window (C1) | Re-run `probe-K2-qsa-route.ps1` only inside an 8192-row prefill arm; criterion: >=3% at 8192 and no regression at 4096 | ~37% of the full-attention prefill step at 8192 `[measured]`; 2.85 s/chunk at 4096 is the reference scale |
| C17 | Strata-style "borrow the head's VRAM for the prompt path" (prefill head alias) | `tmp/Strata/docs/PREFILL_HEAD_ALIAS.md:1-40` (RTX 5070: 768 -> 2048-token chunk at constant reserve, 387 -> 667 tok/s) | Nothing comparable exists in Aila; Aila keeps embed/lm_head handling separate (`AILA_Q4E_HOST_EMBED` gathers on host) and holds ~1.3 GiB of reserve | The Aila reserve decomposition (logits 970 + phase 403 + PLE 540 + HC 225 + staging 77 + misc 60 ~= 2275 MiB, headroom ~1.3 GiB) is exactly the kind of budget this technique reclaims (`qwen38-optimization-roadmap-2026-10-10.md` B3 budget audit, `plan:114`) | First measure what the head/embed actually occupies: `--memory-sample-ms 250` + DXGI during a 4096 block, a short append and cancel-recover (`plan:166`); criterion: quantified headroom before any aliasing design | Unquantified for Aila; Strata's Windows-side result is chunk-size, not tok/s, so port the *measurement* first `[estimate]` |
| C18 | Root-cause the 12x decode K/V re-read with explicit L2 attribution | `r1-kernel` S2.2 (measured point consistent with *partial* L2 absorption) | Explanation only; no counters | Decides whether C2's lever is "share in SLM" or "improve reuse order", i.e. two very different rewrites | Use Level Zero/UMA counters or a variant that reads K/V into SLM per kv-head; criterion: measured DRAM bytes/row within 10% of the 29.36 MB minimum | If reads are really 352 MB/row the headroom is 1.21 -> 0.41 ms/row `[estimate]` |
| C19 | Strata KV upload/prefill overlap pattern (`STRATA_KV_PREFETCH` structure) | `tmp/Strata/docs/KV_PREFETCH.md` (prefill 32K -5.3%, 64K -4.1%; no decode effect) | Aila's KV is VRAM-resident at <=32K so there is nothing to stream today; C13 in the Strata scan already closed Aila's KV residency work for <=32K | Only becomes live at >=64K contexts; the reference design (release/ready events, drain on early return) is useful if Aila ever streams KV | Do not probe now; note as a design reference tied to the same >=64K restart condition as Strata C13 | 0 at the production 32K shape; `-4..-5%` prefill only in the Strata 32-64K regime `[measured there, not here]` |

Explicitly **not** re-listed: host fill, bank_wait, staging copy parallelism,
copy-only queue, 16B alignment, DirectStorage single-file reads, SLM dequant
reuse, 8-token/share-8, swap-budget/frequency, pinned complement, chunk 8192 /
staging 64 / ring 4, prefill W4A8, ESIMD-vs-DPAS, INT8 attention DPAS, W4 slot
prefetch-into-GPU, `SYCL_CACHE_PERSISTENT`, resident-oracle profile (-3.7%),
cross-chunk cold window (charter S2; `r1-prefill-supply` S3.3).

---

## 2. Measurement gaps (no current-code measurement exists)

| # | Missing measurement | Evidence that it is missing | What to add | Cost |
|---|---|---|---|---|
| M1 | Decode attention device time at the production shape | The only decode-attention numbers are kernel-bench results (2 rows, len 14336, bf16 1.5833 ms; int8 1.0532 ms) from `qwen38-int8kv-vec-2026-10-12.md:23`; production per-token attention time is folded into `router_wait` (queue residual) with no field (`r1-decode` S2 last row) | A decode-safe event interval around `attention_decode` for the 12 full layers (see C4); separate from the 28-54 ms serialized arm | M |
| M2 | `attention_decode_rows` / fused multi-row JM device time | No arm exists at all; the `r1-kernel` table row e is a 2-row single-kernel measurement, the multi-row path is described only in prose (AGENTS S3) | New bench arm (C8) | S (test-only) |
| M3 | MoE-front / shared-expert submit breakdown | `r1-cost-model:132` "unprinted residual" derived by subtraction; `kStageRouter` not printed (`r1-decode:259`) | Print the existing stage enum values | S |
| M4 | QSA scoring / top-k / selected-attention split inside the full-attention *prefill* step | `r1-kernel` S2.4: "no timing arm for the qsa prefill path"; `--qsa-prefill-only` is correctness only; the only proxies are single-row decode numbers (82.2-85.4 us scoring, 385-589 us TopK per row at 4096-8192) | Extend the repaired `--qsa-probe` guidance (`qwen38-r1-qsa-probe-fix-2026-10-15.md` S3) to the prefill tile path with breadcrumbs | M |
| M5 | Full-attention prefill kernel at the 4096 block (237.5 ms/layer vs 2.05 ms XMX floor) | `r1-kernel` S1.1 row d, S2.4: unexplained 20x-116x, per-stage evidence WEAK | Timing arm for `RunAttentionWindow` including HC and the batched QSA prefill | M |
| M6 | GDN prefill kernel at the 4096 block (113.1 ms/layer vs 132.8 ms stream model) | `r1-kernel` S1.2 row c': "no kernel arm"; `AILA_Q4E_GDN_CHUNK_MODE` is e2e-only | Kernel arm (K4 subagent commissioned, `agents/k4-gdn.txt`) | M |
| M7 | PLE workspace lazy growth / peak | Launch comment "leave room for the lazy PLE workspace to grow during long prefill" (`tools/server/start-qwen38.ps1:21-23`); no `[Q4E ...]` line reports the PLE workspace size, and no `PLE_WORKSPACE`-shaped switch exists in `src/models/Qwen4ExpPle.{hpp,cpp}` | Add a load/forward print of the allocated PLE workspace bytes and run `--memory-sample-ms` | S |
| M8 | USM/DXGI peak reservation for the current shapes | `plan:166` item 4 explicitly asks for a separate diagnostic round using the existing `--memory-sample-ms` plus DXGI monitoring of the real VRAM peak, covering 4096-token blocks, a short append, cancel and recovery; the plan's record (S8) only reports items 1-3, and the status doc still marks this `[unverified]` (`qwen38-qwen4exp-pipeline-status-2026-10-13.md` S7) | `--memory-sample-ms 250` + DXGI monitor on 4096 blocks, short append, cancel-recover; then the reserve 3584 -> 3328 single-variable arm if headroom allows | M |
| M9 | Checkpoint / interior-snapshot overhead | `AILA_Q4E_INLINE_CHECKPOINT` (default "auto-30/26", `qwen38-qwen4exp-pipeline-status` S2.1) and `SnapshotPolicy::ReusablePrefix` are wired (`src/models/Qwen4ExpBnb4Backend.cpp:1809-1818,2469`) but no report gives the wall cost of capturing vs not capturing the interior snapshot | A/B with the checkpoint disabled on a 14096-token prompt; field = TTFT delta and `cached_prompt_tokens` on the continuation | S |
| M10 | 2-session behavior and aggregate throughput on the adaptive pipeline | The only probe attempt failed before readiness: `Server exited before readiness (exit=1)` with `AilaServer: MEM_LARGE_PAGES allocation failed: bytes=1272971264 win32_error=1450` (`tmp/q4e-sprint-r1-20261014/service/run-s2-a1/result.json`, `worker.log`); no `[Q4E AdaptiveWork]` decode rows were produced | Fix the large-page/readiness precondition, then re-run `probe_sessions2_qwen4exp.py`; fields `peak_active_requests`, `metrics.aggregate_decode_rate_pair_window`, `adaptive_work.decode_fingerprint.duplicate_pos_values` | M |
| M11 | Prefetch switches in the adaptive pipeline | `AILA_Q4E_PREFETCH` defaults true (`src/models/Qwen4ExpBnb4Backend.cpp:2009`) and its sub-switches `_PREFETCH_DECODE`/`_PREFETCH_PREFILL` default true (`:2012-2013`), but the profile JSON sets both to `0` and the prediction path is gated by `const bool predict_next = !use_adaptive_pipeline_ && ...` (`:2253-2254`) - i.e. speculative prefetch is unreachable from the adaptive pipeline that all production runs use | Decide: delete the dead path, or re-express next-layer prediction as an adaptive-cache hint (correctness-neutral). Then A/B once | S (source read) / M (arm) |
| M12 | AOT vs JIT startup split per kernel group | `docs/Qwen38FlashNext_A770_Handoff.md:386-387` item 5 never done; the L10 split (router 50.4 s + full 19.8 + linear 7.3 + hc 1.2) is the closest data | AOT scratch build + `AILA_Q4E_LOAD_TIMING=1` cold start | L |
| M13 | `--qsa-batches` batched-selector cost at start in {2044...131071} x rows {4,64} | `r1-kernel` S3 probe specification; `qwen38-r1-qsa-probe-fix-2026-10-15.md` S3.2 keeps `--qsa-batches` non-fatal; no log archived | Run the repaired probe (no src work) | S |
| M14 | 12x K/V re-read attribution (L2 vs DRAM) | `r1-kernel` S2.2 is a fit to two candidate models, not a counter measurement | Counters or an SLM-staged variant (C18) | M |
| M15 | Slot elasticity on the current pair | `plan:96-97`/`B3` elasticity (0.45-0.5%/100 slots) came from `AILA_Q4E_MAX_GPU_SLOTS=2000` on `8101788`+hotset profile; no current-pair re-measurement, and the int8-KV layout changes the slot budget | Re-run the 2000-slot arm on `retained23`; field `[Q4E AdaptiveWork]` hits/misses + decode rate | S |

---

## 3. Mapping to charter S3 questions 1-8

| # | Charter question (`qwen38-performance-research-charter-2026-10-14.md` S3) | State after R1 | Evidence |
|---|---|---|---|
| 1 | Prefill layer-major prototype (H2D 310 -> ~62 GB, ttft -13..-18%, charter-tagged unverified) | **Half open.** Mechanism validated (H2D -74/-75%, uploads 33,108 -> 8,349 on the 8192 window); production form still wall-negative (-67%), so the ttft claim is unproven. Next step = cross-layer pipeline (C1) | L7, L13, L17, L27-L31; `r1-layer-major-proto` S8 |
| 2 | Decode 64 ms/token contemporary cost model, per item | **Mostly closed.** Refreshed: wall 66.01 ms = submitted 62.20 + sample 3.56 + prepare/drain 0.11; experts_host 22.11 (cpu_join 10.00), router_wait 12.69, residual 9.09, ple_host 6.65, hc 6.22. `router_wait` = queue depth (router work 0.17-0.29 s/chunk). Open sub-items: decoder-attention device time, MoE-front split (C5, M1, M3) | L3, L11, L12; `r1-cost-model` S2, S3 |
| 3 | MTP extension (per-session gate; acceptance 74.8% -> 95%) | **Open, deferred.** Equivalence gate failed at pos=1492 (rejected verify row 0); per-session value is near zero because production sessions start at 14K while the gate requires <=2048; the +2.9% first answer is below the +5% line | `qwen38-mtp-acceptance-2026-10-11.md`; `qwen38-mtp-align-2026-10-11.md`; roadmap B4-6 |
| 4 | Expert supply tiering and capacity (how much of the 24.2 MiB/token goes CPU vs staging; is the CPU path at the DRAM wall) | **Effectively closed for the production profile.** 100% of misses go to CPU (`staged 0`, `h2d_mb 0`, `COLD_MODE=2`), CPU compute 13.88 ms/token at ~45 GB/s over 626 MiB/token = at the practical DRAM wall; the alternative (staging) is 3x worse (L6) | `r1-decode` S2 supply mix; L6 |
| 5 | Prefill GDN (4.07 s) and full attention (2.85 s) vs their kernel ceilings | **Open.** Both reproduced within 5% on the current pair (L11) but neither has a kernel timing arm; GDN sits at 0.85x of its own degraded stream model, full attention is 20x-116x above compute floors `[estimate]` | `r1-kernel` S1.1-S1.2, S2.4, S2.5; M5, M6 |
| 6 | Service-mode aggregate throughput ceiling (multi-session / `--decode-batch-size>1`) | **Open.** Code-level answer is complete (round-robin serial; `SupportsDecodeBatch()==false`, null batch key; upper bounds 15.6 / 21.1 / 32.4 tok/s at B=2), but the GPU probe failed before readiness with `MEM_LARGE_PAGES ... 1450`, so no adaptive-pipeline 2-session data exists | `qwen38-r1-service-2026-10-14.md` S1-S3; `service/run-s2-a1/result.json`; M10 |
| 7 | Startup and memory (cold 116 s / warm 68 s; nightly peak; multi-shard parallel read) | **Half open.** Cold start 116.6 -> 63.8 s measured but opt-in and hot-cache-negative; the parallel pre-read lever is quantified (8.6-9.8 GB/s); first-request ~80 s now has an opt-in warmup + the F1 PLE fix; the USM/DXGI peak audit and reserve 3584 -> 3328 experiment were never run | `qwen38-c7-*`; `qwen38-strata-c7-fill-pipeline-2026-10-13.md`; L21-L26; `plan:166`; M8 |
| 8 | Kernel-family remaining headroom audit under the A770 skill (roofline/ESIMD/JM/GRF) | **Half open.** K1 (decode MoE grouped GEMV) measured 0.12343 ms vs a 0.0871 ms 1.5x threshold - headroom exists; K2 (QSA route) repaired and measured (+36-37% at 8192, marginal at 4096); decode attention is the largest measured distance-to-bound with no variant yet; GDN/full-attention prefill have no timing arms | L8, L25; `r1-kernel` S3-S4; `qwen38-r1-qsa-probe-fix-2026-10-15.md`; C2, M1, M4-M6 |

Open count: questions 1, 3, 5, 6 fully open; 2, 7, 8 partially open; 4
effectively closed for the production profile. Net: **5.5 of 8 still open**
(1, 3, 5, 6 fully; 2, 7, 8 residual sub-items).

---

## 4. Sources

- `docs/plans/qwen38-performance-research-charter-2026-10-14.md` (S0 hardware, S1 baseline, S2 negatives, S3 questions, S5 discipline).
- `docs/plans/qwen38-r1-cost-model-2026-10-14.md` (S1-S8, ledger L1-L31).
- `docs/plans/qwen38-r1-decode-2026-10-14.md` (decode table, if-X-free, probe specs).
- `docs/plans/qwen38-r1-prefill-supply-2026-10-14.md` (block table, H2D, layer-major sizing).
- `docs/plans/qwen38-r1-kernel-2026-10-14.md` (roofline table, gap attribution, K1/K2).
- `docs/plans/qwen38-r1-service-2026-10-14.md` (batching verdict, upper bounds, probe spec).
- `docs/plans/qwen38-r1-ple-firstchunk-2026-10-15.md`, `qwen38-r1-layer-major-proto-2026-10-15.md`, `qwen38-r1-qsa-probe-fix-2026-10-15.md`.
- `docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md`, `qwen38-optimization-roadmap-2026-10-10.md`, `qwen38-a770-agent-performance-plan-2026-10-12.md`, `qwen38-strata-candidates-execution-2026-10-13.md`, `qwen38-strata-design-scan-2026-10-13.md`.
- `docs/plans/qwen38-mtp-acceptance-2026-10-11.md`, `qwen38-mtp-align-2026-10-11.md`, `qwen38-int8kv-vec-2026-10-12.md`, `qwen38-int8kv-prod-2026-10-12.md`, `qwen38-strata-c7-fill-pipeline-2026-10-13.md`, `qwen38-c7-deep-load-breakdown-and-directstorage-2026-10-13.md`.
- `docs/Qwen38FlashNext_A770_Handoff.md`, `docs/Environment_Variables.md`.
- Source: `src/models/Qwen4ExpBnb4Backend.cpp` (2007-2030 prefetch switches, 2253-2254 predict gate, 1809-1818 checkpoint, 2489-2505 AdaptiveWork), `src/models/Qwen4ExpPle.cpp` (521-550 prefetch, 577 call), `src/models/Qwen4ExpLayerMajor.cpp`, `src/models/Qwen4ExpPrefill.cpp` (16-131, 559-573), `src/inference/Scheduler.cpp` (386-455), `tools/server/start-qwen38.ps1`, `docs/plans/qwen38-server-mtp-2026-10-07.json`.
- Artifacts: `tmp/q4e-sprint-r1-20261014/{r1-diag,r1-stages2,lm-a1,lm-b2,lm-b3,lm-b5,ple-a,ple-b,ple-a2,ple-b2,service/run-s2-a1,agents}`.
- Reference project: `tmp/Strata/docs/{PREFILL_HEAD_ALIAS,KV_PREFETCH,BATCHED_DMA,VRAM_ELASTIC,HOW_IT_WORKS}.md`.
