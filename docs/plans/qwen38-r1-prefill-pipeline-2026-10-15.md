# Qwen4Exp R1 E6: Prefill Supply Pipeline (deeper staging banks + in-layer lookahead)

Base: HEAD 3ef4606 (pair retained27). Scope: src/models/Qwen4ExpAdaptiveCache.{hpp,cpp},
src/models/Qwen4ExpBnb4Backend.cpp RunAdaptivePrefill staging call section only.
Status: code landed behind a default-off env: AILA_Q4E_PREFILL_STAGING_BANKS (default 2).
No build, no GPU arm was run by E6; arms for the coordinator are in section 3.

## 1. Dependency chain and what can actually be looked ahead

Layer order inside the 48-layer loop (Qwen4ExpBnb4Backend.cpp):
1. RunAttentionWindow(ctx, session, L, ...) at :2202 submits this layer's attention
   kernels (buf_.y = attn_out).
2. RunHyperConnectionCombined(ctx, layer.mlp_hc, count, true, buf_.y, hc_inject)
   at :2220-2222 folds the attention residual into buf_.x (post-attention hidden).
3. dense_rows(layer.router_gate, layer.router_w, buf_.x, buf_.router_scores) at :2290
   computes the router FROM buf_.x, i.e. from the post-attention hidden.
4. wait_actual_route at :2332 blocks the host until the top-k D2H (pinned) lands.
5. RunAdaptivePrefill is reached at :2522 (count>4 path) and is the only prefill
   consumer of StageColdExpertsAsync.

Consequence (confirmed by reading the code, matches the A1-v4 note): routing(L)
strictly requires attention(L) + mlp_hc(L) to have executed. Therefore the cold
expert SET of layer L+1 cannot be known while layer L's MoE kernels run, and a
cross-layer fill/DMA prefetch is structurally impossible. The only legal lookahead
is INSIDE layer L: once wait_actual_route(L) returns, all of layer L's cold chunks
are known at once, and chunk k's staging has no dependency on chunk k-1's kernels
except the shared bank resource.

Where the current 19.5 s/block goes (E1 measured evidence, not re-derived here):
- cost arm: copy queue alone (48 layers x 403 cold experts = 49.0 GB/block) sustains
  5.73 s wall (fill 4.93 s + H2D 4.96 s overlapped, link 9.89 GB/s);
  tmp/q4e-sprint-r1-20261014/block-bench-A1.log.
- production block 19.5 s = host-busy 11.2 s (fill 2.4 + copy wait 8.1 + submit)
  + router_wait 8.0 s; device work ~8 s (GDN 4.1 + full 2.7 + hc/router 0.6 +
  expert kernels 0.55). Supply and device are effectively serialized.
- With 2 banks the host loop is stage(k) -> run(k) -> stage(k+1) (HEAD code
  Qwen4ExpBnb4Backend.cpp:1503-1515). StageColdExpertsAsync blocks the host until
  the DMA that last read THIS bank's host buffer completes
  (Qwen4ExpAdaptiveCache.cpp:615), and the next H2D waits for the bank's last
  consumer kernel event (cpp:653-654). At banks=2 that means the host cadence
  (pending-DMA wait + 16-slot fill + RunGroupedExperts host map build + submit)
  is serialized one-deep with the DMA chain, which is what the measured
  bank_wait (~8.1 s/block, ~6.5 ms per 38.7 MiB chunk) reflects.

## 2. V1 implementation (built) and V2 (design-only, trigger-gated)

### V1: AILA_Q4E_PREFILL_STAGING_BANKS (default 2, range 2..4)

Change points:
- Qwen4ExpAdaptiveCache.hpp:26-47: kMaxStagingBanks=4, config field staging_banks=2,
  StagingBanks() accessor; :192 GetStagingSlotDevicePtr normalizes bank =
  index mod banks; :219-222 RecordStagingConsumer same normalization;
  :335-347 bank arrays widened from 2 to kMaxStagingBanks.
- Qwen4ExpAdaptiveCache.cpp:122-129 and :193-199: env read + clamp applied in both
  Allocate overloads (2 = historical ping-pong, byte-identical default).
  :160-167 / :230-237 per-bank allocation loops; :246-254 null checks;
  :269-306 drain/free loops over all banks; :566-572 ExecuteSwapsAsync drains all
  banks before reusing host banks 0/1; :613 bank resolve in StageColdExpertsAsync.
  :347-352 GetStagingSlotDev previously indexed staging_experts_[pp&1] while the
  pointer used pp - a latent descriptor/table mismatch once pp>1; fixed by
  normalizing once for both.
- Qwen4ExpBnb4Backend.cpp:1506-1543 RunAdaptivePrefill cold loop is now a sliding
  window of lookahead = banks-1 chunks: pre-stage chunks 0..lookahead-1, then for
  each chunk c: RunGroupedExperts(c) gated on chunk c's own ready_event ->
  RecordStagingConsumer(bank_c) at :1538-1539 -> stage chunk c+lookahead at :1541.
  One-time "[Q4E PrefillStage] banks=.. lookahead=.. slots_per_bank=.." log at
  :1516-1520.

Dependency/sync argument:
- Bank cycle is banks, one chunk per iteration, so chunk k always uses bank k mod
  banks and its DMA waits on the kernel that consumed that bank (chunk k-banks).
  With lookahead = banks-1 every bank stays live; the ring array is
  kMaxStagingBanks=4 entries >= lookahead+1.
- Kernel gating is unchanged: RunGroupedExperts(chunk c) receives chunk c's DMA
  event (backend :1538), and RecordStagingConsumer(chunk c) is called before the
  next stage, so the fence chain is exactly the historical one, just delayed by
  one bank cycle.
- All stage calls still happen inside the same RunAdaptivePrefill call, after
  wait_actual_route; no staging state crosses a forward boundary. Descriptor
  host arrays (expert_map_*_host_) are written only by RunGroupedExperts, whose
  ordering is untouched.
- banks==2 ordering proof: pre-stage(0); loop c=0: run(0), record(0), stage(1);
  c=1: run(1), record(1), stage(2)... which equals HEAD's stage(k), run(k),
  record(k), stage(k+1) exactly; ping_pong_index_ advance (i+1) mod 2 equals
  HEAD's ^=1 from any starting value.

VRAM / host cost:
- banks=2 (default): zero change (allocation, ordering, fences identical).
- banks=4: +2 device banks and +2 host banks = 2 x 16 x 2,534,400 B = 81,100,800 B
  = 77.35 MiB device (+77.35 MiB host USM). NOTE: the load budget reservation at
  Qwen4ExpBnb4Backend.cpp:784 still counts 2 banks; the extra 77.35 MiB sits
  outside that reservation (well inside the 3584 MiB reserve env). If a future
  arm pairs this with AILA_Q4E_STAGING_SLOTS>16, revisit that line.
- AILA_Q4E_STAGE_COPY_PARALLEL (existing, default off) parallelizes the 16-slot
  host fill via the executor arena (Qwen4ExpAdaptiveCache.cpp:630-638,
  synchronous; bit-exact). Kept independent; offered as a second B-family knob
  only if fill_ms stays the residual after banks=4.

Bit-exact argument:
- Same cold expert set per layer, same chunk sizes/order, same staging slots per
  expert, same kernels with identical arguments, same accumulate. Only the host
  issue order of H2D copies relative to host map-building work changes. Fill path
  is byte-identical (parallel fill stays off unless its env is set).

Rollback: unset AILA_Q4E_PREFILL_STAGING_BANKS (or =2). No rebuild.

### V2: whole-layer device staging (AILA_Q4E_PREFILL_LAYER_STAGE) - NOT built

Design on record (build only if the V1 arm proves depth is not the limiter):
- One device buffer sized for the worst-case cold union of a layer, 403..512 slots
  = 973 MiB..1.21 GiB, filled from the existing 16-slot host banks in 26+ chunked
  H2D transfers, each chunk gated on the previous consumer of its device slice;
  one reader fence per layer instead of per bank.
- Expected ceiling is the same DMA stream (~4.96 s/block at 9.89 GB/s): V2 does
  not reduce bytes, does not raise the link rate, and the E1 block bench showed
  banks=4 already making no difference to the pure-copy replay, so the marginal
  gain over V1 is small for +973 MiB..1.21 GiB VRAM near the budget.
- Build trigger (explicit): V1 arm shows block wall >= 15 s AND bank_wait_ms
  collapsed to ~0 yet the copy chain still stars on per-chunk kernel fences
  (visible as staged=... with fill_ms << copy wait). If instead bank_wait_ms
  stays ~8 s at banks=4, depth is not the limiter at all and the correct next
  lever is byte reduction (E1 path 1: per-prompt cold-key union), not V2.

## 3. Coordinator arms (GPU-exclusive, back-to-back, no sleep loops)

Precondition (must print nothing):
  Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue

Build once, then A/A/B/B/A on the standard 14K+prime diagnostic arm:
  pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
  python tmp/q4e-sprint-r1-20261014/run.py e6-p1-a1 --prime --diag --generate=8 AILA_Q4E_PREFILL_STAGING_BANKS=2
  python tmp/q4e-sprint-r1-20261014/run.py e6-p1-b1 --prime --diag --generate=8 AILA_Q4E_PREFILL_STAGING_BANKS=4
  python tmp/q4e-sprint-r1-20261014/run.py e6-p1-b2 --prime --diag --generate=8 AILA_Q4E_PREFILL_STAGING_BANKS=4
  python tmp/q4e-sprint-r1-20261014/run.py e6-p1-a2 --prime --diag --generate=8 AILA_Q4E_PREFILL_STAGING_BANKS=2

Read (per run: <label>-worker.log, count=4096 forward lines):
- Confirm "[Q4E PrefillStage] banks=2|4 lookahead=1|3" appears once.
- Per count=4096 [Q4E AdaptiveWork] line: submitted_ms (per-forward host wall,
  the block wall proxy), fill_ms, bank_wait_ms, experts_host_ms, staged, h2d_mb.
  Also [Q4E Supply] is cold-path only and will not print here; do not use it.
- TTFT secondary: context-6000/context-14000 first_content_seconds in the run
  output; cached_len (reuse) must match across arms.
- Bit-exactness: assistant text bytes of all replies identical A vs B (the probe
  output JSON), and continuation tok/s within the A/A noise band.

Pass criteria (frozen):
- median submitted_ms over steady count=4096 blocks <= 15.0 s (baseline 19.5 s,
  -25%) with bank_wait_ms materially lower than the A arms; 2/2 pairs B better;
  A/A |delta| <= 1%; all texts byte-identical; no UR errors, no h2d_mb change.
- If bank_wait_ms drops but wall does not, escalate to the AILA_Q4E_STAGE_COPY_PARALLEL=1
  arm on top of banks=4 before touching V2 (A2/B2 pair).

## 4. Risks and TODOs

- Depth may not be the binding constraint: the E1 bench replay showed banks=4
  equal to banks=2 in a pure copy pipeline (block-bench-A2-banks4.log). If the
  production B arms come out flat, the 8.1 s bank_wait is a symptom of the
  DMA-gated-on-kernel fence chain, not of host cadence, and the documented next
  lever is byte reduction, not more banks.
- VRAM: +77.35 MiB device at banks=4 is outside the :784 reservation (see above).
- LayerMajor path (Qwen4ExpLayerMajor.cpp:612-623) still advances the bank index
  with ^=1, so it uses a 2-bank subset by construction; harmless (its own
  AILA_Q4E_LAYER_MAJOR_PIPELINE is a depth-2 schedule). Unify if layer-major
  ever becomes production and wants deeper banks.
- Legacy decode staging path (Qwen4ExpBnb4Backend.cpp:1331-1333) also toggles ^=1;
  with banks>=3 it uses a consistent 2-bank subset. Left untouched by scope.
- TODO(coordinator): if a later arm wants banks=4 with AILA_Q4E_STAGING_SLOTS>16,
  update Qwen4ExpBnb4Backend.cpp:784-785 to reserve banks * slots.

## 5. Sources

- HEAD 3ef4606; brief evidence: tmp/q4e-sprint-r1-20261014/block-bench-A1.log,
  block-bench-A2-banks4.log; docs/plans/qwen38-r1-prefill-400-2026-10-15.md.
- Code refs as cited inline (line numbers at HEAD plus this change).
