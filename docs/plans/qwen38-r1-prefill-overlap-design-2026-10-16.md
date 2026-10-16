# Qwen38 R1 P2: prefill supply/device overlap - archaeology and design verdict

Date: 2026-10-16. Role: P2 (supply/device-overlap archaeology + design).
Base: HEAD `2e25fb8` (worktree clean; src at `5cc1cad`).

Method: source + existing receipts + `tmp/Strata` read-only. Zero GPU runs, zero
builds, zero src edits. Every claim carries a source; estimates are marked
`[est]`. Charter section 2 (the negative list) is assumed closed and is not
re-opened except where a mechanism is provably different.

## 0. The question and the answer

Question: under (i) in-order queues, (ii) routing needs the post-attention hidden
state, (iii) the bit-exact requirement - does an un-negated supply/device overlap
mechanism still exist for prefill?

Answer: **no mechanism that can move the 4096-block wall by >= 3%**. Four
independent receipt families show that H2D transfer time does not hide behind
memory-bound device work on this box (section 1), and every reorder/replace
attempt so far pays that cost somewhere else (section 5). Sections 7-8 price what
is actually left, including the one structural mechanism with a >20% ceiling
(candidate Z: consumer kernels read the cold experts over PCIe), which is not an
overlap mechanism but a supply-elimination one.

## 1. The measured law: H2D time is additive with memory-bound device work

| # | receipt | setup | result |
|---|---|---|---|
| L1 | `--queue-overlap-probe` (`tests/ops/Qwen4ExpKernelTests.cpp:4374`), numbers in `docs/plans/qwen38-strata-c2-copy-queue-screen-2026-10-13.md` section 2(a) | ALU burn 3.60 ms + H2D 3.45-3.48 ms, two-bank ping-pong | 4.04-4.18 ms = max(3.60, 3.48) - **overlaps** |
| L2 | same probe | memory burn 3.70-3.73 ms (256 MiB x 6 passes) + same ping-pong | 6.70 ms vs serial sum 7.16 and max 3.73 - **adds** |
| L3 | `c1-e1` arm, `docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md` section 2 | direct DMA from the large-page arena (fill 0.75 s, bank_wait 0.05 s vs baseline fill 7.2 / bank_wait 3.77) | chunk wall unchanged 20.3 s, TTFT 72.23 s (+2.9%) |
| L4 | `AilaQwen4ExpLayerStageBench`, `tmp/q4e-sprint-r1-20261014/layer-stage/bench-A1.log:18-19`, `bench-A2-g8.log:18-19`, doc `docs/plans/qwen38-r1-cost-model-2026-10-14.md` 6.32/6.36 | GDN prefill load (2 calls, 150.5 ms) + 1.15 GB staging | overlap 255.4 ms = load + 104.9 ms; at 8 calls (load 601.8 ms) the inflation is 105.4 ms - **constant, = the copy time** |
| L5 | layer-stage e2e arms, `docs/plans/qwen38-r1-layer-stage-2026-10-15.md` section 8; `tmp/q4e-sprint-r1-20261014/g5-b1-worker.log:88-99` | whole-layer arena staging overlapped with the device drain | bank path retired (bank_wait 3.60 -> 0.05 s, h2d 46.8 -> 0.218 GB) but block 19.87-20.59 s vs 19.05-19.90 s baseline; the entire supply reappears in `router_wait` (6.9-8.2 -> 18.4-20.3 s) |

Law (qualitative and quantitative): for these byte counts, on this
DG2/Windows/L0 stack, the transfer cost of S bytes is S / (9.81-9.89 GB/s)
[measured link rate, section 7] and that time appears in the block wall
essentially undiminished, whether it is spent host-blocked in the pump, deferred
into a later bank wait, or run concurrently with streaming kernels. Only ALU-only
loads overlap (L1). The 46.8 GiB of expert bytes per 4096-token block therefore
costs ~4.96 s that no queue placement removes.

Conservation channel (not a new measurement, but the accounting that makes the
law visible): banks 2 -> 4 moves -1,655 ms out of `bank_wait` into `router_wait`;
slots 16 -> 32 moves -1,732 ms (tables in
`docs/plans/qwen38-r1-prefill-overlap-2026-10-15.md` section 3).

## 2. What the prefill pipeline actually is (plus one correction)

Code map (current line numbers, HEAD `2e25fb8`):

- Production prefill is chunk-major `RunAdaptivePrefill`
  (`src/models/Qwen4ExpBnb4Backend.cpp:1540`), reached from the layer loop after
  the route wait (`:2535` router GEMM, `:2558` top-k, `:2596`/`:2718` the two
  wait sites).
- The staging pump: `stage_chunk` (`:1689`) calls
  `StageColdExpertsAsync(host_arena_, *copy_q_, ...)` (`:1691`) with lookahead
  `staging_banks - 1` (`:1672`, `:1698`); each chunk's kernels are gated on that
  chunk's DMA event (`:1701`) and the bank's consumer event is recorded for the
  next refill (`:1702` -> `src/models/Qwen4ExpAdaptiveCache.hpp:219-222` ->
  `src/models/Qwen4ExpAdaptiveCache.cpp:652-654` `depends_on(staging_reader_[pp])`).
- `StageColdExpertsAsync` internals (`src/models/Qwen4ExpAdaptiveCache.cpp:606-665`):
  host-blocking wait on the bank's previous DMA (`:613-617`, the `bank_wait_ms`
  source), the fill (`:629-645`; parallel host copy only behind the opt-in
  `AILA_Q4E_STAGE_COPY_PARALLEL`), the H2D submit with the consumer fence
  (`:649-655`).
- The DMA queue is the separate in-order queue `copy_q_`, created at
  `src/models/Qwen4ExpBnb4Backend.cpp:885` and reused for the decode cold path
  (`:1453`) and the layer-stage arena (`:2835`).

Correction to the working hypothesis recorded in the coordination note ("the
supply and the device compute share the same in-order `ctx.queue()`"): they do
not. The staging copies already run on a second in-order queue, and the worker
already sets `UR_L0_USE_COPY_ENGINE=1` /
`SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1` at startup
(`src/utils/EnvUtils.hpp:179-201`, `src/worker/WorkerMain.cpp:494`; stated as
already-in-place in the C2 screen, section 1). The conservation is therefore not
a queue-placement defect; it is the memory-path law of section 1.

Per-block accounting (steady 4096 block, `e6-banks4-b1`,
`docs/plans/qwen38-r1-prefill-overlap-2026-10-15.md` sections 2-4):

| term | ms/block | source |
|---|---:|---|
| submitted total | 18,076.8 | `[Q4E AdaptiveWork]` |
| router_wait (device drain, 48 sync points) | 8,583 | 99.9% device queue depth |
| supply (host pump) | 9,366 | = fill 2,472 + bank_wait 6,283 + ~600 submits |
| bank_wait detail | 6,283 | = 4,962 DMA (46.8 GiB at 9.89 GB/s) + 1,321 fence/queue |

Supply shape: 1,210 chunks/block of 16 slots (40.55 MB/chunk); 19,360 staged
experts/block; resident pool 2,792 of 24,576 experts
(`docs/plans/qwen38-r1-layer-stage-2026-10-15.md` section 2.3).

## 3. Archaeology of the charter section 2 items (what / why / kind)

| item | what was actually done | why it failed | kind | restart condition |
|---|---|---|---|---|
| C1 stream-ahead | Aila minimal screen: parallel fill, direct DMA, queue-overlap micro-probe (`docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md`) | the transfer is additive with the streaming kernels (L2/L3); removing *all* host-side cost left the chunk wall unchanged | mechanism (memory path), not implementation | a driver/stack change that removes the additivity (re-run the L1/L2 probe) |
| C2 copy-only queue | copy-engine ON/OFF probe + local L0 probe `tmp/q4e-strata-c2/l0_copy_probe.cpp` (`docs/plans/qwen38-strata-c2-copy-queue-screen-2026-10-13.md`) | the A770 exposes copy-only ordinal 1 (9.81 vs 9.76 GB/s per 32 MiB) but Aila already runs copies off the compute engine via UR; ALU loads overlap, memory-bound loads add; Strata's own Alchemist result is +1.4% with a hang history (`tmp/Strata/docs/INTEL.md:824-829`) | mechanism for the variant that matters; **one cell unmeasured** (see section 4) | only if the section 4 residual is ever wanted |
| C3 16B alignment | layout audit + `--decode-gu-bench` split-load variant (`docs/plans/qwen38-strata-c3-aligned-loads-screen-2026-10-13.md`) | premise absent: Aila NF4 rows/strides are already 16B multiples (1280 B / 320 B / 32 B); the only misaligned 16B load is the L1-resident activation stream; measured -0.5% (noise) | premise/parameter | none (no layout problem to fix) |
| C4 profile mix | held-out export + normalized mix ranking (`docs/plans/qwen38-strata-c4-profile-mix-screen-2026-10-13.md`) | +1.1pp decode-slot coverage vs agent6k, below the >=3pp implementation gate; the remaining 9.68pp headroom is instance-specific, not domain-specific | parameter/coverage | a genuinely different-domain corpus (Chinese long-form/docs/code repo) |
| C6 verify route tail skip | source/verdict only (`docs/plans/qwen38-strata-c6-c11-verdicts-2026-10-13.md`) | precondition missing: production had no >=2-row window (T1, decode-batch 1, MTP gated). Value-changing by construction (dropped experts count 0, no renormalization) so it needs its own quality gate | precondition + value-changing | when a multi-row verify path is default (MTP is now opt-in with Top-1 >=0.95 relaxed; still not default) |
| C9 suffix lookup | offline trigram probe on the real token streams (`docs/plans/qwen38-strata-c9-suffix-lookup-screen-2026-10-13.md`) | acceptance 32.3-35.5% < the 50% gate (token/window 1.66-1.80 passes); needs a multi-row verify path; Strata itself measures A770 draft/verify overlap as a loss (`tmp/Strata/docs/A770_SPEED_SPEC.md:254-257`) | parameter (acceptance) + precondition | real edit/reference corpora + a credible multi-row verify |
| C10 batch-MTP | verdict only (`docs/plans/qwen38-strata-c5-c10-c12-c13-verdicts-2026-10-13.md`) | precondition: MTP equality gate had failed and Strata's own batch-MTP is -21.9% when experts are fully resident; the transferable part is only the policy rule | precondition | MTP usable by default and multi-session batching in the default path |
| C12 small-chunk CPU share | verdict only, same file | Aila's optimum chunk is 4096 (2048/4096/8192 = 83.89/70.6/89.22 s) and the CPU-MoE prefill is excluded: 40,960 route rows are per *layer*, 1,966,080 rows/block at 53.7 us -> ~105 s/block (`docs/plans/qwen38-r1-cpu-moe-bench-2026-10-15.md:116-120,179`) | parameter + precondition | a chunk-policy change (e.g. small last chunk) or a much faster CPU expert kernel |
| C13 KV residency/prefetch | verdict only, same file | Aila KV lives in VRAM (int8 opt-in), decode staging is already zero; host KV would add per-layer PCIe on a 9.6-9.9 GB/s link; Strata's gains appear at 32K/64K only | parameter (context size) | >=64K contexts in real use |
| C5 adaptive/asynchronous swap + rotation | verdict only (same file) | swap costs decode -8.03% when on; rotation's precondition is a fully pinned, GPU-mapped host complement, which collides with the measured 51 GiB pin failure (`UR_OUT_OF_RESOURCES`) | precondition | a driver that removes the pin limit, or a much smaller pinned set |
| C8 methodology / C11 build items | adopted / verified (`docs/plans/qwen38-strata-c6-c11-verdicts-2026-10-13.md`) | not candidates | n/a | n/a |

## 4. C2 deep dive: is "copy-only queue" the same mechanism as "H2D on a second queue"?

Two distinct implementations share one family name:

1. **Aila's current form** ("H2D on a second queue"): the staging copies go to
   the separate in-order `copy_q_` (`src/models/Qwen4ExpBnb4Backend.cpp:885`,
   `:1691`) with `UR_L0_USE_COPY_ENGINE=1` set at worker startup
   (`src/utils/EnvUtils.hpp:179-201`, `src/worker/WorkerMain.cpp:494`). Measured
   in the same binary with only the env flipped
   (`docs/plans/qwen38-strata-c2-copy-queue-screen-2026-10-13.md` section 2a):
   compute-waits-copy 5.15-5.34 ms ON vs **7.09 ms OFF**; ALU ping-pong
   4.04-4.18 vs **7.08**; memory-burn ping-pong **6.70 vs 7.17** (additive
   either way). So the "second queue/engine" mechanism is already ON and is
   already proven to overlap ALU work and to fail for streaming loads.
2. **Strata's form**: find the copy-only queue group
   (`zeDeviceGetCommandQueueGroupProperties`, ordinal 1), build an immediate
   command list, wrap it with `sycl::make_queue<backend::ext_level_zero>`
   (`tmp/Strata/sycl/include/strata/copy_queue.hpp:1-26`,
   `tmp/Strata/sycl/src/core/copy_queue.cpp:30-70`; rationale in
   `tmp/Strata/docs/INTEL.md:824-829`: with the L0 v2 adapter every SYCL queue
   ran on the compute engine and 0 of 2,000 copies overlapped a kernel).

Evidence bearing on the same question, locally measured: the A770 **does**
expose the copy-only group, and its standalone bandwidth equals the default
queue's - ordinal 0 (COMPUTE+COPY) 3.436 ms per 32 MiB = 9.76 GB/s, ordinal 1
(COPY only) 3.422 ms = 9.81 GB/s
(`docs/plans/qwen38-strata-c2-copy-queue-screen-2026-10-13.md` section 6,
probe `tmp/q4e-strata-c2/l0_copy_probe.cpp`). Strata's own Alchemist number is
+1.4% with one hang on the completion-flag pattern
(`tmp/Strata/docs/INTEL.md:829`).

Concrete failing numbers for the "second queue saves the transfer" hypothesis
(all with the second queue/engine active): block wall 19.05-19.90 s baseline vs
19.87-20.59 s with the bank path fully retired (L5); chunk wall 20.3 s with
fill+bank_wait driven to ~0 (L3); overlap inflation = copy time at two load
sizes (L4).

**Residual cell (explicitly unmeasured)**: running a SYCL interop queue on
ordinal 1 with a *memory-bound* load and re-testing overlap. Predicted negative
on the strength of (a) equal standalone bandwidth on both ordinals and (b) L1/L2
showing overlap only for ALU loads. Cost: medium-high (vendored L0 headers +
interop plumbing + probe). Recommendation: keep on record, do not spend the next
GPU slot on it (section 9, P3).

## 5. Timeline of queue / asynchronous-copy / event-dependency experiments

| when | experiment | what | result | disposition |
|---|---|---|---|---|
| 10-13 | C1 screen | parallel fill, direct DMA, queue-overlap micro-probe | fill conserved into bank_wait; direct DMA leaves wall unchanged; ALU overlaps / memory adds | closed (mechanism) |
| 10-13 | C2 screen | copy-engine ON/OFF + L0 ordinal probe | see section 4 | closed except the interop cell |
| 10-14 | banks 2 -> 4 (`d7a5f6f`) | deeper ping-pong ring + sliding lookahead | bank_wait 7,893 -> 6,224 ms, submitted -2.4%, TTFT 69.4 -> 65.9 s; -1,655 ms moved into router_wait | adopted opt-in; tail deferred, not saved |
| 10-15 | slots 16 -> 32 (`e6-b4s32`) | double chunk size | bank_wait 4,551-4,612 ms (-26%) but submitted 18,496 ms (flat/worse) | no |
| 10-15 | layer-major v3 (`lm-b5`) | per-layer windows for an 8192 chunk | H2D 42.37 GiB (-45.8% vs 78.15) but per-layer wall sum 61.57 s vs 37.06 s chunk-major TTFT; bank_wait 32.09 s | no |
| 10-15 | layer-major v4 ring | pre-stage while the previous layer computes | source verdict: structurally impossible for router-derived sets; only the first batch is pre-staged; -0.29 s of 61.57 s [est]; not run (below gate) | closed by dependency (E6) |
| 10-15/16 | E7 + G5 layer-stage arena | whole non-resident layer staged on `copy_q_` during the drain | bench `no_go` (inflation 104.9/105.4 ms); e2e block flat/worse, supply conserved into router_wait; decode 6.83/10.81 tok/s | NO-GO as implemented |
| 10-15 | D1 decode ordering (`CPU_ROUTE_EARLY`, `STAGEIN_HOIST`) | move CPU route classification / stage-in earlier in the decode window | 60.82 ms vs 59.46 ms baseline (neutral-to-negative); `CPU_DEFER_ARENA_WAIT=1` no effect | default off (ledger L54/L57) |
| 10-15 | C3 startup hint overlap (`fe1bf9e`, `626fada`) | event-based async arena/`PrefetchVirtualMemory` | startup 135.4 -> 58.5 s | adopted; startup domain only, no TTFT effect |
| 10-16 | topk-split patch (`tmp/q4e-sprint-r1-20261014/prefill-topk-split.patch`) | publish the route as two row-half launches and wait the first half early | source-level self-negation: the first-half event still sits behind the whole attention chain on the in-order queue, so it leads by about half the top-k kernel's own device time; the host work that could fill the window is already submitted before the wait; the value form (per-half chain) needs a non-default attention tile and M-dependent GEMM splits, i.e. not bit-exact | pre-registered: do not spend arms unless the receipt itself is wanted (|delta| <= 0.2%) |

## 6. Design verdict: the overlap families that are closed

F1 **Relocate the same bytes** (deeper ring, bigger chunks, direct DMA, separate
queue, whole-layer arena, layer-major ring): closed by section 1 plus the
conservation channel.
F2 **Engine placement** (strata C2): closed for the SYCL/UR form Aila runs; one
interop cell remains open with a predicted-negative outcome (section 4).
F3 **Speculative whole-layer supply** (E7/G5): closed e2e (L5).
F4 **Cross-layer prefetch of routed sets**: structurally impossible - the router
reads the post-attention hidden (`dense_rows` on `buf_.x` at
`src/models/Qwen4ExpBnb4Backend.cpp:2535`; proof and order in
`docs/plans/qwen38-r1-prefill-pipeline-2026-10-15.md` section 1).
F5 **Route-publication split** (E2 fallback): self-negated at source level
(section 5, last row).
F6 **Bigger chunk / staging 64 / ring 4** (charter section 2): measured hard
no-go (chunk 8192: TTFT 134.8 s, bank_wait 56.2 s, cost model 6.36).
F7 **Per-chunk microlaunch overlap**: already implemented (lookahead at
`:1672-1701`); there is nothing left to add at that level.

Residual R1: the L0 ordinal-1 interop overlap test (F2's open cell).
Residual R2: the never-run per-chunk fixed-cost falsification (`e2-slots8`,
`docs/plans/qwen38-r1-prefill-overlap-2026-10-15.md` section 6) - it prices the
pump residual, not the law.

**Conclusion for the P2 question: no un-negated bit-exact overlap mechanism
survives.** What remains is byte reduction, transfer-rate recovery, device work,
and one supply-elimination design (candidate Z).

## 7. The remaining true paths, with numbers

Baselines used below: block = 4096 tokens; 19,360 staged experts x 2,534,400 B =
46.8 GiB = 50.27 GB per block; link 9.81-9.89 GB/s; DMA floor 4.96 s/block;
device work 6.8 s attention (GDN 4.1 + full 2.7) + 0.6 s HC/router + 0.55-0.72 s
expert kernels = ~8.0 s; measured block 19.05-19.90 s (all from
`docs/plans/qwen38-r1-cost-model-2026-10-14.md` 6.29/6.31 and the E2 audit).

### P-A Reduce bytes (46.8 GiB/block)

- A1 residency: staged 19,360 vs 21,784 non-resident -> 89% coverage already;
  the ceiling is -11% bytes = -0.5 s/block (-2.7%) [est] and it needs VRAM the
  decode path also wants. Low value.
- A2 fewer blocks per prompt: chunk 8192 and layer-major both remove ~half of
  the re-staging, and both measured worse (134.8 s TTFT; 61.57 s layer window).
  The causal explanation is missing from every receipt I read (uncertainty U3).
  This is the only path with a >30% byte reduction, so it deserves one
  source-level archaeology before any more arms.
- A3 compute the experts on the CPU instead of shipping bytes: excluded -
  40,960 rows is per layer; 1,966,080 rows/block at the measured rate is ~105 s
  (`docs/plans/qwen38-r1-cpu-moe-bench-2026-10-15.md:116-120`). Note this
  supersedes cost model 6.33's 2.2-3 s/block estimate.
- A4 smaller weights: denied (W4A8 <= 0.7%); NF4 is the floor.

### P-B Recover the transfer rate

Measured: link 9.81-9.89 GB/s standalone; the production pump delivers
5.6 GB/s effective (banks 4/slots 16 cycle 7.24 ms per 40.55 MB chunk) and the
bench-form pump 8.5 GB/s (49 GB in 5.73 s). The gap between the bench pump and
the production pump is fill + fence + submit overhead = ~3.8 s/block [est].
Every knob tried so far only moved this gap between markers; capturing it needs
a construction change (fewer host-mediated steps), not a knob. Ceiling here:
~3-4 s/block (-15-20%) if the pump ever ran at link rate with zero fence cost.

### P-C Reduce device work

Attention 6.8 s/block: GDN prefill anchored at 74.82 ms/layer, 516.7 GB/s
effective, L2-served (`docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md`); SLM
variants measured -61..-86% and closed. Full attention 2.7 s/block; expert
kernels 0.55-0.72 s/block (68-89 GB/s effective - not at the wall, but too small
to matter: 2.8% of the block). Decode-attention G=4 is decode-side only.
Nothing here moves prefill.

### P-D The arithmetic that bounds the goal

Current: 19.4 s = device 8.6 + DMA 4.96 + pump overhead 3.8 + slack ~2.0.
Add-law floor (perfect pump, DMA still additive): device + DMA = ~13.5 s/block
-> 14K TTFT ~46-48 s = ~300-310 tok/s [est]. 400 tok/s needs 10.24 s/block: the
DMA must be genuinely hidden (sections 1/5 say it is not) or the bytes must drop
~35%. Cross-reference: cost model 6.31 quotes E1's rev2 "bytes+device alone reach
only ~368 tok/s"; I could not re-derive that arithmetic from the receipts (U4),
but it points the same way.

## 8. Candidate Z: fold the supply into the consumer kernel (device reads host memory)

Mechanism: instead of staging cold experts into VRAM and then reading them, let
the expert kernels read the cold experts directly from host memory over PCIe
(USM host / device-readable host buffers). No copy engine involvement, no bank
DMA, no new queue; the byte stream becomes part of the kernel's own memory
traffic. Per-layer ordering, row order and all arithmetic are unchanged, and the
bytes are the same bytes, so the result is bit-exact by construction. Resident
experts are untouched.

Why it is different from every denied item:

- vs C1 (stream-ahead / deeper ring): C1 keeps a copy and tries to hide it; Z
  removes the copy and pays the bytes inside the consumer kernel, so there is no
  second engine to contend with - the law of section 1 becomes the design's
  baseline instead of an obstacle.
- vs C2 (copy-only queue): Z does not place a copy anywhere; section 4's
  engine-placement question becomes irrelevant.
- vs E7/G5 layer-stage: no speculative set, no arena H2D, no extra VRAM, no
  `WaitReady`, no +12% bytes.
- vs topk-split: no route or chain split, no M-dependent GEMM change.

Two variants:

- **Z1 (low risk)**: keep the fill, delete the H2D. The cold chunk is filled into
  the existing staging bank - which is already `sycl::malloc_host`
  (`src/models/Qwen4ExpAdaptiveCache.cpp:237`) - and the kernel reads that bank.
  The `usm_host_allocations` aspect is already queried in production
  (`src/models/Qwen4ExpExpertExecutor.cpp:658-659`) and the top-k kernel already
  writes into host USM (`router_host_ids_` allocated via `ctx.alloc_host`,
  `src/models/Qwen4ExpBnb4Backend.cpp:1127`).
  Wall [est] = 8.03 (device minus the old VRAM MoE reads) + 50.27 GB / R + 2.47
  (fill) + 1.32 (fence) + ~0.6 (submits), serial as today.
  R = 10 GB/s -> 17.4 s/block (-10%); R = 12 -> 16.6 (-15%).
- **Z2 (higher risk)**: read the large-page arena directly, deleting fill and
  fence as well. Wall [est] = 8.03 + 50.27 / R + 0.6.
  R = 8 -> 14.9 s (-23%); R = 10 -> 13.7 (-29%).

Decision input is one measurement: the effective kernel-read rate R from host
USM (section 9, P2). The kernels currently consume weights at only 68-89 GB/s
from VRAM, so at R >= 8 GB/s the read becomes the kernel limiter and the ALU
rides along; below R = 6 GB/s the mechanism is negative and should be closed.

Minimal change points (for the coordinator to assign, not applied here):

1. A host-pointer slot descriptor: mirror `GetStagingSlotDev`
   (`src/models/Qwen4ExpAdaptiveCache.hpp:206-215`) with a host base pointer, and
   let the pump fill `adaptive_prefill_slots_[e]` from it
   (`src/models/Qwen4ExpBnb4Backend.cpp:1693-1695`).
2. In `StageColdExpertsAsync` (`src/models/Qwen4ExpAdaptiveCache.cpp:649-655`),
   behind a new default-off env, skip the H2D submit and return the fill plus the
   bank's consumer event as the consumer gate. The fence semantics
   (`RecordStagingConsumer`) must stay, because the next fill must not overwrite
   a bank the previous kernel is still reading. No kernel math changes: the
   descriptors carry plain pointers.
3. Cancel/drain discipline: the same `copy_q_` drain plus a host-bank in-use
   counter, per `AGENTS.md` section 6.

Most likely failure modes, in order:

1. Kernel reads from host USM on DG2/WDDM are latency-bound far below the link
   rate (uncached 4B/16B sectors over PCIe). P2 decides this.
2. Z2's arena is a large-page allocation
   (`src/models/Qwen4ExpHostArena.cpp:146`, `memory::AllocateLargePages`), not a
   USM host allocation; making 51 GiB device-readable may force regular pages,
   which would regress the CPU-side expert path (45 GB/s reads) and the startup
   fill. A dual representation is impossible on a 64 GB box with a 51 GiB
   complement. Z1 avoids this at the price of the fill + fence terms.
3. If the read is fast but the fence (1.32 s) is not hideable, Z1 stalls at the
   -10% level.
4. L2 pollution: host reads may evict attention/KV working sets; not modelled.

## 9. Probes the coordinator can run (ranked)

P1 - zero code, one arm (~5 min GPU): falsify the pump's per-chunk fixed cost
model (E2 section 6; never ran). Command:

    python tmp/q4e-sprint-r1-20261014/run.py e2-slots8 --prime --diag --generate=8 \
        AILA_Q4E_STAGE_COPY_PARALLEL=1 AILA_Q4E_PREFILL_STAGING_BANKS=4 \
        AILA_Q4E_STAGING_SLOTS=8

Read `[Q4E PrefillStage] slots_per_bank=8` then `fill_ms` / `bank_wait_ms` /
`router_wait_ms` / `submitted_ms`. If the fixed cost is per chunk, the cycle is
~5.2 ms -> supply ~12.6 s (+44%); if it is per byte, supply stays 5-8 s. Either
way this prices the pump residual (P-B) and closes E2's open model.

P2 - needs ~60-80 lines in `tests/ops/Qwen4ExpKernelTests.cpp` plus one build and
one GPU run (~1 min), and it decides candidate Z: a host-USM streaming-read
probe. Spec: query `device.has(sycl::aspect::usm_host_allocations)`; allocate
512 MB with `sycl::malloc_host`; fill with a known pattern and bit-check on the
host; run a streaming-read kernel with the expert-kernel access shape (uint32
per lane plus a per-row gather) over the whole buffer; report effective GB/s and
a finite/sentinel guard; optionally repeat with `vec8<bf16>` loads to test the
16B pattern. Decision table: R >= 10 -> Z1 and Z2 both live; 8-10 -> Z2 live;
6-8 -> marginal, Z2 only; < 6 -> close Z (write the negative result).

P3 - optional, medium cost: the L0 ordinal-1 SYCL-interop overlap probe
(section 4 residual). Predicted negative; do it only if the C2 family needs a
closed cell for the record.

Drop: the topk-split A/B/B/A arms (pre-registered |delta| <= 0.2%,
`tmp/q4e-sprint-r1-20261014/prefill-topk-split-README.md` section 3).

## 10. Uncertainties

- U1: the layer-stage bench's `overlap_stage_ms` equals the overlap round's load
  time, so part of its accounting may reflect the arena-consumer gate rather
  than pure bandwidth contention. The additivity conclusion is nonetheless
  corroborated by L1/L2 (micro-probe) and L5 (e2e conservation).
- U2: the e2e layer-stage interference (+11.3 s of router_wait) exceeds the
  additive prediction (+5.3 s of DMA). The excess (~6 s) is unexplained
  (fill-vs-DMA contention? VRAM/arena side effects? WaitReady tail?). The
  qualitative conclusion (the supply is not hidden) is firm; the quantitative
  model is approximate.
- U3: the chunk-8192 collapse (bank_wait 56.2 s, TTFT 134.8 s) has no causal
  explanation in any receipt. It is the only path with a >30% byte reduction, so
  a source-level archaeology is worth an agent before more arms.
- U4: cost model 6.31's "~368 tok/s" (E1 rev2) could not be re-derived from the
  receipts; my own floor estimate is ~300-310 tok/s.
- U5: it is unproven whether UR's copy-engine env actually lands the copies on
  the L0 ordinal-1 copy-only group (only the overlap behavior is measured).
- U6: device-to-host USM *read* throughput on this box is entirely unmeasured.
  Production only exercises small host USM writes (top-k ids) and the default-off
  `AILA_Q4E_CPU_RESULT_HOST_READ` path
  (`src/models/Qwen4ExpExpertExecutor.cpp:655-660`). This is exactly what P2 is
  for.

## Appendix: citation index

- Section 1: `tests/ops/Qwen4ExpKernelTests.cpp:4374`;
  `docs/plans/qwen38-strata-c1-prefill-staging-screen-2026-10-13.md` sections 2-4;
  `docs/plans/qwen38-strata-c2-copy-queue-screen-2026-10-13.md` sections 2, 6;
  `tmp/q4e-sprint-r1-20261014/layer-stage/bench-A{1,2-g8}.log:18-19`;
  `docs/plans/qwen38-r1-cost-model-2026-10-14.md` 6.29/6.32/6.36;
  `docs/plans/qwen38-r1-layer-stage-2026-10-15.md` sections 3, 8;
  `tmp/q4e-sprint-r1-20261014/g5-b1-worker.log:88-99`.
- Section 2: `src/models/Qwen4ExpBnb4Backend.cpp:885,1453,1540,1672-1702,2535,2558,2596,2718,2835`;
  `src/models/Qwen4ExpAdaptiveCache.cpp:606-665`;
  `src/models/Qwen4ExpAdaptiveCache.hpp:219-222`;
  `src/utils/EnvUtils.hpp:179-201`; `src/worker/WorkerMain.cpp:494`;
  `docs/plans/qwen38-r1-prefill-overlap-2026-10-15.md` sections 2-4.
- Section 3: the six strata C-docs under `docs/plans/qwen38-strata-*` plus
  `docs/plans/qwen38-strata-candidates-execution-2026-10-13.md`;
  `docs/plans/qwen38-r1-cpu-moe-bench-2026-10-15.md:116-120,179`.
- Section 4: `tmp/Strata/sycl/include/strata/copy_queue.hpp:1-26`;
  `tmp/Strata/sycl/src/core/copy_queue.cpp:30-70`;
  `tmp/Strata/docs/INTEL.md:824-829`; `tmp/q4e-strata-c2/l0_copy_probe.cpp`.
- Section 5: `tmp/q4e-sprint-r1-20261014/COORD-NOTE-2026-10-15.md`;
  `docs/plans/qwen38-r1-cost-model-2026-10-14.md` 6.31/6.36/6.38/6.39 and ledger
  L54/L57; `docs/plans/qwen38-r1-layer-major-v4-2026-10-15.md` sections 4-5;
  `tmp/q4e-sprint-r1-20261014/prefill-topk-split-README.md` section 2.
- Sections 7-8: `docs/plans/qwen38-r1-gdn-prefill-2026-10-15.md`;
  `docs/plans/qwen38-r1-prefill-400-2026-10-15.md`;
  `src/models/Qwen4ExpHostArena.cpp:146`;
  `src/models/Qwen4ExpAdaptiveCache.cpp:237`;
  `src/models/Qwen4ExpExpertExecutor.cpp:655-660`;
  `src/models/Qwen4ExpBnb4Backend.cpp:1127-1128`.
