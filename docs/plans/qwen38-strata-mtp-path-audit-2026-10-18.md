# Qwen4Exp MTP path audit vs Strata (2026-10-18)

Scope: current Qwen4Exp MTP path, production loader, draft layer, target
verification window, decode economics, and the Strata residency/dispatch
principles. This is a read-only source and receipt audit. No GPU command was
run, no build was performed, and no source file was changed.

Base revision: `ab28c3c` (`docs(q4e): audit cold expert CPU/GPU split`).
The working tree already contained another agent's uncommitted cold-split
document before this audit began; it was read for cross-checking but was not
staged or committed here.

Inputs:

- `tmp/q4e-strata-principle/REPORT.md`
- `docs/plans/qwen38-r1-mtp-longctx-2026-10-15.md`
- `docs/plans/qwen38-mtp-align-2026-10-11.md`
- `docs/plans/qwen38-mtp-acceptance-2026-10-11.md`
- `docs/plans/qwen38-qwen4exp-pipeline-status-2026-10-13.md`
- `docs/plans/qwen38-r1-mtp-step-budget-2026-10-15.md`
- `docs/plans/qwen38-r1-mtp-width-economics-2026-10-16.md`
- `docs/plans/qwen38-r1-mtp-top1-2026-10-15.md`
- `docs/plans/qwen38-strata-cold-expert-split-audit-2026-10-18.md`
- `docs/plans/qwen38-r1-copyq-graph-falsification-2026-10-17.md`

The filename requested as `qwen38-mtp-alignment-20261011.md` does not exist in
this tree. The current equivalents are `qwen38-mtp-align-2026-10-11.md` and
`qwen38-mtp-acceptance-2026-10-11.md`; both were used.

## 0. Verdict

1. **MTP is not a host-compute pipeline.** The dense MTP layer, HC, router,
   shared expert, routed-expert GPU kernels, draft KV, target KV, and the
   target recurrent-state checkpoint all execute or reside on the device. The
   host paths are embedding-row staging, route visibility, expert fill/DMA,
   and the target verify window's CPU cold-expert work.
2. **The current MTP cannot load in the 32K production server.** The hard
   load gate is still `maxSequenceLength <= 2048`, `GDN_CHUNK_MODE == 0`, and
   `!AILA_Q4E_LM_HEAD_Q8` (`src/inference/Model.cpp:800-802`).
   `Qwen4ExpMtpRuntime::EffectiveCapacity` exists but is not wired into the
   loader or reservation path in the current tree.
3. **The 64-slot draft cache is not the main wall.** Its direct H2D upper
   bound is under 1 ms/token. A 128-slot expansion was already measured at
   `19.95 -> 15.38 tok/s`; full 512-slot residency costs a full 1.208 GiB and
   displaces 512 target-expert slot equivalents. Do not import Strata's
   708 MB draft number: Aila's NF4 slot geometry is different.
4. **Host embedding is real but too small to lead.** The draft copies one
   5,120-byte row per Step and blocks on a host-to-device copy. At the measured
   small-copy cost this is roughly 0.04-0.14 ms/window. Moving the 1.184 GiB
   table to VRAM would displace about 502 target slots, a clear negative trade.
5. **Width 2 is already the only live width.** The target verify alone costs
   92.26 ms. Even if every non-verify MTP cost were free, the width-2 window
   would be only 1.748 / 0.0926 = 18.88 tok/s. The target verify must lose at
   least another 5.2 ms before 20 tok/s is possible.
6. **The best remaining MTP-specific structural candidate is a shared Q8 LM
   head.** It saves 568.4 MiB (about 235 target slots) and should remove
   2.3-3.0 ms/window because the head is read by both the draft Step and the
   target verify. It is currently hard-disabled for MTP.
7. **The second structural candidate is the per-request 2048 window, not a
   larger global window.** It restores short-session MTP availability under a
   32K server, but its measured throughput is neutral and it is not a route to
   20 tok/s. Raising the default window to 14K/32K adds seconds of catch-up
   and is NO-GO.

## 1. Strata principle vs Aila current placement

| Component | Strata | Aila current | Verdict |
|---|---|---|---|
| Gated residual / HC | GPU-resident | HC weights and all HC launches are GPU; main HC reads about 1.279 GB/token | aligned |
| Router | GPU-resident, device-side doorbell | Router weights GPU; host waits for IDs once per layer/Step | compute aligned, handoff is host |
| Shared expert | GPU-resident | 48 shared experts GPU; MTP shared expert GPU | aligned |
| Token embedding | GPU-resident | BF16 table in host large-page memory, 1.184 GiB; one row gathered and H2D per Step | mismatch, but cheap per token |
| LM head | GPU-resident, much smaller quantized head | BF16 GPU head 1.184 GiB; Q8 head 644.1 MiB is opt-in and hard-blocks MTP | structural mismatch |
| MTP dense layer | GPU-resident | about 80.5 MB dense MTP weights on GPU | aligned |
| MTP routed experts | all 512 draft experts resident | 64-slot GPU cache, 154.7 MiB; remaining experts filled from host on miss | capacity mismatch |
| Target routed experts | GPU hot cache plus PCIe/CPU cold paths | 2800 GPU slots plus host arena and CPU verify path | broadly aligned, host lane is costly |
| Target KV | active window on GPU; optional host streaming for very long contexts | GPU-resident; no KV streaming implemented | aligned for current 32K |
| Draft KV | per-MTP-layer GPU KV | per-session GPU KV, separate from target KV | aligned |
| Target recurrent state | device state with checkpointing | GPU state plus 118,038,528-byte device checkpoint | aligned compute, host-orchestrated copy |
| Whole-token graph / host doorbell | CUDA graph split plus doorbell | Limited SYCL graph cannot publish host-visible route until graph exit on Windows/L0 | NO-GO, already measured |

The key distinction is that Aila's host embedding is a **gather/staging path**,
not a hidden CPU transformer. The expensive host work is the target verify
window, not the MTP dense layer.

## 2. Current capacity gate and long-context cost

Current loader behavior:

```text
enableMtp =
    qwen4exp && requestedMtp &&
    maxSequenceLength >= 2 &&
    maxSequenceLength <= 2048 &&
    GDN_CHUNK_MODE == 0 &&
    !LM_HEAD_Q8
```

The production server starts with `--context 32768`, so `enableMtp` is false
before any request is admitted. `Qwen4ExpMtpRuntime::SessionBytes(Capacity())`
and the reservation also use the full session capacity in the current code.
Removing only the `<= 2048` condition would therefore allocate the wrong MTP
window and pay a long prompt catch-up cost. The existing long-context design
patch (`EffectiveCapacity`) is not merged.

Window-scaled MTP memory:

| Window | Draft KV/session | Hidden rows | Window subtotal | MTP total including fixed pieces | Catch-up / verdict |
|---:|---:|---:|---:|---:|---|
| 2048 | 4.64 MiB | 40.00 MiB | 44.64 MiB | about 409-414 MB | short sessions only; low-risk availability candidate |
| 14000 | 31.64 MiB | 273.44 MiB | 305.07 MiB | about 654 MB | needs 4.2-18.2 s prompt catch-up; not a default |
| 32768 | 74.02 MiB | 640.00 MiB | 714.02 MiB | about 1.04 GB | catch-up and VRAM both worsen; NO-GO as a global default |

The 2048 row uses the measured `SessionBytes(2048) = 4,870,144` bytes and the
existing `hiddenRows = [cap, 10240]` geometry. The 14K and 32K rows are
derived from the same arithmetic. The fixed MTP pieces include the checkpoint,
linear inputs, draft dense weights, and the 64-slot expert cache.

The correct capacity change is:

1. Use a per-model MTP window (default 2048) independent of the 32K session
   capacity.
2. Reserve and allocate with `EffectiveCapacity(maxSequenceLength)`, not the
   full session capacity.
3. Invalidate MTP only for sessions or prefill chunks outside that window.
4. Keep the short-session enablement and long-session T1 fallback separate;
   do not silently pay MTP prompt catch-up on 14K requests.

This change is availability work. It does not make MTP a 20 tok/s path.

## 3. Draft expert cache: 64 slots vs 512

Aila's MTP cache uses:

```text
kMtpExpertSlotBytes = 2,534,400
kMtpExpertSlots     = 64
64 * 2,534,400      = 162,201,600 B = 154.7 MiB
```

Measured 128-token draft behavior:

| Cache | Cache VRAM | Draft hit rate | Misses / 128 tokens | H2D bytes / 128 tokens | Draft time | Target side | Outcome |
|---:|---:|---:|---:|---:|---:|---|---|
| 64 slots | 154.7 MiB | 66.3% | 431 | 1,092,326,400 | 7.525 ms | baseline | current |
| 128 slots | +154.69 MiB | 80.8% | 246 | 623,462,400 | 6.843 ms | verify 77.38 -> 99.09 ms; target slots 3418 -> 3352 | 19.95 -> 15.38 tok/s, NO-GO |
| 512 slots | 1,208.0 MiB | 100% | 0 | 0 | H2D upper bound only | costs 448 extra target slots over 64 | NO-GO on 16 GB |

The 512-slot line is the important Strata correction:

- Aila's 512-slot draft cache is **1.208 GiB**, not Strata's 708 MB.
- Its total VRAM cost is equal to **512 target-expert slot equivalents**.
- The incremental cost over the current 64-slot cache is 448 target slots.
- Even a perfect elimination of draft H2D is bounded by about 0.93 ms/token
  (1.62 ms/window at the measured 1.748 tokens/window), before any loss of
  target hot-set coverage.
- The 128-slot experiment already proved the target-side loss is much larger
  than the draft-side gain.

The correct conclusion is not "copy Strata's 512 draft experts". On this
machine, full draft residency is a cache-budget regression unless the target
expert cache is deliberately reduced or more than 1.1 GiB is freed elsewhere.
Neither condition is present in the current 16 GB production bundle.

A 64-slot MTP-specific hotset could reduce misses without increasing VRAM, but
its entire H2D ceiling is below 1 ms/token and the current LRU already reaches
two-thirds hit rate. It is a secondary probe, not a structural answer.

## 4. Host embedding and the shared LM head

### 4.1 Host embedding

`Qwen4ExpMtpDraft::Step` does:

```text
host memcpy: 2,560 BF16 values = 5,120 B
Context::memcpy_h2d(...): blocking small copy
GPU embedding RMS + FC
```

The measured small host-pointer H2D class is about 22 us on the same queue;
the stack-source class is 72-79 us. The MTP window executes one full Step and,
on acceptance, one `kvOnly` Step, so the average is about 1.748 copies/window.
The resulting upper bound is 0.04-0.14 ms/window, or well below 0.2% of the
112.5 ms window.

The full embedding table is 1,184.0 MiB and would displace about 502 target
expert slots if uploaded. That is a large negative trade for a 5 KB row
operation. Keep the host table; if this path is ever revisited, use a small
hot-row cache or a device-side host-USM gather, not the full table.

### 4.2 Q8 shared head

The head is shared by the target and MTP draft:

- BF16 weight: 1,271,398,400 B = 1.184 GiB.
- Q8 weight plus scales: 675,430,400 B = 644.1 MiB.
- Saving: 595,968,000 B = 568.4 MiB, about 235 target-slot equivalents.
- Measured T1 A/B/B/A: `17.451 -> 18.258 tok/s` (+4.62%), sample window
  3.35 -> about 2.1 ms.

The Q8 head is read twice per MTP window, once by the draft Step and once by
the target verification forward. A conservative 1.1-1.5 ms saving per read
gives 2.3-3.0 ms/window. That moves W from 112.5 to about 109.5-110.2 ms,
or 15.9 tok/s. The slot saving can improve target cache pressure further, but
it does not close the 20 tok/s gap by itself.

This is not a one-line gate removal. In the current backend the Q8 dispatch
lives inside the local `forward_head` lambda, while
`Qwen4ExpMtpDraft::Step` calls `target_.lm_head_.forward(...)` directly:

- With `AILA_Q4E_LM_HEAD_Q8=1`, the BF16 `lm_head_` is not initialized.
- The MTP constructor rejects the Q8 configuration.

The minimal structural implementation is a shared backend helper such as
`ForwardLmHeadRows(ctx, input, rows)`, used by both the target forward and the
MTP draft Step. Then remove the MTP Q8 rejection and retain the Top-1 gate.

## 5. Width 2, acceptance, and the hard upper bounds

Current production semantics:

```text
width 2 = pending token + one draft
tokens/window = 1 + p1
```

Measured long prompt:

| Quantity | Value |
|---|---:|
| p1 acceptance | 74.8% |
| tokens/window | 1.748 |
| target verify | 92.26 ms |
| draft plus state/select | 19.9 ms |
| total window W | 112.5 ms |
| measured rate | 15.54-15.59 tok/s |
| T1 comparison | 17.12-17.33 tok/s, about 57.7 ms/token |

Width estimates from the same per-row path:

| Width | Rows / drafts | Tokens/window T | Current W estimate | Current rate | Verdict |
|---:|---|---:|---:|---:|---|
| 2 | 2 / 1 | 1.748 | 112.5 ms measured | 15.5 tok/s | current |
| 3 | 3 / 2 | 2.153 | 155-165 ms | about 13.1 tok/s | NO-GO |
| 4 | 4 / 3 | 2.371 | 196-210 ms | 10.6-13.4 tok/s | NO-GO |

Strata's 4-token window benefits from about 3.05x expert dedup. Aila's CPU
merge experiment measured only `f = 0.154` and regressed
`16.31 -> 16.19 tok/s` (-0.75%) with identical output. Aila cannot currently
reproduce Strata's wide-window economics.

Independent "if X were free" upper bounds at width 2:

| X removed | Resulting W | Resulting rate |
|---|---:|---:|
| nothing | 112.5 ms | 15.54 tok/s |
| entire draft Step | 104.3-106.3 ms | 16.44-16.76 tok/s |
| capture/restore/select/sync | 98.8-100.8 ms | 17.34-17.69 tok/s |
| all non-verify MTP work | 92.6 ms | 18.88 tok/s |
| Q8 head only | 109.5-110.2 ms | 15.86-15.96 tok/s |
| full 512 draft cache | 110.9-112.0 ms | 15.61-15.76 tok/s |
| host embedding only | about 112.4 ms | 15.55 tok/s |
| target state capture only | 110.8-111.0 ms | 15.75-15.78 tok/s |

The central result is the third row: even a completely free draft, capture,
selection, and synchronization path leaves the 92.26 ms target verify and
only 18.88 tok/s. To reach 20 tok/s, the verify window itself must lose at
least 5.2 ms. At 92.6 ms, a width-2 acceptance of `p1 >= 0.852` would be
required; the measured long-prompt p1 is 0.748. Acceptance improvement is not
the first lever.

The user's quality waiver is compatible with this analysis: the remaining MTP
obstacle is speed and availability, not tensor-bitwise equality. The gate for
any future Q8 or width change is teacher-forced Top-1 agreement >= 0.95, plus
the ordinary text/output check. The existing width-4 short sample is 61/63 =
0.968, but it is not a large enough sample to justify a new default.

## 6. Cold experts and host rendezvous

The previous cold-split audit (committed as `ab28c3c`) already established:

- Canonical adaptive T1 sends every cold miss to CPU because
  `CPU_MAX_EXPERTS=10` equals top-k.
- `COLD_POLICY=3` and `COLD_PCIE_NUM` are not connected to the adaptive
  `StartArenaCold` path.
- A cap=4 fixed split sent about 24% of misses to GPU staging and regressed
  the retained20 token from 66.006 to 95.518 ms; end-to-end continuation fell
  15.50 -> 9.40 tok/s (-39%).
- A token-scoped 40/256 split with direct large-page DMA is the only admissible
  new cold-split probe, and its ideal T1 upper bound is only about +6%.
- Cross-row CPU expert merge is closed: f = 0.154 and -0.75%.

For MTP verification, the same caution applies more strongly:

```text
width-2 verify: 587 cold CPU tasks
cpu_compute:    29.47 ms
cpu_join:       24.84 ms
experts_host:   53.97 ms
staged/H2D:     0
```

An all-GPU staging arm was measured at 253 ms and about 5.9-6.5 tok/s. A fixed
CPU/DMA dose sweep is therefore prohibited by the existing evidence. The
MTP-specific cold-split probe should remain deferred until a direct-DMA
independent lane exists and is first proven on canonical T1.

The host route rendezvous is also part of the target verify cost:
`route.wait_and_throw()` precedes CPU expert dispatch, and the whole-token
graph attempt could not publish a host-visible route payload before graph exit
on Windows/L0. That graph path is NO-GO and should not be retried as an MTP
fix.

## 7. KV and target state

MTP draft KV is per-session and on device:

| Capacity | Draft K/V + indexer | Notes |
|---:|---:|---|
| 2048 | 4.64 MiB | small relative to the 112.6 MiB checkpoint |
| 14000 | 31.64 MiB | long-context draft dense attention costs about 0.34 GB/step, roughly +0.77 ms at 450 GB/s |
| 32768 | 74.02 MiB | not the reason to reject long MTP; catch-up and target cost dominate |

The target KV is not copied to host. The only large target checkpoint is the
linear/GDN state:

```text
checkpoint: 118,038,528 B = 112.6 MiB
capture:    about 1.13-1.31 ms/window
restore:    about 1.10-1.45 ms on the 25% reject path
```

The checkpoint is device memory and uses device-to-device copies. It is not a
host KV path. If it were completely free, the width-2 rate would be only about
15.75 tok/s. The earlier attempt to save the first-token state regressed the
full window (19.89156 -> 19.62932 tok/s), so that branch is closed.

KV streaming is a Strata mechanism for contexts beyond the resident window.
Aila's current 32K target already keeps KV on device and does not need a
host-KV streaming subsystem. Adding one solely for MTP would create a host
path without addressing the 92.26 ms verify wall.

## 8. Candidate ranking

### C1 - GO to one probe: shared Q8 LM head for MTP

Mechanism:

- Add a single backend helper for BF16/Q8 head dispatch.
- Use it from both the normal target forward and `Qwen4ExpMtpDraft::Step`.
- Remove the MTP hard rejection of Q8.
- Keep the target's Top-1 gate at >= 0.95; do not use tensor bitwise equality
  as the gate.

Upper bound:

- 568.4 MiB saved, about 235 target expert slots.
- About 2.3-3.0 ms/window from the two head reads.
- Expected width-2 rate about 15.9 tok/s; useful stack, not sufficient alone.

Probe:

```text
A: MTP=1, LM_HEAD_Q8=0
B: MTP=1, LM_HEAD_Q8=1
```

Use the existing MTP bundle (agent6k + int8 KV) and a short, generalizable
prompt above the QSA enable threshold. Gate on actual Q8 load, MTP
`windows > 0`, teacher-forced Top-1 >= 0.95, unchanged output semantics, and
the normal end-to-end threshold. A single A/B is enough for the first screen;
only a default-on decision needs the full receipt.

### C2 - GO to one availability probe: per-request 2048 MTP window

Mechanism:

- Wire `EffectiveCapacity` into reservation, `CacheBytes`, construction, and
  `AfterForward`.
- Keep the 2048 default window.
- Allow short sessions under a 32K server to use MTP; invalidate sessions or
  prefill chunks outside the window without failing the request.

Upper bound:

- About 409-414 MB total MTP footprint at 2048.
- Long 14K/32K sessions do not gain speed; they fall back to T1.
- This is an availability fix, not a 20 tok/s candidate.

Probe:

- 32K server, prompt <= 2048: compare T1 vs per-request MTP; require the
  long-session arm to report `windows=0` and identical content.
- 14K prompt: require unchanged TTFT and decode rate within the A/A band.
- Short arm Top-1 >= 0.95 on at least two prompt lengths.

### Not selected

| Item | Disposition |
|---|---|
| 512 MTP draft experts resident | NO-GO on 16 GB: 1.208 GiB, 512 target-slot equivalents |
| GPU-resident full embedding | NO-GO: 1.184 GiB, about 502 target slots for a 5 KB row |
| Width 3/4 | NO-GO: per-row cost, 10.6-13.4 tok/s estimate |
| Cross-row CPU expert merge | NO-GO: f=0.154, -0.75% |
| Fixed CPU/DMA dose sweep | NO-GO: cap=4 split -39% end-to-end |
| Target state capture rewrite | minor: <2 ms upper bound; first-state variant regressed |
| Whole-token graph / host doorbell | NO-GO on current Windows/L0 path |
| Out-of-order queue conversion | already NO-GO; not retried |

## 9. Final answer

MTP has the correct broad placement for the dense and KV path: HC, router,
shared expert, LM head, target KV, draft KV, and state checkpoint are on the
device. It does not have Strata's capacity or execution topology:

- The 32K server never constructs MTP because of the 2048 hard gate.
- The draft cache is 64 slots, not 512; full residency costs 1.208 GiB and is
  not a valid 16 GB trade.
- Host embedding is a small real path, not the bottleneck.
- Width 2 is the only viable width; even a free draft cannot exceed 18.88
  tok/s because the target verify alone is 92.26 ms.
- The remaining useful MTP-specific structural change is the shared Q8 head.
- The only other clean structural probe is per-request 2048 enablement, and it
  is an availability change rather than a speed path.

No current evidence supports enabling MTP by default. Keep it opt-in until a
candidate demonstrates both the Top-1 gate and the end-to-end threshold.

## 10. Receipts

- `docs/plans/qwen38-r1-mtp-step-budget-2026-10-15.md`
- `docs/plans/qwen38-r1-mtp-width-economics-2026-10-16.md`
- `docs/plans/qwen38-r1-mtp-longctx-2026-10-15.md`
- `docs/plans/qwen38-mtp-window-cache-8724-2026-10-06.md`
- `docs/plans/qwen38-mtp-projection-reuse-2026-10-06.md`
- `docs/plans/qwen38-strata-cold-expert-split-audit-2026-10-18.md`
- `docs/plans/qwen38-r1-copyq-graph-falsification-2026-10-17.md`
- `tmp/q4e-resident-speed-20261009/g1-b1-worker.log`
- `tmp/q4e-resident-speed-20261009/g1-b2-worker.log`
- `tmp/q4e-sprint-r1-20261014/mtp-top1/top1-gate.json`
- `tmp/q4e-sprint-r1-20261014/e10-bundle-worker.log`
- `tmp/q4e-sprint-r1-20261014/decode/retained20-dec-cpu4.json`
