# Qwen4Exp MTP long-context per-request enablement (R1, 2026-10-15)

E4 deliverable. Goal: remove the load-time "MTP requires max_seq <= 2048" hard
gate so MTP becomes a per-request decision: sessions inside the draft window get
MTP, longer sessions silently keep the T1 decode path.

Status: implementation done in the MTP-owned sources (uncommitted). The load
gate itself lives in `src/inference/Model.cpp`, which this task does not own;
the exact 5-hunk patch is in section 3.2. No build, no GPU run (sprint
discipline: build.ps1 and the GPU were owned by the main agent).

## 1. Current disable conditions (all of them)

Load time (worker process, once per model load):

| # | Location | Condition | Effect |
|---|---|---|---|
| L1 | `Model.cpp:726` | `AILA_Q4E_SERVER_MTP` (read-once static) not set | no runtime |
| L2 | `Model.cpp:727` | family != Qwen4Exp | no runtime |
| L3 | `Model.cpp:800-802` | `options.maxSequenceLength > 2048` | no runtime (**the hard gate**) |
| L4 | `Model.cpp:801` | `AILA_Q4E_GDN_CHUNK_MODE != 0` | no runtime |
| L5 | `Model.cpp:802` | `AILA_Q4E_LM_HEAD_Q8` set | no runtime |
| L6 | `Qwen4ExpMtpRuntime.cpp:12` | ctor belt: `cap > 2048` (now `cap > max_seq_len()`), GDN chunk, Q8 head | throw at load |
| L7 | `Qwen4ExpMtpDraft.cpp:154` (CreateState at 148) | draft state: `capacity > 2048` or `> indexer_budget` | throw on session create |

Why L3/L7 existed: `C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2/config.json` has
`indexer_budget = 2048`, `indexer_compress_ratio = 4`. The draft attends
densely over `position+1` rows (`Qwen4ExpMtpDraft.cpp:213`,
`attention_decode(..., position + 1, ..., active_indices=nullptr)`), so the gate
is a *cost* bound (draft attention stays <= what the target pays per QSA-budget
layer), not a correctness bound. The draft reuses target decode scratch sized
by `max_seq_len_`, so `capacity <= max_seq_len()` is the real safety bound.

Per session / per request (already existed, unchanged):

| # | Location | Condition |
|---|---|---|
| S1 | `Model.cpp:229-230` | `!mtp \|\| !valid \|\| length != CurrentLength() \|\| len < 1 \|\| len + 2 > Capacity()` -> `nullopt` |
| S2 | `Request.cpp:306-311` | greedy only, penalties 1/0/0, no thinking budget, `outputSlots >= 2`, `>= 2` new tokens left |
| S3 | `Model.cpp:291` | when an MTP runtime is loaded, prompt chunks pass `prefill_remaining = -1` |

Production runner evidence for L3: `tmp/q4e-sprint-r1-20261014/run.py` starts the
server with `--context 32768` (prompts 6000 and 14000), so `maxSequenceLength =
32768` and MTP was structurally impossible there. The 2048-context MTP arms
(`tmp/q4e-resident-speed-20261009/run_mtp_ab.py`, `--context 2048 --chunk 512
--prompt-tokens 1500`) are the only place MTP ever ran.

## 2. Per-request semantics

`capacity` (a.k.a. MTP window) is set once per model load; each session decides
at run time whether it may use it:

* Session stays inside the window for every forward -> draft windows exactly as
  today (Request.cpp eligibility decides per step).
* Session (or one prefill chunk) reaches past the window -> `AfterForward`
  invalidates the draft state *before* staging. `State.valid = false` makes
  S1 return `nullopt` forever, and `Request.cpp:331` then runs the plain
  `session->Execute({pendingToken})` T1 path. No exception, no device fault, no
  per-step MTP work (AfterForward returns at its first line).
* `Reset()` re-arms (`length = 0, valid = true`, `Model.cpp:256`) and
  `Truncate(0)` re-arms via the existing `valid = restored == 0` rule
  (`Model.cpp:269`); a non-zero truncate stays invalid (conservative).
* T1 output for a falling-back session is the pre-MTP code path byte-for-byte;
  the only intentional coupling left is S3, which the patch in 3.2 restricts to
  chunks inside the window.

## 3. Minimal implementation

### 3.1 Applied (uncommitted) in MTP-owned files

1. `Qwen4ExpMtpRuntime.cpp:12` - ctor gate relaxed from `cap > 2048` to
   `cap > target.max_seq_len()`; same GDN/Q8 checks, message updated.
2. `Qwen4ExpMtpRuntime.cpp:40` - new `EffectiveCapacity(maxSequenceLength)`:

   ```cpp
   static const int requested = aila::env::read_int("AILA_Q4E_MTP_MAX_CONTEXT", 2048);
   return std::clamp(requested, 2, std::max(2, maxSequenceLength));
   ```

   The default reproduces the old hard gate exactly; policy now lives in one
   place that the Model.cpp call sites share.
3. `Qwen4ExpMtpRuntime.hpp:22,27` - `EffectiveCapacity` and `Capacity()`
   accessors.
4. `Qwen4ExpMtpRuntime.cpp:65` - `AfterForward` window early-out:

   ```cpp
   if (start < 0 || int64_t(start) + int64_t(count) > int64_t(capacity)) {
       state.valid = false;
       state.length = start + int(count);
       return;
   }
   ```

   Placed before the `hiddenRows` staging copy: that copy writes `count` rows
   into a `capacity`-row buffer, and `draft->Step` later writes position
   `start + count - 1`, so both must be bounded before any work is issued.
   `Qwen4ExpMtpDraft::Step` keeps its own `position < capacity` `Require(...)`
   as the second line of defense (it throws before touching KV).
5. `Qwen4ExpMtpDraft.cpp:154` - draft state gate relaxed to
   `capacity <= target_.max_seq_len()` with the QSA-budget rationale in a
   comment.

### 3.2 Required Model.cpp patch (owner of that file; not applied by E4)

```diff
@@ Model.cpp:221 (CacheBytes)
-        return cache->CacheBytes() + (model->mtp ? Qwen4ExpMtpRuntime::SessionBytes(Capacity()) : 0);
+        return cache->CacheBytes() + (model->mtp ? Qwen4ExpMtpRuntime::SessionBytes(model->mtp->Capacity()) : 0);

@@ Model.cpp:229-230 (window admission)
             if (!model->mtp || !mtpState->valid || mtpState->length != cache->CurrentLength() ||
-                cache->CurrentLength() < 1 || cache->CurrentLength() + 2 > Capacity()) return std::nullopt;
+                cache->CurrentLength() < 1 || cache->CurrentLength() + 2 > model->mtp->Capacity()) return std::nullopt;

@@ Model.cpp:291 (ForwardPrefill)
-        const int remaining = model->mtp ? -1 : prefillRemaining;
+        // MTP phase coupling only for chunks that stay inside the draft window;
+        // longer chunks keep the production T1 prefill hints byte-for-byte.
+        const bool mtpChunk = model->mtp &&
+            cache->CurrentLength() + int(tokenIds.size()) <= model->mtp->Capacity();
+        const int remaining = mtpChunk ? -1 : prefillRemaining;

@@ Model.cpp:800-806 (load gate + reservation)
     enableMtp = enableMtp && options.maxSequenceLength >= 2 &&
         aila::env::read_int("AILA_Q4E_GDN_CHUNK_MODE", 0) == 0 &&
         !aila::env::read_flag("AILA_Q4E_LM_HEAD_Q8", false);
     if (enableMtp) {
         reservation.deviceBytes = aila::memory::checked_memory_sum(reservation.deviceBytes,
-            Qwen4ExpMtpRuntime::SessionBytes(options.maxSequenceLength));
+            Qwen4ExpMtpRuntime::SessionBytes(
+                Qwen4ExpMtpRuntime::EffectiveCapacity(options.maxSequenceLength)));
     }

@@ Model.cpp:836 (construction)
         if (enableMtp) loaded->mtp = std::make_unique<Qwen4ExpMtpRuntime>(context,
-            static_cast<Qwen4ExpBnb4Backend&>(*loaded->backend), *loaded->weights, options.maxSequenceLength);
+            static_cast<Qwen4ExpBnb4Backend&>(*loaded->backend), *loaded->weights,
+            Qwen4ExpMtpRuntime::EffectiveCapacity(options.maxSequenceLength));
```

Notes: the `<= 2048` in the `enableMtp` expression is deleted (L3); L4/L5 stay.
`tests/inference/Qwen4ExpSessionTests.cpp:35` hard-codes `SessionBytes(128)` and
uses `maxSequenceLength = 128`, where `EffectiveCapacity(128) = 128`, so that
assertion is unaffected. S3 (hunk 3) is mandatory for the "TTFT not worse"
judge: without it every long prefill chunk loses the `prefill_remaining` /
`next_prefill_tokens` hints that the 66 s production TTFT was measured with.

## 4. Memory and cost at production scale

Fixed runtime pieces (one per loaded model, independent of the window):

| Piece | Bytes | Source |
|---|---|---|
| checkpoint (linear states + PLE conv) | 118,038,528 (112.6 MiB) | `Qwen4ExpMtpRuntime.cpp:17`, matches session host reservation 117,669,888 |
| linearInputs `[48,4,16480]` bf16 | 6,328,320 (6.0 MiB) | `Qwen4ExpMtpRuntime.cpp:18` |
| draft resident weights (nf4 fc/qkv/o/shared + bf16 hc/indexer/router) | ~80.5 MB (76.8 MiB) | sum of non-expert `mtp.*` safetensors bytes; experts stay host-side (1,238 MiB in mmap) |
| draft expert cache 64 x 2,534,400 | 162.2 MB (154.7 MiB) | `Qwen4ExpMtpDraft.cpp:14-15` |
| plePrefix/ids/greedy/logits row | < 1 MB | |

Window-scaled pieces:

| Window T | shared `hiddenRows` = T*20,480 B | per-session `SessionBytes(T)` | shared MTP subtotal |
|---|---|---|---|
| 2048 (default; old gate) | 41.9 MB (40.0 MiB) | 4,870,144 B (4.6 MiB) | ~409 MB |
| 14000 (production prompt) | 286.7 MB (273.4 MiB) | 33,172,480 B (31.6 MiB) | ~654 MB |
| 32768 (production capacity) | 671.1 MB (640.0 MiB) | 77,615,104 B (74.0 MiB) | ~1,038 MB |

Per-session bytes are exact: `2304 B/token` (k 1024 + v 1024 + indexer_k 256,
bf16) `+ 256 B per 4-token indexer block + 20,480 B lastHidden`. The 2048 row
reproduces the recorded `mtp_extra_session_device_bytes: 4,870,144` in
`docs/plans/qwen38-server-mtp-2026-10-07.json`. Even the 32768 row is small next
to the production ledger (static weights 4208.6 MB + expert cache 9035 MB +
session 129.6 MB @max_seq 640); the default 2048 window costs *nothing* extra
versus today's MTP runs because the runtime allocation is identical.

Costs that scale with the window:

1. Prompt catch-up: one draft `Step(..., kvOnly=true)` per prompt token caught
   up inside the window (`Qwen4ExpMtpRuntime.cpp:71`). Anchor: the shape-real
   acceptance probe measured 3 catch-up rows in ~3.7 ms (~1.2 ms/row,
   `[MTP CYCLES]` in `tmp/q4e-sprint-r1-20261014/mtp-live-w2-g1.log`,
   probe code `tests/models/Qwen4ExpMtpAcceptance.hpp:165-166`); real-draft
   estimate 0.3-1.3 ms/row -> 2048: +0.6-2.7 s TTFT, 14000: +4.2-18.2 s TTFT.
   With the default window the production first chunk is 4096 tokens > 2048, so
   the early-out fires on chunk 1 and the catch-up cost is **exactly zero** for
   the 14K/6K production sessions.
2. Draft dense attention per full draft step: `24 heads * P * 256 * 2 B` for K
   plus V -> 0.05 GB at P=2048, 0.34 GB at P=14000 (~+0.1 ms vs +0.77 ms at
   ~450 GB/s). Measured draft steps are 5.5-8.5 ms at short context
   (`[MTP LIVE] draft_ms`), so +0.6 ms at 14K is tolerable; it is the reason
   for the original gate.
3. Verify window and capture/restore are unaffected by the window: capture is
   two fixed 118 MB device copies (`[MTP STATE] capture_ms ~1.13-1.31`,
   `restore_ms ~1.10-1.45`), and the target's own sparse QSA attention stays
   inside its 2048-token budget at any context length.

Conclusion: memory is not the blocker at any window up to 32768. The blocker
for windows >= 14000 is catch-up time (+4-18 s TTFT) unless catch-up is
batched (multi-row draft step) or made sparse; both are follow-ups, not part of
this minimal change.

## 5. Verification, A/B/B/A, criteria

Discipline: before every arm `Get-Process AilaWorker,AilaServer -ErrorAction
SilentlyContinue` must be empty, and arms are spaced >= 15 s so the large-page
pool drains. Both runners strip inherited AILA_/SYCL_/UR_L0_ variables, and
`run.py` applies trailing KEY=VALUE arguments after its own MTP=0 default, so
the arms below override it.

Long-session fallback (14K prompt, production runner, `--prime` also sends the
6000-token request):

```powershell
python tmp/q4e-sprint-r1-20261014/run.py e4-fb-a1 --prime --diag AILA_Q4E_SERVER_MTP=0
python tmp/q4e-sprint-r1-20261014/run.py e4-fb-b1 --prime --diag AILA_Q4E_SERVER_MTP=1 AILA_Q4E_MTP_MAX_CONTEXT=2048
python tmp/q4e-sprint-r1-20261014/run.py e4-fb-b2 --prime --diag AILA_Q4E_SERVER_MTP=1 AILA_Q4E_MTP_MAX_CONTEXT=2048
python tmp/q4e-sprint-r1-20261014/run.py e4-fb-a2 --prime --diag AILA_Q4E_SERVER_MTP=0
```

Short-session enablement (<= 2048 prompt, the only shape MTP may serve by
default), 2 x A/B/B/A via the MTP runner plus the E1 Top-1 comparator:

```powershell
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py e4-short-a1 AILA_Q4E_SERVER_MTP=0
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py e4-short-b1 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_MTP_MAX_CONTEXT=2048
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py e4-short-b2 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_MTP_MAX_CONTEXT=2048
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py e4-short-a2 AILA_Q4E_SERVER_MTP=0
# then the paste-ready Top-1 / window-sweep procedure from
# docs/plans/qwen38-r1-mtp-top1-2026-10-15.md section 5
```

Judges:

* J1 correctness (long): the 14K response `content_sha256` is identical in
  A1/B1/B2/A2 (hash identity implies Top-1 agreement 1.0). A differing hash is
  a fallback leak: re-run once, and if it persists score the pair with the E1
  teacher-forced Top-1 comparator and require >= 0.95, otherwise NO-GO.
* J2 no regression (long): TTFT (`first_content_seconds`) and decode tok/s of
  both B arms stay inside the A/A spread (target <= 1.5% delta); a B arm outside
  that band is re-run once before judging.
* J3 mechanism: worker log contains `[Q4E Server MTP] enabled=1 ...
  capacity=2048` at load and `[Q4E Server MTP] windows=0 accepted=0` at
  teardown for the long pair -> silent fallback happened, nothing was drafted.
* J4 enablement (short): `windows > 0` and the sprint criterion - decode
  >= 20 tok/s or e2e >= +5% versus T1; otherwise MTP stays opt-in.
* J5 quality (short): teacher-forced Top-1 agreement >= 0.95 against the T1
  stream on two prompt lengths (the >= 0.95 claim needs roughly 280 scored
  positions for a one-sided 95% bound).
* Default gate: `AILA_Q4E_MTP_MAX_CONTEXT` stays 2048 by default; any raise
  requires its own A/B/B/A with the same J1-J4 judges, plus a catch-up probe
  (J2 would otherwise fail on TTFT).

## 6. Risks, NO-GO conditions

* Do not apply the load-gate hunk without S3 hunk 3: long sessions would lose
  the prefill hints they were measured with, and (through expert-residency /
  hotset history) could even change numbers.
* `AfterForward`'s early-out must stay before the `hiddenRows` copy; moving it
  after the copy is a device-buffer overflow for chunks larger than the window.
* Windows at `CurrentLength() = Capacity()-1` are rejected (same conservative
  `len + 2 > C` rule as today); no behavior change, one wasted window slot.
* Windows >= 14000: NO-GO until catch-up is batched or sparse - estimate
  +4-18 s TTFT against a 66 s baseline.
* MTP remains not bitwise-identical to T1 by design (acceptance 74.8%,
  `bitwise_t1_guarantee=0` in the load log); quality is judged by the E1 Top-1
  gate, per the user's relaxed criterion.
* `EffectiveCapacity` clamps to `maxSequenceLength`, so `AILA_Q4E_MTP_MAX_CONTEXT`
  can never exceed the session capacity; a value below 2 clamps to 2, and the
  off switch stays `AILA_Q4E_SERVER_MTP=0`.

## 7. Files

* Applied, uncommitted: `src/models/Qwen4ExpMtpRuntime.cpp`,
  `src/models/Qwen4ExpMtpRuntime.hpp`, `src/models/Qwen4ExpMtpDraft.cpp`.
* Required follow-up by the source owner: the 5-hunk `Model.cpp` patch in 3.2.
* Not touched: `Qwen4ExpBnb4Backend.*`, `Qwen4ExpExpertExecutor.*` (owned by
  other agents this sprint).
* Rebuild before any GPU arm; no build was run here.
