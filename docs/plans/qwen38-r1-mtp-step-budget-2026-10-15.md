# R1-G4: MTP step budget and the road to 20 tok/s

Date: 2026-10-15. HEAD `df1f45a`. Scope: Qwen4Exp decode, MTP permitted.
Method: document only - no build, no GPU run, no commit. Every number below
carries its source; numbers marked `[est]` are estimates.

Task brief inputs (treated as measured, cross-checked against the receipts):
- T1 (2048 ctx, prompt 1500, generate 128): 17.33/17.12 tok/s.
- MTP width-2 + `AILA_Q4E_VERIFY_CPU_COLD=1`: 15.54/15.59 tok/s, acceptance
  113/151 = 74.8%.
- verify window (count=2) submitted 92.26 ms (router_wait 12.64,
  experts_host 53.97 = cpu_compute 29.47 + cpu_join 24.84, staged/h2d = 0).
- Break-even: W_total <= (1+p) * W_T1 ~= 100.9 ms; 20 tok/s needs <= 87 ms;
  current W_total ~= 112.5 ms.

## 0. Notation - brief label vs code semantics (read this first)

`width` in code and logs = number of rows in the verify forward
(`[Q4E Server MTP] ... width=2` is emitted by the runtime itself,
`src/models/Qwen4ExpMtpRuntime.cpp:27`):

- width 2 (production) = 2 rows = pending token + 1 draft
  (`proposal{pending, Select(1)[0].index}`, `MtpRuntime.cpp:100`;
  `ForwardImpl(ids, 2, decode_verification=true)`, `:108-110`).
  Accept iff target row-0 argmax == draft (`:115`); emitted tokens per window
  = 1 + accept.
  Measured, `g1-b1`/`g1-b2` (the destructor counters cover the whole run,
  including warmup): 151 windows, 113 accepted, p = 113/151 = 0.748;
  window-emitted tokens = windows + accepted = 264 vs 266 generated
  (2-token request-boundary slack). Continue request alone: 74 windows,
  128 tokens, rate 15.543/15.592 tok/s
  (`tmp/q4e-resident-speed-20261009/g1-b1-worker.log`,
  `g1-{b1,b2}/result.json`).

- The task brief says "width=3 (3 rows verify), tokens/step = 1+3p". Those two
  cannot both hold under code semantics: 3 rows = 2 drafts, so
  tokens/step = 1 + p + p2 (chain). The formula 1+3p corresponds to
  3 *drafts* = 4 rows - which is also what the brief's "(4 rows CPU batch)"
  phrase in task 2b implies. This document therefore reports both:
  a "code width 3" (3 rows / 2 drafts) and a "brief width 4" (4 rows / 3
  drafts, the 1+3p model). Both are gated the same way.

## 1. Where the ~20 ms per-window draft step goes

### 1.1 Measured residual

Per-window total today = (1+p)/rate = 1.748 / 15.543 = 112.5 ms (g1-b1
continue request). Verify (count=2) submitted median = 92.57 ms
(g1-b1 req3, n=74; g1-b2 req3 92.59, n=74; b1 req2 92.02). Residual =
112.5 - 92.6 = **19.9 ms/window** (brief says 20.4; same number within
accounting of the first token). This is the "draft step" line item.

### 1.2 Code inventory of one width-2 window (all of it except verify)

From `src/models/Qwen4ExpMtpRuntime.cpp:94-142`:

| # | Step | Code line | Content |
|---|---|---|---|
| 1 | `draft->Step(...)` full | :99 | full MTP layer forward (see 1.3) |
| 2 | `Select(1)` | :100 | GPU greedy over vocab + blocking D2H (1 row) |
| 3 | `Capture(s)` | :101 | per-linear-layer (linear_state+conv_state) + PLE conv = 117,669,888 + 368,640 B = 118.0 MB D2D |
| 4 | ids H2D | :102 | 8 bytes, blocking |
| 5 | verify | :108-110 | `submitted_ms` = 92.3-92.6 (payload below) |
| 6 | hiddenRows copy | :112 | 2 x 10240 bf16 = 40 KB D2D |
| 7 | `Select(2)` | :113 | GPU greedy 2 rows + blocking D2H |
| 8a | accept: `draft->Step(kvOnly=true)` | :137 | embedding + 2 FC + qkv + KV/indexer append, no attention/MoE/head |
| 8b | reject (25.2% today): plePrefix copy 40 KB, `Restore()` 118.0 MB D2D, per-linear-layer GDN replay `run_linear_delta_prefill_gpu_batched` x1 row, PLE ring writeback, ngram recompute | :117-135 | |
| 9 | lastHidden copy | :139 | 20 KB D2D |
| 10 | `ctx.synchronize()` | :142 | full queue drain per window |

### 1.3 Does the draft repeat a full 1-layer forward?

Yes - the draft step is a complete MTP transformer layer *plus the shared LM
head*. `Qwen4ExpMtpDraft::Step` (`src/models/Qwen4ExpMtpDraft.cpp:162-242`)
submits, in order: embed H2D (:173) + embedding RMS + `fcEmbedding` NF4 GEMM;
10240-wide RMS + `fcHidden` NF4 GEMM with 4-row streams (:180, rows=4 reuse
path `Bnb4BitLinear.cpp:2357`); attn HC (:186); fused QKV (:187);
`decode_prepare_qgkv` KV append (:192); indexer GEMV + QSA compress (:198,
:201); `attention_decode` over its own KV (:208); o_proj (:212); mlp HC
(:213); router GEMV + topk (:217); shared expert gate/up/down; 10-expert MoE
(:232, :234); reader fences (:238); inject (:239); mixer HC (:241);
**`lm_head_.forward(ctx_, b.x, b.logits, 1)` (:242)**.

- Hidden reuse: already implemented. The accept path feeds the verify's own
  hidden row back into the next Step (`:137`, `hiddenRows` row 0), and the
  4-stream FC uses `AILA_Q4E_MTP_HIDDEN_REUSE=1`
  (`Bnb4BitLinear.cpp:2356-2364`).
- KV reuse: not possible by design. The draft owns a separate KV/indexer
  ("scheme A... never KV state", `Qwen4ExpMtpDraft.hpp` header) and its own
  layer-0 attention weights; the verify KV cannot serve the draft attention.
  Only the hidden can be, and it is.
- The expensive irreducibles inside the Step: the shared LM head reads
  ~1271 MB per row (vocab x 2560 bf16) - E5 measured the T1 head at 2.7 ms
  clean / 4.37 ms at the 291 GB/s bound
  (`docs/plans/qwen38-r1-decode-misc-2026-10-15.md`); the draft MoE moves
  10 x 2.417 MiB = 24.2 MiB of expert weights (staging misses observed
  431 misses / 1092 MB per 128 tokens in the 10-06 epoch,
  `docs/plans/qwen38-mtp-window-cache-8724-2026-10-06.md`).

### 1.4 Filling the 19.9 ms (measured anchors + remainder)

Last measured per-part decomposition, 10-06 epoch, ctx ~1960, same code
family (`qwen38-mtp-projection-reuse-2026-10-06.md`,
`qwen38-mtp-window-cache-8724-2026-10-06.md`): draft 7.58-8.23 ms,
checkpoint+input 1.30-1.48, commit/replay 0.65-0.89 (warm window, advance
1.848). Selecting those numbers as a prior and reconciling with today's
residual:

| Item | ms/window | Basis |
|---|---:|---|
| draft full Step | 7.5-8.2 | 10-06 measured (same Step code) |
| Capture + ids H2D | 1.3-1.5 | 10-06 measured |
| Select(1) + Select(2) + accept D2H | 2-4 `[est]` | T1 sample is 3.42 ms for 1 select + IPC (`d1-smoke-worker.log`); 2 blocking selects inline |
| accept commit (kvOnly Step) | 0.7-0.9 | 10-06 measured |
| reject path averaged over 25.2% | 1.3-3.8 `[est]` | reject = Restore 118 MB D2D + per-linear-layer GDN replay + PLE + ngram; unit cost unknown (5-15 ms `[est]`) |
| per-window sync + scheduler/SSE | 1-3 `[est]` | `ctx.synchronize()` at :142 is outside `submitted_ms` |
| Sum | 15.3-22.7 | vs 19.9 measured |

The exact split is *measurable with existing test code* - see arm R1 in
section 3.4 (`[MTP LIVE]` already prints per-cycle `draft_ms`, `verify_ms`,
`replay_ms`, and with `AILA_Q4E_MTP_SELECT_PROFILE=1` also
`checkpoint_input_ms`, `select_copy_ms`, `select_cpu_ms`,
`draft_select_copy_ms`, `draft_select_cpu_ms` -
`tests/models/Qwen4ExpMtpProbe.hpp:315,321-322,691-724`).

### 1.5 Reducible items and their arithmetic

| Item | Size | Can save | Notes / blockers |
|---|---:|---:|---|
| reject path (Restore + 24x GDN replay) | 5-15 `[est]`/reject, 1.3-3.8 avg | 1-3 `[est]` | cheaper replay (state-only variants exist in the probe); raising p is not actionable |
| draft LM head (shared, 1271 MB) | 2.7-4.4 | 1.3-1.5 | requires Q8 head; today MTP is *hard-gated* against it: constructor throws if `target.lmHeadQ8Enabled_` (`Qwen4ExpMtpRuntime.cpp:12`) |
| Select(1)+Select(2)+sync | 3-7 `[est]` | 1-2 | async D2H + one wait; fuse commit D2H |
| draft MoE staging misses | ~8.6 MB transfer /token @ 10-06 rate | 0.5-1 | 64-slot cache; 64->128 was measured and reverted (hotset damage, 10-06) |
| verify itself | 92.6 (82% of window) | up to 8.8 for T1 = same shave 1:1 | this is the T1 host-shave (P2); it is the only item with 5-15 ms potential |

No draft-side item alone yields 5-15 ms. Draft-side realistic total: -3 to -6
ms `[est]`; the path to 87.4 ms must come mostly from the verify, i.e. from
the same machinery as the T1 shave.

## 2. Width arithmetic (width 3 and width 4) and the verdict

### 2.1 Tokens/step and required W (rate = T / W, 20 tok/s <=> W <= 50*T)

Three acceptance models are shown because the brief's 1+3p is the optimistic
linear model; the code chain model and the one conditional-acceptance
measurement (q2 ~= 0.63, 24/38 in
`docs/plans/qwen3.8-flash-next-p4-mtp-results-2026-09-30.md` per E2) are
tighter:

| p | width 2 (1 draft): 1+p | width 3 (2 drafts) | | | width 4 (3 drafts) | | |
|---:|---:|---:|---:|---:|---:|---:|---:|
| | | linear 1+2p | chain 1+p+p^2 | cond 1+p+p*q2 | linear 1+3p | chain 1+p+p^2+p^3 | cond q2=q3=.63 |
| 0.60 | 1.600 | 2.200 | 1.960 | 1.978 | 2.800 | 2.176 | 2.240 |
| 0.748 | 1.748 | 2.496 | 2.307 | 2.219 | 3.244 | 2.725 | 2.516 |
| 0.85 | 1.850 | 2.700 | 2.573 | 2.386 | 3.550 | 3.186 | 2.800 |

W_max = 50 x T at 20 tok/s (ms):

| p | width 2 | width 3 (lin / chain / cond) | width 4 (lin / chain / cond) |
|---:|---:|---:|---:|
| 0.60 | 80.0 | 110.0 / 98.0 / 98.9 | 140.0 / 108.8 / 112.0 |
| 0.748 | 87.4 | 124.8 / 115.4 / 111.0 | 162.2 / 136.3 / 125.8 |
| 0.85 | 92.5 | 135.0 / 128.6 / 119.3 | 177.5 / 159.3 / 140.0 |

Width-2 sanity: today W = 112.5 -> rate = 1.748/0.1125 = 15.5 tok/s
(measured 15.54/15.59). T1 break-even W = (1+p)*57.7 = 100.9 (brief).

### 2.2 W(3) and W(4) estimate from measured pieces

Marginal verify row on the same build: T1 count=1 (d1-smoke, submitted 55.0,
cpu_compute 13.25, cpu_join 9.08, experts_host 20.65) vs width-2 count=2
(submitted 92.0, cpu_compute 29.7, cpu_join 25.4, experts_host 54.0) ->
**+37.0 ms on submitted, of which +16.4 cpu_compute, +16.0 cpu_join,
~+4.5 non-expert** (`d1-smoke-worker.log`,
`tmp/q4e-resident-speed-20261009/g1-b2-worker.log`). The per-row CPU cold
batch is serialized per row (G1 risk note,
`docs/plans/qwen38-r1-mtp-verify-cpu-2026-10-15.md` sec 4), so linear
row scaling is the right first-order model; the 4-token CPU batch API
(`cpu_expert_batch_forward` accepts tokens<=4,
`Qwen4ExpCpuExpertOps.cpp:1581`; >4 falls back scalar `:1372/1475/1514`,
E9b) is *not* wired for verify rows today.

Non-verify rest: 19.9 measured at width 2; each extra draft adds one full
Step + one Select + bookkeeping = +8.5-9.7 ms `[est]`.

| Config | verify `[est]` | rest `[est]` | W `[est]` | rate at p=.748 (chain / cond) |
|---|---:|---:|---:|---|
| width 2 | 92.6 (measured) | 19.9 (measured) | 112.5 (measured) | 15.5 (measured) |
| width 3 (2 drafts) | 125-133 | 29-31 | **155-165** | 2.307/0.160=14.4 / 2.219/0.160=13.9 |
| width 4 (3 drafts) | 158-170 | 38-43 | **196-210** | 2.725/0.203=13.4 / 2.516/0.203=12.4 |

### 2.3 Verdicts

(a) Can width 3 reach >= 20 tok/s? **No, under every p from 0.60 to 0.85.**
Required W(3) = 111-135 ms (model-dependent); estimated 155-165 ms, i.e.
-20 to -50 ms short. Even the brief's optimistic linear model requires the
3-row window to cost <= 124.8 ms, and that budget is already below
today's 2-row verify alone (92.6) plus rest (30) = 122.6 - i.e. width 3
would need a *marginal third row of ~+2 ms* (linear) or a negative one
(chain/cond).

Width 4 under the brief's 1+3p model: budget 162.2 ms vs estimated
196-210 -> **No** as well; it would need the two extra rows at <= 14.8 ms
each (vs 37 observed) or a substantially cheaper CPU row pipeline.

(b) New bottleneck at width 3/4: **the CPU cold-expert path.**
cpu_compute 29.7 -> ~44/61 ms and cpu_join 25.4 -> ~38/50 ms for 3/4 rows;
together they are 60-70% of the window, and because each row is its own
serialized batch (G1), the join chain per (layer,row) is the critical path.
The second bottleneck is the per-extra-draft cost: one full MTP layer +
shared LM head read (2.7-4.4 ms) + MoE + selects = 8.5-9.7 ms per extra
draft. The lever that could plausibly change verdict (b) is the single
4-token CPU batch (weight fetch amortized across rows, cap-preserving
per-row semantics + >cap overflow per G1 sec 4.1) - it is a prediction,
not a measurement.

## 3. <=3 paths to 20+ tok/s

### 3.1 P1 - MTP width 2: shave W from 112.5 to <= 87.4 (p=.748; <= 92.5 at p=.85)

- Need: -25 ms (-22%). Composition: verify-side -12..-18 (the verify is the
  same 48-layer loop, so every T1 host shave lands 1:1 - E2 sec 2.5:
  router_wait 12.7 queue residual, residual family ~9, cpu_join 25.4) +
  draft-side -4..-7 (reject path 1-3, Q8 head 1.3-1.5 if unlocked for MTP,
  selects/sync 1-2).
- Effort M-L. Risk: MTP is not bit-equal to T1
  (`bitwise_t1_guarantee=0`, runtime log `:27`); any default flip needs a
  quality position first (E2 sec 3.1). T1-side changes must stay bit-exact.
- Minimal arm (after the probe arms in 3.4 confirm the split):

```powershell
Get-Process AilaWorker,AilaServer,Aila -ErrorAction SilentlyContinue   # must be empty
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py mtp-shave-a1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py mtp-shave-b1 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_WORK_SUMMARY=1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py mtp-shave-b2 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_WORK_SUMMARY=1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py mtp-shave-a2
```

  Pass: continue rate >= 20.0 tok/s, 2/2 pairs over the A/A floor
  (0.9-2.6%, E2 sec 1.1), count=2 submitted median <= 87.4
  (diagnostic: W_hat = (1+p_hat)/rate_b). Baseline today: g1-b1/b2
  15.543/15.592 tok/s, submitted 92.57/92.59.

### 3.2 P2 - T1 (no MTP): best bundle 54.77/55.46 ms -> <= 50 ms

- Current best *measured* bundle: e10-bundle (MOE_FRONT_FUSED=1 +
  AILA_KV_QUANT=int8 + agent6k hotset profile; note `MOE_FRONT_FUSED` and
  `KV_QUANT` are already in that arm's worker_environment,
  `tmp/q4e-sprint-r1-20261014/e10-bundle-b2/result.json`): ForwardWall
  median 54.772 (b2 55.461), submitted 51.006, 14K-continue 17.563 tok/s
  (`tmp/q4e-sprint-r1-20261014/e10_parse.py` output). The brief's 55.12 ms
  figure matches the same family.
- Need: -4.8 to -6.5 ms (-9 to -12%) to reach 50.0 ms wall / ~46.4 ms
  backend. Measured candidates: residual family ~9.0 (moe_front 1.8 +
  topk 0.9 + shared_submit 3.7 + router_other 1.9 + shared_add 0.6),
  cpu_join 9.1 inside experts_host 20.7 (submit/join pipelining), sample
  3.4 (fuse toward IPC). Bit-exact requirement on the SSE text.
- Warning: the F1 fused-HC arm measured *worse* in its only run
  (f1-b1: ForwardWall 67.19, 14.18 tok/s,
  `tmp/q4e-sprint-r1-20261014/f1-b1-worker.log`) - needs a clean A/B/B/A
  before any conclusion.
- Minimal arm (bit-exact gate):

```powershell
python tmp/q4e-sprint-r1-20261014/run.py p2-shave-a1 --prime --diag
python tmp/q4e-sprint-r1-20261014/run.py p2-shave-b1 --prime --diag <mechanism env>
python tmp/q4e-sprint-r1-20261014/run.py p2-shave-b2 --prime --diag <mechanism env>
python tmp/q4e-sprint-r1-20261014/run.py p2-shave-a2 --prime --diag
```

  Pass: continued rate >= 20.0, 2/2 pairs, content sha identical to the A
  arms.

### 3.3 P3 - width 3/4 (conditional, pre-registered NO-GO)

Gate (from 2.2/2.3): width 3 may be reconsidered only if a measured 3-row
verify comes in at <= ~95 ms (needs marginal row <= ~2 ms) under the linear
model, or the whole W(3) lands <= 111-115 ms. Width 4 only if 4-row verify
<= ~122 ms (marginal row <= ~15 ms). Both require the single-batch 4-token
CPU path plus a cheaper draft chain. Until that exists: **do not spend GPU
on width 3/4.** If it is ever measured, use the existing probe (3.4 R2).

### 3.4 Verification arms (existing code, no new src change required)

The MTP live probe already exposes the whole window split and widths
1/2/4 in one process (`tests/models/Qwen4ExpMtpProbe.hpp`,
`tests/models/Qwen4ExpMtpAcceptance.hpp`,
driver `AilaQwen4ExpToleranceTests.exe --mtp-model <dir>`;
per-cycle arrays `draft_ms`, `verify_ms`, `replay_ms`, `cycle_ms`,
`advance`, and with select profile `checkpoint_input_ms`, `select_*`).

R1 - width-2 attribution (settles section 1.4/1.5):

```powershell
Get-Process AilaWorker,AilaServer,Aila -ErrorAction SilentlyContinue   # must be empty
$env:AILA_Q4E_MTP_MAX_CONTEXT='2048'; $env:AILA_Q4E_MTP_PROMPT='512'
$env:AILA_Q4E_MTP_ANCHORS='124'; $env:AILA_Q4E_MTP_REPEATS='2'
$env:AILA_Q4E_MTP_LIVE_WIDTH='2'
$env:AILA_Q4E_MTP_SELECT_PROFILE='1'
$env:AILA_Q4E_MTP_GPU_GREEDY='1'; $env:AILA_Q4E_MTP_GREEDY_ROWS='1'
$env:AILA_Q4E_VERIFY_CPU_COLD='1'; $env:AILA_Q4E_CPU_MAX_EXPERTS='10'
$env:AILA_Q4E_CPU_THREADS='18'
build-refactor\AilaQwen4ExpToleranceTests.exe --mtp-model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 *> tmp/q4e-sprint-r1-20261014/mtp-step/live-w2.log
```

  Read: `[MTP LIVE]` medians of draft_ms / verify_ms / replay_ms /
  checkpoint_input_ms / select_* vs the 19.9 ms residual; check against
  section 1.4 priors. GPU mutex pre-check; no fixed sleeps; one arm only.

R2 - width sweep (marginal row + q2, settles section 2):

```powershell
$env:AILA_Q4E_MTP_LIVE_COMPARE_WIDTHS='1'   # runs widths 1, 2, 4 in-process
... same env as R1 (drop LIVE_WIDTH) ...
build-refactor\AilaQwen4ExpToleranceTests.exe --mtp-model C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2 *> tmp/q4e-sprint-r1-20261014/mtp-step/live-w124.log
```

  Read: verify_ms(4)-verify_ms(2) / 2 = marginal row (predict 33-40 ms);
  draft_ms(4)-draft_ms(2) / 2 = per-draft cost (predict 8.5-9.7 ms);
  `advance` array at width 4 gives the conditional acceptance q2 (predict
  ~0.63). Width 3 itself is not accepted by the probe
  (`Require(liveWidth==1||2||4)`, probe hpp :337) - interpolate or extend
  the Require only if 2.3 stays inside the gray zone (it is not, as of
  today's numbers).

R3 - after P2 lands: re-run P1's arm; verify inherits T1 shaves 1:1
(E2 sec 2.5).

## 4. Risks / open items

1. All W(3)/W(4) numbers are `[est]` built from a 1-row->2-row same-build
   delta; row scaling may be sub-linear if the single 4-token CPU batch
   lands (that is exactly the gate in 3.3). R2 settles it.
2. q2 (conditional acceptance) is only known from the 09-30 probe
   (24/38 ~= 0.63); R2 measures it on retained code.
3. Do not layer `AILA_Q4E_DECODE_STAGE_EVENTS=1` on top of any MTP probe:
   the observer effect is +11-13 ms per token
   (`docs/plans/qwen38-r1-decode-boundary-2026-10-15.md`).
4. MTP remains non-bit-equal to T1 and is gated (<=2048 ctx, GDN chunk 0,
   BF16 head - `src/inference/Model.cpp:726,800`); no default flips without
   a quality position.
5. The production runtime hardcodes width 2 (`proposal` array size,
   `Select(2)`, capacity guard `length+2 <= Capacity()` at
   `Model.cpp:231`). Width 3/4 would be a code change, not a config.
6. TODO(probe): R1, R2 (commands above). TODO(decision): close width>=3 as a
   20 tok/s path unless R2 shows marginal row <= 15 ms with the batched CPU
   path.

## 5. Sources

- `tmp/q4e-resident-speed-20261009/g1-b1-worker.log`, `g1-b2-worker.log`,
  `g1-b1/result.json`, `g1-b2/result.json` (windows=151 accepted=113;
  count=2 medians; 15.543/15.592 tok/s).
- `tmp/q4e-sprint-r1-20261014/d1-smoke-worker.log` (T1 55.0/count=1),
  `d1-a1b/a2/b1/b2-worker.log` (A/A), `e10-bundle*.log` + `e10_parse.py`
  (best bundle 54.772/55.461, 17.563 tok/s), `f1-b1-worker.log`.
- `docs/plans/qwen38-mtp-window-cache-8724-2026-10-06.md`,
  `qwen38-mtp-projection-reuse-2026-10-06.md` (draft 7.58-8.23,
  checkpoint+input 1.30-1.48, commit/replay 0.65-0.89, advance 1.848).
- `docs/plans/qwen38-r1-mtp-verify-cpu-2026-10-15.md` (G1; per-row CPU
  serialization, bit-parity argument, read rules),
  `docs/plans/qwen38-r1-decode-20-2026-10-15.md` (E2; T1 table,
  break-even formula, threshold tables),
  `docs/plans/qwen38-r1-decode-misc-2026-10-15.md` (E5; head 2.7 ms /
  1271 MB), `qwen38-r1-decode-boundary-2026-10-15.md` (observer effect),
  `qwen38-r1-cpu-moe-bench-2026-10-15.md` (E9b; tokens<=4 batch API).
- Code: `src/models/Qwen4ExpMtpRuntime.cpp:12,27,94-142`,
  `src/models/Qwen4ExpMtpDraft.cpp:162-242`,
  `src/models/Qwen4ExpBnb4Backend.cpp:1263,1324-1325,2417`,
  `src/inference/Model.cpp:231,726,800`,
  `tests/models/Qwen4ExpMtpProbe.hpp:315-337,691-724`,
  `tests/models/Qwen4ExpMtpAcceptance.hpp:236-241`,
  `tests/models/Qwen4ExpMtpAlign.hpp:10,141-146`.
