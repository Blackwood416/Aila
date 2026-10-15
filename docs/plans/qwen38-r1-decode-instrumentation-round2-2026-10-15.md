# R1 decode instrumentation round 2 (C2): naming the misc_dev bucket

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, HEAD `fe1bf9e`, pair
`retained25`. Scope: `src/models/Qwen4ExpBnb4Backend.cpp` only (diagnostics;
no execution semantics, no default-path change). Zero GPU, zero build, zero
commit by this agent. All line numbers are post-edit. ASCII only.

## 1. Current state and goal (why misc_dev needed a name)

B6 round 1 (`docs/plans/qwen38-r1-decode-instrumentation-2026-10-15.md`)
added the gated queue-span markers. In the measured arm
(`tmp/q4e-sprint-r1-20261014/dec-instr-b2-worker.log`, n=128 count=1 tokens,
parse via `tmp/q4e-sprint-r1-20261014/dec_instr_dev_parse.py`) the dev spans
sum to 71.223 ms/token median:

    experts 27.489 | misc 11.762 | hc 9.696 | linear 7.743 | full 5.370
    shared 3.541 | head 3.375 | router 1.936 | resid 0.311

`misc_dev` was 16.5% of all device time and had no name. The readback rule is
`bucket(i) += start(i) - end(i-1)` (Backend.cpp round-1 code, now
`:2597-2604`), so with the old marker set (pre-loop anchor + one marker after
PLE and one before the router gate, both tagged `kDevMisc`) it could only be
the sum of exactly two windows per layer:

    W1: previous layer resid-marker end -> PLE-marker start (48x/token)
    W2: MLP-HC-marker end -> pre-router-marker start (48x/token)

Diagnostic evidence from the same log (all medians unless noted):

- `corr(misc_dev_ms, ple_host_ms) = 0.998` over 128 tokens; both are bimodal
  with the same token pattern (`ple_host_ms` quintiles 1.67 / 1.86 / 7.93 /
  8.56 / 9.74 ms) -> W1 tracks the `kStagePle` host window, which contains the
  L==1 PLE block (fresh-prepare vs ticket-reuse modes).
- `misc_dev_ms - ple_host_ms`: median 3.98, mean 3.99, min 2.66, max 4.53,
  n=128, zero negative values -> W2 is an approximately constant ~4 ms/token.
- `misc_dev_ms` itself: median 11.762, mean 10.104, min 5.405, max 14.065
  (5.6 in the low `ple_host` mode, ~12.6-13.4 in the high mode).

Goal: give every window a named bucket, keep the span partition exact, and
keep `misc_dev_ms` in the line as a must-print-zero self-check. No new
synchronization: the two new markers are in-order queue commands (same API as
round 1, `ext_oneapi_submit_barrier`) and the readback is still the single
end-of-forward wait.

## 2. New sub-buckets and marker sites (post-edit line numbers)

Marker -> credited window (start attributed to the marker's bucket):

| field | marker | credited window | submitted / waited inside |
|---|---|---|---|
| `inter_dev_ms` | `:2102` | prev-layer resid marker (`:2462`) end -> this | loop header, dump check; barrier dispatch. For L=0 the pre-loop anchor `:2093` -> `:2102`. |
| `ple_dev_ms` | `:2122` | `:2102` -> this | `pending_mlp.Materialize` + `ple_.prepare/consume/commit` at L==1 (Qwen4ExpPle.cpp); ~empty on the other 47 layers. |
| `stagein_dev_ms` | `:2200` | MLP-HC marker (`:2173`) end -> this | lease/branch/env checks + `StageResidentInput` D2H memcpy on `ctx_->queue()` + the previous input copy's `wait_and_throw` (`src/models/Qwen4ExpExpertExecutor.cpp:723-730`). |
| `prerouter_dev_ms` | `:2209` | `:2200` -> this | `gpu_nf4_q8` / `moe_front_fused` flag expressions only (`:2201-2208`). |
| `misc_dev_ms` | no marker | - | kept for backward compat; no marker credits it, so it must print `0.000`. |

Changes vs round 1: `:2110` re-bucketed `kDevMisc -> kDevPle`; `:2194`
re-bucketed `kDevMisc -> kDevPreRouter`; new markers at `:2102` (`kDevInter`)
and `:2200` (`kDevStageIn`). The old W1/W2 spans are split exactly at the new
markers, so `inter + ple + stagein + prerouter` is a partition of the old
`misc`. Enum and print append-only (`:2083-2085`, fields at `:2625-2626`,
args at `:2642-2644`); the parser form is unchanged key=value.

Marker count per decode token: 434 -> 530 (11/layer + anchor + head, +96).

Candidates from the brief that are NOT misc (verified, no new marker needed):

- MoE-front device span (gate GEMV / top-k / classify): already
  `router_dev_ms` (marker `:2257`; window `:2209` -> `:2257`).
- ordered accumulate / resident decode: inside `experts_dev_ms` (marker
  `:2429`; window covers `RunResidentDecode` submissions + route-event
  bubble).
- token H2D / prepare and sampling / head tail: these run OUTSIDE this
  translation unit after the last marker (kDevHead `:2532`) and are already
  accounted by the caller span `[Q4E ForwardWall]`
  prepare_ms/backend_ms/sample_ms/drain_ms (`src/inference/Model.cpp:380-396`;
  sampling + token D2H at `:530-568`). They cannot be captured by queue-span
  markers inside `ForwardImpl` (TODO section 6).

## 3. Coordinator validation arm

GPU mutex first, then build (this agent did neither):

    Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue  # must be empty
    pwsh -NoProfile -File build.ps1 -BuildDir build-refactor

    python tmp/q4e-sprint-r1-20261014/run.py c2-round2-b1 --prime --diag --AILA_Q4E_DECODE_STAGE_EVENTS=1
    python tmp/q4e-sprint-r1-20261014/run.py c2-round2-b2 --prime --diag --AILA_Q4E_DECODE_STAGE_EVENTS=1

`--diag` sets `AILA_Q4E_WORK_SUMMARY=1` (`run.py:37-39`); the `--KEY=VALUE`
form is honored (`run.py:55-62`). Requirements unchanged from round 1: the
queue must be profiling-enabled (worker path needs the
`AILA_Q4E_DECODE_STAGE_EVENTS` enabler in `src/inference/Model.cpp:40`,
otherwise the one-time warning prints and every `dev_*` is 0); do NOT combine
with `AILA_PROFILE_Q4E_STAGES=1`; this is an attribution arm, never a rate
receipt (530 barrier commands/token).

Fields to read on every count=1 `[Q4E AdaptiveWork]` line (>=100 consecutive
tokens):
`inter_dev_ms ple_dev_ms stagein_dev_ms prerouter_dev_ms misc_dev_ms dev_events`
plus the four `*_host` fields that share the windows
(`ple_host_ms router_wait_ms experts_host_ms hc_host_ms`) for cross-checks.

Pass criteria:

1. `misc_dev_ms=0.000` on every line (non-zero = a marker lost its bucket).
2. Conservation: median of `inter + ple + stagein + prerouter` reproduces the
   round-1 `misc` reference: |sum - 11.762| <= 0.5 ms/token (median basis).
3. A/A: |sum(b1) - sum(b2)| <= 0.5 ms/token.
4. `dev_events=1` and no profiling-off warning.

## 4. Overhead and risk

- Gate off (default): unchanged. `dev_mark` stays one bool check per site; no
  queue op, no wait, empty vector. Summary-off implies `dev_trace` false
  (`:2063-2064`).
- Gate on: +2 barrier commands/layer (434 -> 530 per token) plus 2x2
  `get_profiling_info` reads per layer; still exactly one `wait` at the end.
  Measured cost of round 1's gate is bounded in the B6 doc (A2 vs B2).
- Risk 1: the extra barriers add dispatch time inside the very gaps being
  measured; sub-0.5 ms buckets (`inter`, `prerouter`) should be read against
  the A/A spread, not absolutely.
- Risk 2: `stagein_dev_ms` semantics depend on `StageResidentInput` running
  on the main queue; that is asserted by
  `Qwen4ExpExpertExecutor.cpp:729`. With `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1`
  the host wait at `:728` disappears but the window remains valid.
- Risk 3 (alternative outcome): if the arm instead shows a large `inter_dev`
  and ~0 `stagein_dev`, the constant ~4 ms offset lives in the boundary gap
  (loop header / dump path), not in the handoff. The arm is what decides;
  both outcomes are actionable (section 5).
- Exceptions mid-forward discard the marker vector unread (same as round 1).
- NOT verified by this agent: no build/GPU. First compiler contact is the
  coordinator's build.

## 5. Predicted misc split (conservation vs 11.762 ms/token median)

Basis: (a) structural - `inter` is loop-header-only and `prerouter` is two
bool expressions, both near the barrier dispatch floor; (b) the correlation
0.998 and the constant offset 3.98 ms (n=128) pin the bulk to `ple` +
`stagein`.

| bucket | predicted median | range | main driver |
|---|---:|---:|---|
| `inter_dev_ms` | 0.4 | 0.2 - 1.0 | 48x loop header + dispatch |
| `ple_dev_ms` | 7.1 | 6.4 - 7.7 | L==1 PLE block; bimodal (low ~1.3, high ~7.7) |
| `stagein_dev_ms` | 3.8 | 3.2 - 4.4 | constant-offset mass = pre-router handoff block |
| `prerouter_dev_ms` | 0.4 | 0.1 - 0.9 | flag tail + dispatch |
| sum | 11.7 | must satisfy 11.762 +/- 0.5 | conservation |

Mode cross-check (predicts the bimodality lives entirely in `ple`):

- low mode (`ple_host` ~1.7, observed `misc` 5.6-6.0): 0.3 + 1.4 + 3.6 + 0.4
  = ~5.7.
- high mode (`ple_host` ~8.5, observed `misc` 12.6-13.4): 0.5 + 7.7 + 4.1 +
  0.4 = ~12.7.

If `stagein_dev` confirms ~4 ms, the follow-up lever is the resident-input
handoff itself: a half-arm with `AILA_Q4E_CPU_DEFER_ARENA_WAIT=1` (removes the
prior-copy wait at `Qwen4ExpExpertExecutor.cpp:728`) or hoisting
`StageResidentInput` earlier in the layer - measure with the same fields
instead of guessing.

## 6. API gaps / TODO (cannot be expressed with current markers)

- Sampling / token H2D / next-token prepare happen outside `ForwardImpl`
  after the last marker; they already have a caller-side span
  (`[Q4E ForwardWall]`), so nothing is missing, but queue-span buckets here
  will never include them by construction.
- `Qwen4ExpLayerMajor.cpp:477` prints its own shorter `[Q4E AdaptiveWork]`
  line without dev fields; porting the buckets there is out of scope for
  this round.

## 7. Sources

- `src/models/Qwen4ExpBnb4Backend.cpp:2077-2085` (enum + rationale),
  `:2093` (anchor), `:2102/:2122/:2200/:2209` (new markers),
  `:2625-2626` + `:2642-2644` (print fields/args, 43 conversions = 43 args).
- `src/models/Qwen4ExpExpertExecutor.cpp:723-730` (`StageResidentInput` D2H
  on `ctx_->queue()`).
- `src/inference/Model.cpp:380-396` (`[Q4E ForwardWall]`), `:530-568`
  (sampling / token D2H).
- `tmp/q4e-sprint-r1-20261014/dec-instr-b2-worker.log` +
  `dec_instr_dev_parse.py` (n=128 medians used above).
- `docs/plans/qwen38-r1-decode-instrumentation-2026-10-15.md` (round 1:
  attribution rule, arms, discipline).
