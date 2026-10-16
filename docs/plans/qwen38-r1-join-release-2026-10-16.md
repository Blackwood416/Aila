# R1 decode join-release hook + prefill event-fence fix (P0/P1, 2026-10-16)

Date: 2026-10-16. Repo `E:\RiderProjects\Aila`, base HEAD `bb07130`.
Follows `docs/plans/qwen38-r1-bottleneck-audit-2026-10-16.md` (levers 2 and 4).

## 0. Conclusions

1. **P0 (correctness, adopted)**: the two prefill expert dispatchers selected
   their returned fence with `x_ev != sycl::event{}`, which is always true on
   this toolchain (audit section 5), so a chunk whose single/ pair/ quad / eight
   bucket did not launch could return a queue-less default event that callers
   recorded as a staging-bank reader fence. Both dispatchers now track
   "was submitted" explicitly and return the last *actually submitted* bucket
   event (`src/ops/MoeExpertOps.cpp`, +32/-8).
   Verified with the audit probe rebuilt against the fixed lib: layer 20 now
   reports `[PROBE2 DEFAULT EVENTS] ... : 0` (the audit measured 48 across its
   6 probe arms); the probe's ALU/XMX arms reproduce the audit's numbers
   (K=140.5, KH=277.1, DX=39.3, DXH=146.8 ms/layer; 3.58x / 1.89x).
2. **P1 (decode join-release, measured below gate, stays opt-in)**: querying the
   last submitted event's status once before the decode CPU join
   (`AILA_Q4E_JOIN_RELEASE=1`, default 0) measured **+1.8%** on the 14K continue
   rate (A median 17.305, B median 17.617 tok/s) with **A/A spread 2.24%** -
   i.e. the effect is inside this series' own noise and far below the +5%
   default-on gate. Not adopted; the hook stays as a measurement flag.
3. The audit's synthetic `-14%` (29.2 -> 25.4 ms/token) does **not** transfer at
   this magnitude: 5 of the 6 audit overlap-arm logs show the release saving
   ~100 us/layer, but one shows zero difference on identical env
   (`submit-*.log`), and in production the resident hot kernels are only
   ~0.08 ms/layer, so the hideable window is ~3.8 ms/token *before* any overlap
   that already happens.

## 1. P0: which event the dispatchers return

| Item | Before | After |
|---|---|---|
| Selection | `if (single_ev != sycl::event{}) return single_ev; ...` (always true) | `if (single_submitted) return single_ev; ... else return {}` |
| Failure mode | no-single chunk returned a queue-less event -> `RecordStagingConsumer(empty)` -> next H2D's `depends_on` is an empty fence | returns the real last bucket event (single -> pair -> quad -> eight, reverse submission order) |
| Sites | `MoeExpertOps.cpp` gate_up `:6056-6058`, down `:6253-6255` | both dispatchers, +4 bools each |

The ~80 other `*dep != sycl::event{}` occurrences are `depends_on` guards; per
the audit they are harmless and were left untouched. The dead
`FinishResidentCpuRead` `== sycl::event{}` throw (`Qwen4ExpExpertExecutor.cpp:1275`)
is also left as-is: its input comes from always-submitting merge kernels, and a
queue-less event cannot be detected by comparison on this toolchain (the
producers now carry the real guarantee).

Receipts: `tmp/q4e-bottleneck-audit-20261016/expert2-fixed-L20.log`
(`DEFAULT EVENTS: 0`), probe rebuilt via `build_probe.ps1`.

## 2. P1: release point and mechanism

| Item | Value |
|---|---|
| Flag | `AILA_Q4E_JOIN_RELEASE` (int, default 0) |
| Hook | `src/models/Qwen4ExpBnb4Backend.cpp:1509-1516`, immediately before the `JoinResidentCold` / `JoinArenaColdRowsRow` ternary in `RunResidentDecode` (the count<=4 path called at `:2821`) |
| Mechanism | `(void)last.get_info<sycl::info::event::command_execution_status>()` - the same call the audit's `submit_probe` used to release the L0 v1 regular command batch |
| Semantics | scheduling only; no arithmetic change. All four arms' SSE content sha are identical (`a9aa5132...`) |

## 3. A/B/B/A (production timing, no --diag, 14K continue, n=32)

| arm | AILA_Q4E_JOIN_RELEASE | rate tok/s |
|---|---|---|
| jr-a1 | 0 | 17.497 |
| jr-b1 | 1 | 17.383 |
| jr-b2 | 1 | 17.850 |
| jr-a2 | 0 | 17.113 |

* A median 17.305 / B median 17.617 -> **B - A = +1.80%**; A/A spread 2.24%,
  B spread 2.66%: the effect is below the series' noise floor.
* Gate: default-on requires >= +5% (audit lever 2 estimate was -4..-10 ms/token
  = +8..+20%). Measured far below -> **not adopted, opt-in only**.
* Bundle env (all arms): `AILA_Q4E_HOTSET_PROFILE=<abs agent6k>` + int8 KV, via
  `tmp/q4e-sprint-r1-20261014/run-join-release.ps1`; receipts under
  `tmp/q4e-sprint-r1-20261014/jr-*/`.

## 4. Interpretation and next steps

1. The release is cheap and harmless but the production decode does not stall
   enough resident GPU work at the join to matter: the hideable set is the hot
   kernels (~80 us/layer) plus whatever else shares the batch, and the condvar
   wait may already overlap part of it.
2. Stronger decode candidates from the submit-path audit
   (`docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md`): the
   per-layer resident-input wait (double-buffered `x_host_`), fused-HC
   (`AILA_Q4E_DECODE_FUSED_HC=1`), shared fusion
   (`AILA_Q4E_MOE_SHARED_FUSED=1`), and the 64 B weights H2D - all rank above
   further join-release tuning.
3. The prefill XMX redesign (Erdos plan
   `docs/plans/qwen38-r1-prefill-expert-xmx-plan-2026-10-16.md`, quality gate
   `docs/plans/qwen38-r1-expert-xmx-quality-gate-recipe-2026-10-16.md`) is the
   next large lever: block 19.9 s -> ~13.5 s target.

## 5. Evidence

* `tmp/q4e-bottleneck-audit-20261016/expert2-fixed-L20.log` (P0 probe).
* `tmp/q4e-sprint-r1-20261014/jr-{a1,b1,b2,a2}/result.json` (rates, SSE sha).
* `tmp/q4e-sprint-r1-20261014/run-join-release.ps1` (arm sequence).
* Audit probe overlap arms: `tmp/q4e-bottleneck-audit-20261016/submit-*.log`.
