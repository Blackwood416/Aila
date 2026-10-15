# R1 decode instrumentation (B6): close the two decode cost-model measurement holes

Date: 2026-10-15. Repo: `E:\RiderProjects\Aila`, HEAD `40239cd`, pair `retained24`.
Scope: `src/models/Qwen4ExpBnb4Backend.cpp` only (diagnostics; no execution
semantics, no default-path change). Zero GPU, zero build, zero commit by this
agent. All line numbers are post-edit. ASCII only.

## 1. Problem statement (what was still unknown)

The retained20 decode refresh (n=512) measures 66.01 ms/token wall, of which
`[Q4E AdaptiveWork]` could only account for two derived leftovers
(`docs/plans/qwen38-r1-cost-model-2026-10-14.md:30-33`,
`docs/plans/qwen38-r1-decode-2026-10-14.md:50`):

1. "Unprinted MoE-front/shared residual" 8.684-9.091 ms/token (13.5-13.8%),
   recoverable only as `submitted_ms - (router_wait + all printed stages)`.
   It mixes: router gate GEMV submit, top-k submit, shared-expert submits,
   `hc_inject_add`, and the two stage buckets `kStageRouter`/`kStageShared`
   that the summary line never printed (line 50: "Backend.cpp:2134-2301,
   2336-2352 (kStageRouter at 2315 is not printed)").
2. hc / linear / full had only host-submit intervals (`hc_host_ms`,
   `linear_host_ms`, `full_host_ms`); their true device cost was unknown.
   The only earlier split (cost model section 1.2 table, `r1-stages2` arm) is
   the stage-serialized mode whose 384 syncs per forward change the workload
   and cannot be used for decode (`qwen38-r1-decode-2026-10-14.md:64,171`).

Backlog rows C5 (#0.3 / M3) asked exactly for this split:
`docs/plans/qwen38-r1-backlog-mining-2026-10-15.md:52-57,78,109`.

## 2. What was added

### 2.1 (a) Host-side submit intervals (always on with the summary line)

Declaration block: `src/models/Qwen4ExpBnb4Backend.cpp:2042-2055`. Pure
`steady_clock` deltas around the submission blocks; no queue interaction, no
new sync. Accumulated only when `AILA_Q4E_WORK_SUMMARY=1` (the line that
prints them would otherwise not exist); zero cost otherwise
(`host_split` = `s_work_summary`, `:2046`).

| Field | Definition | Measured at |
|---|---|---|
| `moe_front_ms` | Router gate projection submit: fused `ops::bf16_moe_front_fused` or `dense_rows` (raw-weight route: profile sets `AILA_Q4E_MOE_FRONT_FUSED=0`) | `:2195` (begin) / `:2210` (end) |
| `topk_submit_ms` | Router softmax+top-k device submit (plus resident classify if that arm is on) | `:2220` / `:2241` |
| `shared_submit_ms` | Shared expert submits: shared gate GEMV + fused gate/up/SiLU or 3-kernel fallback + shared down | `:2331` / `:2369` |
| `shared_add_ms` | `add_shared_expert_to_y` submit when `AILA_Q4E_MOE_SHARED_FUSED=0` (production) | `:2425` / `:2430` |
| `hc_inject_ms` | MLP residual block: `pending_mlp.Defer()` or `ops::hc_inject_add` submit | `:2433` / `:2445` |
| `router_other_ms` | `kStageRouter` minus `router_wait_ms`, `moe_front_ms`, `topk_submit_ms`, `shared_submit_ms`; residual of the router stage: lease/flag setup, `StageResidentInput` (D2H + its host wait), `have_pred` bookkeeping, `bind_actual_route`, the `stage()` call itself | computed `:2600-2601` |
| `shared_other_ms` | `kStageShared` minus `shared_add_ms`, `hc_inject_ms`; miss-rate bookkeeping | computed `:2602` |
| `tail_ms` | `submitted_ms_now - sum(stage_ms[8])`: everything after the last `stage()` call (state publication, snapshot capture, stats read) plus any gap before the first stage timer | `:2596-2599`, printed `:2625` |

### 2.2 (b) Device spans (gated, default off)

Gate: `AILA_Q4E_DECODE_STAGE_EVENTS=1` (`:2063`), default 0. Effective only
when the summary is on (`:2064`) and the queue was created with
`sycl::property::queue::enable_profiling` (`:2067`). If the gate is on but the
queue is not profiling-enabled, one warning is logged (`:2072`) and every
`dev_*_ms` prints 0.

Design: the phase APIs (`RunHyperConnection`, `RunAttentionWindow`, `lm_head_`,
shared/dense submits) return no `sycl::event`, and widening their signatures
would touch other translation units. Instead the gate inserts one no-op
in-order barrier per phase boundary (`ctx.queue().ext_oneapi_submit_barrier()`,
`:2082-2086`; same API already used at `Qwen4ExpExpertCache.cpp:2250`) and
reads its `event_profiling` start/end after a single wait on the last marker
(`:2582-2589`). Interval for marker pair `(i-1, i)` is
`command_start(i) - command_end(i-1)` accumulated into bucket `i`; barrier
self time is excluded, queue bubbles are included.

Bucket -> submissions covered (marker sites):

| Field | Marker sites | Covers |
|---|---|---|
| `misc_dev_ms` | `:2086` (initial), `:2110` (after PLE), `:2194` (pre-router gap) | Pre-loop, per-layer PLE (L1) + previous-layer tail/dump, lease setup, `StageResidentInput` D2H |
| `hc_dev_ms` | `:2129`, `:2161` | Attn HC and MLP HC (`RunHyperConnection*`), incl. `pending_mlp` materialization (profile sets `AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=1`, so the residual inject is folded here, not into `hc_inject_ms`) |
| `linear_dev_ms` / `full_dev_ms` | `:2145` | `RunAttentionWindow` (bucket chosen by `layer.is_linear`) |
| `router_dev_ms` | `:2242` | Router gate GEMV + top-k (and resident classify) |
| `shared_dev_ms` | `:2370` | Shared gate + gate/up/SiLU + shared down |
| `experts_dev_ms` | `:2414` | `RunResidentDecode` span, including any in-order bubble while the host waits for the route event |
| `resid_dev_ms` | `:2447` | `add_shared_expert_to_y` + `hc_inject_add` (the latter ~empty under the production `pending_mlp` setting) |
| `head_dev_ms` | `:2517` | Final mixer HC + LM head (+ swap-DMA bubble only if `AILA_Q4E_ADAPTIVE_SWAP_FREQ>0`, default 0) |

`dev_events=0|1` is appended so consumers can tell "inert" from "measured".
Markers observe `ctx.queue()` only; copies issued on `copy_q_` or expert
supply queues (staging H2D, swap DMA) are invisible by design.

### 2.3 Account identity (checkable from one log line)

With `S = sum(stage_ms[all 8])` and `tail = submitted - S`:

    old_unprinted_residual = Router + Shared + tail - router_wait
                           = moe_front_ms + topk_submit_ms + shared_submit_ms
                             + router_other_ms + shared_add_ms + hc_inject_ms
                             + shared_other_ms + tail_ms

(`router_wait_ms` cancels because it is a component of `kStageRouter`.)
Sanity target: the sum should reproduce the historical 8.68-9.09 ms/token.

The printed line format is backward compatible: fields are appended after
`h2d_mb` (`:2603-2610`), and the tree's parser is key=value regex
(`tmp/q4e-agent-plan-audit-20261012/audit_inventory.py:119`).

## 3. Validation arms for the coordinator

GPU mutex first (no fixed sleeps; poll for process exit by hand/launcher):

    Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue  # must be empty
    pwsh -NoProfile -File build.ps1 -BuildDir build-refactor

Arm A1/A2 - host split (also the A/A noise floor for the new fields):

    python tmp/q4e-sprint-r1-20261014/run.py b6-host-a1 --diag --prime
    python tmp/q4e-sprint-r1-20261014/run.py b6-host-a2 --diag --prime

`--diag` sets `AILA_Q4E_WORK_SUMMARY=1` (`run.py:39`), so the new fields
appear with no extra env. Read, per `[Q4E AdaptiveWork]` line with `count=1`
(>=100 consecutive tokens; parse key=value like `audit_inventory.py:119`):
`moe_front_ms topk_submit_ms shared_submit_ms shared_add_ms hc_inject_ms
router_other_ms shared_other_ms tail_ms`, plus the identity check in 2.3.
Expected shape from the old residual (8.68-9.09 total): shared/`moe_front`
host submits are the prime suspects; `hc_inject_ms` should be ~0 under
`AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL=1`.

Environment routing: the server->worker child environment strips only the
`XPTI_`/`UR_`/`ZES_`/`ZET_`/`ZE_ENABLE_`/`OCL_ICD_`/`SYCL_CACHE_`/
`__KMP_REGISTERED_LIB_` prefixes (`src/runtime/ChildEnvironment.cpp:28-39`),
so `AILA_Q4E_*` overrides exported by `run.py` reach the worker without
`AILA_WORKER_ENV_PASSTHROUGH`.

Arm B1 - gate-inertness on the worker (today's worker queue is NOT
profiling-enabled; `DeviceState` builds `Context context{trace.get()}` at
`src/inference/Model.cpp:40`, engine gate `AILA_PROFILE_Q4E_COLD_DEVICE` only
exists at `src/engine/EngineLoad.cpp:15`):

    python tmp/q4e-sprint-r1-20261014/run.py b6-dev-b1 --diag --prime AILA_Q4E_DECODE_STAGE_EVENTS=1

Pass criteria: exactly one `[Q4E DecodeStageEvents] ... profiling is off`
warning; every line carries `dev_events=0` with all `dev_*_ms=0`; 14K
continue tok/s within the A1/A2 spread. This proves the gate is inert where
the queue cannot support it.

Arm B2 - real device spans (requires a one-line enabler OUTSIDE this agent's
file scope; coordinator decides whether to fund it):

    src/inference/Model.cpp:40 ->
    Context context{trace.get(), aila::env::read_flag("AILA_Q4E_DECODE_STAGE_EVENTS", false)};

then rerun `run.py b6-ev-b2 --diag --prime AILA_Q4E_DECODE_STAGE_EVENTS=1`.
Pass criteria: no warning, `dev_events=1`, `hc_dev_ms + linear_dev_ms +
full_dev_ms + head_dev_ms` <= `hc_host_ms + linear_host_ms + full_host_ms +
head_host_ms` (host intervals include queue wait; device spans cannot exceed
them by more than marker noise), and the dev arm's tok/s compared against
`b6-host-a2` bounds the profiling+barrier overhead for the record.

Discipline: back-to-back arms, same binary/pair, no other GPU work; the dev
arm is an attribution arm - never a rate receipt (the gate adds 434 barrier
commands per 48-layer forward = 9/layer + initial + head, plus one wait).

## 4. Overhead argument and risks

- Gate off, summary off: 2 cached env reads per process + one predictable
branch per `dev_mark` site; no waits, no queue ops, empty vector. The default
decode path is unchanged.
- Summary on, gate off: +10 `steady_clock::now()` per layer (~480/token,
~20-25 ns each ~= 10-12 us/token ~= 0.02% of 66 ms), which only runs in the
diagnostic arms that already pay the stage machinery (`stage()` itself calls
`now()` 8x/layer).
- Gate on: +9 barrier commands/layer (434/token) + one wait per forward +
~434x2 `get_profiling_info` reads. Barrier commands are cheap but nonzero;
the end-of-forward wait would otherwise be paid by the sampling D2H anyway
(same in-order queue). Bound measured via A2 vs B2.
- Risks: (1) bubbles from unseen queues land in whichever marker interval they
fall into; (2) `AILA_PROFILE_Q4E_STAGES=1` must NOT be combined with the dev
arm - `stage()` synchronizes every boundary and would dominate both host and
device fields; (3) an exception thrown mid-forward discards the marker vector
without read-back (harmless); (4) profiling-enabled queues change per-command
overhead, so `dev_events=1` numbers are for attribution only.
- NOT verified by this agent: no build/GPU was run; the edit is code-reviewed
only. First compiler contact is the coordinator's build.

## 5. Interaction with existing switches

- `AILA_Q4E_WORK_SUMMARY=1`: required; it owns the `[Q4E AdaptiveWork]` line.
Host fields (a) print whenever it is on; device fields (b) additionally need
`AILA_Q4E_DECODE_STAGE_EVENTS=1` and a profiling queue.
- `AILA_PROFILE_Q4E_STAGES=1`: keep separate (see risk 2). Its
`kStageRouter`/`kStageShared` semantics are unchanged.
- `AILA_PROFILE_Q4E_DECODE` / `AILA_Q4E_PROFILE_COMPONENTS`: independent
prints; neither gates the new fields.
- Data is per `Forward` call = per decode token in the standard arm; no
cross-call statics are involved.

TODO / out of scope:
- The layer-major path prints its own shorter `[Q4E AdaptiveWork]`
(`src/models/Qwen4ExpLayerMajor.cpp:477`) without these fields.
- Worker-path profiling enabler (arm B2 one-liner) is a deliberate outside
change for the coordinator to approve or reject.
- If `misc_dev_ms` turns out to be non-trivial after B2, add one more marker
between `stage(kStageHc)` and the router gate to split `StageResidentInput`.

## 6. Section C4 follow-up: worker queue profiling enabler (opt-in, decoupled)

Date: 2026-10-15. Repo HEAD `fe1bf9e`, pair `retained25`. Scope of this
section: `src/inference/Model.cpp` only (worker queue construction; no
execution semantics, no default-path change). Zero GPU, zero build, zero
commit by this agent. Line numbers are post-edit in the working tree; the
backend file is owned by the round-2 (C2) window, so backend citations below
name the committed B6 state (`4c7abad`) unless stated otherwise.

### 6.1 Change

Arm B2 (section 3) asked for a worker-path enabler. That one-liner had in fact
already been folded into B6's commit (`4c7abad`):
`Context context{trace.get(), aila::env::read_flag("AILA_Q4E_DECODE_STAGE_EVENTS", false)}`,
which made the stage-event gate silently turn the shared worker queue into a
profiling queue - so arm B1's "inert gate" property no longer held at HEAD. C4
splits the two switches:

| File:line | Change |
|---|---|
| `src/inference/Model.cpp:22` | `#include <exception>` for the fallback catch |
| `src/inference/Model.cpp:42-59` | new `detail::MakeDeviceContext()` |
| `src/inference/Model.cpp:43-44` | read-once flag `AILA_Q4E_DEVICE_QUEUE_PROFILING`, default 0 |
| `src/inference/Model.cpp:45` | flag unset -> `Context(trace)`, i.e. the unchanged `property_list{in_order()}` (`src/core/Context.hpp:35`) |
| `src/inference/Model.cpp:46-58` | flag set -> `Context(trace, true)`; if queue creation rejects `enable_profiling`, build the plain queue instead and warn once (`:54-55`) |
| `src/inference/Model.cpp:65` | `DeviceState::context{MakeDeviceContext(trace.get())}` (was the stage-event-gated one-liner) |
| `src/core/Context.hpp:26-35` | not touched: already parameterized, keeps `in_order()` in both branches and only adds `enable_profiling()` (`:33-34`) |

Env name: `AILA_Q4E_DEVICE_QUEUE_PROFILING`, truthy per `std::atoi != 0`
(`src/utils/EnvUtils.hpp:18-36`, the convention shared by every other
`AILA_Q4E_*` switch). Default / unset: the worker queue stays the plain
in-order queue (`src/core/Context.hpp:35`), i.e. the path is unchanged with no
`AILA_Q4E_*` env set.

### 6.2 Read-path self-check (why the property is sufficient)

Committed B6 protocol (`git show 4c7abad:src/models/Qwen4ExpBnb4Backend.cpp`):
gate `:2063`, `has_property<enable_profiling>()` `:2067`, warn-once `:2072`,
one `ext_oneapi_submit_barrier()` per phase boundary `:2084`, a single
`wait_and_throw` on the last marker `:2583`, then `command_end(i-1)` /
`command_start(i)` `:2586-2587`. That is exactly the profiling contract; the
working tree (C2) keeps the same read-back at `:2597-2604`.

No second consumer assumes a non-profiling queue: every other
`get_profiling_info` reader is guarded - `Qwen4ExpExpertCache.cpp:1900` (own
`cache_trace_` flag plus the same `has_property` check),
`Qwen4ExpExpertCache.cpp:4112-4119` (try/catch around the copy-queue DMA read),
plus test-local queues (`tests/ops/GdnChunkTests.cpp:175,201,270`,
`tests/models/Qwen4ExpIntegerMoeProbe.hpp:268`).

Empirical: the enabler's predecessor already ran the A-shape once -
`tmp/q4e-sprint-r1-20261014/dec-instr-b2-worker.log` (worker build id
`aila-0.2.0-abi1-40239cda...`, `dec-instr-b2/stdout.log:1`) carries
`dev_events=1` with non-zero `hc/linear/full/router/shared/experts/resid/head_dev_ms`
on its `count=1` lines, so marker events do return timestamps on this
device/plugin stack once the queue has the property.

### 6.3 Coordinator arms (run.py form)

Prereq as in section 3: GPU mutex (`Get-Process AilaWorker,AilaServer` empty),
one build (`build.ps1 -BuildDir build-refactor`), both arms on the same binary,
back to back. `--diag` already sets `AILA_Q4E_WORK_SUMMARY=1` (`run.py:39`);
the `KEY=VALUE` arguments are applied by `run.py:55-58`.

Arm A - real device spans:

    python tmp/q4e-sprint-r1-20261014/run.py c4-queueprof-a1 --diag --prime AILA_Q4E_DECODE_STAGE_EVENTS=1 AILA_Q4E_DEVICE_QUEUE_PROFILING=1

Read from `c4-queueprof-a1-worker.log`: no `queue profiling is off` warning;
the `count=1` `[Q4E AdaptiveWork]` lines carry `dev_events=1` and non-zero
`*_dev_ms` buckets (9 in the B6 line, 13 with C2; the C2 parser
`tmp/q4e-sprint-r1-20261014/dec_instr_dev_parse.py` already consumes them).
Attribution arm only - a profiling queue changes per-command overhead, so its
tok/s is not a rate receipt.

Arm B - gate inert, zero device fields:

    python tmp/q4e-sprint-r1-20261014/run.py c4-queueprof-b1 --diag --prime AILA_Q4E_DECODE_STAGE_EVENTS=1

Pass: exactly one `[Q4E DecodeStageEvents] ... queue profiling is off`
warning, every line `dev_events=0` with all `*_dev_ms=0`, tok/s within the
`dec-instr-a1` / `b6-host-a2` host spread. Neither arm may be combined with
`AILA_PROFILE_Q4E_STAGES=1` (section 5).

### 6.4 Risks

- Behavior change vs HEAD: `AILA_Q4E_DECODE_STAGE_EVENTS=1` alone no longer
  enables profiling in the worker (it did between `4c7abad` and this change);
  both envs are now required. Nothing changes when both are unset.
- Not compiled or run by this agent (zero build); first compiler contact is the
  coordinator's build. Returning the non-movable `Context` by value relies on
  C++17 guaranteed copy elision (`CMakeLists.txt:3`).
- The fallback catch is defensive only: DPC++/L0 has not been observed to
  reject the property on this device. TODO - drop it if arm A shows it never
  fires.
- Two backend strings are stale after C4 but live in the peer-owned file
  (`src/models/Qwen4ExpBnb4Backend.cpp`, numbers as of writing): the comment
  at `:2059-2062` still says the worker "owns a non-profiling Context", and
  the one-time warning at `:2072-2073` still ends with "queue property fixed
  in Context.hpp:32-35". TODO for a later pass - the switch now lives in
  `Model.cpp:42-59`.
