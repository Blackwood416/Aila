# R1-S service/multi-session quick evaluation (2026-10-14)

Role: S (service / multi-session), sub-agent of the Qwen3.8-Flash-Next A770
performance sprint. Scope: answer charter section 3 item 6 at code level,
derive the aggregate-throughput upper bound table, and write exactly one
minimal probe request. This role ran **zero GPU work**: no Worker/Server/Aila
process, no build, no git write. All numbers below are copied from an existing
source with a `file:line` or artifact path; every prediction is marked
`[估算]`.

Revision anchor: HEAD `946c644`, pair `retained20`
(`tmp/q4e-mtp-alignment-20261011/retained20/pair.json`:
worker `4d75c59c...`, server `211aabb0...`).

---

## 0. Conclusion summary

1. **There is no true cross-session decode batch for `qwen4exp_nf4` in the
   current code.** With `--sessions 2 --decode-batch-size 2` the two requests
   are admitted concurrently but executed **round-robin, one token per
   request per scheduling quantum** on the single submission thread and the
   shared device mutex. The `--decode-batch-size` flag is accepted, validated
   and reported by `/health`, but is inert for this backend.
2. **Aggregate decode throughput upper bound at B=2 today = the single-session
   rate**, 15.6 tok/s at the charter's 64 ms/token basis (per-session 7.8
   tok/s `[估算]`). The ceiling rises only if a real batched step amortizes
   supply (21.1 tok/s at B=2) or supply/compute overlap (32.4 tok/s); both
   need new code (Model-level `ForwardDecodeBatch` for Q4E + multi-session
   expert staging + per-session KV rows), not a flag.
3. **Capacity is not the blocker**: two sessions at context 4096 reserve
   ~223.6 MiB/session `[估算]` (units note in section 2.1), i.e. that much
   beyond what the load-time accounting already deducts (one `max_seq_len`
   session inside the 3584 MiB headroom). At production context 32768 the
   second session's 1000.2 (MiB-convention) session state comes out of the
   same headroom.
4. **2-session operation on the adaptive pipeline is unverified** (status
   `qwen38-qwen4exp-pipeline-status-2026-10-13.md:253`). The only precedents
   are in-process C=128 interleave runs from 2026-10-07 on the older cache
   layout (`tmp/q4e-server-p1-session2.log:51` PASS). No multi-session
   throughput number exists anywhere in `docs/plans` or `tmp`.
5. One minimal probe is specified in section 3/4 (2 sessions, prompt target
   2560 tokens, rendered ~2472-2490 so the QSA sparse-prefill gate at 2052
   stays engaged, `generate=32`, `decode-batch-size=2`). Predicted aggregate
   rate: 14.8-15.6 tok/s
   `[估算]`, i.e. no batching gain; the probe mainly buys (a) the first
   adaptive-pipeline 2-session evidence and (b) per-session
   `[Q4E AdaptiveWork]` decode rows for the R/D cost models.

---

## 1. Code-path verdict: how decode actually runs

### 1.1 The flag path (accepted, propagated, inert)

- `src/server/ServerMain.cpp:84-85` parses `--sessions 1..4` and
  `--decode-batch-size 1..4`.
- `src/server/HttpServer.cpp:998-1001` validates `decode-batch-size` against
  `maxActiveRequests = --sessions`; `HttpServer.cpp:1006-1014` copies both
  into the client limits and `clientOptions.maxDecodeBatchSize`;
  `HttpServer.cpp:542` reports `max_decode_batch_size` in `/health`.
- `src/worker/InferenceWorker.cpp:113-125` (`CoreLimits`) maps the negotiated
  limits into `SchedulerOptions` (`maxActiveRequests`, `maxDecodeBatchSize`),
  `InferenceWorker.cpp:490-505` builds the `Scheduler` with them and echoes
  the bound in the configure response.
- Capability negotiation does **not** filter this per backend:
  `continuous_batching_v1` is advertised globally
  (`src/ipc/InferenceProtocol.cpp:625`), required by the client whenever
  `maxDecodeBatchSize > 1` (`src/runtime/InferenceClient.cpp:1201-1202`), and
  `validate_decode_batch_size` only checks the numeric bound
  (`src/ipc/InferenceProtocol.cpp:643-650`, `928-938`; the model-capability
  list at `630-641` is not consulted there). So a Qwen4Exp server starts
  happily with `--decode-batch-size 2`.

### 1.2 The scheduler only batches rows with a non-null model batch key

- `src/inference/Scheduler.cpp:436-446`: a batch is formed only when
  `batchKey = EligibleBatchKey(current.front())` is non-null and peers share
  it. `EligibleBatchKey` (`Scheduler.cpp:386-395`) returns the stored
  `decodeBatchKey`.
- `src/inference/Scheduler.cpp:693`: `entry->decodeBatchKey =
  entry->request->DecodeBatchKey()` at submit time.
- `src/inference/Scheduler.cpp:454-455`: `current.size() == 1` -> `RunOne`.
  A null key therefore always yields single-row execution; with
  `maxActiveRequests = 2` both requests live in `runnable` and are rotated at
  complete-forward boundaries (`Scheduler.cpp:409-413`, `428-431`,
  `462-473`; design note `Scheduler.hpp:102-113`).

### 1.3 Qwen4Exp returns a null key and has no batch entry point

- `src/models/Qwen4ExpBnb4Backend.hpp:254-256`:
  `bool SupportsDecodeBatch() const noexcept { return false; }` with the
  comment "Qwen4Exp has no cross-session decode batch path; the Model session
  backend also excludes ForwardDecodeBatch at compile time."
- `src/inference/Model.cpp:218-221`: `DecodeBatchKey()` returns
  `model.get()` only when `SupportsDecodeBatch()`, else `nullptr`.
- Belt and braces on the execution side: `src/inference/Model.cpp:430-434`
  throws `UnsupportedModel` with the literal message "Qwen4Exp does not
  support cross-session decode batching";
  `src/inference/Session.cpp:212-218` and
  `src/inference/Request.cpp:376-381` throw the same way one level down.

Therefore: `--sessions 2 --decode-batch-size 2` -> 2 admitted sessions,
strict alternation of one-token quanta, one serialized GPU pipeline
(device mutex: `src/inference/Model.cpp:198-200` and `459`). **Round-robin
serial, not true batching.**

### 1.4 `count > 1` inside Qwen4Exp means "rows of one sequence"

- `src/models/Qwen4ExpBnb4Backend.cpp:2317-2319`: adaptive dispatch is
  `count > 4 -> RunAdaptivePrefill` else `RunResidentDecode`.
- `RunResidentDecode` (`Qwen4ExpBnb4Backend.cpp:1171-1350`) iterates
  `t = 0..count-1` over `x_t = x_base + t * hidden_size_` and the per-row
  router ids of the *same* forward (`1250-1254`, `1266-1277`); the multi-row
  branch at `1199-1246` is the opt-in arena verify-rows path for a
  verification window.
- `Qwen4ExpBnb4Backend.cpp:1985`: `is_prefill = !options.decode_verification
  && (options.prefill_remaining >= 0 || count > 1)`.
- `src/models/IModelBackend.hpp:18`: `decode_verification` is "Explicit
  multi-token decode, not a new prompt slice"; the only in-tree producer is
  MTP (`src/models/Qwen4ExpMtpRuntime.cpp:109`), and MTP is off in production
  (`tools/server/start-qwen38.ps1:19`).
- Cross-reference in docs:
  `docs/plans/qwen3.8-flash-next-a770-next-optimization.md:290` ("Q4E's
  `count>1` currently represents same-sequence prefill, not multi-session
  decode batch"); `docs/plans/qwen38-server-p1-2026-10-07.md:11`
  ("cross-session decode batch key is null, unsupported batch calls excluded
  at compile time"); status `qwen38-qwen4exp-pipeline-status-2026-10-13.md:253`
  ("unverified" multi-session path, the status document's own label).

Consequence for the service question: any B>1 aggregate-throughput gain must
come from overlapping/admitting more work, not from a wider decode step.
The current code cannot overlap the per-token supply and per-token compute of
two requests either, because one submission thread drives both.

---

## 2. Aggregate decode throughput upper bounds

### 2.1 Basis and split

Units note: memory values below are quoted exactly as printed by their source;
the worker/plan "MB" values use `bytes / 1048576` (MiB convention), while raw
byte counts are given in bytes. Derived values state their unit explicitly.

Charter section 1.1 (canonical basis, used as instructed): single-session
decode 15.2-16.0 tok/s; the 64 ms/token nominal is 1/15.625. Split from the
same section plus `[Q4E AdaptiveWork]` field names
(`src/models/Qwen4ExpBnb4Backend.cpp:2489-2505`):

| term | value | source |
|---|---:|---|
| router_wait_ms | 13.4 | charter section 1.1 |
| experts_host_ms | 19.7 | charter section 1.1 (cpu_compute 12.9 + join 9.1 overlap inside) |
| supply s = router_wait + experts_host | **33.1** | sum, ms/token |
| rest c = 64 - 33.1 | **30.9** | ms/token (dense/GDN/attn/head/sampling/submit/IPC) |

Cross-check: the 2026-10-12 profile
(`docs/plans/qwen38-a770-agent-performance-plan-2026-10-12.md:47-56`) gives
router_wait 11.924 + experts_host 21.213 = 33.14 ms, i.e. the same 33.1/30.9
split from a different night. Both splits are means over 16 single-row
samples and treat `router_wait` as queue-inclusive, not router-exclusive; the
charter section 3 item 2 re-measurement is the D role's deliverable and may
move these numbers by a few ms.

Same-day A/A reference (arm A1, `retained20`,
`tmp/q4e-sprint-r1-20261014/aa-a1-run.log`): continuation decode 14.762
tok/s = 67.7 ms/token; its first round was anomalously slow (155.67 s for a
13944-token prompt vs the charter's 70.2 s), so it is quoted only as a
sensitivity band (scale all numbers below by 64/67.7 = 0.945).

### 2.2 Upper-bound table `[估算]`

Let B = concurrent sessions, T1 = 64 ms, s = 33.1 ms, c = 30.9 ms.

- (a) No sharing at all (strictly serial pipeline; **this is today's code**):
  `T(B) = B * T1`, `R_agg = 1 / T1`, `R_session = R_agg / B`.
- (b) Supply served once per step, only the fixed supply/launch/join cost
  amortized; compute still per row: `T(B) = s + B * c`.
- (c) Supply and compute fully in parallel (per-step critical path
  `max(s, B * c)`): for B >= 2 the bound is `R_agg = 1 / c`.

| B | (a) agg / per-session | (b) agg / per-session | (c) agg / per-session |
|---:|---:|---:|---:|
| 2 | 15.6 / 7.8 | 21.1 / 10.5 | 32.4 / 16.2 |
| 3 | 15.6 / 5.2 | 23.8 / 7.9 | 32.4 / 10.8 |
| 4 | 15.6 / 3.9 | 25.5 / 6.4 | 32.4 / 8.1 |

All values tok/s `[估算]`. Worked examples: (b) B=2 -> 33.1 + 61.8 = 94.9 ms
per two-token step -> 2/0.0949 = 21.1; (b) B=4 -> 156.7 ms -> 25.5;
(c) -> 1/0.0309 = 32.4.

Notes:

- (a) is an equality, not a bound: the current round-robin costs exactly T1
  per token per session (plus scheduler overhead), so B>1 buys nothing.
- (b) already assumes the union of B rows' misses is filled, joined and
  routed once per step; a realistic implementation without expert-set sharing
  lands between (a) and (b), closer to (a).
- Absolute ceiling if supply became free and shared as well:
  `R_agg = B / c` = 64.7 / 97.1 / 129.4 tok/s `[估算]`. Unreachable with the
  current MoE-per-row supply; listed only to bound the search space.
- At B=2 the interesting target is therefore ~21 tok/s aggregate (+35% over
  the serial fallback) and it requires new batched-decode code, not
  configuration.

### 2.3 Capacity constraints at B=2

- Resident expert slots: 2792 with bf16 KV (plan
  `qwen38-a770-agent-performance-plan-2026-10-12.md:259-270`, P2 table);
  slot size 2.416992 MiB (`plan:58`) -> 2792 * 2.416992 MiB = 6.59 GiB
  (6748 MiB) of the expert cache; hard cap
  `AILA_Q4E_MAX_GPU_SLOTS` 2800 and staging 2 x 16 slots
  (`Qwen4ExpBnb4Backend.cpp:751-757`; status section 2.1).
- KV: 768.0 (MiB convention) at the production 32768 layout / bf16 =
  24.0 KiB/token (P2 table `plan:267`; the standard runner uses
  `--context 32768`, status section 4.3 and
  `tmp/q4e-sprint-r1-20261014/aa-a1/launch.json` show the same command
  shape); 14K single session ~336 MiB by the same rate, int8 KV halves the
  code bytes (384.0 at 32K).
- Session state: 1000.2 (MiB convention) at C=32768 (P2 table); 176,234,496 B
  = 168.08 MiB at C=2048 (`docs/plans/qwen38-server-p1-2026-10-07.md:28`).
  Two-point linear fit: ~112.6 MiB + 0.0271 MiB/token -> ~223.6 MiB
  (~234 MB) per session at C=4096 `[估算]`.
- Load-time admission reserves `AILA_Q4E_VRAM_RESERVE_MB` (production 3584,
  read as MiB: `<< 20`)
  **plus one** `max_seq_len` session state and nothing per extra session
  (`Qwen4ExpBnb4Backend.cpp:712-722`, `reserve = max_reserve +
  total_session_state`). For B=2 this under-counts one session: ~223.6 MiB at
  C=4096, ~1000.2 at C=32768 `[估算]`, both drawn from the 3584 MiB headroom.
  Context for the risk: the same code comments that ~1 GB free already slowed
  kernels ~2x under WDDM paging (`Qwen4ExpBnb4Backend.cpp:693-695`).
- The server never sets wire `maxSessionCacheBytes` (no assignment anywhere
  under `src/server`; `HttpServer.cpp:1005-1031` sets device/pinned/session-host
  budgets only), so the only 2-session guards are `maxSessions` and the
  worker's session admission (`InferenceWorker.cpp:615-618`).

Net: B=2 fits, but it consumes headroom that the single-session design point
reserved for the desktop/prefill workspace; this is a monitoring item for the
probe (the load-time `[Qwen4Exp] Memory Admission Accounting` line plus
`memory-sample-ms` if desired), not a predicted failure.

---

## 3. Minimal probe: specification and risks

### 3.1 Why a new probe instead of an existing test asset

- `tests/server/batching_gpu_tests.py`: its `--backend` choices include
  `qwen4exp_nf4` (`socket_tests.py:33-34`) but it hardcodes `args.context =
  512` (`batching_gpu_tests.py:429`), uses 16-32-token prompts
  (`batching_gpu_tests.py:36-41`) and `MAX_DEVICE_BYTES = 8 GiB` (`:32`),
  below the Qwen4Exp layout (expert cache alone is 6.59 GiB before static
  weights; an older Qwen3.8 run logged a 13.945 GiB device ledger peak,
  `docs/plans/qwen3.8-flash-next-supply-h7-results-2026-10-03.md:105`); and
  its own scope note
  (`:42-46`) requires the outer guard to prove *native* decode_batch
  execution, which Qwen4Exp cannot satisfy. Not usable.
- `tests/server/socket_tests.py --mode gpu --backend qwen4exp_nf4 --sessions
  2 --decode-batch-size 2 --case concurrent_1_2_4_fixed_seed`: supported
  (`:1441-1445`) and it does check 1/2/3/4-way admission, overlap and
  fixed-seed semantic equality (`concurrency()` `:903-942`, peak_active check
  `:939`). But prompts are 16-token and context defaults to 512 -> **no QSA
  coverage and no throughput claim**. Usable as an optional cheap functional
  companion, not as the QSA-compliant throughput probe.
- `tests/server/batching_socket_tests.py`: CPU fake Worker only ("The CPU
  fake Worker does not advertise a native batch executor", `:3-5`) - it
  documents the serial-fallback semantics but makes no GPU claim.

### 3.2 Probe (1 item)

Script (written, ASCII, no GPU run by this role):
`tmp/q4e-sprint-r1-20261014/service/probe_sessions2_qwen4exp.py`.

What it does: starts its own server via the existing
`tests/server/socket_tests.py` helpers, submits two **different** prompts
(target 2560 tokens; the deterministic filler renders 2472 local tokens each,
plus chat-template overhead, so both stay above the 2052 QSA gate)
concurrently (barrier-submitted SSE), each `generate=32`,
`--sessions 2 --decode-batch-size 2 --context 4096 --chunk 4096`, waits by
polling `/health` and SSE terminals (no fixed sleeps), then parses the worker
log for `[Q4E AdaptiveWork]`.

QSA compliance: the sparse-prefill predicate is
`(length + count) / 4 > block_topk` with `block_topk = 2048 / 4 = 512`
(`Qwen4ExpBnb4Backend.cpp:1028`, `2931-2935`), i.e. the QSA path is active
from 2052 total tokens (513 complete blocks; 2051 is the last dense
position). The probe refuses `--prompt-tokens < 2052` and additionally
verifies the server-reported `usage.prompt_tokens >= 2052`.

Log hygiene: the engine warmup emits `count=2` at pos=0 and `count=1` at
pos=2 (`src/engine/EngineLoad.cpp:597-620`); the parser keeps only decode
rows at pos >= 2052 and flags any `count` 2..4 row at pos > 0.

### 3.3 Risks and failure criteria

- Prior evidence: 2-session operation was validated in-process at C=128 on
  2026-10-07 (`tmp/q4e-server-p1-session2.log:51` PASS "shared T1: memory,
  interleave, append, reset, truncate, lifetime") and for MTP isolation
(`tmp/q4e-server-mtp-session-final.log` last line PASS). Both predate the
adaptive-only pipeline and carry no throughput numbers. Status line 253
remains the correct label: **unverified on the current adaptive pipeline**.
- No multi-session failure mode is documented: no
  `UR_OUT_OF_RESOURCES`/session-capacity incident exists in `docs/plans` or
  `tmp` (the documented OOR case is full pinned host memory, roadmap
  `qwen38-optimization-roadmap-2026-10-10.md:98-101`).
- Failure criteria for the probe: request error/timeout or
  `health.status=faulted`; worker log lines "USM allocation failed" or
  "UR_OUT_OF_RESOURCES"; IPC error "Session count or cache reservation limit
  reached"; `usage.prompt_tokens < 2052`; any `[Q4E AdaptiveWork]` row with
  `count` 2..4 at pos > 0 (MTP is off, so none is expected); no overlap
  (`min(done_at) <= max(first_content_at)`); or a decode `pos` sequence with
  zero repeated values (no interleave proof).
- Interpretation of a clean pass: aggregate decode rate should land at or
  below the table row (b) for B=2 (<= 21.1 tok/s `[估算]`), most likely
  ~14.8-15.6; that confirms "no batching gain today" and closes the service
  question at the upper-bound level.

---

## 4. Probe request (exactly one)

Owner: coordinator (this role cannot run GPU). Pair: `retained20`.

```powershell
# 0) GPU mutual exclusion (must print nothing)
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue
# 1) production worker environment (same as A/A arm A1), MTP off, chunk 4096
$config = Get-Content docs/plans/qwen38-server-mtp-2026-10-07.json -Raw | ConvertFrom-Json
foreach ($entry in $config.worker_environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($entry.Name, [string]$entry.Value, 'Process')
}
$env:AILA_Q4E_SERVER_MTP = '0'; $env:AILA_Q4E_VRAM_RESERVE_MB = '3584'; $env:AILA_Q35_PREFILL_CHUNK = '4096'
# 2) probe (script adds: --sessions 2 --decode-batch-size 2 --context 4096
#    --chunk 4096 --prompt-tokens 2560 --generate 32; it sets
#    AILA_Q4E_WORK_SUMMARY=1 and AILA_SERVER_WORKER_LOG itself; do NOT add
#    --production-timing, it would strip the AdaptiveWork diagnostics.
#    Defaults: build-refactor-server/AilaServer.exe, build-refactor/AilaWorker.exe,
#    C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2; override with
#    --server/--worker/--model if the pair moved.)
python tmp/q4e-sprint-r1-20261014/service/probe_sessions2_qwen4exp.py --output tmp/q4e-sprint-r1-20261014/service/run-s2-a1
```

Expected wall time: worker load ~64-68 s (charter section 1.1) + 2 prefills
~15 s each + 2 x 32 decode tokens ~4 s + shutdown ~10 s
(`[估算]` ~110-120 s); the script's timeouts are 300 s startup / 45 s
shutdown and there are no fixed sleeps.

Fields to extract from `run-s2-a1/result.json`:

- `requests[].semantic.usage.prompt_tokens` (>= 2052 required),
  `completion_tokens`, `finish_reason`;
- `metrics.per_session_decode_rate` [(ctok-1)/(done-first), charter
  section 5 item 8], `metrics.aggregate_decode_rate_pair_window`,
  `metrics.overlap_window_seconds`, `metrics.decode_overlapped`;
- `peak_active_requests` (must reach 2);
- `adaptive_work.small_multi_rows` (must be empty) and
  `adaptive_work.decode_fingerprint.duplicate_pos_values` (must be > 0);
- `adaptive_work.decode_submitted_ms` / `decode_router_wait_ms` /
  `decode_experts_host_ms` / `decode_cpu_compute_ms` / `decode_cpu_join_ms`
  min/median/p90 (feeds the D/R 64 ms model);
- `adaptive_work.admission` (the load-time
  `[Qwen4Exp] Memory Admission Accounting` line; single-session reserve);
- `failures` (must be empty) and `cleanup.exit_code` (must be 0);
  `launch.artifacts` sha256 = pair record.

Decision rule (section 5 discipline): this is a single-arm screening probe,
not an adoption gate. If it reproduces the predicted <= 21.1 tok/s `[估算]`,
the service question is answered (no batching gain today) and no A/B/B/A
receipt is owed; if it ever shows an aggregate above that band, a paired
A/B/B/A run with `tools/perf/ab_receipt.py` must be scheduled before any
config change. `--decode-batch-size` default stays 1 (default-value gate).
