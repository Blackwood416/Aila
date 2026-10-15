# R1-G1: MTP verify window - CPU cold experts (AILA_Q4E_VERIFY_CPU_COLD)

Date: 2026-10-15. Scope: Qwen4Exp decode path, MTP verify windows only.
New gate: `AILA_Q4E_VERIFY_CPU_COLD` (read_flag, default 0 = off).
Files: `src/models/Qwen4ExpBnb4Backend.cpp`, `src/models/Qwen4ExpExpertExecutor.cpp`,
`src/models/Qwen4ExpExpertExecutor.hpp`. No build, no GPU run, no commit in this delivery;
both TUs pass `icx-cl -fsyntax-only` (exit 0, no .obj; script in
`tmp/q4e-sprint-r1-20261014/mtp-verify-cpu/syntax-check.cmd`).

## 1. Current state and the exact closure points (why cpu_compute=0)

### 1.1 Measured today (retained build, MTP T2, width 2)

`tmp/q4e-resident-speed-20261009/mtp-reanchor-b1-worker.log:23`
(one line per verify window, all 48 layers aggregated):

```
pos=19 count=2 submitted_ms=253.316 router_wait_ms=135.805 experts_host_ms=87.131
hc_host_ms=5.469 linear_host_ms=4.775 full_host_ms=3.888 head_host_ms=1.073
cpu_compute_ms=0.000 cpu_join_ms=0.000 cpu_tasks=0
hits=365 misses=595 staged=595 h2d_mb=1438.110
```

- misses == staged == 595 (all 48 layers x 2 rows x 10 ranks = 960 ranks; 595 cold),
  so every cold rank is moved over PCIe instead of being served from RAM;
- `cpu_tasks=0` / `cpu_compute=0` / `cpu_join=0`: the CPU worker is never asked;
- window wall submitted_ms runs 229-264 ms over lines 23-28
  (`tmp/q4e-resident-speed-20261009/mtp-reanchor-b1-worker.log:23-28`).

T1 contrast (`tmp/q4e-sprint-r1-20261014/d1-smoke-worker.log:26`, count=1):
`misses=209 cpu_tasks=209 staged=0 h2d_mb=0.000 cpu_compute_ms=21.185 cpu_join_ms=16.498`
- T1 puts every per-layer miss (<= `max_cpu_credits_`) on the CPU and never pays PCIe.

End-to-end from the same harness runs
(`tmp/q4e-resident-speed-20261009/mtp-reanchor-{a1,b1}/result.json`,
`context-1500-continue`, `after_first_content_tokens_s`):

| arm | MTP | continue tok/s | completion_tokens | content sha (12) |
|---|---|---|---|---|
| g1-reanchor-a1 | off (T1) | 17.17 | 128 | d76fb23bc060 |
| g1-reanchor-b1 | on | 5.88 | 84 | a2501de8b231 |

`[Q4E Server MTP] windows=137 accepted=88` (b1 worker log:176) -> p ~= 0.642 for this
prompt; `(1+p)/W = 1.642/0.2533 = 6.48 tok/s` matches the measured 5.88-6.53.
So today MTP is ~3x SLOWER than T1 on this build; the verify window is the reason.

### 1.2 Why no CPU path is reachable for count=2 (line numbers, post-change tree)

1. `Qwen4ExpBnb4Backend.cpp:1222-1224`: the only multi-row CPU route inside the
   adaptive decode (`arena_verify`) is gated by
   `expert_executor_.CanExecuteArenaVerifyRows(...)`.
2. `Qwen4ExpExpertExecutor.cpp:1262-1272`: `CanExecuteArenaVerifyRows` returns
   false unless `AILA_Q4E_ARENA_VERIFY_ROWS=1`. That env is NOT in the MTP profile
   (`docs/plans/qwen38-server-mtp-2026-10-07.json` worker_environment; it carries
   `VERIFY_CPU_ROWS=4`, `VERIFY_ASYNC_ROUTE=1`, `SMALL_ROWS_DECODE=1`,
   `SMALL_ROWS_HC=1`, `CPU_MAX_EXPERTS=10`, `COLD_MODE=2`, `CPU_ASYNC_ROUTE=1`).
3. `Qwen4ExpBnb4Backend.cpp:2329-2330` computes `verify_cpu_rows` but its only
   consumer `cpu_verify_async` (`:2343`) requires `!use_adaptive_pipeline_`
   (same for `cpu_async_route` `:2336`); the legacy pipeline was removed
   (`:2609` throws without the adaptive pipeline), so this cluster is inert
   on every production arm.
4. The per-token loop's CPU branch was `count==1 && cpu_overlap_enabled()`
   (now `:1344-1345`). For count=2 it never fired, so `cpu_count=0` and the whole
   miss list went to `adaptive_cache_.StageColdExpertsAsync(...)` (`:1347-1355`)
   -> `staged=595`, `h2d_mb=1438` (Section 1.1).

Conclusion: for count>=2 the adaptive pipeline had NO reachable CPU cold path.
The window's cold experts were always GPU-staged, unlike T1.

## 2. Variant design and the bit-parity argument

### 2.1 Gate and shape

- New env `AILA_Q4E_VERIFY_CPU_COLD` (default 0;
  `Qwen4ExpBnb4Backend.cpp:1282`).
- Active only for `verify_rows && 2<=count<=4 && count<=max_cpu_tokens()`
  (i.e. `options.decode_verification`, `Qwen4ExpBnb4Backend.cpp:1283-1284`;
  MTP sets it in `Qwen4ExpMtpRuntime.cpp:109`).
- One `StageResidentInput(x_base, count)` per layer (`:1285`) so every row's job
  indexes the same pinned image (`Qwen4ExpExpertExecutor.cpp:723-730`).
- Per row, the miss list is resolved with the SAME single-row arena batch as T1:
  `StartArenaCold(..., miss_count, x_base, t, count)` (`:1349-1350`);
  `x_base == x_t` when `count==1`, so the count==1 call is byte-for-byte the old one.
- `StartArenaCold` extension (`Qwen4ExpExpertExecutor.cpp:891-915`,
  declaration `Qwen4ExpExpertExecutor.hpp:193-198`): added `row_token`/`tokens`
  (defaults keep every existing caller unchanged). The job keeps
  `tokens=count` and stamps `tasks[c]={source,row_token}`.

### 2.2 Why each verify row's expert stage is byte-identical to a sequential T1 decode

1. Same cap: `cold_count_=min(miss_count, max_cpu_credits_)`
   (`Qwen4ExpExpertExecutor.cpp:906`), i.e. `AILA_Q4E_CPU_MAX_EXPERTS=10` per row,
   exactly the T1 split; overflow (if any) still goes to the unchanged GPU staging
   path (`Qwen4ExpBnb4Backend.cpp:1355-1361`, `ping_pong_index_` toggle included).
2. Same sources: `arena.GetExpertSlotHost(layer, ids[c])` (`:911-915`) - the same
   native RAM slots T1 would use.
3. Same CPU arithmetic per (row, expert) task: `cpu_expert_batch_forward` computes
   each task independently - gate/down tiles read `input + tasks[c].token*kHiddenDim`
   and write `h + c*...`/`out + c*...`
   (`Qwen4ExpCpuExpertOps.cpp:818-903`), and the activation quantization is a
   per-token loop (`:905-928`). Task count and job token count do not enter any
   task's arithmetic, so task (row t, expert e) in a 2-row job equals the same task
   in a 1-row job bit for bit. Same backend selection (`resident_prepared=true` +
   the same `AILA_Q4E_CPU_NF4_Q8*` flags -> `Nf4Q8Activation64Avx2`).
4. Same staging event: `job.activation = resident_input_event_` (the copy staged in
   2.1) vs T1's per-layer `StageResidentInput(buf_.x, 1)` - same bytes for the same row.
5. Same merge: per row `JoinResidentCold(down)` uploads the f32 rank rows into the
   same per-row accumulator, and `moe_ordered_accumulate_down_rows` merges with
   `resident_weights_` (`Qwen4ExpBnb4Backend.cpp:1405-1416`) - the exact
   kernel/order T1 uses; CPU rows are read back only after the merge event
   (`FinishResidentCpuRead`).

### 2.3 Runtime fallback instead of a hard failure

New per-layer guard `cpu_cold_ok` (`:1328-1331`): if any routed miss has no native
RAM owner (`host_arena_.GetExpertPointer == nullptr`, the same ownership test the
arena-verify cover check uses at `:1229`), the layer falls back to the unchanged
all-GPU staging path instead of throwing inside `StartArenaCold`. count==1 keeps
its historical throw semantics.

### 2.4 What this does NOT fix

The window's non-expert stages (attention/GDN/HC/shared/router/LM head) are still
the count>=2 batched path; T1 bit-parity for the whole window additionally needs
the `VERIFY_SMALL_ROWS` alignment set (`docs/plans/qwen38-mtp-align-2026-10-11.md`
B4-3), which measured 6.28 tok/s and is out of this arm's scope. This change makes
the expert stage (595 of 960 ranks per window) T1-identical - a necessary step,
not the whole alignment gate.

## 3. Verification arms (A/B/B/A, GPU mutex, no fixed sleeps)

```powershell
Get-Process AilaWorker,AilaServer,Aila -ErrorAction SilentlyContinue   # must be empty
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py g1-a1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py g1-b1 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_WORK_SUMMARY=1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py g1-b2 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_WORK_SUMMARY=1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py g1-a2
```

`run_mtp_ab.py` clears AILA_*/SYCL_*/UR_* and re-applies the MTP profile, so the
b arms differ from a arms only by `SERVER_MTP=1` plus the new gate
(`tmp/q4e-resident-speed-20261009/run_mtp_ab.py:15-27`).

Read (in order):

1. Worker log `[Q4E AdaptiveWork]` lines with `count=2` (one per window; exclude the
   `count=19`/wide prefill lines): median `submitted_ms`.
   - Strong (20+ tok/s): <=88 ms at the brief's p=0.748 (>=18.7 tok/s at this run's
     p=0.642). GO (19+ tok/s): <=92 ms at p=0.748, <=86 ms at p=0.642.
   - Reject: >100 ms (<=17.5 tok/s even at p=0.748); >=2/3 windows must qualify.
   - `staged` must be 0 (or only the rare per-row >10 overflow) and `h2d_mb` ~= 0;
     `cpu_tasks` ~= `misses`; `cpu_compute_ms` / `cpu_join_ms` > 0.
2. `result.json` per request `after_first_content_tokens_s` (continue):
   target >=19 tok/s (a-arms stay ~17.2).
3. `[Q4E Server MTP] windows=N accepted=M` (destructor line): report p=M/N.
4. Text: compare continue-request `content_sha256` g1-b1 vs g1-b2 (same-flag
   repeatability) and against the pre-fix b1 sha `a2501de8b231...`
   (`tmp/q4e-resident-speed-20261009/mtp-reanchor-b1/result.json`) to attribute any
   content change to this gate. Full T1-parity text is NOT expected on this arm
   (Section 2.4); the byte-parity arm is the existing alignment harness:

```powershell
$env:AILA_Q4E_MTP_ALIGN_T1='decode'; $env:AILA_Q4E_VERIFY_CPU_COLD='1'
$env:AILA_Q4E_VERIFY_SMALL_ROWS='1'; $env:AILA_Q4E_VERIFY_GDN_ORDER='1'
$env:AILA_Q4E_VERIFY_DECODE_PREPARE='1'; $env:AILA_Q4E_VERIFY_DECODE_ATTN='1'
build-refactor\AilaQwen4ExpToleranceTests.exe --mtp-align C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2
```

read `logit_different`, per-row `first_diff_layer`, `[MTP ALIGN STATE]` rows
(`tests/models/Qwen4ExpMtpAlign.hpp`; baseline reproduction set per
`qwen38-mtp-align-2026-10-11.md` B4-3). Run it once WITHOUT the new gate as control and
once WITH it; the diff must not regress, and the expert-stage divergence (if any)
must disappear.

## 4. Risks / open items

1. Window wall may land ~82-120 ms (two serialized per-row CPU joins per layer).
   If `submitted_ms` stays >100 with `staged=0`, the per-row serialization is the
   binding cost; the fast follow is the single 2-row CPU batch
   (`StartArenaVerifyRows` machinery) plus an explicit >cap overflow staging path -
   faster join but must reproduce the per-row cap to stay T1-identical.
2. CPU becomes the floor: T1 cpu_compute is 21.2 ms/token with 209 tasks
   (`d1-smoke-worker.log:26`); a window carries ~2x that in ~595 tasks. 18 CPU
   threads should absorb it, but `cpu_join_ms` growth is the early warning.
3. Bit-parity argument is constructional (Section 2.2), verified by reading the
   per-task DAG; the runtime proof is the `--mtp-align` arm. If it diverges at the
   expert stage, suspect slot/back-end differences, not arithmetic.
4. Cross-row buffer reuse stays inside the existing contract: the worker waits
   `cpu_output_reader_event_` (previous merge) before rewriting `cpu_out_host_`,
   and `StartArenaCold` waits `prev_cpu_h2d_ev_` (`Qwen4ExpExpertExecutor.cpp:898`).
5. Do NOT combine with `AILA_Q4E_ARENA_VERIFY_ROWS=1` (that early-return path wins
   per layer and has different routing semantics) or with `VERIFY_SMALL_ROWS=1`
   (measured 6.28 tok/s) in the same arm.
6. Zero new allocations; `verify_down_/verify_weights_` are untouched by this path.
7. First build after this change compiles the three touched TUs; syntax-only has
   passed, but the link and the arm are unverified by design (discipline).

## 5. Sources

- `tmp/q4e-resident-speed-20261009/mtp-reanchor-b1-worker.log:23-28,176`
- `tmp/q4e-sprint-r1-20261014/d1-smoke-worker.log:26`
- `tmp/q4e-resident-speed-20261009/mtp-reanchor-{a1,b1}/result.json`
- `docs/plans/qwen38-server-mtp-2026-10-07.json` (worker_environment)
- `docs/plans/qwen38-r1-decode-20-2026-10-15.md` (T1 table, MTP arithmetic)
- `docs/plans/qwen38-mtp-align-2026-10-11.md` (B4-3/B4-6 alignment set and debt)
- `docs/plans/qwen38-arena-verify-2026-10-11.md` (arena verify design)
