# R1-P3 C1: CPU verify-window expert merge (implementation note)

Date: 2026-10-16. Base: HEAD `2e25fb8` plus this uncommitted change set.
Scope: `AILA_Q4E_VERIFY_CPU_MERGE` (default 0) for the 2..4-row MTP verify
window cold CPU service.
Method: CPU-only implementation and review. No build, no GPU run, no commit.
Only these files were touched:

| file | lines |
|---|---|
| `src/ops/Qwen4ExpCpuExpertOps.hpp` | +27 |
| `src/ops/Qwen4ExpCpuExpertOps.cpp` | +102 / -13 |
| `src/models/Qwen4ExpExpertExecutor.hpp` | +8 |
| `src/models/Qwen4ExpExpertExecutor.cpp` | +85 / -5 |

## 1. Mechanism

The window-wide cold job already carries a strict row-major partition of
(row, expert) pairs (`Qwen4ExpExpertExecutor.cpp:1064-1124`, built by the G1b
batch path of `Qwen4ExpBnb4Backend.cpp:1341-1366`). The merge adds, inside
`StartArenaColdRows` and only when the gate is on:

- a group map built over those pairs, keyed by expert id
  (`Qwen4ExpExpertExecutor.cpp:1132-1185`): group ids assigned by first
  appearance, member lists flattened and kept contiguous, groups with
  `member_count > 4` or `groups == task_count` are rejected (mergeable=false);
- three new fields on the worker job (`Qwen4ExpExpertExecutor.cpp:83-89`):
  `groups[]` (`ops::cpu::CpuExpertGroup`, `slot` + member span),
  `group_members[]` and `group_count`;
- a dispatch split in the worker (`Qwen4ExpExpertExecutor.cpp:494-508`):
  `group_count > 0` calls the new
  `ops::cpu::cpu_expert_batch_forward_groups`, otherwise the byte-identical
  `cpu_expert_batch_forward` call is used as before;
- the grouped kernel (`src/ops/Qwen4ExpCpuExpertOps.cpp:1696-1750`) validates
  the map (members cover every pair exactly once, tokens in range, NF4-Q8
  backend family) and forwards to `nf4_q8_batch_forward` with the group map
  (`:907-915`).

Inside `nf4_q8_batch_forward` the two tile loops were re-expressed so that one
(task, tile) body is shared:

- gate: body `src/ops/Qwen4ExpCpuExpertOps.cpp:965-1049`, ungrouped loop
  `:1050-1064` (one call per task per tile), grouped loop iterating
  `group x tile` and calling the body once per member (`:1054-1061`);
- down: body `:1082-1131`, grouped loop `:1132-1145`.

Only the loop headers/tails and the two `continue` -> `return` conversions
changed; every per-(task, tile) statement appears as a context line in
`git diff` (see section 4).

## 2. The read-only counter (f) and the NO-GO gate

When the gate is on, `StartArenaColdRows` accumulates per build:

- `pairs` = original task count of one layer batch (the same number the
  ungrouped job would dispatch);
- `groups` = merged task count;
- `f` = 1 - groups / pairs for the build, and `cum_f` over all builds.

Log line (throttled to the first 4 builds and every 48th build, i.e. about
once per window, `Qwen4ExpExpertExecutor.cpp:1171-1183`):

    [Q4E VerifyCpuMerge] build=<n> layer=<l> pairs=<p> groups=<g> f=<x>
    cum_pairs=<P> cum_groups=<G> cum_f=<X>

Reading rules:

- `cpu_tasks` in `[Q4E AdaptiveWork]` stays 587 per window by design (the pair
  table and therefore `stats_.cpu_tasks_dispatched` are unchanged); the merge
  count is `groups`, not `cpu_tasks`. Do not read "cpu_tasks unchanged" as
  "merge did not engage" - read the `[Q4E VerifyCpuMerge]` lines.
- If the flag is set without `AILA_Q4E_VERIFY_CPU_BATCH=1` (the per-row G1
  path), one line says so and the merge is inert:
  `[Q4E VerifyCpuMerge] inactive: window job requires
  AILA_Q4E_VERIFY_CPU_BATCH=1` (`Qwen4ExpExpertExecutor.cpp:915-925`).

GO/NO-GO on f (per the P3 economics estimate, saving ~= f * 16.7 ms of
window, band 0.5..1.0 of that):

| f | expected window saving | verdict |
|---|---|---|
| < 0.10 | < 1.7 ms [est] | NO-GO |
| 0.30 | 5.0 ms [est] | NO-GO (below the 5 ms bar) |
| 0.50 | 8.4 ms [est] | GO, window ~104 ms |
| 0.80 | 13.4 ms [est] | GO, window ~99 ms |

Independence null for scale: if two adjacent rows picked 10 of 256 experts
independently, the expected shared count is 10*10/256 ~= 0.39 per layer, i.e.
~19 pairs over 48 layers, f ~= 0.03. The measured f is therefore a direct
statement about router correlation between a token and its draft token, not a
tautology.

## 3. Bit-exactness argument

1. The merged kernel keeps `tasks[c]` as the owner of both the activation row
   (`tasks[c].token`) and the output rows (`h + c*intermediate`,
   `out + c*hidden`). Every member executes the identical body it would
   execute in the ungrouped loop, with identical arguments: same slot pointer,
   same `values`/`scales`/`output` addresses, same dot function, same
   accumulation order inside a task.
2. Writes are disjoint per task (each (task, tile) writes its own rows of `h`
   and `out`), and no shared mutable state is written inside the bodies, so
   the interleaving across independent tasks (grouped vs task-major) cannot
   change any byte. The only schedule-visible side effect is
   `mark_a64_specialize_enabled(mode)`, a report-only latch with the same
   value for all members of a group.
3. The quantize stage, the down stage inputs (`intermediate` indexed by task)
   and the coordinator-side publishing (`JoinArenaColdRowsRow`,
   `cpu_out_host_ + c*hidden`) are unchanged, so rank-span uploads stay
   byte-identical.
4. The extracted bodies were verified by `git diff`: the gate/down per-tile
   bodies show up as context lines; the only changes are the loop
   headers/tails and `continue` -> `return` in the two A64 dispatch branches.
5. When `AILA_Q4E_CPU_VERIFY_*` shadow checks are enabled, their reference
   call still runs the ungrouped `cpu_expert_batch_forward` over the same
   pair table and memcmp's against the merged output, so the flag turns the
   merge into a self-checking change in those arms.

Fallbacks (all fail-safe, no silent reschedule):

- backends outside the NF4-Q8 family: exact ungrouped call
  (`src/ops/Qwen4ExpCpuExpertOps.cpp:1736`);
- `AILA_Q4E_CPU_PER_EXPERT_DAG=1` (default off): falls back to the ungrouped
  DAG schedule rather than the tile loops (`:943-951`);
- `groups == task_count` (no duplicate expert in the window): group_count
  stays 0 and the ungrouped call is used;
- invalid maps (span/coverage/token) throw before any compute.

## 4. A/B/B/A arm (what to run)

The merge lives in the window-wide job, so both arms must enable the batch
path; the merge is the only delta between A and B:

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # must be empty
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py s3-a1 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_VERIFY_CPU_BATCH=1 AILA_Q4E_WORK_SUMMARY=1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py s3-b1 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_VERIFY_CPU_BATCH=1 AILA_Q4E_VERIFY_CPU_MERGE=1 AILA_Q4E_WORK_SUMMARY=1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py s3-b2 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_VERIFY_CPU_BATCH=1 AILA_Q4E_VERIFY_CPU_MERGE=1 AILA_Q4E_WORK_SUMMARY=1
python tmp/q4e-resident-speed-20261009/run_mtp_ab.py s3-a2 AILA_Q4E_SERVER_MTP=1 AILA_Q4E_VERIFY_CPU_COLD=1 AILA_Q4E_VERIFY_CPU_BATCH=1 AILA_Q4E_WORK_SUMMARY=1
```

Judgement (pre-registered):

1. `[Q4E VerifyCpuMerge]` lines present in B arms and `cum_f` read
   (NO-GO if `cum_f < 0.3`).
2. Window: count=2 `submitted_ms` median <= 106.7 ms (5% better than the
   112.2 ms measured base) or continue rate >= 16.0 tok/s; B median over the
   A median with 2/2 pairs above the A/A floor (<= 1%).
3. `content_sha256` identical in all four arms (bitwise output claim).
4. Secondary reads: `cpu_compute_ms` (expect -0.5..1.0 x f x 16.7 ms),
   `experts_host_ms`, acceptance `windows/accepted` unchanged.

Note the batch path's own measured cost (-1.0% vs the per-row G1 arm,
s2x A/B/B/A) rides along in both arms; this arm measures merge-on-top-of-batch.

## 5. Verification done here (no build, no GPU)

- Syntax-only, exact flags from `build-refactor/compile_commands.json`
  (icx-cl 2026.1, `vcvars64` from VS 18, `-fsycl -fsyntax-only`):
  `Qwen4ExpCpuExpertOps.cpp` -> exit 0; `Qwen4ExpExpertExecutor.cpp` -> exit 0.
- Negative control: the same command on a temp copy of the ops file with an
  injected syntax error -> 3 errors, exit 1, so the check parses and reports.
- `git diff` review of the two extracted bodies (context-only).
- Not verified here (needs the user's build + arm): f value, window timing,
  content sha, and any compile warnings from the real build.

## 6. Risks / open items

1. f unknown: < 0.3 kills the lever (the first B arm decides).
2. Cache effect size: the saving is L1/L2 reuse of the packed tile bytes; if
   the kernel is compute-bound rather than stream-bound, only the stream part
   of the duplicate task disappears (band 0.5..1.0 x f x 16.7 ms [est]).
3. Tile-granular member iteration only shares within a (group, tile) window
   (~41 KB gate, ~61 KB down per member pass); finer row-level sharing would
   need dot-level fusion and is out of scope.
4. `groups == task_count` windows bypass the merged schedule entirely (f=0),
   which is correct but means a single low-overlap window cannot judge the
   lever; read `cum_f` instead.
5. If a build warning appears for the guarded `groups[unit]` dereference in the
   grouped loops, hoist it behind the `group_count > 0` branch (cosmetic).

## 7. Sources

- Code: `src/ops/Qwen4ExpCpuExpertOps.hpp:22-30,74-90`;
  `src/ops/Qwen4ExpCpuExpertOps.cpp:907-951,957-1065,1077-1145,1696-1745`;
  `src/models/Qwen4ExpExpertExecutor.cpp:83-89,494-508,915-925,1064-1185`;
  `src/models/Qwen4ExpExpertExecutor.hpp:378-385`.
- Baseline and arithmetic: `docs/plans/qwen38-r1-mtp-width-economics-2026-10-16.md`
  (f saving model, 112.2 ms base, 106.7 ms bar);
  `tmp/q4e-resident-speed-20261009/s2x-*` (batch-arm cost rode along).
