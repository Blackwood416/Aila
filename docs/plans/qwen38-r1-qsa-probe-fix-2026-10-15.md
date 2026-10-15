# R1-A3: QSA probe fix and instrumentation (2026-10-15)

Role: A3 (kernel probe repair), zero-GPU/no-build recovery work for the K2 arm
of the R1 sprint. Base revision `a3ee21e`. Everything below is either measured
in this repo, measured with a local pwsh repro, or explicitly marked.

## 1. Reported symptom and raw evidence

`build-refactor/AilaQwen4ExpKernelTests.exe --qsa-probe` produced a 150-byte
log on 2026-10-14 23:34:

- `tmp/q4e-sprint-r1-20261014/kernel/qsa-cost-curve-A1.log` (150 B, UTF-16LE
  from the old runner): BOM + `Starting Qwen4Exp Kernel Tests on device:
  Intel(R) Arc(TM) A770 Graphics` + CRLF, and nothing else. The last line is
  complete, so no partial write was truncated mid-line.
- `tmp/q4e-sprint-r1-20261014/kernel/env-sha256.txt`: exe sha
  `9372B8FB363BC648DD13826F0FC28CD08EAA02BE8920772BB45BF8E929613B60`,
  `utc=2026-10-14T15:34:32Z`; log mtime 23:34:35 local -> the process produced
  its last byte of output three seconds after the meta record was written.
- The same exe ran K1 fine the same evening: `decode-gu-A1.log` (378 B) and
  `decode-gu-A2.log` (374 B) both contain `[decode-gu-bench] ... median_ms=
  0.123427/0.12343`, so the binary, driver stack and runner worked for
  `--decode-gu-bench`.
- Windows event log, last 3 days (Application + System, all providers): the
  only Application Error record is an unrelated `IntelGraphicsSoftware.exe`
  crash (2026-10-15 13:32:51, `CoreMessagingXP.dll`, `0xc000027b`). There is
  no faulting record for `AilaQwen4ExpKernelTests.exe` even though WER is
  demonstrably active on this host.
- Cost-model ledger note agrees with the observation:
  `docs/plans/qwen38-r1-cost-model-2026-10-14.md:441-444` ("the exe aborts
  right after the device banner, no [QSA COST] rows, 150 bytes").

## 2. Root cause analysis

### 2.1 Falsified: runner stderr handling did not kill the probe

The rev-1 runner used `$ErrorActionPreference='Stop'` with `*>` capture, and
the first stderr producer in this code path is real: `aila::log` writes Info
level messages to `std::cerr` by default (`src/profile/Profiling.cpp:20`
`g_min_level = LogLevel::Info`, `:24-26` `default_log_handler` -> `std::cerr`),
and the one-time `[JM]` banner fires on the first `attention_decode` call
(`src/ops/AttentionOps.cpp:1716-1729`).

Measured counter-evidence on this host (pwsh 7.6.7):

- `$PSNativeCommandUseErrorActionPreference` defaults to `False` here; with
  `$ErrorActionPreference='Stop'` a `cmd` child writing to stderr did not
  terminate the script, the stderr line was merged into the `*>` log (21-byte
  repro log contained `OUT1`, `OUT2` and `ERR1`), and execution continued.
- The 2026-10-14 K2 log contains no stderr-derived text at all, so the exe
  had not reached the first `attention_decode` before output stopped.

Conclusion: stderr-triggered script abort is NOT the failure mode; the runner
is still hardened in rev 2 (explicit flag off, separate streams) so the result
no longer depends on the PowerShell default.

### 2.2 Root cause of the diagnostic blind spot (proven)

The old probe body (`tests/ops/Qwen4ExpQsaProbe.hpp:123-197`, unchanged since
2026-09-30 20:58, last commit touch `5ea0171`) prints nothing between the
device banner and the first `[QSA CHECK]` line at `:176`. Everything before
that - host fill of 34.5M bf16 values, ten `Tensor::allocate` calls, five H2D
uploads, `qsa_compute_block_scores_compressed`, `qsa_select_topk_blocks`,
`attention_decode` and a 32768-token f64 CPU reference - is silent. Any stall,
driver-kill or operator kill in that window is observationally identical to
"starts and immediately exits".

The probe itself is not a regression: on 2026-09-30 it ran the full 9-length
curve three times with the same 2e-4 f64 threshold
(`docs/plans/qwen3.8-flash-next-qsa-gap-audit-2026-09-30.md:74`: "9 lengths
matched the CPU oracle, 2 heads checked at 2e-4, three GPU processes exited
normally, watchdog not triggered").

### 2.3 Ranked remaining hypotheses (to be discriminated by the fixed probe)

1. First-use kernel setup on a fresh process (strong candidate). The R1
   ledger documents a single 19-token first forward paying router 50.4 s +
   full-attn 19.8 s + linear 7.3 s + hc 1.2 s of one-time setup
   (`docs/plans/qwen38-r1-cost-model-2026-10-14.md:469`, `L10`). The K2 probe
   touches the QSA score/select kernels and the head_dim=256 exact-partial
   path for the first time in the process; with zero progress output this is
   indistinguishable from a hang.
2. A driver- or kernel-level stall at a shape that CI does not cover (the
   probe's 512-length first iteration drives `topk=512 > blocks=128`).
3. Operator kill after a silent no-output window (log mtime evidence: only
   three seconds of captured output).

The fixed probe discriminates all three: the last `[QSA PROBE]` breadcrumb
names the stalled stage, and `warm_ms=` reports the first-touch cost instead
of hiding it.

## 3. Fix (allowed files only)

### 3.1 `tests/ops/Qwen4ExpKernelTests.cpp`

| item | line | note |
|---|---|---|
| `<cstdlib>` / `<exception>` added | 19, 21 | `std::abort`, `std::get/set_terminate` |
| `run_qsa_route_cost_probe(Context&)` | 4905 | replaces `test_qsa_decode_cost_curve` for `--qsa-probe` |
| terminate handler (`[QSA PROBE TERMINATE]`) | 4911-4916 | no silent `std::terminate` |
| breadcrumbs `stage=host_fill/allocate/upload` | 4931, 4941, 4955, 4964 | flushed before each phase |
| `stage=warm_done ... warm_ms=` | 4977 | exposes first-use kernel cost |
| per-length `stage=score/select/verify_*` | 4997-5050 | breadcrumb before each GPU op |
| selector validation vs CPU oracle | 5013, 5022 | invalid count, ID mismatch -> throw |
| f64 reference + finiteness | 5033-5078 | unchanged 2e-4 protocol, 2 reference heads |
| `[QSA COST] {"length",...,"dense_attention_ms","routed_post_projection_ms","route_net_pct"}` | 5093-5097 | parser-ready schema |
| `[QSA PROBE FAIL] stage=... length=... what=...` + rethrow | 5103-5105 | failures are always logged |
| branch wired to the new probe | 5200 | `--qsa-batches`/`--qsa-profile` untouched |

Probe contract: lengths 2048 / 2051 (dense controls, route keeps every block
and only pays selector + scoring overhead), 2052 (first sparse length: 513
blocks -> 512 selected), 4096, 8192; 2 rounds x 15 synchronized reps per arm;
`routed_post_projection_ms` = compressed-K scoring + top-512 selection +
attention over the selected 512-block set (tail tokens included via the
kernel's own `cached_len % 4` rule).

`tests/ops/Qwen4ExpQsaProbe.hpp` is intentionally left untouched (the
2026-09-30 probe body stays for reference, now unreferenced from the CLI).

### 3.2 `tmp/q4e-sprint-r1-20261014/kernel/probe-K2-qsa-route.ps1` (rev 2)

- `$PSNativeCommandUseErrorActionPreference = $false` when present (line 26-27).
- `Start-Process` + `WaitForExit(ms)` (lines 51-64): raw stdout/stderr to
  `*.log` / `*.err.log`, real `ExitCode`, bounded timeout that kills the
  process and reports the last `[QSA PROBE]` breadcrumb.
- `qsa-probe-meta.txt`: exit code + wall seconds + UTC per arm (lines 67-73).
- Automatic parse of `[QSA COST]` rows + verdict (lines 76-112): >=3% at BOTH
  4096 and 8192 -> cross-head lever; <1% at either -> selector/scoring lever;
  otherwise gray, repeat once (A/A).
- `--qsa-batches` stays non-fatal (lines 115-124); GPU-exclusivity check and
  `env-sha256.txt` are unchanged.
- Static checks: PowerShell parser reports 0 errors; the JSON/verdict path was
  exercised offline against a synthetic log (4096: 19.93%, 8192: 3.92% ->
  "cross-head lever").

## 4. Verification command for the coordinator

```powershell
Get-Process AilaWorker,AilaServer -ErrorAction SilentlyContinue   # must be empty (charter 5.4)
pwsh -NoProfile -File build.ps1 -BuildDir build-refactor          # rebuild AilaQwen4ExpKernelTests
pwsh -NoProfile -File tmp/q4e-sprint-r1-20261014/kernel/probe-K2-qsa-route.ps1
```

Read, in order:

1. `qsa-cost-curve-A1.err.log` - expected: one `[JM] mode=...` Info banner;
   anything else there is new evidence.
2. `qsa-cost-curve-A1.log` - `[QSA PROBE]` breadcrumbs, then `[QSA CHECK]` and
   `[QSA COST]` rows; `[QSA PROBE FAIL]` or a missing breadcrumb after
   `stage=warm_done`/`stage=score` names the exact faulting phase.
3. `qsa-probe-meta.txt` + `env-sha256.txt` - exit code, wall time, exe hash as
   the A-side receipt.

If hypothesis 2.3-1 holds, expect large `warm_ms` on the first length and a
clean run afterwards; if it dies silently, the last breadcrumb localizes it.

## 5. Decision table (unchanged from the K2 README)

| condition on `[QSA COST]` | reading | lever for the K ranking |
|---|---|---|
| route_net_pct >= 3% at BOTH 4096 and 8192 | route net-positive | K/V cross-head sharing variant (kill the 12x re-read) |
| route_net_pct < 1% at either | route net-negative | selector/scoring work (route overhead not amortized) |
| 1-3% | gray | repeat the arm once (A/A) before ranking |

2048/2051 are dense controls only; sparse rows must come from length >= 2052.

## 6. Non-claims and residual risk

- No build and no GPU run were executed by this role; compile/link and runtime
  behavior of the new probe are unverified until the coordinator builds it.
- Brace/paren/bracket balance of the edited translation unit was checked
  offline (5312/839/660 pairs balanced), and the 16-argument
  `attention_decode` call shape matches `src/ops/Ops.hpp:348-360` defaults, but
  this is not a substitute for a compiler.
- The 2e-4 f64 threshold and the 1/16 attention scale are inherited from the
  2026-09-30 protocol, not re-derived.
- `README.md` in the probe directory still lists the old output filenames;
  rev 2 keeps `qsa-cost-curve-A1.log` for stdout and adds `.err.log` +
  `qsa-probe-meta.txt`, so no file list change is required.
