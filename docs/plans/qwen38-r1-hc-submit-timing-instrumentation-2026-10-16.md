# Q4E HC submit timing instrumentation (2026-10-16)

Implements the "single cheapest experiment" from
`docs/plans/qwen38-r1-hc-submit-attribution-2026-10-16.md` section 5: an
env-gated `steady_clock` around the HC decode `ctx.queue().submit(...)` calls,
printed on the existing `[Q4E AdaptiveWork]` line next to `hc_host_ms`. Host
side only; the decode arm itself is the main agent's step. Tree base `7baa063`;
the change also rides over the pre-existing uncommitted
`AILA_Q4E_RUNTIME_MIN_CAP` hunk in the same backend file (untouched here).

## 1. Gate

- `AILA_Q4E_HC_SUBMIT_TIMING` (int, default `0`): read once via
  `aila::env::read_int_raw` in `src/ops/HyperConnectionOps.cpp`
  (`hc_submit_timing_enabled`), same read-once pattern as
  `AILA_Q4E_JOIN_RELEASE` (`Qwen4ExpBnb4Backend.cpp:1311`). Any non-zero value
  enables it.
- Gate off (default): the helpers only test a cached `bool`; no `steady_clock`
  read is executed and no counter is touched, so the submit path keeps the
  production codegen modulo one predictable branch.

## 2. What is measured (and what is not)

Interval = the `ctx.queue().submit(...)` call only: started immediately before
the call, ended immediately after it returns.

- Included: `sycl::queue::submit` itself (the UR/L0 submit) plus the
  command-group function body that SYCL runs inside `submit`
  (`local_accessor` construction, kernel argument marshalling).
- Excluded: everything before the call (range/`num_wgs` arithmetic, pointer
  casts, `inv_groups`, capture setup) and everything after it (event handling,
  `hsplit_add`, stage markers).

This is the submit-only split the attribution doc asked for (section 5.1:
"wrap each `ctx.queue().submit(...)`"); no separate "total = prep + submit"
timer was added because the plan's read map (`sum(submit_us)` vs `hc_host_ms`)
is defined on the submit-only interval. Counters are updated on the submission
thread (worker scheduler thread, or CLI thread) and read back on the same
thread: no atomics, no locks.

| id | log key prefix | op | site |
|---|---|---|---|
| 0 | `hc_submit_norm*` | single-row fused inject+add+norm (P1 pending path, inventory A1) | `try_hc_decode_inject_add_and_grouped_rms_norm` submit |
| 1 | `hc_submit_normfb*` | multi-row fused inject+add+norm (inventory B1) | `hc_inject_add_and_grouped_rms_norm` submit |
| 2 | `hc_submit_down*` | down/act+inj (inventory A2) | `hc_decode_down_inject_impl` (all 3 work-group instantiations) |
| 3 | `hc_submit_up*` | up/mix (inventory A3) | `hc_decode_up_mix_impl` + `hc_decode_up_mix_streams_impl` |

The plain `grouped_rms_norm` submit (no-pending-residual arm) is intentionally
not instrumented: it is not one of the HC submit sites enumerated in the
attribution doc (section 5.1) and is not on the e10 T1 decode path.

## 3. API

`src/ops/HyperConnectionOps.hpp` gains `ops::HcSubmitTiming` (members
`norm_pending`, `norm_fallback`, `down`, `up`; each
`count`/`total_ns`/`min_ns`/`max_ns`) and two accessors:

- `ops::HcSubmitTimingTotals()` - cumulative since process start, never reset.
- `ops::HcSubmitTimingTakeWindow()` - stats since the previous call; the
  window is then reset (cumulative totals untouched). The backend calls this
  once per forward at the `[Q4E AdaptiveWork]` print point, so the printed
  values are exactly this forward's submits (identical whether you diff two
  `Totals()` snapshots or take the window).

## 4. Log fields (appended to `[Q4E AdaptiveWork]`)

16 new fields; all zero when the gate is off:

```
hc_submit_norm_us hc_submit_norm_n hc_submit_norm_min_us hc_submit_norm_max_us
hc_submit_normfb_us hc_submit_normfb_n hc_submit_normfb_min_us hc_submit_normfb_max_us
hc_submit_down_us hc_submit_down_n hc_submit_down_min_us hc_submit_down_max_us
hc_submit_up_us hc_submit_up_n hc_submit_up_min_us hc_submit_up_max_us
```

- `*_us` = sum of submit time in the forward (compare against `hc_host_ms`;
  at T1 the counts are expected 48/48/97 x `count` for
  norm-pending/norm-fallback+down/up-ish - read `*_n` rather than assuming).
- `*_n` = submit count in the window (sanity: ties to the 291/token budget).
- `*_min_us`/`*_max_us` = window extremes, i.e. the bimodality read from the
  attribution doc: cold ~40-60 us vs warm ~6 us per submit.

Read map (attribution doc section 5): ~5-7 us/submit => the 5x is **not**
inside `sycl::queue::submit`, so the block attribution (C3) plus "no candidate
in our code" (C4) is the answer; ~20-35 us => the 5x is the UR/L0 submit under
production queue state (C1/C2/C5), and the per-op split separates the norm
submit (C2) from all three (C1/C5).

Arm env: the e10 bundle plus `AILA_Q4E_HC_SUBMIT_TIMING=1` (the fields print
under the existing `AILA_Q4E_WORK_SUMMARY=1` gate of the line).

## 5. Default behavior

Unchanged. With the variable unset: no clock reads, counters stay zero, the
only observable difference is that the `[Q4E AdaptiveWork]` line carries the
16 new fields with `0.000`/`0` values.

## 6. Build verification

Both builds green on 2026-10-16 (Asia/Shanghai), no GPU test run from this
change; the tree also carries the owner-approved default flips
(`AILA_Q4E_PREFILL_EXPERT_XMX` on + MIN_M=256/ROWS=4096,
`AILA_Q4E_STAGE_COPY_PARALLEL` on) in the backend file.

- Runtime: `pwsh -NoProfile -File build.ps1 -BuildDir build-refactor` exit 0.
  Race guard: `HyperConnectionOps.cpp.obj` 21:41:17 and
  `Qwen4ExpBnb4Backend.cpp.obj` 21:41:38 are both later than every touched src
  mtime (hpp 21:33:53, ops cpp 21:34:53, backend cpp 21:39:40), so the
  artifacts below contain the instrumentation plus the default flips.
- Artifacts: `build-refactor/AilaWorker.exe` **2026-10-16 21:43:00**
  (12,633,600 bytes); `Aila.exe` 21:43:01; `AilaShared.dll` 21:41:43;
  `AilaInference.dll` 21:41:38.
- Server: `pwsh -NoProfile -File tools/server/build.ps1 -SkipDependencies`
  exit 0; `build-refactor-server/AilaServer.exe` **2026-10-16 21:44:06**
  (2,627,072 bytes; the server never links the SYCL ops, this is the
  discipline build).
- Instrumentation landed as commit `ffb2b08`; the 16 backend log fields rode
  in `6f9a41e` together with the default flips.

## 7. Files changed

- `src/ops/HyperConnectionOps.hpp` (timing struct + accessors)
- `src/ops/HyperConnectionOps.cpp` (read-once gate, counters, 4 submit sites)
- `src/models/Qwen4ExpBnb4Backend.cpp` (window take + 16 appended log fields)
- this file
- `tmp/q4e-structural-20261016/probe_submit_shape.cpp` + `probe_submit_shape.exe`
  + `run_shape_probe.ps1` (section 8; `tmp/` is not tracked)

## 8. Structural submit-shape probe (2026-10-16)

Follow-up for the 21-26 us/submit attribution. hct-a1 receipt: `hc_submit_norm`
~21 us and `hc_submit_normfb` ~26 us per call in the production worker (min
13.4-17.0, max 39-68 us) while the bare `submit_probe.cpp` measured ~6.5 us per
submit in the same L0 environment. The probe re-measures that gap as a function
of the CG shape alone and then repeats it under simulated CPU contention:
`tmp/q4e-structural-20261016/probe_submit_shape.cpp` ->
`probe_submit_shape.exe` (2026-10-16 21:57:11, 143,872 bytes). Compile + link
verified; deliberately not executed here (GPU is scheduled by the main agent).

| arm | shape |
|---|---|
| `A_plain` | `q.parallel_for(range<1>(1024), scalar body)`; the 6.5 us control |
| `S_sync` | `q.single_task([] {})`; pure submit call with no CG work |
| `B_slm_empty` | `submit` + `local_accessor<float,1>(256)` + `nd_range<1>(4*256,256)` + `[[reqd_sub_group_size(16)]]`, near-empty body |
| `C_down_slm` | B's CG shape + the real down captures (~12 values) + the full `hc_decode_down_inject_impl` body, SLM kept |
| `R_down_real` | exact production down CG: same captures and body, no SLM |

Phases: `idle`, then (default; skipped by `--cg-only`) `loaded18` = 18
in-process CPU threads each streaming a private 10 MB buffer read+write, the
18-expert-thread / fill-thread contention model. `A->B->C/R` prices hypothesis
(a) "the HC CG shape itself is expensive"; `idle->loaded18` prices hypothesis
(b) "the loaded production machine is expensive".

`--cg-only` adds two extra pieces: `cg_no_kernel` (the SLM + capture CG without
`parallel_for`; prints `unsupported` if the runtime rejects zero-command command
groups) and `cg_capture_host` (host-only closure construction with the same
capture pack, pricing the capture copy the call site does before `submit`,
which the arm intervals exclude by construction).

Run (GPU; refuses to start when a worker/server holds the card):

```
pwsh -NoProfile -File tmp/q4e-structural-20261016/run_shape_probe.ps1 -Log tmp/q4e-structural-20261016/shape-L20.log
tmp/q4e-structural-20261016/probe_submit_shape.exe --n 256 --warmup 64 --wait-every 32
```

Defaults `--n 256 --warmup 64 --wait-every 32` (bounded 32-deep in-flight
window; `--wait-every 0` submits the whole arm before one wait); the runner sets
the production L0 knobs (regular command lists, copy engine on, batch size 0,
cache persistent off). `run_shape_probe.ps1 -Cpu` smoke-tests the harness on the
CPU device; those numbers are not representative of the L0 submit path.
Output: one `[SHAPE ARM] name=... phase=... n=... p10_us=... median_us=...
p90_us=... mean_us=... min_us=... max_us=...` line per arm. Build (project build
untouched):

```
pwsh -NoProfile -File tmp/q4e-bottleneck-audit-20261016/build_probe.ps1 -Source tmp/q4e-structural-20261016/probe_submit_shape.cpp -Output tmp/q4e-structural-20261016/probe_submit_shape.exe
```

## 9. Per-submit kernel metadata path at the HC sites (DPC++ 2026.1 audit)

Source-level answer to "do `local_accessor` and `[[reqd_sub_group_size]]` trigger
a per-submit kernel specialization or metadata lookup". Not a re-specialization
of the device image, but `submit` does per-call template erasure and bookkeeping.
Anchors in the installed headers
(`.../oneAPI/compiler/2026.1/include/sycl/handler.hpp`):

- `handler.hpp:632`, called at `:885`/`:917`: every `parallel_for` runs
  `verifyUsedKernelBundleInternal(Info.Name)`. With no explicit kernel bundle set
  (Aila never calls `use_kernel_bundle`), this is the documented fast path
  ("Verifies the kernel bundle to be used if any is set"): a name check, no
  device-image lookup - but it is executed once per HC submit.
- `handler.hpp:893`/`:922`: `StoreLambda<...>(std::move(KernelFunc))` per submit -
  "Stores lambda to the template-free object. Also initializes the kernel name
  and prepares for arguments to be extracted from the lambda in
  handler::finalize()". Per-submit side effects: `MHostKernel.reset(new
  detail::HostKernel<...>)` (`handler.hpp:651`, one heap allocation) and
  `setDeviceKernelInfo<KernelName>(...)` (`handler.hpp:695`).
- `handler.hpp:700`/`:1027`: `getOrInsertHandlerKernelBundle(Insert=true)` per CG.
- `handler.hpp:473`/`:478` + `:2641`: each `local_accessor` construction appends a
  `shared_ptr<LocalAccessorImplHost>` to the handler's `MLocalAccStorage` (one
  heap allocation per accessor per submit) - the two norm ops carry one each; the
  down/up ops carry none.
- `[[reqd_sub_group_size]]` is a kernel function attribute consumed when the
  device image is built; it adds no per-submit specialization. The project build
  already passes `-fsycl-default-sub-group-size=16`, so the attribute matches the
  compile default.
- `handler::finalize()` (runtime side, outside the headers) then resolves the
  kernel in the per-context program/kernel cache (the first submit per kernel
  pays JIT; later submits are cache hits - inference from DPC++ behaviour, not a
  header anchor), marshals the ~12 captured arguments into UR launch parameters
  and sets the dynamic local-memory size for the SLM kernels.

So each HC submit pays: one bundle-name check, one `HostKernel` heap allocation,
~10-12 captured-argument collection and cache lookups, all inside the measured
interval; section 8's `A -> S -> B -> C/R` ladder prices exactly these
increments against the 6.5 us control.

## 10. Gap buckets (2026-10-16)

The `wait-every` scan showed the HC submit cost is batch-state bound (W=1
~22.2 us, W=8 7.0 us, W=32 3.7 us) while production measured 21-27 us per
submit - i.e. production appears to pay a drained round trip per submit. The
prime mechanism candidate: the host gap between two adjacent HC submits exceeds
the UR/L0 dynamic-batch timeout and the batch is closed. The instrumentation
now buckets that gap.

Definition: `gap = hc_submit_begin(now) - previous hc_submit_end`, one shared
chain across all four HC sites (attn_hc norm/down/up, mlp_hc norm/down/up,
mixer) on the submission thread. The first HC submit after process start (no
previous end) contributes no gap sample; the chain is not reset by the window
take. Buckets: `b0 < 10 us`, `b1 < 50 us`, `b2 < 200 us`, `b3 >= 200 us`.

Fields appended to `[Q4E AdaptiveWork]` (per-forward window; same gate,
`AILA_Q4E_HC_SUBMIT_TIMING=1` plus the line's `AILA_Q4E_WORK_SUMMARY=1`):

- `hc_gap_b0_n`/`hc_gap_b0_us` ... `hc_gap_b3_n`/`hc_gap_b3_us`: the count of
  submits whose preceding gap fell into the bucket and the mean submit cost of
  exactly those submits (bucket sum / count; 0.000 with no samples).
- `hc_gap_max_us`: the largest gap sample in the window.

Read: if the 21-27 us submits concentrate in the high-gap buckets (b2/b3) while
b0/b1 carry the warm numbers (~4-7 us), the drained-round-trip mechanism is
confirmed and the lever is removing/flushing host gaps (fewer boundaries,
larger batches) rather than the CG shape. If b0 (<10 us gaps) already shows
~21-27 us, the gap/batch-timeout story is refuted for that portion.

Implementation notes: the gap state is one physical chain shared by the totals
and window scopes (`g_hc_last_submit_end`, `g_hc_pending_gap_*` in
`HyperConnectionOps.cpp`) and is deliberately not reset by
`HcSubmitTimingTakeWindow`; only the bucket counters are scoped. Gate off: the
helpers return before any clock read, exactly as before.

Line-position note (2026-10-16 late session): the `[Q4E AdaptiveWork]` message
is truncated by the logger's fixed buffer at ~1025 chars, and with the HC block
appended at the end the `hc_submit_down/up` and all `hc_gap_*` fields were cut
off (gap2-k3 receipt: the line ended mid-`hc_submit_down_us`). Fixed with both
mechanisms:

1. The 16 `hc_submit_*` + 9 `hc_gap_*` fields lead the `[Q4E AdaptiveWork]`
   message, directly after `pos=/count=/ple_reused=`; the HC block ends at
   rendered char 689 of the 1024 cap (verified locally), so every HC field
   survives; the cut now only trims the trailing device-ms fields.
2. A dedicated `[Q4E HcSubmit] pos=... count=... hc_host_ms=... <all 25
   hc_* fields>` line, printed only when `AILA_Q4E_HC_SUBMIT_TIMING=1`
   (28/28 format/arg pairs, rendered length 687 - no field can be cut).

Field naming is uniform across all four submit stages:
`<name>_us`, `<name>_n`, `<name>_min_us`, `<name>_max_us`.

Final build receipts after the line fix: `build-refactor/AilaWorker.exe`
**2026-10-16 23:17:32**, `Aila.exe` 23:17:39, `AilaShared.dll` 23:16:19,
`AilaInference.dll` 23:16:15, `build-refactor-server/AilaServer.exe`
**2026-10-16 23:18:33** (server rebuilt in the same window so the handshake
identity matches). Race guard: both AilaLib objs (23:15:55 / 23:16:14) are later
than their src mtimes. Binary check: `[Q4E HcSubmit]` and the gap field groups
are present in `AilaWorker.exe`.

Build-window note (2026-10-16 late session): the concurrent GPU arm sequence
re-locks `build-refactor\libhwloc-15.dll` (loaded by every running
`AilaWorker.exe`) and `AilaWorker.exe` itself, so `build.ps1` runs that land
inside an arm fail in the `copy_runtime_dependencies` step with
`Error copying file (if different) ... libhwloc-15.dll`. The runtime build is
therefore gated on an arm-free window (`tmp/q4e-structural-20261016/
build_on_window.ps1`: poll 10 s, build immediately when the processes vanish,
retry on collisions).
