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

Pending (waiting for the running GPU arm to exit before building).

## 7. Files changed

- `src/ops/HyperConnectionOps.hpp` (timing struct + accessors)
- `src/ops/HyperConnectionOps.cpp` (read-once gate, counters, 4 submit sites)
- `src/models/Qwen4ExpBnb4Backend.cpp` (window take + 16 appended log fields)
- this file
