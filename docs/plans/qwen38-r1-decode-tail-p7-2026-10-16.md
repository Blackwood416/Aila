# P7: T1 decode tail accounting (the last ~5 ms) + two candidates

Date: 2026-10-16. Repo `E:\RiderProjects\Aila`, HEAD `2e25fb8`. Owner: subagent P7.
Zero GPU, zero build, zero tree edits (one `git apply --check` receipt below).
Sources: `tmp/q4e-sprint-r1-20261014/e10-bundle{,-b2}-worker.log` (best bundle:
agent6k profile + int8 KV + MOE_FRONT_FUSED default), `g5-a1/a2-worker.log`
(default config, `AILA_PROFILE_Q4E_DECODE=1`), each arm's `result.json`
(`worker_environment` + client stats), `src/inference/Model.cpp`,
`src/models/Qwen4ExpBnb4Backend.cpp`, `src/ops/GreedyOps.hpp`,
`tests/server/socket_tests.py`, `tools/server/qwen38_context_probe.py`.

## 1. Per-token table (e10-bundle-a, 14K-continue segment, 31 decode steps)

Means over the segment; segment = `[Q4E ForwardWall] ... sample=1` lines.

| item | ms | source |
|---|---:|---|
| prepare (slot Ensure + token H2D) | 0.09 | `Model.cpp:362-364`, log `prepare_ms` |
| backend, submitted stages (host wall of `Forward`) | 51.26 | log `submitted_ms` mean |
| - router stage (sync + submits) | 19.06 | `router_wait 11.45` + front 1.81 + topk 0.89 + shared_submit 2.78 + other 1.87 |
| - experts stage | 18.21 | `cpu_compute 11.44` + `cpu_join 7.21` (+submits) |
| - hc stage (2 calls/layer) | 6.62 | log `hc_host_ms` |
| - attention (36 GDN + 12 full) | 6.06 | `linear_host 3.22` + `full_host 2.84` |
| - ple / head / shared_add | 0.24 / 0.16 / 0.70 | log |
| backend tail (`backend_ms - submitted_ms`) | 0.23 | log; no per-token sync in this arm |
| sample (greedy 2 launches + 16 B D2H, blocking) | 3.35 | `Model.cpp:563-577`, `GreedyOps.hpp:17` |
| drain (`ctx.synchronize()` after sample) | 0.02 | `Model.cpp:411` |
| scope wrapper delta (total - backend - sample) | 0.12 | log |
| **ForwardWall total** | **54.84 mean / 54.67 med** | 128-token median 54.77 |
| client inter-token interval (probe metric) | 56.94 | `result.json` rate 17.563; `socket_tests.py:536-537` |

## 2. Where the ~5 ms is

`ForwardWall total` (54.77) minus the submitted median (49.99-52.55):

1. `sample_ms` **3.35** - this is *not* host argmax. The greedy path is already
   device-side (`ops::greedy_bf16`, 64x256 threads, two launches) and reads back
   16 B (`GreedyResult`). The blocking D2H waits for the queue in front of it,
   which is the bf16 LM head: `[2560,248320] x 2 B = 1271.4 MB` at ~470 GB/s =
   **2.71 ms** (E5 `qwen38-r1-decode-kernel-roofline-2026-10-15.md` section 2),
   plus greedy ~0.15 and D2H/queue ~0.4.
2. backend tail + prepare + drain + scope = **0.46**.
3. outside `ForwardWall`: publish/IPC/SSE residual ~**0.45 [est]** plus a
   **1.65 ms metric artifact** (see 4).

Cross-check that the tail is device work, not a host scan: in `g5-a1/a2`
(`AILA_PROFILE_Q4E_DECODE=1`) a `ctx.synchronize()` runs at the end of every
backend forward (`Qwen4ExpBnb4Backend.cpp:3124-3126`), and the same tail shows
up as `backend - submitted = 3.49` while `sample` drops to 0.38. Tail sum is
3.6-3.9 ms in both arms. **Do not run decode arms with
`AILA_PROFILE_Q4E_DECODE=1`** (extra full drain + `[Q4E Adaptive Profile]` line
per token); e10-bundle already ran with it at 0.

## 3. Candidates

### C1 `AILA_Q4E_LM_HEAD_Q8=1` (wired, default off)

- Mechanism: quantize the head weight to Q8 at load, halving the 1271 MB read
  to ~675 MB (`Qwen4ExpBnb4Backend.cpp:540-566`, forward `:2854-2861`).
- Expected: `sample_ms` 3.35 -> ~2.1; wall **-1.2..-1.6 ms [est]**.
- Not bit-exact (bf16 row -> q8, 64-block input scales). Gate with the E1
  teacher-forced Top-1 comparator `>= 0.95` (the standard the user already
  accepted for MTP) plus text equality on the decode stream.
- Note: MTP load requires `!AILA_Q4E_LM_HEAD_Q8` (`Model.cpp:800-802`), so keep
  `AILA_Q4E_SERVER_MTP=0` in these arms.
- Section 2 check: not listed; the decode-misc doc calls it "out of scope" for
  the head roofline, not disproven.

### C2 Speculative CPU cold-expert precompute from the previous token's route

- Mechanism: the host already gets `router_host_ids_` per layer every token;
  the CPU pool is idle during the router wait (`router_wait 11.45 ms/token`).
  Keep the previous token's route per layer, dispatch the CPU cold jobs for
  that set at layer entry (before the route D2H), then on route arrival drop
  the non-matching jobs and compute the remainder.
- Expected: **-2..-6 ms [est]**, proportional to `cold_cover` (fraction of the
  current cold set that the previous route already covers).
- Files: `Qwen4ExpExpertExecutor.*` (dispatch/cancel), hook at
  `Qwen4ExpBnb4Backend.cpp:2568-2600` (`bind_actual_route`).
- Section 2 check: not listed. `CPU_ROUTE_EARLY`/`STAGEIN_HOIST` were neutral
  (ledger L57) and `CPU_EARLY_START` is inert in the adaptive pipeline; this is
  speculative precompute, a different mechanism.
- Probe first (counter only, no behavior change): apply
  `tmp/q4e-sprint-r1-20261014/p7/route-replay-probe.patch` (38 insertions, one
  file, gated by `AILA_Q4E_ROUTE_REPLAY_COUNT`, default off) and read
  `[Q4E RouteReplay] ... overlap_rate cold_cover`. Gate: `cold_cover >= 0.70`
  -> build the precompute; `<= 0.40` -> close.

### Harness note (affects every gate)

`rate = (completion_tokens-1) / (duration - first_content)` where `duration` is
the client's terminal time (`socket_tests.py:485-537`). A completed request
runs one extra no-logits forward before the terminal
(`Request.cpp:337-346`, measured 45.1-51.3 ms), so ~1.5 ms/token is charged to
the interval although generation stopped earlier. Measuring
`(N-1) / (t_last_content - t_first_content)` would restore ~+2.7% rate on every
past arm; keep using the current formula for comparability until the ledger is
re-baselined.

## 4. Commands and gates

```powershell
# C1 A/B/B/A (GPU mutex first; bundle env on every arm)
python tmp/q4e-sprint-r1-20261014/run.py p7-a1 --prime --diag --generate=32 `
  AILA_Q4E_HOTSET_PROFILE=E:\RiderProjects\Aila\perf\q4exp\profiles\qwen38-nf4-agent6k-v1.profile `
  AILA_KV_QUANT=1 AILA_KV_QUANT_FORMAT=int8 AILA_SERVER_KV_TYPE=int8
python tmp/q4e-sprint-r1-20261014/run.py p7-b1 --prime --diag --generate=32 `
  AILA_Q4E_HOTSET_PROFILE=... AILA_KV_QUANT=1 AILA_KV_QUANT_FORMAT=int8 AILA_SERVER_KV_TYPE=int8 `
  AILA_Q4E_LM_HEAD_Q8=1
python tmp/q4e-sprint-r1-20261014/run.py p7-b2 --prime --diag --generate=32 ... AILA_Q4E_LM_HEAD_Q8=1
python tmp/q4e-sprint-r1-20261014/run.py p7-a2 --prime --diag --generate=32 ...
```

- Read: `[Q4E LM Head Q8] actual_enabled=1` in B logs; `sample_ms` median
  (A ~3.4 -> B <= 2.6); `ForwardWall total_ms` median; continue rate.
- Gate: 2/2 pairs on `total_ms` direction, A/A <= 1.5%, `sample_ms` down >= 20%,
  Top-1 >= 0.95 vs the A token stream; otherwise leave the flag off.
- C2 counter: apply the patch, rebuild, one arm with the bundle env plus
  `AILA_Q4E_ROUTE_REPLAY_COUNT=1`; read `cold_cover`.
- Optional attribution (same config): `AILA_Q4E_DEVICE_QUEUE_PROFILING=1` +
  `AILA_Q4E_DECODE_STAGE_EVENTS=1` prints `head_dev_ms` and the last inter-gap,
  confirming the 2.71 ms head split directly.

## 5. Not resolvable from current data

1. Production (non-diag) per-token logging cost is unmeasured (every decode arm
   is `--diag`); estimate < 0.2 ms.
2. The publish/IPC/SSE residual (~0.45 ms) cannot be split worker vs server
   without a worker-side timestamp at `SendEvent` (`InferenceWorker.cpp:960-1000`).
3. The head/greedy/D2H split inside `sample_ms` is inferred from E5's event arm,
   not measured on this config (one `DECODE_STAGE_EVENTS` arm would settle it).
4. `lm_head.weight` is stored bf16 (file 131 = 1,271,398,400 B = 248320x2560x2),
   so there is no NF4 head to reuse; an NF4-at-load head would need a new C++
   NF4 quantizer (only `quantize_bf16_weights_q8_64` exists).

## 6. Patch receipt

```
git apply --check --verbose tmp/q4e-sprint-r1-20261014/p7/route-replay-probe.patch
Checking patch src/models/Qwen4ExpBnb4Backend.cpp... exit 0
1 file changed, 38 insertions(+)
```
