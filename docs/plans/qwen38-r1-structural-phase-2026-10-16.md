# R1 structural phase: OOO answer, submit-cost anatomy, XMX/stage-copy adopted (2026-10-16 night)

Owner directives for this phase: structural changes over tuning; the audit's
out-of-order-queue question and submit/launch overhead are the priority;
Routine rounds = single A/B (A/B/B/A only for default-on gates); XMX admission
gate relaxed where the deviation is provably rounding-order bound.

## Adopted (default-on, committed)

`6f9a41e` flips four defaults (escape hatch `=0` on each; rationale doc
`qwen38-r1-xmx-stagecopy-adoption-2026-10-16.md`):

| flag | effect on the primed 14K TTFT |
|---|---|
| `AILA_Q4E_PREFILL_EXPERT_XMX=1` (+MIN_M 256, ROWS 4096) | 69.6-71.2 s -> 63.99 s |
| `AILA_Q4E_STAGE_COPY_PARALLEL=1` | 63.99 s -> 53.2-54.1 s; steady blocks 17.0 -> 14.6 s; fill 7.4 -> 2.3 s |

Confirmed by two independent B arms (`xpb-b1` 54.06 s, `xpb-b2` 53.20 s);
`xpb-b2` ran the h4/bfloat16-KV bundle, so its decode rate is not comparable,
but the fill/bank/block timings reproduced exactly. Quality: owner waiver
recorded (KL 0.019 class, rounding-order bound; ALU arms stay content-identical,
XMX arms drift run-to-run at 32-token greedy).

## Answered / falsified

1. **Out-of-order queue: no.** Probe (`8f4950e`, `qwen38-r1-ooo-queue-probe`):
   OOO chain + depends_on is 7% slower and costs +27% host submit
   (6.45 -> 8.17 us); two truly independent chains do not overlap more than
   in-order (0.43 vs 0.42); removing the per-layer wait is what collapses the
   synthetic (-40%, 14.4 -> 8.6 ms); the event-status-query release behaves
   identically on both queue types. Hooke's anatomy doc: 203 `.submit(` sites,
   only 33 `depends_on`; decode relies 100% on implicit in-order; OOO blast
   radius 31 TUs + shared scratch + a test assertion. No OOO conversion.
2. **The "5x submit inflation" is really 2.7x** (Hooke): 18.5 us/submit vs
   6.5-6.8 us bare; the 5x was a denominator artifact (291 HC submits over 97
   calls). Absolute gap ~12.8 ms/token.
3. **`Context::queue()` is not the cost** (1 acquire load + 1 null check).
   `Bnb4BitLinear.cpp:2709`'s in-order check is the Q8 LM-head contract; a
   violating queue would silently disable LM_HEAD_Q8.
4. **chunk-8192 stays parked**: `6498159` falsified the min-cap hypothesis -
   the 55.5 s first-block cost is in the topk submit path, unchanged by
   `AILA_Q4E_RUNTIME_MIN_CAP` (133.5 vs 132.9 s TTFT).

## Submit-cost anatomy (new receipts)

| evidence | value |
|---|---|
| production, `hct-a1` (`AILA_Q4E_HC_SUBMIT_TIMING=1`) | HC submits 20.8 us (norm) / 26.0 us (norm-fb) per call; HC host 7.5 ms/token |
| E0 probe, direct production kernel, AilaLib-linked, drain-per-call | 23.8-27.5 us |
| E0 probe, param-table ABI + SYCL submit | 20.3-20.9 us |
| E0 probe, param-table ABI, burst (64 back-to-back) | 5.2-5.4 us |
| E0 probe, single-kernel graph replay | 32.8-49.2 us (no win alone) |
| shape probe (`probe_submit_shape`, wait-every-32, local kernels) | real down kernel 3.7 us; +SLM 5.0; plain 10.7; empty single_task 2.9 |
| shape probe under 18-thread DRAM contention | medians 6.7-15.2 us, p90 33-45 us |

**E0 verdict: the param-table ABI is bit-exact on the real `hc_decode_down_inject`
kernel** (3 shapes x 1000 replays, echo-pointer proof the table is live,
`all_bitexact=1 failures=0`). The graph/parameter-table route is viable for
correctness; its economics depend on the open regime question below.

**Open contradiction (next discriminator):** the same true kernel submits at
3.7 us (local binary, burst) but 23.8-27.5 us (AilaLib binary, drained) and
21-27 us in production. Hypothesis list: (a) kernel-program-cache size effect
in the big `per_kernel`-split binary, (b) drained-vs-batched launch regime,
(c) L0 batch flush behavior at production cadence (8 submits per wait).
The decisive probe: same kernel, same binary, sweep `wait_every` {1,2,4,8,16,32}
on both binaries.

## Noise floors (use for gates)

| metric | A/A spread |
|---|---|
| 14K TTFT (primed, single arm) | ~0.9% (53.2/54.1/53.5/54.0) |
| decode, 19-token continue, agent6k+int8, production timing | 1.05% (16.372/16.202) |
| decode, 19-token continue, h4+bf16-KV | 8% (15.02/16.24) - do not gate on this |
| decode, n=128, agent6k+int8 | 17.15 tok/s (baseline 17.38-17.50, unchanged by adoption) |

## Goal status

- Prefill 14K TTFT now 53.1-54.1 s (~258 tok/s; from 69.6-71.2 s ALU).
  400 tok/s needs ~35 s; remaining slices per steady block: attention +
  router wait ~7.5 s, experts (fill 2.3 + H2D/readers ~2.1 + compute) ~6.7 s.
- Decode 17.15 tok/s (n=128, canonical bundle) vs 20 tok/s target (-13%).
  Levers ranked: submission-cost regime (above), per-layer attention-subchain
  graph (multi-kernel graph + param table), fewer sync points.

## Next experiment queue (ranked)

1. Submit-regime discriminator: `wait_every` sweep on both binaries (30 GPU-min).
2. Multi-kernel attention-subchain graph using the proven param-table ABI
   (build + 12 GPU-min; graph replay only pays with many kernels per graph).
3. Only if (1) shows the 21-27 us regime is real and production-bound:
   batch submissions / restructure sync cadence around the router readback.
