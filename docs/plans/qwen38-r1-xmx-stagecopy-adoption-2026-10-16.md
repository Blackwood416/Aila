# Adoption: expert-XMX + parallel stage copy default-on (2026-10-16, owner waiver)

## Decision

Four R1 flags flip to default-on in the source tree (each still overridable
with the documented `=0` escape hatch):

| flag | old default | new default | measured effect (14K TTFT) |
|---|---|---|---|
| `AILA_Q4E_PREFILL_EXPERT_XMX` | 0 | **1** | 69.6-71.2 s -> 63.99 s (-8..-10%) |
| `AILA_Q4E_PREFILL_EXPERT_XMX_MIN_M` | 0 | **256** | min-band phase 2: -2.0% (kept) |
| `AILA_Q4E_PREFILL_EXPERT_XMX_ROWS` | 2048 | **4096** | with MIN_M 256: 64.2 s, blocks 17.0-17.7 s |
| `AILA_Q4E_STAGE_COPY_PARALLEL` | 0 | **1** | 63.99 -> 54.06 s (-15.5%); blocks 17.0 -> 14.6 s |

## Evidence (receipts in `tmp/q4e-sprint-r1-20261014/`)

Steady 4096-row blocks of the primed 14K request (three blocks each):

| arm | config | TTFT | blocks (s) | fill (s) | bank_wait (s) | experts_host (s) |
|---|---|---:|---|---|---|---|
| `fx-a1` | XMX+min-band | 63.99 | 16.84/17.54/17.65 | 7.37-7.51 | ~0.009 | 9.60-9.79 |
| `xpb-b1` | + stage-copy | 54.06 | 14.60/14.85/14.91 | 2.33-2.38 | 2.10-2.19 | 6.70-6.97 |
| `xpb-b2` | + stage-copy (repeat) | 53.20 | 14.23/14.60/14.68 | 2.29-2.46 | 2.04-2.11 | 6.72-6.91 |

The two B arms reproduce the win; the decode rate is unaffected
(16.1-14.8 tok/s in the same arms, within the 14K-continue spread).

## Quality / waiver

- Teacher-forced, ALU vs XMX-bf16 over 128 positions: mean KL 0.019, NRMSE
  0.298, top-1 120/128 (contract was KL 0.01 / top-1 122/128).
- The deviation is accumulation-order bound, not dtype: two XMX variants
  differ by 0.0099 KL (same class); fp16 weight storage tightened the kernel
  nrmse 2.1x with no e2e change (phase 3 receipts).
- Run-to-run: the 32-token greedy continuation drifts between runs with the
  XMX path (three distinct content shas across fx/xpb-b1/xpb-b2); the
  ALU-expert arms (pfx-a1/a2/b1/b2) are content-identical across all four.
  This is the recorded rounding-order class, not a new defect.
- Owner ruling 2026-10-16: with the rounding-order evidence, relax the
  admission gate. Recorded here as the standing waiver for the expert-XMX
  path (`AILA_Q4E_PREFILL_EXPERT_XMX=0` returns to the exact ALU path).

## Out of scope

- Decode paths are unchanged; stage copy only affects the prefill staging
  banks, expert-XMX only the prefill MoE segment.
- `AILA_Q4E_RUNTIME_MIN_CAP` stays opt-in (its hypothesis was falsified,
  see `qwen38-r1-chunk8192-first-block-falsification-2026-10-16.md`).
