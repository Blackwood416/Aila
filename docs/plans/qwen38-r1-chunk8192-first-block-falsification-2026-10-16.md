# Qwen3.8-Flash chunk-8192 first-block blowup: attribution and min-cap falsification (2026-10-16)

Receipts: `tmp/q4e-sprint-r1-20261014/x8-a1*`, `tmp/q4e-sprint-r1-20261014/x8m-a1*`.
Config both arms: XMX prefill expert path (`AILA_Q4E_PREFILL_EXPERT_XMX=1`,
MIN_M=256, ROWS=4096), `AILA_Q4E_STAGE_COPY_PARALLEL=1`, int8 KV, agent6k
hotset, chunk 8192, no prime, 14K request.

## Result

| arm | extra | TTFT 14K | first block (count=8192) | topk_submit_ms | steady block (count=5752) |
|---|---|---:|---:|---:|---:|
| `x8-a1` | - | 132.9 s | ~118 s | 55.7 s | 18.6 s |
| `x8m-a1` | `AILA_Q4E_RUNTIME_MIN_CAP=8192` | 133.5 s | 113.8 s | 55.5 s | 18.7 s |

`x8m-a1` first-block ledger (worker log):
`submitted_ms=113800.9 router_wait_ms=14479.9 topk_submit_ms=55476.9
linear_host_ms=11202.8 full_host_ms=21072.6 experts_host_ms=9373.1 fill_ms=2507.8
bank_wait_ms=2211.8 hc_host_ms=1849.8`.

## Verdict

- **Falsified**: the blowup is not caused by runtime-cap growth
  (`max(seq, cap*2)` paging). With a preallocation floor of 8192
  (`AILA_Q4E_RUNTIME_MIN_CAP`) and PHASE_SCRATCH squeeze, `topk_submit_ms`
  is unchanged (55.5 s vs 55.7 s) and TTFT is unchanged (133.5 s vs 132.9 s,
  within run noise).
- The ~55 s first-block cost lives inside the **topk submit path** itself
  (one-time, not proportional to count: 1.97 ms at count=5752). Next probe
  should isolate allocation/first-touch in that path (prefill top-k submission
  and its staging buffers), not runtime scratch.
- Steady-state ledger confirms chunk-8192 is otherwise healthy:
  router_wait 14.5 s / 8192 rows = 1.77 ms/row, matching 4096-chunk
  (7.1 s / 4096 = 1.73 ms/row). router_wait is linear in rows and is the
  dominant prefill term at both chunk sizes.

## Disposition

- `AILA_Q4E_RUNTIME_MIN_CAP` is kept as an opt-in preallocation floor
  (default 0 = old behavior). It is not a fix for the topk first-touch.
- chunk 8192 stays parked until the topk-submit first-touch is attributed;
  chunk 4096 remains the production chunk.
