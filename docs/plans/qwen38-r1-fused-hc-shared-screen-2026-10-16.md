# R1 decode fused-HC + shared-fusion screen: NO-GO (2026-10-16)

Date: 2026-10-16. Base HEAD `e93c0ff`. Candidate source: submit-path audit
`docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md` (fused-HC -96
submits, shared fusion -48 submits).

## Result

A/B/B/A on the production decode bundle (hotset profile + int8 KV, production
timing, 14K continue, n=32), receipts `tmp/q4e-sprint-r1-20261014/fh-*/`:

| arm | flags | rate tok/s | SSE content sha | cached prompt tokens |
|---|---|---|---|---:|
| fh-a1 | none | 17.445 | `a9aa5132...` | 13976 |
| fh-b1 | `DECODE_FUSED_HC=1` + `MOE_SHARED_FUSED=1` | 13.734 | `2d1eae00...` | absent |
| fh-b2 | same | 14.328 | `2d1eae00...` | absent |
| fh-a2 | none | 17.549 | `a9aa5132...` | 13976 |

A median 17.497 / B median 14.031 -> **B - A = -19.8%**.

Two independent disqualifiers:

1. **Speed**: -19.8% on the 14K continue rate; the B arms also ran ~368 s wall
   vs 266-268 s for the A arms (the extra ~100 s is the prefix cache not
   engaging: all four B-arm requests report no `cached_tokens`, so the 14K
   continue paid a full prefill, TTFT ~71 s).
2. **Numerics**: the SSE content sha changes between A and B
   (`a9aa5132...` vs `2d1eae00...`), so the fused pair is not bit-exact - it
   cannot be adopted without the section-4 quality gate even if it were fast.

## Disposition

NO-GO. Both flags stay default 0. Do not spend more GPU time on the pair: a
-20% regression with a content change is out of scope for tuning. If either flag
is ever revisited, screen it individually (this arm only tested the pair, and
the prefix-cache side effect was not investigated).

## Evidence

* `tmp/q4e-sprint-r1-20261014/fh-{a1,b1,b2,a2}/result.json` (rates, sha, usage).
* `tmp/q4e-sprint-r1-20261014/run-fused-hc-shared.ps1` (arm sequence).
