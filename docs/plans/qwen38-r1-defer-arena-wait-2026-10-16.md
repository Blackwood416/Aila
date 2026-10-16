# R1 decode `CPU_DEFER_ARENA_WAIT` screen: NO-GO (2026-10-16)

Date: 2026-10-16. Base HEAD `e36ff4d`+`54d2bd2` (join-release receipts).
Candidate source: `docs/plans/qwen38-r1-decode-submit-path-audit-2026-10-16.md`
rank-1 next experiment (est -1.5..-2.8 ms/token).

## Result

A/B/B/A on the production decode bundle (hotset profile + int8 KV, production
timing, 14K continue, n=32), receipts `tmp/q4e-sprint-r1-20261014/da-*/`:

| arm | AILA_Q4E_CPU_DEFER_ARENA_WAIT | rate tok/s |
|---|---|---|
| da-a1 | 0 | 17.347 |
| da-b1 | 1 | 17.302 |
| da-b2 | 1 | 17.193 |
| da-a2 | 0 | 17.525 |

A median 17.436 (A/A spread 1.03%) / B median 17.248 -> **B - A = -1.08%**.
SSE content sha identical in all four arms. Below the +5% gate and below zero:
**NO-GO**, the flag stays default 0 (it already was).

## Reading

This is the third decode control-plane candidate measured at ~0 on the same
bundle (`AILA_Q4E_JOIN_RELEASE` +1.80% within a 2.24% A/A spread; pair2 -1.6%
in `qwen38-r1-decode-attn-pair2-gate-2026-10-16.md`). The audit's isolated-loop
estimates do not transfer 1:1 to the production decode loop: whatever the
condvar/arena-wait deferral changes, either it is already overlapped or it costs
as much as it saves in this configuration. Per the audit's own ranking, the
remaining decode items are the per-layer resident-input wait (needs code:
double-buffered `x_host_`), the fused-HC/shared launches, and the 64 B weights
H2D; those are queued as `tmp/q4e-sprint-r1-20261014/run-fused-hc-shared.ps1`.

## Evidence

* `tmp/q4e-sprint-r1-20261014/da-{a1,b1,b2,a2}/result.json` (rates, SSE sha).
* `tmp/q4e-sprint-r1-20261014/run-defer-arena.ps1` (arm sequence, 15 s drains).
