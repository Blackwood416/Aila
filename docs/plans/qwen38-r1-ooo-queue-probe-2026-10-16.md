# R1 probe: out-of-order queue vs in-order, and what actually gates decode submits (2026-10-16)

Owner question: the bottleneck audit's batch-hold finding suggested the in-order
queue may be the structural problem; would an out-of-order queue (explicit event
dependencies) remove the submit/wait cost?

Probe: `tmp/q4e-structural-20261016/probe_ooo.cpp` (built with the audit's
`build_probe.ps1`, run with `run_probe.ps1` under the production L0 knobs:
regular command lists, batch 0, L0 v1, copy engine). Receipt:
`tmp/q4e-structural-20261016/ooo-probe-L20.log`. Kernel mix and layer shape
match the decode-shaped audit probe (48 layers x 20 chained kernels,
wait-on-last-event per layer, 20 tokens, A770, oneAPI 2026.1).

## Result

| arm | wall ms/token | host submit us/kernel | device busy ms | note |
|---|---:|---:|---:|---|
| in-order chain, wait/layer | 13.92 | 6.45 | - | replicates the audit (13.8-14.6) |
| out-of-order chain + depends_on, wait/layer | 14.93 | 8.17 | - | +7% wall, +27% submit cost |
| in-order, two independent chains, wait/layer | 14.43 | 6.92 | 6.22 | overlap 0.43 |
| out-of-order, two independent chains, wait/layer | 15.25 | 8.53 | 6.39 | overlap 0.42 |
| out-of-order, two independent chains, no per-layer wait | 8.61 | 8.28 | 5.89 | overlap 0.68 |
| hold pattern (2 kernels + 230 us host spin), in-order | no-flush 337 / status-query 278 us | | | replicates audit flush finding |
| hold pattern, out-of-order | no-flush 325 / status-query 278 us | | | identical behavior |

## Conclusions

1. **Out-of-order queue is not the lever.** Explicit dependency bookkeeping
   raises host submit cost 6.45 -> 8.17 us/kernel (+27%) and the per-layer
   chain gets 7% *slower*. Two truly independent chains do not overlap any
   more than they already do in-order (overlap 0.43 vs 0.42); the per-layer
   SYCL wait gates both queue types equally.
2. **The lever is the wait cadence.** Removing the per-layer wait drops the
   same kernel set 14.43 -> 8.61 ms/token (-40%), matching the audit's
   one-wait-at-end arm (8.3 ms). In production the per-layer wait is the
   router top-k readback, which the host needs for routing - so the structural
   paths are (a) fewer/cheaper sync points around that readback, (b) fewer
   submissions (fusion / graph replay), (c) amortize the readback over more
   tokens (MTP / verify batching).
3. **The status-query release works identically on both queue types**
   (~59 us saved per hold; 337 -> 278 in-order, 325 -> 278 out-of-order).
   `AILA_Q4E_JOIN_RELEASE` already applies this at the decode CPU join; the
   probe says queue order must not be part of that fix.
4. Negative result recorded: no further out-of-order conversion work is
   planned for the decode path. `Context` keeps the in-order queue.
