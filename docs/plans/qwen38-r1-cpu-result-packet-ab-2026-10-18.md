# Qwen3.8 R1: CPU result packet A/B (2026-10-18)

Scope: A770 Windows, current paired Worker/Server binaries,
`Worker 2e603c73...`, `Server f0fc1aef...`.

Test: deterministic 2,472-token prompt, above the QSA enable threshold,
agent6k hotset, INT8 KV, ALU prefill (`AILA_Q4E_PREFILL_EXPERT_XMX=0`) so
cross-process greedy content is comparable.

Receipts:

- `tmp/q4e-sprint-r1-20261014/packet-clean-a2{,-run.log}`
- `tmp/q4e-sprint-r1-20261014/packet-clean-b2{,-run.log}`

## Result

`AILA_Q4E_CPU_RESULT_PACKET` replaces up to one rank-span H2D per CPU expert
with one fixed packet H2D per layer. The merge still walks the original rank
order.

| request | A tok/s | B tok/s | delta | content hash |
|---|---:|---:|---:|---|
| warmup | 14.076 | 14.615 | +3.83% | equal |
| 2,472-token first answer | 14.684 | 14.997 | +2.13% | equal |
| 2,472-token continuation | 14.489 | 15.077 | +4.06% | equal |

All three content hashes are identical. Both arms used the same Worker/Server
identity.

## Verdict

The mechanism is real and content-preserving, but the best measured window is
+4.06%, below the project's +5% default-on gate.

- keep `AILA_Q4E_CPU_RESULT_PACKET=0` as the production default;
- retain the packet path as a validated opt-in;
- do not run another packet sweep now; its remaining value is below the token
  graph P1 path.

## Rejected comparison

An earlier 6K/14K A/B used the production XMX prefill and therefore inherited
the documented XMX run-to-run greedy drift. Its decode deltas were +10.38% and
-0.90%, but content differed between processes. It is not evidence for or
against packetization and is not used in the verdict.
