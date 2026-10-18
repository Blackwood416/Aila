# Qwen3.8 R1 token graph P8: structural hardening (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2.

## 0. Verdict

The graph foundations now fail closed before state-writing graph nodes, support
independent counters for all 12 full-attention layers, and contain the first
production ownership pieces for a one-in-flight token graph.

This is not yet a production whole-token graph. It removes the blockers found
by the P7 source audit and preserves the real-weight L3 receipt.

## 1. Graph control and QSA

- QGKV preparation, QSA preparation and compression validate position/block
  count before their first cache write.
- A producer-side check permits exactly one pending completed block; stale or
  future counters publish sticky abort without writing.
- Compression now uses the unified `GraphControl.abort`; its old independent
  invalid flag is gone.
- Layer 0 keeps the original compressed-block slot. Layers 1 to 11 use
  independent slots, so one executable can advance 12 full-attention layers
  without sharing counters. Slot 3 remains reserved for layer-0 diagnostics.
- Sparse block IDs and selected tail token IDs are validated in a separate
  single-workgroup graph node before attention. Invalid IDs publish sticky
  abort and attention leaves output and partials unchanged.

The separate validator is intentional. A first attempt placed a
workgroup-reduced check inside the tile kernel; a mid-kernel abort could make
some work-items exit before barriers while peers continued. The GPU hung for
167.7 seconds. The separate kernel fixed the divergence; the same test now
completes in 15.4 seconds.

## 2. State identity

- `TokenGraphKey` now includes a backend-assigned monotonic Session instance
  ID, PLE workspace generation and Session-scoped PLE graph-view generation.
- `MakeTokenGraphKey()` rejects mixed host/device GDN authority, mRoPE state
  and embedding overrides for the current graph body.
- GDN steady decode distinguishes `device_authority` from
  `device_synced`; a stale host mirror remains readable only while device
  authority owns it.
- `SubmitTokenGraph()` rejects unpublished heads, marks GDN state dirty after
  submission, and requires later event-side abort reporting to poison the
  Session.
- Snapshot capture synchronizes device-authoritative GDN heads and rejects a
  poisoned Session.
- MTP Restore advances an independent PLE ticket epoch, rejecting speculative
  tickets prepared against rolled-back state.

## 3. Runtime scaffolding

- `Context::alloc_host_uncached()` allocates Level Zero host USM with
  `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` and accounts it in the existing
  pinned-host ledger.
- `Qwen4ExpTokenGraphRuntime.hpp` adds stable P3 ring PODs and a non-copyable
  `TokenGraphRuntime` state machine:

```text
Empty/Ready -> Capturing -> Ready -> InFlight -> Ready
                                      |
                                      +-> Aborting -> Poisoned
```

It enforces one in-flight launch, sequence matching, gate validation and
permanent poisoning. It intentionally does not own a queue, executable,
thread or packet service yet.

## 4. Receipts

Full runtime build:

```text
291/291 targets linked
```

CPU/ABI:

```text
ctest -L "cpu|abi": 20/20 passed
AilaInferenceCApiGpuTests skipped by label
```

Final focused GPU suite:

```text
uncached-host-allocation         PASS  16.6 s
QSA per-layer counters           PASS   2.7 s
QSA graph abort/stale checks     PASS   2.6 s
block-native attention graph     PASS  15.4 s
QGKV graph                       PASS   6.1 s
GDN graph authority              PASS   0.6 s
token graph key generations      PASS   0.6 s
real-weight L3 QSA graph         PASS  28.4 s
```

Real L3 remains bitwise over `start_pos=2052..2055`, blocks `513..514` and
tiles `17/17/17/16`.

Receipts:

- `tmp/q4e-token-graph-probe-20261017/gpu-structural-final-suite.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-uncached-host-posthardening.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-attention-graph-final.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-qsa-real-l3-posthardening.log`

## 5. Limits

- No production `ForwardTokenGraph` is wired to `ForwardImpl` yet.
- `TokenGraphRuntime` does not own or drain its event/executable resources.
- The real L3 fixture still seeds its hidden/cache data; it is not an upstream
  2052-token prefill checkpoint.
- MTP target verification is still direct. Draft remains outside the token
  graph and is intentionally deferred.
- Dense QSA, INT8/FP8 QGKV, KV streaming and multi-session graph sharing remain
  out of scope.
- Residency/descriptor freeze during an in-flight token graph is not enforced
  by the new runtime container yet.

## 6. Next gate

Wire the proven primitives into one production single-in-flight orchestration:

```text
prepare runtime/PLE and freeze residency before key construction
  -> one executable and one event
  -> same orchestration thread services CPU packets
  -> validator/control gate
  -> commit host mirrors only after successful event completion
  -> abort, drain and poison on any failure
```

The first production body should stop before LM head/sampling and must not
capture `ForwardImpl`. A two-row target-verification variant is the next MTP
step only after the single-token path has an e2e receipt.
