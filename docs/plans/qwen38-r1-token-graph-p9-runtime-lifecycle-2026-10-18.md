# Qwen3.8 R1 token graph P9: backend runtime lifecycle (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2.

## 0. Verdict

The one-in-flight token-graph state machine is now owned by the Qwen4Exp
backend and sits on the real direct-path invalidation boundaries. Default
direct execution remains `Empty`, allocation-free and unchanged.

This is lifecycle integration only. No production graph body is captured yet.

## 1. Backend integration

- `Qwen4ExpBnb4Backend` owns a non-copyable `TokenGraphRuntime` value member.
- `BeginCapture`, `MarkReady`, `Launch`, `Complete`, `AbortTokenGraph` and
  `AbortAndPoisonTokenGraph` expose the minimal lifecycle.
- `Launch` verifies the key's Session identity and permits only one in-flight
  graph. L3-only `QsaLayerProof` bodies do not require or dirty GDN state;
  whole-token bodies retain the GDN authority checks.
- Direct `ForwardImpl`, Reset, Truncate, GDN authority transitions,
  embedding/mRoPE mutation and topology setters abort before mutating graph
  visible state.
- Session invalidation drives the same runtime abort. A poisoned Session still
  rejects graph keys and checkpoint capture.
- Session teardown retires its runtime key before synchronizing/freeing cache
  state, so a Ready graph cannot retain a pointer to a destroyed Session.

## 2. Correctness fixes found by the focused test

Two issues were caught before merge:

1. A zero-snapshot Session returned from checkpoint capture before the poison
   check. Poison is now checked first.
2. Session destruction left the runtime key pointing at freed Session memory.
   `TokenGraphRuntime::RetireSession()` now enters `Aborting` and clears the
   key identity.

## 3. Receipts

Focused GPU-mutex suite:

```text
token-graph-runtime-only  PASS  0.6 s
token-graph-key-only      PASS  0.5 s
gdn-graph-state-only      PASS  0.4 s
```

The post-build runtime check passed again at `75c239e` in 0.7 s after the
full 291-target runtime/Worker build.

The runtime test covers:

- default direct path remains `Empty` and performs no graph allocation;
- direct forward aborts a Ready graph;
- second in-flight launch is rejected;
- successful completion returns to Ready;
- Reset, Truncate and MTP restore invalidation abort before mutation;
- Session destruction clears the key and aborts;
- poison is permanent for runtime, key and checkpoint paths.

Receipts:

- `tmp/q4e-token-graph-probe-20261017/gpu-token-graph-runtime-p9.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-token-graph-runtime-p9-final.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-token-graph-key-p9.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-gdn-graph-state-p9.log`

## 4. Limits

- The runtime is backend-wide, not per-Session. This matches one GPU device
  mutex and one in-order compute queue, but one abort blocks further graph
  capture until backend/session recreation.
- `Aborting` has no return-to-Ready transition. The future owner must drain,
  poison and release executable resources explicitly.
- No executable, event ownership, uncached ring service or graph body is wired
  here yet.
- Default direct execution remains the production path.

## 5. Next gate

Implement the narrow production single-layer body:

```text
real L3 weights/session caches
  -> qkv/index_qk/QGKV/QSA/selector/validator/attention/o_proj
  -> one executable and one runtime launch
  -> control gate and host compressed-count commit
```

Use the fixed `buf_.x`/`buf_.y` rows, graph ordinal 0 for model L3, BF16 K/V,
no mRoPE/embedding override/Q8/INT8/LoRA, and an explicit
`AILA_Q4E_QSA_LAYER_GRAPH=1` opt-in. Do not capture `ForwardImpl` or add a
per-layer graph chain.
