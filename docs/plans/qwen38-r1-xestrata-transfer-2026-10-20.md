# Qwen3.8 R1: XeStrata transfer results (2026-10-20)

Reference: `tmp/XeStrata` xe0.1.45, A770 Windows, oneAPI 2026.1.

## 0. Verdict

The first graph-relevant XeStrata transfer is a clear win:

```text
QSA graph selector, 8192 complete blocks
legacy 32-pass bit selector: 2244.3 us
4 x 8-bit radix selector:      233.8 us
speedup:                        9.60x
selected IDs:                  exact
```

The same selector is exact for 513 blocks, 514 blocks with tail, and 8192
blocks with NaN/tie cases. It is now the default graph selector.

The attempted active-token SLM cache in block-native attention was bitwise
correct but slower and remains diagnostic-only.

## 1. Adopted changes

### Graph radix selector

- Independently reimplemented the 4 x 8-bit radix idea using Aila's existing
  integer score ordering, subgroup-local histograms, vec4 loads, stable
  compaction, lower-ID tie break and tail append.
- Fixed one-workgroup geometry; all dynamic values still come from the
  per-full-layer GraphControl slot.
- `selector_mode=0` keeps the legacy graph selector for in-process A/B.
- `selector_mode=1` is the default graph path.

Receipt:

```text
blocks=513 tail=0 selected=512 legacy=exact radix=exact
blocks=514 tail=3 selected=512 legacy=exact radix=exact
blocks=8192 tail=2 selected=512 legacy=exact radix=exact
legacy_us=2244.3 radix_us=233.8 speedup=9.59923 exact=1
```

### Graph lifecycle hardening

- `TokenGraphRuntime::event_status()` adds non-throwing
  `NoEvent/Pending/Done/Failed` event diagnostics. Query errors report
  `Failed`, never `Pending`.
- P10 records logical capture nodes and rejects an empty recording before
  finalize.
- Every P10 executable release now passes through
  `ReleaseQsaLayerGraph(ctx, idle_confirmed)`; it destroys immediately only
  when the event is known complete, otherwise drains first.
- Backend teardown uses the same release path before caches/scratch die.

## 2. Negative result: attention active-token SLM cache

The cached variant computes each selected `cache_t` once per tile and reuses
it for QK and PV. It is bitwise at `2052..2055` and at a 32768-capacity,
non-contiguous sparse-ID case, but it is slower:

```text
old replay wall:    0.4301 ms
cached replay wall: 0.5016 ms
```

Keep `attention_decode_exact_head256_block_native_graph` as production. The
token-cache entry remains only for future register/SLM layout experiments.

## 3. Real L3 regression check

The real-weight L3 probe still passes after the radix selector change:

```text
start_pos=2052..2055
blocks=513..514
active=512
bit_exact=1
wall=27.8 s
```

## 4. XeStrata findings not adopted

- No device-side cold lane exists. XeStrata still uses host-scheduled CPU
  misses and device-side waiting. Aila's all-CPU cold policy remains valid.
- Copy-only queue is already a measured A770 negative for this workload.
- Multi-layer commit-batch is deferred until Aila has an independent
  accepted-prefix/MTP commit stage.
- `stream_fork/join`, second compute queue and MTP side stream do not fit the
  A770 one-compute-queue topology.

## 5. Next structural step

Productize the P3 whole-token protocol:

```text
one graph launch
  -> one event/status query to release the L0 batch
  -> host polls uncached route ring while CPU experts compute misses
  -> device doorbell wait + graph copy kernel consumes CPU packet
  -> final event done + gate
  -> commit host mirrors, or poison/drain on failure
```

Do this only after a real production-session L3 `2052..2055` receipt. Do not
replace `TokenGraphRuntime` with XeStrata's raw graph pointer or pair key.

## 6. Receipts

- `tmp/q4e-token-graph-probe-20261017/gpu-qsa-selector-radix-ab-xestrata.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-attention-token-cache-ab-xestrata.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-qsa-per-layer-xestrata.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-qsa-graph-xestrata.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-qsa-layer-xestrata.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-token-graph-runtime-xestrata.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-qsa-real-l3-xestrata.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-xestrata-transfer-suite.log`
