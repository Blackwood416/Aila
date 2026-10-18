# Qwen3.8 R1 token graph P5: QSA/full-attention primitives (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2.

This gate makes the full-attention/QSA decode operators replayable with
device-position control. It does not yet capture a complete L3 layer.

## 0. Verdict

The fixed graph ABI now covers:

- packed Q/gate split, Q/K RMSNorm + partial RoPE, and K/V row writes;
- QSA query preparation and raw-K caching;
- at-most-one completed 4-token block compression;
- fixed-grid compressed-block scoring;
- one fixed 256-thread selector for dense, 513 complement, and 514+ exact
  cutoff paths;
- exact head-dim 256 block-native sparse attention with fixed max tile grid.

All focused direct-vs-graph tests pass bitwise. Production dispatch remains
unchanged and no graph is enabled by default.

## 1. Graph control ABI

`src/ops/GraphControl.hpp` defines a fixed 16-word device int32 table:

```text
[0] start_pos
[1] compressed_blocks
[2] abort
[3..15] reserved
```

The table pointer is baked at capture time; only values change between
replays. Out-of-range block counts or a non-zero abort flag fail closed.

## 2. Receipts

### QSA scoring and selector

`gpu-qsa-attention-graph.log`:

```text
pos=2051..2055
blocks=513..514
scores/active-blocks bitwise against the host-sized path
```

Covered cases:

- 513 blocks complement selection with tail lengths 0..3;
- 514 blocks general exact top-512 selection;
- selected block IDs ascending, followed by trailing token IDs;
- score ordering `(score descending, block id ascending)` and NaN ordering.

### Fixed-grid block-native attention

`gpu-qsa-attention-graph.log`:

```text
start_pos=2052..2055
actual tiles=17,17,17,16
output and partial buffers bitwise
abort=1 leaves output/partials unchanged
```

Phase 1 always launches `24 * 17` groups of 128 threads; phase 2 always
launches 24 groups. Groups beyond the device-derived actual tile count exit
before touching memory or barriers.

### Device-position QGKV preparation

`gpu-qgkv-graph.log`:

```text
pos=2052..2055
q/gate/K/V scratch and full BF16 K/V cache bitwise
abort=1 leaves the K cache unchanged
```

The graph path derives RoPE, K/V row, and quantization-scale indices from
`control[start_pos]`; no host position is required during replay.

### Full QSA regression

`gpu-qsa-full-after-graph.log` keeps the existing 1428 selector cases,
prefill scoring/selection, and block-native decode tests green in addition to
the new graph tests.

## 3. API surface

```cpp
qsa_prepare_q_and_cache_k_graph(...)
qsa_compress_one_completed_block_graph(...)
qsa_compute_block_scores_compressed_graph(...)
qsa_select_topk_blocks_graph(...)
attention_decode_exact_head256_block_native_graph(...)
decode_prepare_qgkv_packed_partial_graph(...)
```

All graph variants validate fixed shapes before capture and preserve the
existing host APIs.

## 4. What is not proven

- The operators have not yet been composed into one L3 attention sublayer.
- QSA indexer projection and full QKV/o projections are not in those tests.
- Dense QSA above/short-context topology remains a separate graph variant.
- Per-full-layer compressed-block counters are not solved; the current
  control word is sufficient for one-layer proof only.
- No production graph lease, session poison, or forward integration exists.

## 5. Next gate

Capture one complete L3 sparse-QSA attention sublayer:

```text
HC -> packed QKV -> graph QGKV prep -> index_qk projection
   -> QSA q/raw-K/compress/score/select
   -> block-native attention -> o_proj
```

Start from a real 2052-token prefill checkpoint and replay four tokens over
`2053..2056` visible length. Compare q, z, new KV row, indexer K, compressed
block 513, active blocks, attention output and o-projection output bitwise.
