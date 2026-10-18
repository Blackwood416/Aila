# Qwen3.8 R1 token graph P10: production L3 body scaffold (2026-10-18)

Scope: A770 Windows, oneAPI 2026.1, Level Zero 1.28.2.

## 0. Verdict

The first production-facing L3 sparse-QSA graph body is wired behind the
opt-in `AILA_Q4E_QSA_LAYER_GRAPH=1`. It compiles and fails closed, but it has
not yet run against a real 2052-token production Session.

Default inference remains on the direct path.

## 1. Implemented body

The backend now has one fixed `QsaLayerGraphState` and captures:

```text
packed QKV projection
  -> index_qk GEMV
  -> graph QGKV preparation
  -> QSA query/raw-K preparation
  -> one completed-block compression
  -> compressed-block scoring
  -> fixed top-512 selector
  -> active-token validator
  -> block-native attention
  -> o_projection
```

The body uses model layer 3 but GraphControl ordinal 0. The control table is a
fixed 16-word device tensor; its values are rewritten before every replay.

## 2. Eligibility and key

The graph is eligible only when all of the following hold:

- decode `count == 1`, model layer 3, and `start_pos == CurrentLength()`;
- the position is in the sparse phase, with at least 513 complete blocks;
- the compressed count is either the current target or exactly one block
  behind it, so the graph may publish one completed block;
- BF16 K/V and indexer caches, no KV scales, mRoPE, embedding overrides, Q8
  attention, LoRA/reference overrides or unsupported topology;
- packed graph decode paths exist for QKV and o-projection;
- real production `plus_one_norm` tensors are BF16 with sizes
  `256/256/128/128`;
- all fixed runtime buffers and partial/selector scratch have production
  capacity.

The L3 key retains backend/Session identity, session generation, runtime
generation/capacity and topology generation. It excludes PLE, expert residency
and compressed-count generations because this body does not read them.

## 3. Commit and abort behavior

- A successful graph event waits, controls are copied to the host, and the
  layer-0 compressed count must equal `floor((start_pos + 1) / 4)`.
- Only then is `Session.layerCaches_[3].num_compressed_blocks` committed.
- Capture failure, event failure, control abort or a gate mismatch drains,
  destroys the executable and poisons the Session.
- A poisoned Session now rejects every subsequent forward; it cannot silently
  fall back to direct execution after partial KV/indexer writes.
- A direct count>1 or explicit session mutation aborts/destroys the retained
  L3 graph.

## 4. Evidence

Compile and CPU contracts:

```text
AilaQwen4ExpModuleTests      102/102 linked
AilaQwen4ExpModelSpecTests  PASSED
token-graph-runtime-only    PASS  0.6 s
token-graph-key-only        PASS  0.5 s
```

The review pass found and fixed:

- production norms wrongly required F32 instead of their real BF16 layout;
- pending-one compression at positions like 2055 was wrongly rejected;
- in-flight capture conflict did not drain before poison;
- poisoned Sessions could still enter `ForwardImpl`.

Receipts:

- `tmp/q4e-token-graph-probe-20261017/gpu-token-graph-runtime-p10.log`
- `tmp/q4e-token-graph-probe-20261017/gpu-token-graph-key-p10.log`

## 5. Limits

- No real model Session has executed this production L3 graph yet.
- No direct-vs-graph tensor receipt or A/B performance measurement exists.
- A 32K Session makes scoring and selector iterate the full compressed-block
  capacity. This is graph-stable but may dominate at short sparse prefixes.
- The graph remains one layer only. It is not a whole-token graph.
- The opt-in must remain off by default until real-model parity and the
  established end-to-end performance gate pass.

## 6. Next gate

Add one focused production-session probe:

```text
real model weights + real Session caches
capacity 4096 or 8192 for a bounded first run
real tokenized prefix >= 2052
start_pos 2052..2055
direct-vs-graph bitwise rows and control counter
```

If this passes, measure one A/B pair on the real decode path. Do not expand to
multiple layers or enable the graph by default before that receipt.
