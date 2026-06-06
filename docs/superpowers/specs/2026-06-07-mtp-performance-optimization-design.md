# MTP Performance Optimization Design

Date: 2026-06-07

## Problem

Aila's Qwen3.5 MTP speculative decoding produces correct output but is **35% slower**
than the baseline (79 tok/s vs 107 tok/s). The root cause is an MTP draft match rate
of only 2–15%, making the extra computation a net loss.

### Baseline numbers (Qwen3.5-0.8B, greedy decode, Arc A770)

| Mode | tok/s | Match rate | Effective forward calls |
|------|-------|------------|------------------------|
| No MTP | 106.8 | — | N |
| MTP | 79.1 | 2–15% | ~1.3×N |

### Comparison with reference implementations

Both [vllm](vllm/vllm/model_executor/models/qwen3_5_mtp.py) and
[llama.cpp](llama.cpp/common/speculative.cpp) achieve >50% match rates with their MTP
implementations. Key architectural differences:

| Aspect | vllm / llama.cpp | Aila (current) |
|--------|------------------|----------------|
| MTP KV cache position | Follows main model positions | Always position 0 |
| MTP KV cache length | Grows with each MTP call | Always 1 |
| MTP RoPE encoding | Matches main model positions | Effectively disabled (pos=0) |

However, single-token self-attention with RoPE is mathematically invariant to position
(RoPE rotation preserves dot products when Q and K share the same angle). The position
difference alone does NOT explain the 2% match rate — there is a deeper computation bug.

## Design

### Phase 1: Diagnose and fix MTP computation correctness

The MTP logits are essentially random relative to the main model's logits. We need to
pinpoint where the computation diverges.

**Step 1.1: Compare hidden state extraction**

Verify that `buf_.hidden[current_len_ - 1]` is the correct final hidden state.
Compare against the main model's logits computation path to ensure the hidden state
is captured at the right point (post final-FFN residual, pre final-norm).

**Step 1.2: Compare embedding lookup**

Compare MTP's `embedding_lookup` output against the main model's embedding for the
same token ID. They share `embed_weight_` so should be identical.

**Step 1.3: Validate FC projection weight shape**

Confirm `mtp.fc.weight` transposed shape is `[hidden_size, hidden_size * 2]` as
expected by `mtp_fc_.init(in=hidden*2, out=hidden, preprocessed=true)`.

**Step 1.4: Validate MTP QKV dimensions**

The MTP layer reuses `full_fused_qkv_dim_` from the main model. Verify the actual
MTP Q/K/V weight shapes match. If the MTP layer has different head dimensions,
this would produce garbage QKV projections.

**Step 1.5: Compare attention output**

For single-token self-attention, the output should approximately equal the input V
vector. Verify this holds for the MTP attention.

**Step 1.6: Compare final logits**

After fixing any issues found above, compare MTP logits against main model logits
for a fixed token. Target: top-1 agreement >50%.

**Expected outcome:** Match rate improves from ~10% to >50%.

### Phase 2: Align RoPE positions with vllm/llama.cpp

Once MTP computation is correct, align the KV cache management with reference
implementations.

**Changes in `Qwen35HybridTextBackend`:**
- Add `int mtp_current_len_ = 0;` member to track MTP KV cache length
- In `reset()`: set `mtp_current_len_ = 0`
- In `forward_mtp()`:
  - `start_pos = mtp_current_len_` (increments each call)
  - `cached_len = mtp_current_len_ + 1` (growing KV cache)
  - After attention: `mtp_current_len_++`

**Changes in `truncate_kv_cache()`:**
- When the main model truncates, also truncate `mtp_current_len_` proportionally

**Expected outcome:** RoPE positions match training distribution. Match rate may
improve further, and multi-token drafting (Phase 3) requires this foundation.

### Phase 3: Multi-token speculation

Predict 2–3 draft tokens per main model decode step instead of 1.

**Architecture:**
- If `mtp_num_hidden_layers >= num_drafts`: use one layer per draft token
- If `mtp_num_hidden_layers < num_drafts`: cycle layers (`step_idx % num_layers`)

**Changes in `forward_mtp()`:**
- New signature: `forward_mtp(ctx, start_token_id, num_drafts)` → returns `Tensor*` with `num_drafts` rows of logits
- Loop: for each draft step, use previous step's output hidden state as input, run MTP decoder layer, sample next token

**Changes in `Engine.hpp` MTP decode loop:**
- Accept 0..num_drafts draft tokens per verify step
- Verify each draft token against main model's predictions
- On first mismatch, accept all previous drafts and fall back to main model

**Expected outcome:** With 50%+ match rate and 3 draft tokens, theoretical speedup
approaches 2× (net ~1.3–1.5× after overhead).

### Phase 4: Per-step overhead reduction

Reduce the fixed cost of each MTP decode step.

**4.1: Pre-allocate device token buffers**
- Replace per-call `alloc_device`/`free_device` for token IDs with persistent buffers
- Allocate once in `ensure_runtime_buffers()`, reuse across calls

**4.2: Fuse rms_norm + concatenation kernel**
- Current: 2× `ops::rms_norm` (separate kernels) + 1× concat kernel
- Fused: single kernel that reads embedding + hidden state, applies norms, writes concatenated output
- Eliminates 2 kernel launch overheads and intermediate buffer reads

**4.3: Remove redundant sync points**
- `ops::argmax` internally calls `ctx.synchronize()`. For MTP draft sampling,
  we can pipeline the argmax with the next forward call.

**Expected outcome:** 10–20% reduction in per-step overhead, especially for short
generations where overhead dominates.

## Implementation Order

Phases MUST be implemented in order:
1. **Phase 1 first** — without fixing match rate, all other optimizations are wasted
2. **Phase 2 second** — required foundation for Phase 3
3. **Phase 3 third** — multi-token speculation depends on correct KV cache
4. **Phase 4 last** — overhead reduction after correctness and speculation width are fixed

Each phase includes its own verification step: run the same benchmark prompt and
confirm tok/s improvement.

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| MTP match rate | 2–15% | >50% |
| MTP tok/s vs baseline | 74% (slower) | >120% (faster) |
| Output quality | Matches baseline | Matches baseline (must not regress) |
| Multi-token drafts | 1 | 2–3 |

## Risk: This model wasn't trained for MTP

The Qwen3.5-0.8B checkpoint may have MTP weights that are randomly initialized or
from a different training run. If Phase 1 diagnostics reveal that the MTP weights
produce inherently poor predictions even with correct computation, the fallback is:
- Accept the low match rate and skip Phases 2–3
- Focus purely on overhead reduction (Phase 4) to minimize MTP's cost
- Or disable MTP and pursue alternative speedups (e.g., PLD / n-gram speculation)

## Reference files

- `src/models/Qwen35HybridTextBackend.cpp` — MTP forward pass, weight loading
- `src/models/Qwen35HybridTextBackend.hpp` — MTP member declarations
- `include/engine/Engine.hpp` — MTP speculative decode loop
- `vllm/vllm/model_executor/models/qwen3_5_mtp.py` — vllm reference
- `llama.cpp/common/speculative.cpp` — llama.cpp MTP draft/verify
