# MTP Performance Optimization — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why MTP draft match rate is only 2–15% and fix the root cause, targeting >50% match rate.

**Architecture:** Add diagnostic shape logging during MTP weight loading, then add a debug verification function that compares MTP logits against the main model's lm_head logits for the same token position. Use the diagnostic output to identify and fix the computation bug.

**Tech Stack:** C++17, SYCL/oneDNN, Intel Arc A770 GPU

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/models/Qwen35HybridTextBackend.cpp` | MTP weight loading (add shape logging), forward_mtp (add verification), forward (existing) |
| `src/models/Qwen35HybridTextBackend.hpp` | Declare debug verification function |
| `include/engine/Engine.hpp` | Warmup / MTP decode loop (add verification call) |

---

### Task 1: Add MTP weight shape diagnostic logging

**Files:**
- Modify: `src/models/Qwen35HybridTextBackend.cpp:920-990`

**Purpose:** Print the actual shapes of every MTP weight tensor during loading so we can verify they match the expected dimensions derived from `full_fused_qkv_dim_`, `full_q_dim_`, `hidden_size_`, etc.

- [ ] **Step 1: Add shape logging after fc weight load**

In `Qwen35HybridTextBackend::load()`, after line 927 (`mtp_fc_.init(...)`), add:

```cpp
AILA_LOG_INFO("[MTP-Diag] mtp.fc.weight after transpose: shape=(%lld,%lld) expected_in=%d expected_out=%d",
              fc_w->shape(0), fc_w->shape(1), hidden_size_ * 2, hidden_size_);
```

- [ ] **Step 2: Add shape logging for each MTP layer's Q/K/V/O weights**

Inside the MTP layer loop (after line 947, before the `fuse_three_cols` call), add:

```cpp
AILA_LOG_INFO("[MTP-Diag] MTP layer %d weight shapes:", i);
AILA_LOG_INFO("[MTP-Diag]   q_proj: (%lld,%lld)  k_proj: (%lld,%lld)  v_proj: (%lld,%lld)  o_proj: (%lld,%lld)",
              q_w->shape(0), q_w->shape(1),
              k_w->shape(0), k_w->shape(1),
              v_w->shape(0), v_w->shape(1),
              o_w->shape(0), o_w->shape(1));
AILA_LOG_INFO("[MTP-Diag]   Expected qkv fused dims: full_q_proj_dim=%d full_kv_dim=%d full_fused_qkv_dim=%d",
              full_q_proj_dim_, full_kv_dim_, full_fused_qkv_dim_);
AILA_LOG_INFO("[MTP-Diag]   Expected: full_q_dim=%d hidden_size=%d full_kv_heads=%d full_head_dim=%d",
              full_q_dim_, hidden_size_, full_kv_heads_, full_head_dim_);
```

- [ ] **Step 3: Add shape logging for MTP gate/up/down weights**

After line 975 (the `transpose_weight` calls for gate/up/down), add:

```cpp
AILA_LOG_INFO("[MTP-Diag]   gate_proj: (%lld,%lld)  up_proj: (%lld,%lld)  down_proj: (%lld,%lld)",
              gate_w->shape(0), gate_w->shape(1),
              up_w->shape(0), up_w->shape(1),
              down_w->shape(0), down_w->shape(1));
AILA_LOG_INFO("[MTP-Diag]   Expected: ff_dim=%d hidden_size=%d", ff_dim_, hidden_size_);
```

- [ ] **Step 4: Build, run, and capture diagnostic output**

```bash
pwsh build.ps1
echo '{"messages":[{"role":"user","content":"hi"}]}' | ./build/Aila.exe -m ./models/Qwen3.5-0.8B --greedy --no-stream --messages-json - 2>&1 | grep 'MTP-Diag'
```

Verify all shapes match expectations. If any shape mismatches, that's the root cause — fix the dimension used in `init()`.

- [ ] **Step 5: Commit diagnostic logging**

```bash
git add src/models/Qwen35HybridTextBackend.cpp
git commit -m "debug(mtp): add weight shape diagnostic logging

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add MTP vs main model logit comparison

**Files:**
- Modify: `src/models/Qwen35HybridTextBackend.hpp:27` (add declaration)
- Modify: `src/models/Qwen35HybridTextBackend.cpp` (add implementation)
- Modify: `include/engine/Engine.hpp:635` (call verification after warmup)

**Purpose:** After warmup prefill, run forward_mtp on the token that the main model predicts (argmax of warmup logits), and compare the MTP logits against the main model's logits to quantify how far apart they are.

- [ ] **Step 1: Declare debug function in header**

In `Qwen35HybridTextBackend.hpp`, after line 27 (`bool has_mtp() const override`), add:

```cpp
void debug_compare_mtp_logits(Context& ctx, int token_id);
```

- [ ] **Step 2: Implement debug_compare_mtp_logits**

In `Qwen35HybridTextBackend.cpp`, add after the `forward_mtp` function (around line 2787):

```cpp
void Qwen35HybridTextBackend::debug_compare_mtp_logits(Context& ctx, int token_id) {
    if (!has_mtp_) return;

    // Run MTP on this token
    Tensor* mtp_logits = forward_mtp(ctx, token_id);
    if (!mtp_logits) return;

    // Run main model forward on same token to get baseline logits
    int* dev_token = static_cast<int*>(ctx.alloc_device(sizeof(int)));
    ctx.memcpy_h2d(dev_token, &token_id, sizeof(int));
    Tensor& main_logits = forward(ctx, dev_token, 1);
    ctx.free_device(dev_token);
    ctx.synchronize();

    // Read back first 20 logit values from both
    int vocab = cfg_.vocab_size;
    std::vector<float> mtp_host(std::min(20, vocab));
    std::vector<float> main_host(std::min(20, vocab));

    if (mtp_logits->dtype() == dnnl::memory::data_type::bf16) {
        std::vector<bf16> tmp(20);
        ctx.memcpy_d2h(tmp.data(), mtp_logits->data(), 20 * sizeof(bf16));
        for (int i = 0; i < 20; ++i) mtp_host[i] = static_cast<float>(tmp[i]);
        ctx.memcpy_d2h(tmp.data(), main_logits.data(), 20 * sizeof(bf16));
        for (int i = 0; i < 20; ++i) main_host[i] = static_cast<float>(tmp[i]);
    } else {
        ctx.memcpy_d2h(mtp_host.data(), mtp_logits->data(), 20 * sizeof(float));
        ctx.memcpy_d2h(main_host.data(), main_logits.data(), 20 * sizeof(float));
    }

    // Find argmax for both
    int mtp_argmax = 0, main_argmax = 0;
    float mtp_max = -1e9f, main_max = -1e9f;
    std::vector<float> mtp_full, main_full;

    // Read full logits for argmax (expensive but diagnostic)
    mtp_full.resize(vocab);
    main_full.resize(vocab);
    if (mtp_logits->dtype() == dnnl::memory::data_type::bf16) {
        std::vector<bf16> tmp(vocab);
        ctx.memcpy_d2h(tmp.data(), mtp_logits->data(), vocab * sizeof(bf16));
        for (int i = 0; i < vocab; ++i) mtp_full[i] = static_cast<float>(tmp[i]);
        ctx.memcpy_d2h(tmp.data(), main_logits.data(), vocab * sizeof(bf16));
        for (int i = 0; i < vocab; ++i) main_full[i] = static_cast<float>(tmp[i]);
    }
    for (int i = 0; i < vocab; ++i) {
        if (mtp_full[i] > mtp_max) { mtp_max = mtp_full[i]; mtp_argmax = i; }
        if (main_full[i] > main_max) { main_max = main_full[i]; main_argmax = i; }
    }

    AILA_LOG_INFO("[MTP-Diag] Token %d: MTP argmax=%d (max=%.4f)  Main argmax=%d (max=%.4f)  match=%d",
                  token_id, mtp_argmax, mtp_max, main_argmax, main_max,
                  mtp_argmax == main_argmax ? 1 : 0);
    AILA_LOG_INFO("[MTP-Diag] First 20 MTP logits:  [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                  mtp_host[0], mtp_host[1], mtp_host[2], mtp_host[3], mtp_host[4],
                  mtp_host[5], mtp_host[6], mtp_host[7], mtp_host[8], mtp_host[9],
                  mtp_host[10], mtp_host[11], mtp_host[12], mtp_host[13], mtp_host[14],
                  mtp_host[15], mtp_host[16], mtp_host[17], mtp_host[18], mtp_host[19]);
    AILA_LOG_INFO("[MTP-Diag] First 20 Main logits: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                  main_host[0], main_host[1], main_host[2], main_host[3], main_host[4],
                  main_host[5], main_host[6], main_host[7], main_host[8], main_host[9],
                  main_host[10], main_host[11], main_host[12], main_host[13], main_host[14],
                  main_host[15], main_host[16], main_host[17], main_host[18], main_host[19]);

    // Compute cosine similarity over full vocab
    double dot = 0.0, norm_mtp = 0.0, norm_main = 0.0;
    for (int i = 0; i < vocab; ++i) {
        dot += (double)mtp_full[i] * (double)main_full[i];
        norm_mtp += (double)mtp_full[i] * (double)mtp_full[i];
        norm_main += (double)main_full[i] * (double)main_full[i];
    }
    double cosine = dot / (std::sqrt(norm_mtp) * std::sqrt(norm_main));
    AILA_LOG_INFO("[MTP-Diag] Cosine similarity (full vocab): %.6f", cosine);

    // Roll back the main model's KV cache (forward() incremented current_len_)
    if (current_len_ > 0) {
        truncate_kv_cache(current_len_ - 1);
    }
}
```

> **Note:** This function calls `forward(ctx, dev_token, 1)` which increments `current_len_` and modifies the main model's KV cache. We roll back with `truncate_kv_cache(current_len_ - 1)` afterward. This is safe because we're still in warmup and `reset()` will be called next.

- [ ] **Step 3: Call debug_compare_mtp_logits from Engine.hpp warmup**

In `Engine.hpp`, after the warmup `argmax` call (around line 637), before `backend_->reset()`, add:

```cpp
// Diagnostic: compare MTP logits against main model logits
int warmup_predicted_token;
ctx_->memcpy_d2h(&warmup_predicted_token, warmup_argmax, sizeof(int));
ctx_->synchronize();
AILA_LOG_INFO("[MTP-Diag] Warmup predicted token: %d", warmup_predicted_token);
backend_->debug_compare_mtp_logits(*ctx_, warmup_predicted_token);
```

- [ ] **Step 4: Build, run, and capture comparison output**

```bash
pwsh build.ps1
echo '{"messages":[{"role":"user","content":"hi"}]}' | ./build/Aila.exe -m ./models/Qwen3.5-0.8B --greedy --no-stream --messages-json - 2>&1 | grep 'MTP-Diag'
```

Analyze the output:
- If `match=0`: MTP argmax differs from main model argmax → computation bug confirmed
- If `match=1` but cosine < 0.9: MTP distribution is shifted
- If `match=1` and cosine > 0.99: MTP is essentially correct, low match rate is inherent to the model

- [ ] **Step 5: Commit comparison diagnostic**

```bash
git add src/models/Qwen35HybridTextBackend.hpp src/models/Qwen35HybridTextBackend.cpp include/engine/Engine.hpp
git commit -m "debug(mtp): add MTP vs main model logit comparison diagnostic

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Fix root cause (conditional — based on diagnostic findings)

**Files:**
- Modify: `src/models/Qwen35HybridTextBackend.cpp`

**Purpose:** Apply the fix based on what the diagnostics revealed. The most likely issues and their fixes:

#### Scenario A: Weight shape mismatch

If any diagnostic log shows weight shapes != expected dimensions:

Fix the `init()` call to use the correct dimensions. For example, if MTP QKV uses `q_proj_dim` instead of `full_q_proj_dim_`:

```cpp
// Before (wrong):
layer.qkv_proj.init(ctx, fused_weights_.back(), hidden_size_, full_fused_qkv_dim_, true);
// After (fix — use actual weight dims):
int64_t actual_qkv_dim = fused_weights_.back().shape(1);
layer.qkv_proj.init(ctx, fused_weights_.back(), hidden_size_, (int)actual_qkv_dim, true);
```

#### Scenario B: Cosine similarity near zero (completely different logits)

This indicates the MTP computation path is fundamentally broken. Possible causes:

1. **Buffer aliasing**: MTP uses `buf_.normed`, `buf_.full_qkv`, etc. which may contain stale data from the main model's forward pass. Fix: add `ctx.synchronize()` after MTP RMS norm to ensure buffers are written before reading.

2. **Attention reading garbage**: `cached_len = 1` with `start_pos = 0` might not work correctly with the JM attention kernel. Fix: zero-fill the MTP KV cache before first use:

```cpp
// In load(), after MTP KV cache allocation:
ctx.queue().memset(cache.k.data(), 0, cache.k.size_bytes());
ctx.queue().memset(cache.v.data(), 0, cache.v.size_bytes());
```

3. **MTP uses wrong embedding**: If `mtp_use_dedicated_embeddings` is true but we use the main model's `embed_weight_`, the embeddings will be wrong. Fix: load dedicated embedding if available:

```cpp
if (cfg_.mtp_use_dedicated_embeddings && weights.has("mtp.embed_tokens.weight")) {
    mtp_embed_weight_ = &weights.get("mtp.embed_tokens.weight");
} else {
    mtp_embed_weight_ = embed_weight_;  // shared with main model
}
```

#### Scenario C: Cosine > 0.9 but argmax differs

MTP distribution is close but not identical. This is expected behavior for a smaller draft model. Fix: use top-k sampling for draft acceptance instead of exact match:

In `Engine.hpp` MTP loop, change accept condition:

```cpp
// Before: exact match only
if (real_next_token == draft_token) { ... }

// After: accept if draft_token is in top-3 of real logits
// (requires reading top-k from logits_current)
```

- [ ] **Step 1: Apply the fix identified by diagnostics**

Execute the specific fix code for the scenario identified.

- [ ] **Step 2: Rebuild and verify match rate improvement**

```bash
pwsh build.ps1
echo '{"messages":[{"role":"user","content":"Write a short poem about AI."}]}' | ./build/Aila.exe -m ./models/Qwen3.5-0.8B --greedy --no-stream --messages-json - --mtp 2>&1 | grep -E '(MTP-Stats|Match rate)'
```

Target: match rate > 30%. If not achieved, re-run diagnostics with the fix in place to identify remaining issues.

- [ ] **Step 3: Commit the fix**

```bash
git add src/models/Qwen35HybridTextBackend.cpp include/engine/Engine.hpp
git commit -m "fix(mtp): [describe specific fix applied]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Remove diagnostic code and verify baseline

**Files:**
- Modify: `src/models/Qwen35HybridTextBackend.cpp`
- Modify: `src/models/Qwen35HybridTextBackend.hpp`
- Modify: `include/engine/Engine.hpp`

- [ ] **Step 1: Remove diagnostic logging from MTP weight loading**

Remove the `AILA_LOG_INFO("[MTP-Diag]...")` lines added in Task 1.

- [ ] **Step 2: Remove debug_compare_mtp_logits**

Remove the function declaration from the header and implementation from the .cpp file.

- [ ] **Step 3: Remove warmup diagnostic call**

Remove the `debug_compare_mtp_logits` call from Engine.hpp warmup.

- [ ] **Step 4: Verify baseline output quality still matches**

```bash
pwsh build.ps1
echo '{"messages":[{"role":"user","content":"hi"}]}' | ./build/Aila.exe -m ./models/Qwen3.5-0.8B --greedy --no-stream --messages-json - 2>&1 | grep -E 'Generated|Hello'
echo '{"messages":[{"role":"user","content":"hi"}]}' | ./build/Aila.exe -m ./models/Qwen3.5-0.8B --greedy --no-stream --messages-json - --mtp 2>&1 | grep -E 'Generated|Hello|MTP-Stats'
```

Expected: "Hello! How can I help you today?" for both. MTP match rate should be improved.

- [ ] **Step 5: Commit cleanup**

```bash
git add src/models/Qwen35HybridTextBackend.cpp src/models/Qwen35HybridTextBackend.hpp include/engine/Engine.hpp
git commit -m "chore(mtp): remove Phase 1 diagnostic code

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase 1 Success Gate

Phase 1 is complete when:
- [ ] MTP match rate > 30% on a standard prompt (baseline > 2× improvement)
- [ ] Baseline output quality matches non-MTP mode
- [ ] No diagnostic code remains in production paths
- [ ] All commits are clean and build passes

Proceed to Phase 2 only after this gate is passed.
