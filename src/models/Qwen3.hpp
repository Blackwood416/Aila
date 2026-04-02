#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../ops/Ops.hpp"
#include "../memory/KVCache.hpp"
#include "../utils/SafeTensors.hpp"
#include "engine/Types.hpp"
#include <vector>
#include <string>

// ============================================================
// Qwen3 Model: weights + forward pass
// ============================================================
class Qwen3Model {
public:
    Qwen3Model() = default;

    // Load weights from ModelWeights and initialize all layers
    void load(Context& ctx, ModelWeights& weights, const Qwen3Config& config, int max_seq_len = 4096);

    // Forward pass: token_ids -> logits
    // Returns reference to internal logits buffer
    // start_pos = kv_cache current length (for RoPE)
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len);

    // Reset KV cache for new conversation
    void reset();

    const Qwen3Config& config() const { return config_; }

private:
    Qwen3Config config_;

    // Per-layer components
    struct TransformerLayer {
        Linear q_proj, k_proj, v_proj;
        Linear qkv_proj, o_proj;
        Linear gate_proj, up_proj;
        Linear gate_up_proj, down_proj;
        Tensor* input_ln_weight = nullptr;    // RMSNorm gamma [hidden_size]
        Tensor* post_attn_ln_weight = nullptr;
        Tensor* q_norm_weight = nullptr;       // Qwen3 QK-norm [head_dim]
        Tensor* k_norm_weight = nullptr;       // Qwen3 QK-norm [head_dim]
    };

    // Embedding
    Tensor* embed_weight_ = nullptr;   // [vocab_size, hidden_size]

    // Layers
    std::vector<TransformerLayer> layers_;
    std::vector<Tensor> fused_weights_;

    // Final norm
    Tensor* final_norm_weight_ = nullptr;  // [hidden_size]

    // lm_head shares embed_weight_ (tie_word_embeddings)
    Linear lm_head_;

    // KV Cache
    KVCache kv_cache_;

    // Pre-allocated activation buffers
    struct Buffers {
        Tensor hidden;       // [max_seq, hidden_size]
        Tensor residual;     // [max_seq, hidden_size]
        Tensor normed;       // [max_seq, hidden_size]
        Tensor qkv;          // [max_seq, q_dim + kv_dim + kv_dim]
        Tensor q;            // [max_seq, num_heads * head_dim]
        Tensor k;            // [max_seq, num_kv_heads * head_dim]
        Tensor v;            // [max_seq, num_kv_heads * head_dim]
        Tensor attn_out;     // [max_seq, num_heads * head_dim]
        Tensor gate_up;      // [max_seq, 2 * intermediate_size]
        Tensor gate;         // [max_seq, intermediate_size]
        Tensor up;           // [max_seq, intermediate_size]
        Tensor ffn_out;      // [max_seq, hidden_size]
        Tensor logits;       // [1, vocab_size]
        Tensor decode_scores;// [num_heads, max_seq] for decode attention
        Tensor scores;       // [num_heads, max_seq, max_seq] for prefill attention
        Tensor rope_freq;    // [head_dim/2] precomputed RoPE frequency table (f32)
    } buf_;

    int max_seq_len_ = 0;
};
