#pragma once

#include "IModelBackend.hpp"
#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../ops/Ops.hpp"
#include "../memory/KVCache.hpp"
#include "../utils/SafeTensors.hpp"
#include "engine/Types.hpp"
#include <vector>
#include <string>

class Qwen3DenseBackend : public IModelBackend {
public:
    Qwen3DenseBackend() = default;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen3Dense; }

private:
    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void ensure_prefill_scores(Context& ctx, int seq_len);
    void ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len);

    Qwen3Config cfg_{};
    int max_seq_len_ = 0;

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

    Tensor* embed_weight_ = nullptr;
    std::vector<TransformerLayer> layers_;
    std::vector<Tensor> fused_weights_;
    Tensor* final_norm_weight_ = nullptr;
    Linear lm_head_;
    KVCache kv_cache_;

    struct Buffers {
        Tensor hidden;       // [max_seq, hidden_size]
        Tensor normed;       // [max_seq, hidden_size]
        Tensor qkv;          // [max_seq, q_dim + 2*kv_dim]
        Tensor q;            // [max_seq, num_heads * head_dim]
        Tensor k;            // [max_seq, num_kv_heads * head_dim]
        Tensor v;            // [max_seq, num_kv_heads * head_dim]
        Tensor attn_out;     // [max_seq, num_heads * head_dim]
        Tensor gate_up;      // [max_seq, 2 * intermediate_size]
        Tensor gate;         // [max_seq, intermediate_size]
        Tensor up;           // [max_seq, intermediate_size]
        Tensor logits;       // [1, vocab_size]
        Tensor decode_scores;// [num_heads, max_seq] for decode attention
        Tensor scores;       // [num_heads, max_seq, max_seq] for prefill
        Tensor incr_scores;  // [num_heads, seq_len, total_len] for incremental prefill
        Tensor rope_freq;    // [head_dim/2] precomputed RoPE frequency table (f32)
        Tensor decode_attn_partials; // cache for head_dim=256 decode exact partials
    } buf_;

    int runtime_seq_capacity_ = 0;
    int prefill_scores_capacity_ = 0;
    int incr_prefill_seq_cap_ = 0;
    int incr_prefill_total_cap_ = 0;
};
