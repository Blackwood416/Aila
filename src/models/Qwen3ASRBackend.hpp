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
#include <deque>

using bf16 = sycl::ext::oneapi::bfloat16;

class Qwen3ASRBackend : public IModelBackend {
public:
    Qwen3ASRBackend() = default;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen3ASR; }
    int get_current_context_len() const override { return current_len_; }

    // Multimodal injection for audio features
    bool supports_vision_embedding_override() const override { return true; }
    void set_embedding_overrides(const std::vector<int>& positions,
                                 const std::vector<bf16>& embeddings,
                                 int hidden_size) override;
    void clear_embedding_overrides() override;
    void set_mrope_positions(Context& ctx,
                             const std::vector<int>& pos_t,
                             const std::vector<int>& pos_h,
                             const std::vector<int>& pos_w,
                             int text_pos_delta) override;
    void clear_mrope_positions() override;

private:
    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void ensure_prefill_scores(Context& ctx, int seq_len);
    void ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len);

protected:
    Qwen3Config cfg_{};
    RopeSpec rope_{};
    int max_seq_len_ = 0;
    int current_len_ = 0;

    struct TransformerLayer {
        Linear q_proj, k_proj, v_proj;
        Linear qkv_proj, o_proj;
        Linear gate_proj, up_proj;
        Linear gate_up_proj, down_proj;
        Tensor* input_ln_weight = nullptr;
        Tensor* post_attn_ln_weight = nullptr;
        Tensor* q_norm_weight = nullptr;
        Tensor* k_norm_weight = nullptr;
    };

    Tensor* embed_weight_ = nullptr;
    std::vector<TransformerLayer> layers_;
    std::vector<Tensor> fused_weights_;
    Tensor* final_norm_weight_ = nullptr;
    Linear lm_head_;
    KVCache kv_cache_;

    struct Buffers {
        Tensor hidden;
        Tensor normed;
        Tensor qkv;
        Tensor q;
        Tensor k;
        Tensor v;
        Tensor attn_out;
        Tensor gate_up;
        Tensor gate;
        Tensor up;
        Tensor logits;
        Tensor decode_scores;
        Tensor scores;
        Tensor incr_scores;
        Tensor decode_attn_partials; // cache for head_dim=256 decode exact partials
    } buf_;

private:
    int runtime_seq_capacity_ = 0;
    int prefill_scores_capacity_ = 0;
    int incr_prefill_seq_cap_ = 0;
    int incr_prefill_total_cap_ = 0;

    // Embedding override state (for audio token injection)
    bool has_embedding_overrides_ = false;
    std::vector<int> override_positions_;
    std::vector<bf16> override_embeddings_;
    int override_hidden_size_ = 0;
    Tensor override_buf_;      // GPU buffer for override values
    Tensor override_pos_buf_;  // GPU buffer for override positions (persistent)

    // MRoPE state
    bool has_mrope_positions_ = false;
    std::vector<int> mrope_pos_t_;
    std::vector<int> mrope_pos_h_;
    std::vector<int> mrope_pos_w_;
    int mrope_text_pos_delta_ = 0;
    Tensor mrope_pos_t_dev_;
    Tensor mrope_pos_h_dev_;
    Tensor mrope_pos_w_dev_;
    int mrope_pos_capacity_ = 0;

    void upload_mrope_positions(Context& ctx);
};
