#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/Ops.hpp"
#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../memory/KVCache.hpp"
#include "engine/Types.hpp"
#include <cstdint>
#include <string>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

struct Qwen3ASRBnb4Layer {
    Tensor* input_ln_weight = nullptr;
    Tensor* post_attn_ln_weight = nullptr;
    Tensor* q_norm_weight = nullptr;
    Tensor* k_norm_weight = nullptr;

    Bnb4BitLinear qkv_proj;      // fused Q/K/V (3-column horizontal concat)
    Bnb4BitLinear o_proj;
    Bnb4BitLinear gate_up_proj;  // fused gate/up (2-column horizontal concat)
    Bnb4BitLinear down_proj;
};

class Qwen3ASRBnb4Backend : public IModelBackend {
public:
    ~Qwen3ASRBnb4Backend() override;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;
    bool apply_lora(Context& ctx, const aila::lora::LoraAdapter& adapter,
                    std::string* error_message = nullptr) override;
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen3ASR; }

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

protected:
    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void ensure_prefill_scores(Context& ctx, int seq_len);
    void ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len);

    Qwen3Config cfg_{};
    RopeSpec rope_{};
    int hidden_size_ = 0;
    int ff_dim_ = 0;
    int max_seq_len_ = 0;
    int current_len_ = 0;

    int q_dim_ = 0;
    int kv_dim_ = 0;
    int fused_qkv_dim_ = 0;  // q_dim_ + 2 * kv_dim_

    std::vector<Qwen3ASRBnb4Layer> layers_;
    std::vector<Tensor> fused_weights_;
    Bnb4BitLinearScratch linear_scratch_;
    KVCache kv_cache_;

    Tensor* embed_weight_ = nullptr;
    Tensor* final_norm_weight_ = nullptr;
    Linear lm_head_;  // Tied copy, dense

    struct Buffers {
        Tensor hidden, normed, qkv;
        Tensor q, k, v, attn_out;
        Tensor gate_up, gate, up, logits;
        Tensor decode_scores, scores, incr_scores;
        Tensor rope_freq;
        Tensor decode_attn_partials; // cache for head_dim=256 decode exact partials
    } buf_;

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

private:
    int runtime_seq_capacity_ = 0;
    int prefill_scores_capacity_ = 0;
    int incr_prefill_seq_cap_ = 0;
    int incr_prefill_total_cap_ = 0;
};
