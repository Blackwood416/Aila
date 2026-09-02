#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include "engine/Types.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Weight views and runtime projections for one DiffusionGemma text layer
// (encoder == decoder weights).
struct DiffusionGemmaLayerWeights {
    Bnb4BitWeightRef q_proj_ref;
    Bnb4BitWeightRef k_proj_ref;
    Bnb4BitWeightRef v_proj_ref;  // absent on full-attention layers (k_eq_v)
    Bnb4BitWeightRef o_proj_ref;

    Bnb4BitWeightRef mlp_gate_up_ref;
    Bnb4BitWeightRef mlp_up_ref;
    Bnb4BitWeightRef mlp_down_ref;
    Bnb4BitLinear q_proj;
    Bnb4BitLinear k_proj;
    Bnb4BitLinear v_proj;
    Bnb4BitLinear o_proj;
    Bnb4BitLinear mlp_gate;
    Bnb4BitLinear mlp_up;
    Bnb4BitLinear mlp_gate_up;
    Bnb4BitLinear mlp_down;
    bool has_fused_gate_up = false;

    // MoE experts are stored as one packed tensor per layer with all experts
    // concatenated along the row axis:
    //   gate_up rows: num_experts * 2 * moe_intermediate_size
    //   down    rows: num_experts * moe_intermediate_size
    Bnb4BitWeightRef moe_gate_up_ref;
    Bnb4BitWeightRef moe_down_ref;

    Tensor* input_ln = nullptr;
    Tensor* post_attn_ln = nullptr;
    Tensor* pre_ff_ln = nullptr;
    Tensor* post_ff_ln = nullptr;
    Tensor* pre_ff_ln_2 = nullptr;
    Tensor* post_ff_ln_1 = nullptr;
    Tensor* post_ff_ln_2 = nullptr;
    Tensor* q_norm = nullptr;
    Tensor* k_norm = nullptr;

    Tensor* router_proj = nullptr;      // (num_experts, hidden)
    Tensor* router_scale = nullptr;     // (hidden)
    Tensor* per_expert_scale = nullptr; // (num_experts)
    std::vector<sycl::ext::oneapi::bfloat16> per_expert_scale_host;
    float layer_scalar = 1.0f;
    Linear router_linear;
};

class DiffusionGemmaBnb4Backend : public IModelBackend {
public:
    struct SoftEmbedTiming {
        // Host-side submission windows.  The GPU work remains asynchronous;
        // the following soft-embed D2H wait in the generator accounts for the
        // actual device execution without adding a profiling synchronization.
        double softmax_submit_ms = 0.0;
        double gemm_submit_ms = 0.0;
    };

    ~DiffusionGemmaBnb4Backend() override;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;

    // Standard AR forward is not used by the DiffusionGemma generation path.
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;

    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int get_current_context_len() const override { return current_len_; }
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.text.vocab_size; }
    ModelFamily family() const override { return ModelFamily::DiffusionGemma; }

    bool supports_block_diffusion() const override { return true; }
    bool diffusion_encode(Context& ctx, const int* token_ids_device,
                          int seq_len,
                          std::string* error_message) override;
    Tensor& diffusion_decode_canvas(
        Context& ctx, const int* canvas_ids_device,
        const sycl::ext::oneapi::bfloat16* sc_embeds_device,
        int canvas_len, float temperature = 1.0f) override;
    Tensor& diffusion_compute_soft_embeds(Context& ctx, Tensor& logits_f32,
                                          int canvas_len) override;

    int* diffusion_argmax_device() override {
        return buf_.d_argmax.valid() ? static_cast<int*>(buf_.d_argmax.data()) : nullptr;
    }
    float* diffusion_entropy_device() override {
        return buf_.d_entropy.valid() ? static_cast<float*>(buf_.d_entropy.data()) : nullptr;
    }
    bool supports_vision_embedding_override() const override { return true; }
    void set_embedding_overrides(
        const std::vector<int>& positions,
        const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
        int hidden_size) override;
    void clear_embedding_overrides() override;

    const SoftEmbedTiming& last_soft_embed_timing() const {
        return last_soft_embed_timing_;
    }

    const DiffusionGemmaConfig& config() const { return cfg_; }

private:
    std::vector<int> embed_override_positions_;
    std::vector<sycl::ext::oneapi::bfloat16> embed_override_values_;
    int embed_override_hidden_size_ = 0;
    struct LayerRuntime {
        Tensor k_cache;  // [kv_heads, max_seq_len, head_dim]
        Tensor v_cache;  // [kv_heads, max_seq_len, head_dim]
    };

    struct Buffers {
        // Per-layer attention scratch.  These are raw capacity buffers rather
        // than fixed-stride matrices so run_layer_forward() can create a
        // compact view for either the full-attention or sliding-attention
        // geometry without allocating on every layer.
        Tensor attn_q;
        Tensor attn_k;
        Tensor attn_v;
        Tensor attn_out;
        Tensor attn_k_full;
        Tensor attn_v_full;
        Tensor attn_mask;
        Tensor attn_scores;
        Tensor hidden;
        Tensor normed;
        Tensor attn_o;
        Tensor mlp_gate, mlp_up, mlp_gate_up_out, mlp_proj, mlp_out;
        Tensor ff1, ff2;
        Tensor moe_device_out;
        Tensor moe_gpu_ref;
        Tensor moe_transfer;
        Tensor moe_token_ids;
        Tensor moe_weights;
        Tensor router_scores;
        Tensor sc_normed, sc_gate, sc_up, sc_proj, sc_down;
        Tensor logits_bf16;
        Tensor logits_f32;
        Tensor soft_embeds;
        Tensor probs_bf16;
        Tensor row_max;
        Tensor row_inv_sum;
        Tensor d_argmax;
        Tensor d_entropy;
        Tensor expert_ids_dev;
        Tensor expert_weights_dev;
        Tensor ones;
        Bnb4BitLinearScratch qkv_scratch;
        Bnb4BitLinearScratch o_scratch;
        Bnb4BitLinearScratch mlp_scratch;
        Bnb4BitLinearScratch mlp_down_scratch;
        Bnb4BitLinearScratch sc_scratch;
        Linear lm_head;
        Linear soft_embed;
    };

    struct ProfileTotals {
        double attention_ms = 0.0;
        double qkv_proj_ms = 0.0;
        double attention_core_ms = 0.0;
        double o_proj_ms = 0.0;
        double dense_mlp_ms = 0.0;
        double dense_gate_up_ms = 0.0;
        double dense_activation_ms = 0.0;
        double dense_down_ms = 0.0;
        double router_ms = 0.0;
        double moe_ms = 0.0;
        double post_ms = 0.0;
    };

    ModelSpec spec_;
    DiffusionGemmaConfig cfg_{};
    int max_seq_len_ = 0;
    int current_len_ = 0;
    int buf_q_cap_ = 0;
    int buf_kv_cap_ = 0;
    int attn_q_dim_cap_ = 0;
    int attn_kv_dim_cap_ = 0;

    std::vector<DiffusionGemmaLayerWeights> layers_;
    std::vector<LayerRuntime> runtimes_;
    std::vector<std::unique_ptr<Bnb4GroupedMoE>> moes_;
    Buffers buf_;
    ProfileTotals profile_totals_;
    SoftEmbedTiming last_soft_embed_timing_;

    Tensor* embed_weight_ = nullptr;
    Tensor* final_norm_weight_ = nullptr;
    Tensor* sc_pre_norm_ = nullptr;
    Bnb4BitLinear sc_gate_;
    Bnb4BitLinear sc_up_;
    Bnb4BitLinear sc_down_;
    float embed_scale_ = 1.0f;

    void ensure_buffers(Context& ctx, int q_len, int kv_len);
    void ensure_logits_buffers(Context& ctx);
    void fill_attention_mask(Context& ctx, Tensor& mask, int q_len, int kv_len,
                             int cached_len, bool bidirectional,
                             bool sliding, int window);
    void concat_kv(Context& ctx, Tensor& dst, const Tensor& cache,
                   const Tensor& new_kv, int seq_len, int start_pos,
                   int kv_heads, int head_dim, int max_seq_len);
    void run_layer_forward(Context& ctx, DiffusionGemmaLayerWeights& layer,
                           LayerRuntime& runtime, Tensor& hidden,
                           int seq_len, int start_pos, bool bidirectional);
};
