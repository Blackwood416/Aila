#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include "engine/Types.hpp"
#include <cstdint>
#include <string>
#include <vector>

// Weight views for one DiffusionGemma text layer (encoder == decoder weights).
struct DiffusionGemmaLayerWeights {
    Bnb4BitWeightRef q_proj;
    Bnb4BitWeightRef k_proj;
    Bnb4BitWeightRef v_proj;  // absent on full-attention layers (k_eq_v)
    Bnb4BitWeightRef o_proj;

    Bnb4BitWeightRef mlp_gate_up;
    Bnb4BitWeightRef mlp_down;

    // MoE experts are stored as one packed tensor per layer with all experts
    // concatenated along the row axis:
    //   gate_up rows: num_experts * 2 * moe_intermediate_size
    //   down    rows: num_experts * moe_intermediate_size
    Bnb4BitWeightRef moe_gate_up;
    Bnb4BitWeightRef moe_down;

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
};

class DiffusionGemmaBnb4Backend : public IModelBackend {
public:
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

private:
    ModelSpec spec_;
    DiffusionGemmaConfig cfg_{};
    int max_seq_len_ = 0;
    int current_len_ = 0;

    std::vector<DiffusionGemmaLayerWeights> layers_;
    Tensor* embed_weight_ = nullptr;
    Tensor* final_norm_weight_ = nullptr;

    Tensor* sc_pre_norm_ = nullptr;
    Bnb4BitWeightRef sc_gate_;
    Bnb4BitWeightRef sc_up_;
    Bnb4BitWeightRef sc_down_;
};
