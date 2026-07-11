#pragma once

#include "CpuBnb4.hpp"
#include "engine/Types.hpp"

#include <memory>
#include <atomic>
#include <string>
#include <vector>

namespace cpu_q35 {

void q35_rms_norm(const float* input,
                  const float* raw_weight,
                  int n,
                  float eps,
                  float* output);

void sigmoid_gate(const float* input,
                  const float* gate,
                  int n,
                  float* output);

float silu(float x);

}  // namespace cpu_q35

int parse_cpu_q35_prefill_batch(std::string_view value);

struct CpuQ35Layer {
    bool is_linear = false;

    std::vector<float> input_ln_weight;
    std::vector<float> post_attn_ln_weight;
    std::vector<float> q_norm_weight;
    std::vector<float> k_norm_weight;

    CpuBnb4WeightRef full_q_proj;
    CpuBnb4WeightRef full_k_proj;
    CpuBnb4WeightRef full_v_proj;
    CpuBnb4WeightRef full_o_proj;

    CpuBnb4WeightRef linear_qkv_proj;
    CpuBnb4WeightRef linear_z_proj;
    CpuBnb4WeightRef linear_a_proj;
    CpuBnb4WeightRef linear_b_proj;
    CpuBnb4WeightRef linear_o_proj;
    std::vector<float> linear_A_negexp;
    std::vector<float> linear_dt_bias;
    std::vector<float> linear_norm_weight;
    std::vector<float> linear_conv;

    CpuBnb4WeightRef mlp_gate_proj;
    CpuBnb4WeightRef mlp_up_proj;
    CpuBnb4WeightRef mlp_down_proj;
};

struct CpuQ35LayerCache {
    std::vector<float> full_k;
    std::vector<float> full_v;
    std::vector<float> linear_state;
    std::vector<float> linear_conv_state;
};

class CpuQ35HybridModel {
public:
    CpuQ35HybridModel() = default;

    bool load(const std::string& model_dir,
              const ModelSpec& spec,
              int max_seq_len,
              std::string* error);
    bool load_from_store(const CpuSafetensorsStore& store,
                         const ModelSpec& spec,
                         int max_seq_len,
                         std::string* error);

    bool loaded() const { return loaded_; }
    size_t dense_weight_cache_bytes() const;
    size_t embedding_cache_bytes() const;
    size_t projection_cache_bytes() const;
    CpuBnb4CacheMode weight_cache_mode() const { return weight_cache_mode_; }
    void reset();

    bool consume_one(int token_id, std::string* error);
    bool forward_one(int token_id, std::vector<float>& logits, std::string* error);
    bool prefill(const std::vector<int>& token_ids,
                 int micro_batch,
                 const std::atomic_bool* abort_requested,
                 std::vector<float>* logits,
                 std::string* error);

private:
    void clear_loaded();
    void reset_runtime_state();
    bool load_layers(std::string* error);
    bool run_linear_attention(CpuQ35Layer& layer,
                              CpuQ35LayerCache& cache,
                              const std::vector<float>& input,
                              std::vector<float>& output);
    bool run_full_attention(CpuQ35Layer& layer,
                            CpuQ35LayerCache& cache,
                            const std::vector<float>& input,
                            std::vector<float>& output);
    bool run_mlp(CpuQ35Layer& layer,
                 const std::vector<float>& input,
                 std::vector<float>& output);
    bool compute_logits(const std::vector<float>& hidden,
                        std::vector<float>& logits,
                        std::string* error) const;
    bool forward_one_impl(int token_id,
                          std::vector<float>* logits,
                          std::string* error);

    const CpuSafetensorsStore* store_ = nullptr;
    std::unique_ptr<CpuSafetensorsStore> owned_store_;
    const CpuTensorView* embed_weight_ = nullptr;
    ModelSpec spec_{};
    Qwen35TextConfig cfg_{};
    int max_seq_len_ = 0;
    int current_len_ = 0;
    bool loaded_ = false;
    CpuBnb4CacheMode weight_cache_mode_ = CpuBnb4CacheMode::Fp16;

    int hidden_size_ = 0;
    int ff_dim_ = 0;
    int full_q_heads_ = 0;
    int full_kv_heads_ = 0;
    int full_head_dim_ = 0;
    int full_q_dim_ = 0;
    int full_kv_dim_ = 0;
    int full_q_proj_dim_ = 0;
    int linear_q_heads_ = 0;
    int linear_kv_heads_ = 0;
    int linear_head_dim_ = 0;
    int linear_value_head_dim_ = 0;
    int linear_q_dim_ = 0;
    int linear_kv_dim_ = 0;
    int linear_qkv_dim_ = 0;
    int linear_z_dim_ = 0;
    int linear_conv_kernel_dim_ = 0;
    int linear_conv_channels_ = 0;
    int full_rotary_dim_ = 0;

    std::vector<CpuQ35Layer> layers_;
    std::vector<CpuQ35LayerCache> layer_caches_;
    std::vector<float> final_norm_weight_;
    std::vector<uint16_t> embedding_f16_;

    std::vector<float> hidden_;
    std::vector<float> normed_;
    std::vector<float> mixer_out_;
    std::vector<float> mlp_out_;
    std::vector<float> scratch_a_;
    std::vector<float> scratch_b_;
    std::vector<float> scratch_c_;
    std::vector<float> scratch_d_;
    std::vector<float> scratch_e_;
};
