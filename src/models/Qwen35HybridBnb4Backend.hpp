#pragma once

#include "IModelBackend.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include "engine/Types.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

struct Qwen35HybridBnB4Layer {
    bool is_linear = false;

    Tensor* input_ln_weight = nullptr;
    Tensor* post_attn_ln_weight = nullptr;
    Tensor* q_norm_weight = nullptr;
    Tensor* k_norm_weight = nullptr;

    Bnb4BitLinear qkv_proj;
    Bnb4BitLinear o_proj;

    Bnb4BitLinear gate_up_proj;
    Bnb4BitLinear down_proj;

    Bnb4BitLinear linear_all_proj;
    Bnb4BitLinear linear_o_proj;
    Bnb4BitLinear linear_qkv_proj;
    Bnb4BitLinear linear_z_proj;
    Tensor* linear_norm_weight = nullptr;
    Tensor* linear_A_log = nullptr;
    Tensor* linear_dt_bias = nullptr;
    Tensor* linear_conv1d_weight = nullptr;
    Tensor* linear_conv = nullptr;

    std::vector<float> host_linear_A_negexp;
    std::vector<float> host_linear_conv;
    std::vector<float> host_linear_dt_bias;
    std::vector<float> host_linear_norm;
};

typedef Qwen35HybridBnB4Layer Layer;

struct Qwen35HybridBnB4LayerCache {
    Tensor k;
    Tensor v;
    Tensor linear_state;
    Tensor linear_conv_state;
    std::vector<float> host_linear_state;
    std::vector<float> host_linear_conv_state;
    int linear_conv_head = 0;
    bool device_state_dirty = false;
};

struct Qwen35HybridDeltaScratch {
    std::vector<sycl::ext::oneapi::bfloat16> h_qkv, h_z, h_a, h_b;
    std::vector<float> conv_in, conv_out, z, a, b, q, k, v, d, out, sk;
    std::vector<sycl::ext::oneapi::bfloat16> out_bf16;
};

typedef Qwen35HybridBnB4LayerCache LayerCache;

struct Qwen35HybridBnB4Buffers {
    Tensor hidden, normed, qkv, linear_all;
    Tensor q, k, v, a, b, z;
    Tensor attn_out, full_qkv, gate_up, gate, up;
    Tensor scores, incr_scores, logits;
    Tensor decode_scores, decode_attn_partials;
};

class Qwen35HybridBnb4Backend : public IModelBackend {
public:
    ~Qwen35HybridBnb4Backend() override;

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;
    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    bool truncate_kv_cache(int new_len) override;
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen35Hybrid; }

    bool supports_vision_embedding_override() const override { return true; }
    void set_embedding_overrides(const std::vector<int>& positions,
                                 const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
                                 int hidden_size) override;
    void clear_embedding_overrides() override;
    void set_mrope_positions(Context& ctx, const std::vector<int>& pos_t,
                             const std::vector<int>& pos_h,
                             const std::vector<int>& pos_w,
                             int text_pos_delta) override;
    void clear_mrope_positions() override;

private:
    ModelSpec spec_;
    Qwen35TextConfig cfg_;
    int hidden_size_ = 0, ff_dim_ = 0, max_seq_len_ = 0, current_len_ = 0;
    int full_q_heads_ = 0, full_kv_heads_ = 0, full_head_dim_ = 0;
    int full_q_dim_ = 0, full_kv_dim_ = 0, full_q_proj_dim_ = 0, full_fused_qkv_dim_ = 0;
    int linear_q_heads_ = 0, linear_kv_heads_ = 0, linear_head_dim_ = 0;
    int linear_q_dim_ = 0, linear_kv_dim_ = 0, linear_qkv_dim_ = 0;
    int linear_z_dim_ = 0, linear_all_dim_ = 0, linear_conv_channels_ = 0;
    int linear_conv_kernel_dim_ = 0;
    int max_qkv_dim_ = 0, max_attn_dim_ = 0, max_attn_heads_ = 0;
    bool use_delta_linear_ = false;
    std::vector<Qwen35HybridBnB4Layer> layers_;
    std::vector<Qwen35HybridBnB4LayerCache> layer_caches_;
    std::vector<Tensor> fused_weights_;
    Qwen35HybridBnB4Buffers buf_;
    Bnb4BitLinearScratch linear_scratch_;
    Qwen35HybridDeltaScratch linear_delta_scratch_;
    Linear lm_head_;

    Tensor* embed_weight_ = nullptr;
    Tensor* final_norm_weight_ = nullptr;

    int runtime_seq_capacity_ = 0;
    int prefill_scores_capacity_ = 0;
    int incr_prefill_seq_cap_ = 0, incr_prefill_total_cap_ = 0;

    Context* mrope_ctx_ = nullptr;
    int* mrope_pos_t_ = nullptr;
    int* mrope_pos_h_ = nullptr;
    int* mrope_pos_w_ = nullptr;
    int mrope_prompt_len_ = 0;
    int mrope_text_pos_delta_ = 0;

    std::vector<int> embed_override_positions_;
    std::vector<sycl::ext::oneapi::bfloat16> embed_override_values_;
    int embed_override_hidden_size_ = 0;

    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void ensure_prefill_scores(Context& ctx, int seq_len);
    void ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len);

    void run_linear_delta_decode_gpu(Context& ctx, Layer& layer,
                                      LayerCache& cache,
                                      Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src);
    void run_linear_delta_decode_gpu(Context& ctx, Layer& layer,
                                      LayerCache& cache,
                                      Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src,
                                      Tensor& out_dst);
    void run_linear_delta_prefill_gpu_batched(Context& ctx, Layer& layer,
                                               LayerCache& cache,
                                               Tensor& linear_all_src, Tensor& out_dst, int seq_len);
    void run_linear_delta_host(Context& ctx, Layer& layer,
                                LayerCache& cache,
                                Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src,
                                int seq_len);
    void run_linear_delta_decode_host(Context& ctx, Layer& layer,
                                       LayerCache& cache,
                                       Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src,
                                       int seq_len);
    void debug_compare_linear_delta_decode(Context& ctx, int layer_idx,
                                            Layer& layer,
                                            LayerCache& cache,
                                            Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src);

    static float softplus(float x);
    static float silu(float x);
    void head_l2_norm_inplace(std::vector<float>& x, int seq_len, int num_heads,
                               int head_dim, float eps);
    void head_rms_norm_and_silu_gate(std::vector<float>& x, const std::vector<float>& norm_weight,
                                      const std::vector<float>& z, int seq_len, int num_heads,
                                      int head_dim, float eps);
};
