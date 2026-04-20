#pragma once

#include "IModelBackend.hpp"
#include "../ops/Ops.hpp"
#include <vector>
#include <sycl/sycl.hpp>

class Qwen35HybridTextBackend : public IModelBackend {
public:
    ~Qwen35HybridTextBackend() override;

    bool load(Context& ctx,
              ModelWeights& weights,
              const ModelSpec& spec,
              int max_seq_len,
              std::string* error_message) override;

    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override;
    void reset() override;
    void truncate_kv_cache(int new_len) override;
    int max_seq_len() const override { return max_seq_len_; }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen35Hybrid; }

    void set_embedding_overrides(const std::vector<int>& positions,
                                 const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
                                 int hidden_size);
    void clear_embedding_overrides();
    void set_mrope_positions(Context& ctx,
                             const std::vector<int>& pos_t,
                             const std::vector<int>& pos_h,
                             const std::vector<int>& pos_w,
                             int text_pos_delta);
    void clear_mrope_positions();

private:
    struct Layer {
        bool is_linear = false;

        Tensor* input_ln_weight = nullptr;
        Tensor* post_attn_ln_weight = nullptr;

        // full attention branch
        Linear q_proj, k_proj, v_proj, o_proj;
        Linear qkv_proj;
        Tensor* qkv_weight_jm = nullptr; // bf16 [full_fused_qkv_dim, hidden]
        Tensor* o_weight_jm = nullptr; // bf16 [hidden, full_q_dim]
        Tensor* q_norm_weight = nullptr;
        Tensor* k_norm_weight = nullptr;

        // linear attention branch
        Linear linear_qkv_proj, linear_z_proj, linear_o_proj;
        Linear linear_all_proj;
        Tensor* linear_all_weight_jm = nullptr; // bf16 [linear_all_dim, hidden]
        Tensor* linear_o_weight_jm = nullptr; // bf16 [hidden, linear_kv_dim]
        Linear linear_a_proj, linear_b_proj;
        Tensor* linear_norm_weight = nullptr; // f32
        Tensor* linear_A_log = nullptr;       // f32
        Tensor* linear_dt_bias = nullptr;     // bf16
        Tensor* linear_conv1d_weight = nullptr; // bf16 [channels, 1, kernel]

        // FFN
        Linear gate_proj, up_proj, down_proj;
        Linear gate_up_proj;
        Tensor* gate_up_weight = nullptr; // bf16 [hidden, 2 * ff_dim]
        Tensor* down_weight = nullptr;    // bf16 [ff_dim, hidden]
        Tensor* gate_up_weight_jm = nullptr; // bf16 [2 * ff_dim, hidden]
        Tensor* down_weight_jm = nullptr;    // bf16 [hidden, ff_dim]

        std::vector<float> host_linear_A_negexp;
        std::vector<float> host_linear_dt_bias;
        std::vector<float> host_linear_conv;      // [channels * kernel]
        std::vector<float> host_linear_norm;      // [head_dim]
    };

    struct LayerCache {
        Tensor k;
        Tensor v;
        // DeltaNet recurrent state S per head: [num_heads, key_dim, value_dim] (f32)
        Tensor linear_state;
        Tensor linear_conv_state; // [kernel-1, conv_channels] (f32)
        std::vector<float> host_linear_state;
        std::vector<float> host_linear_conv_state;
        bool device_state_dirty = false;
        int linear_conv_head = 0; // Ring-buffer head for the oldest conv row.
    };

    struct Buffers {
        Tensor hidden;
        Tensor normed;
        Tensor qkv;
        Tensor linear_all;
        Tensor q;
        Tensor k;
        Tensor v;
        Tensor z;
        Tensor a;
        Tensor b;
        Tensor attn_out;
        Tensor full_qkv;
        Tensor gate_up;
        Tensor gate;
        Tensor up;
        Tensor logits;
        Tensor decode_scores;
        Tensor decode_attn_partials;
        Tensor scores;
        Tensor incr_scores;
    } buf_;

    void ensure_runtime_buffers(Context& ctx, int seq_len);
    void ensure_prefill_scores(Context& ctx, int seq_len);
    void ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len);
    void run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                     Tensor& qkv_src, Tensor& z_src,
                                     Tensor& a_src, Tensor& b_src);
    void run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                     Tensor& qkv_src, Tensor& z_src,
                                     Tensor& a_src, Tensor& b_src,
                                     Tensor& out_dst);
    void run_linear_delta_prefill_gpu_batched(Context& ctx, Layer& layer, LayerCache& cache,
                                              Tensor& fused_all, Tensor& out_dst, int seq_len);
    bool use_decode_jm_custom_path(int seq_len) const;
    bool use_decode_ffn_custom_path(int seq_len) const;
    void run_decode_ffn_gate_up_swiglu_custom(Context& ctx, const Layer& layer,
                                              Tensor& input, Tensor& output);
    void run_decode_ffn_down_custom(Context& ctx, const Layer& layer,
                                    Tensor& input, Tensor& output);
    void run_decode_jm_matvec_custom(Context& ctx, Tensor& input,
                                     const Tensor& weight_jm,
                                     int in_dim, int out_dim, Tensor& output);
    void debug_compare_linear_delta_decode(Context& ctx, int layer_idx,
                                           Layer& layer, LayerCache& cache,
                                           Tensor& qkv_src, Tensor& z_src,
                                           Tensor& a_src, Tensor& b_src);
    void run_linear_delta_host(Context& ctx, Layer& layer, LayerCache& cache,
                               Tensor& qkv_src, Tensor& z_src, Tensor& a_src, Tensor& b_src,
                               int seq_len);
    static float softplus(float x);
    static float silu(float x);
    static void head_l2_norm_inplace(std::vector<float>& x, int seq_len, int num_heads, int head_dim, float eps);
    static void head_rms_norm_and_silu_gate(std::vector<float>& x,
                                            const std::vector<float>& norm_weight,
                                            const std::vector<float>& z,
                                            int seq_len, int num_heads, int head_dim, float eps);

    ModelSpec spec_{};
    Qwen35TextConfig cfg_{};
    Tensor* embed_weight_ = nullptr;
    Tensor* final_norm_weight_ = nullptr;
    Linear lm_head_;

    std::vector<Layer> layers_;
    std::vector<LayerCache> layer_caches_;
    std::vector<Tensor> fused_weights_;

    int max_seq_len_ = 0;
    int current_len_ = 0;
    int runtime_seq_capacity_ = 0;
    int prefill_scores_capacity_ = 0;
    int incr_prefill_seq_cap_ = 0;
    int incr_prefill_total_cap_ = 0;

    int full_q_heads_ = 0;
    int full_kv_heads_ = 0;
    int full_head_dim_ = 0;
    int full_q_dim_ = 0;
    int full_kv_dim_ = 0;
    int full_q_proj_dim_ = 0;
    int full_fused_qkv_dim_ = 0;

    int linear_q_heads_ = 0;
    int linear_kv_heads_ = 0;
    int linear_head_dim_ = 0;
    int linear_q_dim_ = 0;
    int linear_kv_dim_ = 0;
    int linear_qkv_dim_ = 0;
    int linear_all_dim_ = 0;
    int linear_z_dim_ = 0;
    int linear_conv_kernel_dim_ = 4;
    int linear_conv_channels_ = 0;

    int hidden_size_ = 0;
    int ff_dim_ = 0;
    int max_attn_heads_ = 0;
    int max_qkv_dim_ = 0;
    int max_attn_dim_ = 0;
    bool use_delta_linear_ = false;
    bool decode_ffn_custom_enabled_ = false;

    struct LinearDeltaScratch {
        std::vector<sycl::ext::oneapi::bfloat16> h_qkv;
        std::vector<sycl::ext::oneapi::bfloat16> h_z;
        std::vector<sycl::ext::oneapi::bfloat16> h_a;
        std::vector<sycl::ext::oneapi::bfloat16> h_b;
        std::vector<float> conv_in;
        std::vector<float> z;
        std::vector<float> a;
        std::vector<float> b;
        std::vector<float> conv_out;
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        std::vector<float> out;
        std::vector<float> sk;
        std::vector<float> d;
        std::vector<sycl::ext::oneapi::bfloat16> out_bf16;
    } linear_delta_scratch_;

    std::vector<int> embed_override_positions_;
    std::vector<sycl::ext::oneapi::bfloat16> embed_override_values_;
    int embed_override_hidden_size_ = 0;

    Context* mrope_ctx_ = nullptr;
    int* mrope_pos_t_ = nullptr;
    int* mrope_pos_h_ = nullptr;
    int* mrope_pos_w_ = nullptr;
    int mrope_prompt_len_ = 0;
    int mrope_text_pos_delta_ = 0;
};
