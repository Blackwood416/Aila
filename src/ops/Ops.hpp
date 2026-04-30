#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "engine/Types.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>

// ============================================================
// Linear 层 (oneDNN MatMul 封装)
// ============================================================
class Linear {
public:
    Linear() = default;

    // weight: [out_features, in_features] (如果 preprocessed=false)
    // weight: [in_features, out_features]  (如果 preprocessed=true)
    void init(Context& ctx, Tensor& weight, int in_features, int out_features, bool preprocessed = false);

    // forward: output = input @ weight^T
    // 如果权重已预转置，则计算 input @ weight_ptr
    void forward(Context& ctx, Tensor& input, Tensor& output, int seq_len);
    void forward_bias(Context& ctx, Tensor& input, Tensor& bias,
                      Tensor& output, int seq_len);
    void forward_bias_gelu_tanh(Context& ctx, Tensor& input, Tensor& bias,
                                Tensor& output, int seq_len);

private:
    Tensor* weight_ = nullptr;
    int in_features_ = 0;
    int out_features_ = 0;
    bool preprocessed_ = false; // 默认 false (原始 [out, in] 布局)

    // 缓存 decode (seq_len=1) 模式的 primitive
    dnnl::matmul decode_prim_;
    dnnl::memory::desc decode_src_md_;
    dnnl::memory::desc decode_weight_md_;
    dnnl::memory::desc decode_dst_md_;
    bool decode_inited_ = false;

    // 缓存 decode (seq_len=1) 模式创建的 memory 句柄 (避免 USM 重复绑定的极高开销)
    dnnl::memory decode_src_mem_;
    dnnl::memory decode_weight_mem_;
    dnnl::memory decode_dst_mem_;
    dnnl::memory decode_scratchpad_mem_;
    Tensor decode_scratchpad_;
    bool decode_mem_inited_ = false;
    void* decode_src_ptr_ = nullptr;
    void* decode_weight_ptr_ = nullptr;
    void* decode_dst_ptr_ = nullptr;

    // 预缓存 decode args map，避免热循环中每次重构
    std::unordered_map<int, dnnl::memory> decode_args_;

    void ensure_primitive(Context& ctx, int seq_len);
    void ensure_bias_primitive(Context& ctx, int seq_len,
                               dnnl::memory::data_type bias_dtype);
    void ensure_bias_gelu_tanh_primitive(Context& ctx, int seq_len,
                                         dnnl::memory::data_type bias_dtype);

    // 运行时 primitive 缓存 (非 decode 模式)
    struct CachedPrimitive {
        dnnl::matmul prim;
        dnnl::memory::desc src_md;
        dnnl::memory::desc weight_md;
        dnnl::memory::desc dst_md;

        // 句柄缓存
        dnnl::memory src_mem;
        dnnl::memory weight_mem;
        dnnl::memory dst_mem;
        dnnl::memory scratchpad_mem;
        Tensor scratchpad;
        bool mem_inited = false;
        void* src_ptr = nullptr;
        void* weight_ptr = nullptr;
        void* dst_ptr = nullptr;

        // 预缓存 args map
        std::unordered_map<int, dnnl::memory> args;
    };

    struct CachedBiasPrimitive {
        dnnl::matmul prim;
        dnnl::memory::desc src_md;
        dnnl::memory::desc weight_md;
        dnnl::memory::desc bias_md;
        dnnl::memory::desc dst_md;

        dnnl::memory src_mem;
        dnnl::memory weight_mem;
        dnnl::memory bias_mem;
        dnnl::memory dst_mem;
        dnnl::memory scratchpad_mem;
        Tensor scratchpad;
        bool mem_inited = false;
        void* src_ptr = nullptr;
        void* weight_ptr = nullptr;
        void* bias_ptr = nullptr;
        void* dst_ptr = nullptr;

        std::unordered_map<int, dnnl::memory> args;
    };

    struct CachedBiasGeluTanhPrimitive {
        dnnl::matmul prim;
        dnnl::memory::desc src_md;
        dnnl::memory::desc weight_md;
        dnnl::memory::desc bias_md;
        dnnl::memory::desc dst_md;

        dnnl::memory src_mem;
        dnnl::memory weight_mem;
        dnnl::memory bias_mem;
        dnnl::memory dst_mem;
        dnnl::memory scratchpad_mem;
        Tensor scratchpad;
        bool mem_inited = false;
        void* src_ptr = nullptr;
        void* weight_ptr = nullptr;
        void* bias_ptr = nullptr;
        void* dst_ptr = nullptr;

        std::unordered_map<int, dnnl::memory> args;
    };

    std::unordered_map<int, CachedPrimitive> prim_cache_;
    std::unordered_map<uint64_t, CachedBiasPrimitive> bias_cache_;
    std::unordered_map<uint64_t, CachedBiasGeluTanhPrimitive> bias_gelu_tanh_cache_;
};

// ============================================================
// SYCL Kernel 算子
// ============================================================
namespace ops {

    // Embedding lookup: output[i] = table[token_ids[i]]
    // table: [vocab_size, hidden_size], output: [seq_len, hidden_size]
    void embedding_lookup(Context& ctx, Tensor& table,
                          const int* token_ids_device, int seq_len,
                          Tensor& output, int hidden_size);

    // RMSNorm: output = x / sqrt(mean(x^2) + eps) * weight
    // input/output: [seq_len, hidden_size], weight: [hidden_size]
    void rms_norm(Context& ctx, Tensor& input, Tensor& weight,
                  float eps, Tensor& output, int seq_len, int hidden_size);

    // LayerNorm: output = ((x - mean) / sqrt(var + eps)) * weight + bias
    // input/output: [rows, cols], weight/bias: [cols]
    void layer_norm(Context& ctx, Tensor& input, Tensor& weight, Tensor& bias,
                    float eps, Tensor& output, int rows, int cols);

    // Fused: output = rms_norm(input + residual, weight) (writes back to input)
    void fused_add_rms_norm(Context& ctx, Tensor& input, Tensor& residual,
                            Tensor& weight, float eps, Tensor& output,
                            int seq_len, int hidden_size);

    // Per-head RMSNorm (Qwen3 QK-norm)
    // x: [seq_len, num_heads * head_dim], weight: [head_dim]
    // Applies RMSNorm independently to each head's [head_dim] vector
    void head_rms_norm(Context& ctx, Tensor& x, Tensor& weight,
                       float eps, int seq_len, int num_heads, int head_dim);

    // output = sigmoid(gate) * input (element-wise)
    void sigmoid_mul(Context& ctx, Tensor& input, Tensor& gate,
                     Tensor& output, int n);

    // RoPE: 对 Q 和 K 施加旋转位置编码 (in-place)
    // q: [seq_len, num_heads_q * head_dim]
    // k: [seq_len, num_kv_heads * head_dim]
    void apply_rope(Context& ctx, Tensor& q, Tensor& k,
                    int seq_len, int start_pos,
                    int num_heads_q, int num_kv_heads, int head_dim,
                    float theta);

    // RoPE with explicit rotary dimension (must be even and <= head_dim).
    void apply_rope_partial(Context& ctx, Tensor& q, Tensor& k,
                            int seq_len, int start_pos,
                            int num_heads_q, int num_kv_heads, int head_dim,
                            int rotary_dim,
                            float theta,
                            bool interleaved = false,
                            const int* pos_t = nullptr,
                            const int* pos_h = nullptr,
                            const int* pos_w = nullptr,
                            int prompt_pos_len = 0,
                            int text_pos_delta = 0,
                            int mrope_section_t = 0,
                            int mrope_section_h = 0,
                            int mrope_section_w = 0);

    // 将新 K/V 写到 cache 的指定位置
    // new_kv: [seq_len, num_kv_heads * head_dim]
    // cache: [num_kv_heads, max_seq_len, head_dim] (预分配)
    void copy_to_cache(Context& ctx, Tensor& new_kv, Tensor& cache,
                       int seq_len, int start_pos,
                       int num_heads, int head_dim, int max_seq_len);

    // Decode fused prep (seq_len=1):
    // Q/K per-head RMSNorm + RoPE, and write K/V to cache at start_pos.
    void decode_prepare_qkv(Context& ctx,
                            Tensor& q, Tensor& k, Tensor& v,
                            Tensor& rope_freq,
                            Tensor& q_norm_weight, Tensor& k_norm_weight,
                            Tensor& k_cache, Tensor& v_cache,
                            int start_pos,
                            int num_heads_q, int num_kv_heads, int head_dim,
                            float eps, float theta);

    // Decode fused prep with partial RoPE / mRoPE support (seq_len=1):
    // Q/K per-head RMSNorm + partial RoPE, then write K/V to cache.
    void decode_prepare_qkv_partial(Context& ctx,
                                    Tensor& q, Tensor& k, Tensor& v,
                                    Tensor& q_norm_weight, Tensor& k_norm_weight,
                                    Tensor& k_cache, Tensor& v_cache,
                                    int start_pos,
                                    int num_heads_q, int num_kv_heads, int head_dim,
                                    float eps, int rotary_dim, float theta,
                                    bool interleaved = false,
                                    const int* pos_t = nullptr,
                                    const int* pos_h = nullptr,
                                    const int* pos_w = nullptr,
                                    int prompt_pos_len = 0,
                                    int text_pos_delta = 0,
                                    int mrope_section_t = 0,
                                    int mrope_section_h = 0,
                                    int mrope_section_w = 0);

    // Decode fused prep for packed Q+gate layout (seq_len=1):
    // split packed q/gate -> q_out/gate_out, Q/K RMSNorm + partial RoPE, then write K/V to cache.
    void decode_prepare_qgkv_packed_partial(Context& ctx,
                                            Tensor& q_gate_packed, Tensor& k, Tensor& v,
                                            Tensor& q_out, Tensor& gate_out,
                                            Tensor& q_norm_weight, Tensor& k_norm_weight,
                                            Tensor& k_cache, Tensor& v_cache,
                                            int start_pos,
                                            int num_heads_q, int num_kv_heads, int head_dim,
                                            float eps, int rotary_dim, float theta,
                                            bool interleaved = false,
                                            const int* pos_t = nullptr,
                                            const int* pos_h = nullptr,
                                            const int* pos_w = nullptr,
                                            int prompt_pos_len = 0,
                                            int text_pos_delta = 0,
                                            int mrope_section_t = 0,
                                            int mrope_section_h = 0,
                                            int mrope_section_w = 0);

    // GQA Attention (decode 模式, seq_len=1)
    // q:       [1, num_heads * head_dim]
    // k_cache: [num_kv_heads, cached_len, head_dim]
    // v_cache: [num_kv_heads, cached_len, head_dim]
    // output:  [1, num_heads * head_dim]
    void attention_decode(Context& ctx,
                          Tensor& q, Tensor& k_cache, Tensor& v_cache,
                          Tensor& output, Tensor& scores_buf,
                          int num_heads, int num_kv_heads, int head_dim,
                          int cached_len,
                          Tensor* exact_partials_buf = nullptr);

    // Prefill Attention (seq_len > 1, initial prefill with start_pos=0)
    // q: [seq_len, num_heads * head_dim]
    // k: [seq_len, num_kv_heads * head_dim]  (本次 K, 已 RoPE)
    // v: [seq_len, num_kv_heads * head_dim]  (本次 V)
    // output: [seq_len, num_heads * head_dim]
    // 内含 causal masking + softmax
    void attention_prefill(Context& ctx,
                           Tensor& q, Tensor& k, Tensor& v,
                           Tensor& output, Tensor& scores_buf,
                           int seq_len,
                           int num_heads, int num_kv_heads, int head_dim);

    // Bidirectional Attention (vision prefill)
    // q/k/v/output: [seq_len, num_heads * head_dim]
    // scores_buf: [num_heads, seq_len, seq_len]
    void attention_bidi(Context& ctx,
                        Tensor& q, Tensor& k, Tensor& v,
                        Tensor& output, Tensor& scores_buf,
                        int seq_len, int num_heads, int head_dim);

    // Incremental Prefill Attention (seq_len > 1, start_pos > 0)
    // Q attends to full KV cache [0, start_pos + seq_len) with causal masking.
    // New K/V must already be written to cache before calling this.
    // q:       [seq_len, num_heads * head_dim]
    // k_cache: [num_kv_heads, max_seq_len, head_dim]
    // v_cache: [num_kv_heads, max_seq_len, head_dim]
    // output:  [seq_len, num_heads * head_dim]
    // scores_buf: [num_heads, seq_len, total_len] where total_len = start_pos + seq_len
    void attention_prefill_cached(Context& ctx,
                                  Tensor& q, Tensor& k_cache, Tensor& v_cache,
                                  Tensor& output, Tensor& scores_buf,
                                  int seq_len, int start_pos,
                                  int num_heads, int num_kv_heads,
                                  int head_dim, int max_seq_len);

    // SwiGLU: output = silu(gate) * up
    // gate, up, output: [n] (element-wise)
    void swiglu(Context& ctx, Tensor& gate, Tensor& up,
                Tensor& output, int n);

    // Fused SwiGLU on concatenated [gate|up] tensor (decode optimization)
    // gate_up: [2 * ff_dim], output: [ff_dim]
    // Reads first half as gate, second half as up, computes silu(gate)*up
    void fused_gate_up_swiglu(Context& ctx, Tensor& gate_up,
                               Tensor& output, int ff_dim);

    // GELU-Tanh activation: x = gelu(x) (in-place)
    void gelu_tanh_inplace(Context& ctx, Tensor& input, int n);

    // Bias add: input[row, col] += bias[col] (in-place)
    void bias_add_inplace(Context& ctx, Tensor& input, Tensor& bias, int rows, int cols);

    // 残差连接: a += b (in-place)
    void residual_add(Context& ctx, Tensor& a, Tensor& b, int n);

    // 复制 tensor: dst = src
    void copy_tensor(Context& ctx, Tensor& src, Tensor& dst, int n);

    // Split fused projection output:
    // qkv: [seq_len, q_dim + kv_dim + kv_dim] -> q/k/v
    void split_qkv(Context& ctx, Tensor& qkv, Tensor& q, Tensor& k, Tensor& v,
                   int seq_len, int q_dim, int kv_dim);
    void split_qkv_bias(Context& ctx, Tensor& qkv, Tensor& bias,
                        Tensor& q, Tensor& k, Tensor& v,
                        int seq_len, int q_dim, int kv_dim);

    // Split fused FFN projection output:
    // gate_up: [seq_len, 2 * ff_dim] -> gate/up
    void split_gate_up(Context& ctx, Tensor& gate_up, Tensor& gate, Tensor& up,
                       int seq_len, int ff_dim);

    // Split fused linear-attention projection output:
    // linear_all: [seq_len, qkv_dim + z_dim + a_dim + b_dim]
    //   -> qkv: [seq_len, qkv_dim]
    //   -> z:   [seq_len, z_dim]
    //   -> a:   [seq_len, a_dim]
    //   -> b:   [seq_len, b_dim]
    void split_linear_all(Context& ctx, Tensor& linear_all, Tensor& qkv,
                          Tensor& z, Tensor& a, Tensor& b,
                          int seq_len, int qkv_dim, int z_dim,
                          int a_dim, int b_dim);

    // Split gated attention Q projection output where per-head layout is
    // [q_head | gate_head] and heads are packed consecutively:
    // q_gate: [seq_len, num_heads * (2 * head_dim)]
    //   -> q:    [seq_len, num_heads * head_dim]
    //   -> gate: [seq_len, num_heads * head_dim]
    void split_q_gate(Context& ctx, Tensor& q_gate, Tensor& q, Tensor& gate,
                      int seq_len, int num_heads, int head_dim);

    // Argmax: 返回 logits 中最大值索引 (旧版，带内部分配)
    int argmax(Context& ctx, Tensor& logits, int vocab_size);

    // Argmax: 将结果写入预先分配好的 d_result 显存指针，避免内部的 allocation 和 sync
    void argmax(Context& ctx, Tensor& logits, int vocab_size, int* d_result);

    // Top-k sampling with temperature
    int topk_sample(Context& ctx, Tensor& logits, int vocab_size,
                    float temperature, int top_k);

    // Set sampling RNG seed for deterministic sampling runs.
    void set_sampling_seed(uint64_t seed);

    // Draw a single uniform random number in [0, 1) from the sampling RNG.
    float next_sampling_uniform();

    // Apply repetition/presence/frequency penalties to logits (CPU side).
    // logits_f: float logits array on CPU (size = vocab_size)
    // generated_ids: all tokens generated so far in this turn
    void apply_penalties(float* logits_f, int vocab_size,
                         const std::vector<int>& generated_ids,
                         float repetition_penalty,
                         float presence_penalty,
                         float frequency_penalty);

    // Unified sampling: D2H logits → penalties → temperature → top-k → sample/argmax
    // Returns sampled token ID.
    int sample_with_config(Context& ctx, Tensor& logits, int vocab_size,
                           const GenerationConfig& gen_config,
                           const std::vector<int>& generated_ids);

    // Check whether the current generation config can use the device-side
    // no-penalty sampling fast path.
    bool can_use_device_sampling(int vocab_size, const GenerationConfig& gen_config);

    // Device-side sampling for the no-penalty fast path. Writes sampled token ID
    // to d_result without forcing a host synchronization.
    void sample_with_config_device(Context& ctx, Tensor& logits, int vocab_size,
                                   const GenerationConfig& gen_config,
                                   float random_u, int* d_result);

    // Vision patchify + normalize: RGB u8 -> patch rows bf16
    // patches: [num_patches, 3 * patch_size * patch_size]
    void vision_patchify_rgb_u8(Context& ctx, const uint8_t* rgb_device, Tensor& patches,
                                int width, int height, int patch_size,
                                float mean0, float mean1, float mean2,
                                float std0, float std1, float std2);

    // Vision position embedding add with optional bilinear interpolation.
    void vision_add_position_embedding(Context& ctx, Tensor& tokens, Tensor& pos_embed,
                                       int patch_grid_w, int patch_grid_h, int hidden_size);

    // Reorder tokens into merge-block order expected by Qwen3.5 vision blocks.
    void vision_reorder_merge_blocks(Context& ctx, Tensor& src, Tensor& dst,
                                     int grid_w, int grid_h, int hidden_size, int merge_size);

    // Vision mRoPE in-place for [num_tokens, num_heads * head_dim].
    void vision_mrope_inplace(Context& ctx, Tensor& x,
                              int num_tokens, int num_heads, int head_dim,
                              const int* pos_y, const int* pos_x);

    // Physical transpose: dst = src^T
    // src: [R, C], dst: [C, R]
    void transpose(Context& ctx, Tensor& src, Tensor& dst);

} // namespace ops
