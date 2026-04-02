#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "engine/Types.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <unordered_map>
#include <string>

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
    bool decode_mem_inited_ = false;
    void* decode_src_ptr_ = nullptr;
    void* decode_weight_ptr_ = nullptr;
    void* decode_dst_ptr_ = nullptr;

    void ensure_primitive(Context& ctx, int seq_len);

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
        bool mem_inited = false;
        void* src_ptr = nullptr;
        void* weight_ptr = nullptr;
        void* dst_ptr = nullptr;
    };
    std::unordered_map<int, CachedPrimitive> prim_cache_;
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

    // Fused: output = rms_norm(input + residual, weight) (writes back to input)
    void fused_add_rms_norm(Context& ctx, Tensor& input, Tensor& residual,
                            Tensor& weight, float eps, Tensor& output,
                            int seq_len, int hidden_size);

    // Per-head RMSNorm (Qwen3 QK-norm)
    // x: [seq_len, num_heads * head_dim], weight: [head_dim]
    // Applies RMSNorm independently to each head's [head_dim] vector
    void head_rms_norm(Context& ctx, Tensor& x, Tensor& weight,
                       float eps, int seq_len, int num_heads, int head_dim);

    // RoPE: 对 Q 和 K 施加旋转位置编码 (in-place)
    // q: [seq_len, num_heads_q * head_dim]
    // k: [seq_len, num_kv_heads * head_dim]
    void apply_rope(Context& ctx, Tensor& q, Tensor& k,
                    int seq_len, int start_pos,
                    int num_heads_q, int num_kv_heads, int head_dim,
                    float theta);

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

    // GQA Attention (decode 模式, seq_len=1)
    // q:       [1, num_heads * head_dim]
    // k_cache: [num_kv_heads, cached_len, head_dim]
    // v_cache: [num_kv_heads, cached_len, head_dim]
    // output:  [1, num_heads * head_dim]
    void attention_decode(Context& ctx,
                          Tensor& q, Tensor& k_cache, Tensor& v_cache,
                          Tensor& output, Tensor& scores_buf,
                          int num_heads, int num_kv_heads, int head_dim,
                          int cached_len);

    // Prefill Attention (seq_len > 1)
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

    // SwiGLU: output = silu(gate) * up
    // gate, up, output: [n] (element-wise)
    void swiglu(Context& ctx, Tensor& gate, Tensor& up,
                Tensor& output, int n);

    // 残差连接: a += b (in-place)
    void residual_add(Context& ctx, Tensor& a, Tensor& b, int n);

    // 复制 tensor: dst = src
    void copy_tensor(Context& ctx, Tensor& src, Tensor& dst, int n);

    // Split fused projection output:
    // qkv: [seq_len, q_dim + kv_dim + kv_dim] -> q/k/v
    void split_qkv(Context& ctx, Tensor& qkv, Tensor& q, Tensor& k, Tensor& v,
                   int seq_len, int q_dim, int kv_dim);

    // Split fused FFN projection output:
    // gate_up: [seq_len, 2 * ff_dim] -> gate/up
    void split_gate_up(Context& ctx, Tensor& gate_up, Tensor& gate, Tensor& up,
                       int seq_len, int ff_dim);

    // Argmax: 返回 logits 中最大值索引 (旧版，带内部分配)
    int argmax(Context& ctx, Tensor& logits, int vocab_size);

    // Argmax: 将结果写入预先分配好的 d_result 显存指针，避免内部的 allocation 和 sync
    void argmax(Context& ctx, Tensor& logits, int vocab_size, int* d_result);

    // Top-k sampling with temperature
    int topk_sample(Context& ctx, Tensor& logits, int vocab_size,
                    float temperature, int top_k);

    // Physical transpose: dst = src^T
    // src: [R, C], dst: [C, R]
    void transpose(Context& ctx, Tensor& src, Tensor& dst);

} // namespace ops
