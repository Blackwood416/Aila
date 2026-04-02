#include "Qwen3.hpp"
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

using bf16 = sycl::ext::oneapi::bfloat16;

// ============================================================
// Load model weights from SafeTensors
// ============================================================

void Qwen3Model::load(Context& ctx, ModelWeights& weights, const Qwen3Config& config, int max_seq_len) {
    config_ = config;
    max_seq_len_ = max_seq_len;

    int H = config.hidden_size;
    int QD = config.num_attention_heads * config.head_dim;  // 2048
    int KVD = config.num_key_value_heads * config.head_dim;  // 1024
    int FF = config.intermediate_size;  // 3072
    fused_weights_.clear();
    fused_weights_.reserve(static_cast<size_t>(config.num_hidden_layers) * 2);

    // Helper: 物理转置权重 [out, in] -> [in, out]
    auto transpose_weight = [&](const std::string& name) {
        Tensor& src = weights.get(name);
        int64_t out_f = src.shape(0);
        int64_t in_f = src.shape(1);
        
        // 分配转置后的空间
        Tensor dst = Tensor::allocate(ctx, {in_f, out_f}, src.dtype());
        ops::transpose(ctx, src, dst);
        
        // 替换原始 tensor
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    auto fuse_three_cols = [&](Tensor& a, Tensor& b, Tensor& c) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t c_cols = c.shape(1);
        int64_t total_cols = a_cols + b_cols + c_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* c_ptr = static_cast<bf16*>(c.data());
        bf16* o_ptr = static_cast<bf16*>(out.data());

        ctx.queue().parallel_for(sycl::range<2>(rows, total_cols),
            [=](sycl::id<2> idx) {
                int r = idx[0];
                int cidx = idx[1];
                int64_t out_idx = static_cast<int64_t>(r) * total_cols + cidx;
                if (cidx < a_cols) {
                    o_ptr[out_idx] = a_ptr[static_cast<int64_t>(r) * a_cols + cidx];
                } else if (cidx < a_cols + b_cols) {
                    int bc = cidx - static_cast<int>(a_cols);
                    o_ptr[out_idx] = b_ptr[static_cast<int64_t>(r) * b_cols + bc];
                } else {
                    int cc = cidx - static_cast<int>(a_cols + b_cols);
                    o_ptr[out_idx] = c_ptr[static_cast<int64_t>(r) * c_cols + cc];
                }
            });
        return out;
    };

    auto fuse_two_cols = [&](Tensor& a, Tensor& b) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t total_cols = a_cols + b_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* o_ptr = static_cast<bf16*>(out.data());

        ctx.queue().parallel_for(sycl::range<2>(rows, total_cols),
            [=](sycl::id<2> idx) {
                int r = idx[0];
                int cidx = idx[1];
                int64_t out_idx = static_cast<int64_t>(r) * total_cols + cidx;
                if (cidx < a_cols) {
                    o_ptr[out_idx] = a_ptr[static_cast<int64_t>(r) * a_cols + cidx];
                } else {
                    int bc = cidx - static_cast<int>(a_cols);
                    o_ptr[out_idx] = b_ptr[static_cast<int64_t>(r) * b_cols + bc];
                }
            });
        return out;
    };

    // --- Embedding ---
    embed_weight_ = &weights.get("model.embed_tokens.weight");
    std::cout << "[Qwen3] embed_tokens loaded" << std::endl;

    // --- Layers ---
    layers_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        auto& layer = layers_[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        layer.input_ln_weight = &weights.get(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = &weights.get(prefix + "post_attention_layernorm.weight");

        // 加载并转置权重
        Tensor* qw = transpose_weight(prefix + "self_attn.q_proj.weight");
        Tensor* kw = transpose_weight(prefix + "self_attn.k_proj.weight");
        Tensor* vw = transpose_weight(prefix + "self_attn.v_proj.weight");
        layer.q_proj.init(ctx, *qw, H, QD, true);
        layer.k_proj.init(ctx, *kw, H, KVD, true);
        layer.v_proj.init(ctx, *vw, H, KVD, true);
        fused_weights_.push_back(fuse_three_cols(*qw, *kw, *vw));
        layer.qkv_proj.init(ctx, fused_weights_.back(), H, QD + 2 * KVD, true);

        Tensor* ow = transpose_weight(prefix + "self_attn.o_proj.weight");
        layer.o_proj.init(ctx, *ow, QD, H, true);

        // Qwen3 QK-norm: per-head RMSNorm on Q and K
        layer.q_norm_weight = &weights.get(prefix + "self_attn.q_norm.weight");
        layer.k_norm_weight = &weights.get(prefix + "self_attn.k_norm.weight");

        Tensor* gw = transpose_weight(prefix + "mlp.gate_proj.weight");
        Tensor* uw = transpose_weight(prefix + "mlp.up_proj.weight");
        layer.gate_proj.init(ctx, *gw, H, FF, true);
        layer.up_proj.init(ctx, *uw, H, FF, true);
        fused_weights_.push_back(fuse_two_cols(*gw, *uw));
        layer.gate_up_proj.init(ctx, fused_weights_.back(), H, 2 * FF, true);

        Tensor* dw = transpose_weight(prefix + "mlp.down_proj.weight");
        layer.down_proj.init(ctx, *dw, FF, H, true);
    }
    std::cout << "[Qwen3] " << config.num_hidden_layers << " transformer layers loaded" << std::endl;

    // --- Final norm ---
    final_norm_weight_ = &weights.get("model.norm.weight");

    // --- LM Head ---
    if (weights.has("lm_head.weight")) {
        Tensor* lw = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lw, H, config.vocab_size, true);
        std::cout << "[Qwen3] lm_head loaded (standalone)" << std::endl;
    } else {
        // lm_head shares embed_weight_, but it is [vocab, hidden]
        // Actually, in Qwen, lm_head is typically tied with embed,
        // and embed is [vocab, hidden]. Linear expects [hidden, vocab] after transpose.
        // We'll treat tied case differently or just transpose a copy.
        Tensor& src = *embed_weight_;
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        weights.put("lm_head.weight_preprocessed", std::move(dst));
        lm_head_.init(ctx, weights.get("lm_head.weight_preprocessed"), H, config.vocab_size, true);
        std::cout << "[Qwen3] lm_head (tied, preprocessed copy)" << std::endl;
    }

    // --- KV Cache ---
    kv_cache_.init(ctx, config, max_seq_len);

    // --- Activation buffers ---
    buf_.hidden   = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.residual = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.normed   = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.qkv      = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)(QD + 2 * KVD)});
    buf_.q        = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)QD});
    buf_.k        = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)KVD});
    buf_.v        = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)KVD});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)QD});
    buf_.gate_up  = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)(2 * FF)});
    buf_.gate     = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)FF});
    buf_.up       = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)FF});
    buf_.ffn_out  = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.logits   = Tensor::allocate(ctx, {1, (int64_t)config.vocab_size});
    buf_.decode_scores = Tensor::allocate(ctx,
        {(int64_t)config.num_attention_heads, (int64_t)max_seq_len},
        dnnl::memory::data_type::f32);
    // Prefill attention scores: [num_heads, max_seq, max_seq] in f32
    buf_.scores   = Tensor::allocate(ctx,
        {(int64_t)config.num_attention_heads, (int64_t)max_seq_len, (int64_t)max_seq_len},
        dnnl::memory::data_type::f32);
    buf_.rope_freq = Tensor::allocate(ctx,
        {(int64_t)(config.head_dim / 2)},
        dnnl::memory::data_type::f32);
    {
        std::vector<float> rope_freq_host(static_cast<size_t>(config.head_dim / 2));
        for (int d = 0; d < config.head_dim / 2; ++d) {
            rope_freq_host[static_cast<size_t>(d)] =
                1.0f / std::pow(config.rope_theta, (2.0f * d) / static_cast<float>(config.head_dim));
        }
        ctx.memcpy_h2d(buf_.rope_freq.data(), rope_freq_host.data(),
                       rope_freq_host.size() * sizeof(float));
    }

    std::cout << "[Qwen3] Model fully loaded and initialized" << std::endl;
}

// ============================================================
// Forward pass
// ============================================================

Tensor& Qwen3Model::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    int H = config_.hidden_size;
    int QD = config_.num_attention_heads * config_.head_dim;
    int KVD = config_.num_key_value_heads * config_.head_dim;
    int FF = config_.intermediate_size;
    int start_pos = kv_cache_.current_length();
    int cached_len = start_pos + seq_len;

    // 1. Embedding lookup
    ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, H);

    // 2. Initial norm for the first layer
    ops::rms_norm(ctx, buf_.hidden, *layers_[0].input_ln_weight,
                  config_.rms_norm_eps, buf_.normed, seq_len, H);

    // 3. Iterate through transformer layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        auto& layer = layers_[i];
        Tensor q_decode_view;
        Tensor* q_for_attn = &buf_.q;

        if (seq_len == 1) {
            // Decode path: fused QKV projection to reduce tiny matmul launches
            layer.qkv_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
            bf16* qkv_ptr = static_cast<bf16*>(buf_.qkv.data());
            q_decode_view = Tensor::view(ctx, qkv_ptr, {1, (int64_t)QD});
            Tensor k_view = Tensor::view(ctx, qkv_ptr + QD, {1, (int64_t)KVD});
            Tensor v_view = Tensor::view(ctx, qkv_ptr + QD + KVD, {1, (int64_t)KVD});
            q_for_attn = &q_decode_view;

            ops::decode_prepare_qkv(ctx, q_decode_view, k_view, v_view, buf_.rope_freq,
                                    *layer.q_norm_weight, *layer.k_norm_weight,
                                    kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                    start_pos,
                                    config_.num_attention_heads, config_.num_key_value_heads,
                                    config_.head_dim, config_.rms_norm_eps, config_.rope_theta);
        } else {
            // Prefill path: keep split projections for better throughput on larger M
            layer.q_proj.forward(ctx, buf_.normed, buf_.q, seq_len);
            layer.k_proj.forward(ctx, buf_.normed, buf_.k, seq_len);
            layer.v_proj.forward(ctx, buf_.normed, buf_.v, seq_len);

            // 4.5 Qwen3 QK-norm: per-head RMSNorm on Q and K (before RoPE)
            ops::head_rms_norm(ctx, buf_.q, *layer.q_norm_weight,
                               config_.rms_norm_eps, seq_len,
                               config_.num_attention_heads, config_.head_dim);
            ops::head_rms_norm(ctx, buf_.k, *layer.k_norm_weight,
                               config_.rms_norm_eps, seq_len,
                               config_.num_key_value_heads, config_.head_dim);

            // 5. Apply RoPE to Q and K
            ops::apply_rope(ctx, buf_.q, buf_.k, seq_len, start_pos,
                            config_.num_attention_heads, config_.num_key_value_heads,
                            config_.head_dim, config_.rope_theta);

            // 6. Write K, V to cache
            ops::copy_to_cache(ctx, buf_.k, kv_cache_.k_cache(i),
                               seq_len, start_pos,
                               config_.num_key_value_heads, config_.head_dim,
                               kv_cache_.max_length());
            ops::copy_to_cache(ctx, buf_.v, kv_cache_.v_cache(i),
                               seq_len, start_pos,
                               config_.num_key_value_heads, config_.head_dim,
                               kv_cache_.max_length());
        }

        // 7. Attention
        if (seq_len == 1) {
            // Decode mode: GEMV attention
            ops::attention_decode(ctx, *q_for_attn, kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                  buf_.attn_out, buf_.decode_scores,
                                  config_.num_attention_heads, config_.num_key_value_heads,
                                  config_.head_dim, cached_len);
        } else {
            // Prefill mode: full attention matrix
            ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v,
                                   buf_.attn_out, buf_.scores,
                                   seq_len,
                                   config_.num_attention_heads, config_.num_key_value_heads,
                                   config_.head_dim);
        }

        // 8. O projection
        // Use buf_.gate as a temporary addition buffer
        layer.o_proj.forward(ctx, buf_.attn_out, buf_.gate, seq_len);

        // 9. Fused: hidden += addition, then normed = norm(hidden)
        ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate,
                                *layer.post_attn_ln_weight, config_.rms_norm_eps,
                                buf_.normed, seq_len, H);

        if (seq_len == 1) {
            // Decode path: fused gate/up projection
            layer.gate_up_proj.forward(ctx, buf_.normed, buf_.gate_up, seq_len);
            bf16* gate_up_ptr = static_cast<bf16*>(buf_.gate_up.data());
            Tensor gate_view = Tensor::view(ctx, gate_up_ptr, {1, (int64_t)FF});
            Tensor up_view = Tensor::view(ctx, gate_up_ptr + FF, {1, (int64_t)FF});
            ops::swiglu(ctx, gate_view, up_view, buf_.gate, seq_len * FF);
        } else {
            // Prefill path
            layer.gate_proj.forward(ctx, buf_.normed, buf_.gate, seq_len);
            layer.up_proj.forward(ctx, buf_.normed, buf_.up, seq_len);
            ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * FF);
        }
        
        // 11. Down projection -> buf_.up (temp addition)
        layer.down_proj.forward(ctx, buf_.gate, buf_.up, seq_len);

        // 12. Final fused for sub-layer
        if (i < config_.num_hidden_layers - 1) {
            // hidden += addition, then normed = input_norm of NEXT layer
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                    *layers_[i + 1].input_ln_weight, config_.rms_norm_eps,
                                    buf_.normed, seq_len, H);
        } else {
            // Last layer: residual + final norm
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                    *final_norm_weight_, config_.rms_norm_eps,
                                    buf_.normed, seq_len, H);
        }
    }

    // 17. LM Head: logits = normed[-1] @ embed_weight^T
    // Only compute logits for the last token
    if (seq_len > 1) {
        // Create a view pointing to the last token's hidden state
        bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * H;
        Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)H});
        lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
    } else {
        lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
    }

    // Update KV cache position
    kv_cache_.advance(seq_len);

    return buf_.logits;
}

void Qwen3Model::reset() {
    kv_cache_.reset();
}
